"""Pure contracts and lazy MLX primitives for GLM-5.3 multi-stream mHC."""

from __future__ import annotations

import importlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

OFFICIAL_MHC_LAYERS: Final = tuple(range(45))
TRANSFORMERS_REFERENCE: Final = "eb4d9e2a64a013bec12289288b85d0b1210ba0aa"
_MHC_NAME_RE: Final = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.(hc_(?:attn|ffn)_(?:fn|base|scale))$"
)


class MHCContractError(ValueError):
    """The configuration or weights do not implement pinned GLM-5.3 mHC."""


@dataclass(frozen=True, slots=True)
class MHCConfig:
    hidden_size: int = 4096
    streams: int = 4
    eps: float = 1e-6
    sinkhorn_iters: int = 20
    rms_norm_eps: float = 1e-5

    @property
    def mix_size(self) -> int:
        return (2 + self.streams) * self.streams


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text = config.get("text_config", config)
    if not isinstance(text, Mapping):
        raise MHCContractError("text_config must be an object")
    return text


def _exact(container: Mapping[str, Any], field: str, expected: Any) -> None:
    actual = container.get(field)
    if actual != expected or (
        isinstance(expected, int)
        and not isinstance(expected, bool)
        and isinstance(actual, bool)
    ):
        raise MHCContractError(
            f"text_config.{field} changed: expected {expected!r}, found {actual!r}"
        )


def validate_mhc_config(config: Mapping[str, Any]) -> MHCConfig:
    """Validate the official four-stream, 20-iteration mHC contract."""

    if not isinstance(config, Mapping):
        raise MHCContractError("config must be an object")
    text = _text_config(config)
    for field, expected in {
        "model_type": "glm5_next_text",
        "hidden_size": 4096,
        "num_hidden_layers": 45,
        "mhc": True,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        "rms_norm_eps": 1e-5,
    }.items():
        _exact(text, field, expected)
    return MHCConfig()


_MHC_SPECS: Final[dict[str, tuple[tuple[int, ...], str]]] = {
    "hc_attn_fn": ((24, 16384), "BF16"),
    "hc_attn_base": ((24,), "F32"),
    "hc_attn_scale": ((3,), "F32"),
    "hc_ffn_fn": ((24, 16384), "BF16"),
    "hc_ffn_base": ((24,), "F32"),
    "hc_ffn_scale": ((3,), "F32"),
}


def _tensor_metadata(value: Any) -> tuple[tuple[int, ...] | None, str | None]:
    if isinstance(value, Mapping):
        shape, dtype = value.get("shape"), value.get("dtype")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        shape, dtype = value, None
    else:
        shape, dtype = getattr(value, "shape", None), getattr(value, "dtype", None)
    parsed_shape = None
    if shape is not None:
        try:
            parsed_shape = tuple(int(dim) for dim in shape)
        except (TypeError, ValueError) as exc:
            raise MHCContractError(f"invalid tensor shape metadata: {shape!r}") from exc
    if dtype is not None:
        dtype = str(dtype).upper().replace("BFLOAT16", "BF16").replace("FLOAT32", "F32")
    return parsed_shape, dtype


def validate_mhc_weights(weights: Mapping[str, Any]) -> None:
    """Validate the six mHC tensors on every main decoder layer (not MTP)."""

    if not isinstance(weights, Mapping):
        raise MHCContractError("weights must be an object")
    found: dict[int, dict[str, Any]] = {}
    for name, value in weights.items():
        if not isinstance(name, str):
            raise MHCContractError("weight names must be strings")
        if name.startswith("model.layers.") and "hc_" in name:
            raise MHCContractError("glm_moe_dsa weight aliases are forbidden")
        match = _MHC_NAME_RE.match(name)
        if match is not None:
            found.setdefault(int(match.group(1)), {})[match.group(2)] = value
        elif name.startswith("model.language_model.layers.") and ".hc_" in name:
            raise MHCContractError(f"unexpected mHC tensor: {name}")

    expected_layers = set(OFFICIAL_MHC_LAYERS)
    if set(found) != expected_layers:
        missing = sorted(expected_layers - set(found))
        extra = sorted(set(found) - expected_layers)
        raise MHCContractError(
            f"mHC layer placement changed: missing={missing}, extra={extra}"
        )
    for layer in OFFICIAL_MHC_LAYERS:
        tensors = found[layer]
        missing = sorted(set(_MHC_SPECS) - set(tensors))
        if missing:
            raise MHCContractError(f"mHC layer {layer} is missing tensors: {missing}")
        for suffix, (expected_shape, expected_dtype) in _MHC_SPECS.items():
            shape, dtype = _tensor_metadata(tensors[suffix])
            if shape is not None and shape != expected_shape:
                raise MHCContractError(
                    f"mHC layer {layer} {suffix} shape changed: expected {expected_shape}, found {shape}"
                )
            if dtype is not None and dtype != expected_dtype:
                raise MHCContractError(
                    f"mHC layer {layer} {suffix} dtype changed: expected {expected_dtype}, found {dtype}"
                )


@lru_cache(maxsize=1)
def _mlx_runtime() -> tuple[Any, Any]:
    return importlib.import_module("mlx.core"), importlib.import_module("mlx.nn")


@lru_cache(maxsize=1)
def _compiled_mhc_residual():
    """Compile the residual placement+mix graph for decode-sized calls."""

    mx, _ = _mlx_runtime()

    def _run(post, comb, branch_output, residual):
        dtype = residual.dtype
        placed = post.astype(dtype)[..., None] * branch_output[..., None, :]
        mixed = mx.matmul(comb.astype(dtype).swapaxes(-1, -2), residual)
        return placed + mixed

    return mx.compile(_run)


def apply_mhc_residual(post, comb, branch_output, residual):
    """Apply the exact post-placement and transposed stream mixer equation."""

    if (
        os.environ.get("GLM5_NEXT_MHC_COMPILE", "1") == "1"
        and residual.shape[1] <= 4
    ):
        return _compiled_mhc_residual()(post, comb, branch_output, residual)
    mx, _ = _mlx_runtime()
    dtype = residual.dtype
    placed = post.astype(dtype)[..., None] * branch_output[..., None, :]
    mixed = mx.matmul(comb.astype(dtype).swapaxes(-1, -2), residual)
    return placed + mixed


def _mhc_mix_pyre(
    hidden_streams,
    fn_pre,
    fn_post,
    fn_comb,
    base_pre,
    base_post,
    base_comb,
    scale_pre,
    scale_post,
    scale_comb,
    config: MHCConfig,
):
    """Pure functional form of the mHC mixer, exactly the class semantics.

    Exposed separately so decode can run the identical math through a compiled
    graph: the 20-iteration Sinkhorn loop otherwise launches ~40 tiny kernels
    per call, which dominates single-token decode latency.
    """

    mx, _ = _mlx_runtime()
    input_dtype = hidden_streams.dtype
    flat = hidden_streams.reshape(*hidden_streams.shape[:2], -1).astype(mx.float32)
    variance = mx.mean(flat * flat, axis=-1, keepdims=True)
    flat = flat * mx.rsqrt(variance + config.rms_norm_eps)
    hc = config.streams
    pre_w = flat @ fn_pre.astype(mx.float32).swapaxes(-1, -2)
    post_w = flat @ fn_post.astype(mx.float32).swapaxes(-1, -2)
    comb_w = flat @ fn_comb.astype(mx.float32).swapaxes(-1, -2)
    base_pre = base_pre.astype(mx.float32)
    base_post = base_post.astype(mx.float32)
    base_comb = base_comb.astype(mx.float32)
    pre_scale = scale_pre.astype(mx.float32)
    post_scale = scale_post.astype(mx.float32)
    comb_scale = scale_comb.astype(mx.float32)
    pre = mx.sigmoid(pre_w * pre_scale + base_pre) + config.eps
    post = 2.0 * mx.sigmoid(post_w * post_scale + base_post)
    comb_logits = comb_w.reshape(*comb_w.shape[:-1], hc, hc) * comb_scale
    comb_logits = comb_logits + base_comb.reshape(hc, hc)
    comb = mx.softmax(comb_logits, axis=-1, precise=True) + config.eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + config.eps)
    for _ in range(config.sinkhorn_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + config.eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + config.eps)
    collapsed = (pre[..., None] * hidden_streams.astype(mx.float32)).sum(axis=2)
    return post, comb, collapsed.astype(input_dtype)


@lru_cache(maxsize=4)
def _compiled_mhc_mix(streams: int, eps: float, rms_eps: float, iters: int):
    """Compile the pure mixer once per config; shapeless for prefill/decode."""

    mx, _ = _mlx_runtime()
    config = MHCConfig(
        streams=streams, eps=eps, rms_norm_eps=rms_eps, sinkhorn_iters=iters
    )

    def _run(
        hidden_streams,
        fn_pre,
        fn_post,
        fn_comb,
        base_pre,
        base_post,
        base_comb,
        scale_pre,
        scale_post,
        scale_comb,
    ):
        return _mhc_mix_pyre(
            hidden_streams,
            fn_pre,
            fn_post,
            fn_comb,
            base_pre,
            base_post,
            base_comb,
            scale_pre,
            scale_post,
            scale_comb,
            config,
        )

    # Concrete-shape compilation: shapeless tracing cannot infer the
    # comb (.., hc*hc) -> (.., hc, hc) reshape, so prefill stays eager.
    return mx.compile(_run)


@lru_cache(maxsize=1)
def make_mhc_class():
    """Return the lazily constructed MLX ``Glm5NextMHC`` module class."""

    mx, nn = _mlx_runtime()

    class Glm5NextMHC(nn.Module):
        def __init__(self, config: MHCConfig | None = None):
            super().__init__()
            config = MHCConfig() if config is None else config
            self.config = config
            # Names intentionally mirror Transformers' fn/base/scale module ABI.
            self.fn = (
                mx.random.normal((config.mix_size, config.streams * config.hidden_size))
                * 0.02
            )
            self.base = mx.zeros((config.mix_size,), dtype=mx.float32)
            self.scale = mx.ones((3,), dtype=mx.float32)

        def __call__(self, hidden_streams):
            if hidden_streams.ndim != 4:
                raise ValueError(
                    "mHC input must have shape [batch, sequence, streams, hidden]"
                )
            if hidden_streams.shape[2:] != (
                self.config.streams,
                self.config.hidden_size,
            ):
                raise ValueError(
                    "mHC input tail changed: expected "
                    f"({self.config.streams}, {self.config.hidden_size}), "
                    f"found {hidden_streams.shape[2:]}"
                )
            input_dtype = hidden_streams.dtype
            if (
                os.environ.get("GLM5_NEXT_MHC_COMPILE", "1") == "1"
                and hidden_streams.shape[1] <= 4
            ):
                # Decode/verify-sized calls: the compiled graph eliminates
                # ~40 tiny kernel launches plus per-call Python graph
                # construction.  Prefill keeps the eager path (identical
                # math, no reshape-specialization risk on long sequences).
                hc = self.config.streams
                if not getattr(self, "_compiled_logged", False):
                    self._compiled_logged = True
                    import logging

                    logging.getLogger(__name__).info(
                        "GLM5-Next compiled mHC decode engaged"
                    )
                mixer = _compiled_mhc_mix(
                    hc,
                    self.config.eps,
                    self.config.rms_norm_eps,
                    self.config.sinkhorn_iters,
                )
                # Split eagerly (views) so the compiled graph holds only
                # matmul/elementwise/reduction primitives.
                fn, base, scale = self.fn, self.base, self.scale
                return mixer(
                    hidden_streams,
                    fn[:hc],
                    fn[hc : 2 * hc],
                    fn[2 * hc :],
                    base[:hc],
                    base[hc : 2 * hc],
                    base[2 * hc :],
                    scale[0],
                    scale[1],
                    scale[2],
                )
            flat = hidden_streams.reshape(*hidden_streams.shape[:2], -1).astype(
                mx.float32
            )
            variance = mx.mean(flat * flat, axis=-1, keepdims=True)
            flat = flat * mx.rsqrt(variance + self.config.rms_norm_eps)
            logits = flat @ self.fn.astype(mx.float32).swapaxes(-1, -2)
            hc = self.config.streams
            pre_w, post_w, comb_w = mx.split(logits, [hc, 2 * hc], axis=-1)
            pre_b, post_b, comb_b = mx.split(self.base.astype(mx.float32), [hc, 2 * hc])
            pre_scale, post_scale, comb_scale = [
                self.scale[i].astype(mx.float32) for i in range(3)
            ]

            pre = mx.sigmoid(pre_w * pre_scale + pre_b) + self.config.eps
            post = 2.0 * mx.sigmoid(post_w * post_scale + post_b)
            comb_logits = comb_w.reshape(*comb_w.shape[:-1], hc, hc) * comb_scale
            comb_logits = comb_logits + comb_b.reshape(hc, hc)
            comb = mx.softmax(comb_logits, axis=-1, precise=True) + self.config.eps
            comb = comb / (comb.sum(axis=-2, keepdims=True) + self.config.eps)
            for _ in range(self.config.sinkhorn_iters - 1):
                comb = comb / (comb.sum(axis=-1, keepdims=True) + self.config.eps)
                comb = comb / (comb.sum(axis=-2, keepdims=True) + self.config.eps)
            collapsed = (pre[..., None] * hidden_streams.astype(mx.float32)).sum(axis=2)
            return post, comb, collapsed.astype(input_dtype)

    Glm5NextMHC.__name__ = "Glm5NextMHC"
    Glm5NextMHC.__qualname__ = "Glm5NextMHC"
    return Glm5NextMHC


@lru_cache(maxsize=1)
def make_hyper_head_class():
    """Return the official unweighted final stream-collapse module class."""

    mx, nn = _mlx_runtime()

    class Glm5NextHyperHead(nn.Module):
        def __call__(self, hidden_streams):
            return mx.mean(hidden_streams, axis=2)

    Glm5NextHyperHead.__name__ = "Glm5NextHyperHead"
    Glm5NextHyperHead.__qualname__ = "Glm5NextHyperHead"
    return Glm5NextHyperHead


__all__ = [
    "MHCConfig",
    "MHCContractError",
    "OFFICIAL_MHC_LAYERS",
    "TRANSFORMERS_REFERENCE",
    "apply_mhc_residual",
    "make_hyper_head_class",
    "make_mhc_class",
    "validate_mhc_config",
    "validate_mhc_weights",
]
