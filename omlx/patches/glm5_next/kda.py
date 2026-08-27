"""GLM-5.3-Flash Kimi Delta Attention contracts and MLX primitive.

The source contract in this module is pinned to ``zai-org/GLM-5.3-Flash``
revision ``84c6a6aa9497188e15a635ba793b0f95a79b1033`` and the corresponding
Transformers implementation (``eb4d9e2a64a013bec12289288b85d0b1210ba0aa``).

Importing this module is intentionally MLX-free.  Runtime classes are created
by :func:`make_kda_class`; this lets conversion and source validation run on
machines which do not have MLX installed and avoids initializing Metal in
controller processes.
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

OFFICIAL_KDA_LAYERS: Final = tuple(i for i in range(45) if i % 4 != 3)
OFFICIAL_DSA_LAYERS: Final = tuple(range(3, 45, 4))
TRANSFORMERS_REFERENCE: Final = "eb4d9e2a64a013bec12289288b85d0b1210ba0aa"

_PREFIX: Final = "model.language_model.layers."
_KDA_NAME_RE: Final = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.self_attn\.(.+)$"
)


class KDAContractError(ValueError):
    """The configuration or weights do not implement pinned GLM-5.3 KDA."""


@dataclass(frozen=True, slots=True)
class KDAConfig:
    hidden_size: int = 4096
    num_heads: int = 64
    head_dim: int = 128
    conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"

    @property
    def projection_dim(self) -> int:
        return self.num_heads * self.head_dim


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text = config.get("text_config", config)
    if not isinstance(text, Mapping):
        raise KDAContractError("text_config must be an object")
    return text


def _exact(container: Mapping[str, Any], field: str, expected: Any, where: str) -> None:
    actual = container.get(field)
    if actual != expected or (
        isinstance(expected, int)
        and not isinstance(expected, bool)
        and isinstance(actual, bool)
    ):
        raise KDAContractError(
            f"{where}.{field} changed: expected {expected!r}, found {actual!r}"
        )


def validate_kda_config(config: Mapping[str, Any]) -> KDAConfig:
    """Validate and return the exact official KDA sub-configuration.

    ``config`` may be the top-level multimodal config or its ``text_config``.
    The layer numbers are deliberately zero-based; Kimi Linear's historical
    one-based schedule must not be reused for this architecture.
    """

    if not isinstance(config, Mapping):
        raise KDAContractError("config must be an object")
    text = _text_config(config)
    for field, expected in {
        "model_type": "glm5_next_text",
        "hidden_size": 4096,
        "num_hidden_layers": 45,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
    }.items():
        _exact(text, field, expected, "text_config")

    linear = text.get("linear_attn_config")
    if not isinstance(linear, Mapping):
        raise KDAContractError("text_config.linear_attn_config must be an object")
    for field, expected in {
        "num_heads": 64,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": list(OFFICIAL_KDA_LAYERS),
        "full_attn_layers": list(OFFICIAL_DSA_LAYERS),
    }.items():
        _exact(linear, field, expected, "text_config.linear_attn_config")

    expected_types = [
        "linear_attention" if i in OFFICIAL_KDA_LAYERS else "deepseek_sparse_attention"
        for i in range(45)
    ]
    _exact(text, "layer_types", expected_types, "text_config")
    return KDAConfig()


_KDA_WEIGHT_SPECS: Final[dict[str, tuple[tuple[int, ...], str]]] = {
    "A_log": ((64,), "F32"),
    "b_proj.weight": ((64, 4096), "BF16"),
    "dt_bias": ((8192,), "F32"),
    "f_a_proj.weight": ((128, 4096), "BF16"),
    "f_b_proj.weight": ((8192, 128), "BF16"),
    "g_a_proj.weight": ((128, 4096), "BF16"),
    "g_b_proj.weight": ((8192, 128), "BF16"),
    "k_conv1d.weight": ((8192, 1, 4), "BF16"),
    "k_proj.weight": ((8192, 4096), "BF16"),
    "o_norm.weight": ((128,), "BF16"),
    "o_proj.weight": ((4096, 8192), "BF16"),
    "q_conv1d.weight": ((8192, 1, 4), "BF16"),
    "q_proj.weight": ((8192, 4096), "BF16"),
    "v_conv1d.weight": ((8192, 1, 4), "BF16"),
    "v_proj.weight": ((8192, 4096), "BF16"),
}


def _tensor_metadata(value: Any) -> tuple[tuple[int, ...] | None, str | None]:
    """Read metadata from a safetensors header or an array without loading it."""

    if isinstance(value, Mapping):
        shape = value.get("shape")
        dtype = value.get("dtype")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        shape, dtype = value, None
    else:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
    parsed_shape = None
    if shape is not None:
        try:
            parsed_shape = tuple(int(dim) for dim in shape)
        except (TypeError, ValueError) as exc:
            raise KDAContractError(f"invalid tensor shape metadata: {shape!r}") from exc
    if dtype is not None:
        dtype = str(dtype).upper().replace("BFLOAT16", "BF16").replace("FLOAT32", "F32")
    return parsed_shape, dtype


def validate_kda_weights(weights: Mapping[str, Any]) -> None:
    """Validate all 34 official KDA layer families, shapes, and source dtypes.

    Values may be safetensors header dictionaries, arrays, bare shapes, or shard
    names.  Shape/dtype validation is performed whenever that metadata exists;
    a model index still receives strict name and placement validation.
    """

    if not isinstance(weights, Mapping):
        raise KDAContractError("weights must be an object")
    found: dict[int, dict[str, Any]] = {}
    unexpected: list[str] = []
    misplaced: set[int] = set()
    kda_suffixes = set(_KDA_WEIGHT_SPECS)
    # ``o_proj.weight`` is shared with DSA/MTP.  Every other suffix is specific
    # to this KDA checkpoint family and therefore proves illegal placement.
    placement_markers = kda_suffixes - {"o_proj.weight"}
    for name, value in weights.items():
        if not isinstance(name, str):
            raise KDAContractError("weight names must be strings")
        match = _KDA_NAME_RE.match(name)
        if match is None:
            if name.startswith("model.layers.") and any(
                name.endswith(s) for s in kda_suffixes
            ):
                raise KDAContractError("glm_moe_dsa weight aliases are forbidden")
            continue
        layer, suffix = int(match.group(1)), match.group(2)
        if layer in OFFICIAL_KDA_LAYERS:
            if suffix not in kda_suffixes:
                unexpected.append(name)
            else:
                found.setdefault(layer, {})[suffix] = value
        elif suffix in placement_markers:
            misplaced.add(layer)

    if unexpected:
        raise KDAContractError(f"unexpected KDA tensor: {unexpected[0]}")

    expected_layers = set(OFFICIAL_KDA_LAYERS)
    actual_layers = set(found) | misplaced
    if actual_layers != expected_layers:
        missing = sorted(expected_layers - set(found))
        extra = sorted(actual_layers - expected_layers)
        raise KDAContractError(
            f"KDA layer placement changed: missing={missing}, extra={extra}"
        )

    for layer in OFFICIAL_KDA_LAYERS:
        tensors = found[layer]
        missing = sorted(kda_suffixes - set(tensors))
        if missing:
            raise KDAContractError(f"KDA layer {layer} is missing tensors: {missing}")
        for suffix, (expected_shape, expected_dtype) in _KDA_WEIGHT_SPECS.items():
            shape, dtype = _tensor_metadata(tensors[suffix])
            if shape is not None and shape != expected_shape:
                raise KDAContractError(
                    f"KDA layer {layer} {suffix} shape changed: "
                    f"expected {expected_shape}, found {shape}"
                )
            if dtype is not None and dtype != expected_dtype:
                raise KDAContractError(
                    f"KDA layer {layer} {suffix} dtype changed: "
                    f"expected {expected_dtype}, found {dtype}"
                )


@dataclass(frozen=True, slots=True)
class KDACacheSnapshot:
    q_conv: Any
    k_conv: Any
    v_conv: Any
    recurrent: Any
    offset: int


class KDACache:
    """Four-array recurrent cache with explicit transactional rollback.

    Recurrent KDA state cannot be trimmed by slicing.  Snapshots retain MLX
    array references (MLX arrays are immutable), so speculative rollback is
    exact without eagerly copying the roughly 4 MiB FP32 state per layer.
    """

    def __init__(self) -> None:
        self.cache: list[Any] = [None, None, None, None]
        self.offset = 0
        self.lengths = None
        self.left_padding = None
        self._transactions: list[KDACacheSnapshot] = []

    def __getitem__(self, index: int) -> Any:
        return self.cache[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self.cache[index] = value

    @property
    def state(self) -> list[Any]:
        return self.cache

    @state.setter
    def state(self, value: Sequence[Any]) -> None:
        if len(value) != 4:
            raise ValueError(
                "KDA cache state must contain q, k, v, and recurrent arrays"
            )
        self.cache = list(value)

    @property
    def meta_state(self) -> tuple[str]:
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, value: Sequence[Any]) -> None:
        if len(value) != 1:
            raise ValueError("KDA cache metadata must contain only its token offset")
        self.offset = int(value[0])

    def is_trimmable(self) -> bool:
        return False

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return all(value is None for value in self.cache)

    @property
    def nbytes(self) -> int:
        return sum(int(value.nbytes) for value in self.cache if value is not None)

    def snapshot(self) -> KDACacheSnapshot:
        return KDACacheSnapshot(*self.cache, self.offset)

    def restore(self, snapshot: KDACacheSnapshot) -> None:
        if not isinstance(snapshot, KDACacheSnapshot):
            raise TypeError("snapshot must be a KDACacheSnapshot")
        self.cache = [
            snapshot.q_conv,
            snapshot.k_conv,
            snapshot.v_conv,
            snapshot.recurrent,
        ]
        self.offset = snapshot.offset

    def begin_update(self) -> KDACacheSnapshot:
        snapshot = self.snapshot()
        self._transactions.append(snapshot)
        return snapshot

    def commit(self, snapshot: KDACacheSnapshot | None = None) -> None:
        if not self._transactions:
            raise RuntimeError("no KDA cache transaction is active")
        expected = self._transactions[-1]
        if snapshot is not None and snapshot is not expected:
            raise RuntimeError("KDA cache transactions must commit in LIFO order")
        self._transactions.pop()

    def rollback(self, snapshot: KDACacheSnapshot | None = None) -> None:
        if snapshot is None:
            if not self._transactions:
                raise RuntimeError("no KDA cache transaction is active")
            snapshot = self._transactions[-1]
        if self._transactions:
            if snapshot is not self._transactions[-1]:
                raise RuntimeError(
                    "KDA cache transactions must roll back in LIFO order"
                )
            self._transactions.pop()
        self.restore(snapshot)

    def update(
        self, q_conv: Any, k_conv: Any, v_conv: Any, recurrent: Any, tokens: int
    ) -> None:
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("tokens must be a non-negative integer")
        self.cache = [q_conv, k_conv, v_conv, recurrent]
        self.offset += tokens

    def trim(self, _: int) -> int:
        """Signal that recurrent state needs snapshot restoration, not slicing."""

        return 0

    def advance(self, tokens: int) -> None:
        self.offset += tokens
        if self.lengths is not None:
            self.lengths -= tokens
        if self.left_padding is not None:
            self.left_padding -= tokens

    def make_mask(self, length: int):
        # This method intentionally imports MLX only on the batched-cache path.
        if self.left_padding is None and self.lengths is None:
            return None
        mx = importlib.import_module("mlx.core")
        positions = mx.arange(length)
        if self.left_padding is not None:
            return positions >= self.left_padding[:, None]
        return positions < self.lengths[:, None]


def _validate_recurrence_backend(module: Any) -> tuple[Any, Any]:
    """Prove the mlx-lm recurrence ABI before dispatching to it."""

    ops = getattr(module, "gated_delta_ops", None)
    kernel = getattr(module, "gated_delta_kernel", None)
    expected = ("q", "k", "v", "g", "beta", "state", "mask")
    for name, function in (("gated_delta_ops", ops), ("gated_delta_kernel", kernel)):
        if not callable(function):
            raise RuntimeError(f"mlx-lm is missing required {name}")
        parameters = tuple(inspect.signature(function).parameters)
        if parameters != expected:
            raise RuntimeError(
                f"mlx-lm {name} ABI changed: expected {expected}, found {parameters}"
            )
    return ops, kernel


@lru_cache(maxsize=1)
def _mlx_runtime() -> tuple[Any, Any, Any, Any]:
    mx = importlib.import_module("mlx.core")
    nn = importlib.import_module("mlx.nn")
    delta = importlib.import_module("mlx_lm.models.gated_delta")
    ops, kernel = _validate_recurrence_backend(delta)
    return mx, nn, ops, kernel


_LOGGED: set = set()


def _log_once(key: str, message: str) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    import logging

    logging.getLogger(__name__).info("%s", message)


@lru_cache(maxsize=2)
def _compiled_kda_decode(heads: int, head_dim: int, eps: float, gate_lower: float):
    """One compiled graph for every KDA layer at decode shapes.

    Replays ``Glm5NextKDA.__call__`` exactly for the mask-free single-row
    decode case: short-conv state updates, fp32 normalization, the pinned
    gated-delta Metal kernel, and the output gate.  Weights and states are
    arguments so all 34 KDA layers share one compiled graph.  Returns the
    layer output plus the four updated cache states; the caller owns the
    cache mutation.
    """

    mx, nn, delta_ops, delta_kernel = _mlx_runtime()

    def _conv_tap(conv_input, weight, length):
        # Exact groups=C kernel-4 Conv1d without the module wrapper.
        # MLX Conv1d weight layout is (channels, kernel, in/groups=1).
        out = None
        for tap in range(4):
            piece = conv_input[:, tap : tap + length, :] * weight[:, tap, 0]
            out = piece if out is None else out + piece
        return out

    def _run(
        x,
        qw,
        kw,
        vw,
        qcw,
        kcw,
        vcw,
        fa_w,
        fb_w,
        b_w,
        ga_w,
        gb_w,
        onorm_w,
        ow,
        a_log,
        dt_bias,
        q_state,
        k_state,
        v_state,
        recurrent,
    ):
        batch, length, _ = x.shape
        input_dtype = x.dtype
        keep = 3

        def _proj(w):
            return x @ w.swapaxes(-1, -2)

        conv_input = mx.concatenate([q_state.astype(x.dtype), _proj(qw)], axis=1)
        q = nn.silu(_conv_tap(conv_input, qcw, length))
        new_q_state = mx.contiguous(conv_input[:, -keep:, :])
        conv_input = mx.concatenate([k_state.astype(x.dtype), _proj(kw)], axis=1)
        k = nn.silu(_conv_tap(conv_input, kcw, length))
        new_k_state = mx.contiguous(conv_input[:, -keep:, :])
        conv_input = mx.concatenate([v_state.astype(x.dtype), _proj(vw)], axis=1)
        v = nn.silu(_conv_tap(conv_input, vcw, length))
        new_v_state = mx.contiguous(conv_input[:, -keep:, :])

        shape = (batch, length, heads, head_dim)
        q = q.reshape(shape).astype(mx.float32)
        k = k.reshape(shape).astype(mx.float32)
        v = v.reshape(shape).astype(mx.float32)
        q = q / mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        q = q * (head_dim**-0.5)

        gate_logits = ((_proj(fa_w) @ fb_w.swapaxes(-1, -2))).astype(mx.float32)
        gate_logits = gate_logits.reshape(shape) + dt_bias.astype(mx.float32).reshape(
            1, 1, heads, head_dim
        )
        rate = mx.exp(a_log.astype(mx.float32)).reshape(1, 1, heads, 1)
        log_decay = gate_lower * mx.sigmoid(rate * gate_logits)
        decay = mx.exp(log_decay)
        beta = mx.sigmoid(_proj(b_w).astype(mx.float32))

        recurrent_f = recurrent.astype(mx.float32)
        output, recurrent_f = delta_kernel(
            q, k, v, decay, beta, recurrent_f, None
        )
        recurrent_f = recurrent_f.astype(mx.float32)

        gate = (_proj(ga_w) @ gb_w.swapaxes(-1, -2)).astype(mx.float32).reshape(shape)
        normed = output.astype(mx.float32)
        variance = mx.mean(normed * normed, axis=-1, keepdims=True)
        normed = normed * mx.rsqrt(variance + eps)
        normed = normed * onorm_w.astype(mx.float32)
        normed = normed * mx.sigmoid(gate)
        normed = normed.astype(input_dtype).reshape(batch, length, -1)
        out = (normed @ ow.swapaxes(-1, -2)).astype(input_dtype)
        return out, new_q_state, new_k_state, new_v_state, recurrent_f

    return mx.compile(_run)


@lru_cache(maxsize=1)
def make_kda_class():
    """Return the lazily constructed MLX ``Glm5NextKDA`` module class."""

    mx, nn, delta_ops, delta_kernel = _mlx_runtime()

    class Glm5NextKDA(nn.Module):
        def __init__(self, config: KDAConfig | None = None):
            super().__init__()
            config = KDAConfig() if config is None else config
            self.config = config
            hidden = config.hidden_size
            projected = config.projection_dim
            heads, dim = config.num_heads, config.head_dim

            self.q_proj = nn.Linear(hidden, projected, bias=False)
            self.k_proj = nn.Linear(hidden, projected, bias=False)
            self.v_proj = nn.Linear(hidden, projected, bias=False)
            self.q_conv1d = nn.Conv1d(
                projected,
                projected,
                config.conv_kernel_size,
                groups=projected,
                bias=False,
            )
            self.k_conv1d = nn.Conv1d(
                projected,
                projected,
                config.conv_kernel_size,
                groups=projected,
                bias=False,
            )
            self.v_conv1d = nn.Conv1d(
                projected,
                projected,
                config.conv_kernel_size,
                groups=projected,
                bias=False,
            )
            self.f_a_proj = nn.Linear(hidden, dim, bias=False)
            self.f_b_proj = nn.Linear(dim, projected, bias=False)
            self.b_proj = nn.Linear(hidden, heads, bias=False)
            self.g_a_proj = nn.Linear(hidden, dim, bias=False)
            self.g_b_proj = nn.Linear(dim, projected, bias=False)
            self.o_norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.o_proj = nn.Linear(projected, hidden, bias=False)
            # These two checkpoint tensors are strict FP32 state-space parameters.
            self.A_log = mx.zeros((heads,), dtype=mx.float32)
            self.dt_bias = mx.zeros((projected,), dtype=mx.float32)

        def _short_conv(self, conv, x, state, mask, lengths):
            if mask is not None:
                x = mx.where(mask[..., None], x, mx.array(0, dtype=x.dtype))
            keep = self.config.conv_kernel_size - 1
            if state is None:
                state = mx.zeros((x.shape[0], keep, x.shape[-1]), dtype=x.dtype)
            conv_input = mx.concatenate([state.astype(x.dtype), x], axis=1)
            out = nn.silu(conv(conv_input))
            if lengths is None:
                new_state = mx.contiguous(conv_input[:, -keep:, :])
            else:
                ends = mx.clip(lengths, 0, x.shape[1])
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                new_state = mx.take_along_axis(conv_input, positions, axis=1)
            return out, new_state

        def __call__(self, x, mask=None, cache=None, *, use_kernel: bool = True):
            batch, length, _ = x.shape
            input_dtype = x.dtype
            if mask is None and cache is not None and hasattr(cache, "make_mask"):
                mask = cache.make_mask(length)
            if mask is not None:
                x = mx.where(mask[..., None], x, mx.array(0, dtype=x.dtype))

            states = (
                [None, None, None, None]
                if cache is None
                else [cache[i] for i in range(4)]
            )
            lengths = None if cache is None else getattr(cache, "lengths", None)
            if (
                use_kernel
                and length <= 4
                and mask is None
                and lengths is None
                and states[3] is not None
                and os.environ.get("GLM5_NEXT_KDA_COMPILE", "1") == "1"
                and mx.default_device() == mx.gpu
                and mx.metal.is_available()
            ):
                _log_once("kda", "GLM5-Next compiled KDA decode engaged")
                out, q_s, k_s, v_s, rec = _compiled_kda_decode(
                    self.config.num_heads,
                    self.config.head_dim,
                    self.config.rms_norm_eps,
                    self.config.gate_lower_bound,
                )(
                    x,
                    self.q_proj.weight,
                    self.k_proj.weight,
                    self.v_proj.weight,
                    self.q_conv1d.weight,
                    self.k_conv1d.weight,
                    self.v_conv1d.weight,
                    self.f_a_proj.weight,
                    self.f_b_proj.weight,
                    self.b_proj.weight,
                    self.g_a_proj.weight,
                    self.g_b_proj.weight,
                    self.o_norm.weight,
                    self.o_proj.weight,
                    self.A_log,
                    self.dt_bias,
                    states[0],
                    states[1],
                    states[2],
                    states[3],
                )
                if isinstance(cache, KDACache):
                    cache.update(q_s, k_s, v_s, rec, length)
                else:
                    cache[0], cache[1], cache[2], cache[3] = q_s, k_s, v_s, rec
                    if hasattr(cache, "advance"):
                        cache.advance(length)
                return out

            q, q_state = self._short_conv(
                self.q_conv1d, self.q_proj(x), states[0], mask, lengths
            )
            k, k_state = self._short_conv(
                self.k_conv1d, self.k_proj(x), states[1], mask, lengths
            )
            v, v_state = self._short_conv(
                self.v_conv1d, self.v_proj(x), states[2], mask, lengths
            )

            shape = (batch, length, self.config.num_heads, self.config.head_dim)
            q = q.reshape(shape).astype(mx.float32)
            k = k.reshape(shape).astype(mx.float32)
            v = v.reshape(shape).astype(mx.float32)
            q = q / mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
            k = k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
            q = q * (self.config.head_dim**-0.5)

            gate_logits = (
                self.f_b_proj(self.f_a_proj(x)).astype(mx.float32).reshape(shape)
            )
            gate_logits = gate_logits + self.dt_bias.astype(mx.float32).reshape(
                1, 1, self.config.num_heads, self.config.head_dim
            )
            rate = mx.exp(self.A_log.astype(mx.float32)).reshape(
                1, 1, self.config.num_heads, 1
            )
            log_decay = self.config.gate_lower_bound * mx.sigmoid(rate * gate_logits)
            decay = mx.exp(log_decay)
            beta = mx.sigmoid(self.b_proj(x).astype(mx.float32))

            recurrent = states[3]
            if recurrent is None:
                recurrent = mx.zeros(
                    (
                        batch,
                        self.config.num_heads,
                        self.config.head_dim,
                        self.config.head_dim,
                    ),
                    dtype=mx.float32,
                )
            else:
                recurrent = recurrent.astype(mx.float32)

            can_kernel = (
                use_kernel and mx.default_device() == mx.gpu and mx.metal.is_available()
            )
            backend = delta_kernel if can_kernel else delta_ops
            output, recurrent = backend(q, k, v, decay, beta, recurrent, mask)
            recurrent = recurrent.astype(mx.float32)

            if cache is not None:
                if isinstance(cache, KDACache):
                    cache.update(q_state, k_state, v_state, recurrent, length)
                else:
                    cache[0], cache[1], cache[2], cache[3] = (
                        q_state,
                        k_state,
                        v_state,
                        recurrent,
                    )
                    if hasattr(cache, "advance"):
                        cache.advance(length)

            gate = self.g_b_proj(self.g_a_proj(x)).astype(mx.float32).reshape(shape)
            normed = output.astype(mx.float32)
            variance = mx.mean(normed * normed, axis=-1, keepdims=True)
            normed = normed * mx.rsqrt(variance + self.config.rms_norm_eps)
            normed = normed * self.o_norm.weight.astype(mx.float32)
            normed = normed * mx.sigmoid(gate)
            normed = normed.astype(input_dtype).reshape(batch, length, -1)
            return self.o_proj(normed).astype(input_dtype)

    Glm5NextKDA.__name__ = "Glm5NextKDA"
    Glm5NextKDA.__qualname__ = "Glm5NextKDA"
    return Glm5NextKDA


__all__ = [
    "KDACache",
    "KDACacheSnapshot",
    "KDAConfig",
    "KDAContractError",
    "OFFICIAL_DSA_LAYERS",
    "OFFICIAL_KDA_LAYERS",
    "TRANSFORMERS_REFERENCE",
    "make_kda_class",
    "validate_kda_config",
    "validate_kda_weights",
]
