# SPDX-License-Identifier: Apache-2.0
"""Strict GLM-5.3-Flash MoE checkpoint boundary.

The official checkpoint stores 288 routed experts as individual block-FP8
matrices.  MLX switch layers consume one packed expert axis.  This module
performs only that lossless packing; it deliberately keeps the FP8 inverse
scale grids beside the packed codes.  Dequantising the whole expert bank here
would transiently turn a 320B checkpoint into a dense BF16 model.

There are no MLX imports or arrays at module import time.  The runtime class is
resolved lazily so source inspection and conversion planning never allocate a
288-expert model.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any, Final

OFFICIAL_EXPERTS: Final = 288
OFFICIAL_TOP_K: Final = 8
OFFICIAL_HIDDEN_SIZE: Final = 4_096
OFFICIAL_EXPERT_SIZE: Final = 2_048
OFFICIAL_FP8_BLOCK: Final = 128
GLM5_NEXT_BLOCK_FP8_RUNTIME_READY: Final = True

_EXPERT_KEY = re.compile(
    r"^(?P<prefix>.+\.mlp)\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_PACKED_SUFFIX = ".mlp.switch_mlp.gate_proj.weight"


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _text_config(config: Any) -> Any:
    nested = _get(config, "text_config")
    return config if nested is None else nested


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def validate_moe_config(config: Any) -> None:
    """Require the exact published sigmoid/noaux_tc top-8 router contract."""

    config = _text_config(config)
    expected = {
        "hidden_size": OFFICIAL_HIDDEN_SIZE,
        "moe_intermediate_size": OFFICIAL_EXPERT_SIZE,
        "n_routed_experts": OFFICIAL_EXPERTS,
        "n_shared_experts": 1,
        "num_experts_per_tok": OFFICIAL_TOP_K,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "moe_router_dtype": "float32",
        "n_group": 1,
        "topk_group": 1,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
    }
    mismatches = [
        f"{name}={_get(config, name)!r} (expected {wanted!r})"
        for name, wanted in expected.items()
        if _get(config, name) != wanted
    ]
    if mismatches:
        raise ValueError("Unsupported GLM5-Next MoE config: " + "; ".join(mismatches))


def _expected_projection_shape(projection: str) -> tuple[int, int]:
    if projection == "down_proj":
        return (OFFICIAL_HIDDEN_SIZE, OFFICIAL_EXPERT_SIZE)
    return (OFFICIAL_EXPERT_SIZE, OFFICIAL_HIDDEN_SIZE)


def _expected_scale_shape(projection: str) -> tuple[int, int]:
    rows, columns = _expected_projection_shape(projection)
    return (
        (rows + OFFICIAL_FP8_BLOCK - 1) // OFFICIAL_FP8_BLOCK,
        (columns + OFFICIAL_FP8_BLOCK - 1) // OFFICIAL_FP8_BLOCK,
    )


def _validate_shape(
    weights: Mapping[str, Any], key: str, expected: tuple[int, ...]
) -> None:
    if key not in weights:
        raise ValueError(f"GLM5-Next MoE is missing checkpoint tensor: {key}")
    actual = _shape(weights[key])
    if actual != expected:
        raise ValueError(
            f"Invalid GLM5-Next MoE tensor {key}: found {actual}, expected {expected}"
        )


def _raw_prefixes(weights: Mapping[str, Any]) -> set[str]:
    return {
        match.group("prefix")
        for key in weights
        if (match := _EXPERT_KEY.match(key)) is not None
    }


def _packed_prefixes(weights: Mapping[str, Any]) -> set[str]:
    return {
        key[: -len(_PACKED_SUFFIX)] + ".mlp"
        for key in weights
        if key.endswith(_PACKED_SUFFIX)
    }


def validate_moe_weight_layout(weights: Mapping[str, Any], prefix: str) -> None:
    """Validate one raw or packed official MoE at ``prefix``.

    ``prefix`` ends in ``.mlp``.  The scale grids are mandatory: accepting an
    FP8 code tensor without its inverse scales would silently change weights.
    """

    raw_probe = f"{prefix}.experts.0.gate_proj.weight"
    packed_probe = f"{prefix}.switch_mlp.gate_proj.weight"
    raw = raw_probe in weights
    packed = packed_probe in weights
    if raw == packed:
        state = "both raw and packed" if raw else "neither raw nor packed"
        raise ValueError(f"GLM5-Next MoE {prefix} contains {state} expert tensors")

    _validate_shape(
        weights, f"{prefix}.gate.weight", (OFFICIAL_EXPERTS, OFFICIAL_HIDDEN_SIZE)
    )
    _validate_shape(
        weights, f"{prefix}.gate.e_score_correction_bias", (OFFICIAL_EXPERTS,)
    )

    affine = (
        packed
        and f"{prefix}.switch_mlp.gate_proj.scales" in weights
        and f"{prefix}.switch_mlp.gate_proj.biases" in weights
    )
    nvfp4 = packed and f"{prefix}.switch_mlp.gate_proj.global_scale" in weights
    if affine and nvfp4:
        raise ValueError(f"GLM5-Next MoE {prefix} mixes affine and NVFP4 tensors")
    for projection in ("gate_proj", "up_proj", "down_proj"):
        shape = _expected_projection_shape(projection)
        scale_shape = _expected_scale_shape(projection)
        if raw:
            for expert in range(OFFICIAL_EXPERTS):
                base = f"{prefix}.experts.{expert}.{projection}.weight"
                _validate_shape(weights, base, shape)
                _validate_shape(weights, f"{base}_scale_inv", scale_shape)
        else:
            base = f"{prefix}.switch_mlp.{projection}.weight"
            if nvfp4:
                _validate_shape(
                    weights,
                    base,
                    (OFFICIAL_EXPERTS, shape[0], shape[1] // 8),
                )
                _validate_shape(
                    weights,
                    f"{prefix}.switch_mlp.{projection}.scales",
                    (OFFICIAL_EXPERTS, shape[0], shape[1] // 16),
                )
                _validate_shape(
                    weights,
                    f"{prefix}.switch_mlp.{projection}.global_scale",
                    (OFFICIAL_EXPERTS,),
                )
            elif affine:
                if base not in weights:
                    raise ValueError(
                        f"GLM5-Next MoE is missing checkpoint tensor: {base}"
                    )
                packed_shape = _shape(weights[base])
                scales_key = f"{prefix}.switch_mlp.{projection}.scales"
                biases_key = f"{prefix}.switch_mlp.{projection}.biases"
                if scales_key not in weights or biases_key not in weights:
                    raise ValueError(
                        f"GLM5-Next affine MoE requires weight/scales/biases: {base}"
                    )
                scales_shape = _shape(weights[scales_key])
                if (
                    len(packed_shape) != 3
                    or packed_shape[:2] != (OFFICIAL_EXPERTS, shape[0])
                    or len(scales_shape) != 3
                    or scales_shape[:2] != (OFFICIAL_EXPERTS, shape[0])
                    or _shape(weights[biases_key]) != scales_shape
                    or shape[1] % scales_shape[-1]
                ):
                    raise ValueError(f"Invalid GLM5-Next affine MoE geometry at {base}")
                bits = 32 * packed_shape[-1] // shape[1]
                group_size = shape[1] // scales_shape[-1]
                if bits not in (4, 8) or group_size <= 0:
                    raise ValueError(
                        f"GLM5-Next converted MoE must be affine Q4/Q8 at {base}"
                    )
            else:
                _validate_shape(weights, base, (OFFICIAL_EXPERTS, *shape))
                _validate_shape(
                    weights,
                    f"{base}_scale_inv",
                    (OFFICIAL_EXPERTS, *scale_shape),
                )

    shared = f"{prefix}.shared_experts"
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"{shared}.{projection}.weight"
        if nvfp4:
            shape = _expected_projection_shape(projection)
            global_key = f"{shared}.{projection}.global_scale"
            scales_key = f"{shared}.{projection}.scales"
            if global_key in weights:
                _validate_shape(weights, base, (shape[0], shape[1] // 8))
                _validate_shape(
                    weights,
                    scales_key,
                    (shape[0], shape[1] // 16),
                )
                global_shape = _shape(weights[global_key])
                if global_shape not in ((), (1,)):
                    raise ValueError(
                        "GLM5-Next NVFP4 shared expert global scale must be scalar: "
                        f"{shared}.{projection}.global_scale"
                    )
            else:
                # LibertAI's routed-only ModelOpt artifact intentionally keeps
                # shared experts in BF16.
                if scales_key in weights:
                    raise ValueError(
                        f"GLM5-Next NVFP4 shared expert is missing tensor: {global_key}"
                    )
                _validate_shape(weights, base, shape)
        elif affine:
            # The converted artifact quantizes shared dense experts through
            # standard MLX affine Linear triples as well.
            for suffix in ("weight", "scales", "biases"):
                key = f"{shared}.{projection}.{suffix}"
                if key not in weights:
                    raise ValueError(
                        f"GLM5-Next affine shared expert is missing tensor: {key}"
                    )
        else:
            _validate_shape(weights, base, _expected_projection_shape(projection))
            _validate_shape(
                weights, f"{base}_scale_inv", _expected_scale_shape(projection)
            )


def _mlx_stack(values: list[Any]) -> Any:
    # Importing MLX here (rather than at module scope) is an intentional part
    # of the conversion planner's no-model-allocation contract.
    import mlx.core as mx

    return mx.stack(values, axis=0)


def sanitize_moe_weights(
    weights: Mapping[str, Any],
    *,
    stack_fn: Callable[[list[Any]], Any] | None = None,
) -> dict[str, Any]:
    """Pack every complete official routed-expert family losslessly.

    Unrelated keys are retained verbatim, notably the raw layer-45 MTP and all
    vision tensors.  The operation is idempotent for an already packed family.
    ``stack_fn`` exists for streaming converters/tests that provide their own
    virtual stack implementation; the default is ``mlx.stack``.
    """

    sanitized = dict(weights)
    raw_prefixes = _raw_prefixes(sanitized)
    packed_prefixes = _packed_prefixes(sanitized)
    overlap = raw_prefixes & packed_prefixes
    if overlap:
        raise ValueError(
            "GLM5-Next MoE has ambiguous raw/packed expert families: "
            + ", ".join(sorted(overlap))
        )
    for prefix in sorted(raw_prefixes | packed_prefixes):
        validate_moe_weight_layout(sanitized, prefix)

    stack = _mlx_stack if stack_fn is None else stack_fn
    for prefix in sorted(raw_prefixes):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            values: list[Any] = []
            scales: list[Any] = []
            for expert in range(OFFICIAL_EXPERTS):
                base = f"{prefix}.experts.{expert}.{projection}.weight"
                values.append(sanitized.pop(base))
                scales.append(sanitized.pop(f"{base}_scale_inv"))
            packed = f"{prefix}.switch_mlp.{projection}.weight"
            sanitized[packed] = stack(values)
            sanitized[f"{packed}_scale_inv"] = stack(scales)

    return sanitized


def _runtime_geometry(config: Any, *, validate_official: bool) -> dict[str, Any]:
    config = _text_config(config)
    if validate_official:
        validate_moe_config(config)
    names = (
        "hidden_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "routed_scaling_factor",
        "norm_topk_prob",
        "swiglu_limit",
    )
    values = {name: _get(config, name) for name in names}
    integer_names = (
        "hidden_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
    )
    if any(
        isinstance(values[name], bool)
        or not isinstance(values[name], int)
        or values[name] <= 0
        for name in integer_names
    ):
        raise ValueError("GLM5-Next MoE runtime dimensions must be positive integers")
    if values["num_experts_per_tok"] > values["n_routed_experts"]:
        raise ValueError("GLM5-Next MoE top-k cannot exceed the expert count")
    if not isinstance(values["routed_scaling_factor"], (int, float)):
        raise ValueError("GLM5-Next routed scaling factor must be numeric")
    if values["norm_topk_prob"] is not True:
        raise ValueError("GLM5-Next runtime requires normalized top-k probabilities")
    if not isinstance(values["swiglu_limit"], (int, float)):
        raise ValueError("GLM5-Next SwiGLU limit must be numeric")
    return values


@lru_cache(maxsize=2)
def _compiled_nvfp4_sparse_decode():
    """One compiled graph for every NVFP4 sparse MoE block at decode shapes.

    Mirrors ``_FP32Router`` + ``_ClampedSwitchGLU`` (ScaledNVFP4SwitchLinear)
    + ``_SharedExperts`` (plain dense) exactly.  Weights are passed as
    arguments so all 43 blocks share a single compiled graph.  Only used for
    single-token-style calls; prefill stays on the eager modules.
    """

    import mlx.core as mx
    import mlx.nn as nn

    def _qmm(x, weight, scales, global_scale, indices):
        out = mx.gather_qmm(
            x,
            weight,
            scales,
            rhs_indices=indices,
            transpose=True,
            group_size=16,
            bits=4,
            mode="nvfp4",
            sorted_indices=False,
        )
        selected = global_scale[indices].astype(out.dtype)
        return out * selected[..., None, None]

    def _run(
        x,
        gate_w,
        e_bias,
        sw_gate_w,
        sw_gate_s,
        sw_gate_g,
        sw_up_w,
        sw_up_s,
        sw_up_g,
        sw_down_w,
        sw_down_s,
        sw_down_g,
        sh_gate_w,
        sh_up_w,
        sh_down_w,
        top_k,
        scaling,
        limit,
    ):
        logits = x.astype(mx.float32) @ gate_w.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        selection = scores + e_bias.astype(mx.float32)
        indices = mx.argpartition(-selection, kth=top_k - 1, axis=-1)[..., :top_k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = weights * scaling

        xe = mx.expand_dims(x, (-2, -3))
        up = _qmm(xe, sw_up_w, sw_up_s, sw_up_g, indices)
        gate = _qmm(xe, sw_gate_w, sw_gate_s, sw_gate_g, indices)
        gate = mx.minimum(gate, mx.array(limit, dtype=gate.dtype))
        up = mx.clip(up, -limit, limit)
        routed = _qmm(nn.silu(gate) * up, sw_down_w, sw_down_s, sw_down_g, indices)
        routed = routed.squeeze(-2)
        routed = mx.sum(routed * weights[..., None], axis=-2).astype(x.dtype)

        shared_gate = x @ sh_gate_w.swapaxes(-1, -2)
        shared_up = x @ sh_up_w.swapaxes(-1, -2)
        shared_gate = mx.minimum(shared_gate, mx.array(limit, dtype=shared_gate.dtype))
        shared_up = mx.clip(shared_up, -limit, limit)
        shared = (nn.silu(shared_gate) * shared_up) @ sh_down_w.swapaxes(-1, -2)
        return routed + shared

    return mx.compile(_run)


def _nvfp4_decode_args(block: Any):
    """Return the compiled-graph argument tuple, or None when ineligible."""

    switch = getattr(block, "switch_mlp", None)
    shared = getattr(block, "shared_experts", None)
    gate = getattr(block, "gate", None)
    if switch is None or shared is None or gate is None:
        return None
    if type(switch.gate_proj).__name__ != "ScaledNVFP4SwitchLinear":
        return None
    if "bias" in switch.gate_proj or "bias" in shared.gate_proj:
        return None
    return (
        gate.weight,
        gate.e_score_correction_bias,
        switch.gate_proj["weight"],
        switch.gate_proj["scales"],
        switch.gate_proj["global_scale"],
        switch.up_proj["weight"],
        switch.up_proj["scales"],
        switch.up_proj["global_scale"],
        switch.down_proj["weight"],
        switch.down_proj["scales"],
        switch.down_proj["global_scale"],
        shared.gate_proj.weight,
        shared.up_proj.weight,
        shared.down_proj.weight,
        gate.top_k,
        gate.routed_scaling_factor,
        switch.limit,
    )


@lru_cache(maxsize=2)
def _implementation_class(validate_official: bool = True):
    """Create the exact router/FP8 wrapper only if a runtime instantiates it."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    def _dequant_block_fp8(weight, scale_inv, dtype):
        if weight.ndim < 2:
            raise ValueError("block-FP8 weight must have at least two dimensions")
        rows, columns = weight.shape[-2:]
        expected = (
            (rows + OFFICIAL_FP8_BLOCK - 1) // OFFICIAL_FP8_BLOCK,
            (columns + OFFICIAL_FP8_BLOCK - 1) // OFFICIAL_FP8_BLOCK,
        )
        if tuple(scale_inv.shape[-2:]) != expected or tuple(
            scale_inv.shape[:-2]
        ) != tuple(weight.shape[:-2]):
            raise ValueError(
                "GLM5-Next block-FP8 scale geometry changed: "
                f"weight={tuple(weight.shape)}, scale={tuple(scale_inv.shape)}"
            )
        padded_rows = expected[0] * OFFICIAL_FP8_BLOCK
        padded_columns = expected[1] * OFFICIAL_FP8_BLOCK
        decoded = mx.from_fp8(weight, dtype=dtype)
        decoded = mx.pad(
            decoded,
            [(0, 0)] * (decoded.ndim - 2)
            + [(0, padded_rows - rows), (0, padded_columns - columns)],
        )
        decoded = decoded.reshape(
            *decoded.shape[:-2],
            expected[0],
            OFFICIAL_FP8_BLOCK,
            expected[1],
            OFFICIAL_FP8_BLOCK,
        )
        scaled = decoded * scale_inv.astype(dtype)[..., :, None, :, None]
        scaled = scaled.reshape(*weight.shape[:-2], padded_rows, padded_columns)
        return scaled[..., :rows, :columns]

    class _BlockFP8Linear(nn.Linear):
        def __init__(self, input_dims, output_dims, bias=False):
            super().__init__(input_dims, output_dims, bias=bias)
            self.weight_scale_inv = mx.ones(
                (
                    (output_dims + OFFICIAL_FP8_BLOCK - 1) // OFFICIAL_FP8_BLOCK,
                    (input_dims + OFFICIAL_FP8_BLOCK - 1) // OFFICIAL_FP8_BLOCK,
                ),
                dtype=mx.float32,
            )

        def __call__(self, x):
            if self.weight.dtype == mx.uint8:
                weight = _dequant_block_fp8(self.weight, self.weight_scale_inv, x.dtype)
                output = mx.matmul(x, weight.swapaxes(-1, -2))
                return output + self.bias if "bias" in self else output
            return super().__call__(x)

    class _BlockFP8SwitchLinear(SwitchLinear):
        def __init__(self, input_dims, output_dims, num_experts, bias=False):
            super().__init__(input_dims, output_dims, num_experts, bias=bias)
            self.weight_scale_inv = mx.ones(
                (
                    num_experts,
                    (output_dims + OFFICIAL_FP8_BLOCK - 1) // OFFICIAL_FP8_BLOCK,
                    (input_dims + OFFICIAL_FP8_BLOCK - 1) // OFFICIAL_FP8_BLOCK,
                ),
                dtype=mx.float32,
            )

        def __call__(self, x, indices, sorted_indices=False):
            if self.weight.dtype != mx.uint8:
                return super().__call__(x, indices, sorted_indices=sorted_indices)
            if sorted_indices:
                raise ValueError(
                    "block-FP8 switch execution does not accept sorted indices"
                )
            weight = self.weight[indices]
            scales = self.weight_scale_inv[indices]
            weight = _dequant_block_fp8(weight, scales, x.dtype)
            output = mx.matmul(x, weight.swapaxes(-1, -2))
            if "bias" in self:
                output = output + mx.expand_dims(self.bias[indices], -2)
            return output

        def to_quantized(self, group_size=64, bits=4, mode="affine", **_kwargs):
            layer = QuantizedSwitchLinear(
                self.input_dims,
                self.output_dims,
                self.num_experts,
                bias="bias" in self,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            layer.weight, layer.scales, *biases = mx.quantize(
                self.weight,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            layer.biases = biases[0] if biases else None
            if "bias" in self:
                layer.bias = self.bias
            return layer

    class _FP32Router(nn.Module):
        def __init__(self, geometry):
            super().__init__()
            self.top_k = geometry["num_experts_per_tok"]
            self.routed_scaling_factor = geometry["routed_scaling_factor"]
            self.weight = mx.zeros(
                (geometry["n_routed_experts"], geometry["hidden_size"])
            )
            self.e_score_correction_bias = mx.zeros(
                (geometry["n_routed_experts"],), dtype=mx.float32
            )

        def __call__(self, x):
            logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
            scores = mx.sigmoid(logits)
            selection = scores + self.e_score_correction_bias.astype(mx.float32)
            indices = mx.argpartition(-selection, kth=self.top_k - 1, axis=-1)[
                ..., : self.top_k
            ]
            weights = mx.take_along_axis(scores, indices, axis=-1)
            # Match the published denominator exactly, including its guard.
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
            weights = weights * self.routed_scaling_factor
            return indices, weights

    class _ClampedSwitchGLU(nn.Module):
        def __init__(self, geometry):
            super().__init__()
            hidden = geometry["hidden_size"]
            intermediate = geometry["moe_intermediate_size"]
            experts = geometry["n_routed_experts"]
            self.gate_proj = _BlockFP8SwitchLinear(
                hidden, intermediate, experts, bias=False
            )
            self.up_proj = _BlockFP8SwitchLinear(
                hidden, intermediate, experts, bias=False
            )
            self.down_proj = _BlockFP8SwitchLinear(
                intermediate, hidden, experts, bias=False
            )
            self.limit = geometry["swiglu_limit"]

        def __call__(self, x, indices):
            # Keep the simple unsorted path: it supports both affine gather-qmm
            # modules and selected-expert block-FP8 dequantisation exactly.
            x = mx.expand_dims(x, (-2, -3))
            up = self.up_proj(x, indices, sorted_indices=False)
            gate = self.gate_proj(x, indices, sorted_indices=False)
            gate = mx.minimum(gate, mx.array(self.limit, dtype=gate.dtype))
            up = mx.clip(up, -self.limit, self.limit)
            output = self.down_proj(nn.silu(gate) * up, indices, sorted_indices=False)
            return output.squeeze(-2)

    class _SharedExperts(nn.Module):
        def __init__(self, geometry):
            super().__init__()
            hidden = geometry["hidden_size"]
            intermediate = (
                geometry["moe_intermediate_size"] * geometry["n_shared_experts"]
            )
            self.gate_proj = _BlockFP8Linear(hidden, intermediate, bias=False)
            self.up_proj = _BlockFP8Linear(hidden, intermediate, bias=False)
            self.down_proj = _BlockFP8Linear(intermediate, hidden, bias=False)
            self.limit = geometry["swiglu_limit"]

        def __call__(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            gate = mx.minimum(gate, mx.array(self.limit, dtype=gate.dtype))
            up = mx.clip(up, -self.limit, self.limit)
            return self.down_proj(nn.silu(gate) * up)

    class _Glm5NextSparseMoeBlock(nn.Module):
        def __init__(self, config: Any):
            super().__init__()
            geometry = _runtime_geometry(config, validate_official=validate_official)
            self.num_experts_per_tok = geometry["num_experts_per_tok"]
            self.gate = _FP32Router(geometry)
            self.switch_mlp = _ClampedSwitchGLU(geometry)
            self.shared_experts = _SharedExperts(geometry)

        def __call__(self, x):
            if x.shape[1] <= 4 and os.environ.get("GLM5_NEXT_MOE_COMPILE", "1") == "1":
                args = _nvfp4_decode_args(self)
                if args is not None:
                    if not getattr(self, "_compiled_logged", False):
                        self._compiled_logged = True
                        import logging

                        logging.getLogger(__name__).info(
                            "GLM5-Next compiled NVFP4 MoE decode engaged"
                        )
                    return _compiled_nvfp4_sparse_decode()(x, *args)
            indices, scores = self.gate(x)
            routed = self.switch_mlp(x, indices)
            routed = mx.sum(routed * scores[..., None], axis=-2).astype(x.dtype)
            return routed + self.shared_experts(x)

    _Glm5NextSparseMoeBlock.__name__ = "Glm5NextSparseMoeBlock"
    _Glm5NextSparseMoeBlock.__qualname__ = "Glm5NextSparseMoeBlock"
    _Glm5NextSparseMoeBlock.__module__ = __name__
    return _Glm5NextSparseMoeBlock


def make_sparse_moe_class(*, validate_official: bool = True):
    """Return the lazy runtime class; relaxed geometry is for parity tests."""

    return _implementation_class(validate_official)


class Glm5NextSparseMoeBlock:
    """Lazy MLX constructor for the GLM5-Next FP32-router MoE."""

    def __new__(cls, config: Any):
        return _implementation_class(True)(config)


Glm5NextTextMoE = Glm5NextSparseMoeBlock


__all__ = [
    "Glm5NextSparseMoeBlock",
    "Glm5NextTextMoE",
    "GLM5_NEXT_BLOCK_FP8_RUNTIME_READY",
    "OFFICIAL_EXPERTS",
    "OFFICIAL_EXPERT_SIZE",
    "OFFICIAL_FP8_BLOCK",
    "OFFICIAL_HIDDEN_SIZE",
    "OFFICIAL_TOP_K",
    "sanitize_moe_weights",
    "make_sparse_moe_class",
    "validate_moe_config",
    "validate_moe_weight_layout",
]
