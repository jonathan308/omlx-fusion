# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 Flash Next (``qwen4_exp``) gated-delta primitive.

The published model deliberately reuses Qwen3.5's split-projection
GatedDeltaNet.  It does *not* use Qwen3-Next's fused ``qkvz`` / ``ba``
layout.  The only trained math difference in this block is the sigmoid
output gate.  We therefore reuse the pinned ``mlx_lm`` recurrence and its
cache implementation, while keeping the Qwen4-Exp validation and output
normalisation here.

MLX imports are delayed until construction.  Configuration and checkpoint
layout checks stay pure Python so discovery/conversion can fail closed
without allocating model weights (or even importing Metal).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

_EXPECTED = {
    "hidden_size": 2560,
    "linear_num_value_heads": 48,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "hidden_act": "silu",
    "output_gate_type": "sigmoid",
    "mamba_ssm_dtype": "float32",
}


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def validate_gdn_config(config: Any, layer_idx: int | None = None) -> None:
    """Validate the exact Flash-Next GDN contract.

    These values are architecture, not tuning knobs.  Accepting a nearby
    Qwen3.5/Qwen3-Next layout would let weights load under the wrong graph and
    produce plausible-looking but invalid output.
    """

    mismatches = {
        name: (expected, _get(config, name))
        for name, expected in _EXPECTED.items()
        if _get(config, name) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})"
            for name, (expected, actual) in mismatches.items()
        )
        raise ValueError(f"Unsupported qwen4_exp GatedDeltaNet config: {details}")

    num_v_heads = int(_get(config, "linear_num_value_heads"))
    num_k_heads = int(_get(config, "linear_num_key_heads"))
    if num_v_heads % num_k_heads:
        raise ValueError(
            "qwen4_exp linear_num_value_heads must be divisible by linear_num_key_heads"
        )

    if layer_idx is not None:
        layer_types = _get(config, "layer_types")
        if layer_types is not None:
            if not 0 <= layer_idx < len(layer_types):
                raise ValueError(f"qwen4_exp layer_idx {layer_idx} is out of range")
            if layer_types[layer_idx] != "linear_attention":
                raise ValueError(
                    f"qwen4_exp GatedDeltaNet cannot serve layer type "
                    f"{layer_types[layer_idx]!r} at layer {layer_idx}"
                )


def validate_gdn_weight_layout(weights: Mapping[str, Any], prefix: str) -> None:
    """Fail closed unless *prefix* contains the official split GDN weights.

    ``prefix`` is the path through ``linear_attn`` without a trailing dot.
    Both Hugging Face's Conv1d shape ``(C, 1, 4)`` and its already-sanitized
    MLX shape ``(C, 4, 1)`` are accepted.  Fused Qwen3-Next projections are
    explicitly rejected.
    """

    forbidden = (f"{prefix}.in_proj_qkvz.weight", f"{prefix}.in_proj_ba.weight")
    present_forbidden = [key for key in forbidden if key in weights]
    if present_forbidden:
        raise ValueError(
            "qwen4_exp requires split in_proj_qkv/z/b/a weights; found "
            + ", ".join(present_forbidden)
        )

    required_shapes = {
        f"{prefix}.in_proj_qkv.weight": (10240, 2560),
        f"{prefix}.in_proj_z.weight": (6144, 2560),
        f"{prefix}.in_proj_b.weight": (48, 2560),
        f"{prefix}.in_proj_a.weight": (48, 2560),
        f"{prefix}.dt_bias": (48,),
        f"{prefix}.A_log": (48,),
        f"{prefix}.norm.weight": (128,),
        f"{prefix}.out_proj.weight": (2560, 6144),
    }
    missing = [key for key in required_shapes if key not in weights]
    conv_key = f"{prefix}.conv1d.weight"
    if conv_key not in weights:
        missing.append(conv_key)
    if missing:
        raise ValueError(
            "qwen4_exp GatedDeltaNet is missing checkpoint weights: "
            + ", ".join(missing)
        )

    bad_shapes: list[str] = []
    for key, expected in required_shapes.items():
        shape = tuple(getattr(weights[key], "shape", ()))
        if shape != expected:
            bad_shapes.append(f"{key}={shape}, expected {expected}")
    conv_shape = tuple(getattr(weights[conv_key], "shape", ()))
    if conv_shape not in {(10240, 1, 4), (10240, 4, 1)}:
        bad_shapes.append(
            f"{conv_key}={conv_shape}, expected (10240, 1, 4) or (10240, 4, 1)"
        )
    if bad_shapes:
        raise ValueError(
            "Invalid qwen4_exp GatedDeltaNet checkpoint layout: "
            + "; ".join(bad_shapes)
        )


@lru_cache(maxsize=1)
def _implementation_class():
    """Build the real MLX module only when a model is instantiated."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import GatedDeltaNet as PinnedGatedDeltaNet

    class SigmoidRMSNormGated(nn.Module):
        """Official norm-before-gate calculation with a sigmoid gate."""

        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            self.weight = mx.ones((hidden_size,))
            self.eps = eps
            self.activation = "sigmoid"

        def __call__(self, hidden_states, gate=None):
            input_dtype = hidden_states.dtype
            hidden_f32 = hidden_states.astype(mx.float32)
            variance = mx.mean(mx.square(hidden_f32), axis=-1, keepdims=True)
            normalized = hidden_f32 * mx.rsqrt(variance + self.eps)
            normalized = normalized * self.weight.astype(mx.float32)
            if gate is not None:
                normalized = normalized * mx.sigmoid(gate.astype(mx.float32))
            return normalized.astype(input_dtype)

    class _Qwen4ExpGatedDeltaNet(PinnedGatedDeltaNet):
        """Qwen3.5 recurrence plus Qwen4-Exp's trained sigmoid gate."""

        def __init__(self, config: Any, layer_idx: int):
            validate_gdn_config(config, layer_idx)
            super().__init__(config)
            self.layer_idx = layer_idx
            self.layer_type = "linear_attention"
            self.mamba_ssm_dtype = "float32"
            self.norm = SigmoidRMSNormGated(
                self.head_v_dim, eps=self.layer_norm_epsilon
            )

        def __call__(self, inputs, mask=None, cache=None):
            # ``mlx_lm.models.gated_delta.gated_delta_update`` creates new
            # state in fp32.  Coerce restored/legacy cache state too so the
            # recurrent product cannot silently fall back to bf16.
            if cache is not None and cache[1] is not None:
                cache[1] = cache[1].astype(mx.float32)
            output = super().__call__(inputs, mask=mask, cache=cache)
            if cache is not None and cache[1] is not None:
                cache[1] = cache[1].astype(mx.float32)
            return output

    _Qwen4ExpGatedDeltaNet.__name__ = "Qwen4ExpGatedDeltaNet"
    _Qwen4ExpGatedDeltaNet.__qualname__ = "Qwen4ExpGatedDeltaNet"
    _Qwen4ExpGatedDeltaNet.__module__ = __name__
    return _Qwen4ExpGatedDeltaNet


class Qwen4ExpGatedDeltaNet:
    """Lazy constructor for the MLX Qwen4-Exp GatedDeltaNet module.

    ``Qwen4ExpGatedDeltaNet(config, layer_idx)`` returns an ``mlx.nn.Module``
    with the standard ``(inputs, mask=None, cache=None)`` call signature.
    """

    def __new__(cls, config: Any, layer_idx: int):
        return _implementation_class()(config, layer_idx)


# Official Transformers-style spelling used by model wrappers.
Qwen4ExpTextGatedDeltaNet = Qwen4ExpGatedDeltaNet


__all__ = [
    "Qwen4ExpGatedDeltaNet",
    "Qwen4ExpTextGatedDeltaNet",
    "validate_gdn_config",
    "validate_gdn_weight_layout",
]
