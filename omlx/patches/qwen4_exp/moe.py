# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp packed MoE primitive and checkpoint sanitizer.

The official graph is Qwen3-Next's top-k MoE: 512 packed routed experts,
top-10 routing, plus one sigmoid-gated shared SwiGLU expert.  The checkpoint
stores routed ``gate`` and ``up`` matrices fused as
``experts.gate_up_proj``.  Pinned mlx-lm's switch layer stores them as two
packed switch linears, so conversion is a lossless split on the intermediate
axis.

Sanitization is deliberately prefix-agnostic.  It handles both backbone and
``mtp.layers.0`` weights and never filters ``mtp.*`` keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def validate_moe_config(config: Any) -> None:
    expected = {
        "hidden_size": 2560,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "hidden_act": "silu",
    }
    mismatches = {
        name: (wanted, _get(config, name))
        for name, wanted in expected.items()
        if _get(config, name) != wanted
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {wanted!r})"
            for name, (wanted, actual) in mismatches.items()
        )
        raise ValueError(f"Unsupported qwen4_exp MoE config: {details}")
    norm_topk_prob = _get(config, "norm_topk_prob")
    if norm_topk_prob not in (None, False, True):
        raise ValueError(
            f"qwen4_exp norm_topk_prob must be bool or null, got {norm_topk_prob!r}"
        )


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def validate_moe_weight_layout(weights: Mapping[str, Any], prefix: str) -> None:
    """Validate official packed routed and shared-expert tensors.

    ``prefix`` is the path through ``mlp`` without a trailing dot.
    """

    expected = {
        f"{prefix}.gate.weight": (512, 2560),
        f"{prefix}.experts.gate_up_proj": (512, 1280, 2560),
        f"{prefix}.experts.down_proj": (512, 2560, 640),
        f"{prefix}.shared_expert.gate_proj.weight": (640, 2560),
        f"{prefix}.shared_expert.up_proj.weight": (640, 2560),
        f"{prefix}.shared_expert.down_proj.weight": (2560, 640),
        f"{prefix}.shared_expert_gate.weight": (1, 2560),
    }
    missing = [key for key in expected if key not in weights]
    if missing:
        raise ValueError(
            "qwen4_exp MoE is missing checkpoint weights: " + ", ".join(missing)
        )
    bad_shapes = [
        f"{key}={_shape(weights[key])}, expected {shape}"
        for key, shape in expected.items()
        if _shape(weights[key]) != shape
    ]
    if bad_shapes:
        raise ValueError(
            "Invalid qwen4_exp packed MoE checkpoint layout: " + "; ".join(bad_shapes)
        )


def sanitize_moe_weights(weights: Mapping[str, Any]) -> dict[str, Any]:
    """Losslessly split all official packed routed-expert tensors.

    The returned mapping retains every unrelated key, including all MTP
    tensors.  A missing partner or a non-official shape fails closed rather
    than silently constructing a different MoE.
    """

    sanitized = dict(weights)
    suffix = ".experts.gate_up_proj"
    gate_up_keys = sorted(key for key in sanitized if key.endswith(suffix))
    for gate_up_key in gate_up_keys:
        mlp_prefix = gate_up_key[: -len(suffix)]
        down_key = f"{mlp_prefix}.experts.down_proj"
        if down_key not in sanitized:
            raise ValueError(
                f"qwen4_exp packed MoE {gate_up_key} has no matching {down_key}"
            )

        gate_up = sanitized[gate_up_key]
        down = sanitized[down_key]
        expected_gate_up = (512, 1280, 2560)
        expected_down = (512, 2560, 640)
        if _shape(gate_up) != expected_gate_up or _shape(down) != expected_down:
            raise ValueError(
                "Invalid qwen4_exp packed MoE tensors at "
                f"{mlp_prefix}: gate_up={_shape(gate_up)} expected "
                f"{expected_gate_up}; down={_shape(down)} expected "
                f"{expected_down}"
            )

        sanitized.pop(gate_up_key)
        sanitized.pop(down_key)
        sanitized[f"{mlp_prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :640, :]
        sanitized[f"{mlp_prefix}.switch_mlp.up_proj.weight"] = gate_up[..., 640:, :]
        sanitized[f"{mlp_prefix}.switch_mlp.down_proj.weight"] = down

    return sanitized


@lru_cache(maxsize=1)
def _implementation_class():
    # Qwen's official implementation subclasses Qwen3-Next's sparse MoE.
    # The pinned mlx-lm class has the same router, precise softmax, SwitchGLU,
    # shared expert, and distributed sharding behavior.
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    class _Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):
        def __init__(self, config: Any):
            validate_moe_config(config)
            super().__init__(config)

    _Qwen4ExpSparseMoeBlock.__name__ = "Qwen4ExpSparseMoeBlock"
    _Qwen4ExpSparseMoeBlock.__qualname__ = "Qwen4ExpSparseMoeBlock"
    _Qwen4ExpSparseMoeBlock.__module__ = __name__
    return _Qwen4ExpSparseMoeBlock


class Qwen4ExpSparseMoeBlock:
    """Lazy constructor for the pinned Qwen3-Next-compatible MLX MoE."""

    def __new__(cls, config: Any):
        return _implementation_class()(config)


Qwen4ExpTextSparseMoeBlock = Qwen4ExpSparseMoeBlock


__all__ = [
    "Qwen4ExpSparseMoeBlock",
    "Qwen4ExpTextSparseMoeBlock",
    "sanitize_moe_weights",
    "validate_moe_config",
    "validate_moe_weight_layout",
]
