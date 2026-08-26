# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp gated residual (hyper-connection) primitive.

Qwen3.8 Flash Next carries four 2560-wide residual streams.  Before every
attention and MoE block it learns a rank-320 input mixture, then injects the
block result back into all four streams.  A final mixer uses the same input
mixing contract without ``block_inject_weight``.

The zero-centred grouped RMSNorm is important: checkpoint ``hc_norm.weight``
is applied as ``1 + weight``.  Substituting MLX's ordinary RMSNorm would make
fresh/official zero-valued norm weights erase the stream.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def validate_hc_config(config: Any) -> None:
    expected = {
        "hidden_size": 2560,
        "hc_count": 4,
        "hc_lowrank": 320,
        "rms_norm_eps": 1e-6,
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
        raise ValueError(f"Unsupported qwen4_exp hyper-connection config: {details}")


def validate_hc_weight_layout(
    weights: Mapping[str, Any], prefix: str, *, use_combine: bool = True
) -> None:
    """Validate official HC names and dimensions beneath *prefix*."""

    required_shapes = {
        f"{prefix}.hc_norm.weight": (10240,),
        f"{prefix}.input_mix_weight_down.weight": (320, 10240),
        f"{prefix}.input_mix_weight_up.weight": (10240, 320),
    }
    inject_key = f"{prefix}.block_inject_weight.weight"
    if use_combine:
        required_shapes[inject_key] = (4, 10240)
    elif inject_key in weights:
        raise ValueError(
            f"qwen4_exp final hyper_connection_mixer must not contain {inject_key}"
        )

    missing = [key for key in required_shapes if key not in weights]
    if missing:
        raise ValueError(
            "qwen4_exp hyper-connection is missing checkpoint weights: "
            + ", ".join(missing)
        )
    bad_shapes = []
    for key, expected in required_shapes.items():
        shape = tuple(getattr(weights[key], "shape", ()))
        if shape != expected:
            bad_shapes.append(f"{key}={shape}, expected {expected}")
    if bad_shapes:
        raise ValueError(
            "Invalid qwen4_exp hyper-connection checkpoint layout: "
            + "; ".join(bad_shapes)
        )


@lru_cache(maxsize=1)
def _implementation_class():
    import mlx.core as mx
    import mlx.nn as nn

    class ZeroCenteredGroupRMSNorm(nn.Module):
        def __init__(self, dim: int, group_size: int, eps: float):
            super().__init__()
            if dim % group_size:
                raise ValueError(
                    f"hidden_size ({dim}) must be divisible by group_size "
                    f"({group_size})"
                )
            self.weight = mx.zeros((dim,))
            self.group_size = group_size
            self.eps = eps

        def __call__(self, x):
            input_dtype = x.dtype
            grouped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, self.group_size)
            variance = mx.mean(mx.square(grouped), axis=-1, keepdims=True)
            normalized = grouped * mx.rsqrt(variance + self.eps)
            normalized = normalized.reshape(*x.shape)
            normalized = normalized * (1.0 + self.weight.astype(mx.float32))
            return normalized.astype(input_dtype)

    class _Qwen4ExpGatedResidual(nn.Module):
        def __init__(self, config: Any, use_combine: bool = True):
            super().__init__()
            validate_hc_config(config)
            self.hc_count = int(_get(config, "hc_count"))
            self.hidden_size = int(_get(config, "hidden_size"))
            self.hc_lowrank = int(_get(config, "hc_lowrank"))
            hc_hidden_size = self.hc_count * self.hidden_size
            self.hc_norm = ZeroCenteredGroupRMSNorm(
                hc_hidden_size,
                group_size=self.hidden_size,
                eps=float(_get(config, "rms_norm_eps")),
            )
            self.input_mix_weight_down = nn.Linear(
                hc_hidden_size, self.hc_lowrank, bias=False
            )
            self.input_mix_weight_up = nn.Linear(
                self.hc_lowrank, hc_hidden_size, bias=False
            )
            self.block_inject_weight = (
                nn.Linear(hc_hidden_size, self.hc_count, bias=False)
                if use_combine
                else None
            )

        def __call__(self, hyper_input):
            expected = self.hc_count * self.hidden_size
            if hyper_input.shape[-1] != expected:
                raise ValueError(
                    f"Expected {expected} hyper-connection features, got "
                    f"{hyper_input.shape[-1]}"
                )
            hyper_input_normed = self.hc_norm(hyper_input)
            input_mix_weight = nn.silu(
                self.input_mix_weight_down(hyper_input_normed) / self.hc_count
            )
            input_mix_weight = mx.sigmoid(self.input_mix_weight_up(input_mix_weight))
            input_mix_weight = input_mix_weight.reshape(
                *input_mix_weight.shape[:-1], self.hc_count, self.hidden_size
            )
            streams = hyper_input_normed.reshape(
                *hyper_input_normed.shape[:-1], self.hc_count, self.hidden_size
            )
            mixed_input = mx.mean(input_mix_weight * streams, axis=-2)
            if self.block_inject_weight is None:
                return mixed_input
            injection_weights = 2 * mx.sigmoid(
                self.block_inject_weight(hyper_input_normed) / self.hc_count
            )
            return mixed_input, hyper_input, injection_weights

    _Qwen4ExpGatedResidual.__name__ = "Qwen4ExpGatedResidual"
    _Qwen4ExpGatedResidual.__qualname__ = "Qwen4ExpGatedResidual"
    _Qwen4ExpGatedResidual.__module__ = __name__
    return _Qwen4ExpGatedResidual


class Qwen4ExpGatedResidual:
    """Lazy MLX constructor with the official HC call contract."""

    def __new__(cls, config: Any, use_combine: bool = True):
        return _implementation_class()(config, use_combine=use_combine)


class Qwen4ExpHyperConnectionMixer:
    """Construct the final four-stream-to-hidden mixer.

    The returned module has no ``block_inject_weight`` and returns only the
    mixed ``[..., 2560]`` tensor.
    """

    def __new__(cls, config: Any):
        return _implementation_class()(config, use_combine=False)


def expand_hyper_residual(
    block_output: Any,
    hyper_input: Any,
    injection_weights: Any,
    *,
    hc_count: int = 4,
) -> Any:
    """Apply the official post-block injection and flatten the four streams."""

    if hc_count != 4:
        raise ValueError(f"qwen4_exp requires hc_count=4, got {hc_count}")
    if injection_weights.shape[-1] != hc_count:
        raise ValueError(
            f"Expected {hc_count} injection weights, got {injection_weights.shape[-1]}"
        )
    hidden_size = block_output.shape[-1]
    if hyper_input.shape[-1] != hc_count * hidden_size:
        raise ValueError(
            "hyper_input and block_output do not satisfy the qwen4_exp "
            "four-stream contract"
        )
    injection = block_output[..., None, :] * injection_weights[..., :, None]
    return hyper_input + injection.reshape(*hyper_input.shape)


Qwen4ExpTextGatedResidual = Qwen4ExpGatedResidual


__all__ = [
    "Qwen4ExpGatedResidual",
    "Qwen4ExpHyperConnectionMixer",
    "Qwen4ExpTextGatedResidual",
    "expand_hyper_residual",
    "validate_hc_config",
    "validate_hc_weight_layout",
]
