# SPDX-License-Identifier: Apache-2.0
"""The exp inside MLX's Sigmoid, for kernels that mirror a compiled sigmoid.

MLX's Sigmoid functor is ``y = 1 / (1 + exp(|x|)); x < 0 ? y : 1 - y``.
Through mlx 0.32.2 it called ``metal::exp``, which runtime-compiled kernels
(compiled graphs such as ``nn.silu``, custom kernels, JIT builds) evaluate
with the fast exp. ml-explore/mlx#4461 (mlx 0.32.3) made it call
``metal::precise::exp`` so that compiled and eager sigmoids agree.

Fused kernels that reproduce a compiled sigmoid bit for bit write it as
``metal::exp(metal::abs(...))`` and pass their source through
``compiled_sigmoid``, which switches that call to the precise exp on
mlx 0.32.3 and later. Kernels that mirror the eager sigmoid already use
``metal::precise::exp`` (the release metallib is built with -fno-fast-math)
and are left alone. ``OMLX_MLX_PRECISE_SIGMOID=1`` or ``0`` overrides the
version check for MLX builds from between releases.
"""

from __future__ import annotations

import os

import mlx.core as mx

_FAST_CALL = "metal::exp(metal::abs("
_PRECISE_CALL = "metal::precise::exp(metal::abs("


def _mlx_release(version: str) -> tuple[int, ...]:
    """(major, minor, patch) of an MLX version such as 0.32.3.dev20260929+abc."""
    base = version.split("+")[0].split(".dev")[0]
    parts = []
    for part in base.split(".")[:3]:
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits or 0))
    return tuple(parts)


def precise_compiled_sigmoid(version: str | None = None) -> bool:
    """Whether MLX's compiled Sigmoid evaluates ``metal::precise::exp``."""
    override = os.environ.get("OMLX_MLX_PRECISE_SIGMOID")
    if override in ("0", "1"):
        return override == "1"
    return _mlx_release(mx.__version__ if version is None else version) >= (0, 32, 3)


def compiled_sigmoid(source: str) -> str:
    """``source`` with its compiled-sigmoid exp as this MLX evaluates it."""
    if not precise_compiled_sigmoid():
        return source
    return source.replace(_FAST_CALL, _PRECISE_CALL)
