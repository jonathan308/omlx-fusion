# SPDX-License-Identifier: Apache-2.0
"""Fused 4x4 Sinkhorn kernel for GLM-5.3 mHC.

The 20-iteration Sinkhorn loop is ~38 tiny reduction kernels per mHC call;
at 90 calls per token that dominated decode.  This kernel runs the exact
eager sequence (precise row softmax, +eps, column normalize, then 19
row/column normalization iterations) serially in fp32 on one thread per
position — 16 floats of work — replacing the reduction chain with a single
launch.  Operation order matches the eager path step for step.

No MLX import or Metal compilation at module import time.
"""

from __future__ import annotations

import os
from typing import Any

# One thread per position; the whole 4x4 matrix lives in registers.
_SOURCE = r"""
    const int pos = thread_position_in_grid.x;
    device const float* in = comb_logits + (long)pos * 16;

    float c[4][4];
    float mxv[4];
    for (int i = 0; i < 4; ++i) {
        mxv[i] = -INFINITY;
        for (int j = 0; j < 4; ++j) {
            float v = in[i * 4 + j];
            c[i][j] = v;
            mxv[i] = max(mxv[i], v);
        }
    }
    // precise softmax over the last axis, then + eps
    for (int i = 0; i < 4; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 4; ++j) sum += exp(c[i][j] - mxv[i]);
        for (int j = 0; j < 4; ++j) c[i][j] = exp(c[i][j] - mxv[i]) / sum;
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) c[i][j] += Eps;
    for (int j = 0; j < 4; ++j) {
        float s0 = 0.0f;
        for (int i = 0; i < 4; ++i) s0 += c[i][j];
        for (int i = 0; i < 4; ++i) c[i][j] /= (s0 + Eps);
    }
    for (int it = 0; it < 19; ++it) {
        for (int i = 0; i < 4; ++i) {
            float s0 = 0.0f;
            for (int j = 0; j < 4; ++j) s0 += c[i][j];
            for (int j = 0; j < 4; ++j) c[i][j] /= (s0 + Eps);
        }
        for (int j = 0; j < 4; ++j) {
            float s0 = 0.0f;
            for (int i = 0; i < 4; ++i) s0 += c[i][j];
            for (int i = 0; i < 4; ++i) c[i][j] /= (s0 + Eps);
        }
    }
    device float* outp = comb + (long)pos * 16;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) outp[i * 4 + j] = c[i][j];
"""

_kernel_cache: dict[float, Any] = {}


def _get_kernel(eps: float):
    import mlx.core as mx

    kernel = _kernel_cache.get(eps)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="omlx_glm5_next_sinkhorn4x4",
            input_names=["comb_logits"],
            output_names=["comb"],
            source=_SOURCE.replace("Eps", repr(float(eps))),
        )
        _kernel_cache[eps] = kernel
    return kernel


def sinkhorn4x4(comb_logits: Any, eps: float):
    """Exact eager-equivalent 20-iteration Sinkhorn on (..., 4, 4) fp32."""

    import mlx.core as mx

    shape = comb_logits.shape
    flat = mx.contiguous(comb_logits.reshape(-1, 4, 4).astype(mx.float32))
    positions = flat.shape[0]
    (out,) = _get_kernel(eps)(
        inputs=[flat],
        grid=(positions, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[mx.float32],
    )
    return out.reshape(shape)


def sinkhorn4x4_available() -> bool:
    if os.environ.get("GLM5_NEXT_SINKHORN_FUSED", "1") != "1":
        return False
    try:
        import mlx.core as mx

        return mx.default_device() == mx.gpu and mx.metal.is_available()
    except Exception:
        return False


__all__ = ["sinkhorn4x4", "sinkhorn4x4_available"]
