# SPDX-License-Identifier: Apache-2.0
"""Fused NVFP4 gather-GEMV decode kernel for GLM-5.3-Flash MoE layers.

The stock ``mx.gather_qmm(mode="nvfp4")`` path reaches only ~105 GB/s at
single-token decode shapes on M3 Ultra; this fused Metal kernel streams the
eight selected experts' packed E2M1 rows, per-16 E4M3 scales, and ModelOpt
FP32 global scales in one pass at memory-bandwidth rates.  The math mirrors
``ScaledNVFP4SwitchLinear`` exactly: per-16 group dequantization, FP32
accumulation, then the retained per-expert global scale.

No MLX import or Metal compilation happens at module import time.
"""

from __future__ import annotations

import os
from typing import Any

NVFP4_GEMV_READY: bool = True

_HEADER = r"""
constant float OMLX_E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

inline float omlx_e2m1_decode(uint code) {
    float v = OMLX_E2M1_TABLE[code & 7u];
    return (code & 8u) ? -v : v;
}

inline float omlx_e4m3fn_decode(uchar b) {
    uint e = (b >> 3) & 0xFu;
    uint m = b & 0x7u;
    float v;
    if (e == 0u) {
        v = (float)m * 0x1p-9f;
    } else if (e == 15u && m == 7u) {
        v = NAN;
    } else {
        v = ldexp(1.0f + (float)m * 0.125f, (int)e - 7);
    }
    return (b & 0x80u) ? -v : v;
}
"""

# One thread per output element.  Weight rows stream as uint4 (four packed
# words = two scale groups per load) and scales as uchar2, so a thread's
# inner loop is 128 vector iterations over its 2 KiB row.
_SOURCE = r"""
    constexpr int TG = 256;
    const uint tid = thread_position_in_threadgroup.x;
    const int o = threadgroup_position_in_grid.x * TG + tid;
    const int k = threadgroup_position_in_grid.y;
    if (o >= OUT) return;
    const int e = idx[k];

    const int GROUPS = IN / 16;
    const int WORDS = IN / 8;

    threadgroup InT xs[4096];
    for (int i = tid; i < (int)IN; i += TG) xs[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device uint* wrow = w + ((long)e * OUT + o) * WORDS;
    const device uchar* srow = s + ((long)e * OUT + o) * GROUPS;
    const device uint4* wrow4 = (const device uint4*)wrow;
    const device uchar2* srow2 = (const device uchar2*)srow;

    float acc = 0.0f;
    for (int g2 = 0; g2 < GROUPS / 2; ++g2) {
        const uint4 wv = wrow4[g2];
        const uchar2 sv = srow2[g2];
        const uint wa[2] = {wv.x, wv.z};
        const uint wb[2] = {wv.y, wv.w};
        const uchar sc[2] = {sv.x, sv.y};
        for (int gg = 0; gg < 2; ++gg) {
            const int g = 2 * g2 + gg;
            float partial = 0.0f;
            for (int j = 0; j < 8; ++j) {
                partial += omlx_e2m1_decode((wa[gg] >> (4 * j)) & 0xFu)
                    * (float)xs[16 * g + j];
                partial += omlx_e2m1_decode((wb[gg] >> (4 * j)) & 0xFu)
                    * (float)xs[16 * g + 8 + j];
            }
            acc += partial * omlx_e4m3fn_decode(sc[gg]);
        }
    }
    out[(long)k * OUT + o] = (InT)(acc * gs[e]);
"""

# Cooperative variant: one threadgroup (256 threads) per (k, o) output;
# thread i owns scale-group i, so weight/scale reads are fully coalesced.
_SOURCE_COOP = r"""
    constexpr int TG = 256;
    const uint tid = thread_position_in_threadgroup.x;
    const int o = threadgroup_position_in_grid.x;
    const int k = threadgroup_position_in_grid.y;
    const int e = idx[k];

    const int GROUPS = IN / 16;
    const int WORDS = IN / 8;

    const device uint* wrow = w + ((long)e * OUT + o) * WORDS;
    const device uchar* srow = s + ((long)e * OUT + o) * GROUPS;

    float partial = 0.0f;
    if ((int)tid < GROUPS) {
        const int g = tid;
        const uint w0 = wrow[2 * g];
        const uint w1 = wrow[2 * g + 1];
        float dot = 0.0f;
        for (int j = 0; j < 8; ++j) {
            dot += omlx_e2m1_decode((w0 >> (4 * j)) & 0xFu) * (float)x[16 * g + j];
            dot += omlx_e2m1_decode((w1 >> (4 * j)) & 0xFu) * (float)x[16 * g + 8 + j];
        }
        partial = dot * omlx_e4m3fn_decode(srow[g]);
    }
    float acc = simd_sum(partial);
    threadgroup float parts[TG / 32];
    if (thread_index_in_simdgroup == 0) parts[simdgroup_index_in_threadgroup] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < TG / 32; ++i) total += parts[i];
        out[(long)k * OUT + o] = (InT)(total * gs[e]);
    }
"""

_kernel_cache: dict[tuple[Any, ...], Any] = {}


def _get_kernel(dtype: Any, coop: bool = False):
    import mlx.core as mx

    key = (dtype, coop)
    kernel = _kernel_cache.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=(
                "omlx_glm5_next_nvfp4_gather_gemv_coop"
                if coop
                else "omlx_glm5_next_nvfp4_gather_gemv"
            ),
            input_names=["x", "w", "s", "gs", "idx", "IN", "OUT"],
            output_names=["out"],
            source=_SOURCE_COOP if coop else _SOURCE,
            header=_HEADER,
        )
        _kernel_cache[key] = kernel
    return kernel


def nvfp4_gather_gemv(
    x: Any,
    weight: Any,
    scales: Any,
    global_scale: Any,
    indices: Any,
    out_dtype: Any = None,
) -> Any:
    """Gather-selected NVFP4 GEMV for decode rows.

    x: (IN,) activation row.  weight: (E, OUT, IN//8) uint32 E2M1 carriers,
    scales: (E, OUT, IN//16) uint8 E4M3 group scales, global_scale: (E,)
    FP32 ModelOpt scale, indices: (K,) int32 expert ids.  Returns
    (K, OUT) in ``x.dtype`` (or ``out_dtype``).
    """

    import mlx.core as mx

    in_features = x.shape[-1]
    out_features = weight.shape[1]
    k = indices.shape[-1]
    x = mx.contiguous(x.reshape(-1).astype(x.dtype))
    idx = mx.contiguous(indices.reshape(-1).astype(mx.int32))
    coop = os.environ.get("GLM5_NEXT_NVFP4_GEMV_COOP", "1") == "1"
    if coop:
        grid = (256 * out_features, k, 1)
    else:
        grid = (256 * (out_features // 256), k, 1)
    (out,) = _get_kernel(x.dtype, coop)(
        inputs=[x, weight, scales, global_scale, idx, in_features, out_features],
        template=[("InT", x.dtype)],
        grid=grid,
        threadgroup=(256, 1, 1),
        output_shapes=[(k, out_features)],
        output_dtypes=[x.dtype if out_dtype is None else out_dtype],
    )
    return out


def nvfp4_gather_gemv_available() -> bool:
    """Affirmative runtime probe used by the switch-layer fast path."""

    if os.environ.get("GLM5_NEXT_NVFP4_GEMV", "1") != "1":
        return False
    try:
        import mlx.core as mx

        return mx.default_device() == mx.gpu and mx.metal.is_available()
    except Exception:
        return False


__all__ = [
    "NVFP4_GEMV_READY",
    "nvfp4_gather_gemv",
    "nvfp4_gather_gemv_available",
]
