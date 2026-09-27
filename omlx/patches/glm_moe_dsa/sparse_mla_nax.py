"""Tensor-unit (NAX) sparse MLA prefill attention for GLM-5.3 on M5 GPUs.

GLM-5.3's DSA layers attend every query over its own top-k latent rows
(2048 + 3 tail slots, 64 heads, 512-wide NoPE latent; values are the same
latent rows). The native ``glm_dsa_sparse_mla_attention`` kernel computes
this in fp32 on the classic SIMD matrix units at ~10-13 TFLOPS: ~42-54 ms
per layer per 2048-token chunk (4k-64k context, M5 Ultra).

This kernel keeps the native kernel's arithmetic -- fp32 scores scaled by
``scale * log2(e)``, unused slots excluded, online ``exp2`` softmax in
fp32, fp32 probabilities times bf16 values accumulated in fp32, one
division at the end -- but runs both products on the tensor units with
fp32 accumulation, so only the summation order differs. The fp32
probabilities enter the PV product as three bf16 parts (hi + mid + lo ==
p exactly, 8 + 8 + 8 significant bits), each through an exact bf16 x bf16
-> fp32 tensor op: a single fp32 operand would need a non-relaxed tensor
op (half rate, ~10% slower kernel), and with ``relaxed_precision`` the
tensor unit rounds fp32 operands to ~11 significant bits (measured), which
the native kernel does not do.

Layout: one threadgroup per (query, 32-head half), 8 simdgroups. Per tile
of 128 top-k slots, simdgroup ``(hg, kq)`` computes the scores of heads
``hg * 16 .. + 16`` for slots ``kq * 32 .. + 32`` over the whole latent
(key rows are read through the top-k indices, never gathered into memory)
into threadgroup memory; then simdgroup ``(hg, dq)`` runs the online
softmax of its head group over the tile and multiplies the fp32
probabilities with its 128-wide value slice. Tiles without any usable slot
(the indexer sorts unused slots last) are skipped.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import mlx.core as mx

_ENV = os.environ.get("OMLX_GLM_SPARSE_MLA_NAX", "1").strip().lower()
_ENABLED = _ENV not in {"0", "false", "off"}

_D_LATENT = 512
_HEADS_PER_GROUP = 32

_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

# Fragment layout (MLX BaseNAXFrag, 16x16): lane -> rows fm, fm + 8 and
# columns fn .. fn + 3. Lanes sharing a row differ in lane bits 0 and 3.
_SOURCE = """
    constexpr int D = 512;
    constexpr int BK = 128;
    const int L = params[0];
    const int Kn = params[1];
    const int TOPK = params[2];
    const int q_off = params[3];
    const float scale_log2 = scale[0] * 1.44269504088896341f;
    const int qi = int(threadgroup_position_in_grid.x);
    const int hh = int(threadgroup_position_in_grid.y);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const int hg = int(sg) / 4;
    const int dq = int(sg) % 4;   // QK: key quarter; PV: dim quarter
    const int q_abs = q_off + qi;
    const int head0 = hh * 32 + hg * 16;

    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);

    threadgroup float s_tile[2 * 16 * BK];
    threadgroup int sel[2][BK];
    threadgroup int live[2][BK / 32];

    constexpr auto qk_desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    // PV: the fp32 probabilities enter as three bf16 parts (hi + mid + lo
    // == p exactly: 8 + 8 + 8 significant bits), each multiplied with the
    // bf16 values by an exact bf16 x bf16 -> fp32 tensor op.
    constexpr auto pv_desc = matmul2d_descriptor(
        16, 32, 16, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    auto pa = pv_op.template get_left_input_cooperative_tensor<bfloat, T, float>();
    auto pm = pv_op.template get_left_input_cooperative_tensor<bfloat, T, float>();
    auto pl = pv_op.template get_left_input_cooperative_tensor<bfloat, T, float>();
    auto pb = pv_op.template get_right_input_cooperative_tensor<bfloat, T, float>();
    auto o0 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o1 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o2 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o3 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    for (short e = 0; e < 16; ++e) {
        o0[e] = 0.0f;
        o1[e] = 0.0f;
        o2[e] = 0.0f;
        o3[e] = 0.0f;
    }
    float m_run[2] = {-FLT_MAX, -FLT_MAX};
    float l_run[2] = {0.0f, 0.0f};

    const device T* qr0 = q + (ulong(head0 + fm) * L + qi) * D + fn;
    const device T* qr1 = q + (ulong(head0 + fm + 8) * L + qi) * D + fn;
    const device int32_t* idx_row = idx + ulong(qi) * TOPK;

    const int n_tiles = (TOPK + BK - 1) / BK;
    for (int t = 0; t < n_tiles; ++t) {
        const int buf = t & 1;
        const int tile_keys = min(BK, TOPK - t * BK);
        if (tid < uint(BK)) {
            const int slot = t * BK + int(tid);
            int kp = slot < TOPK ? int(idx_row[slot]) : -1;
            if (kp < 0 || kp >= Kn || kp > q_abs) {
                kp = -1;
            }
            sel[buf][tid] = kp;
            const bool any_live = simd_any(kp >= 0);
            if (lane == 0) {
                live[buf][sg] = any_live ? 1 : 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Unused slots sort last in the indexer's top-k rows: skip tiles
        // with no live key (uniform across the threadgroup).
        if ((live[buf][0] | live[buf][1] | live[buf][2] | live[buf][3]) == 0) {
            continue;
        }

        if (dq * 32 < tile_keys) {
            auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
            auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
            auto sc = qk_op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
            for (short e = 0; e < 16; ++e) {
                sc[e] = 0.0f;
            }
            const device T* kr[2][2];
            for (short tn = 0; tn < 2; ++tn) {
                for (short i = 0; i < 2; ++i) {
                    const int kp = sel[buf][dq * 32 + tn * 16 + fm + i * 8];
                    kr[tn][i] = kv + ulong(max(kp, 0)) * D + fn;
                }
            }
            for (short kk = 0; kk < D; kk += 16) {
                for (short j = 0; j < 4; ++j) {
                    qa[j] = qr0[kk + j];
                    qa[4 + j] = qr1[kk + j];
                }
                for (short tn = 0; tn < 2; ++tn) {
                    for (short i = 0; i < 2; ++i) {
                        for (short j = 0; j < 4; ++j) {
                            kb[tn * 8 + i * 4 + j] = kr[tn][i][kk + j];
                        }
                    }
                }
                qk_op.run(qa, kb, sc);
            }
            threadgroup float* sp = s_tile + hg * 16 * BK + dq * 32;
            for (short tn = 0; tn < 2; ++tn) {
                for (short i = 0; i < 2; ++i) {
                    for (short j = 0; j < 4; ++j) {
                        sp[(fm + i * 8) * BK + tn * 16 + fn + j] = sc[tn * 8 + i * 4 + j];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float* st = s_tile + hg * 16 * BK;
        const int n_ks = (tile_keys + 15) / 16;
        float rmax[2] = {m_run[0], m_run[1]};
        for (short ks = 0; ks < n_ks; ++ks) {
            const int key0 = ks * 16 + fn;
            for (short i = 0; i < 2; ++i) {
                const int r = fm + i * 8;
                for (short j = 0; j < 4; ++j) {
                    if (sel[buf][key0 + j] >= 0) {
                        rmax[i] = max(rmax[i], st[r * BK + key0 + j] * scale_log2);
                    }
                }
            }
        }
        float factor[2];
        float rsum[2] = {0.0f, 0.0f};
        for (short i = 0; i < 2; ++i) {
            rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], ushort(1)));
            rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], ushort(8)));
            factor[i] = fast::exp2(m_run[i] - rmax[i]);
            m_run[i] = rmax[i];
        }
        for (short e = 0; e < 16; ++e) {
            const float f = factor[(e >> 2) & 1];
            o0[e] *= f;
            o1[e] *= f;
            o2[e] *= f;
            o3[e] *= f;
        }
        for (short ks = 0; ks < n_ks; ++ks) {
            const int key0 = ks * 16 + fn;
            for (short i = 0; i < 2; ++i) {
                const int r = fm + i * 8;
                for (short j = 0; j < 4; ++j) {
                    const float e = sel[buf][key0 + j] < 0
                        ? 0.0f
                        : fast::exp2(st[r * BK + key0 + j] * scale_log2 - rmax[i]);
                    const bfloat hi = bfloat(e);
                    const float r1 = e - float(hi);
                    const bfloat mid = bfloat(r1);
                    pa[i * 4 + j] = hi;
                    pm[i * 4 + j] = mid;
                    pl[i * 4 + j] = bfloat(r1 - float(mid));
                    rsum[i] += e;
                }
            }
            const int kp0 = sel[buf][ks * 16 + fm];
            const int kp1 = sel[buf][ks * 16 + fm + 8];
            const device T* v0 = kv + ulong(max(kp0, 0)) * D + dq * 128 + fn;
            const device T* v1 = kv + ulong(max(kp1, 0)) * D + dq * 128 + fn;
            for (short np = 0; np < 4; ++np) {
                for (short tn = 0; tn < 2; ++tn) {
                    for (short j = 0; j < 4; ++j) {
                        pb[tn * 8 + j] = v0[np * 32 + tn * 16 + j];
                        pb[tn * 8 + 4 + j] = v1[np * 32 + tn * 16 + j];
                    }
                }
                if (np == 0) {
                    pv_op.run(pa, pb, o0);
                    pv_op.run(pm, pb, o0);
                    pv_op.run(pl, pb, o0);
                } else if (np == 1) {
                    pv_op.run(pa, pb, o1);
                    pv_op.run(pm, pb, o1);
                    pv_op.run(pl, pb, o1);
                } else if (np == 2) {
                    pv_op.run(pa, pb, o2);
                    pv_op.run(pm, pb, o2);
                    pv_op.run(pl, pb, o2);
                } else {
                    pv_op.run(pa, pb, o3);
                    pv_op.run(pm, pb, o3);
                    pv_op.run(pl, pb, o3);
                }
            }
        }
        for (short i = 0; i < 2; ++i) {
            rsum[i] += simd_shuffle_xor(rsum[i], ushort(1));
            rsum[i] += simd_shuffle_xor(rsum[i], ushort(8));
            l_run[i] = l_run[i] * factor[i] + rsum[i];
        }
    }

    for (short i = 0; i < 2; ++i) {
        device T* orow = out + (ulong(head0 + fm + i * 8) * L + qi) * D + dq * 128 + fn;
        const float denom = l_run[i] > 0.0f ? l_run[i] : 1.0f;
        for (short tn = 0; tn < 2; ++tn) {
            for (short j = 0; j < 4; ++j) {
                const short e = tn * 8 + i * 4 + j;
                orow[tn * 16 + j] = T(o0[e] / denom);
                orow[32 + tn * 16 + j] = T(o1[e] / denom);
                orow[64 + tn * 16 + j] = T(o2[e] / denom);
                orow[96 + tn * 16 + j] = T(o3[e] / denom);
            }
        }
    }
"""

# P @ V with the fp32 probabilities as two fp16 pieces (hi + lo, sum within
# 2^-24 absolute of p: the precision of p's own fp32 rounding at 1.0) times
# the bf16 values, as in the Qwen4 QSA tensor-unit kernel; 2 tensor ops per
# value fragment instead of 3.
_SOURCE_HALF2 = """
    constexpr int D = 512;
    constexpr int BK = 128;
    const int L = params[0];
    const int Kn = params[1];
    const int TOPK = params[2];
    const int q_off = params[3];
    const float scale_log2 = scale[0] * 1.44269504088896341f;
    const int qi = int(threadgroup_position_in_grid.x);
    const int hh = int(threadgroup_position_in_grid.y);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const int hg = int(sg) / 4;
    const int dq = int(sg) % 4;   // QK: key quarter; PV: dim quarter
    const int q_abs = q_off + qi;
    const int head0 = hh * 32 + hg * 16;

    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);

    threadgroup float s_tile[2 * 16 * BK];
    threadgroup int sel[2][BK];
    threadgroup int live[2][BK / 32];

    constexpr auto qk_desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    // PV: the fp32 probabilities enter as three bf16 parts (hi + mid + lo
    // == p exactly: 8 + 8 + 8 significant bits), each multiplied with the
    // bf16 values by an exact bf16 x bf16 -> fp32 tensor op.
    constexpr auto pv_desc = matmul2d_descriptor(
        16, 32, 16, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    auto pa = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
    auto pm = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
    auto pb = pv_op.template get_right_input_cooperative_tensor<half, T, float>();
    auto o0 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o1 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o2 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o3 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    for (short e = 0; e < 16; ++e) {
        o0[e] = 0.0f;
        o1[e] = 0.0f;
        o2[e] = 0.0f;
        o3[e] = 0.0f;
    }
    float m_run[2] = {-FLT_MAX, -FLT_MAX};
    float l_run[2] = {0.0f, 0.0f};

    const device T* qr0 = q + (ulong(head0 + fm) * L + qi) * D + fn;
    const device T* qr1 = q + (ulong(head0 + fm + 8) * L + qi) * D + fn;
    const device int32_t* idx_row = idx + ulong(qi) * TOPK;

    const int n_tiles = (TOPK + BK - 1) / BK;
    for (int t = 0; t < n_tiles; ++t) {
        const int buf = t & 1;
        const int tile_keys = min(BK, TOPK - t * BK);
        if (tid < uint(BK)) {
            const int slot = t * BK + int(tid);
            int kp = slot < TOPK ? int(idx_row[slot]) : -1;
            if (kp < 0 || kp >= Kn || kp > q_abs) {
                kp = -1;
            }
            sel[buf][tid] = kp;
            const bool any_live = simd_any(kp >= 0);
            if (lane == 0) {
                live[buf][sg] = any_live ? 1 : 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Unused slots sort last in the indexer's top-k rows: skip tiles
        // with no live key (uniform across the threadgroup).
        if ((live[buf][0] | live[buf][1] | live[buf][2] | live[buf][3]) == 0) {
            continue;
        }

        if (dq * 32 < tile_keys) {
            auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
            auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
            auto sc = qk_op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
            for (short e = 0; e < 16; ++e) {
                sc[e] = 0.0f;
            }
            const device T* kr[2][2];
            for (short tn = 0; tn < 2; ++tn) {
                for (short i = 0; i < 2; ++i) {
                    const int kp = sel[buf][dq * 32 + tn * 16 + fm + i * 8];
                    kr[tn][i] = kv + ulong(max(kp, 0)) * D + fn;
                }
            }
            for (short kk = 0; kk < D; kk += 16) {
                for (short j = 0; j < 4; ++j) {
                    qa[j] = qr0[kk + j];
                    qa[4 + j] = qr1[kk + j];
                }
                for (short tn = 0; tn < 2; ++tn) {
                    for (short i = 0; i < 2; ++i) {
                        for (short j = 0; j < 4; ++j) {
                            kb[tn * 8 + i * 4 + j] = kr[tn][i][kk + j];
                        }
                    }
                }
                qk_op.run(qa, kb, sc);
            }
            threadgroup float* sp = s_tile + hg * 16 * BK + dq * 32;
            for (short tn = 0; tn < 2; ++tn) {
                for (short i = 0; i < 2; ++i) {
                    for (short j = 0; j < 4; ++j) {
                        sp[(fm + i * 8) * BK + tn * 16 + fn + j] = sc[tn * 8 + i * 4 + j];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float* st = s_tile + hg * 16 * BK;
        const int n_ks = (tile_keys + 15) / 16;
        float rmax[2] = {m_run[0], m_run[1]};
        for (short ks = 0; ks < n_ks; ++ks) {
            const int key0 = ks * 16 + fn;
            for (short i = 0; i < 2; ++i) {
                const int r = fm + i * 8;
                for (short j = 0; j < 4; ++j) {
                    if (sel[buf][key0 + j] >= 0) {
                        rmax[i] = max(rmax[i], st[r * BK + key0 + j] * scale_log2);
                    }
                }
            }
        }
        float factor[2];
        float rsum[2] = {0.0f, 0.0f};
        for (short i = 0; i < 2; ++i) {
            rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], ushort(1)));
            rmax[i] = max(rmax[i], simd_shuffle_xor(rmax[i], ushort(8)));
            factor[i] = fast::exp2(m_run[i] - rmax[i]);
            m_run[i] = rmax[i];
        }
        for (short e = 0; e < 16; ++e) {
            const float f = factor[(e >> 2) & 1];
            o0[e] *= f;
            o1[e] *= f;
            o2[e] *= f;
            o3[e] *= f;
        }
        for (short ks = 0; ks < n_ks; ++ks) {
            const int key0 = ks * 16 + fn;
            for (short i = 0; i < 2; ++i) {
                const int r = fm + i * 8;
                for (short j = 0; j < 4; ++j) {
                    const float e = sel[buf][key0 + j] < 0
                        ? 0.0f
                        : fast::exp2(st[r * BK + key0 + j] * scale_log2 - rmax[i]);
                    const half hi = half(e);
                    pa[i * 4 + j] = hi;
                    pm[i * 4 + j] = half(e - float(hi));
                    rsum[i] += e;
                }
            }
            const int kp0 = sel[buf][ks * 16 + fm];
            const int kp1 = sel[buf][ks * 16 + fm + 8];
            const device T* v0 = kv + ulong(max(kp0, 0)) * D + dq * 128 + fn;
            const device T* v1 = kv + ulong(max(kp1, 0)) * D + dq * 128 + fn;
            for (short np = 0; np < 4; ++np) {
                for (short tn = 0; tn < 2; ++tn) {
                    for (short j = 0; j < 4; ++j) {
                        pb[tn * 8 + j] = v0[np * 32 + tn * 16 + j];
                        pb[tn * 8 + 4 + j] = v1[np * 32 + tn * 16 + j];
                    }
                }
                if (np == 0) {
                    pv_op.run(pa, pb, o0);
                    pv_op.run(pm, pb, o0);
                } else if (np == 1) {
                    pv_op.run(pa, pb, o1);
                    pv_op.run(pm, pb, o1);
                } else if (np == 2) {
                    pv_op.run(pa, pb, o2);
                    pv_op.run(pm, pb, o2);
                } else {
                    pv_op.run(pa, pb, o3);
                    pv_op.run(pm, pb, o3);
                }
            }
        }
        for (short i = 0; i < 2; ++i) {
            rsum[i] += simd_shuffle_xor(rsum[i], ushort(1));
            rsum[i] += simd_shuffle_xor(rsum[i], ushort(8));
            l_run[i] = l_run[i] * factor[i] + rsum[i];
        }
    }

    for (short i = 0; i < 2; ++i) {
        device T* orow = out + (ulong(head0 + fm + i * 8) * L + qi) * D + dq * 128 + fn;
        const float denom = l_run[i] > 0.0f ? l_run[i] : 1.0f;
        for (short tn = 0; tn < 2; ++tn) {
            for (short j = 0; j < 4; ++j) {
                const short e = tn * 8 + i * 4 + j;
                orow[tn * 16 + j] = T(o0[e] / denom);
                orow[32 + tn * 16 + j] = T(o1[e] / denom);
                orow[64 + tn * 16 + j] = T(o2[e] / denom);
                orow[96 + tn * 16 + j] = T(o3[e] / denom);
            }
        }
    }
"""

_KERNEL = None


_PV_MODE = os.environ.get("OMLX_GLM_SPARSE_MLA_NAX_PV", "half2").strip().lower()


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        half2 = _PV_MODE == "half2"
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_glm_sparse_mla_nax" + ("_h2" if half2 else ""),
            input_names=["q", "kv", "idx", "params", "scale"],
            output_names=["out"],
            header=_HEADER,
            source=_SOURCE_HALF2 if half2 else _SOURCE,
        )
    return _KERNEL


@lru_cache(maxsize=1)
def nax_sparse_mla_available() -> bool:
    if not _ENABLED:
        return False
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


def sparse_mla_attention_nax(
    q_latent: mx.array,
    kv_latent: mx.array,
    topk_indices: mx.array,
    scale: float,
) -> Optional[mx.array]:
    """Causal sparse MLA prefill for NoPE latents on the tensor units.

    q_latent: [1, H, L, 512], kv_latent: [1, 1, K, 512] (bf16/fp16, the
    last L rows are the queries' own positions), topk_indices:
    [1, 1, L, TOPK] int32 key rows (negative, >= K or past the query's
    position = unused slot). Returns [1, H, L, 512] or None when the inputs
    are outside what the kernel handles.
    """
    if not nax_sparse_mla_available():
        return None
    if (
        q_latent.ndim != 4
        or kv_latent.ndim != 4
        or topk_indices.ndim != 4
        or q_latent.shape[0] != 1
        or kv_latent.shape[:2] != (1, 1)
        or topk_indices.shape[:2] != (1, 1)
        or q_latent.shape[-1] != _D_LATENT
        or kv_latent.shape[-1] != _D_LATENT
        or q_latent.shape[1] % _HEADS_PER_GROUP != 0
        or q_latent.dtype not in (mx.float16, mx.bfloat16)
        or kv_latent.dtype != q_latent.dtype
        or topk_indices.dtype not in (mx.int32, mx.uint32)
    ):
        return None
    _, H, L, _ = q_latent.shape
    K = kv_latent.shape[2]
    topk = topk_indices.shape[-1]
    if L < 1 or K < L or topk_indices.shape[2] != L or topk < 1:
        return None
    idx = topk_indices[0, 0]
    if idx.dtype != mx.int32:
        idx = idx.astype(mx.int32)
    params = mx.array([L, K, topk, K - L], dtype=mx.int32)
    out = _kernel()(
        inputs=[q_latent[0], kv_latent[0, 0], idx, params, mx.array([scale], mx.float32)],
        template=[("T", q_latent.dtype)],
        grid=(L * 256, H // _HEADS_PER_GROUP, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(H, L, _D_LATENT)],
        output_dtypes=[q_latent.dtype],
    )[0]
    return out[None]
