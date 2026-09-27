# SPDX-License-Identifier: Apache-2.0
"""Exact fused decode/verify kernels for GLM-5.3-Flash (glm5_next).

Single-token decode and short verify blocks (L <= 8) are dominated by
thousands of tiny dependent dispatches, which cost both GPU time and host
encode time. The kernels here fuse chains of them while reproducing the
stock MLX arithmetic bit for bit:

* hyper-connections: ``hc_mix`` (fp32 RMS + mix GEMV), ``hc_expand_one``
  (the one-token NAX relaxed-precision comb product + epilogue);
* MoE: ``moe_router`` (logits GEMV + sigmoid/bias, top-k select with the
  stable-sort tie order), ``moe_gate_up_swiglu`` and ``moe_down_combine``
  (routed + shared experts, clamped SwiGLU, routing-weighted sum);
* KDA linear attention: ``kda_decode_step`` (short conv, SiLU, l2norm,
  gate projections, vector-gated delta rule, RMSNormGated);
* DSA indexer: ``dsa_decode_scores`` and ``dsa_expand_topk``.

Exactness rules: every reduction replays the order of the MLX kernel it
replaces (qmv/qmv_quad lane mapping, gemv shuffle ladders, row_reduce
orders, Steel/NAX MMA fragments); every intermediate is rounded where the
reference materializes it; and a product is never contracted into an add
that consumed it in a different reference kernel (separate statements or
``volatile``). ``tests/test_glm5_next_decode_kernels.py`` checks bitwise
equality against the reference op graphs, per kernel and end to end.
"""

from __future__ import annotations

import os
import platform
import re
from collections import Counter
from functools import lru_cache
from typing import Optional

import mlx.core as mx

# Successful fused dispatches by kernel family (graph-build time counts; used
# by tests and profilers to confirm the fused paths engage).
STATS: Counter = Counter()
# Fused families switched off (STATS keys); the callers then take the
# reference path. OMLX_GLM5_DECODE_DISABLE (comma separated) replaces the
# default set, e.g. "" enables everything, "router_rows" disables only that.
# The latent attention kernels are exact but slower than the reference ops in
# the model (one call in flight: 16x32x16 NAX chains of up to 128 dependent
# ops at low occupancy), so they are off unless enabled.
DEFAULT_DISABLED = frozenset({"latent_attn", "latent_sparse_rows"})
_DISABLE_ENV = os.environ.get("OMLX_GLM5_DECODE_DISABLE")
DISABLED = (
    set(DEFAULT_DISABLED)
    if _DISABLE_ENV is None
    else set(filter(None, _DISABLE_ENV.split(",")))
)

_QMV_HEADER = r"""
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <int bits>
constexpr int glm_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits);
}

template <int bits>
constexpr int glm_bytes_per_pack() {
  return ((bits & (bits - 1)) == 0) ? 4 : (bits == 5 ? 5 : 3);
}

// Verbatim copy of MLX quantized.h load_vector (U = float).
template <typename T, int values_per_thread, int bits>
inline float glm_load_vector(const device T* x, thread float* x_thread) {
  float sum = 0;
  if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  } else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

// Verbatim copy of MLX quantized.h qdot (U = float).
template <int values_per_thread, int bits>
inline float glm_qdot(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0;
  if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;
      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  } else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;
      accum += (w[0] & 0x3f) * x_thread[0];
      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);
      accum += (w[2] & 0xfc) * x_thread[3];
    }
  } else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}

// qmv_fast_impl for RPS consecutive rows of one [N, K] affine matrix
// (row pointers already offset to the first row), one simdgroup.  Leaves the
// per-lane partial sums in `result`; the caller simd_sums them.
template <typename T, int K, int group_size, int bits, int RPS>
inline void glm_qmv_rows(
    const device uint8_t* ws,
    const device T* scales,
    const device T* biases,
    const device T* x,
    uint simd_lid,
    thread float* result) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = glm_pack_factor<bits>();
  constexpr int bytes_per_pack = glm_bytes_per_pack<bits>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
  constexpr int in_vec_size_g = K / group_size;

  thread float x_thread[values_per_thread];
  ws += simd_lid * packs_per_thread * bytes_per_pack;
  scales += simd_lid / scale_step_per_thread;
  biases += simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

  for (int k = 0; k < K; k += block_size) {
    float sum = glm_load_vector<T, values_per_thread, bits>(x, x_thread);
    for (int row = 0; row < RPS; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;
      float s = sl[0];
      float b = bl[0];
      result[row] += glm_qdot<values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }
}

// Verbatim copy of MLX quantized.h dequantize (U = float) for 4/5/6/8 bits.
template <int N, int bits>
inline void glm_dequantize(const device uint8_t* w, float scale, float bias, thread float* w_local) {
  const float s = float(scale);
  const float b = float(bias);
  if (bits == 4) {
    float sc[2] = {s, s / 16.0f};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = static_cast<float>(sc[0] * (w[i] & 0x0f) + b);
      w_local[2 * i + 1] = static_cast<float>(sc[1] * (w[i] & 0xf0) + b);
    }
  } else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;
      w_local[0] = static_cast<float>((w[0] & 0x1f) * s + b);
      w_local[1] =
          static_cast<float>((((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * s + b);
      w_local[2] = static_cast<float>(((w[1] & 0x7c) >> 2) * s + b);
      w_local[3] =
          static_cast<float>((((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * s + b);
      w_local[4] =
          static_cast<float>((((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * s + b);
      w_local[5] = static_cast<float>(((w[3] & 0x3e) >> 1) * s + b);
      w_local[6] =
          static_cast<float>((((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * s + b);
      w_local[7] = static_cast<float>(((w[4] & 0xf8) >> 3) * s + b);
    }
  } else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = static_cast<float>((w[0] & 0x3f) * s + b);
      w_local[1] =
          static_cast<float>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
      w_local[2] =
          static_cast<float>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
      w_local[3] = static_cast<float>(((w[2] >> 2) & 0x3f) * s + b);
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = static_cast<float>(s * w[i] + b);
    }
  }
}

// MLX qmv_wide_impl (affine, k_lanes = 8) for one weight row and NV input
// vectors: each lane reduces groups k_lane, k_lane + 8, ... in 8-value
// sub-chunks; the caller applies the 4/2/1 shuffle-down ladder.
template <typename T, int K, int GS, int BITS, int NV>
inline void glm_qmv_wide_row(
    const device uint8_t* wrow,
    const device T* srow,
    const device T* brow,
    const device T* x,
    int nv,
    int k_lane,
    thread float* result) {
  constexpr int sub = 8;
  constexpr int G = K / GS;
  for (int g = k_lane; g < G; g += 8) {
    float scale = srow[g];
    float bias = brow[g];
    for (int sc = 0; sc < GS / sub; sc++) {
      const int k0 = g * GS + sc * sub;
      const device uint8_t* wc = wrow + k0 * BITS / 8;
      float w_dq[sub];
      glm_dequantize<sub, BITS>(wc, scale, bias, w_dq);
      for (int v = 0; v < NV; v++) {
        if (v < nv) {
          const device T* xc = x + v * K + k0;
          float acc = 0;
          for (int i = 0; i < sub; i++) {
            acc += static_cast<float>(xc[i]) * w_dq[i];
          }
          result[v] += acc;
        }
      }
    }
  }
}

// Same expressions as MLX's Sigmoid / Minimum / Maximum functors.
template <typename T>
inline T glm_sigmoid(T x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
// MLX's Sigmoid as its precompiled kernels evaluate it: the release metallib
// is built with -fno-fast-math, so metal::exp is the precise exp there,
// while runtime-compiled kernels (custom kernels, compiled graphs, JIT
// builds) get the default one. See eager_sigmoid_precise().
template <typename T>
inline T glm_sigmoid_precise(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
template <typename T>
inline T glm_minimum(T x, T y) {
  if (metal::isnan(x)) {
    return x;
  }
  return x < y ? x : y;
}
template <typename T>
inline T glm_maximum(T x, T y) {
  if (metal::isnan(x)) {
    return x;
  }
  return x > y ? x : y;
}

// The router's top-k selection (the select kernel's loop, one simdgroup):
// argpartition order of the biased scores, i.e. descending values with ties
// to the lower expert index and NaNs last (lowest index first). Every lane
// ends with the same picked[].
template <int E, int TOPK, typename P>
inline void glm_router_topk(P bz, uint lane, thread int* picked) {
  constexpr int PER = (E + 31) / 32;
  float vals[PER];
  bool taken[PER];
  for (int j = 0; j < PER; j++) {
    const int e = j * 32 + int(lane);
    vals[j] = e < E ? bz[e] : -INFINITY;
    taken[j] = e >= E;
  }
  for (int r = 0; r < TOPK; r++) {
    float best = -INFINITY;
    int best_e = 0x7fffffff;
    for (int j = 0; j < PER; j++) {
      const int e = j * 32 + int(lane);
      if (!taken[j] && !isnan(vals[j]) &&
          (best_e == 0x7fffffff || vals[j] > best || (vals[j] == best && e < best_e))) {
        best = vals[j];
        best_e = e;
      }
    }
    for (ushort off = 16; off >= 1; off >>= 1) {
      float ob = simd_shuffle_xor(best, off);
      int oe = simd_shuffle_xor(best_e, off);
      const bool other_better = oe != 0x7fffffff &&
          (best_e == 0x7fffffff || ob > best || (ob == best && oe < best_e));
      if (other_better) {
        best = ob;
        best_e = oe;
      }
    }
    if (best_e == 0x7fffffff) {
      for (int j = 0; j < PER; j++) {
        const int e = j * 32 + int(lane);
        if (!taken[j] && e < best_e) {
          best_e = e;
        }
      }
      for (ushort off = 16; off >= 1; off >>= 1) {
        best_e = min(best_e, simd_shuffle_xor(best_e, off));
      }
    }
    picked[r] = best_e;
    if ((best_e % 32) == int(lane)) {
      taken[best_e / 32] = true;
    }
  }
}

// Glm5NextClampedSwiGLU / Glm5NextMLP epilogue on bfloat16 projections:
//   silu(minimum(gate, limit)) * minimum(maximum(up, -limit), limit)
template <typename T>
inline T glm_clamped_swiglu(T gate, T up, T limit, T neg_limit) {
  T g = glm_minimum(gate, limit);
  T s = g * glm_sigmoid(g);
  T u = glm_minimum(glm_maximum(up, neg_limit), limit);
  return s * u;
}
"""


# Fused routed-expert (+ optional shared-expert) gate/up projection with the
# clamped SwiGLU epilogue.  One threadgroup z-slice per (token, route); route
# TOPK (when HAS_SHARED) is the shared expert.
_GATE_UP_SOURCE = r"""
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
#if SLOT_MAJOR
  // Route slots vary fastest, so the slots of an expert that several
  // tokens share read its row block back to back (cache hits).
  const int tile = int(threadgroup_position_in_grid.z);
  const int z = int(threadgroup_position_in_grid.y);
#else
  const int tile = int(threadgroup_position_in_grid.y);
  const int z = int(threadgroup_position_in_grid.z);
#endif
  const T lim = T(limit[0]);
  const T neg_lim = T(-limit[0]);
#if SHARED_WIDE
  // Last z slice: the shared expert for all NTOK tokens with MLX's
  // multi-row qmv_wide arithmetic (8 lanes per row, 4 rows per simdgroup).
  if (z == NTOK * TOPK) {
    const int k_lane = int(simd_lid) % 8;
    const int row = (tile * NSG + int(simd_gid)) * 4 + int(simd_lid) / 8;
    constexpr int WB = K * SBITS / 8;
    constexpr int G = K / SGS;
    float g_res[NTOK];
    float u_res[NTOK];
    for (int v = 0; v < NTOK; v++) {
      g_res[v] = 0.0f;
      u_res[v] = 0.0f;
    }
    glm_qmv_wide_row<T, K, SGS, SBITS, NTOK>(
        (const device uint8_t*)sh_gate_w + size_t(row) * WB, sh_gate_s + row * G,
        sh_gate_b + row * G, x, NTOK, k_lane, g_res);
    glm_qmv_wide_row<T, K, SGS, SBITS, NTOK>(
        (const device uint8_t*)sh_up_w + size_t(row) * WB, sh_up_s + row * G,
        sh_up_b + row * G, x, NTOK, k_lane, u_res);
    for (int v = 0; v < NTOK; v++) {
      g_res[v] += simd_shuffle_down(g_res[v], 4);
      g_res[v] += simd_shuffle_down(g_res[v], 2);
      g_res[v] += simd_shuffle_down(g_res[v], 1);
      u_res[v] += simd_shuffle_down(u_res[v], 4);
      u_res[v] += simd_shuffle_down(u_res[v], 2);
      u_res[v] += simd_shuffle_down(u_res[v], 1);
    }
    if (k_lane == 0) {
      for (int v = 0; v < NTOK; v++) {
        shared_out[size_t(v) * N + row] = glm_clamped_swiglu<T>(
            static_cast<T>(g_res[v]), static_cast<T>(u_res[v]), lim, neg_lim);
      }
    }
    return;
  }
  constexpr int RT = TOPK;
#else
  constexpr int RT = TOPK + HAS_SHARED;
#endif
  const int token = z / RT;
  const int r = z - token * RT;
  const int out_row = (tile * NSG + int(simd_gid)) * RPS;
  const device T* xr = x + token * K;

  float g_res[RPS] = {0};
  float u_res[RPS] = {0};
  if (r < TOPK) {
#if SELECT
    // One token: this simdgroup replays the router's selection on the
    // biased sigmoid scores (no separate select dispatch); slot 0 / tile 0
    // publishes the routes and routing weights for the down kernel.
    int picked[TOPK];
    glm_router_topk<NE, TOPK>(sel_biased, simd_lid, picked);
    const int expert = picked[r];
    if (z == 0 && tile == 0 && simd_gid == 0 && simd_lid == 0) {
      float total = 0.0f;
      float gathered[TOPK];
      for (int q = 0; q < TOPK; q++) {
        gathered[q] = sel_sig[picked[q]];
        total = gathered[q] + total;
      }
      for (int q = 0; q < TOPK; q++) {
        float qv = SEL_NORM ? gathered[q] / total : gathered[q];
        float sv = qv * sel_scaling[0];
        sel_indices[q] = uint(picked[q]);
        sel_scores[q] = sv;
      }
    }
#else
    const int expert = int(indices[token * TOPK + r]);
#endif
    constexpr int WB = K * RBITS / 8;   // bytes per weight row
    constexpr int G = K / RGS;          // groups per row
    // ESTRIDE rows per expert; a fused [gate; up] tensor (ESTRIDE = 2N) is
    // passed as both gate and up with the up rows UP_OFF = N further on.
    const size_t row0 = size_t(expert) * ESTRIDE + out_row;
    const size_t urow0 = row0 + UP_OFF;
    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)gate_w + row0 * WB, gate_s + row0 * G,
        gate_b + row0 * G, xr, simd_lid, g_res);
    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)up_w + urow0 * WB, up_s + urow0 * G,
        up_b + urow0 * G, xr, simd_lid, u_res);
  } else {
#if HAS_SHARED
    constexpr int WB = K * SBITS / 8;
    constexpr int G = K / SGS;
    const size_t row0 = size_t(out_row);
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_gate_w + row0 * WB, sh_gate_s + row0 * G,
        sh_gate_b + row0 * G, xr, simd_lid, g_res);
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_up_w + row0 * WB, sh_up_s + row0 * G,
        sh_up_b + row0 * G, xr, simd_lid, u_res);
#endif
  }
  device T* o = out + size_t(z) * N + out_row;
  for (int row = 0; row < RPS; row++) {
    float gv = simd_sum(g_res[row]);
    float uv = simd_sum(u_res[row]);
    if (simd_lid == 0) {
      o[row] = glm_clamped_swiglu<T>(static_cast<T>(gv), static_cast<T>(uv), lim, neg_lim);
    }
  }
"""


# One token's shared-expert gate/up with the clamped SwiGLU: the shared slot
# of the fused gate/up kernel as its own dispatch. It does not depend on the
# router, so it runs concurrently with the router logits kernel (no barrier
# between them) and hides that kernel's latency under its weight stream.
_SHARED_GATE_UP_SOURCE = r"""
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const int tile = int(threadgroup_position_in_grid.y);
  const T lim = T(limit[0]);
  const T neg_lim = T(-limit[0]);
  const int out_row = (tile * NSG + int(simd_gid)) * RPS;
  float g_res[RPS] = {0};
  float u_res[RPS] = {0};
  {
    constexpr int WB = K * SBITS / 8;
    constexpr int G = K / SGS;
    const size_t row0 = size_t(out_row);
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_gate_w + row0 * WB, sh_gate_s + row0 * G,
        sh_gate_b + row0 * G, x, simd_lid, g_res);
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_up_w + row0 * WB, sh_up_s + row0 * G,
        sh_up_b + row0 * G, x, simd_lid, u_res);
  }
  device T* o = out + out_row;
  for (int row = 0; row < RPS; row++) {
    float gv = simd_sum(g_res[row]);
    float uv = simd_sum(u_res[row]);
    if (simd_lid == 0) {
      o[row] = glm_clamped_swiglu<T>(static_cast<T>(gv), static_cast<T>(uv), lim, neg_lim);
    }
  }
"""


@lru_cache(maxsize=None)
def _shared_gate_up_kernel():
    return mx.fast.metal_kernel(
        name="glm5_moe_shared_gate_up_swiglu",
        input_names=["x", "limit", "sh_gate_w", "sh_gate_s", "sh_gate_b", "sh_up_w", "sh_up_s", "sh_up_b"],
        output_names=["out"],
        header=_QMV_HEADER,
        source=_SHARED_GATE_UP_SOURCE,
    )


# Fused routed down projection + routing-weighted sum (+ shared expert down
# projection and residual-free add), reproducing
#   y = (down(act) * scores[..., None]).sum(-2).astype(T) + shared_down(act_s)
_DOWN_SOURCE = r"""
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
#if SLOT_MAJOR
  // Tokens vary fastest: experts they share are read back to back.
  const int tile = int(threadgroup_position_in_grid.z);
  const int token = int(threadgroup_position_in_grid.y);
#else
  const int tile = int(threadgroup_position_in_grid.y);
  const int token = int(threadgroup_position_in_grid.z);
#endif
  // Activation slots per token: the routed ones, then the shared expert's
  // unless it comes from its own input (SH_SEP).
  constexpr int RT = TOPK + HAS_SHARED * (1 - SH_SEP);
  const int out_row = (tile * NSG + int(simd_gid)) * RPS;

  float acc[RPS] = {0};
  constexpr int WB = K * RBITS / 8;
  constexpr int G = K / RGS;
  for (int r = 0; r < TOPK; r++) {
    const int expert = int(indices[token * TOPK + r]);
    const size_t row0 = size_t(expert) * N + out_row;
    float res[RPS] = {0};
    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)down_w + row0 * WB, down_s + row0 * G,
        down_b + row0 * G, act + (size_t(token) * RT + r) * K, simd_lid, res);
    const float score = scores[token * TOPK + r];
    for (int row = 0; row < RPS; row++) {
      float v = simd_sum(res[row]);
      // The reference rounds the fp32 product in its own Multiply kernel
      // before the Sum; keep the compiler from contracting it into an FMA.
      volatile float weighted = static_cast<float>(static_cast<T>(v)) * score;
      acc[row] += weighted;
    }
  }
#if HAS_SHARED
  float sres[RPS] = {0};
  {
    constexpr int SWB = K * SBITS / 8;
    constexpr int SG = K / SGS;
    const size_t row0 = size_t(out_row);
#if SH_SEP
    const device T* sh_x = sh_act + size_t(token) * K;
#else
    const device T* sh_x = act + (size_t(token) * RT + TOPK) * K;
#endif
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_down_w + row0 * SWB, sh_down_s + row0 * SG,
        sh_down_b + row0 * SG, sh_x, simd_lid, sres);
  }
#endif
#if SHARED_WIDE_DOWN
  // Shared expert down projection of this token with the multi-row qmv_wide
  // arithmetic its own [T, K] matmul uses (8 lanes per row; each token's
  // accumulation is independent of the others).
  {
    static_assert(RPS == 4, "qmv_wide rows per simdgroup");
    constexpr int SWB = K * SBITS / 8;
    constexpr int SG = K / SGS;
    const int k_lane = int(simd_lid) % 8;
    const int srow = out_row + int(simd_lid) / 8;
    float sv[1] = {0.0f};
    glm_qmv_wide_row<T, K, SGS, SBITS, 1>(
        (const device uint8_t*)sh_down_w + size_t(srow) * SWB, sh_down_s + srow * SG,
        sh_down_b + srow * SG, sh_act + size_t(token) * K, 1, k_lane, sv);
    sv[0] += simd_shuffle_down(sv[0], 4);
    sv[0] += simd_shuffle_down(sv[0], 2);
    sv[0] += simd_shuffle_down(sv[0], 1);
    if (k_lane == 0) {
      const int r = int(simd_lid) / 8;
      float a = acc[0];
      for (int row = 1; row < RPS; row++) {
        a = row == r ? acc[row] : a;
      }
      out[size_t(token) * N + srow] = static_cast<T>(a) + static_cast<T>(sv[0]);
    }
  }
  return;
#endif
  device T* o = out + size_t(token) * N + out_row;
  for (int row = 0; row < RPS; row++) {
#if HAS_SHARED
    float sv = simd_sum(sres[row]);
#endif
    if (simd_lid == 0) {
#if HAS_SHARED
      o[row] = static_cast<T>(acc[row]) + static_cast<T>(sv);
#elif ADD_SHARED_Y
      o[row] = static_cast<T>(acc[row]) + shared_y[size_t(token) * N + out_row + row];
#else
      o[row] = static_cast<T>(acc[row]);
#endif
    }
  }
"""


def _source(body: str, **defines) -> str:
    lines = [f"#define {k} {int(v)}" for k, v in defines.items()]
    undef = [f"#undef {k}" for k in defines]
    return "\n".join(lines) + "\n" + body + "\n" + "\n".join(undef) + "\n"


@lru_cache(maxsize=None)
def _gate_up_kernel(
    has_shared: bool,
    shared_wide: bool = False,
    slot_major: bool = False,
    select: bool = False,
):
    routes = ["sel_sig", "sel_biased", "sel_scaling"] if select else ["indices"]
    inputs = ["x"] + routes + ["limit", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b"]
    if has_shared or shared_wide:
        inputs += ["sh_gate_w", "sh_gate_s", "sh_gate_b", "sh_up_w", "sh_up_s", "sh_up_b"]
    if select and not has_shared:
        # The shared expert's own gate/up output: read by nothing here, it
        # orders that dispatch (and the router's) before this one.
        inputs += ["after_shared"]
    suffix = "_widesh" if shared_wide else ("_shared" if has_shared else "")
    suffix += "_sm" if slot_major else ""
    suffix += "_select" if select else ""
    outputs = ["out", "shared_out"] if shared_wide else ["out"]
    if select:
        outputs += ["sel_indices", "sel_scores"]
    return mx.fast.metal_kernel(
        name=f"glm5_moe_gate_up_swiglu{suffix}",
        input_names=inputs,
        output_names=outputs,
        header=_QMV_HEADER,
        source=_source(
            _GATE_UP_SOURCE,
            HAS_SHARED=int(has_shared and not shared_wide),
            SHARED_WIDE=int(shared_wide),
            SLOT_MAJOR=int(slot_major),
            SELECT=int(select),
        ),
    )


@lru_cache(maxsize=None)
def _down_kernel(
    has_shared: bool,
    add_shared_y: bool,
    slot_major: bool = False,
    shared_wide: bool = False,
    shared_sep: bool = False,
):
    inputs = ["act", "indices", "scores", "down_w", "down_s", "down_b"]
    if has_shared:
        inputs += ["sh_down_w", "sh_down_s", "sh_down_b"]
        if shared_wide or shared_sep:
            inputs += ["sh_act"]
    elif add_shared_y:
        inputs += ["shared_y"]
    suffix = "_widesh" if shared_wide else (
        "_shared" if has_shared else ("_add" if add_shared_y else "")
    )
    suffix += "_sm" if slot_major else ""
    suffix += "_shsep" if shared_sep else ""
    return mx.fast.metal_kernel(
        name=f"glm5_moe_down_combine{suffix}",
        input_names=inputs,
        output_names=["out"],
        header=_QMV_HEADER,
        source=_source(
            _DOWN_SOURCE,
            HAS_SHARED=int(has_shared and not shared_wide),
            ADD_SHARED_Y=int(add_shared_y and not has_shared),
            SLOT_MAJOR=int(slot_major),
            SHARED_WIDE_DOWN=int(shared_wide),
            SH_SEP=int(shared_sep and not shared_wide),
        ),
    )


def _qmv_fast_ok(bits: int, group_size: int, n: int, k: int) -> bool:
    """Shapes on which MLX routes a one-token product to qmv_fast."""
    if bits not in (4, 5, 6, 8) or group_size not in (32, 64, 128):
        return False
    pack_factor = 8 if bits == 5 else (4 if bits == 6 else 32 // bits)
    values_per_thread = pack_factor * 2
    if group_size % values_per_thread:
        return False
    return n % 8 == 0 and k % (values_per_thread * 32) == 0


def _affine_parts(layer):
    """(weight, scales, biases, bits, group_size) of an affine quantized layer."""
    if getattr(layer, "mode", "affine") != "affine":
        return None
    biases = layer.get("biases") if hasattr(layer, "get") else getattr(layer, "biases", None)
    if biases is None or "bias" in layer:
        return None
    return layer["weight"], layer["scales"], biases, int(layer.bits), int(layer.group_size)


def moe_gate_up_swiglu(
    x: mx.array,
    indices: mx.array,
    limit: float,
    routed_gate,
    routed_up,
    shared_gate=None,
    shared_up=None,
    *,
    rps: int = 4,
    nsg: int = 2,
    shared_wide: bool = False,
    select=None,
    split_shared: bool = False,
):
    """Clamped-SwiGLU activations for every (token, routed expert[, shared]).

    ``x`` is [T, K] (one row per token), ``indices`` [T, TOPK].  Returns
    [T, TOPK (+1), N] in ``x.dtype`` or None when the shapes are not covered.
    With ``shared_wide`` (2 <= T <= 8) the shared expert uses the multi-row
    qmv_wide arithmetic the reference applies to T > 1 rows and the call
    returns ``(routed [T, TOPK, N], shared [T, N])``. ``routed_up=None``
    means ``routed_gate`` is a fused ``gate_up_proj`` ([E, 2N, *]: gate rows
    then up rows per expert, as the MoE gate/up fusion lays them out).

    ``select = (sig, biased, top_k, scaling, norm_topk_prob)`` (one token,
    the ``moe_router_logits`` outputs) replaces ``indices``: every routed
    threadgroup replays the router's top-k selection, and the call returns
    ``(act, indices [1, top_k] uint32, scores [1, top_k] fp32)`` like
    ``moe_router`` + this kernel, or None when not covered. With
    ``split_shared`` too, the shared expert runs as its own dispatch first
    (independent of the router, so it overlaps the router logits kernel)
    and the call returns ``(act [1, top_k, N], shared_act [1, N], indices,
    scores)`` for ``moe_down_combine(..., shared_act_sep=shared_act)``.
    """
    if "moe_gate_up" in DISABLED:
        return None
    if select is not None:
        if "router_select_fused" in DISABLED or shared_wide or x.ndim != 2 or x.shape[0] != 1:
            return None
        sig, biased, sel_topk, sel_scaling, sel_norm = select
        E_r = sig.shape[-1]
        if sig.shape != (1, E_r) or biased.shape != (1, E_r) or not 1 <= sel_topk <= 32:
            return None
        if sig.dtype != mx.float32 or biased.dtype != mx.float32 or E_r > 1024:
            return None
        indices = mx.zeros((1, sel_topk), dtype=mx.uint32)  # shape only
    fused_gu = routed_up is None
    parts = [_affine_parts(m) for m in ((routed_gate,) if fused_gu else (routed_gate, routed_up))]
    if any(p is None for p in parts) or x.ndim != 2 or indices.ndim != 2:
        return None
    if fused_gu:
        parts = parts * 2
    (gw, gs, gb, rbits, rgs), (uw, us, ub, ubits, ugs) = parts
    if (rbits, rgs) != (ubits, ugs) or gw.shape != uw.shape or gw.ndim != 3:
        return None
    T, K = x.shape
    E, N, _ = gw.shape
    estride, up_off = N, 0
    if fused_gu:
        if N % 2:
            return None
        N //= 2
        estride, up_off = 2 * N, N
    topk = indices.shape[1]
    if x.dtype not in (mx.bfloat16, mx.float16) or gs.dtype != x.dtype or us.dtype != x.dtype:
        return None
    if not _qmv_fast_ok(rbits, rgs, N, K) or N % (rps * nsg):
        return None
    has_shared = shared_gate is not None
    if shared_wide and (not has_shared or not 2 <= T <= 8 or rps != 4):
        return None
    if select is not None:
        routes = [sig, biased, mx.array([sel_scaling], dtype=mx.float32)]
    else:
        routes = [indices]
    inputs = [x] + routes + [mx.array([limit], dtype=mx.float32), gw, gs, gb, uw, us, ub]
    template = [
        ("T", x.dtype), ("K", K), ("N", N), ("TOPK", topk), ("RBITS", rbits),
        ("RGS", rgs), ("RPS", rps), ("NSG", nsg), ("ESTRIDE", estride), ("UP_OFF", up_off),
    ]
    if select is not None:
        template += [("NE", E_r), ("SEL_NORM", int(bool(sel_norm) and sel_topk > 1))]
    if has_shared:
        sparts = [_affine_parts(m) for m in (shared_gate, shared_up)]
        if any(p is None for p in sparts):
            return None
        (sgw, sgs, sgb, sbits, sgsz), (suw, sus, sub, subits, susz) = sparts
        if (sbits, sgsz) != (subits, susz) or sgw.shape[0] != N or suw.shape[0] != N:
            return None
        if sgs.dtype != x.dtype or sus.dtype != x.dtype:
            return None
        if shared_wide:
            # qmv_wide: groups decoded in 8-value sub-chunks, 8 lanes per row.
            if sbits not in (4, 5, 6, 8) or sgsz % 8 or K % sgsz or (K // sgsz) < 1:
                return None
        elif not _qmv_fast_ok(sbits, sgsz, N, K):
            return None
        inputs += [sgw, sgs, sgb, suw, sus, sub]
        template += [("SBITS", sbits), ("SGS", sgsz)]
    if shared_wide and "moe_shared_wide" in DISABLED:
        return None
    slot_major = _slot_major(T)
    if split_shared and select is not None and has_shared and "moe_shared_split" not in DISABLED:
        tiles = N // (rps * nsg)
        limit_arr = inputs[len(routes) + 1]
        act_sh = _shared_gate_up_kernel()(
            inputs=[x, limit_arr] + inputs[-6:],
            template=[("T", x.dtype), ("K", K), ("N", N), ("SBITS", sbits), ("SGS", sgsz),
                      ("RPS", rps), ("NSG", nsg)],
            grid=(32, tiles * nsg, 1),
            threadgroup=(32, nsg, 1),
            output_shapes=[(1, N)],
            output_dtypes=[x.dtype],
        )[0]
        STATS["moe_gate_up"] += 1
        STATS["router_select_fused"] += 1
        STATS["moe_shared_split"] += 1
        act, sel_indices, sel_scores = _gate_up_kernel(False, False, False, True)(
            inputs=inputs[:-6] + [act_sh],
            template=[t for t in template if t[0] not in ("SBITS", "SGS")],
            grid=(32, tiles * nsg, topk),
            threadgroup=(32, nsg, 1),
            output_shapes=[(1, topk, N), (1, topk), (1, topk)],
            output_dtypes=[x.dtype, mx.uint32, mx.float32],
        )
        return act, act_sh, sel_indices, sel_scores
    kernel = _gate_up_kernel(has_shared, shared_wide, slot_major, select is not None)
    STATS["moe_gate_up"] += 1
    tiles = N // (rps * nsg)
    slots = T * topk + 1 if shared_wide else T * (topk + int(has_shared))
    grid = (32, slots * nsg, tiles) if slot_major else (32, tiles * nsg, slots)
    if shared_wide:
        template += [("NTOK", T)]
        routed, shared = kernel(
            inputs=inputs,
            template=template,
            grid=grid,
            threadgroup=(32, nsg, 1),
            output_shapes=[(T, topk, N), (T, N)],
            output_dtypes=[x.dtype, x.dtype],
        )
        STATS["moe_shared_wide"] += 1
        return routed, shared
    rt = topk + int(has_shared)
    if select is not None:
        STATS["router_select_fused"] += 1
        act, sel_indices, sel_scores = kernel(
            inputs=inputs,
            template=template,
            grid=grid,
            threadgroup=(32, nsg, 1),
            output_shapes=[(T, rt, N), (1, topk), (1, topk)],
            output_dtypes=[x.dtype, mx.uint32, mx.float32],
        )
        return act, sel_indices, sel_scores
    return kernel(
        inputs=inputs,
        template=template,
        grid=grid,
        threadgroup=(32, nsg, 1),
        output_shapes=[(T, rt, N)],
        output_dtypes=[x.dtype],
    )[0]


def _slot_major(tokens: int) -> bool:
    """Route-slot-major grids for blocks of 2+ tokens (they share experts)."""
    return tokens > 1 and "moe_slot_major" not in DISABLED


def moe_down_combine(
    act: mx.array,
    indices: mx.array,
    scores: mx.array,
    routed_down,
    shared_down=None,
    shared_y: Optional[mx.array] = None,
    *,
    shared_act: Optional[mx.array] = None,
    shared_act_sep: Optional[mx.array] = None,
    rps: int = 4,
    nsg: int = 2,
) -> Optional[mx.array]:
    """Routed down projections combined with the routing weights (+ shared).

    ``act`` is [T, TOPK (+1), K] from :func:`moe_gate_up_swiglu`, ``scores``
    [T, TOPK] float32.  The shared expert is either projected here from the
    last activation slot (``shared_down``), from ``shared_act`` [T, K] (the
    ``shared_wide`` gate/up output, with the multi-row qmv_wide arithmetic)
    or added from a precomputed ``shared_y`` [T, N].  Returns [T, N] in
    ``act.dtype``. ``shared_act_sep`` [T, K] is the shared expert's one-token
    activation from its own gate/up dispatch (``act`` then has only the
    routed slots); the arithmetic is the in-``act`` shared slot's.
    """
    if "moe_down" in DISABLED:
        return None
    p = _affine_parts(routed_down)
    if p is None or act.ndim != 3 or scores.dtype != mx.float32:
        return None
    dw, ds, db, rbits, rgs = p
    T, rt, K = act.shape
    E, N, _ = dw.shape
    topk = indices.shape[1]
    has_shared = shared_down is not None
    shared_wide = shared_act is not None
    if shared_wide and (not has_shared or shared_y is not None or rps != 4):
        return None
    if shared_wide and ("moe_shared_wide" in DISABLED or not 2 <= T <= 8):
        return None
    shared_sep = shared_act_sep is not None
    if shared_sep and (not has_shared or shared_wide or shared_y is not None):
        return None
    if rt != topk + int(has_shared and not shared_wide and not shared_sep) or ds.dtype != act.dtype:
        return None
    if not _qmv_fast_ok(rbits, rgs, N, K) or N % (rps * nsg):
        return None
    inputs = [act, indices, scores, dw, ds, db]
    template = [
        ("T", act.dtype), ("K", K), ("N", N), ("TOPK", topk), ("RBITS", rbits),
        ("RGS", rgs), ("RPS", rps), ("NSG", nsg),
    ]
    if has_shared:
        sp = _affine_parts(shared_down)
        if sp is None:
            return None
        sdw, sds, sdb, sbits, sgsz = sp
        if sdw.shape[0] != N or sds.dtype != act.dtype:
            return None
        if shared_wide:
            if shared_act.shape != (T, K) or shared_act.dtype != act.dtype:
                return None
            if sbits not in (4, 5, 6, 8) or sgsz % 8 or K % sgsz or K in (64, 128):
                return None
        elif not _qmv_fast_ok(sbits, sgsz, N, K):
            return None
        if shared_sep and (shared_act_sep.shape != (T, K) or shared_act_sep.dtype != act.dtype):
            return None
        inputs += [sdw, sds, sdb] + ([shared_act] if shared_wide else [])
        inputs += [shared_act_sep] if shared_sep else []
        template += [("SBITS", sbits), ("SGS", sgsz)]
    elif shared_y is not None:
        if shared_y.shape != (T, N) or shared_y.dtype != act.dtype:
            return None
        inputs.append(shared_y)
    slot_major = _slot_major(T)
    kernel = _down_kernel(has_shared, shared_y is not None, slot_major, shared_wide, shared_sep)
    STATS["moe_down"] += 1
    if shared_wide:
        STATS["moe_down_shared_wide"] += 1
    tiles = N // (rps * nsg)
    return kernel(
        inputs=inputs,
        template=template,
        grid=(32, T * nsg, tiles) if slot_major else (32, tiles * nsg, T),
        threadgroup=(32, nsg, 1),
        output_shapes=[(T, N)],
        output_dtypes=[act.dtype],
    )[0]


# ---------------------------------------------------------------------------
# Hyper-connection mix: x.astype(f32) -> rms_norm (no weight) -> @ fn.T
# ---------------------------------------------------------------------------
#
# Reproduces MLX's ``rms_looped`` (1024 threads, 4 reads per thread) for the
# inverse RMS and the non-transposed ``gemv`` kernel that MLX selects for a
# [1, HC*D] x [HC*D, MIX] product with MIX < 4096 and K >= 16 * MIX
# (BM=1, BN=8, SM=1, SN=32, TN=4): every output row is reduced by eight
# simdgroups, each lane accumulating 4 contiguous products per 1024-wide K
# block, a shuffle-down ladder inside the simdgroup and a sequential sum over
# the eight simdgroups.  Each threadgroup recomputes the (cheap) RMS and
# owns ROWS_PER_TG output rows, so the product runs on many more cores than
# MLX's 4-rows-per-threadgroup gemv.
_HC_MIX_SOURCE = r"""
  const uint lid = thread_position_in_threadgroup.x;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const int tok = int(threadgroup_position_in_grid.y);
  const int tile = int(threadgroup_position_in_grid.x);
  constexpr int KSZ = HCD;           // flattened HC * D
  const device T* xr = x + size_t(tok) * KSZ;

  // --- rms_looped (lsize = 1024, N_READS = 4) ---
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  float acc = 0;
  for (uint r = 0; r < uint(KSZ); r += 1024 * 4) {
    for (int i = 0; i < 4; i++) {
      float xi = static_cast<float>(xr[r + lid * 4 + i]);
      acc += xi * xi;
    }
  }
  acc = simd_sum(acc);
  if (simd_gid == 0) {
    local_sums[simd_lid] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lid == 0) {
    local_sums[simd_gid] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    acc = simd_sum(local_sums[simd_lid]);
    if (simd_lid == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / KSZ + eps[0]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = local_inv_mean[0];

  // --- gemv rows: 8 simdgroups per row, 4 row slots per threadgroup ---
  const int slot = int(simd_gid) / 8;
  const int sgN = int(simd_gid) % 8;
  threadgroup float partial[4][8];
  for (int rr = 0; rr < ROWS_PER_TG; rr += 4) {
    const int row = tile * ROWS_PER_TG + rr + slot;
    float result = 0;
    if (rr + slot < ROWS_PER_TG && row < MIX) {
      const device float* mrow = fn + size_t(row) * KSZ;
      int bn = (32 * sgN + int(simd_lid)) * 4;
      for (int i = 0; i < KSZ / 1024; ++i) {
        float v_coeff[4];
        float inter[4];
        for (int tn = 0; tn < 4; tn++) {
          v_coeff[tn] = static_cast<float>(xr[bn + tn]) * inv;
        }
        for (int tn = 0; tn < 4; tn++) {
          inter[tn] = mrow[bn + tn];
        }
        for (int tn = 0; tn < 4; tn++) {
          result += inter[tn] * v_coeff[tn];
        }
        bn += 1024;
      }
      for (ushort sn = 16; sn >= 1; sn >>= 1) {
        result += simd_shuffle_down(result, sn);
      }
    }
    if (simd_lid == 0) {
      partial[slot][sgN] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgN == 0 && simd_lid == 0 && rr + slot < ROWS_PER_TG && row < MIX) {
      float total = partial[slot][0];
      for (int s = 1; s < 8; s++) {
        total += partial[slot][s];
      }
      mixes[size_t(tok) * MIX + row] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""


# One mix row per 256-thread threadgroup (MIX threadgroups per token instead
# of MIX / 4 threadgroups of 1024 threads): the same rms_looped reduction tree,
# with each real simdgroup playing four of the 32 virtual 1024-thread
# simdgroups, and the same 8-simdgroup gemv split and sequential final sum.
_HC_MIX1_SOURCE = r"""
  const uint lid = thread_position_in_threadgroup.x;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const int tok = int(threadgroup_position_in_grid.y);
  const int row = int(threadgroup_position_in_grid.x);
  constexpr int KSZ = HCD;
  const device T* xr = x + size_t(tok) * KSZ;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  for (int v = 0; v < 4; v++) {
    const uint vt = uint(v) * 256 + lid;
    float acc = 0;
    for (uint r = 0; r < uint(KSZ); r += 1024 * 4) {
      for (int i = 0; i < 4; i++) {
        float xi = static_cast<float>(xr[r + vt * 4 + i]);
        acc += xi * xi;
      }
    }
    acc = simd_sum(acc);
    if (simd_lid == 0) {
      local_sums[v * 8 + int(simd_gid)] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    float acc = simd_sum(local_sums[simd_lid]);
    if (simd_lid == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / KSZ + eps[0]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = local_inv_mean[0];

  threadgroup float partial[8];
  const int sgN = int(simd_gid);
  float result = 0;
  const device float* mrow = fn + size_t(row) * KSZ;
  int bn = (32 * sgN + int(simd_lid)) * 4;
  for (int i = 0; i < KSZ / 1024; ++i) {
    float v_coeff[4];
    float inter[4];
    for (int tn = 0; tn < 4; tn++) {
      v_coeff[tn] = static_cast<float>(xr[bn + tn]) * inv;
    }
    for (int tn = 0; tn < 4; tn++) {
      inter[tn] = mrow[bn + tn];
    }
    for (int tn = 0; tn < 4; tn++) {
      result += inter[tn] * v_coeff[tn];
    }
    bn += 1024;
  }
  for (ushort sn = 16; sn >= 1; sn >>= 1) {
    result += simd_shuffle_down(result, sn);
  }
  if (simd_lid == 0) {
    partial[sgN] = result;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgN == 0 && simd_lid == 0) {
    float total = partial[0];
    for (int s = 1; s < 8; s++) {
      total += partial[s];
    }
    mixes[size_t(tok) * MIX + row] = total;
  }
"""


@lru_cache(maxsize=None)
def _hc_mix1_kernel():
    return mx.fast.metal_kernel(
        name="glm5_hc_mix_rms_gemv_row",
        input_names=["x", "fn", "eps"],
        output_names=["mixes"],
        source=_HC_MIX1_SOURCE,
    )


@lru_cache(maxsize=None)
def _hc_mix_kernel():
    return mx.fast.metal_kernel(
        name="glm5_hc_mix_rms_gemv",
        input_names=["x", "fn", "eps"],
        output_names=["mixes"],
        source=_HC_MIX_SOURCE,
    )


def hc_mix(x: mx.array, fn: mx.array, eps: float, *, rows_per_tg: int = 0) -> Optional[mx.array]:
    """``(rms_norm(x.astype(f32).flatten(-2)) @ fn.T)`` per token, M=1 exact.

    ``x`` is [B, L, HC, D] bf16/fp16, ``fn`` [MIX, HC*D] float32.  Returns
    [B, L, MIX] float32 or None when the shape is outside the replicated
    kernel configuration.
    """
    if "hc_mix" in DISABLED:
        return None
    if x.ndim != 4 or fn.ndim != 2 or fn.dtype != mx.float32:
        return None
    B, L, hc, d = x.shape
    K = hc * d
    mix = fn.shape[0]
    if fn.shape[1] != K or K % 4096 or mix >= 4096 or K < 16 * mix:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return None
    STATS["hc_mix"] += 1
    if rows_per_tg == 0:
        return _hc_mix1_kernel()(
            inputs=[x, fn, mx.array([eps], dtype=mx.float32)],
            template=[("T", x.dtype), ("HCD", K), ("MIX", mix)],
            grid=(256 * mix, B * L, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(B, L, mix)],
            output_dtypes=[mx.float32],
        )[0]
    tiles = (mix + rows_per_tg - 1) // rows_per_tg
    out = _hc_mix_kernel()(
        inputs=[x, fn, mx.array([eps], dtype=mx.float32)],
        template=[("T", x.dtype), ("HCD", K), ("MIX", mix), ("ROWS_PER_TG", rows_per_tg)],
        grid=(1024 * tiles, B * L, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(B, L, mix)],
        output_dtypes=[mx.float32],
    )[0]
    return out


# ---------------------------------------------------------------------------
# DSA indexer: decode/verify scores and top-k index expansion
# ---------------------------------------------------------------------------
#
# The prefill score kernel (Steel GEMM tile, BM=64) is launched with the
# query rows zero-padded to 64 and only P/64 threadgroups, which makes it the
# single most expensive decode kernel once the context passes 2k tokens.  The
# kernel below computes the same values for up to eight query rows: per head
# it accumulates the 8x8 simdgroup MMAs over D in the same 8-wide K order as
# the Steel tile (float fragments, zero rows for missing queries), and it
# adds max(score, 0) * weight over the heads in the same sequential order.
# Invalid pooled positions receive the same -1e30 sentinel the Python path
# writes with ``mx.where``.
_DSA_SCORES_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint tid = thread_position_in_threadgroup.x + 32 * sg;
  const int key0 = int(threadgroup_position_in_grid.x) * 8;
  const int P = int(pool_len_cap[1]);
  const int pool_len = int(pool_len_cap[0]);
  const int qpos0 = int(qpos[0]);

  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;

  threadgroup float hs[HEADS][8][8];

  // B fragments (K x 8 keys) for this key block, kept in registers.
  simdgroup_matrix<float, 8, 8> bfrag[DIM / 8];
  for (int kb = 0; kb < DIM / 8; kb++) {
    float2 bv = float2(0.0f);
    for (short e = 0; e < 2; e++) {
      int key = key0 + fn + e;
      if (key < P) {
        bv[e] = static_cast<float>(keys[size_t(key) * DIM + kb * 8 + fm]);
      }
    }
    reinterpret_cast<thread float2&>(bfrag[kb].thread_elements()) = bv;
  }

  for (int hh = 0; hh < HEADS / NSG; hh++) {
    const int h = int(sg) * (HEADS / NSG) + hh;
    simdgroup_matrix<float, 8, 8> c = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int kb = 0; kb < DIM / 8; kb++) {
      float2 av = float2(0.0f);
      if (fm < L) {
        const device T* qr = q + (size_t(fm) * HEADS + h) * DIM + kb * 8 + fn;
        av[0] = static_cast<float>(qr[0]);
        av[1] = static_cast<float>(qr[1]);
      }
      simdgroup_matrix<float, 8, 8> a;
      reinterpret_cast<thread float2&>(a.thread_elements()) = av;
      simdgroup_multiply_accumulate(c, a, bfrag[kb], c);
    }
    float2 cv = reinterpret_cast<thread float2&>(c.thread_elements());
    hs[h][fm][fn] = cv[0];
    hs[h][fm][fn + 1] = cv[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < uint(L * 8)) {
    const int row = int(tid) / 8;
    const int j = int(tid) % 8;
    const int key = key0 + j;
    if (key < P) {
      float accum = 0.0f;
      for (int h = 0; h < HEADS; h++) {
        const float weight = static_cast<float>(w[row * HEADS + h]);
        accum += max(hs[h][row][j], 0.0f) * weight;
      }
      const bool valid = key < pool_len && (key + 1) * KPOOL - 1 <= qpos0 + row;
      scores[size_t(row) * P + key] = valid ? static_cast<T>(accum) : static_cast<T>(-1e30f);
    }
  }
"""


@lru_cache(maxsize=None)
def _dsa_scores_kernel():
    return mx.fast.metal_kernel(
        name="glm5_dsa_decode_scores",
        input_names=["q", "keys", "w", "qpos", "pool_len_cap"],
        output_names=["scores"],
        header="#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n",
        source=_DSA_SCORES_SOURCE,
    )


def dsa_decode_scores(
    q: mx.array,
    pool_keys: mx.array,
    weights: mx.array,
    query_pos0: int,
    pool_len: int,
    kpool: int,
    *,
    nsg: int = 8,
) -> Optional[mx.array]:
    """Masked indexer scores for L <= 8 query rows of one sequence.

    ``q`` [1, L, H, D], ``pool_keys`` [1, P, D],
    ``weights`` [1, L, H] (already scaled, q dtype).  Returns [1, L, P]
    scores equal to the padded Steel kernel followed by the validity
    ``mx.where``.
    """
    if "dsa_scores" in DISABLED:
        return None
    if q.ndim != 4 or q.shape[0] != 1 or pool_keys.ndim != 3 or pool_keys.shape[0] != 1:
        return None
    _, L, H, D = q.shape
    P = pool_keys.shape[1]
    if not (1 <= L <= 8) or D % 8 or H % nsg or P == 0:
        return None
    if q.dtype not in (mx.bfloat16, mx.float16) or pool_keys.dtype != q.dtype or weights.dtype != q.dtype:
        return None
    # Inputs are made row contiguous by the kernel launch (a no-op for the
    # pooled cache view, whose rows are contiguous for one sequence).
    STATS["dsa_scores"] += 1
    return _dsa_scores_kernel()(
        inputs=[
            q,
            pool_keys,
            weights,
            mx.array([query_pos0], dtype=mx.int32),
            mx.array([pool_len, P], dtype=mx.int32),
        ],
        template=[("T", q.dtype), ("L", L), ("HEADS", H), ("DIM", D), ("NSG", nsg), ("KPOOL", kpool)],
        grid=(32 * ((P + 7) // 8), nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(1, L, P)],
        output_dtypes=[q.dtype],
    )[0]


# Expands the selected pooled blocks into token indices exactly like
# Glm5NextIndexer.__call__ (validity, kpool expansion, left padding, the
# always-selected tail window and the -1 padding up to the output width).
_DSA_EXPAND_SOURCE = r"""
  const int col = int(thread_position_in_grid.x);
  const int row = int(thread_position_in_grid.y);
  if (col >= OUT_W) {
    return;
  }
  const int qp = int(qpos[0]) + row;
  const int pool_len = int(pool_len_arr[0]);
  const int lp = int(left_padding[0]);
  int v = -1;
  if (col < SEL_K * KPOOL) {
    const int s = int(selected[row * SEL_K + col / KPOOL]);
    const bool valid = s < pool_len && (s + 1) * KPOOL - 1 <= qp;
    if (valid) {
      v = s * KPOOL + (col % KPOOL) + lp;
    }
  } else if (TAIL_W > 0 && col < SEL_K * KPOOL + TAIL_W) {
    const int t = col - SEL_K * KPOOL;
    const int tail_count = (qp + 1) % KPOOL;
    if (t < tail_count) {
      v = qp + 1 - tail_count + t + lp;
    }
  }
  out[row * OUT_W + col] = v;
"""


@lru_cache(maxsize=None)
def _dsa_expand_kernel():
    return mx.fast.metal_kernel(
        name="glm5_dsa_expand_topk",
        input_names=["selected", "qpos", "pool_len_arr", "left_padding"],
        output_names=["out"],
        source=_DSA_EXPAND_SOURCE,
    )


def dsa_expand_topk(
    selected: mx.array,
    query_pos0: int,
    pool_len: int,
    left_padding: mx.array,
    kpool: int,
    tail_width: int,
    output_width: int,
) -> mx.array:
    """[1, L, SEL_K] selected pool rows -> [1, 1, L, output_width] int32.

    ``left_padding`` is the KV cache's [1] padding array (kept on device).
    """
    _, L, sel_k = selected.shape
    return _dsa_expand_kernel()(
        inputs=[
            selected,
            mx.array([query_pos0], dtype=mx.int32),
            mx.array([pool_len], dtype=mx.int32),
            left_padding.astype(mx.int32) if left_padding.dtype != mx.int32 else left_padding,
        ],
        template=[("SEL_K", sel_k), ("KPOOL", kpool), ("TAIL_W", tail_width), ("OUT_W", output_width)],
        grid=(output_width, L, 1),
        threadgroup=(min(256, output_width), 1, 1),
        output_shapes=[(1, 1, L, output_width)],
        output_dtypes=[mx.int32],
    )[0]


# ---------------------------------------------------------------------------
# KDA (linear attention) decode/verify step
# ---------------------------------------------------------------------------
#
# Everything between the fused input projection and o_proj of a
# Glm5NextLinearAttention layer, for one sequence and T <= 8 tokens, in one
# dispatch with one 1024-thread threadgroup per head:
#
#   1. depthwise short conv over [conv_state, q|k|v] + SiLU (bf16), and the
#      new conv state (MLX depthwise_conv_1d + the compiled nn.silu);
#   2. l2-normalization of q (with the 1/sqrt(Dk) scale) and k (fp32, the
#      row_reduce_simple order of MLX's Sum);
#   3. the forget-gate / output-gate low-rank projections (MLX qmv_quad for
#      K == 128), the safe gate g (compiled compute_g_safe) and beta =
#      sigmoid(b);
#   4. the vector-gated delta rule (the vendored gated_delta_step_vec
#      kernel, statement for statement);
#   5. Glm5NextRMSNormGated (fp32, row_reduce_simple order, separate
#      product/sum roundings).
#
# Each reference op is its own kernel there, so every intermediate is rounded
# to its dtype here too and products are never contracted into the adds that
# consumed them in a different kernel.
_SIGMOID_PROBE_SOURCE = r"""
  const uint i = thread_position_in_grid.x;
  default_out[i] = glm_sigmoid<T>(x[i]);
  precise_out[i] = glm_sigmoid_precise<T>(x[i]);
"""

_EAGER_SIGMOID: dict = {}


def eager_sigmoid_precise(dtype) -> Optional[bool]:
    """Whether the eager ``mx.sigmoid`` kernel for ``dtype`` evaluates exp
    precisely (MLX's precompiled kernels, built with -fno-fast-math: the
    release wheels) or like runtime-compiled kernels (source builds that
    JIT their kernels). Decided once per dtype by comparing mx.sigmoid with
    both expressions over a sweep of inputs; None when neither reproduces it
    (or when first asked inside a function transformation)."""
    if dtype in _EAGER_SIGMOID:
        return _EAGER_SIGMOID[dtype]
    try:
        grid = mx.linspace(-24.0, 24.0, 1 << 16)
        noise = mx.random.normal((1 << 16,), key=mx.random.key(7)) * 4.0
        x = mx.concatenate([grid, noise]).astype(dtype)
        kernel = mx.fast.metal_kernel(
            name="glm5_sigmoid_probe",
            input_names=["x"],
            output_names=["default_out", "precise_out"],
            header=_QMV_HEADER,
            source=_SIGMOID_PROBE_SOURCE,
        )
        default, precise = kernel(
            inputs=[x],
            template=[("T", dtype)],
            grid=(x.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape, x.shape],
            output_dtypes=[dtype, dtype],
        )
        ref = mx.sigmoid(x)
        view = {2: mx.uint16, 4: mx.uint32}[ref.dtype.size]
        same_default = mx.array_equal(ref.view(view), default.view(view)).item()
        same_precise = mx.array_equal(ref.view(view), precise.view(view)).item()
    except Exception:  # traced (mx.compile / vmap): decide on an eager call
        return None
    result = True if same_precise and not same_default else (
        False if same_default and not same_precise else None
    )
    _EAGER_SIGMOID[dtype] = result
    return result


_KDA_SOURCE = r"""
  constexpr int CK = 4;
  constexpr int NROW = CK - 1 + TOK;
  constexpr int NP = 3 * QKV;
  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int h = int(threadgroup_position_in_grid.x);
  const float q_scale = consts[0];
  const float l2_eps = consts[1];
  const float norm_eps = consts[2];
  const float lower = consts[3];
  const float inv_n = consts[4];

  threadgroup T qs[TOK][DK];
  threadgroup T ks[TOK][DK];
  threadgroup T vs[TOK][DK];
  threadgroup T as_[TOK][DK];
  threadgroup T gates[TOK][DK];
  threadgroup T ys[TOK][DK];
  threadgroup float gs[TOK][DK];
  threadgroup T betas[TOK];

  // ---- 1. short conv + SiLU -------------------------------------------------
  if (tid < uint(3 * DK)) {
    const int part = int(tid) / DK;
    const int i = int(tid) % DK;
    const int gc = part * QKV + h * DK + i;
    T win[NROW];
    for (int r = 0; r < CK - 1; r++) {
#if HAS_CONV_STATE
      win[r] = conv_state[r * NP + gc];
#else
      win[r] = static_cast<T>(0);
#endif
    }
    for (int t = 0; t < TOK; t++) {
      win[CK - 1 + t] = proj[t * PROJ_W + gc];
    }
    const device T* w = conv_w + gc * CK;
    for (int t = 0; t < TOK; t++) {
      float acc = 0.0;
      for (int j = 0; j < CK; ++j) {
        acc += static_cast<float>(win[t + j]) * w[j];
      }
      T co = static_cast<T>(acc);
      T sgm = glm_sigmoid<T>(co);
      T sv = co * sgm;
      if (part == 0) {
        qs[t][i] = sv;
      } else if (part == 1) {
        ks[t][i] = sv;
      } else {
        vs[t][i] = sv;
      }
    }
    for (int r = 0; r < CK - 1; r++) {
      conv_state_out[r * NP + gc] = win[TOK + r];
    }
  }

  // ---- 3a. low-rank gate projections (qmv_quad rows of this head) ----------
#if PRE_AG
  for (int e = int(tid); e < TOK * DK; e += 1024) {
    const int t = e / DK;
    const int i = e % DK;
    as_[t][i] = a_pre[t * QKV + h * DK + i];
    gates[t][i] = gate_pre[t * QKV + h * DK + i];
  }
#elif GATE5
  // One token, 5-bit K = 128 rows: MLX's qmv (qmv_impl): lanes 0..15 load
  // 8 values each (load_vector_safe / qdot_safe with N = 8, i.e. load_vector
  // / qdot), lanes 16..31 add nothing, one simd_sum per row.
  {
    static_assert(TOK == 1 && DK == 128, "5-bit gate rows: one token");
    constexpr int WBYTES = 128 * 5 / 8;           // 80 bytes per weight row
    constexpr int G = 128 / GS;                   // groups per row
    for (int rr = 0; rr < (2 * DK) / 32; rr++) {
      const int q = int(sg) * ((2 * DK) / 32) + rr;
      const int which = q / DK;
      const int i = q % DK;
      const int row = h * DK + i;
      float result = 0;
      if (lane < 16u) {
        const device T* xin = proj + (which == 0 ? OFF_FA : OFF_GA) + int(lane) * 8;
        float x_thread[8];
        float sum = glm_load_vector<T, 8, 5>(xin, x_thread);
        const device uint8_t* wl = (const device uint8_t*)(which == 0 ? fb_w : gb_w)
            + size_t(row) * WBYTES + int(lane) * 5;
        const device T* sl = (which == 0 ? fb_s : gb_s) + row * G + int(lane) / (GS / 8);
        const device T* bl = (which == 0 ? fb_b : gb_b) + row * G + int(lane) / (GS / 8);
        const float s = sl[0];
        const float b = bl[0];
        result += glm_qdot<8, 5>(wl, x_thread, s, b, sum);
      }
      result = simd_sum(result);
      if (lane == 0) {
        if (which == 0) {
          as_[0][i] = static_cast<T>(result);
        } else {
          gates[0][i] = static_cast<T>(result);
        }
      }
    }
  }
#else
  {
    constexpr int VPT = 32;                       // values per thread (K = 128)
    constexpr int WBYTES = 128 * BITS / 8;        // bytes per weight row
    constexpr int G = 128 / GS;                   // groups per row
    const int quad = int(tid) / 4;
    const int ql = int(tid) % 4;
    const int which = quad / DK;
    const int i = quad % DK;
    const int row = h * DK + i;
    const device uint8_t* wl = (const device uint8_t*)(which == 0 ? fb_w : gb_w)
        + size_t(row) * WBYTES + ql * (VPT * BITS / 8);
    const device T* sl = (which == 0 ? fb_s : gb_s) + row * G + ql / (GS / VPT);
    const device T* bl = (which == 0 ? fb_b : gb_b) + row * G + ql / (GS / VPT);
    const float s = sl[0];
    const float b = bl[0];
    for (int t = 0; t < TOK; t++) {
      const device T* xin = proj + t * PROJ_W + (which == 0 ? OFF_FA : OFF_GA) + ql * VPT;
      float x_thread[VPT];
      float sum = glm_load_vector<T, VPT, BITS>(xin, x_thread);
      float result = 0;
      result += glm_qdot<VPT, BITS>(wl, x_thread, s, b, sum);
      result = quad_sum(result);
      if (ql == 0) {
        if (which == 0) {
          as_[t][i] = static_cast<T>(result);
        } else {
          gates[t][i] = static_cast<T>(result);
        }
      }
    }
  }
#endif
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 2. l2norm(q) * scale, l2norm(k) --------------------------------------
  if (sg < uint(2 * TOK)) {
    const int t = int(sg) / 2;
    const bool is_q = (sg % 2) == 0;
    threadgroup T* row = is_q ? qs[t] : ks[t];
    float x[4];
    float tot = 0.0f;
    for (int e = 0; e < 4; e++) {
      x[e] = static_cast<float>(row[4 * lane + e]);
      float sq = x[e] * x[e];
      tot = sq + tot;
    }
    tot = simd_sum(tot);
    float u = tot + l2_eps;
    float r = metal::precise::rsqrt(u);
    for (int e = 0; e < 4; e++) {
      float xn = x[e] * r;
      if (is_q) {
        float xs = xn * q_scale;
        row[4 * lane + e] = static_cast<T>(xs);
      } else {
        row[4 * lane + e] = static_cast<T>(xn);
      }
    }
  }

  // ---- 3b. g = exp(lower * sigmoid(exp(A_log) * (a + dt_bias))), beta -------
  if (tid < uint(TOK * DK)) {
    const int t = int(tid) / DK;
    const int i = int(tid) % DK;
    float ea = metal::precise::exp(a_log[h]);
    float af = static_cast<float>(as_[t][i]);
    float s1 = af + dt_bias[h * DK + i];
    float s2 = ea * s1;
    float s3 = glm_sigmoid<float>(s2);
    float s4 = lower * s3;
    gs[t][i] = metal::precise::exp(s4);
  }
  if (tid < uint(TOK)) {
#if SIG_B_PRECISE
    betas[tid] = glm_sigmoid_precise<T>(proj[tid * PROJ_W + OFF_B + h]);
#else
    betas[tid] = glm_sigmoid<T>(proj[tid * PROJ_W + OFF_B + h]);
#endif
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 4. vector-gated delta rule --------------------------------------------
  for (int j = 0; j < DK / 32; j++) {
    const int dv_idx = int(sg) + 32 * j;
    constexpr int n_per_t = DK / 32;
    const int dk_idx = int(lane);
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
#if HAS_STATE
      state[i] = static_cast<float>(state_in[(size_t(h) * DK + dv_idx) * DK + s_idx]);
#else
      state[i] = 0.0f;
#endif
    }
    for (int t = 0; t < TOK; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * gs[t][s_idx];
        kv_mem += state[i] * ks[t][s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (vs[t][dv_idx] - kv_mem) * betas[t];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + ks[t][s_idx] * delta;
        out += state[i] * qs[t][s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        ys[t][dv_idx] = static_cast<T>(out);
      }
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state_out[(size_t(h) * DK + dv_idx) * DK + s_idx] = static_cast<float>(state[i]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 5. RMSNormGated ---------------------------------------------------------
  if (sg < uint(TOK)) {
    const int t = int(sg);
    float x[4];
    float tot = 0.0f;
    for (int e = 0; e < 4; e++) {
      x[e] = static_cast<float>(ys[t][4 * lane + e]);
      float sq = x[e] * x[e];
      tot = sq + tot;
    }
    tot = simd_sum(tot);
    float var = tot * inv_n;
    float u = var + norm_eps;
    float r = metal::precise::rsqrt(u);
    for (int e = 0; e < 4; e++) {
      const int c = 4 * lane + e;
      float xn = x[e] * r;
      float wf = static_cast<float>(norm_w[c]);
      float wx = wf * xn;
      float gf = static_cast<float>(gates[t][c]);
#if SIG_G_PRECISE
      float gsg = glm_sigmoid_precise<float>(gf);
#else
      float gsg = glm_sigmoid<float>(gf);
#endif
      float o = wx * gsg;
      y[t * QKV + h * DK + c] = static_cast<T>(o);
    }
  }
"""


@lru_cache(maxsize=None)
def _kda_kernel(
    has_conv_state: bool,
    has_state: bool,
    pre_ag: bool,
    sig_b_precise: bool = False,
    sig_g_precise: bool = False,
    gate5: bool = False,
):
    inputs = ["proj", "conv_w", "a_log", "dt_bias", "norm_w", "consts"]
    if has_conv_state:
        inputs.append("conv_state")
    if has_state:
        inputs.append("state_in")
    if pre_ag:
        inputs += ["a_pre", "gate_pre"]
    else:
        inputs += ["fb_w", "fb_s", "fb_b", "gb_w", "gb_s", "gb_b"]
    return mx.fast.metal_kernel(
        name=(
            f"glm5_kda_decode_c{int(has_conv_state)}_s{int(has_state)}_p{int(pre_ag)}"
            f"_b{int(sig_b_precise)}_g{int(sig_g_precise)}{'_q5' if gate5 else ''}"
        ),
        input_names=inputs,
        output_names=["y", "conv_state_out", "state_out"],
        header=_QMV_HEADER,
        source=_source(
            _KDA_SOURCE,
            HAS_CONV_STATE=int(has_conv_state),
            HAS_STATE=int(has_state),
            PRE_AG=int(pre_ag),
            SIG_B_PRECISE=int(sig_b_precise),
            SIG_G_PRECISE=int(sig_g_precise),
            GATE5=int(gate5),
        ),
    )


def kda_decode_step(
    proj: mx.array,
    conv_state: Optional[mx.array],
    conv_w: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array],
    norm_w: mx.array,
    *,
    heads: int,
    head_dim: int,
    off_fa: int,
    off_ga: int,
    off_b: int,
    q_scale: float,
    l2_eps: float,
    norm_eps: float,
    lower_bound: float,
    f_b=None,
    g_b=None,
    a_pre: Optional[mx.array] = None,
    gate_pre: Optional[mx.array] = None,
):
    """Fused KDA layer body for one sequence and T <= 8 tokens.

    ``proj`` is the fused q|k|v|f_a|g_a|b projection [1, T, W] (bf16/fp16).
    The forget/output gate projections are either the affine quantized
    ``f_b``/``g_b`` layers (K == 128, 4- or 8-bit: MLX's qmv_quad path) or
    precomputed ``a_pre``/``gate_pre`` [1, T, H * Dk]. Returns
    ``(y [1, T, H * Dk], conv_state [1, 3, 3 * H * Dk], state [1, H, Dk, Dk])``
    or None when the shapes are not covered.
    """
    if "kda" in DISABLED:
        return None
    if proj.ndim != 3 or proj.shape[0] != 1 or proj.dtype not in (mx.bfloat16, mx.float16):
        return None
    _, T, width = proj.shape
    qkv = heads * head_dim
    if not 1 <= T <= 8 or head_dim != 128 or heads < 1:
        return None
    if conv_w.shape != (3 * qkv, 4, 1) or conv_w.dtype != proj.dtype:
        return None
    if norm_w.shape != (head_dim,) or norm_w.dtype != proj.dtype:
        return None
    if a_log.size != heads or a_log.dtype != mx.float32:
        return None
    if dt_bias.size != qkv or dt_bias.dtype != mx.float32:
        return None
    if conv_state is not None and (
        conv_state.shape != (1, 3, 3 * qkv) or conv_state.dtype != proj.dtype
    ):
        return None
    if state is not None and (
        state.shape != (1, heads, head_dim, head_dim) or state.dtype != mx.float32
    ):
        return None
    pre = a_pre is not None
    inputs_extra = []
    template = [("T", proj.dtype), ("TOK", T), ("DK", head_dim), ("QKV", qkv),
                ("PROJ_W", width), ("OFF_FA", off_fa), ("OFF_GA", off_ga), ("OFF_B", off_b)]
    if pre:
        if gate_pre is None or a_pre.shape != (1, T, qkv) or gate_pre.shape != (1, T, qkv):
            return None
        if a_pre.dtype != proj.dtype or gate_pre.dtype != proj.dtype:
            return None
        inputs_extra = [a_pre, gate_pre]
        template += [("BITS", 8), ("GS", 64)]
    else:
        parts = [_affine_parts(m) for m in (f_b, g_b)]
        if any(p is None for p in parts):
            return None
        (fw, fs, fbias, fbits, fgs), (gw, gs_, gbias, gbits, ggs) = parts
        if (fbits, fgs) != (gbits, ggs) or fgs not in (32, 64, 128):
            return None
        # 4/8 bits: MLX's qmv_quad (any token count). 5 bits: the one-row
        # qmv (more rows take qmv_wide, which is not replayed).
        if fbits not in (4, 8) and not (fbits == 5 and T == 1):
            return None
        if fw.shape != (qkv, 128 * fbits // 32) or gw.shape != fw.shape:
            return None
        if fs.dtype != proj.dtype or gs_.dtype != proj.dtype:
            return None
        inputs_extra = [fw, fs, fbias, gw, gs_, gbias]
        template += [("BITS", fbits), ("GS", fgs)]
    consts = mx.array(
        [q_scale, l2_eps, norm_eps, lower_bound, 1.0 / head_dim], dtype=mx.float32
    )
    inputs = [proj, conv_w, a_log.reshape(-1), dt_bias.reshape(-1), norm_w, consts]
    if conv_state is not None:
        inputs.append(conv_state)
    if state is not None:
        inputs.append(state)
    inputs += inputs_extra
    # beta = sigmoid(b) and the RMSNormGated gate are eager mx.sigmoid calls
    # in the reference; reproduce whichever exp this MLX build's kernel uses.
    sig_b = eager_sigmoid_precise(proj.dtype)
    sig_g = eager_sigmoid_precise(mx.float32)
    if sig_b is None or sig_g is None:
        return None
    gate5 = not pre and dict(template)["BITS"] == 5
    kernel = _kda_kernel(conv_state is not None, state is not None, pre, sig_b, sig_g, gate5)
    STATS["kda"] += 1
    if gate5:
        STATS["kda_gate5"] += 1
    y, conv_out, state_out = kernel(
        inputs=inputs,
        template=template,
        grid=(1024 * heads, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, T, qkv), (1, 3, 3 * qkv), (1, heads, head_dim, head_dim)],
        output_dtypes=[proj.dtype, proj.dtype, mx.float32],
    )
    return y, conv_out, state_out


# ---------------------------------------------------------------------------
# MoE router (one token): logits GEMV + sigmoid + bias, then top-k selection
# ---------------------------------------------------------------------------
#
# The logits reproduce MLX's non-transposed fp32 gemv for x @ W.T with
# 16 <= E < 4096 outputs and K < 16 * E (BM=4, BN=1, SM=1, SN=32, TM=4, TN=4):
# each simdgroup owns 4 rows, every lane accumulates 4 contiguous products
# per 128-wide K block, then a shuffle-down ladder. The epilogue applies the
# Sigmoid functor and the correction bias as separate roundings (they are
# separate kernels in the reference). The select kernel reproduces
# argpartition (a stable ascending merge sort of -(sigmoid + bias), i.e.
# descending scores with ties to the lower expert index), the gathered
# sigmoid scores, their sequential sum (row_reduce_small), the division and
# the routed scaling factor.
_ROUTER_LOGITS_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int tok = int(threadgroup_position_in_grid.y);
  // One row per simdgroup: MLX's gemv gives each thread TM = 4 rows, but
  // every row's per-lane products and shuffle ladder are independent of TM.
  constexpr int RPS = ROWS_PER_SIMD;
  const int out_row = (int(threadgroup_position_in_grid.x) * 4 + int(sg)) * RPS;
  if (out_row >= E) {
    return;
  }
  const device float* mat = w + size_t(out_row) * K;
  const device T* xv = x + size_t(tok) * K;
  float result[RPS];
  for (int tm = 0; tm < RPS; tm++) {
    result[tm] = 0.0f;
  }
  int bn = int(lane) * 4;
  for (int i = 0; i < K / 128; ++i) {
    float v_coeff[4];
    for (int tn = 0; tn < 4; tn++) {
      v_coeff[tn] = static_cast<float>(xv[bn + tn]);
    }
    int mat_offset = 0;
    for (int tm = 0; tm < RPS; tm++) {
      float inter[4];
      for (int tn = 0; tn < 4; tn++) {
        inter[tn] = mat[mat_offset + bn + tn];
      }
      for (int tn = 0; tn < 4; tn++) {
        result[tm] += inter[tn] * v_coeff[tn];
      }
      mat_offset += K;
    }
    bn += 128;
  }
  for (int tm = 0; tm < RPS; tm++) {
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
      result[tm] += simd_shuffle_down(result[tm], sn);
    }
  }
  if (lane == 0) {
    for (int tm = 0; tm < RPS; tm++) {
      const int e = out_row + tm;
#if SIG_PRECISE
      float sgm = glm_sigmoid_precise<float>(result[tm]);
#else
      float sgm = glm_sigmoid<float>(result[tm]);
#endif
      float biased = sgm + bias[e];
      sig[size_t(tok) * E + e] = sgm;
      biased_out[size_t(tok) * E + e] = biased;
    }
  }
"""

_ROUTER_SELECT_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const int tok = int(threadgroup_position_in_grid.x);
  constexpr int PER = (E + 31) / 32;
  const device float* bz = biased + size_t(tok) * E;
  const device float* sz = sig + size_t(tok) * E;
  float vals[PER];
  bool taken[PER];
  for (int j = 0; j < PER; j++) {
    const int e = j * 32 + int(lane);
    vals[j] = e < E ? bz[e] : -INFINITY;
    taken[j] = e >= E;
  }
  int picked[TOPK];
  for (int r = 0; r < TOPK; r++) {
    // Best remaining candidate of this lane: highest value, lowest index.
    float best = -INFINITY;
    int best_e = 0x7fffffff;
    for (int j = 0; j < PER; j++) {
      const int e = j * 32 + int(lane);
      if (!taken[j] && !isnan(vals[j]) &&
          (best_e == 0x7fffffff || vals[j] > best || (vals[j] == best && e < best_e))) {
        best = vals[j];
        best_e = e;
      }
    }
    for (ushort off = 16; off >= 1; off >>= 1) {
      float ob = simd_shuffle_xor(best, off);
      int oe = simd_shuffle_xor(best_e, off);
      const bool other_better = oe != 0x7fffffff &&
          (best_e == 0x7fffffff || ob > best || (ob == best && oe < best_e));
      if (other_better) {
        best = ob;
        best_e = oe;
      }
    }
    if (best_e == 0x7fffffff) {
      // Only NaNs remain (uniform branch): argpartition's sort places them
      // after every number, lowest index first.
      for (int j = 0; j < PER; j++) {
        const int e = j * 32 + int(lane);
        if (!taken[j] && e < best_e) {
          best_e = e;
        }
      }
      for (ushort off = 16; off >= 1; off >>= 1) {
        best_e = min(best_e, simd_shuffle_xor(best_e, off));
      }
    }
    picked[r] = best_e;
    if ((best_e % 32) == int(lane)) {
      taken[best_e / 32] = true;
    }
  }
  if (lane == 0) {
    float total = 0.0f;
    float gathered[TOPK];
    for (int r = 0; r < TOPK; r++) {
      gathered[r] = sz[picked[r]];
      total = gathered[r] + total;
    }
    for (int r = 0; r < TOPK; r++) {
      float q = NORM ? gathered[r] / total : gathered[r];
      float s = q * scaling[0];
      indices[tok * TOPK + r] = uint(picked[r]);
      scores[tok * TOPK + r] = s;
    }
  }
"""


@lru_cache(maxsize=None)
def _router_logits_kernel(sig_precise: bool = False):
    return mx.fast.metal_kernel(
        name="glm5_router_logits_sigmoid" + ("_precise" if sig_precise else ""),
        input_names=["x", "w", "bias"],
        output_names=["sig", "biased_out"],
        header=_QMV_HEADER,
        source=_source(_ROUTER_LOGITS_SOURCE, SIG_PRECISE=int(sig_precise)),
    )


def _router_sigmoid_precise() -> Optional[bool]:
    """The reference router (group_expert_select) takes the sigmoid of its
    fp32 logits with MLX's eager Sigmoid kernel, whose exp is the precise
    one on release wheels (see eager_sigmoid_precise); None while undecided."""
    return eager_sigmoid_precise(mx.float32)


@lru_cache(maxsize=None)
def _router_select_kernel():
    return mx.fast.metal_kernel(
        name="glm5_router_select",
        input_names=["sig", "biased", "scaling"],
        output_names=["indices", "scores"],
        source=_ROUTER_SELECT_SOURCE,
    )


def moe_router_logits(x: mx.array, weight: mx.array, bias: mx.array):
    """The router logits kernel alone: ``(sigmoid(x @ W.T), sigmoid + bias)``
    [T, E] fp32 with ``moe_router``'s arithmetic, or None when not covered."""
    if "router" in DISABLED or "router_select_fused" in DISABLED:
        return None
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        return None
    T, K = x.shape
    E = weight.shape[0]
    if weight.shape[1] != K or bias.shape[0] != E:
        return None
    if weight.dtype != mx.float32 or bias.dtype != mx.float32:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return None
    if E < 16 or E >= 4096 or K >= 16 * E or K <= 64 or K % 128 or E % 16 or E > 1024:
        return None
    sig, biased = _router_logits_kernel()(
        inputs=[x, weight, bias],
        template=[("T", x.dtype), ("K", K), ("E", E), ("ROWS_PER_SIMD", 1)],
        grid=(128 * (E // 4), T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, E), (T, E)],
        output_dtypes=[mx.float32, mx.float32],
    )
    STATS["router"] += 1
    return sig, biased


def moe_router(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    top_k: int,
    scaling: float,
    norm_topk_prob: bool,
):
    """``group_expert_select(x.astype(f32) @ weight.T, bias, ...)`` for n_group == 1.

    ``x`` [T, K] (one-token rows; bf16/fp16/fp32), ``weight`` [E, K] fp32,
    ``bias`` [E] fp32. Returns ``(indices uint32 [T, top_k], scores fp32
    [T, top_k])`` bit-identical to the reference for rows that the reference
    computes with the one-token gemv, or None when not covered.
    """
    if "router" in DISABLED:
        return None
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        return None
    T, K = x.shape
    E = weight.shape[0]
    if weight.shape[1] != K or bias.shape[0] != E:
        return None
    if weight.dtype != mx.float32 or bias.dtype != mx.float32:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return None
    # Config of the reference gemv (see gemv_axbpy): bm=4, bn=1 needs
    # E < 4096 and K < 16 * E; full 128-wide blocks and whole 16-row tiles.
    if E < 16 or E >= 4096 or K >= 16 * E or K <= 64 or K % 128 or E % 16:
        return None
    if not 1 <= top_k <= min(32, E) or E > 1024:
        return None
    precise = _router_sigmoid_precise()
    if precise is None:
        return None
    rows_per_simd = 1
    sig, biased = _router_logits_kernel(precise)(
        inputs=[x, weight, bias],
        template=[("T", x.dtype), ("K", K), ("E", E), ("ROWS_PER_SIMD", rows_per_simd)],
        grid=(128 * (E // (4 * rows_per_simd)), T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(T, E), (T, E)],
        output_dtypes=[mx.float32, mx.float32],
    )
    threads = 32  # one simdgroup per token (eight selection rounds)
    indices, scores = _router_select_kernel()(
        inputs=[sig, biased, mx.array([scaling], dtype=mx.float32)],
        template=[("E", E), ("TOPK", top_k), ("NORM", int(bool(norm_topk_prob) and top_k > 1))],
        grid=(threads * T, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(T, top_k), (T, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    STATS["router"] += 1
    return indices, scores


# ---------------------------------------------------------------------------
# Hyper-connection expand for one token (L == 1)
# ---------------------------------------------------------------------------
#
# The one-token reference (``hyper_connection._hc_expand_op``) computes
#   bf16(post * float(x) + comb^T @ float(residual))
# where the [HC, HC] x [HC, D] fp32 product runs on MLX's NAX steel GEMM,
# i.e. an MPP matmul2d with relaxed precision. This kernel issues the same
# 16x32x16 relaxed matmul2d on the same zero-padded fragments (bit-identical
# to mx.matmul for this shape) and applies the compiled epilogue with its
# separate multiply/add roundings: one dispatch instead of cast + GEMM +
# elementwise.
_NAX_HEADER = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

_HC_EXPAND1_SOURCE = r"""
  const ushort lane = thread_index_in_simdgroup;
  const int tile = int(threadgroup_position_in_grid.x) * SIMDS + int(simdgroup_index_in_threadgroup);
  if (tile * 32 >= D) {
    return;
  }
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<float, float, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<float, float, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      float>();
  for (short i = 0; i < 8; i++) {
    const short r = fm + (i >> 2) * 8;
    const short c = fn + (i & 3);
    // A = comb^T (rows: output stream, cols: source stream), zero padded.
    ct_a[i] = (r < HC && c < HC) ? comb[c * HC + r] : 0.0f;
    ct_b[i] = (r < HC) ? static_cast<float>(residual[r * D + tile * 32 + c]) : 0.0f;
    ct_b[8 + i] = (r < HC) ? static_cast<float>(residual[r * D + tile * 32 + 16 + c]) : 0.0f;
    ct_c[i] = 0.0f;
    ct_c[8 + i] = 0.0f;
  }
  op.run(ct_a, ct_b, ct_c);
  for (short i = 0; i < 8; i++) {
    const short r = fm + (i >> 2) * 8;
    const short c = fn + (i & 3);
    if (r < HC) {
      for (short hh = 0; hh < 2; hh++) {
        const int col = tile * 32 + hh * 16 + c;
        const float mm = ct_c[hh * 8 + i];
        // Separate roundings, as in the compiled reference epilogue (the
        // MPP headers enable FP contraction for the whole kernel).
        volatile float prod = post[r] * static_cast<float>(x[col]);
        float sum = prod + mm;
        out[r * D + col] = static_cast<T>(sum);
      }
    }
  }
"""


def _atoi(text: str) -> int:
    """C ``atoi`` (how MLX parses its integer environment switches)."""
    m = re.match(r"\s*([+-]?\d+)", text)
    return int(m.group(1)) if m else 0


@lru_cache(maxsize=None)
def nax_available() -> bool:
    """Mirror of ``metal::is_nax_available()``: macOS >= 26.2 and a GPU of
    generation >= 17 (18 for phones). MLX runs fp16/bf16 GEMMs on NAX then."""
    try:
        if not mx.metal.is_available():
            return False
        arch = str(mx.device_info().get("architecture", ""))
        m = re.fullmatch(r"applegpu_g(\d+)([a-z])", arch)
        if m is None or int(m.group(1)) < (18 if m.group(2) == "p" else 17):
            return False
        parts = (platform.mac_ver()[0] or "0").split(".") + ["0"]
        return (int(parts[0]), int(parts[1])) >= (26, 2)
    except Exception:  # noqa: BLE001 - any doubt keeps the reference path
        return False


@lru_cache(maxsize=None)
def nax_relaxed_fp32_matmul() -> bool:
    """True when MLX runs fp32 GEMMs on NAX with relaxed (TF32) precision:
    NAX available and ``env::enable_tf32()`` (MLX_ENABLE_TF32, default 1).
    Kernels that reproduce the fp32 NAX product are only valid then."""
    value = os.environ.get("MLX_ENABLE_TF32")
    if value is not None and _atoi(value) == 0:
        return False
    return nax_available()


def _steel_nax_partition(M: int, N: int, K: int) -> int:
    """K partition width MLX's steel_matmul uses for a NAX GEMM (K itself
    when it does not split K): steel_matmul_axpby case 2 and
    steel_gemm_splitk_axpby_nax."""
    mn = max(M, N)
    if not (K >= 3 * mn or (mn <= 1024 and K > 2 * mn)):
        return K
    if K <= 1024:
        return K // 2
    if K <= 2048:
        return 1024
    if K <= 4096:
        return 2048
    return 4096


@lru_cache(maxsize=None)
def _hc_expand1_kernel():
    return mx.fast.metal_kernel(
        name="glm5_hc_expand_one_token",
        input_names=["x", "residual", "post", "comb"],
        output_names=["out"],
        header=_NAX_HEADER,
        source=_HC_EXPAND1_SOURCE,
    )


def hc_expand_one(
    x: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> Optional[mx.array]:
    """``_hc_expand_op(x, residual, post, comb)`` for a single token.

    ``x`` [1, 1, D], ``residual`` [1, 1, HC, D] (bf16/fp16), ``post``
    [1, 1, HC] and ``comb`` [1, 1, HC, HC] fp32. Returns [1, 1, HC, D] or
    None when not covered.
    """
    if "hc_expand" in DISABLED:
        return None
    if x.ndim != 3 or x.shape[:2] != (1, 1) or residual.ndim != 4:
        return None
    D = x.shape[2]
    hc = residual.shape[2]
    if residual.shape != (1, 1, hc, D) or not 1 <= hc <= 16 or D % 32:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16) or residual.dtype != x.dtype:
        return None
    if post.shape != (1, 1, hc) or comb.shape != (1, 1, hc, hc):
        return None
    if post.dtype != mx.float32 or comb.dtype != mx.float32:
        return None
    if not nax_relaxed_fp32_matmul():
        return None
    simds = 8
    tiles = D // 32
    STATS["hc_expand"] += 1
    return _hc_expand1_kernel()(
        inputs=[x, residual, post, comb],
        template=[("T", x.dtype), ("HC", hc), ("D", D), ("SIMDS", simds)],
        grid=(32 * simds * ((tiles + simds - 1) // simds), 1, 1),
        threadgroup=(32 * simds, 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[x.dtype],
    )[0]


# Multi-row (verify block) router logits. The reference x @ W.T for
# 2 <= L <= 8 rows runs MLX's NAX split-K GEMM: relaxed-precision 16x32x16
# matmul2d ops along each K partition (2048 wide for 2048 < K <= 4096),
# then the partitions summed in order from 0. One simdgroup per (32-expert
# tile, K partition) issues the same op sequence along its partition, and
# the partitions are summed in the same order before the sigmoid/bias
# epilogue.
_ROUTER_NAX_SOURCE = r"""
  const ushort lane = thread_index_in_simdgroup;
  const int p = int(simdgroup_index_in_threadgroup);
  const int n0 = int(threadgroup_position_in_grid.x) * 32;
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<float, float, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<float, float, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      float>();
  for (short i = 0; i < 16; i++) {
    ct_c[i] = 0.0f;
  }
  const int k_begin = p * PART;
  constexpr int CHUNKS = PART / 16;
  // Operands of the next G chunks are loaded ahead of the (accumulator
  // dependent) NAX chain; the chain itself is unchanged. Rows are addressed
  // by offsets (an array of device pointers has been miscompiled).
  const bool xok = fm < M;
  const uint xoff = uint(xok ? fm : 0) * uint(K) + uint(k_begin + fn);
  const uint w0 = uint(n0 + fm) * uint(K) + uint(k_begin + fn);
  const uint w1 = w0 + 8u * uint(K);
  const uint w2 = w0 + 16u * uint(K);
  const uint w3 = w0 + 24u * uint(K);
  float abuf[G][4];
  float bbuf[G][16];
#define GLM5_ROUTER_LOAD(kc, s)                                              \
  {                                                                          \
    const uint k0 = uint(kc) * 16u;                                          \
    _Pragma("unroll") for (short e = 0; e < 4; e++) {                        \
      abuf[s][e] = xok ? static_cast<float>(x[xoff + k0 + e]) : 0.0f;         \
      bbuf[s][e] = w[w0 + k0 + e];                                           \
      bbuf[s][4 + e] = w[w1 + k0 + e];                                       \
      bbuf[s][8 + e] = w[w2 + k0 + e];                                       \
      bbuf[s][12 + e] = w[w3 + k0 + e];                                      \
    }                                                                        \
  }
  _Pragma("unroll") for (short s = 0; s < G; s++) {
    if (s < CHUNKS) GLM5_ROUTER_LOAD(s, s);
  }
  for (int kc0 = 0; kc0 < CHUNKS; kc0 += G) {
    _Pragma("unroll") for (short s = 0; s < G; s++) {
      const int kc = kc0 + s;
      if (kc < CHUNKS) {
        _Pragma("unroll") for (short i = 0; i < 4; i++) {
          ct_a[i] = abuf[s][i];
          ct_a[4 + i] = 0.0f;  // rows fm + 8 >= 8 >= M
        }
        _Pragma("unroll") for (short i = 0; i < 16; i++) {
          ct_b[i] = bbuf[s][i];
        }
        if (kc + G < CHUNKS) GLM5_ROUTER_LOAD(kc + G, s);
        op.run(ct_a, ct_b, ct_c);
      }
    }
  }
#undef GLM5_ROUTER_LOAD
  threadgroup float parts[NPART][16][32];
  for (short i = 0; i < 8; i++) {
    const short r = fm + (i >> 2) * 8;
    const short c = fn + (i & 3);
    parts[p][r][c] = ct_c[i];
    parts[p][r][16 + c] = ct_c[8 + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (p == 0) {
    for (short i = 0; i < 8; i++) {
      const short r = fm + (i >> 2) * 8;
      const short c = fn + (i & 3);
      if (r < M) {
        for (short hh = 0; hh < 2; hh++) {
          const int e = n0 + hh * 16 + c;
          // gemm_splitk_accum: AccT out = 0; out += C_split[q] in order.
          float logit = 0;
          for (int q = 0; q < NPART; q++) {
            logit += parts[q][r][hh * 16 + c];
          }
#if SIG_PRECISE
          float sgm = glm_sigmoid_precise<float>(logit);
#else
          float sgm = glm_sigmoid<float>(logit);
#endif
          float biased = sgm + bias[e];
          sig[r * E + e] = sgm;
          biased_out[r * E + e] = biased;
        }
      }
    }
  }
"""


# Operand chunks (16 inputs each) loaded ahead of the router's NAX chain.
_ROUTER_PREFETCH = int(os.environ.get("OMLX_GLM5_ROUTER_PREFETCH", "8"))


@lru_cache(maxsize=None)
def _router_nax_kernel(sig_precise: bool = False):
    return mx.fast.metal_kernel(
        name="glm5_router_logits_nax_splitk" + ("_precise" if sig_precise else ""),
        input_names=["x", "w", "bias"],
        output_names=["sig", "biased_out"],
        header=_NAX_HEADER + _QMV_HEADER.replace("#include <metal_stdlib>\nusing namespace metal;\n", ""),
        source=_source(_ROUTER_NAX_SOURCE, SIG_PRECISE=int(sig_precise)),
    )


def moe_router_rows(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    top_k: int,
    scaling: float,
    norm_topk_prob: bool,
):
    """Router for 2..8 rows computed together (a verify block), bit-identical
    to the reference's NAX split-K logits GEMM + group_expert_select.

    Only valid (and only used) when MLX runs fp32 GEMMs on NAX with relaxed
    precision; returns None otherwise or when the shape selects another
    GEMM configuration.
    """
    if "router_rows" in DISABLED:
        return None
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        return None
    T, K = x.shape
    E = weight.shape[0]
    if not 2 <= T <= 8 or weight.shape[1] != K or bias.shape[0] != E:
        return None
    if weight.dtype != mx.float32 or bias.dtype != mx.float32:
        return None
    if x.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return None
    # steel_matmul_axpby routes this product to the NAX split-K GEMM when
    # K >= 3 * max(M, N) (or max(M, N) <= 1024 and K > 2 * max(M, N)); the
    # partition size follows steel_gemm_splitk_axpby_nax. Tile sizes only
    # group the per-element 16-wide k chain, so any aligned K is covered.
    mn = max(T, E)
    if not (K >= 3 * mn or (mn <= 1024 and K > 2 * mn)):
        return None
    part = K // 2 if K <= 1024 else 1024 if K <= 2048 else 2048 if K <= 4096 else 4096
    if part % 16 or K % part or not 1 <= K // part <= 8:
        return None
    if E % 32 or E > 1024 or not 1 <= top_k <= min(32, E):
        return None
    if not nax_relaxed_fp32_matmul():
        return None
    precise = _router_sigmoid_precise()
    if precise is None:
        return None
    npart = K // part
    sig, biased = _router_nax_kernel(precise)(
        inputs=[x, weight, bias],
        template=[("M", T), ("K", K), ("E", E), ("PART", part), ("NPART", npart),
                  ("G", _ROUTER_PREFETCH)],
        grid=(32 * npart * (E // 32), 1, 1),
        threadgroup=(32 * npart, 1, 1),
        output_shapes=[(T, E), (T, E)],
        output_dtypes=[mx.float32, mx.float32],
    )
    threads = 32  # one simdgroup per token (eight selection rounds)
    indices, scores = _router_select_kernel()(
        inputs=[sig, biased, mx.array([scaling], dtype=mx.float32)],
        template=[("E", E), ("TOPK", top_k), ("NORM", int(bool(norm_topk_prob) and top_k > 1))],
        grid=(threads * T, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(T, top_k), (T, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    key = (T, K, E, x.dtype, top_k, _ROUTER_PREFETCH)
    if not _router_rows_verified(
        key, indices, scores, x, weight, bias, top_k, scaling, norm_topk_prob
    ):
        return None
    STATS["router_rows"] += 1
    return indices, scores


_ROUTER_ROWS_CHECKED: dict = {}


def _router_rows_verified(key, indices, scores, x, weight, bias, top_k, scaling, norm) -> bool:
    """First call of each configuration: compare with group_expert_select on
    the reference logits (see ``_latent_verified``)."""
    ok = _ROUTER_ROWS_CHECKED.get(key)
    if ok is not None:
        return ok
    from omlx.patches.glm_moe_dsa.deepseek_v32 import group_expert_select

    ref_idx, ref = group_expert_select(
        x.astype(mx.float32) @ weight.T, bias, top_k, 1, 1, scaling, norm
    )
    try:
        ok = bool(
            (mx.array_equal(indices, ref_idx.astype(indices.dtype))
             & mx.array_equal(scores.view(mx.uint32), ref.view(mx.uint32))).item()
        )
    except Exception:  # traced (mx.compile / vmap): check on an eager call
        return False
    _ROUTER_ROWS_CHECKED[key] = ok
    if not ok:
        import logging

        logging.getLogger(__name__).warning(
            "GLM-5.3 fused verify router %s differs from the reference; "
            "using the reference path for it", key,
        )
    return ok


# ---------------------------------------------------------------------------
# NoPE latent attention (GLM-5.3 DSA layers, 512-wide latent keys = values)
# ---------------------------------------------------------------------------
#
# mx.fast.scaled_dot_product_attention takes its op fallback for the 512-wide
# latent head: q * bf16(scale); scores = q @ k^T (NAX GEMM, heads folded into
# M, split along K per steel_matmul's rules); where(mask, scores,
# finfo.min); precise softmax (softmax_single_row); out = p @ v (NAX GEMM,
# split-K for long contexts). The three kernels below replay that: the same
# bf16 16x32x16 matmul2d chains per K partition (two 16-wide steps per
# 32-wide k step, zero-padded tails) summed in partition order, the
# block-softmax reduction tree, and the same rounding points. Sparse decode
# reads the selected (clamped) latent rows directly instead of gathering.
_LATENT_SCORES_SOURCE = r"""
  const int N = dims[0];
  const int NKV = dims[1];
  const int PART = dims[2];
  const int R = dims[3];
  const ushort lane = thread_index_in_simdgroup;
  const int p = int(simdgroup_index_in_threadgroup);
  const int mb = int(threadgroup_position_in_grid.y);
  const int n0 = int(threadgroup_position_in_grid.x) * 32;
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      float>();
  for (short i = 0; i < 16; i++) {
    ct_c[i] = 0.0f;
  }
  const T sc = scale[0];
  // Latent rows feeding this lane's right-operand fragment elements
  // (element offsets; invalid columns read row 0 and are zeroed).
  uint koff[2][2];
  bool kvalid[2][2];
  _Pragma("unroll") for (short hh = 0; hh < 2; hh++) {
    _Pragma("unroll") for (short rr = 0; rr < 2; rr++) {
      const int n = n0 + hh * 16 + fm + rr * 8;
      kvalid[hh][rr] = n < N;
#if GATHER
      int j = n < N ? int(indices[n]) : 0;
      j = j < 0 ? 0 : (j > NKV - 1 ? NKV - 1 : j);
#else
      const int j = n < N ? n : 0;
#endif
      koff[hh][rr] = uint(j) * uint(D);
    }
  }
  const int row0 = mb * 16 + fm;
  const bool rok0 = row0 < R;
  const bool rok1 = row0 + 8 < R;
  const device T* qrow0 = q + size_t(rok0 ? row0 : 0) * D;
  const device T* qrow1 = q + size_t(rok1 ? row0 + 8 : 0) * D;
  const int k_begin = p * PART;
  const int k_end = min(k_begin + PART, D);
  const int chunks = 2 * ((k_end - k_begin + 31) / 32);
  // Operands of the next G chunks are loaded ahead of the (accumulator
  // dependent) matmul chain; the chain itself is unchanged.
  T abuf[G][8];
  T bbuf[G][16];
#define GLM5_SCORES_LOAD(kc, s)                                              \
  {                                                                          \
    const int k0 = k_begin + (kc) * 16 + fn;                                 \
    _Pragma("unroll") for (short e = 0; e < 4; e++) {                        \
      const int k = k0 + e;                                                  \
      const bool kin = k < k_end;                                            \
      const T q0 = (kin && rok0) ? qrow0[k] : static_cast<T>(0.0f);          \
      const T q1 = (kin && rok1) ? qrow1[k] : static_cast<T>(0.0f);          \
      const T s0 = sc * q0;                                                  \
      const T s1 = sc * q1;                                                  \
      abuf[s][e] = (kin && rok0) ? s0 : static_cast<T>(0.0f);                \
      abuf[s][4 + e] = (kin && rok1) ? s1 : static_cast<T>(0.0f);            \
      _Pragma("unroll") for (short rr = 0; rr < 2; rr++) {                   \
        bbuf[s][rr * 4 + e] = (kin && kvalid[0][rr])                         \
            ? keys[koff[0][rr] + uint(k)] : static_cast<T>(0.0f);            \
        bbuf[s][8 + rr * 4 + e] = (kin && kvalid[1][rr])                     \
            ? keys[koff[1][rr] + uint(k)] : static_cast<T>(0.0f);            \
      }                                                                      \
    }                                                                        \
  }
  _Pragma("unroll") for (short s = 0; s < G; s++) {
    if (s < chunks) GLM5_SCORES_LOAD(s, s);
  }
  for (int kc0 = 0; kc0 < chunks; kc0 += G) {
    _Pragma("unroll") for (short s = 0; s < G; s++) {
      const int kc = kc0 + s;
      if (kc < chunks) {
        _Pragma("unroll") for (short i = 0; i < 8; i++) {
          ct_a[i] = abuf[s][i];
        }
        _Pragma("unroll") for (short i = 0; i < 16; i++) {
          ct_b[i] = bbuf[s][i];
        }
        if (kc + G < chunks) GLM5_SCORES_LOAD(kc + G, s);
        op.run(ct_a, ct_b, ct_c);
      }
    }
  }
#undef GLM5_SCORES_LOAD
  // NPART is a template argument: a compile-time branch, not #if.
  threadgroup float parts[NPART][16][32];
  if (NPART > 1) {
    for (short i = 0; i < 8; i++) {
      const short r = fm + (i >> 2) * 8;
      const short c = fn + (i & 3);
      parts[p][r][c] = ct_c[i];
      parts[p][r][16 + c] = ct_c[8 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p != 0) {
      return;
    }
  }
  for (short i = 0; i < 8; i++) {
    const short r = fm + (i >> 2) * 8;
    const short c = fn + (i & 3);
    const int row = mb * 16 + r;
    for (short hh = 0; hh < 2; hh++) {
      const int col = n0 + hh * 16 + c;
      if (col >= N || row >= R) {
        continue;
      }
      float acc;
      if (NPART > 1) {
        // gemm_splitk_accum: AccT out = 0; out += C_split[q] in order.
        acc = 0;
        for (int q2 = 0; q2 < NPART; q2++) {
          acc += parts[q2][r][hh * 16 + c];
        }
      } else {
        acc = ct_c[hh * 8 + i];
      }
      T s = static_cast<T>(acc);
#if MASK_KIND == 1
      const bool keep = int(indices[col]) >= 0;
#elif MASK_KIND == 2
      const bool keep = mask[(row % LQ) * N + col];
#elif MASK_KIND == 3
      const bool keep = (row % LQ) + (N - LQ) >= col;
#else
      const bool keep = true;
#endif
      out[size_t(row) * N + col] = keep ? s : as_type<T>(ushort(FINFO_MIN_BITS));
    }
  }
"""

_LATENT_SOFTMAX_SOURCE = r"""
  const int N = dims[0];
  const int row = int(threadgroup_position_in_grid.x);
  const int lid = int(thread_position_in_threadgroup.x);
  const uint simd_lane_id = thread_index_in_simdgroup;
  const uint simd_group_id = simdgroup_index_in_threadgroup;
  constexpr int N_READS = 4;
  threadgroup float local_max[32];
  threadgroup float local_normalizer[32];
  float ld[N_READS];
  const device T* in = scores + size_t(row) * N + lid * N_READS;
  if (lid * N_READS + N_READS <= N) {
    for (int i = 0; i < N_READS; i++) {
      ld[i] = float(in[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      ld[i] = ((lid * N_READS + i) < N) ? float(in[i]) : Limits<float>::min;
    }
  }
  if (simd_group_id == 0) {
    local_max[simd_lane_id] = Limits<float>::min;
    local_normalizer[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maxval = Limits<float>::finite_min;
  for (int i = 0; i < N_READS; i++) {
    maxval = (maxval < ld[i]) ? ld[i] : maxval;
  }
  maxval = simd_max(maxval);
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    maxval = simd_max(local_max[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_max[0] = maxval;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];
  float normalizer = 0;
  for (int i = 0; i < N_READS; i++) {
    float exp_x = fast::exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
  }
  normalizer = simd_sum(normalizer);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_normalizer[0] = normalizer;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / local_normalizer[0];
  device T* o = probs + size_t(row) * N + lid * N_READS;
  if (lid * N_READS + N_READS <= N) {
    for (int i = 0; i < N_READS; i++) {
      o[i] = T(ld[i] * normalizer);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < N) {
        o[i] = T(ld[i] * normalizer);
      }
    }
  }
"""

_LATENT_VALUES_SOURCE = r"""
  const int N = dims[0];
  const int NKV = dims[1];
  const int PART = dims[2];
  const int R = dims[3];
  const ushort lane = thread_index_in_simdgroup;
  const int p = int(simdgroup_index_in_threadgroup);
  const int mb = int(threadgroup_position_in_grid.y);
  const int n0 = int(threadgroup_position_in_grid.x) * 32;
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
#if GATHER
  // Selected rows (clamped like the reference gather), staged once so the
  // operand loads below do not wait on an index load.
  threadgroup int sel[4096];
  for (int t = int(thread_position_in_threadgroup.x); t < N; t += 32 * NPART) {
    int j = int(indices[t]);
    sel[t] = j < 0 ? 0 : (j > NKV - 1 ? NKV - 1 : j);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#define GLM5_VROW(kb) sel[kb]
#else
#define GLM5_VROW(kb) (kb)
#endif
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      float>();
  for (short i = 0; i < 16; i++) {
    ct_c[i] = 0.0f;
  }
  const int row0 = mb * 16 + fm;
  const bool rok0 = row0 < R;
  const bool rok1 = row0 + 8 < R;
  const device T* prow0 = probs + size_t(rok0 ? row0 : 0) * N;
  const device T* prow1 = probs + size_t(rok1 ? row0 + 8 : 0) * N;
  const int k_begin = p * PART;
  const int k_end = min(k_begin + PART, N);
  const int chunks = 2 * ((k_end - k_begin + 31) / 32);
  // Operands of the next G chunks are loaded ahead of the (accumulator
  // dependent) matmul chain; the chain itself is unchanged.
  T abuf[G][8];
  T bbuf[G][16];
#define GLM5_VALUES_LOAD(kc, s)                                              \
  {                                                                          \
    const int k0 = k_begin + (kc) * 16;                                      \
    _Pragma("unroll") for (short e = 0; e < 4; e++) {                        \
      const int ka = k0 + fn + e;                                            \
      abuf[s][e] = (rok0 && ka < k_end) ? prow0[ka] : static_cast<T>(0.0f);  \
      abuf[s][4 + e] = (rok1 && ka < k_end) ? prow1[ka] : static_cast<T>(0.0f); \
    }                                                                        \
    _Pragma("unroll") for (short rr = 0; rr < 2; rr++) {                     \
      const int kb = k0 + fm + rr * 8;                                       \
      const bool ok = kb < k_end;                                            \
      const device T* vrow = vals + size_t(ok ? GLM5_VROW(kb) : 0) * D + n0 + fn; \
      _Pragma("unroll") for (short e = 0; e < 4; e++) {                      \
        bbuf[s][rr * 4 + e] = ok ? vrow[e] : static_cast<T>(0.0f);           \
        bbuf[s][8 + rr * 4 + e] = ok ? vrow[16 + e] : static_cast<T>(0.0f);  \
      }                                                                      \
    }                                                                        \
  }
  _Pragma("unroll") for (short s = 0; s < G; s++) {
    if (s < chunks) GLM5_VALUES_LOAD(s, s);
  }
  for (int kc0 = 0; kc0 < chunks; kc0 += G) {
    _Pragma("unroll") for (short s = 0; s < G; s++) {
      const int kc = kc0 + s;
      if (kc < chunks) {
        _Pragma("unroll") for (short i = 0; i < 8; i++) {
          ct_a[i] = abuf[s][i];
        }
        _Pragma("unroll") for (short i = 0; i < 16; i++) {
          ct_b[i] = bbuf[s][i];
        }
        if (kc + G < chunks) GLM5_VALUES_LOAD(kc + G, s);
        op.run(ct_a, ct_b, ct_c);
      }
    }
  }
#undef GLM5_VALUES_LOAD
#undef GLM5_VROW
  // NPART is a template argument: a compile-time branch, not #if.
  threadgroup float parts[NPART][16][32];
  if (NPART > 1) {
    for (short i = 0; i < 8; i++) {
      const short r = fm + (i >> 2) * 8;
      const short c = fn + (i & 3);
      parts[p][r][c] = ct_c[i];
      parts[p][r][16 + c] = ct_c[8 + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p != 0) {
      return;
    }
  }
  for (short i = 0; i < 8; i++) {
    const short r = fm + (i >> 2) * 8;
    const short c = fn + (i & 3);
    for (short hh = 0; hh < 2; hh++) {
      float acc;
      if (NPART > 1) {
        // gemm_splitk_accum: AccT out = 0; out += C_split[q] in order.
        acc = 0;
        for (int q2 = 0; q2 < NPART; q2++) {
          acc += parts[q2][r][hh * 16 + c];
        }
      } else {
        acc = ct_c[hh * 8 + i];
      }
      if (mb * 16 + r < R) {
        out[size_t(mb * 16 + r) * D + n0 + hh * 16 + c] = static_cast<T>(acc);
      }
    }
  }
"""


@lru_cache(maxsize=None)
def _latent_kernels(gather: bool, mask_kind: int):
    # Sizes that change every decode step (keys, cache length, K partition)
    # are runtime `dims`, so one compiled pipeline serves all steps.
    scores = mx.fast.metal_kernel(
        name=f"glm5_latent_scores_g{int(gather)}_m{mask_kind}",
        input_names=["q", "keys", "scale", "dims"]
        + (["indices"] if gather or mask_kind == 1 else [])
        + (["mask"] if mask_kind == 2 else []),
        output_names=["out"],
        header=_NAX_HEADER,
        source=_source(_LATENT_SCORES_SOURCE, GATHER=int(gather), MASK_KIND=mask_kind),
    )
    softmax = mx.fast.metal_kernel(
        name="glm5_latent_softmax",
        input_names=["scores", "dims"],
        output_names=["probs"],
        source=_LATENT_SOFTMAX_SOURCE,
    )
    values = mx.fast.metal_kernel(
        name=f"glm5_latent_values_g{int(gather)}",
        input_names=["probs", "vals", "dims"] + (["indices"] if gather else []),
        output_names=["out"],
        header=_NAX_HEADER,
        source=_source(_LATENT_VALUES_SOURCE, GATHER=int(gather)),
    )
    return scores, softmax, values


_FINFO_MIN_BITS = {mx.bfloat16: 0xFF7F, mx.float16: 0xFBFF}
# Operand chunks (16 keys / latent dims each) loaded ahead of the NAX chain.
_LATENT_PREFETCH = int(os.environ.get("OMLX_GLM5_LATENT_PREFETCH", "8"))


def latent_attention(
    q: mx.array,
    keys: mx.array,
    scale: float,
    *,
    indices: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    causal: bool = False,
) -> Optional[mx.array]:
    """``mx.fast.scaled_dot_product_attention(q, k, k, scale, mask)`` for
    GLM's latent MLA, bit-identical to its op fallback.

    ``q`` [1, H, L, D] (queries after embed_q), ``keys`` [1, 1, NKV, D] (the
    latent cache, used as keys and values). ``indices`` [W] selects (L == 1
    sparse decode: clamped rows, masked where negative); otherwise ``mask``
    ([L, NKV] bool) or ``causal`` masks the dense keys. Returns [1, H, L, D]
    or None when not covered.
    """
    if "latent_attn" in DISABLED:
        return None
    if q.ndim != 4 or keys.ndim != 4 or q.shape[0] != 1 or keys.shape[:2] != (1, 1):
        return None
    _, H, L, D = q.shape
    NKV = keys.shape[2]
    if keys.shape[3] != D or keys.dtype != q.dtype or q.dtype not in _FINFO_MIN_BITS:
        return None
    if D % 32 or not 1 <= L <= 8 or not nax_available():
        return None
    gather = indices is not None
    if gather:
        if L != 1 or mask is not None or causal or indices.ndim != 1:
            return None
        N = indices.shape[0]
        mask_kind = 1
    else:
        N = NKV
        if mask is not None:
            if mask.dtype != mx.bool_ or mask.size != L * N or causal:
                return None
            mask_kind = 2
        else:
            mask_kind = 3 if causal else 0
    if N < 2 or N > 4096:  # softmax_single_row range; one key is a gemv
        return None
    R = H * L
    if R < 16:
        # The reference scores q @ k.T then take MLX's gemv_wide route
        # (2..15 rows, transposed right operand), not the NAX GEMM.
        return None
    # steel_matmul NAX routing for scores (M=R, N, K=D) and values (M=R, N=D, K=N).
    s_part = _steel_nax_partition(R, N, D)
    v_part = _steel_nax_partition(R, D, N)
    s_npart = -(-D // s_part)
    v_npart = -(-N // v_part)
    if s_npart > 8 or v_npart > 8 or (s_npart > 1 and s_part % 16) or (v_npart > 1 and v_part % 16):
        return None
    k_scores, k_softmax, k_values = _latent_kernels(gather, mask_kind)
    q2 = q.reshape(R, D)
    kv = keys.reshape(NKV, D)
    scale_arr = mx.array([scale], dtype=q.dtype)
    s_inputs = [q2, kv, scale_arr, mx.array([N, NKV, s_part, R], dtype=mx.int32)]
    if gather:
        s_inputs.append(indices.astype(mx.int32))
    if mask_kind == 2:
        s_inputs.append(mask.reshape(L, N))
    scores = k_scores(
        inputs=s_inputs,
        template=[("T", q.dtype), ("D", D), ("LQ", L), ("NPART", s_npart),
                  ("FINFO_MIN_BITS", _FINFO_MIN_BITS[q.dtype]), ("G", _LATENT_PREFETCH)],
        grid=(32 * s_npart * (-(-N // 32)), -(-R // 16), 1),
        threadgroup=(32 * s_npart, 1, 1),
        output_shapes=[(R, N)],
        output_dtypes=[q.dtype],
    )[0]
    threads = ((-(-N // 4)) + 31) // 32 * 32
    probs = k_softmax(
        inputs=[scores, mx.array([N], dtype=mx.int32)],
        template=[("T", q.dtype)],
        grid=(threads * R, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(R, N)],
        output_dtypes=[q.dtype],
    )[0]
    v_inputs = [probs, kv, mx.array([N, NKV, v_part, R], dtype=mx.int32)]
    v_inputs += [indices.astype(mx.int32)] if gather else []
    out = k_values(
        inputs=v_inputs,
        template=[("T", q.dtype), ("D", D), ("NPART", v_npart), ("G", _LATENT_PREFETCH)],
        grid=(32 * v_npart * (D // 32), -(-R // 16), 1),
        threadgroup=(32 * v_npart, 1, 1),
        output_shapes=[(R, D)],
        output_dtypes=[q.dtype],
    )[0]
    out = out.reshape(1, H, L, D)
    key = (gather, mask_kind, q.dtype, D, L, s_npart, v_npart, _LATENT_PREFETCH)
    if not _latent_verified(key, out, q, keys, scale, indices, mask, causal):
        return None
    STATS["latent_attn"] += 1
    return out


# Compile-time configurations of the latent kernels checked against the
# reference on their first call (the Metal compiler has produced wrong code
# for some unrolled NAX loops, e.g. a 4-deep prefetch ring).
_LATENT_CHECKED: dict = {}


def _latent_verified(key, out, q, keys, scale, indices, mask, causal) -> bool:
    ok = _LATENT_CHECKED.get(key)
    if ok is not None:
        return ok
    if indices is not None:
        width = indices.shape[0]
        rows = mx.clip(indices, 0, keys.shape[2] - 1)
        k = mx.take_along_axis(
            keys, mx.broadcast_to(rows[None, None, :, None], (1, 1, width, keys.shape[3])), axis=2
        )
        ref_mask = (indices >= 0).reshape(1, 1, 1, width)
    else:
        k = keys
        ref_mask = "causal" if causal else (
            None if mask is None else mask.reshape(q.shape[2], keys.shape[2])
        )
    ref = mx.fast.scaled_dot_product_attention(q, k, k, scale=scale, mask=ref_mask)
    bits = mx.uint16 if out.dtype.size == 2 else mx.uint32
    try:
        ok = bool(mx.array_equal(out.view(bits), ref.view(bits)).item())
    except Exception:  # traced (mx.compile / vmap): check on an eager call
        return False
    _LATENT_CHECKED[key] = ok
    if not ok:
        import logging

        logging.getLogger(__name__).warning(
            "GLM-5.3 fused latent attention %s differs from the reference; "
            "using the reference path for it", key,
        )
    return ok


# Sparse verify blocks (2..8 tokens, each with its own selected keys): the
# reference (_gathered_attention) gathers every token's rows and runs the SDPA
# fallback with a batch of L * 64 one-row problems, i.e. MLX's batched gemv
# for the scores (4 keys per simdgroup, 16 products per lane, 16/8/4/2/1
# shuffle ladder) and gemv_t for the values (8 key groups x 4 lanes per
# simdgroup, 4 simdgroups per 64 columns, 16/8/4 ladder). These kernels keep
# each (token, head) reduction identical but load every selected key/value
# row once for a chunk of HCH heads.
_SPARSE_ROWS_SCORES_SOURCE = r"""
  const int W = dims[0];
  const int NKV = dims[1];
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int key0 = int(threadgroup_position_in_grid.x) * 16 + sg * 4;
  const int h0 = int(threadgroup_position_in_grid.y) * HCH;
  const int tok = int(threadgroup_position_in_grid.z);
  const T sc = scale[0];
  const device int* idx = indices + size_t(tok) * W;
  // This lane's 4 dims per 128-wide block (4 blocks) of the 4 keys.
  T kreg[4][16];
  bool kok[4];
  for (int tm = 0; tm < 4; tm++) {
    const int key = key0 + tm;
    kok[tm] = key < W;
    int j = kok[tm] ? idx[key] : 0;
    j = j < 0 ? 0 : (j > NKV - 1 ? NKV - 1 : j);
    const device T* krow = kv + size_t(j) * D;
    for (int i = 0; i < 4; i++) {
      for (int tn = 0; tn < 4; tn++) {
        kreg[tm][i * 4 + tn] = krow[i * 128 + int(lane) * 4 + tn];
      }
    }
  }
  for (int hh = 0; hh < HCH; hh++) {
    const int h = h0 + hh;
    const device T* qrow = q + (size_t(h) * LQ + tok) * D;   // q [H][L][D]
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4; i++) {
      float v_coeff[4];
      for (int tn = 0; tn < 4; tn++) {
        const T qv = qrow[i * 128 + int(lane) * 4 + tn];
        const T qs = sc * qv;
        v_coeff[tn] = static_cast<float>(qs);
      }
      for (int tm = 0; tm < 4; tm++) {
        for (int tn = 0; tn < 4; tn++) {
          result[tm] += kreg[tm][i * 4 + tn] * v_coeff[tn];
        }
      }
    }
    for (int tm = 0; tm < 4; tm++) {
      for (ushort sn = 16; sn >= 1; sn >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], sn);
      }
    }
    if (lane == 0) {
      for (int tm = 0; tm < 4; tm++) {
        if (kok[tm]) {
          const int key = key0 + tm;
          const T s = static_cast<T>(result[tm]);
          out[(size_t(tok) * H + h) * W + key] =
              idx[key] >= 0 ? s : as_type<T>(ushort(FINFO_MIN_BITS));
        }
      }
    }
  }
"""

_SPARSE_ROWS_VALUES_SOURCE = r"""
  const int W = dims[0];
  const int NKV = dims[1];
  const uint lane = thread_index_in_simdgroup;
  const int sgN = int(simdgroup_index_in_threadgroup);   // BN = 4
  const int thrM = int(lane) / 4;
  const int thrN = int(lane) % 4;
  const int out_col = int(threadgroup_position_in_grid.x) * 64 + (sgN * 4 + thrN) * 4;
  const int h0 = int(threadgroup_position_in_grid.y) * HCH;
  const int tok = int(threadgroup_position_in_grid.z);
  const device int* idx = indices + size_t(tok) * W;
  float result[HCH][4];
  for (int hh = 0; hh < HCH; hh++) {
    for (int tn = 0; tn < 4; tn++) {
      result[hh][tn] = 0.0f;
    }
  }
  const int n_iter = W / 32;
  int bm = thrM * 4;
  for (int it = 0; it <= n_iter; it++) {
    const bool leftover = it == n_iter;
    T inter[4][4];
    bool ok[4];
    for (int tm = 0; tm < 4; tm++) {
      ok[tm] = !leftover || (bm + tm < W);
      int j = ok[tm] ? idx[bm + tm] : 0;
      j = j < 0 ? 0 : (j > NKV - 1 ? NKV - 1 : j);
      const device T* vrow = kv + size_t(j) * D + out_col;
      for (int tn = 0; tn < 4; tn++) {
        inter[tm][tn] = ok[tm] ? vrow[tn] : static_cast<T>(0.0f);
      }
    }
    for (int hh = 0; hh < HCH; hh++) {
      const device T* prow = probs + (size_t(tok) * H + h0 + hh) * W;
      for (int tm = 0; tm < 4; tm++) {
        if (!ok[tm]) {
          break;
        }
        const float vc = static_cast<float>(prow[bm + tm]);
        for (int tn = 0; tn < 4; tn++) {
          result[hh][tn] += vc * inter[tm][tn];
        }
      }
    }
    bm += 32;
  }
  for (int hh = 0; hh < HCH; hh++) {
    for (int tn = 0; tn < 4; tn++) {
      for (ushort sm = 4; sm >= 1; sm >>= 1) {
        result[hh][tn] += simd_shuffle_down(result[hh][tn], 4 * sm);
      }
    }
    if (thrM == 0) {
      device T* o = out + (size_t(tok) * H + h0 + hh) * D + out_col;
      for (int tn = 0; tn < 4; tn++) {
        o[tn] = static_cast<T>(result[hh][tn]);
      }
    }
  }
"""


@lru_cache(maxsize=None)
def _sparse_rows_kernels():
    scores = mx.fast.metal_kernel(
        name="glm5_latent_sparse_rows_scores",
        input_names=["q", "kv", "indices", "scale", "dims"],
        output_names=["out"],
        source=_SPARSE_ROWS_SCORES_SOURCE,
    )
    values = mx.fast.metal_kernel(
        name="glm5_latent_sparse_rows_values",
        input_names=["probs", "kv", "indices", "dims"],
        output_names=["out"],
        source=_SPARSE_ROWS_VALUES_SOURCE,
    )
    return scores, values


def latent_attention_sparse_rows(
    q: mx.array, keys: mx.array, indices: mx.array, scale: float, *, heads_per_tg: int = 8
) -> Optional[mx.array]:
    """``_gathered_attention``'s SDPA for 2..8 tokens with per-token selected
    keys, bit-identical: ``q`` [1, H, L, D] (embed_q queries), ``keys``
    [1, 1, NKV, D] latent cache, ``indices`` [L, W] (negative = masked,
    rows clamped). Returns [1, H, L, D] or None."""
    if "latent_sparse_rows" in DISABLED:
        return None
    if q.ndim != 4 or q.shape[0] != 1 or keys.ndim != 4 or keys.shape[:2] != (1, 1):
        return None
    _, H, L, D = q.shape
    NKV = keys.shape[2]
    if keys.shape[3] != D or keys.dtype != q.dtype or q.dtype not in _FINFO_MIN_BITS:
        return None
    if indices.ndim != 2 or indices.shape[0] != L or not 2 <= L <= 8:
        return None
    W = indices.shape[1]
    # gemv (scores): out W >= 4 rows, K = D < 16 * W, 128-wide blocks;
    # gemv_t (values): in W < 8192, out D in [512, 2048) -> bn = 4, 64 cols.
    if D != 512 or W < 16 or W > 4096 or H % heads_per_tg:
        return None
    k_scores, k_values = _sparse_rows_kernels()
    q_rows = q.reshape(H * L, D)
    kv = keys.reshape(NKV, D)
    idx = indices.astype(mx.int32)
    dims = mx.array([W, NKV], dtype=mx.int32)
    fmin = _FINFO_MIN_BITS[q.dtype]
    scores = k_scores(
        inputs=[q_rows, kv, idx, mx.array([scale], dtype=q.dtype), dims],
        template=[("T", q.dtype), ("D", D), ("H", H), ("LQ", L), ("HCH", heads_per_tg),
                  ("FINFO_MIN_BITS", fmin)],
        grid=(128 * (-(-W // 16)), H // heads_per_tg, L),
        threadgroup=(128, 1, 1),
        output_shapes=[(L * H, W)],
        output_dtypes=[q.dtype],
    )[0]
    _, k_softmax, _ = _latent_kernels(False, 0)
    threads = ((-(-W // 4)) + 31) // 32 * 32
    probs = k_softmax(
        inputs=[scores, mx.array([W], dtype=mx.int32)],
        template=[("T", q.dtype)],
        grid=(threads * L * H, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(L * H, W)],
        output_dtypes=[q.dtype],
    )[0]
    out = k_values(
        inputs=[probs, kv, idx, dims],
        template=[("T", q.dtype), ("D", D), ("H", H), ("HCH", heads_per_tg)],
        grid=(128 * (D // 64), H // heads_per_tg, L),
        threadgroup=(128, 1, 1),
        output_shapes=[(L * H, D)],
        output_dtypes=[q.dtype],
    )[0]
    STATS["latent_sparse_rows"] += 1
    return out.reshape(1, L, H, D).transpose(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Several quantized projections of one input in a single dispatch
# ---------------------------------------------------------------------------
#
# One-token rows reproduce MLX's qmv_fast (2 simdgroups x 4 rows per 8-row
# tile, qdot per 512/256-wide K block, simd_sum); 2..8-token rows reproduce
# qmv_wide (8 lanes per row, per-group dequantize in 8-value sub-chunks, 4/2/1
# ladder) -- the kernels the separate projections run on. Each part keeps
# its own contiguous output, so nothing downstream changes.
_MULTI_QMV_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.x) * 8;
  constexpr int WB = K * BITS / 8;
  constexpr int G = K / GS;
  const device uint8_t* w;
  const device T* sc;
  const device T* bi;
  device T* y;
  int local;
  int n_rows;
  if (row0 < N0) {
    w = (const device uint8_t*)w0; sc = s0; bi = b0; y = y0; local = row0; n_rows = N0;
  }
#if NP > 1
  else if (row0 < N0 + N1) {
    w = (const device uint8_t*)w1; sc = s1; bi = b1; y = y1; local = row0 - N0; n_rows = N1;
  }
#endif
#if NP > 2
  else if (row0 < N0 + N1 + N2) {
    w = (const device uint8_t*)w2; sc = s2; bi = b2; y = y2; local = row0 - N0 - N1; n_rows = N2;
  }
#endif
#if NP > 3
  else {
    w = (const device uint8_t*)w3; sc = s3; bi = b3; y = y3; local = row0 - N0 - N1 - N2; n_rows = N3;
  }
#endif
#if TOK == 1
  const int r0 = local + sg * 4;
  float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  glm_qmv_rows<T, K, GS, BITS, 4>(
      w + size_t(r0) * WB, sc + r0 * G, bi + r0 * G, x, lane, result);
  for (int r = 0; r < 4; r++) {
    float v = simd_sum(result[r]);
    if (lane == 0) {
      y[r0 + r] = static_cast<T>(v);
    }
  }
#else
  const int k_lane = int(lane) % 8;
  const int row = local + sg * 4 + int(lane) / 8;
  float res[TOK];
  for (int v = 0; v < TOK; v++) {
    res[v] = 0.0f;
  }
  glm_qmv_wide_row<T, K, GS, BITS, TOK>(
      w + size_t(row) * WB, sc + row * G, bi + row * G, x, TOK, k_lane, res);
  for (int v = 0; v < TOK; v++) {
    res[v] += simd_shuffle_down(res[v], 4);
    res[v] += simd_shuffle_down(res[v], 2);
    res[v] += simd_shuffle_down(res[v], 1);
  }
  if (k_lane == 0) {
    for (int v = 0; v < TOK; v++) {
      y[size_t(v) * n_rows + row] = static_cast<T>(res[v]);
    }
  }
#endif
"""


@lru_cache(maxsize=None)
def _multi_qmv_kernel(n_parts: int, tokens: int):
    inputs = ["x"]
    for i in range(n_parts):
        inputs += [f"w{i}", f"s{i}", f"b{i}"]
    return mx.fast.metal_kernel(
        name=f"glm5_multi_qmv_p{n_parts}_t{tokens}",
        input_names=inputs,
        output_names=[f"y{i}" for i in range(n_parts)],
        header=_QMV_HEADER,
        source=_source(_MULTI_QMV_SOURCE, NP=n_parts, TOK=tokens),
    )


def multi_qmv(x: mx.array, layers) -> Optional[list]:
    """``[linear(x) for linear in layers]`` for 1..8 rows in one dispatch.

    ``x`` [T, K]; ``layers`` 1..4 affine quantized linears (no bias) with the
    same bits/group size, K inputs and output rows divisible by 8. Returns
    the [T, N_i] outputs or None when not covered.
    """
    if "multi_qmv" in DISABLED:
        return None
    if x.ndim != 2 or not 1 <= len(layers) <= 4 or x.dtype not in (mx.bfloat16, mx.float16):
        return None
    T, K = x.shape
    if not 1 <= T <= 8:
        return None
    parts = [_affine_parts(m) for m in layers]
    if any(p is None for p in parts):
        return None
    bits, gs = parts[0][3], parts[0][4]
    if any((p[3], p[4]) != (bits, gs) for p in parts):
        return None
    rows = []
    for w, s, b, _, _ in parts:
        if w.ndim != 2 or w.shape[1] * 32 // bits != K or s.dtype != x.dtype:
            return None
        rows.append(w.shape[0])
    if any(n % 8 for n in rows):
        return None
    if T == 1:
        # qmv_fast only (MLX routes the one-row product there when aligned).
        if not all(_qmv_fast_ok(bits, gs, n, K) for n in rows):
            return None
    elif bits not in (4, 5, 6, 8) or gs % 8 or K % gs or K in (64, 128):
        return None
    inputs = [x]
    for w, s, b, _, _ in parts:
        inputs += [w, s, b]
    template = [("T", x.dtype), ("K", K), ("BITS", bits), ("GS", gs)]
    template += [(f"N{i}", rows[i] if i < len(rows) else 0) for i in range(4)]
    STATS["multi_qmv"] += 1
    return list(_multi_qmv_kernel(len(layers), T)(
        inputs=inputs,
        template=template,
        grid=(64 * (sum(rows) // 8), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(T, n) for n in rows],
        output_dtypes=[x.dtype] * len(rows),
    ))
