# SPDX-License-Identifier: Apache-2.0
"""Tensor-unit (NAX) QSA main attention over per-tile unions of selected blocks.

Qwen4-Exp QSA lets every query attend its own top-512 four-token key blocks
plus the zero-to-three token causal tail. The direct native kernel runs one
(query, KV head) per threadgroup: every query re-streams its 2,051 K/V rows and
the 12 grouped heads (padded to 16) run on the classic fp32 simdgroup MMA
(~12-14 TFLOPS).

Consecutive queries select heavily overlapping blocks (real prompts: the union
of 4 consecutive queries is 1.1x-1.9x one query's 512 blocks). This path groups
``TILE`` = 4 consecutive queries, builds the ascending union of their selected
blocks (plus their tail blocks) with a per-block bitmask of which queries
selected it, and runs one tensor-unit flash-attention pass per (tile, KV head)
over the union, so all 48 query rows share every gathered K/V row. Every
(query, key) pair outside the query's own QSA key set is masked to exactly
``-inf`` before the online softmax, so each query attends exactly its key set.

Numerics: scores are bf16 x bf16 products accumulated in fp32 and the online
softmax runs in fp32, as in the native kernel. The tensor unit truncates a
float operand to tf32, so ``P @ V`` takes the fp32 probabilities as two fp16
pieces, hi = fp16(P) and lo = fp16(P - hi), whose sum is P within 2^-24
absolute (the size of P's own fp32 rounding); the softmax denominator uses the
fp32 P. Only the fp32 summation grouping differs from the native kernel.
"""

from __future__ import annotations

import functools
import os

import mlx.core as mx

# Queries per tile (12 heads each: 4 -> 48 rows = 3 NAX row groups).
TILE = 4
# How P (fp32 probabilities) enters the P @ V tensor-unit MMA (see module doc):
#   "half2" (default): fp16 hi + fp16 lo pieces, |error| <= 2^-24 per probability.
#   "bf16x3": three bf16 pieces (8+8+8 mantissa bits), fp32-exact for normal P,
#             ~1.5x slower.
PV_MODE = os.environ.get("OMLX_QWEN4_QSA_NAX_PV", "half2")
_PV_MODES = {
    "half2": (mx.float16, 2),
    "bf16x3": (mx.bfloat16, 3),
}
KEY_FRAGS = 2  # 16-key fragments per step: 32 keys = 8 union blocks
GQA = 12
HEAD_DIM = 256
COMPRESS = 4
TOPK = 512
# Union builder window: one bit per (query, block) for 32 * UNION_WORDS blocks
# (16384 blocks = 64K tokens) per pass.
UNION_WORDS = 512
UNION_THREADS = 256



def enabled() -> bool:
    """OMLX_QWEN4_QSA_NAX=0 keeps the native direct kernel."""
    return os.environ.get("OMLX_QWEN4_QSA_NAX", "1") != "0"


@functools.lru_cache(maxsize=None)
def nax_available() -> bool:
    """The kernel is built for the tensor units; other GPUs keep the native kernel."""
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:
        return False


_UNION_BITS_SOURCE = r"""
    // One threadgroup per tile of TILE consecutive queries. Per window of
    // 32 * NWORDS blocks: set bit b of plane t for every valid block query t
    // selected (plane TILE holds the tail blocks), then compact the OR of the
    // planes in ascending block order with a popcount prefix sum.
    constexpr int NT = UNION_THREADS;
    constexpr int NW = NWORDS;
    constexpr int WPT = (NW + NT - 1) / NT;  // words per thread
    constexpr int NP = TILE + 1;
    threadgroup atomic_uint planes[NP * NW];
    threadgroup int wsum[NT / 32];

    const int tile = int(threadgroup_position_in_grid.x);
    const int tid = int(thread_position_in_threadgroup.x);
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = simdgroup_index_in_threadgroup;
    const int Lq = params[0];
    const int q_offset = params[1];
    const int umax = params[2];
    const int t0 = tile * TILE;
    const int tvalid = min(TILE, Lq - t0);
    const int p_last = q_offset + t0 + tvalid - 1;
    const int hi = (p_last + 1) >> 2;  // largest tail block

    device int* out_blk = ublk + size_t(tile) * umax;
    device uint* out_bits = ubits + size_t(tile) * umax;
    int base = 0;
    for (int w0 = 0; w0 <= hi; w0 += 32 * NW) {
        for (int i = tid; i < NP * NW; i += NT) {
            atomic_store_explicit(&planes[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = tid; e < tvalid * TOPK; e += NT) {
            const int t = e / TOPK;
            const int i = e - t * TOPK;
            const int p = q_offset + t0 + t;
            const int complete = (p + 1) >> 2;
            if (i < min(TOPK, complete)) {
                const int b = sel[size_t(t0 + t) * TOPK + i] - w0;
                if (b >= 0 && b < 32 * NW) {
                    atomic_fetch_or_explicit(&planes[t * NW + (b >> 5)], 1u << (b & 31), memory_order_relaxed);
                }
            }
        }
        if (tid < tvalid) {
            const int p = q_offset + t0 + tid;
            const int complete = (p + 1) >> 2;
            const int b = complete - w0;
            if ((complete << 2) <= p && b >= 0 && b < 32 * NW) {
                atomic_fetch_or_explicit(&planes[TILE * NW + (b >> 5)], 1u << (b & 31), memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint words[WPT];
        int c = 0;
        for (int j = 0; j < WPT; ++j) {
            const int w = tid * WPT + j;
            uint u = 0u;
            if (w < NW) {
                for (int t = 0; t < NP; ++t) {
                    u |= atomic_load_explicit(&planes[t * NW + w], memory_order_relaxed);
                }
            }
            words[j] = u;
            c += popcount(u);
        }
        // Exclusive prefix over threads: simdgroup scan + one pass over sums.
        const int incl = simd_prefix_inclusive_sum(c);
        if (lane == 31) wsum[sgi] = incl;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int before = 0, total = 0;
        for (uint g = 0; g < NT / 32; ++g) {
            const int v = wsum[g];
            before += g < sgi ? v : 0;
            total += v;
        }
        int pos = base + before + incl - c;
        for (int j = 0; j < WPT; ++j) {
            uint u = words[j];
            const int w = tid * WPT + j;
            while (u != 0u) {
                const int bit = ctz(u);
                u &= u - 1u;
                uint qbits = 0u;
                for (int t = 0; t < TILE; ++t) {
                    qbits |= ((atomic_load_explicit(&planes[t * NW + w], memory_order_relaxed) >> bit) & 1u) << t;
                }
                out_blk[pos] = w0 + w * 32 + bit;
                out_bits[pos] = qbits;
                ++pos;
            }
        }
        base += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        ucount[tile] = base;
    }
"""


_ATTN_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")

// 16x32x16 simdgroup tensor-unit MMA on MLX's NAX fragment layout: each lane
// holds rows (fm, fm + 8) x columns (fn .. fn + 3) of a 16x16 fragment.
template <typename CT, typename AT, typename BT, bool TA, bool TB>
METAL_FUNC void nax_mma_n2(
    thread vec<CT, 8>& c0,
    thread vec<CT, 8>& c1,
    thread const vec<AT, 8>& a,
    thread const vec<BT, 8>& b0,
    thread const vec<BT, 8>& b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, TA, TB, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
  auto ct_b = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>, CT>();
  UNROLL for (short i = 0; i < 8; ++i) {
    ct_a[i] = a[i];
    ct_b[i] = b0[i];
    ct_b[8 + i] = b1[i];
    ct_c[i] = c0[i];
    ct_c[8 + i] = c1[i];
  }
  op.run(ct_a, ct_b, ct_c);
  UNROLL for (short i = 0; i < 8; ++i) {
    c0[i] = ct_c[i];
    c1[i] = ct_c[8 + i];
  }
}
"""

_ATTN_SOURCE = r"""
    // Grid: (tile, kv head). 2 * NG simdgroups: row group rg = sg / 2 owns 16
    // query rows (two heads x TILE queries, head-major), half dh = sg % 2 owns
    // 128 of the 256 head dims (MLX attention_nax_dsplit organization).
    constexpr int D = 256;
    constexpr int TDH = 8;      // 16-wide dim fragments per half
    constexpr int NG = (TILE * GQA) / 16;
    static_assert((TILE * GQA) % 16 == 0, "TILE * 12 rows must fill 16-row groups");

    // Score exchange between the two D halves of a row group. (A double-
    // buffered single-barrier exchange measured 5-8% slower: the second
    // barrier keeps the three row groups' K/V loads in step for L1 reuse.)
    threadgroup float xchg[NG][2][TK * 8 * 32];

    const int tile = int(threadgroup_position_in_grid.x);
    const int kvh = int(threadgroup_position_in_grid.y);
    const ushort sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const short rg = sg >> 1;
    const short dh = sg & 1;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;

    const int Lq = params[0];
    const int q_offset = params[1];
    const int umax = params[2];
    const int kL = params[3];
    const float scale2 = scale[0] * 1.44269504089f;

    // Rows are head-major (row = head * TILE + query): this lane's rows
    // rg * 16 + fm + 8 i share query t and belong to heads hh0, hh0 + 8 / TILE.
    const int t = (rg * 16 + fm) % TILE;
    const int hh0 = (rg * 16 + fm) / TILE;
    const int tq = tile * TILE + t;
    const bool row_ok = tq < Lq;
    const int p = q_offset + tq;
    const int tail_lo = ((p + 1) >> 2) << 2;
    const uint qbit = 1u << t;
    const int h0 = kvh * GQA + hh0;
    const int h1 = h0 + 8 / TILE;

    // Strides (elements): q [1, H, Lq, D], k/v [1, KVH, kL, D]; the last dim
    // is contiguous and rows are 16-byte aligned. Head dims are permuted per
    // lane: fragment pair (2j, 2j + 1) covers the 8 contiguous dims
    // 32 j + 2 fn .. + 7 (the first four in fragment 2j) of Q/K and of V/O, so
    // one 16-byte load feeds both fragments. Q and K share the permutation
    // (only the QK^T summation order changes); O undoes V's at the store.
    const int64_t sqh = q_strides[1], sql = q_strides[2];
    const int64_t skl = k_strides[2], svl = v_strides[2];
    const short fcol = 2 * fn;
    const device bfloat* kb = (const device bfloat*)k + kvh * k_strides[1] + dh * 128 + fcol;
    const device bfloat* vb = (const device bfloat*)v + kvh * v_strides[1] + dh * 128 + fcol;

    // Resident Q half: 8 fragments.
    vec<bfloat, 8> qf[TDH];
    {
      const int tqc = row_ok ? tq : 0;
      const device bfloat* q0 = (const device bfloat*)q + h0 * sqh + tqc * sql + dh * 128 + fcol;
      const device bfloat* q1 = (const device bfloat*)q + h1 * sqh + tqc * sql + dh * 128 + fcol;
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        const vec<bfloat, 8> a = row_ok ? *(const device vec<bfloat, 8>*)(q0 + 32 * jj) : vec<bfloat, 8>(0);
        const vec<bfloat, 8> b = row_ok ? *(const device vec<bfloat, 8>*)(q1 + 32 * jj) : vec<bfloat, 8>(0);
        UNROLL for (short j = 0; j < 4; ++j) {
          qf[2 * jj][j] = a[j];
          qf[2 * jj][4 + j] = b[j];
          qf[2 * jj + 1][j] = a[4 + j];
          qf[2 * jj + 1][4 + j] = b[4 + j];
        }
      }

    }

    vec<float, 8> of[TDH];
    UNROLL for (short id = 0; id < TDH; ++id) {
      of[id] = vec<float, 8>(0.0f);
    }
    float max_s[2] = {-FLT_MAX, -FLT_MAX};
    float sum_s[2] = {0.0f, 0.0f};

    const int U = ucount[tile];
    const device int* blk = ublk + size_t(tile) * umax;
    const device uint* bits = ubits + size_t(tile) * umax;
    constexpr int SB = 4 * TK;  // union blocks (4 tokens each) per step
    const int nsteps = (U + SB - 1) / SB;

    // Union metadata of a step, prefetched one step ahead: the block of each
    // key row this lane loads (keys 16 f + fm + 8 i -> slot (16 f + fm + 8 i) / 4)
    // and the block + selection bits of each S column group (slot 4 f + fn / 4).
    int nrow_b[2 * TK];
    int ncol_b[TK];
    uint ncol_bits[TK];
    auto fetch = [&](int u0) {
      UNROLL for (short r = 0; r < 2 * TK; ++r) {
        const int u = u0 + ((16 * (r >> 1) + fm + 8 * (r & 1)) >> 2);
        nrow_b[r] = u < U ? blk[u] : 0;
      }
      UNROLL for (short f = 0; f < TK; ++f) {
        const int u = u0 + 4 * f + (fn >> 2);
        ncol_b[f] = u < U ? blk[u] : -1;
        ncol_bits[f] = u < U ? bits[u] : 0u;
      }
    };
    fetch(0);

    for (int step = 0; step < nsteps; ++step) {
      const device bfloat* kr[2 * TK];
      const device bfloat* vr[2 * TK];
      UNROLL for (short r = 0; r < 2 * TK; ++r) {
        // The last tail block can extend up to three rows past kL: those keys
        // are masked, but P = 0 must never meet unwritten V (0 * NaN = NaN),
        // so they re-read the last valid row instead.
        const int row = min(nrow_b[r] * 4 + (fm & 3), kL - 1);
        kr[r] = kb + row * skl;
        vr[r] = vb + row * svl;
      }
      int col_b[TK];
      uint col_bits[TK];
      UNROLL for (short f = 0; f < TK; ++f) {
        col_b[f] = ncol_b[f];
        col_bits[f] = ncol_bits[f];
      }
      if (step + 1 < nsteps) {
        fetch((step + 1) * SB);
      }
      // S = Q K^T over this half of D.
      vec<float, 8> s[TK];
      UNROLL for (short f = 0; f < TK; ++f) {
        s[f] = vec<float, 8>(0.0f);
      }
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        UNROLL for (short f = 0; f < TK; f += 2) {
          const vec<bfloat, 8> a0 = *(const device vec<bfloat, 8>*)(kr[2 * f] + 32 * jj);
          const vec<bfloat, 8> a1 = *(const device vec<bfloat, 8>*)(kr[2 * f + 1] + 32 * jj);
          const vec<bfloat, 8> a2 = *(const device vec<bfloat, 8>*)(kr[2 * f + 2] + 32 * jj);
          const vec<bfloat, 8> a3 = *(const device vec<bfloat, 8>*)(kr[2 * f + 3] + 32 * jj);
          vec<bfloat, 8> k0e, k1e, k0o, k1o;
          UNROLL for (short j = 0; j < 4; ++j) {
            k0e[j] = a0[j];
            k0e[4 + j] = a1[j];
            k1e[j] = a2[j];
            k1e[4 + j] = a3[j];
            k0o[j] = a0[4 + j];
            k0o[4 + j] = a1[4 + j];
            k1o[j] = a2[4 + j];
            k1o[4 + j] = a3[4 + j];
          }
          nax_mma_n2<float, bfloat, bfloat, false, true>(s[f], s[f + 1], qf[2 * jj], k0e, k1e);
          nax_mma_n2<float, bfloat, bfloat, false, true>(s[f], s[f + 1], qf[2 * jj + 1], k0o, k1o);
        }
      }

      // Exchange partial sums with the other half of D.
      {
        threadgroup float* mine = xchg[rg][dh];
        const threadgroup float* peer = xchg[rg][1 - dh];
        const short o = lane * 8 * TK;
        UNROLL for (short f = 0; f < TK; ++f) {
          UNROLL for (short i = 0; i < 8; ++i) {
            mine[o + 8 * f + i] = s[f][i];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        UNROLL for (short f = 0; f < TK; ++f) {
          UNROLL for (short i = 0; i < 8; ++i) {
            s[f][i] += peer[o + 8 * f + i];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      // Scale and mask: this lane's columns fn..fn+3 of fragment f are the
      // four tokens of union slot 4 f + fn / 4 of the step.
      UNROLL for (short f = 0; f < TK; ++f) {
        const bool in_u = col_b[f] >= 0 && row_ok;
        const bool selected = in_u && (col_bits[f] & qbit) != 0u;
        UNROLL for (short j = 0; j < 4; ++j) {
          const int tok = col_b[f] * 4 + j;
          const bool ok = selected || (in_u && tok >= tail_lo && tok <= p);
          s[f][j] = ok ? s[f][j] * scale2 : -INFINITY;
          s[f][4 + j] = ok ? s[f][4 + j] * scale2 : -INFINITY;
        }
      }
      // Online softmax per row (rows fm and fm + 8 of the fragments).
      float factor[2];
      UNROLL for (short i = 0; i < 2; ++i) {
        float m = -INFINITY;
        UNROLL for (short f = 0; f < TK; ++f) {
          m = max(m, max(max(s[f][4 * i], s[f][4 * i + 1]), max(s[f][4 * i + 2], s[f][4 * i + 3])));
        }
        m = max(m, simd_shuffle_xor(m, ushort(1)));
        m = max(m, simd_shuffle_xor(m, ushort(8)));
        const float new_max = max(max_s[i], m);
        float rs = 0.0f;
        UNROLL for (short f = 0; f < TK; ++f) {
          UNROLL for (short j = 0; j < 4; ++j) {
            s[f][4 * i + j] = fast::exp2(s[f][4 * i + j] - new_max);
            rs += s[f][4 * i + j];
          }
        }
        rs += simd_shuffle_xor(rs, ushort(1));
        rs += simd_shuffle_xor(rs, ushort(8));
        factor[i] = fast::exp2(max_s[i] - new_max);
        max_s[i] = new_max;
        sum_s[i] = sum_s[i] * factor[i] + rs;
      }
      UNROLL for (short id = 0; id < TDH; ++id) {
        UNROLL for (short j = 0; j < 4; ++j) {
          of[id][j] *= factor[0];
          of[id][4 + j] *= factor[1];
        }
      }
      // O += P V over this half of D (V fragments are [16 keys x 16 dims]).
      // The tensor unit takes P as PV_TERMS pieces of type PT whose sum is P
      // (see PV_MODE in the Python module).
      vec<PT, 8> ph[TK][PV_TERMS];
      UNROLL for (short f = 0; f < TK; ++f) {
        vec<float, 8> rest = s[f];
        UNROLL for (short tt = 0; tt < PV_TERMS; ++tt) {
          UNROLL for (short i = 0; i < 8; ++i) {
            const PT piece = PT(rest[i]);
            ph[f][tt][i] = piece;
            rest[i] -= float(piece);
          }
        }
      }
      UNROLL for (short id = 0; id < TDH; id += 2) {
        UNROLL for (short f = 0; f < TK; ++f) {
          vec<bfloat, 8> v0, v1;
          const vec<bfloat, 8> ra = *(const device vec<bfloat, 8>*)(vr[f * 2] + 16 * id);
          const vec<bfloat, 8> rb = *(const device vec<bfloat, 8>*)(vr[f * 2 + 1] + 16 * id);
          UNROLL for (short j = 0; j < 4; ++j) {
            v0[j] = ra[j];
            v0[4 + j] = rb[j];
            v1[j] = ra[4 + j];
            v1[4 + j] = rb[4 + j];
          }

          UNROLL for (short tt = 0; tt < PV_TERMS; ++tt) {
            nax_mma_n2<float, PT, bfloat, false, false>(
                of[id], of[id + 1], ph[f][tt], v0, v1);
          }
        }
      }
    }

    if (row_ok) {
      const float r0 = 1.0f / sum_s[0];
      const float r1 = 1.0f / sum_s[1];
      device bfloat* o0 = (device bfloat*)out + (size_t(tq) * (2 * GQA) + h0) * D + dh * 128 + fcol;
      device bfloat* o1 = (device bfloat*)out + (size_t(tq) * (2 * GQA) + h1) * D + dh * 128 + fcol;
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        vec<bfloat, 8> w0, w1;
        UNROLL for (short j = 0; j < 4; ++j) {
          w0[j] = bfloat(of[2 * jj][j] * r0);
          w0[4 + j] = bfloat(of[2 * jj + 1][j] * r0);
          w1[j] = bfloat(of[2 * jj][4 + j] * r1);
          w1[4 + j] = bfloat(of[2 * jj + 1][4 + j] * r1);
        }
        *(device vec<bfloat, 8>*)(o0 + 32 * jj) = w0;
        *(device vec<bfloat, 8>*)(o1 + 32 * jj) = w1;
      }
    }
"""


@functools.lru_cache(maxsize=None)
def _union_bits_kernel():
    return mx.fast.metal_kernel(
        name="omlx_qwen4_qsa_tile_union_bits",
        input_names=["sel", "params"],
        output_names=["ublk", "ubits", "ucount"],
        source=_UNION_BITS_SOURCE,
    )


@functools.lru_cache(maxsize=None)
def _attn_kernel():
    return mx.fast.metal_kernel(
        name="omlx_qwen4_qsa_tile_nax_attention",
        input_names=["q", "k", "v", "ublk", "ubits", "ucount", "params", "scale"],
        output_names=["out"],
        source=_ATTN_SOURCE,
        header=_ATTN_HEADER,
        ensure_row_contiguous=False,
    )


def tile_union(selected_blocks: mx.array, q_offset: int):
    """Chronological per-tile block unions: (blocks, bits, counts, umax)."""

    sel = selected_blocks.reshape(-1, TOPK)
    if sel.dtype != mx.int32:
        sel = sel.astype(mx.int32)
    lq = sel.shape[0]
    n_tiles = (lq + TILE - 1) // TILE
    last = q_offset + lq - 1
    umax = min(TILE * (TOPK + 1), (last + 1) // COMPRESS + 1)
    params = mx.array([lq, q_offset, umax], dtype=mx.int32)
    ublk, ubits, ucount = _union_bits_kernel()(
        inputs=[sel, params],
        template=[
            ("TILE", TILE),
            ("TOPK", TOPK),
            ("NWORDS", UNION_WORDS),
            ("UNION_THREADS", UNION_THREADS),
        ],
        grid=(n_tiles * UNION_THREADS, 1, 1),
        threadgroup=(UNION_THREADS, 1, 1),
        output_shapes=[(n_tiles, umax), (n_tiles, umax), (n_tiles,)],
        output_dtypes=[mx.int32, mx.uint32, mx.int32],
    )
    return ublk, ubits, ucount, umax


def sparse_gqa_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    selected_blocks: mx.array,
    *,
    q_offset: int,
    scale: float | None = None,
) -> mx.array:
    """QSA main attention for ``queries`` [1, 24, Lq, 256] at absolute rows
    ``q_offset ..``, K/V [1, 2, kL, 256], chronological ``selected_blocks``
    [1, Lq, 512]. Returns [1, Lq, 24, 256].

    Q/K/V may be strided views (cache buffers with spare capacity, sequence
    slices) but their head dim must be contiguous and 16-byte aligned, as for
    the native kernel.
    """

    lq = queries.shape[2]
    kl = keys.shape[2]
    if scale is None:
        scale = HEAD_DIM**-0.5
    ublk, ubits, ucount, umax = tile_union(selected_blocks, q_offset)
    n_tiles = (lq + TILE - 1) // TILE
    ng = (TILE * GQA) // 16
    params = mx.array([lq, q_offset, umax, kl], dtype=mx.int32)
    (out,) = _attn_kernel()(
        inputs=[
            queries,
            keys,
            values,
            ublk,
            ubits,
            ucount,
            params,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("TILE", TILE),
            ("GQA", GQA),
            ("PT", _PV_MODES[PV_MODE][0]),
            ("PV_TERMS", _PV_MODES[PV_MODE][1]),
            ("TK", KEY_FRAGS),
        ],
        grid=(n_tiles * ng * 2 * 32, keys.shape[1], 1),
        threadgroup=(ng * 2 * 32, 1, 1),
        output_shapes=[(1, lq, 2 * GQA, HEAD_DIM)],
        output_dtypes=[queries.dtype],
    )
    return out
