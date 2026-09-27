# SPDX-License-Identifier: Apache-2.0
"""Tensor-unit (NAX) sorted ``gather_qmm`` for M5 hosts, compiled at runtime.

The MoE prefill path (``SwitchGLU`` with ``sorted_indices=True``) runs every
routed expert GEMM through ``mx.gather_qmm``. On M5 GPUs mlx 0.32.2 sends
it to the ``*_gather_qmm_rhs_nax`` row-block kernel: a threadgroup per
64-row block of the sorted rows, re-running the whole K loop for every
expert present in the block with only that expert's rows active. At real
MoE prefill sizes (tens of rows per expert, a third of the blocks spanning
two experts) a large share of the tensor-unit work is masked, and the
kernel carries two defects (``K % 64 != 0`` tail and an int16 row offset
past 32768 rows) that ``m5_gather_qmm`` works around by dropping to the
slow steel path or splitting the call.

This module runs the same product on the tensor units with segmented tile
scheduling instead, as ``mx.fast.metal_kernel`` kernels on top of the NAX
tile primitives of the installed mlx (``steel/gemm/nax.h``, read from the
package's ``include`` directory):

- a one-threadgroup pre-pass cuts every expert's run of sorted rows into
  (row_start, expert, rows) tiles of at most 64 rows, so partial tiles only
  occur at the end of a run;
- the matmul computes one single-expert 64x64 output tile per threadgroup
  (four simdgroups in a 32x2x2 threadgroup; 1-D threadgroups measured up
  to 16% slower). Two schedules share the tile list: ``seg`` (mlx's
  segmented kernel: weight tile dequantized into threadgroup memory
  between two barriers per K step) and ``db`` (double-buffered weight
  tiles, one barrier per K step) for up to 128 rows per expert. Both skip
  the 16-row activation fragments of a partial tile that hold no rows.

Both schedules dequantize exactly like mlx (fp32 ``scale * q + bias``
rounded once to the activation dtype for affine; ``bfloat(e8m0) * e2m1``
for MXFP4) and issue the same 16x32x16 tensor ops in the same K order, so
every output element is bit-identical to mlx's sorted kernel wherever that
kernel is correct. The K tail zero-fills both operands past K (the stock
kernel reads stale activations there; mlx's fixed kernel still reads the
weight bytes and scales past the row, which can be NaN at the end of the
last expert) and row offsets are 32-bit.

Supported: ``transpose=True``, rhs-indices only, ``x`` of shape
``[M, 1, K]`` with a flat sorted ``uint32`` index of length ``M``, bf16/fp16
activations, affine 4/8-bit with group 32/64/128 (scales and biases in the
activation dtype) and MXFP4 (group 32). Anything else returns None and the
caller keeps the stock path. ``OMLX_M5_GATHER_QMM_NAX=0`` disables the
module; ``OMLX_M5_GATHER_QMM_NAX_SCHEDULE=seg|db`` pins a schedule (testing).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_ENV_ENABLE = "OMLX_M5_GATHER_QMM_NAX"
_ENV_SCHEDULE = "OMLX_M5_GATHER_QMM_NAX_SCHEDULE"
_ENV_TILE = "OMLX_M5_GATHER_QMM_NAX_BM"

# Tile geometry (fixed; the Metal source assumes it).
_BM = 64
_BN = 64
_WM = 2
_WN = 2
# Taller variant (8 simdgroups): one weight-tile stream serves 128 rows.
_BM_TALL = 128

# Largest expert count the one-threadgroup pre-pass handles (its run
# bounds live in threadgroup memory).
_MAX_EXPERTS = 2048

# Use the double-buffered schedule up to this many sorted rows per expert.
# Measured on M5 Ultra: db is up to 12% faster at 14-64 rows per expert and
# on par at 114-128; from 160 rows (mostly full tiles) the single-buffered
# schedule, whose smaller threadgroup footprint keeps more tiles in flight,
# is up to 5% faster.
_DB_MAX_ROWS_PER_EXPERT = 128

_MLX_UTILS_HEADERS = (
    "mlx/backend/metal/kernels/utils.h",
    "mlx/backend/metal/kernels/bf16.h",
    "mlx/backend/metal/kernels/bf16_math.h",
    "mlx/backend/metal/kernels/complex.h",
    "mlx/backend/metal/kernels/defines.h",
    "mlx/backend/metal/kernels/logging.h",
)


# mlx headers the matmul kernels build on: the NAX tile primitives and the
# fp4/fp8 element types.
_MLX_MM_HEADERS = (
    "mlx/backend/metal/kernels/steel/gemm/nax.h",
    "mlx/backend/metal/kernels/fp4.h",
    "mlx/backend/metal/kernels/fp8.h",
)


def _read_mlx_headers(paths: tuple[str, ...]) -> Optional[str]:
    """Flatten mlx kernel headers from the installed package.

    ``mx.fast.metal_kernel`` already prepends mlx's ``utils.h`` preamble, so
    it (and what it includes) is skipped; quoted mlx includes are inlined
    once and ``#pragma once`` dropped, system includes are kept.
    """
    root = Path(mx.__file__).parent / "include"
    if not root.is_dir():
        return None
    seen = {root / p for p in _MLX_UTILS_HEADERS}

    def expand(rel: str) -> str:
        path = root / rel
        if path in seen:
            return ""
        seen.add(path)
        lines = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith('#include "mlx/') and stripped.endswith('"'):
                lines.append(expand(stripped[len('#include "') : -1]))
            elif stripped != "#pragma once":
                lines.append(line)
        return "\n".join(lines)

    try:
        return "\n".join(expand(p) for p in paths)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Tile pre-pass
# ---------------------------------------------------------------------------

_SCAN_HEADER = """
using namespace metal;

// Cuts the sorted rows into (row_start, expert, rows, 0) tiles of at most
// BM rows of one expert, expert-major (the tile order of mlx's segmented
// gather_qmm). One threadgroup: the run bounds of every expert are found in
// parallel over the rows, then a threadgroup scan of the per-expert tile
// counts gives each expert's first tile. At most max_tiles tiles are
// written (a guard for unsorted input, which the contract excludes).
template <int BM>
METAL_FUNC void omlx_gqmm_tile_scan(
    const device uint32_t* idx,
    const constant int* params,
    device uint32_t* tiles,
    device uint32_t* tile_count,
    threadgroup uint32_t* run_start,
    threadgroup uint32_t* run_end,
    threadgroup uint32_t* simd_tot,
    const uint lid,
    const uint tg_size,
    const uint sg,
    const uint lane) {
  const int M = params[0];
  const int E = params[1];
  const uint32_t max_tiles = uint32_t(params[2]);
  for (int e = int(lid); e < E; e += int(tg_size)) {
    run_start[e] = 0;
    run_end[e] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Each thread walks 4 consecutive rows per step, with their neighbours
  // (0xffffffff past either end, never a valid expert).
  for (int g0 = 4 * int(lid); g0 < M; g0 += 4 * int(tg_size)) {
    const int cnt = min(4, M - g0);
    uint32_t v[6];
    v[0] = g0 > 0 ? idx[g0 - 1] : 0xffffffffu;
    for (int j = 0; j < 4; j++) {
      v[j + 1] = j < cnt ? idx[g0 + j] : 0xffffffffu;
    }
    v[5] = g0 + 4 < M ? idx[g0 + 4] : 0xffffffffu;
    for (int j = 0; j < cnt; j++) {
      const uint32_t e = v[j + 1];
      if (e < uint32_t(E)) {
        if (v[j] != e) {
          run_start[e] = uint32_t(g0 + j);
        }
        if (v[j + 2] != e) {
          run_end[e] = uint32_t(g0 + j + 1);
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (tg_size + 31) / 32;
  uint32_t running = 0;
  for (int base = 0; base < E; base += int(tg_size)) {
    const int e = base + int(lid);
    uint32_t start = 0;
    uint32_t cnt = 0;
    if (e < E) {
      start = run_start[e];
      const uint32_t end = run_end[e];
      cnt = end > start ? end - start : 0;
    }
    const uint32_t nt = (cnt + BM - 1) / BM;
    const uint32_t local = simd_prefix_exclusive_sum(nt);
    const uint32_t stot = simd_sum(nt);
    if (lane == 0) {
      simd_tot[sg] = stot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint32_t prefix = 0;
    uint32_t total = 0;
    for (uint s = 0; s < n_simd; s++) {
      const uint32_t v = simd_tot[s];
      prefix += (s < sg) ? v : 0;
      total += v;
    }
    const uint32_t off = running + prefix + local;
    for (uint32_t j = 0; j < nt && off + j < max_tiles; j++) {
      const uint32_t r = start + j * BM;
      *((device uint4*)tiles + off + j) =
          uint4(r, uint32_t(e), min(uint32_t(BM), start + cnt - r), 0);
    }
    running += total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) {
    tile_count[0] = min(running, max_tiles);
  }
}
"""

_SCAN_SOURCE = """
    threadgroup uint32_t run_start[MAXE];
    threadgroup uint32_t run_end[MAXE];
    threadgroup uint32_t simd_tot[32];
    omlx_gqmm_tile_scan<BM>(
        idx, params, tiles, tile_count, run_start, run_end, simd_tot,
        thread_index_in_threadgroup, threads_per_threadgroup.x,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------

_MM_HEADER = """
using namespace metal;
using namespace mlx::steel;

namespace omlx_gqmm {

STEEL_CONST int kBM = 64;
STEEL_CONST int kBN = 64;
STEEL_CONST int kBK = 64;
STEEL_CONST int kWM = 2;
STEEL_CONST int kWN = 2;
STEEL_CONST int kThreads = kWM * kWN * 32;
STEEL_CONST short kSK = 32;
STEEL_CONST short kSM = kBM / kWM;
STEEL_CONST short kSN = kBN / kWN;
STEEL_CONST short kTM = kSM / 16;
STEEL_CONST short kTN = kSN / 16;
STEEL_CONST short kTK = kSK / 16;
// Every loader thread dequantizes kVPT consecutive values of one weight row
// (the split of mlx's QuantizedBlockLoader for a 64x64 tile).
STEEL_CONST int kVPT = kBN * kBK / kThreads;
STEEL_CONST int kTPR = kBK / kVPT;

// Affine: w = scale * q + bias computed in fp32 and rounded once to T, as
// mlx's dequantize() does (scale * q is exact in fp32).
template <typename T, int GS, int BITS>
struct AffineQ {
  using WT = T;
  STEEL_CONST int kBits = BITS;
  STEEL_CONST int kGroup = GS;
  const device T* scales;
  const device T* biases;
  float s;
  float b;

  METAL_FUNC void advance(const size_t n) thread {
    scales += n;
    biases += n;
  }
  METAL_FUNC void load_params(const int g) thread {
    s = float(scales[g]);
    b = float(biases[g]);
  }
  METAL_FUNC WT dq(const uint32_t q) const thread {
    return static_cast<WT>(s * float(q) + b);
  }
};

// MXFP4: e2m1 values times the e8m0 group scale, dequantized to bfloat like
// mlx's fp QuantizedBlockLoader (Wtype = bfloat).
template <int GS>
struct Mxfp4Q {
  using WT = bfloat;
  STEEL_CONST int kBits = 4;
  STEEL_CONST int kGroup = GS;
  const device uint8_t* scales;
  float s;

  METAL_FUNC void advance(const size_t n) thread {
    scales += n;
  }
  METAL_FUNC void load_params(const int g) thread {
    uint8_t sb = scales[g];
    s = float(static_cast<bfloat>(*(thread fp8_e8m0*)(&sb)));
  }
  METAL_FUNC WT dq(const uint32_t q) const thread {
    uint8_t qb = uint8_t(q);
    return static_cast<WT>(s * float(*(thread fp4_e2m1*)(&qb)));
  }
};

// Weight-tile loader: thread lid owns row lid / kTPR of the BN x BK tile
// and the kVPT values from column (lid % kTPR) * kVPT. fetch() reads the
// packed words and group parameters of one K step, store() dequantizes
// them into threadgroup memory (row stride BKP).
template <typename Q>
struct TileLoader {
  using WT = typename Q::WT;
  STEEL_CONST int kBits = Q::kBits;
  STEEL_CONST int kWords = kVPT * kBits / 32;
  STEEL_CONST int kPer = 32 / kBits;
  STEEL_CONST uint32_t kMask = (1u << kBits) - 1u;
  static_assert(kWords * 32 == kVPT * kBits, "whole words per thread");
  static_assert(Q::kGroup % kVPT == 0, "one group per thread and K step");

  const device uint32_t* src;
  Q q;
  const short row;
  const short col;
  uint32_t raw[kWords];

  METAL_FUNC TileLoader(
      const device uint8_t* w_tile,
      const int K,
      thread const Q& q_,
      const uint lid) thread
      : q(q_), row(short(lid / kTPR)), col(short((lid % kTPR) * kVPT)) {
    src = (const device uint32_t*)(w_tile + size_t(row) * (K * kBits / 8) +
                                   col * kBits / 8);
    q.advance(size_t(row) * (K / Q::kGroup));
  }

  METAL_FUNC void fetch(const int kb) thread {
    const device uint32_t* p = src + kb * (kBK * kBits / 32);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      raw[i] = p[i];
    }
    q.load_params((kb * kBK + col) / Q::kGroup);
  }

  METAL_FUNC void store(threadgroup WT* Ws) const thread {
    threadgroup WT* dst = Ws + row * (kBK + 16 / sizeof(WT)) + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      vec<WT, kPer> v;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kPer; j++) {
        v[j] = q.dq((raw[i] >> (kBits * j)) & kMask);
      }
      *(threadgroup vec<WT, kPer>*)(dst + i * kPer) = v;
    }
  }

  METAL_FUNC void zero(threadgroup WT* Ws) const thread {
    threadgroup WT* dst = Ws + row * (kBK + 16 / sizeof(WT)) + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kVPT; i++) {
      dst[i] = WT(0);
    }
  }
};

// seg: mlx's segmented sorted gather kernel (affine_gather_qmm_rhs_seg_nax /
// fp_gather_qmm_rhs_seg_nax): one single-expert BM x BN tile per
// threadgroup, the weight tile dequantized into threadgroup memory between
// two barriers per K step, then tile_matmad_nax. K tail: both operands are
// zero past K. N tail: weight rows past N are zero, stores are bounded.
template <typename T, typename Q, bool ALIGN_N, bool ALIGN_K>
METAL_FUNC void gather_seg(
    const device T* x,
    const device uint8_t* w,
    thread Q& q,
    const device uint32_t* tiles,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint3 tid,
    const uint sgid,
    const uint lane) {
  using WT = typename Q::WT;
  constexpr int BKP = kBK + 16 / sizeof(WT);
  const uint4 desc = *((const device uint4*)tiles + tid.y);
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / kBK;
  const int y_col = int(tid.x) * kBN;
  const short tgp_bn = ALIGN_N ? short(kBN) : short(min(kBN, N - y_col));
  const int k_remain = K - K_it * kBK;

  const size_t w_row = size_t(expert) * N + y_col;
  q.advance(w_row * K_g);
  TileLoader<Q> loader(w + w_row * K_w, K, q, sgid * 32 + lane);
  const bool row_live = ALIGN_N || loader.row < tgp_bn;

  x += size_t(row_start) * K;
  y += size_t(row_start) * N + y_col;

  const short tm = kSM * short(sgid / kWN);
  const short tn = kSN * short(sgid % kWN);
  const short sgp_sm = short(min(int(kSM), max(0, rows - int(tm))));
  const short sgp_sn =
      ALIGN_N ? kSN : short(min(int(kSN), max(0, N - (y_col + tn))));
  const bool sg_active = sgp_sm > 0;

  NAXTile<float, kTM, kTN> Dtile;
  Dtile.clear();
  const device T* xn = x + tm * K;

  dispatch_bool(sgp_sm == kSM, [&](auto kAlignedM) {
    for (int k = 0; k < K_it; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (row_live) {
        loader.fetch(k);
        loader.store(Ws);
      } else {
        loader.zero(Ws);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < kBK; kk1 += kSK) {
        if (sg_active) {
          NAXTile<WT, kTN, kTK> Btile;
          if constexpr (kAlignedM.value) {
            NAXTile<T, kTM, kTK> Atile;

            volatile int compiler_barrier;

            Atile.load(xn + kk1, K);
            Btile.template load<WT, BKP, 1>(Ws + tn * BKP + kk1);

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<true>{});

            (void)compiler_barrier;
          } else {
            // Partial tile: skip the 16-row fragments without rows (the
            // tensor ops of the others are the ones tile_matmad_nax issues).
            Btile.template load<WT, BKP, 1>(Ws + tn * BKP + kk1);
            STEEL_PRAGMA_UNROLL
            for (short mm = 0; mm < kTM; mm++) {
              if (mm * 16 < sgp_sm) {
                NAXTile<T, 1, kTK> Arow;
                Arow.load_safe(
                    xn + mm * 16 * K + kk1, K, short2(kSK, sgp_sm - mm * 16));
                STEEL_PRAGMA_UNROLL
                for (short nn = 0; nn < kTN; nn += 2) {
                  STEEL_PRAGMA_UNROLL
                  for (short kk = 0; kk < kTK; kk++) {
                    BaseNAXFrag::mma(
                        Dtile.frag_at(mm, nn),
                        Dtile.frag_at(mm, nn + 1),
                        Arow.frag_at(0, kk),
                        metal::bool_constant<false>{},
                        Btile.frag_at(nn, kk),
                        Btile.frag_at(nn + 1, kk),
                        metal::bool_constant<true>{});
                  }
                }
              }
            }
          }
        }
      }
      xn += kBK;
    }

    if (!ALIGN_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (row_live && loader.col < k_remain) {
        loader.fetch(K_it);
        loader.store(Ws);
      } else {
        loader.zero(Ws);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < kBK; kk1 += kSK) {
        if (sg_active) {
          NAXTile<T, kTM, kTK> Atile;
          NAXTile<WT, kTN, kTK> Btile;

          volatile int compiler_barrier;

          const short psk = short(min(int(kSK), max(0, k_remain - kk1)));
          Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));
          Btile.template load<WT, BKP, 1>(Ws + tn * BKP + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<false>{},
              Btile,
              metal::bool_constant<true>{});

          (void)compiler_barrier;
        }
      }
    }

    if (kAlignedM.value && sgp_sn == kSN) {
      Dtile.store(y + tm * N + tn, N);
    } else if (sg_active) {
      Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
    }
  });
}

// db: the same tiles and arithmetic with double-buffered weight tiles: the
// packed words of step k + 1 are fetched before the tensor ops of step k and
// dequantized into the other buffer after them, so each K step has a single
// barrier. Activation fragments are read straight from device memory (rows
// past the tile are clamped to its last row and never stored) and 16-row
// fragments without rows of the tile are skipped. Requires K % 64 == 0 and
// N % 64 == 0.
template <typename T, typename Q>
METAL_FUNC void gather_db(
    const device T* x,
    const device uint8_t* w,
    thread Q& q,
    const device uint32_t* tiles,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint3 tid,
    const uint sgid,
    const uint lane) {
  using WT = typename Q::WT;
  constexpr int BKP = kBK + 16 / sizeof(WT);
  constexpr int kTile = kBN * BKP;
  const uint4 desc = *((const device uint4*)tiles + tid.y);
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / kBK;
  const int y_col = int(tid.x) * kBN;

  const size_t w_row = size_t(expert) * N + y_col;
  q.advance(w_row * K_g);
  TileLoader<Q> loader(w + w_row * K_w, K, q, sgid * 32 + lane);

  const int m0 = kSM * int(sgid / kWN);
  const int rows = min(int(kSM), tile_rows - m0);
  const device T* xs =
      x + size_t(row_start + max(0, min(m0, tile_rows - 1))) * K;

  const short2 sc = BaseNAXFrag::get_coord();
  int x_off[kTM][2];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(int(i * 16 + sc.y + h * 8), max(rows, 1) - 1);
      x_off[i][h] = r * K + sc.x;
    }
  }
  const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
  const threadgroup WT* wsg = Ws + (sgid % kWN) * kSN * BKP;

  NAXTile<float, kTM, kTN> D;
  D.clear();

  loader.fetch(0);
  loader.store(Ws);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int kb = 0; kb < K_it; kb++) {
    const bool more = kb + 1 < K_it;
    if (more) {
      loader.fetch(kb + 1);
    }
    const threadgroup WT* wb = wsg + (kb & 1) * kTile;
    STEEL_PRAGMA_UNROLL
    for (short kk1 = 0; kk1 < kBK; kk1 += kSK) {
      NAXTile<WT, kTN, 2> Btile;
      Btile.template load<WT, BKP, 1>(wb + kk1);
      const int k = kb * kBK + kk1;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kTM; i++) {
        if (i < m_frags) {
          NAXTile<T, 1, 2> Atile;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const device T* xp = xs + x_off[i][h] + k;
            const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
            const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              Atile.frag_at(0, 0)[h * 4 + c] = a0[c];
              Atile.frag_at(0, 1)[h * 4 + c] = a1[c];
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < 2; kk++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < kTN; j += 2) {
              BaseNAXFrag::mma(
                  D.frag_at(i, j),
                  D.frag_at(i, j + 1),
                  Atile.frag_at(0, kk),
                  metal::bool_constant<false>{},
                  Btile.frag_at(j, kk),
                  Btile.frag_at(j + 1, kk),
                  metal::bool_constant<true>{});
            }
          }
        }
      }
    }
    if (more) {
      loader.store(Ws + ((kb + 1) & 1) * kTile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  device T* yb = y + size_t(row_start + m0) * N + y_col + kSN * (sgid % kWN);
  if (rows >= kSM) {
    D.store(yb, N);
  } else if (rows > 0) {
    D.store_safe(yb, N, short2(kSN, short(rows)));
  }
}

} // namespace omlx_gqmm
"""

_AFFINE_SOURCE = """
    using Q = omlx_gqmm::AffineQ<T, GS, BITS>;
    constexpr int BKP = omlx_gqmm::kBK + 16 / sizeof(T);
    threadgroup T Ws[(SCHED == 1 ? 2 : 1) * omlx_gqmm::kBN * BKP];
    if (threadgroup_position_in_grid.y >= tile_count[0]) {
        return;
    }
    Q q{scales, biases, 0.0f, 0.0f};
    if constexpr (SCHED == 1) {
        omlx_gqmm::gather_db<T, Q>(
            x, (const device uint8_t*)w, q, tiles, y, params[0], params[1],
            Ws, threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    } else {
        omlx_gqmm::gather_seg<T, Q, ALIGN_N, ALIGN_K>(
            x, (const device uint8_t*)w, q, tiles, y, params[0], params[1],
            Ws, threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    }
"""

_FP_SOURCE = """
    using Q = omlx_gqmm::Mxfp4Q<GS>;
    constexpr int BKP = omlx_gqmm::kBK + 16 / sizeof(bfloat);
    threadgroup bfloat Ws[(SCHED == 1 ? 2 : 1) * omlx_gqmm::kBN * BKP];
    if (threadgroup_position_in_grid.y >= tile_count[0]) {
        return;
    }
    Q q{scales, 0.0f};
    if constexpr (SCHED == 1) {
        omlx_gqmm::gather_db<T, Q>(
            x, (const device uint8_t*)w, q, tiles, y, params[0], params[1],
            Ws, threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    } else {
        omlx_gqmm::gather_seg<T, Q, ALIGN_N, ALIGN_K>(
            x, (const device uint8_t*)w, q, tiles, y, params[0], params[1],
            Ws, threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    }
"""

_SCHED_SEG = 0
_SCHED_DB = 1
_SCHED_NAMES = {_SCHED_SEG: "seg", _SCHED_DB: "db"}

_lock = threading.RLock()
_kernels: dict[str, object] = {}
_header_failed = False
# Self-test verdict per kernel instantiation:
# (dtype, mode, bits, group_size, schedule, align_N, align_K, tile_rows) -> bool.
_verified: dict[tuple, bool] = {}


def enabled() -> bool:
    """False when ``OMLX_M5_GATHER_QMM_NAX`` disables the module."""
    return os.environ.get(_ENV_ENABLE, "1").strip().lower() not in {
        "0",
        "false",
        "off",
    }


def _mm_header(bm: int) -> str:
    """The matmul header for a tile height (``kSM`` stays 32 rows/simdgroup)."""
    if bm == _BM:
        return _MM_HEADER
    return _MM_HEADER.replace(
        "STEEL_CONST int kBM = 64;", f"STEEL_CONST int kBM = {bm};"
    ).replace("STEEL_CONST int kWM = 2;", f"STEEL_CONST int kWM = {bm // 32};")


def _get_kernel(kind: str, bm: int = _BM):
    """Build (once) the ``scan``, ``affine`` or ``fp`` kernel object."""
    global _header_failed
    cache_key = kind if kind == "scan" or bm == _BM else f"{kind}_bm{bm}"
    kernel = _kernels.get(cache_key)
    if kernel is not None or _header_failed:
        return kernel
    with _lock:
        kernel = _kernels.get(cache_key)
        if kernel is not None:
            return kernel
        if kind == "scan":
            kernel = mx.fast.metal_kernel(
                name="omlx_gqmm_tile_scan",
                input_names=["idx", "params"],
                output_names=["tiles", "tile_count"],
                header=_SCAN_HEADER,
                source=_SCAN_SOURCE,
            )
        else:
            mlx_src = _read_mlx_headers(_MLX_MM_HEADERS)
            if mlx_src is None:
                _header_failed = True
                logger.warning(
                    "mlx kernel headers not found under %s; NAX sorted "
                    "gather_qmm disabled",
                    Path(mx.__file__).parent / "include",
                )
                return None
            if kind == "affine":
                kernel = mx.fast.metal_kernel(
                    name="omlx_gqmm_affine" + ("" if bm == _BM else f"_bm{bm}"),
                    input_names=[
                        "x",
                        "w",
                        "scales",
                        "biases",
                        "tiles",
                        "tile_count",
                        "params",
                    ],
                    output_names=["y"],
                    header=mlx_src + _mm_header(bm),
                    source=_AFFINE_SOURCE,
                )
            else:
                kernel = mx.fast.metal_kernel(
                    name="omlx_gqmm_mxfp4" + ("" if bm == _BM else f"_bm{bm}"),
                    input_names=["x", "w", "scales", "tiles", "tile_count", "params"],
                    output_names=["y"],
                    header=mlx_src + _mm_header(bm),
                    source=_FP_SOURCE,
                )
        _kernels[cache_key] = kernel
        return kernel


def _schedule(rows: int, experts: int, K: int, N: int) -> int:
    forced = os.environ.get(_ENV_SCHEDULE, "").strip().lower()
    aligned = K % 64 == 0 and N % 64 == 0
    if forced == "seg" or not aligned:
        return _SCHED_SEG
    if forced == "db":
        return _SCHED_DB
    if rows <= _DB_MAX_ROWS_PER_EXPERT * experts:
        return _SCHED_DB
    return _SCHED_SEG


def _tile_rows(rows: int, experts: int, K: int) -> int:
    """Rows per output tile: 128 when experts average 65-128 rows.

    A 128-row tile streams each expert's weight tile once where two 64-row
    tiles stream it twice (M5 Ultra, GLM-5.3 at 4096-token chunks, 114 rows
    per expert: +5-8% on the expert GEMMs, bit-identical). With fewer rows,
    or a short K, the half-empty taller tile is slower.
    """
    forced = os.environ.get(_ENV_TILE, "").strip()
    if forced in ("64", "128"):
        return int(forced)
    per_expert = rows / max(1, experts)
    if 64 < per_expert <= 128 and K >= 2048:
        return _BM_TALL
    return _BM


def supports(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: Optional[mx.array],
    indices: mx.array,
    group_size: int,
    bits: int,
    mode: str,
) -> bool:
    """True when ``sorted_gather_qmm`` handles this call (layout/dtypes)."""
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if x.ndim != 3 or x.shape[1] != 1 or indices.ndim != 1:
        return False
    M, K = int(x.shape[0]), int(x.shape[2])
    # Fewer than 8 indices would be bound as a constant buffer.
    if M < 8 or indices.shape[0] != M or indices.dtype != mx.uint32:
        return False
    if w.ndim != 3 or w.dtype != mx.uint32:
        return False
    E, N = int(w.shape[0]), int(w.shape[1])
    if E == 0 or E > _MAX_EXPERTS or N == 0 or K % 32:
        return False
    if mode == "affine":
        if bits not in (4, 8) or group_size not in (32, 64, 128):
            return False
        if biases is None or K % group_size:
            return False
        if scales.dtype != x.dtype or biases.dtype != x.dtype:
            return False
        if biases.shape != scales.shape:
            return False
    elif mode == "mxfp4":
        if bits != 4 or group_size != 32 or biases is not None:
            return False
        if scales.dtype != mx.uint8:
            return False
    else:
        return False
    if w.shape[2] * 32 != K * bits:
        return False
    return scales.shape == (E, N, K // group_size)


def _launch(
    x, w, scales, biases, indices, group_size, bits, mode, sched, stream, bm=_BM
):
    scan = _get_kernel("scan")
    mm = _get_kernel("affine" if mode == "affine" else "fp", bm)
    if scan is None or mm is None:
        return None
    M, K = int(x.shape[0]), int(x.shape[2])
    E, N = int(w.shape[0]), int(w.shape[1])
    max_tiles = (M + bm - 1) // bm + min(E, M)
    kw = {} if stream is None else {"stream": stream}
    tiles, tile_count = scan(
        inputs=[indices, mx.array([M, E, max_tiles], dtype=mx.int32)],
        template=[("BM", bm), ("MAXE", _MAX_EXPERTS)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(max_tiles * 4,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
        **kw,
    )
    inputs = [x, w, scales]
    template = [("T", x.dtype), ("GS", group_size)]
    if mode == "affine":
        inputs.append(biases)
        template.append(("BITS", bits))
    inputs += [tiles, tile_count, mx.array([N, K], dtype=mx.int32)]
    template += [
        ("SCHED", int(sched)),
        ("ALIGN_N", N % _BN == 0),
        ("ALIGN_K", K % 64 == 0),
    ]
    n_cols = (N + _BN - 1) // _BN
    return mm(
        inputs=inputs,
        template=template,
        grid=(n_cols * 32, max_tiles * _WN, bm // 32),
        threadgroup=(32, _WN, bm // 32),
        output_shapes=[(M, 1, N)],
        output_dtypes=[x.dtype],
        **kw,
    )[0]


def _stock_gather_qmm():
    """The raw mlx op, also when ``m5_gather_qmm`` has wrapped it."""
    fn = mx.gather_qmm
    if getattr(fn, "_omlx_m5_reroute", False):
        from omlx.patches import m5_gather_qmm

        fn = m5_gather_qmm._original_gather_qmm or fn
    return fn


# Canary routing: an empty expert, runs spanning several 64-row tiles, and
# partial tiles of every size class.
_CANARY_COUNTS = (70, 0, 5, 33, 64, 17, 100, 11)


def _self_test(key: tuple) -> Optional[bool]:
    """Run one kernel instantiation on a small canary.

    Aligned K must be bit-identical to mlx's sorted kernel (correct there);
    ragged K must match an fp32 dequantized reference to bf16 rounding.
    Returns None when the canary could not be evaluated here (e.g. while a
    function transformation is being traced); the caller then retries.
    """
    dtype, mode, bits, group_size, sched, align_n, align_k, bm = key
    E = len(_CANARY_COUNTS)
    N = 128 if align_n else 96
    K = 256 if align_k else 160
    try:
        k_w, k_x = mx.random.split(mx.random.key(0x2267), 2)
        wf = (mx.random.normal((E, N, K), key=k_w) * 0.05).astype(dtype)
        if mode == "affine":
            wq, scales, biases = mx.quantize(wf, group_size=group_size, bits=bits)
            wd = mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits)
        else:
            wq, scales = mx.quantize(wf, group_size=group_size, bits=bits, mode=mode)
            biases = None
            wd = mx.dequantize(
                wq, scales, group_size=group_size, bits=bits, mode=mode
            )
        idx = mx.array(
            [e for e, n in enumerate(_CANARY_COUNTS) for _ in range(n)],
            dtype=mx.uint32,
        )
        M = int(idx.shape[0])
        x = (mx.random.normal((M, 1, K), key=k_x) * 0.5).astype(dtype)
        out = _launch(
            x, wq, scales, biases, idx, group_size, bits, mode, sched, None, bm
        )
        if out is None:
            return False
        if align_k:
            ref = _stock_gather_qmm()(
                x,
                wq,
                scales,
                biases,
                rhs_indices=idx,
                transpose=True,
                group_size=group_size,
                bits=bits,
                mode=mode,
                sorted_indices=True,
            )
            ok = bool(mx.array_equal(out, ref).item())
            detail = "not bit-identical to mlx's sorted kernel"
        else:
            ref = (
                x.astype(mx.float32)
                @ wd[idx].swapaxes(-1, -2).astype(mx.float32)
            )
            err = mx.abs(out.astype(mx.float32) - ref).max().item()
            scale = mx.abs(ref).max().item()
            ok = err <= scale / 64
            detail = f"max err {err:.3g} vs fp32 reference (max {scale:.3g})"
    except Exception as e:  # noqa: BLE001
        if "transformation" in str(e):
            return None
        logger.warning(
            "NAX sorted gather_qmm self-test raised for %s: %s", _describe(key), e
        )
        return False
    if ok:
        logger.info("NAX sorted gather_qmm armed for %s", _describe(key))
    else:
        logger.warning(
            "NAX sorted gather_qmm disabled for %s: canary %s",
            _describe(key),
            detail,
        )
    return ok


def _describe(key: tuple) -> str:
    dtype, mode, bits, group_size, sched, align_n, align_k, bm = key
    return (
        f"{str(dtype).rsplit('.', 1)[-1]} {mode} {bits}-bit gs{group_size} "
        f"({_SCHED_NAMES[sched]}{'' if align_n else ', ragged N'}"
        f"{'' if align_k else ', ragged K'}{'' if bm == _BM else f', {bm}-row tiles'})"
    )


def sorted_gather_qmm(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: Optional[mx.array],
    indices: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
    stream=None,
    schedule: Optional[int] = None,
    verify: bool = True,
) -> Optional[mx.array]:
    """``x @ w[indices].T`` for sorted rows on the tensor units.

    Returns None when the module is disabled, the call is not supported
    (see ``supports``), the kernels cannot be built or the instantiation
    failed its one-time self-test; the caller then keeps the stock path.
    """
    if not enabled() or not supports(
        x, w, scales, biases, indices, group_size, bits, mode
    ):
        return None
    M, K = int(x.shape[0]), int(x.shape[2])
    E, N = int(w.shape[0]), int(w.shape[1])
    sched = _schedule(M, E, K, N) if schedule is None else int(schedule)
    if sched == _SCHED_DB and (K % 64 or N % 64):
        sched = _SCHED_SEG
    bm = _tile_rows(M, E, K)
    if verify:
        key = (x.dtype, mode, bits, group_size, sched, N % _BN == 0, K % 64 == 0, bm)
        ok = _verified.get(key)
        if ok is None:
            with _lock:
                ok = _verified.get(key)
                if ok is None:
                    ok = _self_test(key)
                    if ok is not None:
                        _verified[key] = ok
        if not ok:
            return None
    return _launch(
        x, w, scales, biases, indices, group_size, bits, mode, sched, stream, bm
    )
