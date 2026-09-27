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
  (row_start, expert, rows) tiles of at most BM rows (64, 96 or 128), so
  partial tiles only occur at the end of a run;
- the matmul computes one single-expert BM x 64 output tile per
  threadgroup (BM / 32 x 2 simdgroups, each owning a 32 x 32 block). Two
  schedules share the tile list: ``seg`` (mlx's segmented kernel: the
  weight tile of a K step, 64 or 128 deep, dequantized into threadgroup
  memory between two barriers) and ``db`` (double-buffered 64-deep weight
  tiles, one barrier per K step). Both skip the 16-row activation
  fragments of a partial tile that hold no rows.
- threadgroups are either laid out (column, tile) as mlx does, or with
  the tile index on the grid's x axis in groups of 32 tiles and
  (group, column) on y, so every threadgroup of a row tile shares one x
  coordinate and a tile's columns run 32 threadgroups apart. On M5 Ultra
  the gain tracked how few x coordinates a row tile's threadgroups span
  (activation reuse across its columns): 3-28% from 36 rows per expert.
  Below that weight streaming dominates and the plain layout (each
  expert's column slabs in order) stays faster.

``_plan`` picks the schedule, tile height, K step and layout from the
mean rows per expert and K (measured on M5 Ultra at the Qwen3.8, GLM-5.3
and MiMo-V2.6 expert shapes).

Every configuration dequantizes exactly like mlx (fp32 ``scale * q +
bias`` rounded once to the activation dtype for affine; ``bfloat(e8m0) *
e2m1`` for MXFP4) and issues the same 16x32x16 tensor ops in the same K
order, so every output element is bit-identical to mlx's sorted kernel
wherever that kernel is correct. A K tail (a multiple of 32) runs only its
valid 32-deep sub-steps and never reads weight bytes or scales past K (the
stock kernel reads stale activations there; mlx's fixed kernel still reads
the weight bytes and scales past the row, which can be NaN at the end of
the last expert) and row offsets are 32-bit.

Supported: ``transpose=True``, rhs-indices only, ``x`` of shape
``[M, 1, K]`` with a flat sorted ``uint32`` index of length ``M``, bf16/fp16
activations, affine 4/8-bit with group 32/64/128 (scales and biases in the
activation dtype) and MXFP4 (group 32). Anything else returns None and the
caller keeps the stock path. ``OMLX_M5_GATHER_QMM_NAX=0`` disables the
module; ``OMLX_M5_GATHER_QMM_NAX_PLAN=sched,bm,bk,gx,pad`` (e.g.
``seg,128,128,32,8192``) pins a configuration (testing).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import NamedTuple, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_ENV_ENABLE = "OMLX_M5_GATHER_QMM_NAX"
_ENV_PLAN = "OMLX_M5_GATHER_QMM_NAX_PLAN"

# Output tile width and column simdgroups (fixed; the Metal source assumes
# them). Tile heights are multiples of 32 rows (one row simdgroup each).
_BN = 64
_WN = 2
_TILE_ROWS = (64, 96, 128)

# Largest expert count the one-threadgroup pre-pass handles (its run
# bounds live in threadgroup memory).
_MAX_EXPERTS = 2048

# Row tiles per grid-x group in the tile-on-x layout.
_GX = 32

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

STEEL_CONST int kBN = 64;
STEEL_CONST int kWN = 2;
STEEL_CONST short kSM = 32;
STEEL_CONST short kSN = kBN / kWN;
STEEL_CONST short kSK = 32;
STEEL_CONST short kTM = kSM / 16;
STEEL_CONST short kTN = kSN / 16;
STEEL_CONST short kTK = kSK / 16;

// Tile geometry: BM rows in BM / 32 row simdgroups times kWN column
// simdgroups, K steps BK deep. kLT loader threads dequantize the kBN x BK
// weight tile, each kVPT consecutive values of one weight row: every
// thread when they split the tile evenly, else the largest power of two
// below the thread count (96-row tiles: 128 of 192).
template <int BM, int BK>
struct Geo {
  STEEL_CONST int kBM = BM;
  STEEL_CONST int kBK = BK;
  STEEL_CONST int kWM = BM / kSM;
  STEEL_CONST int kThreads = kWM * kWN * 32;
  STEEL_CONST int kLT = (kThreads & (kThreads - 1)) == 0
      ? kThreads
      : (kThreads > 256 ? 256 : (kThreads > 128 ? 128 : 64));
  STEEL_CONST int kVPT = kBN * BK / kLT;
  STEEL_CONST int kTPR = BK / kVPT;
  static_assert(BM % kSM == 0 && BK % kSK == 0, "tile geometry");
  static_assert(kTPR >= 1 && kTPR * kVPT == BK, "loader split");
};

// Affine: w = scale * q + bias computed in fp32 and rounded once to T, as
// mlx's dequantize() does (scale * q is exact in fp32).
template <typename T, int GS, int BITS>
struct AffineQ {
  using WT = T;
  STEEL_CONST int kBits = BITS;
  STEEL_CONST int kGroup = GS;
  const device T* scales;
  const device T* biases;

  struct P {
    float s;
    float b;
  };

  METAL_FUNC void advance(const size_t n) thread {
    scales += n;
    biases += n;
  }
  METAL_FUNC P params(const int g) const thread {
    return P{float(scales[g]), float(biases[g])};
  }
  METAL_FUNC static WT dq(thread const P& p, const uint32_t q) {
    return static_cast<WT>(p.s * float(q) + p.b);
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

  struct P {
    float s;
  };

  METAL_FUNC void advance(const size_t n) thread {
    scales += n;
  }
  METAL_FUNC P params(const int g) const thread {
    uint8_t sb = scales[g];
    return P{float(static_cast<bfloat>(*(thread fp8_e8m0*)(&sb)))};
  }
  METAL_FUNC static WT dq(thread const P& p, const uint32_t q) {
    uint8_t qb = uint8_t(q);
    return static_cast<WT>(p.s * float(*(thread fp4_e2m1*)(&qb)));
  }
};

// Weight-tile loader: loader thread lid owns row lid / kTPR of the
// kBN x BK tile and the kVPT values from column (lid % kTPR) * kVPT, in
// kNG chunks that each lie in one quantization group. fetch() reads the
// packed words and group parameters of one K step, store() dequantizes
// them into threadgroup memory (row stride BKP). The *_tail variants
// cover a K tail of k_valid (a multiple of 32) columns and never touch a
// word or group at or past it.
template <typename Q, typename G>
struct TileLoader {
  using WT = typename Q::WT;
  using P = typename Q::P;
  STEEL_CONST int kBits = Q::kBits;
  STEEL_CONST int kVPT = G::kVPT;
  STEEL_CONST int kWords = kVPT * kBits / 32;
  STEEL_CONST int kPer = 32 / kBits;
  STEEL_CONST uint32_t kMask = (1u << kBits) - 1u;
  STEEL_CONST int kGV = kVPT < Q::kGroup ? kVPT : Q::kGroup;
  STEEL_CONST int kNG = kVPT / kGV;
  STEEL_CONST int kWPG = kGV * kBits / 32;
  STEEL_CONST int kBKP = G::kBK + 16 / sizeof(WT);
  static_assert(kWords * 32 == kVPT * kBits, "whole words per thread");
  static_assert(kWPG >= 1 && kNG * kWPG == kWords, "group split");

  const device uint32_t* src;
  Q q;
  const short row;
  const short col;
  uint32_t raw[kWords];
  P p[kNG];

  METAL_FUNC TileLoader(
      const device uint8_t* w_tile,
      const int K,
      thread const Q& q_,
      const uint lid) thread
      : q(q_),
        row(short(lid / G::kTPR)),
        col(short((lid % G::kTPR) * kVPT)) {
    src = (const device uint32_t*)(w_tile + size_t(row) * (K * kBits / 8) +
                                   col * kBits / 8);
    q.advance(size_t(row) * (K / Q::kGroup));
  }

  METAL_FUNC void fetch(const int kb) thread {
    const device uint32_t* ptr = src + kb * (G::kBK * kBits / 32);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      raw[i] = ptr[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < kNG; g++) {
      p[g] = q.params((kb * G::kBK + col + g * kGV) / Q::kGroup);
    }
  }

  METAL_FUNC void fetch_tail(const int kb, const int k_valid) thread {
    const device uint32_t* ptr = src + kb * (G::kBK * kBits / 32);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      if (col + i * kPer < k_valid) {
        raw[i] = ptr[i];
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < kNG; g++) {
      if (col + g * kGV < k_valid) {
        p[g] = q.params((kb * G::kBK + col + g * kGV) / Q::kGroup);
      }
    }
  }

  METAL_FUNC void store_words(threadgroup WT* Ws, const int k_valid) const
      thread {
    threadgroup WT* dst = Ws + row * kBKP + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      if (col + i * kPer < k_valid) {
        vec<WT, kPer> v;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kPer; j++) {
          v[j] = Q::dq(p[i / kWPG], (raw[i] >> (kBits * j)) & kMask);
        }
        *(threadgroup vec<WT, kPer>*)(dst + i * kPer) = v;
      }
    }
  }

  METAL_FUNC void store(threadgroup WT* Ws) const thread {
    store_words(Ws, G::kBK);
  }

  METAL_FUNC void zero(threadgroup WT* Ws) const thread {
    threadgroup WT* dst = Ws + row * kBKP + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kVPT; i++) {
      dst[i] = WT(0);
    }
  }
};

// One 32-deep sub-step of a simdgroup's 32 x 32 block: full row blocks run
// tile_matmad_nax; partial ones skip the 16-row fragments without rows
// (the tensor ops of the others are the ones tile_matmad_nax issues).
template <typename T, typename WT, int BKP, bool FULL>
METAL_FUNC void sub_step(
    thread NAXTile<float, kTM, kTN>& Dtile,
    const device T* xn,
    const threadgroup WT* ws,
    const int K,
    const short sgp_sm) {
  NAXTile<WT, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<T, kTM, kTK> Atile;

    volatile int compiler_barrier;

    Atile.load(xn, K);
    Btile.template load<WT, BKP, 1>(ws);

    tile_matmad_nax(
        Dtile,
        Atile,
        metal::bool_constant<false>{},
        Btile,
        metal::bool_constant<true>{});

    (void)compiler_barrier;
  } else {
    Btile.template load<WT, BKP, 1>(ws);
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < kTM; mm++) {
      if (mm * 16 < sgp_sm) {
        NAXTile<T, 1, kTK> Arow;
        Arow.load_safe(xn + mm * 16 * K, K, short2(kSK, sgp_sm - mm * 16));
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

// seg: mlx's segmented sorted gather kernel (affine_gather_qmm_rhs_seg_nax /
// fp_gather_qmm_rhs_seg_nax): one single-expert BM x kBN tile per
// threadgroup, the weight tile of each BK-deep K step dequantized into
// threadgroup memory between two barriers. K tail (K % BK, a multiple of
// 32): only its sub-steps run. N tail: weight rows past N are zero, stores
// are bounded.
template <typename T, typename Q, typename G, bool ALIGN_N, bool ALIGN_K>
METAL_FUNC void gather_seg(
    const device T* x,
    const device uint8_t* w,
    thread Q& q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane) {
  using WT = typename Q::WT;
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;
  const short tgp_bn = ALIGN_N ? short(kBN) : short(min(kBN, N - y_col));
  const int k_remain = K - K_it * G::kBK;

  const size_t w_row = size_t(expert) * N + y_col;
  q.advance(w_row * K_g);
  TileLoader<Q, G> loader(w + w_row * K_w, K, q, sgid * 32 + lane);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);
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
  const threadgroup WT* ws = Ws + tn * BKP;

  dispatch_bool(sgp_sm == kSM, [&](auto kAlignedM) {
    for (int k = 0; k < K_it; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (loads) {
        if (row_live) {
          loader.fetch(k);
          loader.store(Ws);
        } else {
          loader.zero(Ws);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < G::kBK; kk1 += kSK) {
        if (sg_active) {
          sub_step<T, WT, BKP, kAlignedM.value>(
              Dtile, xn + kk1, ws + kk1, K, sgp_sm);
        }
      }
      xn += G::kBK;
    }

    if (!ALIGN_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (loads) {
        if (row_live) {
          loader.fetch_tail(K_it, k_remain);
          loader.store_words(Ws, k_remain);
        } else {
          loader.zero(Ws);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < k_remain; kk1 += kSK) {
        if (sg_active) {
          sub_step<T, WT, BKP, kAlignedM.value>(
              Dtile, xn + kk1, ws + kk1, K, sgp_sm);
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

// db: the same tiles and arithmetic with double-buffered 64-deep weight
// tiles: the packed words of step k + 1 are fetched before the tensor ops
// of step k and dequantized into the other buffer after them, so each K
// step has a single barrier. Activation fragments are read straight from
// device memory (rows past the tile are clamped to its last row and never
// stored) and 16-row fragments without rows of the tile are skipped.
// Requires K % 64 == 0 and N % 64 == 0.
template <typename T, typename Q, typename G>
METAL_FUNC void gather_db(
    const device T* x,
    const device uint8_t* w,
    thread Q& q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane) {
  using WT = typename Q::WT;
  static_assert(G::kBK == 64, "db runs 64-deep K steps");
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  constexpr int kTile = kBN * BKP;
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;

  const size_t w_row = size_t(expert) * N + y_col;
  q.advance(w_row * K_g);
  TileLoader<Q, G> loader(w + w_row * K_w, K, q, sgid * 32 + lane);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);

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

  if (loads) {
    loader.fetch(0);
    loader.store(Ws);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int kb = 0; kb < K_it; kb++) {
    const bool more = kb + 1 < K_it;
    if (more && loads) {
      loader.fetch(kb + 1);
    }
    const threadgroup WT* wb = wsg + (kb & 1) * kTile;
    STEEL_PRAGMA_UNROLL
    for (short kk1 = 0; kk1 < G::kBK; kk1 += kSK) {
      NAXTile<WT, kTN, 2> Btile;
      Btile.template load<WT, BKP, 1>(wb + kk1);
      const int k = kb * G::kBK + kk1;
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
    if (more && loads) {
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

// The row tile and output column of this threadgroup. GX == 0: grid
// (columns, tiles) as mlx lays it out. GX > 0: tile t on grid x t % GX and
// (t / GX, column) on y, so all threadgroups of a row tile share one x
// coordinate and a tile's columns run GX threadgroups apart.
template <int GX>
METAL_FUNC bool tile_of(
    const device uint32_t* tiles,
    const uint tile_count,
    const uint3 tid,
    const int N,
    thread uint4& desc,
    thread int& y_col) {
  uint t;
  uint c;
  if constexpr (GX > 0) {
    const uint n_cols = uint((N + kBN - 1) / kBN);
    t = (tid.y / n_cols) * GX + tid.x;
    c = tid.y % n_cols;
  } else {
    t = tid.y;
    c = tid.x;
  }
  if (t >= tile_count) {
    return false;
  }
  desc = *((const device uint4*)tiles + t);
  y_col = int(c) * kBN;
  return true;
}

} // namespace omlx_gqmm
"""

_MM_SOURCE_TMPL = """
    {q_type}
    using G = omlx_gqmm::Geo<BM, BK>;
    using WT = typename Q::WT;
    constexpr int BKP = BK + 16 / sizeof(WT);
    threadgroup WT Ws[(SCHED == 1 ? 2 : 1) * omlx_gqmm::kBN * BKP +
                      PAD / sizeof(WT)];
    uint4 desc;
    int y_col;
    if (!omlx_gqmm::tile_of<GX>(
            tiles, tile_count[0], threadgroup_position_in_grid, params[0],
            desc, y_col)) {{
        return;
    }}
    {q_init}
    if constexpr (SCHED == 1) {{
        omlx_gqmm::gather_db<T, Q, G>(
            x, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    }} else {{
        omlx_gqmm::gather_seg<T, Q, G, ALIGN_N, ALIGN_K>(
            x, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    }}
"""

_AFFINE_SOURCE = _MM_SOURCE_TMPL.format(
    q_type="using Q = omlx_gqmm::AffineQ<T, GS, BITS>;",
    q_init="Q q{scales, biases};",
)

_FP_SOURCE = _MM_SOURCE_TMPL.format(
    q_type="using Q = omlx_gqmm::Mxfp4Q<GS>;",
    q_init="Q q{scales};",
)

_SCHED_SEG = 0
_SCHED_DB = 1
_SCHED_NAMES = {_SCHED_SEG: "seg", _SCHED_DB: "db"}


class Plan(NamedTuple):
    """One kernel configuration: schedule, tile rows, K step, layout, pad.

    ``gx`` is the row tiles per grid-x group (0: mlx's (column, tile)
    grid); ``pad`` is extra threadgroup memory in bytes (fewer resident
    threadgroups).
    """

    sched: int
    bm: int
    bk: int
    gx: int
    pad: int

    def describe(self) -> str:
        s = f"{_SCHED_NAMES[self.sched]} {self.bm}x{_BN} bk{self.bk}"
        if self.gx:
            s += f" gx{self.gx}"
        if self.pad:
            s += f" pad{self.pad}"
        return s


_lock = threading.RLock()
_kernels: dict[str, object] = {}
_header_failed = False
# Self-test verdict per kernel instantiation:
# (dtype, mode, bits, group_size, plan, align_n, align_k) -> bool.
_verified: dict[tuple, bool] = {}


def enabled() -> bool:
    """False when ``OMLX_M5_GATHER_QMM_NAX`` disables the module."""
    return os.environ.get(_ENV_ENABLE, "1").strip().lower() not in {
        "0",
        "false",
        "off",
    }


def _get_kernel(kind: str):
    """Build (once) the ``scan``, ``affine`` or ``fp`` kernel object."""
    global _header_failed
    kernel = _kernels.get(kind)
    if kernel is not None or _header_failed:
        return kernel
    with _lock:
        kernel = _kernels.get(kind)
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
                    name="omlx_gqmm_affine_v2",
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
                    header=mlx_src + _MM_HEADER,
                    source=_AFFINE_SOURCE,
                )
            else:
                kernel = mx.fast.metal_kernel(
                    name="omlx_gqmm_mxfp4_v2",
                    input_names=["x", "w", "scales", "tiles", "tile_count", "params"],
                    output_names=["y"],
                    header=mlx_src + _MM_HEADER,
                    source=_FP_SOURCE,
                )
        _kernels[kind] = kernel
        return kernel


def _parse_plan(text: str) -> Optional[Plan]:
    parts = [p.strip().lower() for p in text.split(",")]
    if len(parts) != 5 or parts[0] not in ("seg", "db"):
        return None
    try:
        bm, bk, gx, pad = (int(p) for p in parts[1:])
    except ValueError:
        return None
    sched = _SCHED_SEG if parts[0] == "seg" else _SCHED_DB
    if bm not in _TILE_ROWS or bk not in (64, 128) or gx < 0 or not 0 <= pad <= 8192:
        return None
    return Plan(sched, bm, bk, gx, pad)


def _plan(rows: int, experts: int, K: int, N: int) -> Plan:
    """Kernel configuration for a call (mean rows per expert and K).

    Measured on M5 Ultra (real routing profiles, Qwen3.8 / GLM-5.3 /
    MiMo-V2.6 expert shapes at 1024-8192-token chunks):

    - fewer than 36 rows per expert, or K < 1024 (a short down projection)
      below 120 rows: weight streaming dominates; 64-row db tiles in mlx's
      (column, tile) layout;
    - 36-47 rows: 64-row db tiles, tile-on-x layout (+10%);
    - 48-95 rows: 96-row db tiles, tile-on-x layout (+3-26%);
    - 96+ rows (K >= 1024): 128-row seg tiles with 128-deep K steps and 8 KB
      of extra threadgroup memory (fewer resident threadgroups), tile-on-x
      layout (+5-29%);
    - K < 1024 from 120 rows: 96-row seg tiles, 128-deep K steps (+5%).

    Ragged K or N keeps 64-row seg tiles in the plain layout.
    """
    forced = os.environ.get(_ENV_PLAN, "").strip()
    if forced:
        plan = _parse_plan(forced)
        if plan is not None:
            return plan
    if K % 64 or N % 64:
        return Plan(_SCHED_SEG, 64, 64, 0, 0)
    per_expert = rows / max(1, experts)
    if per_expert < 36 or (K < 1024 and per_expert < 120):
        return Plan(_SCHED_DB, 64, 64, 0, 0)
    if K < 1024:
        return Plan(_SCHED_SEG, 96, 128, _GX, 0)
    if per_expert < 48:
        return Plan(_SCHED_DB, 64, 64, _GX, 0)
    if per_expert < 96:
        return Plan(_SCHED_DB, 96, 64, _GX, 0)
    return Plan(_SCHED_SEG, 128, 128, _GX, 8192)


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


def _launch(x, w, scales, biases, indices, group_size, bits, mode, plan, stream):
    scan = _get_kernel("scan")
    mm = _get_kernel("affine" if mode == "affine" else "fp")
    if scan is None or mm is None:
        return None
    M, K = int(x.shape[0]), int(x.shape[2])
    E, N = int(w.shape[0]), int(w.shape[1])
    bm = plan.bm
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
        ("SCHED", int(plan.sched)),
        ("ALIGN_N", N % _BN == 0),
        ("ALIGN_K", K % plan.bk == 0),
        ("BM", bm),
        ("BK", plan.bk),
        ("GX", plan.gx),
        ("PAD", plan.pad),
    ]
    n_cols = (N + _BN - 1) // _BN
    if plan.gx:
        tg_grid = (plan.gx, ((max_tiles + plan.gx - 1) // plan.gx) * n_cols)
    else:
        tg_grid = (n_cols, max_tiles)
    return mm(
        inputs=inputs,
        template=template,
        grid=(tg_grid[0] * 32, tg_grid[1] * _WN, bm // 32),
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


# Canary routing: an empty expert, runs spanning several tiles of every
# height, and partial tiles of every size class.
_CANARY_COUNTS = (70, 0, 5, 33, 64, 17, 140, 11)


def _self_test(key: tuple) -> Optional[bool]:
    """Run one kernel instantiation on a small canary.

    K % 64 == 0 must be bit-identical to mlx's sorted kernel (correct
    there); ragged K must match an fp32 dequantized reference to bf16
    rounding. Returns None when the canary could not be evaluated here
    (e.g. while a function transformation is being traced); the caller then
    retries.
    """
    dtype, mode, bits, group_size, plan, align_n, align_k = key
    E = len(_CANARY_COUNTS)
    N = 128 if align_n else 96
    # Aligned: K % BK == 0. Unaligned: a 64-deep tail (bk 128, still
    # K % 64 == 0) or a 32-deep ragged one (bk 64).
    K = 256 if align_k else (320 if plan.bk == 128 else 160)
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
        out = _launch(x, wq, scales, biases, idx, group_size, bits, mode, plan, None)
        if out is None:
            return False
        if K % 64 == 0:
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
    dtype, mode, bits, group_size, plan, align_n, align_k = key
    return (
        f"{str(dtype).rsplit('.', 1)[-1]} {mode} {bits}-bit gs{group_size} "
        f"({plan.describe()}{'' if align_n else ', ragged N'}"
        f"{'' if align_k else ', K tail'})"
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
    plan: Optional[Plan] = None,
    verify: bool = True,
) -> Optional[mx.array]:
    """``x @ w[indices].T`` for sorted rows on the tensor units.

    ``plan`` pins a configuration (testing); by default ``_plan`` picks
    one. Returns None when the module is disabled, the call is not
    supported (see ``supports``), the kernels cannot be built or the
    instantiation failed its one-time self-test; the caller then keeps the
    stock path.
    """
    if not enabled() or not supports(
        x, w, scales, biases, indices, group_size, bits, mode
    ):
        return None
    M, K = int(x.shape[0]), int(x.shape[2])
    E, N = int(w.shape[0]), int(w.shape[1])
    if plan is None:
        plan = _plan(M, E, K, N)
    if plan.sched == _SCHED_DB and (K % 64 or N % 64 or plan.bk != 64):
        # db runs aligned 64-deep K steps only.
        plan = plan._replace(sched=_SCHED_SEG)
    if verify:
        key = (x.dtype, mode, bits, group_size, plan, N % _BN == 0, K % plan.bk == 0)
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
    return _launch(x, w, scales, biases, indices, group_size, bits, mode, plan, stream)
