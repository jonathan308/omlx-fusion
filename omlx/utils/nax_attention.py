# SPDX-License-Identifier: Apache-2.0
#
# The attention kernel below is MLX's NAX flash-attention kernel
# (mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h,
# Copyright (c) 2024-25 Apple Inc., MIT License,
# https://github.com/ml-explore/mlx) with a separate value head dim.
"""Fused tensor-unit (NAX) prefill attention for a narrower value head.

MLA-style models such as MiMo-V2-Flash use query/key head dim 192 with value
head dim 128. MLX 0.32.2 has no fused kernel for that pair: its prefill SDPA
falls back to the unfused path (bf16 score matrix materialised in memory),
and on M5 (NAX) GPUs its tensor-unit attention kernel only exists for head
dims 64/96/128/256. ``mixed_head_dim_sdpa`` therefore zero-pads Q/K/V to 256
to reach the head-dim-split NAX kernel, which spends a third of its
multiply-adds on zero columns and copies Q/K/V every layer.

This module JIT-compiles (``mx.fast.metal_kernel``) MLX's own NAX attention
kernel with the value head dim as a separate template parameter (BD = 192,
BDV = 128): the query/key loop runs over 192 dims and the output tile and
the P @ V loop over 128, so no padding and no copies are needed. The kernel
body is MLX's (same tiles, same online softmax in fp32, same MPP matmul
calls); the MLX-side function constants (alignment, mask, causal, sinks) are
compile-time constants of one generated kernel per variant. On MLX builds
that carry the native kernel, the output is bit-identical to it.

Inputs are read through their strides (no contiguity copies: KV-cache
slices, transposed projections and the strided key windows of the blocked
sliding-window path are consumed in place); only the head dim must be
contiguous, as for MLX's SDPA. The output is written in MLX's SDPA layout
(``[B, L, H, V]`` rows returned as a ``[B, H, L, V]`` view), so the caller's
``swapaxes(1, 2).reshape(B, L, -1)`` stays free.

Fail-closed: only (192, 128) bf16/fp16 prefill on NAX GPUs is handled, the
first use runs a small self-check against an fp32 reference, and any
unsupported input returns None so the caller keeps its existing route.
Kill switch: ``OMLX_NAX_JIT_ATTENTION=0``.
"""

from __future__ import annotations

import logging
import os
import struct
from functools import lru_cache
from typing import Optional

import mlx.core as mx

from omlx.custom_kernels.nax_tiles import NAX_TILE_HEADER

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("OMLX_NAX_JIT_ATTENTION", "1").strip().lower() not in {
    "0",
    "false",
    "off",
}

# (query/key head dim, value head dim) pairs this kernel is validated for.
SUPPORTED_HEAD_DIMS = frozenset({(192, 128)})

_BQ = 64
_BK = 32
_WM = 4
_THREADS = 32 * _WM

_METAL_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "half"}

_HEADER = NAX_TILE_HEADER + r"""
namespace omlx_nax {

// Scalar parameters of one attention call (MLX's AttnParams without the
// strides, which the kernel reads from the inputs' own stride vectors).
struct AttnParams {
  int B;
  int H;
  int D;
  int qL;
  int kL;
  int gqa_factor;
  float scale;
  int NQ;
  int NK;
  int NQ_aligned;
  int NK_aligned;
  int qL_rem;
  int kL_rem;
  int qL_off;
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

// MLX's attention_nax with a value head dim BDV <= BD. Only the parameter
// plumbing differs: strides come from the inputs, the output is written as
// [B, qL, H, BDV] rows (MLX's SDPA output layout), function constants are
// template arguments, and the mask column stride is honoured.
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int BDV,
    int WM,
    int WN,
    bool align_Q,
    bool align_K,
    bool has_mask,
    bool do_causal,
    bool has_sinks,
    typename MaskType,
    typename AccumType,
    typename StridePtr,
    typename MaskPtr,
    typename SinkPtr>
METAL_FUNC void attention_nax_bdv(
    const device T* Q,
    const device T* K,
    const device T* V,
    device T* O,
    const device AttnParams* params,
    StridePtr q_str,
    StridePtr k_str,
    StridePtr v_str,
    StridePtr m_str,
    MaskPtr mask,
    SinkPtr sinks,
    uint simd_group_id,
    uint3 tid) {
  ulong3 tidl{tid.x, tid.y, tid.z};

  const int64_t Q_strides[3] = {q_str[0], q_str[1], q_str[2]};
  const int64_t K_strides[3] = {k_str[0], k_str[1], k_str[2]};
  const int64_t V_strides[3] = {v_str[0], v_str[1], v_str[2]};
  const int64_t O_strides[3] = {
      int64_t(params->qL) * params->H * BDV, BDV, int64_t(params->H) * BDV};

  Q += tidl.z * Q_strides[0] + // Batch
      tidl.y * Q_strides[1] + // Head
      tidl.x * BQ * Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * K_strides[0] + // Batch
      kv_head_idx * K_strides[1]; // Head

  V += tidl.z * V_strides[0] + // Batch
      kv_head_idx * V_strides[1]; // Head

  O += tidl.z * O_strides[0] + // Batch
      tidl.y * O_strides[1] + // Head
      tidl.x * BQ * O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * m_str[0] + // Batch
        tidl.y * m_str[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kU;
  // Value head dim frags
  constexpr int TDV = BDV / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  static_assert(BDV % (2 * kU) == 0, "BDV must be a multiple of 32");
  using otile_t = NAXTile<AccumType, TQ, TDV>;
  otile_t Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short tm = kU * TQ * simd_group_id;
  Q += tm * int(Q_strides[2]);

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Init row reduction variables
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  // Init to -Inf
  OMLX_NAX_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    // Do S = Q @ K.T
    using stile_t = NAXTile<AccumType, TQ, TK>;
    stile_t Stile;

    Stile.clear();

    OMLX_NAX_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      OMLX_NAX_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;

          const int Q_load_off = iq * kU * int(Q_strides[2]) + id * kU;
          const int K_load_off = ik * kU * int(K_strides[2]) + id * kU;

          if (!align_Q && is_last_q) {
            Qtile.load_rows(
                Q + Q_load_off, int(Q_strides[2]), lim_rows_q - iq * kU);
          } else {
            Qtile.load(Q + Q_load_off, int(Q_strides[2]));
          }

          if (!align_K && is_last_k) {
            Ktile.load_rows(
                K + K_load_off, int(K_strides[2]), lim_rows_k - ik * kU);
          } else {
            Ktile.load(K + K_load_off, int(K_strides[2]));
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale S
    OMLX_NAX_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      OMLX_NAX_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;

          thread auto& fg = Stile.frag_at(iq, ik);

          OMLX_NAX_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            OMLX_NAX_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      OMLX_NAX_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          OMLX_NAX_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            OMLX_NAX_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + BQ <= params->qL && base_col + BK <= params->kL) {
        for (short iq = 0; iq < TQ; iq++) {
          OMLX_NAX_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load(
                mfrag,
                mask,
                int64_t(m_str[2]),
                int64_t(m_str[3]),
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            OMLX_NAX_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      } else {
        OMLX_NAX_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          OMLX_NAX_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load_safe(
                mfrag,
                mask,
                int64_t(m_str[2]),
                int64_t(m_str[3]),
                params->qL,
                params->kL,
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            OMLX_NAX_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      }
    }

    // Do softmax

    // Temp variables
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    // Row Sum
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // Do O = P @ V
    OMLX_NAX_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      OMLX_NAX_UNROLL
      for (short id = 0; id < TDV; id += 2) {
        if constexpr (BDV == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }

        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;

          const int V_load_off = ik * kU * int(V_strides[2]) + id * kU;

          if (!align_K && is_last_k) {
            Vtile.load_rows(
                V + V_load_off, int(V_strides[2]), lim_rows_k - ik * kU);
          } else {
            Vtile.load(V + V_load_off, int(V_strides[2]));
          }

          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    // Prepare for next iteration
    K += BK * int(K_strides[2]);
    V += BK * int(V_strides[2]);
  }

  // Normalize output

  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  OMLX_NAX_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  // Store results
  O += tm * int(O_strides[2]);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    Otile.store_rows(O, int(O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(O_strides[2]));
  }
}

} // namespace omlx_nax
"""

# One generated kernel per variant; the MLX kernel's function constants are
# baked in as template arguments of the call.
_SOURCE = r"""
  omlx_nax::attention_nax_bdv<
      {T}, {BQ}, {BK}, {BD}, {BDV}, {WM}, 1,
      {ALIGN_Q}, {ALIGN_K}, {HAS_MASK}, {DO_CAUSAL}, {HAS_SINKS}, bool, float>(
      q, k, v, out,
      reinterpret_cast<const device omlx_nax::AttnParams*>(params),
      q_strides, k_strides, v_strides, mask_strides,
      mask, sinks,
      simdgroup_index_in_threadgroup,
      threadgroup_position_in_grid);
"""


def _flag(value: bool) -> str:
    return "true" if value else "false"


@lru_cache(maxsize=None)
def _kernel(
    dtype: mx.Dtype,
    bd: int,
    bdv: int,
    align_q: bool,
    align_k: bool,
    has_mask: bool,
    do_causal: bool,
    has_sinks: bool,
):
    source = (
        _SOURCE.replace("{T}", _METAL_TYPES[dtype])
        .replace("{BQ}", str(_BQ))
        .replace("{BK}", str(_BK))
        .replace("{BD}", str(bd))
        .replace("{BDV}", str(bdv))
        .replace("{WM}", str(_WM))
        .replace("{ALIGN_Q}", _flag(align_q))
        .replace("{ALIGN_K}", _flag(align_k))
        .replace("{HAS_MASK}", _flag(has_mask))
        .replace("{DO_CAUSAL}", _flag(do_causal))
        .replace("{HAS_SINKS}", _flag(has_sinks))
    )
    tag = "".join(
        "1" if f else "0" for f in (align_q, align_k, has_mask, do_causal, has_sinks)
    )
    return mx.fast.metal_kernel(
        name=f"omlx_nax_attention_bd{bd}_bdv{bdv}_{tag}",
        input_names=["q", "k", "v", "mask", "sinks", "params"],
        output_names=["out"],
        header=_HEADER,
        source=source,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=1)
def _nax_available() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


def _run(q, k, v, scale, mask, sinks) -> mx.array:
    B, H, qL, D = q.shape
    kL = k.shape[2]
    DV = v.shape[3]
    do_causal = isinstance(mask, str)
    has_mask = isinstance(mask, mx.array)
    NQ = (qL + _BQ - 1) // _BQ
    NK = (kL + _BK - 1) // _BK
    NQ_aligned = qL // _BQ
    NK_aligned = kL // _BK
    params = struct.pack(
        "<6if7i",
        B,
        H,
        D,
        qL,
        kL,
        H // k.shape[1],
        float(scale),
        NQ,
        NK,
        NQ_aligned,
        NK_aligned,
        qL - NQ_aligned * _BQ,
        kL - NK_aligned * _BK,
        kL - qL,
    )
    params = mx.array(memoryview(params), dtype=mx.uint8)
    has_sinks = sinks is not None
    # Unused inputs get a one-element placeholder (never read).
    if has_mask:
        mask = mx.broadcast_to(mask, (B, H, qL, kL))
    else:
        mask = mx.zeros((1,), dtype=mx.bool_)
    if has_sinks:
        sinks = mx.contiguous(sinks.astype(q.dtype))
    else:
        sinks = mx.zeros((1,), dtype=q.dtype)
    kernel = _kernel(
        q.dtype,
        D,
        DV,
        qL % _BQ == 0,
        kL % _BK == 0,
        has_mask,
        do_causal,
        has_sinks,
    )
    out = kernel(
        inputs=[q, k, v, mask, sinks, params],
        grid=(NQ * _THREADS, H, B),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(B, qL, H, DV)],
        output_dtypes=[q.dtype],
    )[0]
    # [B, qL, H, DV] rows viewed as [B, H, qL, DV], like MLX's SDPA output.
    return out.transpose(0, 2, 1, 3)


def _reference(q, k, v, scale, mask):
    """fp32 causal attention (for the self-check)."""
    B, H, qL, _ = q.shape
    Hk, kL = k.shape[1], k.shape[2]
    g = H // Hk
    qf = q.astype(mx.float32).reshape(B, Hk, g, qL, -1) * scale
    s = qf @ k.astype(mx.float32)[:, :, None].swapaxes(-1, -2)
    causal = (mx.arange(qL)[:, None] + (kL - qL)) >= mx.arange(kL)[None]
    s = mx.where(causal, s, -mx.inf)
    p = mx.softmax(s, axis=-1)
    return (p @ v.astype(mx.float32)[:, :, None]).reshape(B, H, qL, -1)


@lru_cache(maxsize=1)
def _self_check_passed() -> bool:
    """Compile and run one small case against fp32 once per process.

    Uses the causal variant with unaligned query/key tails, the one a
    typical first prefill chunk (L - 1 prompt tokens) needs anyway.
    """
    try:
        key = mx.random.key(192128)
        kq, kk, kv = mx.random.split(key, 3)
        q = (0.5 * mx.random.normal((1, 4, 100, 192), key=kq)).astype(mx.bfloat16)
        k = (0.5 * mx.random.normal((1, 2, 300, 192), key=kk)).astype(mx.bfloat16)
        v = (0.5 * mx.random.normal((1, 2, 300, 128), key=kv)).astype(mx.bfloat16)
        scale = 192**-0.5
        out = _run(q, k, v, scale, "causal", None)
        ref = _reference(q, k, v, scale, "causal")
        err = mx.abs(out.astype(mx.float32) - ref).max().item()
        ok = err < 2e-2
    except Exception as exc:  # noqa: BLE001 - any failure disables the route
        logger.warning("NAX JIT attention disabled: self-check failed (%s)", exc)
        return False
    if not ok:
        logger.warning(
            "NAX JIT attention disabled: self-check error %.3g vs fp32", err
        )
        return False
    logger.info("NAX JIT attention (192/128 head dims) enabled")
    return True


def nax_mixed_head_dim_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask=None,
    sinks: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Fused NAX attention for ``qk_dim > v_dim`` prefill; None if unsupported.

    ``queries`` [B, H, L, 192], ``keys`` [B, Hk, S, 192], ``values``
    [B, Hk, S, 128] (bf16 or fp16, any strides with a contiguous head dim);
    ``mask`` None, ``"causal"`` (bottom-right aligned, as in MLX) or a
    boolean array broadcastable to [B, H, L, S]; ``sinks`` [H] or None.
    Returns [B, H, L, 128] in the query dtype.
    """
    if not _ENABLED:
        return None
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return None
    B, H, qL, qk_dim = queries.shape
    _, Hk, kL, k_dim = keys.shape
    v_dim = values.shape[-1]
    if (
        (qk_dim, v_dim) not in SUPPORTED_HEAD_DIMS
        or k_dim != qk_dim
        or queries.dtype not in _METAL_TYPES
        or keys.dtype != queries.dtype
        or values.dtype != queries.dtype
        or keys.shape[0] != B
        or tuple(values.shape[:3]) != (B, Hk, kL)
        or Hk == 0
        or H % Hk
        or qL <= 8
        or kL == 0
    ):
        return None
    if isinstance(mask, str):
        if mask != "causal":
            return None
    elif mask is not None:
        if (
            not isinstance(mask, mx.array)
            or mask.dtype != mx.bool_
            or mask.ndim > 4
            or mask.ndim < 1
        ):
            return None
        try:
            if mx.broadcast_shapes(mask.shape, (B, H, qL, kL)) != (B, H, qL, kL):
                return None
        except ValueError:
            return None
    if sinks is not None and (sinks.ndim != 1 or sinks.shape[0] != H):
        return None
    if not _nax_available() or not _self_check_passed():
        return None
    return _run(queries, keys, values, scale, mask, sinks)
