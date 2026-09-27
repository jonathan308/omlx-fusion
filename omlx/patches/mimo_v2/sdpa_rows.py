# SPDX-License-Identifier: Apache-2.0
"""Vector SDPA for a few query rows at once (MiMo decode / MTP verify).

MLX's vector attention (``sdpa_vector_2pass``) runs one threadgroup per
(KV head, key block) holding one simdgroup per (query head of the group,
query row), so it serves ``rows x GQA <= 32`` and MiMo's full-attention
layers (GQA 16) can only take 2 rows per call: a 3-row MTP verify ran as two
calls, each reading the whole KV cache.

``sdpa_rows`` computes every row of a short forward in one pass over the KV
cache.  Each row's arithmetic is MLX's, operation for operation: the query is
scaled in float32, each lane accumulates its ``D / 32`` products in order and
``simd_sum`` reduces them, the online softmax uses ``fast::exp`` key by key,
the keys are split into MLX's ``blocks`` (same heuristic, same strided
partition), the per-block partial outputs are rounded to the input dtype and
the second pass is MLX's ``sdpa_vector_2pass_2``.  The output is therefore
bit-identical to MLX's kernel on the same rows (and to the row-chunked calls
the decode path made before), as long as MLX would pick the same block count
for every row; ``sdpa_rows`` returns ``None`` (caller keeps its current path)
whenever that, or anything else about the call, is not guaranteed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import mlx.core as mx

# Largest query-row count served (MTP verify forwards are 1 + depth rows).
MAX_ROWS = 4

# Pass 1: grid (KV heads, batch, blocks) threadgroups of G simdgroups; the
# simdgroup of query head g (of the group) carries every query row.  Per row
# the loop body is MLX's sdpa_vector_2pass_1 body; K and V rows are loaded
# once per simdgroup and shared by the rows.
_PASS1_SETUP = r"""
  constexpr int BD = 32;
  constexpr int QK = D / BD;
  constexpr int VP = V / BD;
  typedef float U;

  const int kv_head_idx = threadgroup_position_in_grid.x;
  const int batch_idx = threadgroup_position_in_grid.y;
  const int block_idx = threadgroup_position_in_grid.z;
  const int g = simdgroup_index_in_threadgroup;
  const int simd_lid = thread_index_in_simdgroup;
  const int N = keys_shape[2];
  const int num_q_heads = NKV * G;
  const int q_head_idx = G * kv_head_idx + g;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;

  U q[ROWS][QK];
  U o[ROWS][VP];
  U max_score[ROWS];
  U sum_exp_score[ROWS];
  const U sc = scale[0];
  for (int r = 0; r < ROWS; r++) {
    const device T* qp = queries + batch_idx * queries_strides[0] +
        q_head_idx * queries_strides[1] + r * queries_strides[2] +
        simd_lid * QK * queries_strides[3];
    for (int i = 0; i < QK; i++) {
      q[r][i] = static_cast<U>(sc) * qp[i * queries_strides[3]];
    }
    for (int i = 0; i < VP; i++) {
      o[r][i] = 0;
    }
    max_score[r] = Limits<U>::finite_min;
    sum_exp_score[r] = 0;
    if (HAS_SINKS && block_idx == 0) {
      max_score[r] = static_cast<U>(sinks[q_head_idx]);
      sum_exp_score[r] = 1;
    }
  }

  const int64_t ks = keys_strides[2];
  const int64_t vs = values_strides[2];
  const int64_t k3 = keys_strides[3];
  const int64_t v3 = values_strides[3];
  const device T* kp = keys + batch_idx * keys_strides[0] +
      kv_head_idx * keys_strides[1] + block_idx * ks + simd_lid * QK * k3;
  const device T* vp = values + batch_idx * values_strides[0] +
      kv_head_idx * values_strides[1] + block_idx * vs + simd_lid * VP * v3;
  auto mp = mask + (MASK_KIND ? (size_t(batch_idx) * ROWS * N) : 0);

"""

# Per key of the block: MLX's loop body for every row, with the next key
# of the block loaded while the current one is processed.
_PASS1_LOOP = r"""#define ROW_STEP(KR, VR, KEY)                                             \
    for (int r = 0; r < ROWS; r++) {                                        \
      bool use_key = true;                                                  \
      if (CAUSAL) {                                                         \
        use_key = (KEY) <= (N - ROWS + r);                                  \
      } else if (MASK_KIND == 1) {                                          \
        use_key = mp[r * N + (KEY)];                                        \
      } else if (MASK_KIND == 2) {                                          \
        use_key = (mp[r * N + (KEY)] >= Limits<T>::finite_min);             \
      }                                                                     \
      if (use_key) {                                                        \
        U score = 0;                                                        \
        for (int j = 0; j < QK; j++) {                                      \
          score += q[r][j] * KR[j];                                         \
        }                                                                   \
        score = simd_sum(score);                                            \
        if (MASK_KIND == 2) {                                               \
          score += mp[r * N + (KEY)];                                       \
        }                                                                   \
        U new_max = max(max_score[r], score);                               \
        U factor = fast::exp(max_score[r] - new_max);                       \
        U exp_score = fast::exp(score - new_max);                           \
        max_score[r] = new_max;                                             \
        sum_exp_score[r] = sum_exp_score[r] * factor + exp_score;           \
        for (int j = 0; j < VP; j++) {                                      \
          o[r][j] = o[r][j] * factor + exp_score * VR[j];                   \
        }                                                                   \
      }                                                                     \
    }

#define LOAD_KV(KR, VR, K3, V3)                                           \
    for (int j = 0; j < QK; j++) {                                          \
      KR[j] = kp[j * (K3)];                                                 \
    }                                                                       \
    for (int j = 0; j < VP; j++) {                                          \
      VR[j] = vp[j * (V3)];                                                 \
    }

// Next key of the block loaded while the current one is processed.
#define PASS1_LOOP(K3, V3)                                                \
  {                                                                         \
    T ka[QK], kb[QK];                                                       \
    T va[VP], vb[VP];                                                       \
    int key = block_idx;                                                    \
    if (n_iter > 0) {                                                       \
      LOAD_KV(ka, va, K3, V3)                                               \
    }                                                                       \
    int it = 0;                                                             \
    for (; it + 1 < n_iter; it += 2) {                                      \
      kp += BLOCKS * ks;                                                    \
      vp += BLOCKS * vs;                                                    \
      LOAD_KV(kb, vb, K3, V3)                                               \
      ROW_STEP(ka, va, key)                                                 \
      key += BLOCKS;                                                        \
      if (it + 2 < n_iter) {                                                \
        kp += BLOCKS * ks;                                                  \
        vp += BLOCKS * vs;                                                  \
        LOAD_KV(ka, va, K3, V3)                                             \
      }                                                                     \
      ROW_STEP(kb, vb, key)                                                 \
      key += BLOCKS;                                                        \
    }                                                                       \
    if (it < n_iter) {                                                      \
      ROW_STEP(ka, va, key)                                                 \
    }                                                                       \
  }

  const int n_iter = block_idx < N ? (N - 1 - block_idx) / BLOCKS + 1 : 0;
  if (k3 == 1 && v3 == 1) {
    PASS1_LOOP(1, 1)
  } else {
    PASS1_LOOP(k3, v3)
  }

"""

_PASS1_WRITEBACK = r"""  for (int r = 0; r < ROWS; r++) {
    const int o_offset = q_batch_head_idx * ROWS + r;
    if (simd_lid == 0) {
      sums[o_offset * BLOCKS + block_idx] = sum_exp_score[r];
      maxs[o_offset * BLOCKS + block_idx] = max_score[r];
    }
    device T* op = partials + (size_t(o_offset) * BLOCKS + block_idx) * V + simd_lid * VP;
    for (int j = 0; j < VP; j++) {
      op[j] = static_cast<T>(o[r][j]);
    }
  }
"""


# Pass 2: MLX's sdpa_vector_2pass_2 (1024 threads per (head, row)), writing
# the (B, L, H, V) layout the output projection reads.
_PASS2_SOURCE = r"""
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = V / BD;
  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  const int head_idx = threadgroup_position_in_grid.x;
  const int q_seq_idx = threadgroup_position_in_grid.y;
  const int simd_gid = simdgroup_index_in_threadgroup;
  const int simd_lid = thread_index_in_simdgroup;
  const int q_offset = head_idx * ROWS + q_seq_idx;
  const device T* pp = partials + size_t(q_offset) * BLOCKS * V + simd_gid * V +
      simd_lid * elem_per_thread;
  const device float* sp = sums + q_offset * BLOCKS;
  const device float* mp = maxs + q_offset * BLOCKS;
  const int bi = head_idx / NQ;
  const int hi = head_idx % NQ;
  device T* op = out + ((size_t(bi) * ROWS + q_seq_idx) * NQ + hi) * V +
      simd_gid * elem_per_thread;

  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  for (int b = 0; b < BLOCKS / BN; ++b) {
    max_score = max(max_score, mp[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  for (int b = 0; b < BLOCKS / BN; ++b) {
    U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sp[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  for (int b = 0; b < BLOCKS / BN; ++b) {
    U factor = fast::exp(mp[simd_gid] - max_score);
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(pp[i]);
    }
    mp += BN;
    sp += BN;
    pp += BN * V;
  }

  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      op[i] = static_cast<T>(o[i]);
    }
  }
"""


@lru_cache(maxsize=None)
def _pass1_kernel():
    return mx.fast.metal_kernel(
        name="omlx_sdpa_rows_pass1",
        input_names=["queries", "keys", "values", "scale", "mask", "sinks"],
        output_names=["partials", "sums", "maxs"],
        source=_PASS1_SETUP + _PASS1_LOOP + _PASS1_WRITEBACK,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=None)
def _pass2_kernel():
    return mx.fast.metal_kernel(
        name="omlx_sdpa_rows_pass2",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=_PASS2_SOURCE,
    )


@lru_cache(maxsize=None)
def _device_class() -> str:
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        return str(info.get("architecture", ""))[-1:]
    except Exception:  # noqa: BLE001
        return ""


def mlx_blocks(n_keys: int, n_simds: int, devc: Optional[str] = None) -> int:
    """MLX 0.32's ``sdpa_vector_2pass`` key-block count."""
    devc = _device_class() if devc is None else devc
    N = n_keys
    if devc == "s":
        blocks = 64
        if N > 1024 and n_simds > 4:
            if N <= 8192:
                blocks = 128
            elif N <= 32768:
                blocks = 256
            elif N <= 65536:
                blocks = 512
            else:
                blocks = 1024
    elif devc == "d":
        blocks = 128
        if n_simds <= 2 and N > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= N < 65536:
                blocks = 512
            elif N >= 65536:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    env = os.environ.get("MLX_SDPA_BLOCKS", "")
    if env:
        try:
            value = int(env)
        except ValueError:
            value = 0
        if value > 0:
            blocks = ((value + 31) // 32) * 32
    return blocks


def mlx_uses_2pass(n_q_heads: int, n_kv_heads: int, n_keys: int, devc: Optional[str] = None) -> bool:
    """Whether MLX routes a vector (<= 8 row) SDPA call to the 2-pass kernel."""
    devc = _device_class() if devc is None else devc
    return ((devc in ("d", "s")) and n_keys >= 1024) or (
        n_kv_heads < n_q_heads and n_keys >= 4096
    )


def _mlx_call_rows(L: int, gqa: int):
    """Row chunks the MiMo decode path hands MLX: one call when rows x GQA
    fits the vector kernel, else chunks of ``32 // gqa`` rows."""
    if L * gqa <= 32:
        return [(0, L)]
    step = max(1, 32 // gqa)
    return [(r0, min(L, r0 + step)) for r0 in range(0, L, step)]


def plan_blocks(L: int, H: int, Hk: int, S: int, causal: bool) -> Optional[int]:
    """MLX's block count if every row of this forward would get the same one
    from the 2-pass kernel (today's calls), else ``None``."""
    gqa = H // Hk
    blocks = None
    for r0, r1 in _mlx_call_rows(L, gqa):
        n = S - (L - r1) if causal else S
        if not mlx_uses_2pass(H, Hk, n):
            return None
        b = mlx_blocks(n, gqa * (r1 - r0))
        if blocks is None:
            blocks = b
        elif b != blocks:
            return None
    return blocks


@lru_cache(maxsize=None)
def _scale_array(scale: float):
    return mx.array([scale], dtype=mx.float32)


_DUMMY = {}


def _dummy(dtype):
    arr = _DUMMY.get(dtype)
    if arr is None:
        arr = _DUMMY[dtype] = mx.zeros((1,), dtype=dtype)
    return arr


def sdpa_rows(q, k, v, scale: float, mask, sinks) -> Optional[mx.array]:
    """Attention of ``q (B, H, L, D)`` over ``k (B, Hk, S, D)`` /
    ``v (B, Hk, S, Dv)`` for ``L <= MAX_ROWS`` rows, returned as
    ``(B, L, H * Dv)``; bit-identical to MLX's vector kernel on these rows.

    ``mask`` is ``None``, ``"causal"`` or a bool / additive array broadcastable
    to ``(B, 1, L, S)``.  Returns ``None`` when the call is outside this
    kernel's contract (the caller keeps its current path).
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    B, H, L, D = q.shape
    Hk, S = k.shape[1], k.shape[2]
    Dv = v.shape[3]
    if not (1 <= L <= MAX_ROWS) or k.shape[0] != B or v.shape[0] != B:
        return None
    if v.shape[1] != Hk or v.shape[2] != S or k.shape[3] != D or Hk == 0 or H % Hk:
        return None
    if D % 32 or Dv % 32 or D > 256 or Dv > 256 or L > S:
        return None
    dtype = q.dtype
    if dtype not in (mx.bfloat16, mx.float16) or k.dtype != dtype or v.dtype != dtype:
        return None
    gqa = H // Hk
    if gqa > 32:
        return None
    causal = isinstance(mask, str)
    if causal and mask != "causal":
        return None
    # MLX's single-row GQA-8 variant has its own arithmetic.
    if (mask is None and sinks is None and L == 1 and H == 8 * Hk and D == Dv
            and D in (64, 128) and S >= 8192):
        return None
    blocks = plan_blocks(L, H, Hk, S, causal and L > 1)
    if blocks is None:
        return None

    mask_kind = 0
    mask_arr = _dummy(mx.bool_)
    if mask is not None and not causal:
        if not isinstance(mask, mx.array) or mask.ndim < 1 or mask.ndim > 4:
            return None
        m = mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))
        if m.shape[1] != 1:
            return None
        try:
            m = mx.broadcast_to(m, (B, 1, L, S))
        except ValueError:
            return None
        if m.dtype == mx.bool_:
            mask_kind = 1
        else:
            mask_kind = 2
            m = m.astype(dtype)
        mask_arr = mx.contiguous(m.reshape(B, L, S))
    sinks_arr = _dummy(dtype)
    if sinks is not None:
        if sinks.ndim != 1 or sinks.shape[0] != H:
            return None
        sinks_arr = sinks.astype(dtype)

    partials, sums, maxs = _pass1_kernel()(
        inputs=[q, k, v, _scale_array(float(scale)), mask_arr, sinks_arr],
        template=[
            ("T", dtype),
            ("MT", mask_arr.dtype),
            ("D", int(D)),
            ("V", int(Dv)),
            ("G", int(gqa)),
            ("NKV", int(Hk)),
            ("ROWS", int(L)),
            ("BLOCKS", int(blocks)),
            ("CAUSAL", bool(causal and L > 1)),
            ("MASK_KIND", int(mask_kind)),
            ("HAS_SINKS", sinks is not None),
        ],
        grid=(32 * Hk, gqa * B, blocks),
        threadgroup=(32, gqa, 1),
        output_shapes=[(B * H * L, blocks, Dv), (B * H * L, blocks), (B * H * L, blocks)],
        output_dtypes=[dtype, mx.float32, mx.float32],
    )
    (out,) = _pass2_kernel()(
        inputs=[partials, sums, maxs],
        template=[
            ("T", dtype),
            ("V", int(Dv)),
            ("NQ", int(H)),
            ("ROWS", int(L)),
            ("BLOCKS", int(blocks)),
        ],
        grid=(1024 * B * H, L, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(B, L, H * Dv)],
        output_dtypes=[dtype],
    )
    return out
