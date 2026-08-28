"""Exact gathered QSA prefill for contiguous batch-one text prompts.

This is the portable MLX path proven by Fusion's native Qwen4-Exp bring-up.
It never constructs the full ``[query_tokens, key_tokens]`` main-attention
matrix: QSA selects complete four-token micro-blocks, then main attention
gathers only the selected K/V rows (plus the incomplete causal tail).

The caller owns eligibility.  In particular, this module is only used for a
single contiguous text prompt.  Batched, padded, multimodal and target-verify
requests stay on mlx-vlm's general QSA implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mlx.core as mx


IndexKeyNorm = Callable[[mx.array], mx.array]
IndexRoPE = Callable[[mx.array, mx.array], mx.array]


_NATIVE_QSA_SCORE_DISABLED = False
_NATIVE_QSA_SCORE_PROVEN = False


def contiguous_causal_query_chunk(key_tokens: int) -> int:
    """Keep long-context score sheets bounded without tiny launch overhead."""

    if key_tokens <= 4096:
        return 32
    if key_tokens <= 16384:
        return 64
    return 128


def _batch_gather_tokens(values: mx.array, indices: mx.array) -> mx.array:
    """Gather token rows independently for every batch without a host read."""

    batch, tokens = values.shape[:2]
    trailing = values.shape[2:]
    offset_shape = (batch,) + (1,) * (indices.ndim - 1)
    offsets = mx.arange(batch, dtype=mx.int32).reshape(offset_shape) * tokens
    flat_indices = (indices.astype(mx.int32) + offsets).reshape(-1)
    flat_values = values.reshape(batch * tokens, *trailing)
    return flat_values[flat_indices].reshape(*indices.shape, *trailing)


def _portable_indexer_scores(
    queries: mx.array,
    pooled_keys: mx.array,
    head_dim: int,
) -> mx.array:
    """Current float32 MLX QSA score reference."""

    scores = queries.astype(mx.float32) @ pooled_keys[:, None].astype(
        mx.float32
    ).swapaxes(-1, -2)
    return mx.sum(mx.maximum(scores, 0), axis=-2) / math.sqrt(head_dim)


def _native_indexer_scores(
    queries: mx.array,
    pooled_keys: mx.array,
    *,
    head_dim: int,
    compress_ratio: int,
    mask_q_offset: int,
) -> mx.array | None:
    """Use the narrow native M3 score ABI or fail closed to the MLX path."""

    global _NATIVE_QSA_SCORE_DISABLED, _NATIVE_QSA_SCORE_PROVEN
    if _NATIVE_QSA_SCORE_DISABLED:
        return None
    if (
        queries.ndim != 4
        or queries.shape[0] != 1
        or queries.shape[-2:] != (4, 128)
        or pooled_keys.ndim != 3
        or pooled_keys.shape[0] != 1
        or pooled_keys.shape[-1] != 128
        or queries.dtype != pooled_keys.dtype
        or queries.dtype not in {mx.float16, mx.bfloat16}
        or head_dim != 128
        or compress_ratio != 4
        or mask_q_offset < 0
    ):
        return None

    try:
        from omlx.custom_kernels.glm_moe_dsa import fast

        if not fast.is_native_available() or not fast.has_symbol(
            "qwen4_qsa_indexer_scores"
        ):
            _NATIVE_QSA_SCORE_DISABLED = True
            return None
        # The caller's [B,M,H,D] view transposes back to the GEMM-friendly
        # [B,H,M,D] ABI. The native wrapper only copies when the resulting
        # view is not row-contiguous (for example an offset query chunk).
        scores = fast.qwen4_qsa_indexer_scores(
            queries.transpose(0, 2, 1, 3),
            pooled_keys[:, None],
            mask_ratio=compress_ratio,
            mask_q_offset=mask_q_offset,
        )
        if not _NATIVE_QSA_SCORE_PROVEN:
            # MLX primitives are lazy, so a missing Metal pipeline would not
            # otherwise surface until the enclosing attention graph is
            # evaluated and can no longer fall back. Pay one process-wide
            # synchronization to prove the extension/pipeline pair.
            mx.eval(scores)
            _NATIVE_QSA_SCORE_PROVEN = True
        return scores
    except Exception:
        # A stale binary or rejected ABI should cost one attempt per process.
        # Shape misses were excluded above and remain eligible on later calls.
        _NATIVE_QSA_SCORE_DISABLED = True
        return None


def contiguous_causal_gathered_qsa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    index_queries: mx.array,
    index_keys: mx.array,
    index_position_ids: mx.array,
    *,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    indexer_head_dim: int,
    compress_ratio: int,
    token_budget: int,
    index_key_norm: IndexKeyNorm,
    apply_index_rope: IndexRoPE,
    query_chunk: int | None = None,
) -> mx.array:
    """Run exact QSA over gathered K/V for one contiguous causal prompt.

    ``queries`` and ``keys`` must already carry their main-attention RoPE.
    ``index_queries`` must likewise be normalized and RoPE-rotated.  Raw
    indexer keys remain unrotated because Qwen pools each complete micro-block
    before applying its checkpoint k-norm and the block-start RoPE.
    """

    if queries.ndim != 4 or queries.shape[0] != 1 or queries.shape[2] <= 1:
        raise ValueError(
            "gathered QSA requires rank-four batch-one multi-token queries"
        )
    batch, actual_query_heads, query_tokens, actual_head_dim = queries.shape
    if actual_query_heads != num_query_heads or actual_head_dim != head_dim:
        raise ValueError("QSA queries do not match the configured geometry")
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("QSA keys and values must be matching rank-four arrays")
    if keys.shape[0] != batch or keys.shape[1:] != (
        num_key_value_heads,
        keys.shape[2],
        head_dim,
    ):
        raise ValueError("QSA K/V do not match the configured geometry")
    key_tokens = keys.shape[2]
    if query_tokens > key_tokens:
        raise ValueError("QSA query length cannot exceed cached key length")
    if index_queries.ndim != 4 or index_queries.shape[:2] != (
        batch,
        query_tokens,
    ):
        raise ValueError("QSA index queries do not match the current prompt")
    if index_queries.shape[-1] != indexer_head_dim:
        raise ValueError("QSA index queries have the wrong head dimension")
    if index_keys.shape != (batch, key_tokens, indexer_head_dim):
        raise ValueError("QSA raw index keys do not match cached K/V")
    if (
        index_position_ids.ndim not in {2, 3}
        or index_position_ids.shape[-1] != key_tokens
    ):
        raise ValueError("QSA index positions do not match cached K/V")
    if compress_ratio <= 0 or token_budget <= 0 or token_budget % compress_ratio:
        raise ValueError("QSA token budget must contain complete micro-blocks")
    if num_query_heads % num_key_value_heads:
        raise ValueError("QSA query heads must divide evenly over K/V heads")

    if query_chunk is None:
        query_chunk = contiguous_causal_query_chunk(key_tokens)
    if query_chunk <= 0:
        raise ValueError("QSA query chunk must be positive")

    ratio = compress_ratio
    max_blocks = key_tokens // ratio
    block_budget = token_budget // ratio
    query_start = key_tokens - query_tokens
    key_rows = keys.transpose(0, 2, 1, 3)
    value_rows = values.transpose(0, 2, 1, 3)

    # A contiguous prompt shares the same block bank for every query.  Pool,
    # normalize and rotate it once instead of once per query row.
    if max_blocks:
        block_starts = mx.arange(max_blocks, dtype=mx.int32) * ratio
        pooled = index_keys[:, : max_blocks * ratio].reshape(
            batch, max_blocks, ratio, indexer_head_dim
        )
        pooled = mx.mean(pooled.astype(mx.float32), axis=-2).astype(index_keys.dtype)
        pooled = index_key_norm(pooled)
        pooled_positions = index_position_ids[..., block_starts]
        pooled = apply_index_rope(pooled[:, None], pooled_positions)[:, 0]
    else:
        pooled = None

    outputs: list[mx.array] = []
    groups = num_query_heads // num_key_value_heads
    for start in range(0, query_tokens, query_chunk):
        stop = min(start + query_chunk, query_tokens)
        chunk_tokens = stop - start
        absolute_queries = query_start + mx.arange(start, stop, dtype=mx.int32)
        visible_counts = mx.broadcast_to(
            (absolute_queries + 1)[None], (batch, chunk_tokens)
        )
        complete_counts = visible_counts // ratio

        if max_blocks:
            chunk_index_queries = index_queries[:, start:stop]
            block_scores = _native_indexer_scores(
                chunk_index_queries,
                pooled,
                head_dim=indexer_head_dim,
                compress_ratio=ratio,
                mask_q_offset=query_start + start,
            )
            if block_scores is None:
                block_scores = _portable_indexer_scores(
                    chunk_index_queries,
                    pooled,
                    indexer_head_dim,
                )
                valid_blocks = (
                    mx.arange(max_blocks)[None, None, :]
                    < complete_counts[..., None]
                )
                block_scores = mx.where(
                    valid_blocks,
                    block_scores,
                    mx.finfo(block_scores.dtype).min,
                )

            selected_width = min(max_blocks, block_budget)
            canonical = mx.broadcast_to(
                mx.arange(selected_width, dtype=mx.int32)[None, None],
                (batch, chunk_tokens, selected_width),
            )
            if max_blocks > block_budget:
                ranked = mx.argpartition(
                    block_scores,
                    kth=-block_budget,
                    axis=-1,
                )[..., -block_budget:].astype(mx.int32)
                selected_block_rows = mx.where(
                    (complete_counts <= block_budget)[..., None],
                    canonical,
                    ranked,
                )
            else:
                selected_block_rows = canonical

            selected_count = mx.minimum(complete_counts, block_budget)
            selected_indices = (
                selected_block_rows[..., None] * ratio
                + mx.arange(ratio, dtype=mx.int32)
            ).reshape(batch, chunk_tokens, selected_width * ratio)
            selected_valid = mx.broadcast_to(
                mx.arange(selected_width)[None, None, :, None]
                < selected_count[..., None, None],
                (batch, chunk_tokens, selected_width, ratio),
            ).reshape(batch, chunk_tokens, selected_width * ratio)
        else:
            selected_indices = mx.zeros(
                (batch, chunk_tokens, 0), dtype=mx.int32
            )
            selected_valid = mx.zeros(
                (batch, chunk_tokens, 0), dtype=mx.bool_
            )

        # The zero-to-three visible tokens after the final complete block are
        # always retained by the published QSA contract.
        tail_width = ratio - 1
        tail = complete_counts[..., None] * ratio + mx.arange(
            tail_width, dtype=mx.int32
        )
        tail_valid = tail < visible_counts[..., None]
        selected_indices = mx.concatenate((selected_indices, tail), axis=-1)
        selected_valid = mx.concatenate((selected_valid, tail_valid), axis=-1)
        safe_selected = mx.where(selected_valid, selected_indices, 0).astype(mx.int32)

        selected_keys = _batch_gather_tokens(key_rows, safe_selected).transpose(
            0, 1, 3, 2, 4
        )
        selected_values = _batch_gather_tokens(value_rows, safe_selected).transpose(
            0, 1, 3, 2, 4
        )

        chunk_queries = queries[:, :, start:stop].transpose(0, 2, 1, 3)
        grouped_queries = chunk_queries.reshape(
            batch,
            chunk_tokens,
            num_key_value_heads,
            groups,
            head_dim,
        )
        scores = (
            grouped_queries.astype(mx.float32)
            @ selected_keys.astype(mx.float32).swapaxes(-1, -2)
        ) / math.sqrt(head_dim)
        scores = mx.where(
            selected_valid[:, :, None, None, :],
            scores,
            mx.finfo(scores.dtype).min,
        )
        probabilities = mx.softmax(scores, axis=-1).astype(chunk_queries.dtype)
        output = probabilities @ selected_values
        outputs.append(
            output.reshape(batch, chunk_tokens, num_query_heads, head_dim)
        )

    return mx.concatenate(outputs, axis=1)


__all__ = [
    "contiguous_causal_gathered_qsa",
    "contiguous_causal_query_chunk",
]
