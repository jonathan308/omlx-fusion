# SPDX-License-Identifier: Apache-2.0
"""Exact MLX execution adapter for Qwen3.8-Flash-Next QSA.

The published Qwen4-Exp indexer groups causally visible keys into complete
four-token micro-blocks, scores the mean key of each block, selects at most
512 blocks (2048 tokens), and retains the zero-to-three visible tail tokens.
Only those selected main-attention K/V rows are gathered.  At no point does
this module construct a full ``[query_tokens, key_tokens]`` attention-score
matrix.

This is a portable MLX implementation, not a custom Metal kernel.  It is the
correctness backend and integration seam for a future fused implementation.
It intentionally has no dense-SDPA or token-level DSA fallback.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .qsa import (
    Qwen4ExpQSABackendUnavailableError,
    Qwen4ExpQSAContract,
    Qwen4ExpQSAInputError,
    Qwen4ExpQSARequest,
)

IndexKeyNorm = Callable[[mx.array], mx.array]
RowObserver = Callable[["Qwen4ExpQSARowTrace"], None]


@dataclass(frozen=True)
class Qwen4ExpQSAGeometry:
    """QSA dimensions used by the portable kernel.

    Production is always constructed from :class:`Qwen4ExpQSAContract`.
    Shape-configurable instances exist only so the exact algorithm can be
    tested without allocating 24 x 256 attention heads.
    """

    num_query_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    indexer_query_heads: int
    indexer_head_dim: int
    compress_ratio: int
    token_budget: int

    def __post_init__(self) -> None:
        positive = (
            "num_query_heads",
            "num_key_value_heads",
            "head_dim",
            "rotary_dim",
            "indexer_query_heads",
            "indexer_head_dim",
            "compress_ratio",
            "token_budget",
        )
        if any(
            type(getattr(self, field)) is not int or getattr(self, field) <= 0
            for field in positive
        ):
            raise Qwen4ExpQSAInputError("QSA geometry values must be positive integers")
        if self.num_query_heads % self.num_key_value_heads:
            raise Qwen4ExpQSAInputError(
                "num_query_heads must be divisible by num_key_value_heads"
            )
        if self.rotary_dim > min(self.head_dim, self.indexer_head_dim):
            raise Qwen4ExpQSAInputError(
                "rotary_dim must fit main and indexer head dimensions"
            )
        if self.rotary_dim % 2:
            raise Qwen4ExpQSAInputError("rotary_dim must be even")
        if self.token_budget % self.compress_ratio:
            raise Qwen4ExpQSAInputError(
                "token_budget must be divisible by compress_ratio"
            )

    @property
    def block_budget(self) -> int:
        return self.token_budget // self.compress_ratio

    @classmethod
    def from_contract(cls, contract: Qwen4ExpQSAContract) -> Qwen4ExpQSAGeometry:
        return cls(
            num_query_heads=contract.num_query_heads,
            num_key_value_heads=contract.num_key_value_heads,
            head_dim=contract.head_dim,
            rotary_dim=contract.rotary_dim,
            indexer_query_heads=contract.indexer_query_heads,
            indexer_head_dim=contract.indexer_head_dim,
            compress_ratio=contract.compress_ratio,
            token_budget=contract.token_budget,
        )


@dataclass(frozen=True)
class Qwen4ExpQSARowTrace:
    """Bounded diagnostic proving a row attended only to selected K/V."""

    batch_index: int
    query_index: int
    full_key_tokens: int
    visible_tokens: int
    complete_blocks: int
    selected_blocks: int
    selected_tokens: int


def _require_array(value: Any, name: str) -> mx.array:
    if not isinstance(value, mx.array):
        raise Qwen4ExpQSAInputError(f"{name} must be an mlx.core.array")
    return value


def _rotate_half(value: mx.array) -> mx.array:
    half = value.shape[-1] // 2
    return mx.concatenate((-value[..., half:], value[..., :half]), axis=-1)


def apply_partial_rope(
    states: mx.array,
    cos: mx.array,
    sin: mx.array,
    *,
    rotary_dim: int,
) -> mx.array:
    """Apply Qwen4-Exp partial RoPE to the leading head dimensions."""

    if states.shape[-1] < rotary_dim or cos.shape[-1] != rotary_dim:
        raise Qwen4ExpQSAInputError("partial RoPE dimensions do not match QSA")
    if sin.shape != cos.shape:
        raise Qwen4ExpQSAInputError("RoPE cosine and sine shapes must match")
    rope = states[..., :rotary_dim]
    rotated = rope * cos + _rotate_half(rope) * sin
    return mx.concatenate((rotated, states[..., rotary_dim:]), axis=-1)


def _visible_indices(mask_row: mx.array) -> mx.array:
    """Compact one bool/additive mask row into ascending visible positions."""

    if mask_row.ndim != 1:
        raise Qwen4ExpQSAInputError("attention mask row must be one-dimensional")
    if mask_row.dtype == mx.bool_:
        visible = mask_row
    elif mx.issubdtype(mask_row.dtype, mx.floating):
        visible = mask_row == 0
    else:
        raise Qwen4ExpQSAInputError(
            "attention mask must be bool or an additive floating mask"
        )
    count = int(mx.sum(visible).item())
    positions = mx.arange(mask_row.shape[0], dtype=mx.int32)
    sentinel = mx.array(mask_row.shape[0], dtype=mx.int32)
    compact = mx.sort(mx.where(visible, positions, sentinel))
    return compact[:count]


def select_qsa_token_indices(
    index_query: mx.array,
    raw_index_keys: mx.array,
    position_cos: mx.array,
    position_sin: mx.array,
    visible_mask: mx.array,
    *,
    geometry: Qwen4ExpQSAGeometry,
    index_key_norm: IndexKeyNorm,
) -> tuple[mx.array, int, int]:
    """Select exact Qwen4-Exp micro-block tokens for one query row.

    Returns ``(token_indices, complete_blocks, selected_blocks)``.  The only
    context-wide score sheet has shape ``[complete_blocks, indexer_heads]``;
    main Q/K scores are computed later over the gathered token indices only.
    """

    for value, name in (
        (index_query, "index_query"),
        (raw_index_keys, "raw_index_keys"),
        (position_cos, "position_cos"),
        (position_sin, "position_sin"),
        (visible_mask, "visible_mask"),
    ):
        _require_array(value, name)
    if index_query.shape != (
        geometry.indexer_query_heads,
        geometry.indexer_head_dim,
    ):
        raise Qwen4ExpQSAInputError("index_query has invalid QSA shape")
    if raw_index_keys.ndim != 2 or raw_index_keys.shape[1] != geometry.indexer_head_dim:
        raise Qwen4ExpQSAInputError("raw_index_keys has invalid QSA shape")
    key_tokens = raw_index_keys.shape[0]
    if position_cos.shape != (key_tokens, geometry.rotary_dim):
        raise Qwen4ExpQSAInputError("position_cos has invalid QSA shape")
    if position_sin.shape != position_cos.shape or visible_mask.shape != (key_tokens,):
        raise Qwen4ExpQSAInputError("position or visibility shape does not match keys")
    if not callable(index_key_norm):
        raise Qwen4ExpQSAInputError("QSA requires the checkpoint index k-layernorm")

    visible = _visible_indices(visible_mask)
    visible_count = visible.shape[0]
    complete_blocks = visible_count // geometry.compress_ratio
    complete_tokens = complete_blocks * geometry.compress_ratio

    if complete_blocks:
        block_tokens = visible[:complete_tokens].reshape(
            complete_blocks, geometry.compress_ratio
        )
        grouped_keys = raw_index_keys[block_tokens.reshape(-1)].reshape(
            complete_blocks,
            geometry.compress_ratio,
            geometry.indexer_head_dim,
        )
        # Transformers computes the mean in float32 and casts back before the
        # checkpoint's zero-centered RMSNorm.
        pooled = mx.mean(grouped_keys.astype(mx.float32), axis=1).astype(
            raw_index_keys.dtype
        )
        pooled = index_key_norm(pooled)
        if not isinstance(pooled, mx.array) or pooled.shape != (
            complete_blocks,
            geometry.indexer_head_dim,
        ):
            raise Qwen4ExpQSAInputError(
                "index_key_norm returned an invalid pooled-key shape"
            )
        starts = block_tokens[:, 0]
        pooled = apply_partial_rope(
            pooled,
            position_cos[starts],
            position_sin[starts],
            rotary_dim=geometry.rotary_dim,
        )

        scores = (
            index_query.astype(mx.float32) @ pooled.astype(mx.float32).swapaxes(-1, -2)
        ).swapaxes(-1, -2)
        scores = mx.sum(mx.maximum(scores, 0), axis=-1) / math.sqrt(
            geometry.indexer_head_dim
        )
        selected_blocks = min(complete_blocks, geometry.block_budget)
        if selected_blocks == complete_blocks:
            selected_block_rows = mx.arange(complete_blocks, dtype=mx.int32)
        else:
            selected_block_rows = mx.argpartition(
                -scores, kth=selected_blocks - 1, axis=-1
            )[:selected_blocks].astype(mx.int32)
        selected = block_tokens[selected_block_rows].reshape(-1).astype(mx.int32)
    else:
        selected_blocks = 0
        selected = mx.array([], dtype=mx.int32)

    # The incomplete visible group is never scored or discarded.
    tail = visible[complete_tokens:].astype(mx.int32)
    if tail.shape[0]:
        selected = mx.concatenate((selected, tail), axis=0)
    return selected, complete_blocks, selected_blocks


def sparse_attention_row(
    query: mx.array,
    keys: mx.array,
    values: mx.array,
    selected_indices: mx.array,
    selected_cos: mx.array,
    selected_sin: mx.array,
    query_cos: mx.array,
    query_sin: mx.array,
    *,
    geometry: Qwen4ExpQSAGeometry,
) -> mx.array:
    """Attend one query to gathered Qwen4-Exp K/V rows only."""

    if selected_indices.shape[0] == 0:
        raise Qwen4ExpQSAInputError("a QSA query has no causally visible keys")
    if query.shape != (geometry.num_query_heads, geometry.head_dim):
        raise Qwen4ExpQSAInputError("query has invalid main-attention shape")
    if keys.ndim != 3 or keys.shape[0] != geometry.num_key_value_heads:
        raise Qwen4ExpQSAInputError("keys have invalid main-attention shape")
    if values.shape != keys.shape or keys.shape[-1] != geometry.head_dim:
        raise Qwen4ExpQSAInputError("values must match the QSA key shape")

    query = apply_partial_rope(
        query,
        query_cos,
        query_sin,
        rotary_dim=geometry.rotary_dim,
    )
    # Indexing the already-selected rows is the sparse execution boundary.
    # Transpose to [selected, kv_heads, dim] so per-token RoPE broadcasts.
    selected_keys = keys[:, selected_indices, :].swapaxes(0, 1)
    selected_values = values[:, selected_indices, :]
    selected_keys = apply_partial_rope(
        selected_keys,
        selected_cos[:, None, :],
        selected_sin[:, None, :],
        rotary_dim=geometry.rotary_dim,
    ).swapaxes(0, 1)

    groups = geometry.num_query_heads // geometry.num_key_value_heads
    grouped_query = query.reshape(
        geometry.num_key_value_heads, groups, geometry.head_dim
    )
    scores = (
        grouped_query.astype(mx.float32)
        @ selected_keys.astype(mx.float32).swapaxes(-1, -2)
    ) / math.sqrt(geometry.head_dim)
    probabilities = mx.softmax(scores, axis=-1).astype(query.dtype)
    output = probabilities @ selected_values
    return output.reshape(geometry.num_query_heads, geometry.head_dim)


def _bounded_decode_all_selected(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cos: mx.array,
    sin: mx.array,
    mask: mx.array,
    *,
    geometry: Qwen4ExpQSAGeometry,
) -> mx.array:
    """Vectorized exact decode when the QSA budget retains every visible key."""

    query_cos = cos[:, -1:, :]
    query_sin = sin[:, -1:, :]
    queries = apply_partial_rope(
        queries,
        query_cos[:, None, :, :],
        query_sin[:, None, :, :],
        rotary_dim=geometry.rotary_dim,
    )
    keys = apply_partial_rope(
        keys,
        cos[:, None, :, :],
        sin[:, None, :, :],
        rotary_dim=geometry.rotary_dim,
    )
    batch, _, _, head_dim = queries.shape
    groups = geometry.num_query_heads // geometry.num_key_value_heads
    grouped_queries = queries.reshape(
        batch,
        geometry.num_key_value_heads,
        groups,
        1,
        head_dim,
    )
    scores = (
        grouped_queries.astype(mx.float32)
        @ keys.astype(mx.float32)[:, :, None].swapaxes(-1, -2)
    ) / math.sqrt(head_dim)
    visible = mask[:, :, None]
    if visible.dtype == mx.bool_:
        scores = mx.where(visible, scores, mx.finfo(scores.dtype).min)
    else:
        scores = scores + visible
    probabilities = mx.softmax(scores, axis=-1).astype(queries.dtype)
    output = probabilities @ values[:, :, None]
    output = output.reshape(batch, geometry.num_query_heads, 1, head_dim)
    return output.transpose(0, 2, 1, 3)


_PREFILL_QUERY_CHUNK = 16


def _batch_gather_tokens(values: mx.array, indices: mx.array) -> mx.array:
    """Gather token rows independently for every batch without a host read."""

    batch, tokens = values.shape[:2]
    trailing = values.shape[2:]
    offset_shape = (batch,) + (1,) * (indices.ndim - 1)
    offsets = mx.arange(batch, dtype=mx.int32).reshape(offset_shape) * tokens
    flat_indices = (indices.astype(mx.int32) + offsets).reshape(-1)
    flat_values = values.reshape(batch * tokens, *trailing)
    return flat_values[flat_indices].reshape(*indices.shape, *trailing)


def _compact_visible_indices(visible: mx.array) -> tuple[mx.array, mx.array]:
    """Stable compact of ``[..., key_tokens]`` booleans using GPU scatter.

    Visible and invisible destinations together form a permutation, so the
    scatter has no collisions.  This preserves arbitrary padding/hole masks
    without the row-level ``sum(...).item()`` synchronization used by the
    scalar reference path.
    """

    key_tokens = visible.shape[-1]
    visible_i = visible.astype(mx.int32)
    invisible_i = 1 - visible_i
    counts = mx.sum(visible_i, axis=-1)
    visible_rank = mx.cumsum(visible_i, axis=-1) - 1
    invisible_rank = mx.cumsum(invisible_i, axis=-1) - 1
    destinations = mx.where(
        visible,
        visible_rank,
        counts[..., None] + invisible_rank,
    ).astype(mx.int32)
    positions = mx.arange(key_tokens, dtype=mx.int32)
    positions = mx.broadcast_to(positions, visible.shape)
    sources = mx.where(visible, positions, key_tokens)
    compact = mx.full(visible.shape, key_tokens, dtype=mx.int32)
    return mx.put_along_axis(compact, destinations, sources, axis=-1), counts


def _bounded_prefill_all_selected(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cos: mx.array,
    sin: mx.array,
    mask: mx.array,
    *,
    geometry: Qwen4ExpQSAGeometry,
    query_chunk: int = _PREFILL_QUERY_CHUNK,
) -> mx.array:
    """Exact chunked prefill while every visible block fits the budget."""

    batch, _, query_tokens, _ = queries.shape
    key_tokens = keys.shape[2]
    query_start = key_tokens - query_tokens
    rotated_keys = apply_partial_rope(
        keys,
        cos[:, None, :, :],
        sin[:, None, :, :],
        rotary_dim=geometry.rotary_dim,
    )
    groups = geometry.num_query_heads // geometry.num_key_value_heads
    outputs: list[mx.array] = []
    for start in range(0, query_tokens, query_chunk):
        stop = min(start + query_chunk, query_tokens)
        chunk_tokens = stop - start
        chunk_queries = queries[:, :, start:stop].transpose(0, 2, 1, 3)
        chunk_cos = cos[:, query_start + start : query_start + stop]
        chunk_sin = sin[:, query_start + start : query_start + stop]
        chunk_queries = apply_partial_rope(
            chunk_queries,
            chunk_cos[:, :, None, :],
            chunk_sin[:, :, None, :],
            rotary_dim=geometry.rotary_dim,
        )
        grouped_queries = chunk_queries.reshape(
            batch,
            chunk_tokens,
            geometry.num_key_value_heads,
            groups,
            geometry.head_dim,
        )
        scores = (
            grouped_queries.astype(mx.float32)
            @ rotated_keys.astype(mx.float32)[:, None, :, :, :].swapaxes(-1, -2)
        ) / math.sqrt(geometry.head_dim)
        visible = mask[:, 0, start:stop, None, None, :]
        if visible.dtype == mx.bool_:
            scores = mx.where(visible, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + visible
        probabilities = mx.softmax(scores, axis=-1).astype(chunk_queries.dtype)
        output = probabilities @ values[:, None, :, :, :]
        outputs.append(
            output.reshape(
                batch, chunk_tokens, geometry.num_query_heads, geometry.head_dim
            )
        )
    return mx.concatenate(outputs, axis=1)


def _vectorized_sparse_prefill(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    index_queries: mx.array,
    index_keys: mx.array,
    cos: mx.array,
    sin: mx.array,
    mask: mx.array,
    *,
    geometry: Qwen4ExpQSAGeometry,
    index_key_norm: IndexKeyNorm,
    query_chunk: int = _PREFILL_QUERY_CHUNK,
) -> mx.array:
    """Batch QSA selection/attention over bounded groups of query rows.

    The only dense context dimension is bounded by ``query_chunk``.  Main
    attention always gathers at most ``token_budget + compress_ratio - 1``
    rows, so 10K/20K prompts never construct dense QxK attention scores.
    """

    batch, _, query_tokens, _ = queries.shape
    key_tokens = keys.shape[2]
    ratio = geometry.compress_ratio
    max_blocks = key_tokens // ratio
    block_budget = geometry.block_budget
    query_start = key_tokens - query_tokens
    key_rows = keys.transpose(0, 2, 1, 3)
    value_rows = values.transpose(0, 2, 1, 3)
    outputs: list[mx.array] = []

    for start in range(0, query_tokens, query_chunk):
        stop = min(start + query_chunk, query_tokens)
        chunk_tokens = stop - start
        chunk_mask = mask[:, 0, start:stop]
        visible = chunk_mask if chunk_mask.dtype == mx.bool_ else chunk_mask == 0
        compact, visible_counts = _compact_visible_indices(visible)
        complete_counts = visible_counts // ratio

        if max_blocks:
            block_tokens = compact[..., : max_blocks * ratio].reshape(
                batch, chunk_tokens, max_blocks, ratio
            )
            safe_block_tokens = mx.minimum(block_tokens, key_tokens - 1)
            grouped_keys = _batch_gather_tokens(index_keys, safe_block_tokens)
            pooled = mx.mean(grouped_keys.astype(mx.float32), axis=-2).astype(
                index_keys.dtype
            )
            pooled = index_key_norm(pooled)
            block_starts = safe_block_tokens[..., 0]
            pooled = apply_partial_rope(
                pooled,
                _batch_gather_tokens(cos, block_starts),
                _batch_gather_tokens(sin, block_starts),
                rotary_dim=geometry.rotary_dim,
            )
            chunk_query_cos = cos[:, query_start + start : query_start + stop]
            chunk_query_sin = sin[:, query_start + start : query_start + stop]
            rotated_index_queries = apply_partial_rope(
                index_queries[:, start:stop],
                chunk_query_cos[:, :, None, :],
                chunk_query_sin[:, :, None, :],
                rotary_dim=geometry.rotary_dim,
            )
            block_scores = (
                rotated_index_queries.astype(mx.float32)
                @ pooled.astype(mx.float32).swapaxes(-1, -2)
            ).swapaxes(-1, -2)
            block_scores = mx.sum(mx.maximum(block_scores, 0), axis=-1) / math.sqrt(
                geometry.indexer_head_dim
            )
            valid_blocks = (
                mx.arange(max_blocks)[None, None, :] < complete_counts[..., None]
            )
            block_scores = mx.where(
                valid_blocks, block_scores, mx.finfo(block_scores.dtype).min
            )

            selected_width = min(max_blocks, block_budget)
            canonical = mx.broadcast_to(
                mx.arange(selected_width, dtype=mx.int32)[None, None],
                (batch, chunk_tokens, selected_width),
            )
            if max_blocks > block_budget:
                ranked = mx.argpartition(-block_scores, kth=block_budget - 1, axis=-1)[
                    ..., :block_budget
                ].astype(mx.int32)
                selected_block_rows = mx.where(
                    (complete_counts <= block_budget)[..., None],
                    canonical,
                    ranked,
                )
            else:
                selected_block_rows = canonical
            selected_count = mx.minimum(complete_counts, block_budget)
            selected_blocks = mx.take_along_axis(
                block_tokens,
                selected_block_rows[..., None],
                axis=-2,
            )
            selected_indices = selected_blocks.reshape(
                batch, chunk_tokens, selected_width * ratio
            )
            selected_valid = mx.broadcast_to(
                mx.arange(selected_width)[None, None, :, None]
                < selected_count[..., None, None],
                (batch, chunk_tokens, selected_width, ratio),
            ).reshape(batch, chunk_tokens, selected_width * ratio)
        else:
            selected_indices = mx.zeros((batch, chunk_tokens, 0), dtype=mx.int32)
            selected_valid = mx.zeros((batch, chunk_tokens, 0), dtype=mx.bool_)

        tail_width = ratio - 1
        tail_offsets = mx.arange(tail_width, dtype=mx.int32)
        tail_positions = complete_counts[..., None] * ratio + tail_offsets
        tail_valid = tail_positions < visible_counts[..., None]
        safe_tail_positions = mx.minimum(tail_positions, key_tokens - 1)
        tail = mx.take_along_axis(compact, safe_tail_positions, axis=-1)
        selected_indices = mx.concatenate((selected_indices, tail), axis=-1)
        selected_valid = mx.concatenate((selected_valid, tail_valid), axis=-1)
        safe_selected = mx.where(selected_valid, selected_indices, 0).astype(mx.int32)

        selected_keys = _batch_gather_tokens(key_rows, safe_selected)
        selected_values = _batch_gather_tokens(value_rows, safe_selected)
        selected_cos = _batch_gather_tokens(cos, safe_selected)
        selected_sin = _batch_gather_tokens(sin, safe_selected)
        selected_keys = apply_partial_rope(
            selected_keys,
            selected_cos[..., None, :],
            selected_sin[..., None, :],
            rotary_dim=geometry.rotary_dim,
        ).transpose(0, 1, 3, 2, 4)
        selected_values = selected_values.transpose(0, 1, 3, 2, 4)

        chunk_queries = queries[:, :, start:stop].transpose(0, 2, 1, 3)
        chunk_query_cos = cos[:, query_start + start : query_start + stop]
        chunk_query_sin = sin[:, query_start + start : query_start + stop]
        chunk_queries = apply_partial_rope(
            chunk_queries,
            chunk_query_cos[:, :, None, :],
            chunk_query_sin[:, :, None, :],
            rotary_dim=geometry.rotary_dim,
        )
        groups = geometry.num_query_heads // geometry.num_key_value_heads
        grouped_queries = chunk_queries.reshape(
            batch,
            chunk_tokens,
            geometry.num_key_value_heads,
            groups,
            geometry.head_dim,
        )
        scores = (
            grouped_queries.astype(mx.float32)
            @ selected_keys.astype(mx.float32).swapaxes(-1, -2)
        ) / math.sqrt(geometry.head_dim)
        scores = mx.where(
            selected_valid[:, :, None, None, :],
            scores,
            mx.finfo(scores.dtype).min,
        )
        probabilities = mx.softmax(scores, axis=-1).astype(chunk_queries.dtype)
        output = probabilities @ selected_values
        outputs.append(
            output.reshape(
                batch, chunk_tokens, geometry.num_query_heads, geometry.head_dim
            )
        )

    return mx.concatenate(outputs, axis=1)


def micro_block_sparse_qsa(
    request: Qwen4ExpQSARequest,
    *,
    geometry: Qwen4ExpQSAGeometry,
    index_key_norm: IndexKeyNorm,
    row_observer: RowObserver | None = None,
) -> mx.array:
    """Execute portable sparse QSA for prefill or cached decode states."""

    arrays = {
        name: _require_array(getattr(request, name), name)
        for name in (
            "queries",
            "keys",
            "values",
            "index_queries",
            "index_keys",
            "position_cos",
            "position_sin",
            "attention_mask",
        )
    }
    queries = arrays["queries"]
    keys = arrays["keys"]
    values = arrays["values"]
    index_queries = arrays["index_queries"]
    index_keys = arrays["index_keys"]
    cos = arrays["position_cos"]
    sin = arrays["position_sin"]
    mask = arrays["attention_mask"]

    if queries.ndim != 4:
        raise Qwen4ExpQSAInputError("queries must be rank four")
    batch, query_heads, query_tokens, head_dim = queries.shape
    key_tokens = keys.shape[2] if keys.ndim == 4 else -1
    expected = {
        "queries": (
            batch,
            geometry.num_query_heads,
            query_tokens,
            geometry.head_dim,
        ),
        "keys": (
            batch,
            geometry.num_key_value_heads,
            key_tokens,
            geometry.head_dim,
        ),
        "values": (
            batch,
            geometry.num_key_value_heads,
            key_tokens,
            geometry.head_dim,
        ),
        "index_queries": (
            batch,
            query_tokens,
            geometry.indexer_query_heads,
            geometry.indexer_head_dim,
        ),
        "index_keys": (batch, key_tokens, geometry.indexer_head_dim),
        "position_cos": (batch, key_tokens, geometry.rotary_dim),
        "position_sin": (batch, key_tokens, geometry.rotary_dim),
        "attention_mask": (batch, 1, query_tokens, key_tokens),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise Qwen4ExpQSAInputError(
                f"{name} must have shape {shape}, got {arrays[name].shape}"
            )
    if query_heads != geometry.num_query_heads or head_dim != geometry.head_dim:
        raise Qwen4ExpQSAInputError("queries do not match QSA geometry")
    if query_tokens > key_tokens:
        raise Qwen4ExpQSAInputError("query length cannot exceed key length")

    # Before the 2048-token budget is exceeded, QSA retains every causally
    # visible key. Decode can therefore run the exact selected set as one
    # bounded vectorized operation, avoiding twelve per-layer host syncs.
    if (
        query_tokens == 1
        and key_tokens <= geometry.token_budget
        and row_observer is None
    ):
        return _bounded_decode_all_selected(
            queries,
            keys,
            values,
            cos,
            sin,
            mask,
            geometry=geometry,
        )

    if (
        query_tokens > 1
        and key_tokens <= geometry.token_budget
        and row_observer is None
    ):
        return _bounded_prefill_all_selected(
            queries,
            keys,
            values,
            cos,
            sin,
            mask,
            geometry=geometry,
        )

    if query_tokens > 1 and row_observer is None:
        return _vectorized_sparse_prefill(
            queries,
            keys,
            values,
            index_queries,
            index_keys,
            cos,
            sin,
            mask,
            geometry=geometry,
            index_key_norm=index_key_norm,
        )

    # The request convention is append-only: current query positions are the
    # final query_tokens rows in the full positional cache.
    query_cos = cos[:, key_tokens - query_tokens :, :]
    query_sin = sin[:, key_tokens - query_tokens :, :]
    batches: list[mx.array] = []
    for batch_idx in range(batch):
        rows: list[mx.array] = []
        for query_idx in range(query_tokens):
            index_query = apply_partial_rope(
                index_queries[batch_idx, query_idx],
                query_cos[batch_idx, query_idx],
                query_sin[batch_idx, query_idx],
                rotary_dim=geometry.rotary_dim,
            )
            selected, complete_blocks, selected_blocks = select_qsa_token_indices(
                index_query,
                index_keys[batch_idx],
                cos[batch_idx],
                sin[batch_idx],
                mask[batch_idx, 0, query_idx],
                geometry=geometry,
                index_key_norm=index_key_norm,
            )
            row = sparse_attention_row(
                queries[batch_idx, :, query_idx],
                keys[batch_idx],
                values[batch_idx],
                selected,
                cos[batch_idx, selected],
                sin[batch_idx, selected],
                query_cos[batch_idx, query_idx],
                query_sin[batch_idx, query_idx],
                geometry=geometry,
            )
            rows.append(row)
            if row_observer is not None:
                visible_tokens = int(
                    mx.sum(
                        mask[batch_idx, 0, query_idx]
                        if mask.dtype == mx.bool_
                        else mask[batch_idx, 0, query_idx] == 0
                    ).item()
                )
                row_observer(
                    Qwen4ExpQSARowTrace(
                        batch_index=batch_idx,
                        query_index=query_idx,
                        full_key_tokens=key_tokens,
                        visible_tokens=visible_tokens,
                        complete_blocks=complete_blocks,
                        selected_blocks=selected_blocks,
                        selected_tokens=selected.shape[0],
                    )
                )
        batches.append(mx.stack(rows, axis=0))
    # [batch, query_tokens, query_heads, head_dim], matching the official
    # attention interface before gate flattening and o_proj.
    return mx.stack(batches, axis=0)


class Qwen4ExpQSAKVCache:
    """Append-only raw QSA state for prefill-to-decode continuity."""

    step = 256

    def __init__(self, *, step: int = 256):
        if type(step) is not int or step <= 0:
            raise ValueError("QSA cache step must be a positive integer")
        self.step = step
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.index_keys: mx.array | None = None
        self.position_cos: mx.array | None = None
        self.position_sin: mx.array | None = None
        self.offset = 0

    def _capacity(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        index_keys: mx.array,
        position_cos: mx.array,
        position_sin: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        for value, name in (
            (keys, "keys"),
            (values, "values"),
            (index_keys, "index_keys"),
            (position_cos, "position_cos"),
            (position_sin, "position_sin"),
        ):
            _require_array(value, name)
        if keys.ndim != 4 or values.shape != keys.shape:
            raise Qwen4ExpQSAInputError(
                "cache keys and values must be matching rank-4 arrays"
            )
        batch, _, new_tokens, _ = keys.shape
        if index_keys.ndim != 3 or index_keys.shape[:2] != (batch, new_tokens):
            raise Qwen4ExpQSAInputError("cache index keys do not match K/V tokens")
        if position_cos.ndim != 3 or position_cos.shape[:2] != (batch, new_tokens):
            raise Qwen4ExpQSAInputError("cache positions do not match K/V tokens")
        if position_sin.shape != position_cos.shape:
            raise Qwen4ExpQSAInputError("cache cosine and sine shapes must match")
        if new_tokens <= 0:
            raise Qwen4ExpQSAInputError("QSA cache update cannot be empty")

        previous = self.offset
        required = previous + new_tokens
        if self.keys is None or required > self._capacity():
            capacity = ((required + self.step - 1) // self.step) * self.step
            new_key_store = mx.zeros(
                (batch, keys.shape[1], capacity, keys.shape[3]), dtype=keys.dtype
            )
            new_value_store = mx.zeros(
                (batch, values.shape[1], capacity, values.shape[3]), dtype=values.dtype
            )
            new_index_store = mx.zeros(
                (batch, capacity, index_keys.shape[2]), dtype=index_keys.dtype
            )
            new_cos_store = mx.zeros(
                (batch, capacity, position_cos.shape[2]), dtype=position_cos.dtype
            )
            new_sin_store = mx.zeros(
                (batch, capacity, position_sin.shape[2]), dtype=position_sin.dtype
            )
            if self.keys is not None:
                if (
                    self.keys.shape[0] != batch
                    or self.keys.shape[1] != keys.shape[1]
                    or self.keys.shape[3] != keys.shape[3]
                    or self.index_keys is None
                    or self.index_keys.shape[2] != index_keys.shape[2]
                    or self.position_cos is None
                    or self.position_cos.shape[2] != position_cos.shape[2]
                ):
                    raise Qwen4ExpQSAInputError(
                        "QSA cache geometry changed between updates"
                    )
                new_key_store[..., :previous, :] = self.keys[..., :previous, :]
                new_value_store[..., :previous, :] = self.values[..., :previous, :]
                new_index_store[:, :previous, :] = self.index_keys[:, :previous, :]
                new_cos_store[:, :previous, :] = self.position_cos[:, :previous, :]
                new_sin_store[:, :previous, :] = self.position_sin[:, :previous, :]
            self.keys = new_key_store
            self.values = new_value_store
            self.index_keys = new_index_store
            self.position_cos = new_cos_store
            self.position_sin = new_sin_store

        self.keys[..., previous:required, :] = keys
        self.values[..., previous:required, :] = values
        self.index_keys[:, previous:required, :] = index_keys
        self.position_cos[:, previous:required, :] = position_cos
        self.position_sin[:, previous:required, :] = position_sin
        self.offset = required
        return self.state

    @property
    def state(self) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        if self.keys is None:
            raise Qwen4ExpQSAInputError("QSA cache is empty")
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
            self.index_keys[:, : self.offset, :],
            self.position_cos[:, : self.offset, :],
            self.position_sin[:, : self.offset, :],
        )

    def trim(self, count: int) -> int:
        trimmed = min(max(int(count), 0), self.offset)
        self.offset -= trimmed
        return trimmed

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return sum(value.nbytes for value in self.state)


def _canonical_attention_mask(
    mask: mx.array | None,
    *,
    batch: int,
    query_tokens: int,
    key_tokens: int,
    query_start: int,
) -> mx.array:
    if mask is None:
        query_positions = query_start + mx.arange(query_tokens)
        key_positions = mx.arange(key_tokens)
        causal = key_positions[None, :] <= query_positions[:, None]
        return mx.broadcast_to(causal[None, None], (batch, 1, query_tokens, key_tokens))
    _require_array(mask, "attention_mask")
    if mask.ndim == 2 and mask.shape == (query_tokens, key_tokens):
        mask = mx.broadcast_to(mask[None, None], (batch, 1, query_tokens, key_tokens))
    elif mask.ndim == 3 and mask.shape == (batch, query_tokens, key_tokens):
        mask = mask[:, None]
    if mask.shape != (batch, 1, query_tokens, key_tokens):
        raise Qwen4ExpQSAInputError("attention mask does not match prepared cache")
    return mask


def prepare_qsa_request(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    index_queries: mx.array,
    index_keys: mx.array,
    position_cos: mx.array,
    position_sin: mx.array,
    attention_mask: mx.array | None = None,
    cache: Qwen4ExpQSAKVCache | None = None,
) -> Qwen4ExpQSARequest:
    """Append current QSA state and build the protocol's full-state request.

    Root model integration should call this exactly once per attention layer
    invocation, then pass the result to ``Qwen4ExpQSAExecutor``.
    """

    for value, name in (
        (queries, "queries"),
        (keys, "keys"),
        (values, "values"),
        (index_queries, "index_queries"),
        (index_keys, "index_keys"),
        (position_cos, "position_cos"),
        (position_sin, "position_sin"),
    ):
        _require_array(value, name)
    if queries.ndim != 4 or keys.ndim != 4:
        raise Qwen4ExpQSAInputError("QSA projections must be rank four")
    batch, _, query_tokens, _ = queries.shape
    if keys.shape[0] != batch or keys.shape[2] < query_tokens:
        raise Qwen4ExpQSAInputError(
            "full K/V state must contain every current query position"
        )
    if cache is not None and keys.shape[2] != query_tokens:
        raise Qwen4ExpQSAInputError(
            "a cache append must contain only the current query tokens"
        )
    previous = keys.shape[2] - query_tokens if cache is None else cache.offset
    if cache is None:
        full_keys, full_values = keys, values
        full_index_keys = index_keys
        full_cos, full_sin = position_cos, position_sin
    else:
        (
            full_keys,
            full_values,
            full_index_keys,
            full_cos,
            full_sin,
        ) = cache.update_and_fetch(keys, values, index_keys, position_cos, position_sin)
    key_tokens = full_keys.shape[2]
    mask = _canonical_attention_mask(
        attention_mask,
        batch=batch,
        query_tokens=query_tokens,
        key_tokens=key_tokens,
        query_start=previous,
    )
    return Qwen4ExpQSARequest(
        queries=queries,
        keys=full_keys,
        values=full_values,
        index_queries=index_queries,
        index_keys=full_index_keys,
        position_cos=full_cos,
        position_sin=full_sin,
        attention_mask=mask,
        cache=cache,
    )


class Qwen4ExpMLXQSABackend:
    """Portable exact implementation of ``Qwen4ExpQSASparseBackend``."""

    name = "mlx-qwen4-exp-micro-block"

    def __init__(
        self,
        *,
        index_key_norm: IndexKeyNorm,
        row_observer: RowObserver | None = None,
    ) -> None:
        self.index_key_norm = index_key_norm
        self.row_observer = row_observer

    def supports(self, contract: Qwen4ExpQSAContract) -> bool:
        if not callable(self.index_key_norm):
            return False
        try:
            geometry = Qwen4ExpQSAGeometry.from_contract(contract)
        except (AttributeError, Qwen4ExpQSAInputError):
            return False
        return (
            geometry.num_query_heads == 24
            and geometry.num_key_value_heads == 2
            and geometry.head_dim == 256
            and geometry.rotary_dim == 64
            and geometry.indexer_query_heads == 4
            and geometry.indexer_head_dim == 128
            and geometry.compress_ratio == 4
            and geometry.token_budget == 2048
            and geometry.block_budget == 512
        )

    def execute(
        self,
        request: Qwen4ExpQSARequest,
        *,
        contract: Qwen4ExpQSAContract,
    ) -> mx.array:
        if not self.supports(contract):
            raise Qwen4ExpQSABackendUnavailableError(
                "MLX QSA backend rejected a non-published Qwen4-Exp contract"
            )
        request.validate(contract)
        return micro_block_sparse_qsa(
            request,
            geometry=Qwen4ExpQSAGeometry.from_contract(contract),
            index_key_norm=self.index_key_norm,
            row_observer=self.row_observer,
        )


__all__ = [
    "Qwen4ExpMLXQSABackend",
    "Qwen4ExpQSAGeometry",
    "Qwen4ExpQSAKVCache",
    "Qwen4ExpQSARowTrace",
    "apply_partial_rope",
    "micro_block_sparse_qsa",
    "prepare_qsa_request",
    "select_qsa_token_indices",
    "sparse_attention_row",
]
