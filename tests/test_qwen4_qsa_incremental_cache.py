"""Exactness and lifecycle tests for incremental Qwen4 QSA block caching."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
qsa_fast = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")


def _identity_rope(x, position_ids):
    del position_ids
    return x


def _append(cache, raw_keys, start, stop):
    length = stop - start
    keys = raw_keys[:, start:stop, :4].reshape(1, 1, length, 4)
    values = (keys + 1).astype(keys.dtype)
    cache.update_and_fetch(keys, values)
    cache.update_indexer(
        raw_keys[:, start:stop],
        mx.arange(start, stop, dtype=mx.int32)[None],
    )


def test_qsa_kv_and_raw_index_buffers_grow_geometrically_with_logical_views():
    cache = language.QSAKVCache()
    raw = mx.arange(8194 * 8, dtype=mx.float32).reshape(1, 8194, 8)

    _append(cache, raw, 0, 2050)
    kv_backing = cache.keys
    index_backing = cache._index_keys
    assert cache.keys.shape[2] == 8192
    assert cache._index_keys.shape[1] == 8192
    assert cache.index_keys.shape == (1, 2050, 8)

    _append(cache, raw, 2050, 4098)
    assert cache.keys is kv_backing
    assert cache._index_keys is index_backing
    assert cache.state[0].shape[2] == 4098
    assert cache.state[2].shape[1] == 4098

    _append(cache, raw, 4098, 8194)
    assert cache.keys.shape[2] == 16384
    assert cache._index_keys.shape[1] == 16384
    assert cache.state[0].shape[2] == 8194
    assert cache.state[2].shape[1] == 8194
    assert language.QSAQuantizedKVCache.step == 8192
    assert language.QSAQuantizedKVCache.geometric_growth is True


@pytest.mark.parametrize("chunks", [(2048, 2048, 2048), (2050, 2048, 2047)])
def test_completed_qsa_blocks_match_one_shot_and_only_compute_new_suffix(chunks):
    total = sum(chunks)
    raw = mx.sin(mx.arange(total * 8, dtype=mx.float32)).reshape(1, total, 8)
    incremental = language.QSAKVCache()
    block_calls = []

    def tracked_norm(x):
        block_calls.append(int(x.shape[1]))
        return x * mx.array(1.25, dtype=x.dtype)

    start = 0
    for length in chunks:
        stop = start + length
        incremental.update_indexer(
            raw[:, start:stop],
            mx.arange(start, stop, dtype=mx.int32)[None],
        )
        actual = incremental.pooled_indexer_keys(
            4,
            tracked_norm,
            _identity_rope,
            cache_tag=tracked_norm,
        )
        start = stop

    calls_before_noop = list(block_calls)
    cached_again = incremental.pooled_indexer_keys(
        4,
        tracked_norm,
        _identity_rope,
        cache_tag=tracked_norm,
    )
    assert block_calls == calls_before_noop
    assert block_calls == [512, 512, 512]

    one_shot = qsa_fast.pool_completed_index_keys(
        raw,
        mx.arange(total, dtype=mx.int32)[None],
        compress_ratio=4,
        index_key_norm=lambda x: x * mx.array(1.25, dtype=x.dtype),
        apply_index_rope=_identity_rope,
    )
    mx.eval(actual, cached_again, one_shot)
    assert mx.array_equal(actual, one_shot).item()
    assert mx.array_equal(cached_again, one_shot).item()


@pytest.mark.parametrize("chunks", [(8, 8, 8), (6, 7, 12)])
def test_gathered_qsa_one_shot_and_incremental_appends_are_exact(chunks, monkeypatch):
    total = sum(chunks)
    mx.random.seed(431)
    queries = mx.random.normal((1, 4, total, 8)).astype(mx.float16)
    keys = mx.random.normal((1, 2, total, 8)).astype(mx.float16)
    values = mx.random.normal((1, 2, total, 8)).astype(mx.float16)
    index_queries = mx.random.normal((1, total, 3, 8)).astype(mx.float16)
    index_keys = mx.random.normal((1, total, 8)).astype(mx.float16)
    positions = mx.arange(total, dtype=mx.int32)[None]
    kwargs = dict(
        num_query_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        indexer_head_dim=8,
        compress_ratio=4,
        token_budget=8,
        index_key_norm=lambda x: x,
        apply_index_rope=_identity_rope,
        query_chunk=3,
    )
    monkeypatch.setattr(qsa_fast, "_native_indexer_scores", lambda *a, **k: None)

    expected = qsa_fast.contiguous_causal_gathered_qsa(
        queries,
        keys,
        values,
        index_queries,
        index_keys,
        positions,
        **kwargs,
    )

    cache = language.QSAKVCache()
    outputs = []
    start = 0
    for length in chunks:
        stop = start + length
        cache.update_indexer(index_keys[:, start:stop], positions[:, start:stop])
        pooled = cache.pooled_indexer_keys(
            4,
            kwargs["index_key_norm"],
            kwargs["apply_index_rope"],
            cache_tag=kwargs["index_key_norm"],
        )
        outputs.append(
            qsa_fast.contiguous_causal_gathered_qsa(
                queries[:, :, start:stop],
                keys[:, :, :stop],
                values[:, :, :stop],
                index_queries[:, start:stop],
                cache.index_keys,
                cache.index_position_ids,
                pooled_index_keys=pooled,
                **kwargs,
            )
        )
        start = stop
    actual = mx.concatenate(outputs, axis=1)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_qsa_ephemeral_pool_rebuilds_after_restore_extract_and_rewinds_trim():
    cache = language.QSAKVCache()
    raw = mx.sin(mx.arange(13 * 8, dtype=mx.float32)).reshape(1, 13, 8)
    _append(cache, raw, 0, 13)
    pooled = cache.pooled_indexer_keys(
        4, lambda x: x, _identity_rope, cache_tag=cache
    )
    mx.eval(pooled)
    assert cache._pooled_index_offset == 3
    assert len(cache.state) == 4

    restored = language.QSAKVCache()
    restored.prefix_cache_restore(cache.prefix_cache_snapshot())
    assert restored._pooled_index_keys is None
    restored_pool = restored.pooled_indexer_keys(
        4, lambda x: x, _identity_rope, cache_tag=restored
    )

    extracted = cache.extract(0)
    assert extracted._pooled_index_keys is None
    extracted_pool = extracted.pooled_indexer_keys(
        4, lambda x: x, _identity_rope, cache_tag=extracted
    )
    mx.eval(restored_pool, extracted_pool, pooled)
    assert mx.array_equal(restored_pool, pooled).item()
    assert mx.array_equal(extracted_pool, pooled).item()

    assert cache.trim(3) == 3
    pooled_backing = cache._pooled_index_keys
    assert pooled_backing is not None
    assert cache._pooled_index_offset == 2
    replacement = mx.cos(mx.arange(3 * 8, dtype=mx.float32)).reshape(1, 3, 8)
    _append(cache, mx.concatenate([raw[:, :10], replacement], axis=1), 10, 13)
    block_calls = []

    def tracked_norm(x):
        block_calls.append(int(x.shape[1]))
        return x

    rebuilt = cache.pooled_indexer_keys(
        4, tracked_norm, _identity_rope, cache_tag=cache
    )
    expected = qsa_fast.pool_completed_index_keys(
        cache.index_keys,
        cache.index_position_ids,
        compress_ratio=4,
        index_key_norm=lambda x: x,
        apply_index_rope=_identity_rope,
    )
    mx.eval(rebuilt, expected)
    assert cache._pooled_index_keys is pooled_backing
    assert block_calls == [1]
    assert mx.array_equal(rebuilt, expected).item()


def test_mtp_prefix_snapshot_compacts_qsa_and_preserves_exact_selection_logits():
    from omlx.patches.mlx_lm_mtp import prompt_priming

    total = 2052  # 513 completed blocks: a real top-512 selection.
    mx.random.seed(733)
    raw = mx.random.normal((1, total, 8)).astype(mx.float16)
    cache = language.QSAKVCache()
    _append(cache, raw, 0, total)
    indexer_tag = SimpleNamespace(
        # Model-owned parameters reachable through the identity tag are not
        # retained sidecar payload and must not enter byte accounting.
        weight=mx.ones((4096,), dtype=mx.float16)
    )
    pooled = cache.pooled_indexer_keys(
        4,
        lambda x: x,
        _identity_rope,
        cache_tag=indexer_tag,
    )
    cache._omlx_text_position_ids_qualified = True
    mx.eval(cache.state, pooled)
    assert cache.keys.shape[2] == 8192
    assert cache._index_keys.shape[1] == 8192
    assert cache._pooled_index_keys.shape[1] == 2048

    [snapshot] = prompt_priming._cache_at_offset([cache], total)
    assert type(snapshot) is type(cache)
    assert snapshot.offset == total
    assert snapshot.keys.shape[2] == total
    assert snapshot.values.shape[2] == total
    assert snapshot._index_keys.shape[1] == total
    assert snapshot._index_position_ids.shape[-1] == total
    assert snapshot._pooled_index_keys.shape == (1, total // 4, 8)
    assert snapshot._pooled_index_offset == total // 4
    assert snapshot._pooled_index_ratio == 4
    assert snapshot._pooled_index_tag is indexer_tag
    assert snapshot._omlx_text_position_ids_qualified is True

    # Exercise the same lifecycle as prepare_prefix_context: detach the
    # retained sidecar again into request-owned state. The compact pooled bank
    # and exact indexer identity must survive this second restore as well.
    [restored] = prompt_priming._cache_at_offset([snapshot], total)
    assert restored is not snapshot
    assert restored._pooled_index_keys is not snapshot._pooled_index_keys
    assert restored._pooled_index_keys.shape == (1, total // 4, 8)
    assert restored._pooled_index_offset == total // 4
    assert restored._pooled_index_ratio == 4
    assert restored._pooled_index_tag is indexer_tag
    assert restored._omlx_text_position_ids_qualified is True

    rebuild_rows = []

    def must_not_rebuild(x):
        rebuild_rows.append(int(x.shape[1]))
        return x

    rebuilt = restored.pooled_indexer_keys(
        4,
        must_not_rebuild,
        _identity_rope,
        cache_tag=indexer_tag,
    )
    mx.eval(rebuilt)
    assert rebuild_rows == []
    assert mx.array_equal(rebuilt, pooled).item()

    index_query = mx.random.normal((1, 1, 4, 8)).astype(mx.float16)
    original_scores = qsa_fast._portable_indexer_scores(index_query, pooled, 8)
    restored_scores = qsa_fast._portable_indexer_scores(index_query, rebuilt, 8)
    original_topk = mx.sort(
        mx.argpartition(original_scores, kth=-512, axis=-1)[..., -512:],
        axis=-1,
    )
    restored_topk = mx.sort(
        mx.argpartition(restored_scores, kth=-512, axis=-1)[..., -512:],
        axis=-1,
    )

    queries = mx.random.normal((1, 2, 1, 4)).astype(mx.float16)
    common = dict(
        num_query_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        indexer_head_dim=8,
        compress_ratio=4,
        token_budget=2048,
    )
    original_output = qsa_fast.contiguous_causal_gathered_qsa_decode(
        queries,
        cache.state[0],
        cache.state[1],
        index_query,
        pooled,
        **common,
    )
    restored_output = qsa_fast.contiguous_causal_gathered_qsa_decode(
        queries,
        restored.state[0],
        restored.state[1],
        index_query,
        rebuilt,
        **common,
    )
    lm_head = mx.random.normal((8, 32)).astype(mx.float16)
    original_logits = original_output.reshape(1, 8) @ lm_head
    restored_logits = restored_output.reshape(1, 8) @ lm_head
    mx.eval(
        original_topk,
        restored_topk,
        original_output,
        restored_output,
        original_logits,
        restored_logits,
    )
    assert mx.array_equal(original_topk, restored_topk).item()
    assert mx.array_equal(original_output, restored_output).item()
    assert mx.array_equal(original_logits, restored_logits).item()
    assert mx.array_equal(
        mx.argmax(original_logits, axis=-1),
        mx.argmax(restored_logits, axis=-1),
    ).item()

    # Ties at the top-512 boundary are especially sensitive to any changed
    # pooled row or order. Zero queries make every score equal and prove the
    # restored selector preserves the original deterministic tie result.
    tie_query = mx.zeros_like(index_query)
    original_tie_scores = qsa_fast._portable_indexer_scores(tie_query, pooled, 8)
    restored_tie_scores = qsa_fast._portable_indexer_scores(tie_query, rebuilt, 8)
    original_tie_topk = mx.sort(
        mx.argpartition(original_tie_scores, kth=-512, axis=-1)[..., -512:],
        axis=-1,
    )
    restored_tie_topk = mx.sort(
        mx.argpartition(restored_tie_scores, kth=-512, axis=-1)[..., -512:],
        axis=-1,
    )
    mx.eval(original_tie_topk, restored_tie_topk)
    assert mx.array_equal(original_tie_topk, restored_tie_topk).item()


def test_mtp_prefix_snapshot_trims_live_qsa_to_nondivisible_earlier_boundary():
    from omlx.cache.prefix_cache import _snapshot_value_nbytes
    from omlx.patches.mlx_lm_mtp import prompt_priming

    total = 37
    target = 31
    ratio = 4
    raw = mx.sin(mx.arange(total * 8, dtype=mx.float32)).reshape(1, total, 8)
    cache = language.QSAKVCache()
    _append(cache, raw, 0, total)
    indexer_tag = SimpleNamespace(
        # Model-owned parameters reachable through the identity tag are not
        # retained sidecar payload and must not enter byte accounting.
        weight=mx.ones((4096,), dtype=mx.float16)
    )
    live_pooled = cache.pooled_indexer_keys(
        ratio,
        lambda x: x,
        _identity_rope,
        cache_tag=indexer_tag,
    )
    pending_hidden = mx.ones((1, 1, 8), dtype=mx.float16)
    estimated = prompt_priming._estimate_compact_mtp_snapshot_nbytes(
        [cache], target, pending_hidden
    )

    [snapshot_cache] = prompt_priming._cache_at_offset([cache], target)
    snapshot = prompt_priming._MtpPrefixSnapshot(
        boundary_tokens=target + 1,
        mtp_cache=[snapshot_cache],
        pending_hidden=pending_hidden,
    )
    mx.eval(*prompt_priming._snapshot_arrays(snapshot), live_pooled)
    assert snapshot_cache.offset == target
    assert snapshot_cache._pooled_index_offset == target // ratio
    assert snapshot_cache._pooled_index_keys.shape[1] == target // ratio
    assert snapshot_cache._pooled_index_tag is indexer_tag
    assert snapshot_cache._omlx_text_position_ids_qualified is False
    assert estimated == _snapshot_value_nbytes(snapshot)
    assert estimated < indexer_tag.weight.nbytes

    rebuilt_rows = []

    def must_not_rebuild(x):
        rebuilt_rows.append(int(x.shape[1]))
        return x

    restored_pooled = snapshot_cache.pooled_indexer_keys(
        ratio,
        must_not_rebuild,
        _identity_rope,
        cache_tag=indexer_tag,
    )
    mx.eval(restored_pooled)
    assert rebuilt_rows == []
    assert mx.array_equal(
        restored_pooled,
        live_pooled[:, : target // ratio],
    ).item()


def test_mtp_prefix_snapshot_rejects_stale_pooled_qsa_bank():
    from omlx.patches.mlx_lm_mtp import prompt_priming

    total = 20
    raw = mx.arange(total * 8, dtype=mx.float32).reshape(1, total, 8)
    cache = language.QSAKVCache()
    _append(cache, raw, 0, total)
    cache.pooled_indexer_keys(
        4,
        lambda x: x,
        _identity_rope,
        cache_tag=object(),
    )
    cache._pooled_index_offset -= 1
    assert prompt_priming._cache_at_offset([cache], total) is None


def test_qsa_equal_mrope_text_planes_qualify_once_and_3d_update_revokes():
    cache = language.QSAKVCache()
    raw = mx.random.normal((1, 12, 8))
    text_positions = mx.broadcast_to(
        mx.arange(12, dtype=mx.int32)[None, None],
        (3, 1, 12),
    )
    cache.update_indexer(raw, text_positions)
    assert cache._omlx_text_position_ids_qualified is False

    current = mx.broadcast_to(
        mx.arange(12, 18, dtype=mx.int32)[None, None],
        (3, 1, 6),
    )
    qualified = language._qualified_text_verify_position_ids(current, cache)
    assert qualified.shape == (1, 6)
    assert mx.array_equal(qualified, current[0]).item()
    assert cache._omlx_text_position_ids_qualified is True

    # The direct verify path appends the collapsed 2-D text view and keeps the
    # one-time qualification live for the next speculative cycle.
    cache.update_indexer(mx.random.normal((1, 6, 8)), qualified)
    assert cache._omlx_text_position_ids_qualified is True

    multimodal = mx.stack(
        [
            mx.arange(18, 20, dtype=mx.int32)[None],
            mx.arange(28, 30, dtype=mx.int32)[None],
            mx.arange(38, 40, dtype=mx.int32)[None],
        ]
    )
    cache.update_indexer(mx.random.normal((1, 2, 8)), multimodal)
    assert cache._omlx_text_position_ids_qualified is False


def test_qsa_divergent_mrope_history_fails_closed_as_multimodal():
    cache = language.QSAKVCache()
    raw = mx.random.normal((1, 8, 8))
    positions = mx.stack(
        [
            mx.arange(8, dtype=mx.int32)[None],
            mx.arange(10, 18, dtype=mx.int32)[None],
            mx.arange(20, 28, dtype=mx.int32)[None],
        ]
    )
    cache.update_indexer(raw, positions)
    current = mx.broadcast_to(
        mx.arange(28, 34, dtype=mx.int32)[None, None],
        (3, 1, 6),
    )

    actual = language._qualified_text_verify_position_ids(current, cache)
    assert actual is current
    assert cache._omlx_text_position_ids_qualified is False


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_portable_qsa_flattened_gemm_is_exactly_the_broadcast_reference(dtype):
    mx.random.seed(909)
    queries = mx.random.normal((1, 7, 4, 128)).astype(dtype)
    pooled = mx.random.normal((1, 19, 128)).astype(dtype)
    broadcast = queries.astype(mx.float32) @ pooled[:, None].astype(
        mx.float32
    ).swapaxes(-1, -2)
    expected = mx.sum(mx.maximum(broadcast, 0), axis=-2) / math.sqrt(128)
    actual = qsa_fast._portable_indexer_scores(queries, pooled, 128)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()
