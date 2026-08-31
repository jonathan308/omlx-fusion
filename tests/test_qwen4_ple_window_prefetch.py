# SPDX-License-Identifier: Apache-2.0
"""Exact PLE lookup prefetch contracts for the Qwen4 scalar oracle."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from test_mlx_vlm_qwen4_exp_compat import _tiny_config
from test_qwen4_mtp_terminal_commit import (
    _ack_next,
    _arrays,
    _assert_no_speculative_markers,
    _real_cycle,
)

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import prompt_priming

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import language  # noqa: E402


class _FakeSSDLookup:
    def __init__(self, dims: int):
        self.dims = dims
        self.calls = 0
        self.rows_read = 0
        self.touched_rows: list[tuple[int, ...]] = []

    def __call__(self, indices: mx.array) -> mx.array:
        self.calls += 1
        mx.eval(indices)
        rows = tuple(int(value) for value in indices.reshape(-1).tolist())
        self.rows_read += len(rows)
        self.touched_rows.append(rows)
        return mx.repeat(
            indices[..., None].astype(mx.float32),
            self.dims,
            axis=-1,
        )

    def reset(self) -> None:
        self.calls = 0
        self.rows_read = 0
        self.touched_rows.clear()


@pytest.mark.parametrize(
    ("base_tokens", "tokens"),
    [
        ([9, 10], [5, 6, 7, 8, 9, 10]),
        ([9, 10], [5, 1, 7, 8, 9, 10]),
        ([9, 10], [1, 6, 1, 8, 1, 10]),
        ([1, 10], [5, 6, 7, 8, 9, 10]),
        ([9, 1], [5, 6, 7, 8, 9, 10]),
    ],
    ids=[
        "ordinary",
        "one-eos",
        "repeated-eos",
        "base-leading-eos",
        "base-trailing-eos",
    ],
)
def test_full_window_ngram_ids_and_embeddings_equal_sequential_history(
    base_tokens,
    tokens,
):
    config = _tiny_config().text_config
    module = language.Qwen4ExpNGramEmbedding(
        config,
        config.ple_embed_dim,
        layer_idx=0,
        ple_layer_index=0,
    )
    lookup = _FakeSSDLookup(config.ple_embed_dim // module.ngram_heads)
    module.ngram_embedding = lookup
    base_history = mx.array([base_tokens], dtype=mx.int64)
    prefetch_cache = language.ArraysCache(size=4)
    scalar_cache = language.ArraysCache(size=4)
    prefetch_cache[3] = base_history
    scalar_cache[3] = base_history + 0
    window = mx.array([tokens], dtype=mx.int32)

    ngram_ids, embeddings = module.prefetch_window(window, prefetch_cache)
    assert prefetch_cache[3] is base_history
    scalar_embeddings = []
    scalar_ids = []
    for token in tokens:
        before = len(lookup.touched_rows)
        scalar_embeddings.append(
            module(mx.array([[token]], dtype=mx.int32), scalar_cache)
        )
        scalar_ids.append(
            mx.array(lookup.touched_rows[before], dtype=mx.int64).reshape(
                1, 1, module.ngram_heads
            )
        )
    expected_ids = mx.concatenate(scalar_ids, axis=1)
    expected_embeddings = mx.concatenate(scalar_embeddings, axis=1)
    mx.eval(ngram_ids, embeddings, expected_ids, expected_embeddings)

    assert mx.array_equal(ngram_ids, expected_ids).item()
    assert mx.array_equal(embeddings, expected_embeddings).item()


@pytest.mark.parametrize(
    ("tokens", "used_rows"),
    [([5, 6], 2), ([5, 6, 7, 8, 9, 10], 2), ([5, 1, 7, 8, 9, 10], 6)],
    ids=["m2-full", "m6-early-reject", "m6-eos-full"],
)
def test_prefetched_ple_outputs_and_live_cache_equal_scalar_rows(tokens, used_rows):
    config = _tiny_config().text_config
    ple = language.Qwen4ExpPLELayer(config, layer_idx=0, ple_layer_index=0)
    lookup = _FakeSSDLookup(
        config.ple_embed_dim // ple.ple_embedding.ngram_heads
    )
    ple.ple_embedding.ngram_embedding = lookup
    mx.eval(ple.parameters())
    actual_cache = language.ArraysCache(size=4)
    scalar_cache = language.ArraysCache(size=4)
    prefix_hidden = mx.zeros(
        (1, 2, config.hc_count * config.hidden_size),
        dtype=mx.bfloat16,
    )
    prefix_ids = mx.array([[3, 4]], dtype=mx.int32)
    ple(prefix_hidden, prefix_ids, actual_cache, None)
    ple(prefix_hidden, prefix_ids, scalar_cache, None)

    mx.random.seed(20260903 + len(tokens) + used_rows)
    hidden = mx.random.normal(
        (1, len(tokens), config.hc_count * config.hidden_size)
    ).astype(mx.bfloat16)
    token_array = mx.array([tokens], dtype=mx.int32)
    scalar_outputs = [
        ple(
            hidden[:, row : row + 1],
            token_array[:, row : row + 1],
            scalar_cache,
            None,
        )
        for row in range(used_rows)
    ]

    lookup.reset()
    ids, embeddings = ple.ple_embedding.prefetch_window(
        token_array,
        actual_cache,
    )
    payload = language._PLEWindowPrefetch(
        token_ids=tuple(tokens),
        entries={
            (id(ple.ple_embedding), id(actual_cache)): language._PLEPrefetchEntry(
                module=ple.ple_embedding,
                cache=actual_cache,
                ngram_ids=ids,
                embeddings=embeddings,
                expected_history=actual_cache[3],
            )
        },
    )
    context_token = language._begin_ple_window_prefetch(payload)
    actual_outputs = []
    try:
        for row in range(used_rows):
            row_ids = token_array[:, row : row + 1]
            assert language._activate_ple_prefetch_row(
                payload,
                row,
                tokens[row],
                row_ids,
            )
            actual_outputs.append(
                ple(
                    hidden[:, row : row + 1],
                    row_ids,
                    actual_cache,
                    None,
                )
            )
            assert language._finish_ple_prefetch_row(payload)
    finally:
        language._end_ple_window_prefetch(context_token)

    actual = mx.concatenate(actual_outputs, axis=1)
    expected = mx.concatenate(scalar_outputs, axis=1)
    actual_state = list(_arrays(actual_cache.state))
    scalar_state = list(_arrays(scalar_cache.state))
    mx.eval(actual, expected, *actual_state, *scalar_state)
    assert mx.array_equal(actual, expected).item()
    assert len(actual_state) == len(scalar_state)
    assert all(
        mx.array_equal(left, right).item()
        for left, right in zip(actual_state, scalar_state)
    )
    assert lookup.calls == 1
    assert lookup.rows_read == len(tokens) * ple.ple_embedding.ngram_heads
    assert payload.entries[(id(ple.ple_embedding), id(actual_cache))].next_row == (
        used_rows
    )
    assert language._PLE_WINDOW_PREFETCH.get() is None


def _cycle_snapshot(batch, state):
    queue = [
        (token, source, logprobs + 0)
        for token, logprobs, source in state.queue
    ]
    cache = [list(_arrays(layer.state)) for layer in batch.prompt_cache]
    mx.eval(
        *[item[2] for item in queue],
        *[value for layer in cache for value in layer],
    )
    return queue, cache


@pytest.mark.parametrize(
    ("target_width", "accepted"),
    [(2, 0), (2, 1), (6, 0), (6, 2), (6, 5)],
)
def test_prefetch_full_oracle_cycle_is_array_equal(
    monkeypatch,
    target_width,
    accepted,
):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    monkeypatch.delenv("OMLX_QWEN4_PLE_WINDOW_PREFETCH", raising=False)
    mx.random.seed(20260910 + target_width + accepted)
    _bg, _model_off, batch_off, state_off, _expected = _real_cycle(
        accepted,
        draft_depth=target_width - 1,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    queue_off, cache_off = _cycle_snapshot(batch_off, state_off)

    prepared = []
    original_prepare = language._prepare_ple_window_prefetch

    def tracked_prepare(*args, **kwargs):
        payload = original_prepare(*args, **kwargs)
        prepared.append(payload)
        return payload

    monkeypatch.setattr(language, "_prepare_ple_window_prefetch", tracked_prepare)
    monkeypatch.setenv("OMLX_QWEN4_PLE_WINDOW_PREFETCH", "1")
    mx.random.seed(20260910 + target_width + accepted)
    _bg, _model_on, batch_on, state_on, _expected = _real_cycle(
        accepted,
        draft_depth=target_width - 1,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    queue_on, cache_on = _cycle_snapshot(batch_on, state_on)

    assert len(prepared) == 1 and prepared[0] is not None
    assert all(
        entry.next_row == accepted + 1
        for entry in prepared[0].entries.values()
    )

    assert [(token, source) for token, source, _ in queue_on] == [
        (token, source) for token, source, _ in queue_off
    ]
    assert all(
        mx.array_equal(left[2], right[2]).item()
        for left, right in zip(queue_on, queue_off)
    )
    assert len(cache_on) == len(cache_off)
    for left_layer, right_layer in zip(cache_on, cache_off):
        assert len(left_layer) == len(right_layer)
        assert all(
            mx.array_equal(left, right).item()
            for left, right in zip(left_layer, right_layer)
        )


@pytest.mark.parametrize(("accepted", "terminal_position"), [(0, 0), (2, 0), (2, 1), (2, 2)])
def test_prefetch_preserves_terminal_and_ack_transactions(
    monkeypatch,
    accepted,
    terminal_position,
):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    monkeypatch.setenv("OMLX_QWEN4_PLE_WINDOW_PREFETCH", "1")
    _bg, _model_obj, batch, state, expected = _real_cycle(
        accepted,
        draft_depth=2,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    for position in range(terminal_position):
        token, result = _ack_next(bg, batch, state, terminal=False)
        assert token == expected[position]
        assert result.handled and not result.exact_terminal
    token, result = _ack_next(bg, batch, state, terminal=True)
    assert token == expected[terminal_position]
    assert result.exact_terminal
    assert prompt_priming.target_cache_offset(result.prompt_cache) == len(
        result.all_tokens
    )
    _assert_no_speculative_markers(result.prompt_cache)
    assert language._PLE_WINDOW_PREFETCH.get() is None


def test_prefetch_setup_failure_falls_back_to_sequential_oracle(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    monkeypatch.setenv("OMLX_QWEN4_PLE_WINDOW_PREFETCH", "1")
    monkeypatch.setattr(
        language,
        "_prepare_ple_window_prefetch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected PLE prefetch failure")
        ),
    )
    _bg, _model_obj, _batch, state, _expected = _real_cycle(
        1,
        draft_depth=2,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    assert state.pending_commit.kind == "verify-sequential"
    assert language._PLE_WINDOW_PREFETCH.get() is None


def test_prefetch_keeps_early_reject_target_compute_scalar(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    monkeypatch.setenv("OMLX_QWEN4_PLE_WINDOW_PREFETCH", "1")
    calls = []
    original = bg._call_backbone

    def tracked(model, inputs, cache, n_confirmed=0):
        calls.append((inputs.shape, n_confirmed))
        return original(model, inputs, cache, n_confirmed=n_confirmed)

    monkeypatch.setattr(bg, "_call_backbone", tracked)
    _bg, _model_obj, _batch, state, _expected = _real_cycle(
        0,
        draft_depth=5,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    assert state.pending_commit.accepted == 0
    assert calls == [((1, 1), 0)]


def test_prefetch_scope_cleans_up_after_backbone_exception(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    monkeypatch.setenv("OMLX_QWEN4_PLE_WINDOW_PREFETCH", "1")
    original = bg._call_backbone
    scalar_calls = 0

    def fail_once(model, inputs, cache, n_confirmed=0):
        nonlocal scalar_calls
        result = original(model, inputs, cache, n_confirmed=n_confirmed)
        if n_confirmed == 0:
            scalar_calls += 1
            if scalar_calls == 1:
                raise RuntimeError("injected scalar failure")
        return result

    monkeypatch.setattr(bg, "_call_backbone", fail_once)
    _bg, _model_obj, _batch, state, _expected = _real_cycle(
        1,
        draft_depth=2,
        sized_cache=True,
        expected_pending_kind="verify",
    )
    assert state.pending_commit.kind == "verify"
    assert language._PLE_WINDOW_PREFETCH.get() is None


def test_prefetch_context_restores_nested_scope():
    outer = language._PLEWindowPrefetch(token_ids=(1, 2), entries={})
    inner = language._PLEWindowPrefetch(token_ids=(3, 4), entries={})
    outer_token = language._begin_ple_window_prefetch(outer)
    try:
        inner_token = language._begin_ple_window_prefetch(inner)
        try:
            assert language._PLE_WINDOW_PREFETCH.get() is inner
        finally:
            language._end_ple_window_prefetch(inner_token)
        assert language._PLE_WINDOW_PREFETCH.get() is outer
    finally:
        language._end_ple_window_prefetch(outer_token)
    assert language._PLE_WINDOW_PREFETCH.get() is None


@pytest.mark.parametrize("mismatch", ["input-identity", "history-epoch"])
def test_prefetch_mismatch_fails_closed_to_ordinary_lookup(mismatch):
    config = _tiny_config().text_config
    module = language.Qwen4ExpNGramEmbedding(
        config,
        config.ple_embed_dim,
        layer_idx=0,
        ple_layer_index=0,
    )
    lookup = _FakeSSDLookup(config.ple_embed_dim // module.ngram_heads)
    module.ngram_embedding = lookup
    cache = language.ArraysCache(size=4)
    cache[3] = mx.array([[7, 8]], dtype=mx.int64)
    window = mx.array([[5, 6]], dtype=mx.int32)
    ids, embeddings = module.prefetch_window(window, cache)
    payload = language._PLEWindowPrefetch(
        token_ids=(5, 6),
        entries={
            (id(module), id(cache)): language._PLEPrefetchEntry(
                module=module,
                cache=cache,
                ngram_ids=ids,
                embeddings=embeddings,
                expected_history=cache[3],
            )
        },
    )
    activated_input = window[:, :1]
    actual_input = (
        activated_input + mx.zeros((), dtype=activated_input.dtype)
        if mismatch == "input-identity"
        else activated_input
    )
    if mismatch == "history-epoch":
        cache[3] = cache[3] + mx.zeros((), dtype=cache[3].dtype)
    context_token = language._begin_ple_window_prefetch(payload)
    try:
        assert language._activate_ple_prefetch_row(
            payload,
            0,
            5,
            activated_input,
        )
        result = module(actual_input, cache)
        assert not language._finish_ple_prefetch_row(payload)
    finally:
        language._end_ple_window_prefetch(context_token)
    mx.eval(result)
    assert payload.disabled
    assert lookup.calls == 2  # one window prefetch, one ordinary scalar fallback


def test_quantized_sharded_embedding_batch_lookup_equals_scalar_rows():
    mx.random.seed(20260930)
    embedding = language.ShardedEmbedding(
        num_embeddings=64,
        dims=32,
        num_shards=4,
    )
    embedding.shards = [
        nn.QuantizedEmbedding.from_embedding(shard, group_size=32, bits=4)
        for shard in embedding.shards
    ]
    indices = mx.array([[1, 18, 35, 52, 18, 1]], dtype=mx.int32)
    batched = embedding(indices)
    scalar = mx.concatenate(
        [embedding(indices[:, row : row + 1]) for row in range(indices.shape[1])],
        axis=1,
    )
    mx.eval(batched, scalar)
    assert batched.shape == scalar.shape == (1, 6, 32)
    assert mx.array_equal(batched, scalar).item()
