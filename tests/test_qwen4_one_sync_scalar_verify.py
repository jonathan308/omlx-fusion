# SPDX-License-Identifier: Apache-2.0
"""One-sync canonical scalar-window verification contracts for Qwen4."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from test_qwen4_mtp_terminal_commit import (
    _ack_next,
    _assert_full_cache_parity,
    _real_cycle,
    _wrap_sized_recurrent,
)
from test_qwen4_suffix_local_priming import _assert_target_cache_equal, _model

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import prompt_priming
from omlx.utils.sampling import make_sampler


@pytest.fixture(autouse=True)
def _strict_cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _cycle(monkeypatch, *, one_sync: bool, drafts: int, accepted: int):
    monkeypatch.delenv("OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY", raising=False)
    monkeypatch.delenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", raising=False)
    monkeypatch.setenv(
        (
            "OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY"
            if one_sync
            else "OMLX_QWEN4_SEQUENTIAL_VERIFY"
        ),
        "1",
    )
    mx.random.seed(20260831)
    return _real_cycle(
        accepted,
        draft_depth=drafts,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )


@pytest.mark.parametrize(
    ("drafts", "accepted"),
    [(1, 0), (1, 1), (5, 0), (5, 2), (5, 5)],
)
def test_one_sync_window_is_array_equal_to_sequential_oracle(
    monkeypatch,
    drafts,
    accepted,
):
    one = _cycle(
        monkeypatch,
        one_sync=True,
        drafts=drafts,
        accepted=accepted,
    )
    oracle = _cycle(
        monkeypatch,
        one_sync=False,
        drafts=drafts,
        accepted=accepted,
    )
    _one_bg, _one_model, one_batch, one_state, _one_expected = one
    _oracle_bg, _oracle_model, oracle_batch, oracle_state, _oracle_expected = oracle

    assert [entry[0] for entry in one_state.queue] == [
        entry[0] for entry in oracle_state.queue
    ]
    assert [entry[2] for entry in one_state.queue] == [
        entry[2] for entry in oracle_state.queue
    ]
    for one_entry, oracle_entry in zip(one_state.queue, oracle_state.queue):
        mx.eval(one_entry[1], oracle_entry[1])
        assert mx.array_equal(one_entry[1], oracle_entry[1]).item()

    _assert_target_cache_equal(one_batch.prompt_cache, oracle_batch.prompt_cache)
    mx.eval(
        one_state.next_main,
        oracle_state.next_main,
        one_state.drafts,
        oracle_state.drafts,
    )
    assert mx.array_equal(one_state.next_main, oracle_state.next_main).item()
    assert mx.array_equal(one_state.drafts, oracle_state.drafts).item()
    assert one_state.pending_commit.target_input_ids == (
        oracle_state.pending_commit.target_input_ids
    )


def test_one_sync_window_builds_every_scalar_row_before_one_decision_barrier(
    monkeypatch,
):
    monkeypatch.setenv("OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY", "1")
    monkeypatch.delenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", raising=False)
    calls = []
    barriers = []
    original_call = bg._call_backbone
    original_barrier = bg._qwen4_one_sync_decision_barrier

    def forbidden_detach(*_args, **_kwargs):
        raise AssertionError("one-sync base must retain refs without an eager copy")

    def tracked_call(model, inputs, cache, n_confirmed=0):
        calls.append((tuple(inputs.shape), n_confirmed))
        return original_call(model, inputs, cache, n_confirmed=n_confirmed)

    def tracked_barrier(packet, materialize):
        barriers.append((int(packet.shape[0]), len(materialize)))
        return original_barrier(packet, materialize)

    monkeypatch.setattr(bg, "_call_backbone", tracked_call)
    monkeypatch.setattr(bg, "_qwen4_one_sync_decision_barrier", tracked_barrier)
    monkeypatch.setattr(bg, "_qwen4_detach_snapshot_value", forbidden_detach)
    _bg, _model, batch, state, _expected = _real_cycle(
        0,
        draft_depth=5,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )

    assert calls == [((1, 1), 0)] * 6
    assert len(barriers) == 1
    assert barriers[0][0] == 12  # six target IDs plus six input IDs
    assert barriers[0][1] > 0
    pending = state.pending_commit
    assert pending.verify_width == 6
    assert bg._qwen4_target_offset(batch.prompt_cache) == (
        pending.target_base_offset + 1
    )


def test_one_sync_mid_window_failure_restores_then_uses_wide_verifier(monkeypatch):
    from mlx_lm.models.cache import TokenBuffer

    monkeypatch.setenv("OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY", "1")
    monkeypatch.delenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", raising=False)
    original = bg._call_backbone
    calls = []
    scalar_calls = 0
    context = {}

    class StatefulProcessor:
        def __init__(self):
            self.value = 0
            self.restored = []

        def snapshot_state(self):
            return self.value

        def restore_state(self, value):
            self.value = value
            self.restored.append(value)

        def __call__(self, _tokens, logits):
            self.value += 1
            return logits

    processor = StatefulProcessor()

    def install_processor(batch, _state):
        token_buffer = TokenBuffer(batch.tokens[0][:-1])
        context["batch"] = batch
        context["buffer"] = token_buffer
        context["base_size"] = token_buffer._size
        batch.logits_processors = [[processor]]
        batch._token_context = [token_buffer]

    def fail_second_scalar(model, inputs, cache, n_confirmed=0):
        nonlocal scalar_calls
        if n_confirmed == 1:
            assert processor.value == 0
            assert context["buffer"]._size == context["base_size"] + 6
        result = original(model, inputs, cache, n_confirmed=n_confirmed)
        calls.append((int(inputs.shape[1]), n_confirmed))
        if n_confirmed == 0:
            scalar_calls += 1
            if scalar_calls == 2:
                raise RuntimeError("injected lazy scalar failure")
        return result

    monkeypatch.setattr(bg, "_call_backbone", fail_second_scalar)
    _bg, _model, _batch, state, _expected = _real_cycle(
        2,
        draft_depth=5,
        sized_cache=True,
        expected_pending_kind="verify",
        before_cycle=install_processor,
    )
    assert scalar_calls == 2
    assert any(width == 6 and confirmed == 1 for width, confirmed in calls)
    assert 0 in processor.restored
    assert state.pending_commit.kind == "verify"


def test_one_sync_requires_exact_argmax_marker_not_just_temp_zero(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY", "1")
    model = _model()

    def custom_temp_zero(values):
        return mx.zeros(values.shape[:-1], dtype=mx.uint32)

    custom_temp_zero.temp = 0.0
    custom_temp_zero._omlx_greedy = True
    batch = SimpleNamespace(
        model=model,
        prompt_cache=model.make_cache(),
        samplers=[custom_temp_zero],
        fallback_sampler=None,
    )
    assert not bg._qwen4_one_sync_cycle_eligible(
        batch,
        k=1,
        is_greedy=True,
        two_phase_qwen4=True,
    )

    exact = make_sampler(temp=0.0)
    assert exact._omlx_exact_argmax is True
    batch.samplers = [exact]
    monkeypatch.setattr(bg, "_qwen4_one_sync_world_size", lambda: 2)
    assert not bg._qwen4_one_sync_cycle_eligible(
        batch,
        k=1,
        is_greedy=True,
        two_phase_qwen4=True,
    )

    monkeypatch.setattr(bg, "_qwen4_one_sync_world_size", lambda: 1)
    assert not bg._qwen4_one_sync_cycle_eligible(
        batch,
        k=1,
        is_greedy=True,
        two_phase_qwen4=False,
    )

    batch.prompt_cache = [type("BatchQSAKVCache", (), {})()]
    assert not bg._qwen4_one_sync_cycle_eligible(
        batch,
        k=1,
        is_greedy=True,
        two_phase_qwen4=True,
    )

    batch.prompt_cache = model.make_cache()
    assert bg._qwen4_one_sync_cycle_eligible(
        batch,
        k=1,
        is_greedy=True,
        two_phase_qwen4=True,
    )


def test_one_sync_retained_state_cap_falls_back_without_mutation(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY", "1")
    monkeypatch.delenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", raising=False)
    monkeypatch.setattr(bg, "_QWEN4_ONE_SYNC_RETAINED_BYTES_MAX", 0)
    _bg, _model_obj, _batch, state, _expected = _real_cycle(
        2,
        draft_depth=5,
        sized_cache=True,
        expected_pending_kind="verify",
    )
    assert state.pending_commit.kind == "verify"


def test_one_sync_qsa_capacity_boundary_falls_back_before_scalar_build(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY", "1")
    monkeypatch.delenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", raising=False)

    def shrink_qsa_to_logical_prefix(batch, _state):
        for cache in bg._iter_mtp_cache_leaves(batch.prompt_cache):
            if type(cache).__name__ not in bg._QWEN4_QSA_CACHE_TYPES:
                continue
            offset, _index_offset = bg._qwen4_qsa_offsets(cache)
            cache.keys = mx.contiguous(cache.keys[..., :offset, :])
            cache.values = mx.contiguous(cache.values[..., :offset, :])
            cache._index_keys = mx.contiguous(cache.index_keys)
            cache._index_position_ids = mx.contiguous(cache.index_position_ids)
            cache._index_offset = offset
            cache._geometric_capacity_managed = False
            cache._index_capacity_managed = False

    _bg, _model_obj, _batch, state, _expected = _real_cycle(
        2,
        draft_depth=5,
        sized_cache=True,
        expected_pending_kind="verify",
        before_cycle=shrink_qsa_to_logical_prefix,
    )
    assert state.pending_commit.kind == "verify"


def test_one_sync_pooled_epoch_survives_suffix_trim_and_next_decode(monkeypatch):
    one = _cycle(monkeypatch, one_sync=True, drafts=5, accepted=2)
    oracle = _cycle(monkeypatch, one_sync=False, drafts=5, accepted=2)
    _one_bg, one_model, one_batch, one_state, _one_expected = one
    _oracle_bg, oracle_model, oracle_batch, oracle_state, _oracle_expected = oracle
    expected = bg._qwen4_target_offset(one_batch.prompt_cache)
    assert expected == bg._qwen4_target_offset(oracle_batch.prompt_cache)

    one_qsa = [
        cache
        for cache in bg._iter_mtp_cache_leaves(one_batch.prompt_cache)
        if type(cache).__name__ in bg._QWEN4_QSA_CACHE_TYPES
    ]
    oracle_qsa = [
        cache
        for cache in bg._iter_mtp_cache_leaves(oracle_batch.prompt_cache)
        if type(cache).__name__ in bg._QWEN4_QSA_CACHE_TYPES
    ]
    assert one_qsa and len(one_qsa) == len(oracle_qsa)
    for left, right in zip(one_qsa, oracle_qsa):
        assert left._pooled_index_ratio == right._pooled_index_ratio
        ratio = left._pooled_index_ratio
        assert type(ratio) is int and ratio > 0
        assert left._pooled_index_offset == expected // ratio
        assert left._pooled_index_offset == right._pooled_index_offset
        count = left._pooled_index_offset
        mx.eval(left._pooled_index_keys, right._pooled_index_keys)
        assert mx.array_equal(
            left._pooled_index_keys[:, :count],
            right._pooled_index_keys[:, :count],
        ).item()

    one_logits, one_hidden, _ = bg._call_backbone(
        one_model,
        one_state.next_main[:, None],
        one_batch.prompt_cache,
    )
    oracle_logits, oracle_hidden, _ = bg._call_backbone(
        oracle_model,
        oracle_state.next_main[:, None],
        oracle_batch.prompt_cache,
    )
    mx.eval(
        one_logits,
        one_hidden,
        oracle_logits,
        oracle_hidden,
        *[cache.state for cache in one_batch.prompt_cache],
        *[cache.state for cache in oracle_batch.prompt_cache],
    )
    assert mx.array_equal(one_logits, oracle_logits).item()
    assert mx.array_equal(one_hidden, oracle_hidden).item()
    _assert_target_cache_equal(one_batch.prompt_cache, oracle_batch.prompt_cache)


@pytest.mark.parametrize(("drafts", "accepted"), [(1, 0), (5, 2), (5, 5)])
def test_one_sync_prefix_selection_never_replays_target_rows(
    monkeypatch,
    drafts,
    accepted,
):
    monkeypatch.setenv("OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY", "1")
    monkeypatch.delenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", raising=False)

    def forbidden_replay(*_args, **_kwargs):
        raise AssertionError("one-sync prefix selection must not replay")

    monkeypatch.setattr(bg, "_replay_qwen4_sequential_prefix", forbidden_replay)
    _bg, _model, _batch, state, expected = _real_cycle(
        accepted,
        draft_depth=drafts,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    assert [entry[0] for entry in state.queue] == expected


@pytest.mark.parametrize(
    ("drafts", "accepted"),
    [(1, 0), (1, 1), (5, 0), (5, 2), (5, 5)],
)
def test_one_sync_nonterminal_drain_keeps_skew_and_continues(
    monkeypatch,
    drafts,
    accepted,
):
    monkeypatch.setenv("OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY", "1")
    monkeypatch.delenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", raising=False)
    _bg, _model_obj, batch, state, expected = _real_cycle(
        accepted,
        draft_depth=drafts,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    for _ in expected:
        _token, result = _ack_next(bg, batch, state, terminal=False)
        assert result.handled and not result.exact_terminal
    assert state.pending_commit is None
    assert state.pending_emit is None
    assert not state.queue
    assert bg._qwen4_target_offset(batch.prompt_cache) == len(batch.tokens[0]) - 1
    bg._run_verify_cycle_chain(batch, state)
    assert state.pending_commit is not None
    assert state.pending_commit.kind == "verify-sequential"


@pytest.mark.parametrize(
    ("drafts", "accepted", "terminal_position"),
    [
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (5, 0, 0),
        (5, 2, 0),
        (5, 2, 1),
        (5, 2, 2),
        *[(5, 5, position) for position in range(6)],
    ],
)
def test_one_sync_window_preserves_exact_terminal_base_transaction(
    monkeypatch,
    drafts,
    accepted,
    terminal_position,
):
    monkeypatch.setenv("OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY", "1")
    monkeypatch.delenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", raising=False)
    _bg, model, batch, state, expected = _real_cycle(
        accepted,
        draft_depth=drafts,
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
    assert result.all_tokens == batch.tokens[0]

    reference = _wrap_sized_recurrent(
        model.make_cache(),
        token_count=len(result.all_tokens),
    )
    model._position_ids = None
    model._rope_deltas = None
    with prompt_priming.suppress_capture():
        output = model(
            mx.array(result.all_tokens, dtype=mx.uint32)[None],
            cache=reference,
        )
        mx.eval(output.logits)
    _assert_full_cache_parity(result.prompt_cache, reference)


def test_one_sync_window_is_off_by_default_and_rejects_width_seven(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY", raising=False)
    monkeypatch.delenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", raising=False)
    _bg, _model, _batch, state, _expected = _real_cycle(
        1,
        draft_depth=2,
        expected_pending_kind="verify",
    )
    assert state.pending_commit.sequential_base is None

    monkeypatch.setenv("OMLX_QWEN4_ONE_SYNC_SCALAR_VERIFY", "1")
    _bg, _model, _batch, state, _expected = _real_cycle(
        2,
        draft_depth=6,
        expected_pending_kind="verify",
    )
    assert state.pending_commit.sequential_base is None
