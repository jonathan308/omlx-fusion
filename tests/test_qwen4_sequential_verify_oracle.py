# SPDX-License-Identifier: Apache-2.0
"""Reviewer matrix for the opt-in Qwen4 scalar target oracle.

These tests deliberately exercise the fail-closed seams which are easy to
miss in the end-to-end happy-path suite: admission, bounded prefix selection,
scheduler acknowledgements, exception recovery, and the no-copy QSA snapshot
contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_qwen4_mtp_terminal_commit import (
    _ack_next,
    _assert_no_speculative_markers,
    _CycleBatch,
    _real_cycle,
    _sized_counts,
    _wrap_sized_recurrent,
)
from test_qwen4_suffix_local_priming import (
    _greedy,
    _model,
    _suffix_cycle_fixture,
    _target_continuation,
)

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import prompt_priming


def _prepared_cycle(
    accepted: int,
    *,
    width: int = 2,
    processors=None,
):
    """Build the real tiny-model cycle immediately before target verify."""

    from mlx_lm.models.cache import TokenBuffer

    model = _model()
    target_cache, state, history, next_main = _suffix_cycle_fixture(model)
    target_cache = _wrap_sized_recurrent(
        target_cache,
        token_count=len(history),
    )
    model._position_ids = None
    model._rope_deltas = None
    oracle_next_main, target_tokens = _target_continuation(model, history, width + 2)
    assert oracle_next_main == next_main

    assert 0 <= accepted <= width
    drafts = list(target_tokens[:width])
    if accepted < width:
        drafts[accepted] = (drafts[accepted] + 1) % model.args.vocab_size
    state.depth = max(state.depth, width)
    state.drafts = mx.array(drafts, dtype=mx.uint32)
    zeros = mx.zeros((model.args.vocab_size,), dtype=mx.float32)
    state.draft_lps = [zeros] * width
    state.draft_accept_lps = [zeros] * width

    tokens = [*history.tolist(), next_main]
    processor_list = list(processors or [])
    token_context = [TokenBuffer(tokens)] if processor_list else []
    batch = _CycleBatch(
        model=model,
        prompt_cache=target_cache,
        tokens=[tokens],
        uids=[1],
        samplers=[None],
        fallback_sampler=_greedy,
        logits_processors=[processor_list],
        _token_context=token_context,
        _num_tokens=[0],
        max_tokens=[512],
        state_machines=[],
        _matcher_states=[],
    )
    batch._omlx_mtp_state = state
    return model, batch, state, len(history)


def _primed_capture_fixture(length: int = 40):
    """Build a nontrivial B1 Qwen4 cache with an initialized pooled epoch."""

    model = _model()
    cache = model.make_cache()
    tokens = mx.array(
        [[2 + (index % 30) for index in range(length)]],
        dtype=mx.int32,
    )
    model._position_ids = None
    model._rope_deltas = None
    with prompt_priming.suppress_capture():
        output = model(tokens, cache=cache)
        mx.eval(output.logits)
    batch = SimpleNamespace(model=model, prompt_cache=cache, uids=[71])
    return model, batch, length


def _eligibility_batch(model):
    return SimpleNamespace(
        model=model,
        prompt_cache=model.make_cache(),
        uids=[1],
        samplers=[None],
        fallback_sampler=_greedy,
    )


def test_sequential_oracle_eligibility_is_explicit_and_fail_closed(monkeypatch):
    model = _model()
    batch = _eligibility_batch(model)

    monkeypatch.delenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", raising=False)
    assert not bg._qwen4_sequential_cycle_eligible(
        batch, k=2, is_greedy=True, two_phase_qwen4=True
    )

    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    assert bg._qwen4_sequential_cycle_eligible(
        batch, k=2, is_greedy=True, two_phase_qwen4=True
    )

    batch.fallback_sampler = lambda values: mx.argmax(values, axis=-1)
    assert not bg._qwen4_sequential_cycle_eligible(
        batch, k=2, is_greedy=True, two_phase_qwen4=True
    )
    batch.fallback_sampler = _greedy

    batch_qsa_cache_type = type("BatchQSAKVCache", (), {})
    batch.prompt_cache = [batch_qsa_cache_type()]
    assert not bg._qwen4_sequential_cycle_eligible(
        batch, k=2, is_greedy=True, two_phase_qwen4=True
    )

    batch.prompt_cache = model.make_cache()
    batch.uids = [1, 2]
    assert not bg._qwen4_sequential_cycle_eligible(
        batch, k=2, is_greedy=True, two_phase_qwen4=False
    )
    assert not bg._qwen4_sequential_cycle_eligible(
        batch, k=0, is_greedy=True, two_phase_qwen4=True
    )
    assert not bg._qwen4_sequential_cycle_eligible(
        batch, k=9, is_greedy=True, two_phase_qwen4=True
    )


def test_sequential_oracle_eligibility_rejects_distributed(monkeypatch):
    model = _model()
    batch = _eligibility_batch(model)
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    monkeypatch.setattr(
        mx.distributed,
        "init",
        lambda: SimpleNamespace(size=lambda: 2),
    )
    assert not bg._qwen4_sequential_cycle_eligible(
        batch, k=2, is_greedy=True, two_phase_qwen4=True
    )


@pytest.mark.parametrize(
    ("width", "accepted"),
    [(2, 0), (2, 1), (2, 2), (6, 0), (6, 3), (6, 6)],
    ids=["w2-reject0", "w2-partial", "w2-full", "w6-reject0", "w6-partial", "w6-full"],
)
def test_sequential_oracle_reject_partial_full_matrix(monkeypatch, width, accepted):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    _bg, _model_obj, batch, state, expected = _real_cycle(
        accepted,
        draft_depth=width,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    pending = state.pending_commit
    assert pending.accepted == accepted
    assert pending.verify_width == width + 1
    assert len(state.queue) == accepted + 1
    assert [entry[0] for entry in state.queue] == expected
    expected_offset = pending.target_base_offset + accepted + 1
    assert bg._qwen4_target_offset(batch.prompt_cache) == expected_offset
    assert set(_sized_counts(batch.prompt_cache)) == {expected_offset}


def test_sequential_oracle_clamp_and_boundary_are_bounded(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")

    _bg, _model_obj, batch, state, _expected = _real_cycle(
        5,
        draft_depth=6,
        sized_cache=True,
        clamp_to=2,
        expected_pending_kind="verify-sequential",
    )
    pending = state.pending_commit
    assert pending.accepted == 2
    assert len(state.queue) == 3
    assert bg._qwen4_target_offset(batch.prompt_cache) == pending.target_base_offset + 3

    _bg, _model_obj, batch, state, _expected = _real_cycle(
        6,
        draft_depth=6,
        sized_cache=True,
        alignment_distance=3,
        expected_pending_kind="verify-sequential",
    )
    pending = state.pending_commit
    assert pending.accepted == 3
    assert len(state.queue) == 4
    assert bg._qwen4_target_offset(batch.prompt_cache) == pending.target_base_offset + 4

    # An expanding/invalid model clamp is never trusted. Once canonical target
    # rows have been consumed this is a fail-stop contract, not a silent retry.
    with pytest.raises(
        bg._Qwen4SequentialHardFailure,
        match="clamp returned an invalid prefix",
    ):
        _real_cycle(
            1,
            draft_depth=2,
            sized_cache=True,
            clamp_to=2,
            expected_pending_kind="verify-sequential",
        )


def test_sequential_oracle_normal_ack_commits_only_at_final_queue_receipt(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    _bg, _model_obj, batch, state, expected = _real_cycle(
        3,
        draft_depth=6,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    pending = state.pending_commit
    for position, wanted in enumerate(expected):
        token, result = _ack_next(bg, batch, state, terminal=False)
        assert token == wanted
        assert result.handled and not result.exact_terminal
        if position < len(expected) - 1:
            assert state.pending_commit is pending
            assert not pending.committed
        else:
            assert state.pending_commit is None
            assert pending.committed
    assert state.pending_emit is None
    assert not state.queue
    assert bg._qwen4_target_offset(batch.prompt_cache) == len(batch.tokens[0]) - 1
    _assert_no_speculative_markers(batch.prompt_cache)


_TERMINAL_MATRIX = [
    (width, accepted, position)
    for width, accepted in ((2, 0), (2, 1), (2, 2), (6, 0), (6, 3), (6, 6))
    for position in range(accepted + 1)
]


@pytest.mark.parametrize(("width", "accepted", "position"), _TERMINAL_MATRIX)
def test_sequential_oracle_terminal_is_exact_at_every_queue_position(
    monkeypatch,
    width,
    accepted,
    position,
):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    _bg, _model_obj, batch, state, expected = _real_cycle(
        accepted,
        draft_depth=width,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    for prior in range(position):
        token, result = _ack_next(bg, batch, state, terminal=False)
        assert token == expected[prior]
        assert result.handled and not result.exact_terminal
    token, result = _ack_next(bg, batch, state, terminal=True, reason="tool-stop")
    assert token == expected[position]
    assert result.handled and result.exact_terminal
    assert result.all_tokens == batch.tokens[0]
    assert prompt_priming.target_cache_offset(result.prompt_cache) == len(
        result.all_tokens
    )
    _assert_no_speculative_markers(result.prompt_cache)


def test_sequential_target_logprobs_come_from_each_scalar_target_row(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    calls = []
    original = bg._call_backbone

    def tracked(model, inputs, cache, n_confirmed=0):
        result = original(model, inputs, cache, n_confirmed=n_confirmed)
        calls.append(result[0])
        return result

    monkeypatch.setattr(bg, "_call_backbone", tracked)
    _bg, _model_obj, batch, state, _expected = _real_cycle(
        3,
        draft_depth=6,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    assert len(calls) == len(state.queue) == 4
    for row, (_token, actual, _source) in enumerate(state.queue):
        logits = bg._mtp_prepare_logits(batch, calls[row][:, -1, :])
        canonical = bg._logprobs(logits).squeeze(0)
        mx.eval(actual, canonical)
        assert mx.array_equal(actual, canonical).item()


def test_processor_identity_and_token_buffer_restore_after_scalar_exception(
    monkeypatch,
):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")

    class StatefulProcessor:
        def __init__(self):
            self.count = 0

        def snapshot_state(self):
            return self.count

        def restore_state(self, value):
            self.count = value

        def __call__(self, _tokens, logits):
            self.count += 1
            return logits

    proc = StatefulProcessor()
    _model_obj, batch, state, base = _prepared_cycle(1, processors=[proc])
    base_size = batch._token_context[0]._size
    calls = 0
    original = bg._call_backbone

    def fail_after_second_forward(model, inputs, cache, n_confirmed=0):
        nonlocal calls
        result = original(model, inputs, cache, n_confirmed=n_confirmed)
        calls += 1
        if calls == 2:
            raise RuntimeError("injected scalar failure")
        return result

    monkeypatch.setattr(bg, "_call_backbone", fail_after_second_forward)
    with pytest.raises(bg._Qwen4SequentialRecoveredFallback, match="scalar failure"):
        bg._run_qwen4_sequential_target(
            batch,
            state,
            k=2,
            target_base_offset=base,
            sampler=_greedy,
            procs=[proc],
        )
    assert proc.count == 0
    assert batch._token_context[0]._size == base_size
    assert bg._qwen4_target_offset(batch.prompt_cache) == base

    snapshots = bg._snapshot_qwen4_sequential_processors([proc])
    assert bg._restore_qwen4_sequential_processors([proc], snapshots)
    assert not bg._restore_qwen4_sequential_processors(
        [StatefulProcessor()], snapshots
    )


def test_partial_forward_unprovable_restore_is_a_hard_failure(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    _model_obj, batch, state, base = _prepared_cycle(1)
    original = bg._call_backbone
    calls = 0

    def fail_after_first_partial(model, inputs, cache, n_confirmed=0):
        nonlocal calls
        result = original(model, inputs, cache, n_confirmed=n_confirmed)
        calls += 1
        if calls == 1:
            raise RuntimeError("partial forward")
        return result

    monkeypatch.setattr(bg, "_call_backbone", fail_after_first_partial)
    monkeypatch.setattr(
        bg,
        "_restore_qwen4_sequential_partial_forward",
        lambda *_args, **_kwargs: False,
    )
    with pytest.raises(RuntimeError, match="unprovable live cache"):
        bg._run_qwen4_sequential_target(
            batch,
            state,
            k=2,
            target_base_offset=base,
            sampler=_greedy,
            procs=None,
        )


def test_post_target_staging_failure_restores_then_reconciles_standard(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    _model_obj, batch, state, base = _prepared_cycle(1)

    def fail_staging(*_args, **_kwargs):
        raise RuntimeError("injected head staging failure")

    monkeypatch.setattr(bg, "_chain_next_drafts", fail_staging)
    with pytest.raises(bg._MtpStepFallback, match="post-target staging failed"):
        bg._run_verify_cycle_chain(batch, state)
    assert not state.queue
    assert state.pending_commit is None
    assert state.pending_emit is None
    assert bg._qwen4_target_offset(batch.prompt_cache) == base

    # This is the exact outer patched-next recovery contract: a recovered base
    # may be rebuilt into standard decode, never reused speculatively.
    assert bg._reconcile_mtp_to_standard(batch, state)
    assert batch._omlx_standard_target_exact_v1 is True
    assert bg._qwen4_target_offset(batch.prompt_cache) == len(batch.tokens[0])


class _ArrayShape:
    def __init__(self, ndim, nbytes):
        self.ndim = ndim
        self.nbytes = nbytes
        self.shape = (1,) + (1,) * (ndim - 1)


@pytest.mark.parametrize(
    ("sizes", "message"),
    [
        ((17 * 1024 * 1024, 1, 1, 1), "schema is not bounded"),
        ((9 * 1024 * 1024,) * 4, "exceeds its per-leaf cap"),
    ],
)
def test_recurrent_snapshot_schema_and_byte_caps_fail_closed(sizes, message):
    _model_obj, batch, base = _primed_capture_fixture()
    recurrent = next(
        cache for cache in batch.prompt_cache if type(cache).__name__ == "ArraysCache"
    )
    recurrent.state = [
        _ArrayShape(ndim, nbytes)
        for ndim, nbytes in zip((3, 4, 3, 2), sizes)
    ]
    with pytest.raises(bg._MtpStepFallback, match=message):
        bg._capture_qwen4_sequential_base(batch, base_offset=base)


def test_qsa_snapshot_retains_backing_identity_without_context_copy():
    _model_obj, batch, base = _primed_capture_fixture()
    qsa_cache = next(
        cache for cache in batch.prompt_cache if type(cache).__name__ == "QSAKVCache"
    )
    recurrent_cache = next(
        cache for cache in batch.prompt_cache if type(cache).__name__ == "ArraysCache"
    )
    recurrent_before = tuple(recurrent_cache.state)
    snapshot = bg._capture_qwen4_sequential_base(batch, base_offset=base)
    qsa = snapshot.qsa[0]

    assert qsa.cache is qsa_cache
    assert qsa.keys_backing is qsa_cache.keys
    assert qsa.values_backing is qsa_cache.values
    assert qsa.index_keys_backing is qsa_cache._index_keys
    assert qsa.index_positions_backing is qsa_cache._index_position_ids
    assert qsa.pooled_keys is qsa_cache._pooled_index_keys
    assert all(
        detached is not original
        for detached, original in zip(snapshot.recurrent[0].state, recurrent_before)
    )


def test_pooled_qsa_requires_exact_epoch_and_rejects_pool_none_at_long_base():
    _model_obj, batch, base = _primed_capture_fixture(length=40)
    snapshot = bg._capture_qwen4_sequential_base(batch, base_offset=base)
    qsa = snapshot.qsa[0]
    assert qsa.pooled_offset == base // qsa.pooled_ratio
    assert qsa.pooled_offset > 0
    assert qsa.pooled_keys is not None

    qsa.cache._pooled_index_offset += 1
    with pytest.raises(bg._MtpStepFallback, match="pooled QSA epoch is malformed"):
        bg._capture_qwen4_sequential_base(batch, base_offset=base)

    _model_obj, batch, base = _primed_capture_fixture(length=40)
    qsa_cache = next(
        cache for cache in batch.prompt_cache if type(cache).__name__ == "QSAKVCache"
    )
    qsa_cache._pooled_index_keys = None
    qsa_cache._pooled_index_offset = 0
    qsa_cache._pooled_index_ratio = None
    qsa_cache._pooled_index_tag = None
    with pytest.raises(bg._MtpStepFallback, match="pooled QSA base is malformed"):
        bg._capture_qwen4_sequential_base(batch, base_offset=base)
