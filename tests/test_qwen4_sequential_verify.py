# SPDX-License-Identifier: Apache-2.0
"""Lossless full-model scalar verification oracle contracts for Qwen4."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_qwen4_mtp_terminal_commit import (
    _ack_next,
    _assert_full_cache_parity,
    _assert_no_speculative_markers,
    _real_cycle,
    _sized_counts,
    _wrap_sized_recurrent,
)
from test_qwen4_suffix_local_priming import _greedy, _model

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import prompt_priming


@pytest.mark.parametrize(
    ("width", "accepted"),
    [(2, 0), (2, 1), (2, 2), (6, 0), (6, 2), (6, 6)],
)
def test_sequential_oracle_stops_at_decision_and_returns_target_logprobs(
    monkeypatch,
    width,
    accepted,
):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    calls = []
    original = bg._call_backbone

    def tracked(model, inputs, cache, n_confirmed=0):
        result = original(model, inputs, cache, n_confirmed=n_confirmed)
        calls.append((inputs.shape, n_confirmed, result[0]))
        return result

    monkeypatch.setattr(bg, "_call_backbone", tracked)
    bg_mod, _model_obj, batch, state, expected = _real_cycle(
        accepted,
        draft_depth=width,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )

    assert bg_mod is bg
    assert len(calls) == accepted + 1
    assert all(shape == (1, 1) and confirmed == 0 for shape, confirmed, _ in calls)
    assert [entry[0] for entry in state.queue] == expected
    pending = state.pending_commit
    assert pending.sequential_base is not None
    assert pending.verify_width == width + 1
    assert len(pending.target_input_ids) == width + 1
    assert bg._qwen4_target_offset(batch.prompt_cache) == (
        pending.target_base_offset + accepted + 1
    )
    assert set(_sized_counts(batch.prompt_cache)) == {
        pending.target_base_offset + accepted + 1
    }

    # Accepted drafts must expose the canonical target row, not draft-head
    # logprobs. Row j predicts draft j.
    for row in range(accepted):
        logits = bg._mtp_prepare_logits(batch, calls[row][2][:, -1, :])
        target_lp = bg._logprobs(logits).squeeze(0)
        mx.eval(target_lp, state.queue[row][1])
        assert mx.array_equal(state.queue[row][1], target_lp).item()


_TERMINAL_CASES = [
    (2, 0, 0),
    (2, 1, 0),
    (2, 1, 1),
    (2, 2, 0),
    (2, 2, 1),
    (2, 2, 2),
    (6, 2, 0),
    (6, 2, 1),
    (6, 2, 2),
    *[(6, 6, position) for position in range(7)],
]


@pytest.mark.parametrize(("width", "accepted", "terminal_position"), _TERMINAL_CASES)
def test_sequential_oracle_terminal_every_queue_position_is_exact(
    monkeypatch,
    width,
    accepted,
    terminal_position,
):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    _bg, model, batch, state, expected = _real_cycle(
        accepted,
        draft_depth=width,
        sized_cache=True,
        expected_pending_kind="verify-sequential",
    )
    for position in range(terminal_position):
        token, result = _ack_next(bg, batch, state, terminal=False)
        assert token == expected[position]
        assert result.handled and not result.exact_terminal

    token, result = _ack_next(bg, batch, state, terminal=True)
    assert token == expected[terminal_position]
    assert result.handled and result.exact_terminal
    assert result.all_tokens == batch.tokens[0]
    assert prompt_priming.target_cache_offset(result.prompt_cache) == len(
        result.all_tokens
    )
    _assert_no_speculative_markers(result.prompt_cache)

    reference = _wrap_sized_recurrent(
        model.make_cache(),
        token_count=len(result.all_tokens),
    )
    model._position_ids = None
    model._rope_deltas = None
    with prompt_priming.suppress_capture():
        output = model(
            mx.array(result.all_tokens, dtype=mx.int32)[None],
            cache=reference,
        )
        mx.eval(output.logits)
    _assert_full_cache_parity(result.prompt_cache, reference)


@pytest.mark.parametrize(("width", "accepted"), [(2, 0), (2, 2), (6, 2), (6, 6)])
def test_sequential_oracle_nonterminal_drain_keeps_one_token_skew_and_continues(
    monkeypatch,
    width,
    accepted,
):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    _bg, _model_obj, batch, state, expected = _real_cycle(
        accepted,
        draft_depth=width,
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
    assert set(_sized_counts(batch.prompt_cache)) == {len(batch.tokens[0]) - 1}
    bg._run_verify_cycle_chain(batch, state)
    assert state.pending_commit is not None
    assert state.pending_commit.kind == "verify-sequential"


def test_sequential_oracle_default_is_untouched(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", raising=False)
    _bg, _model_obj, _batch, state, _expected = _real_cycle(
        1,
        draft_depth=2,
        expected_pending_kind="verify",
    )
    assert state.pending_commit.sequential_base is None


def test_sequential_oracle_requires_explicit_greedy_and_single_device(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    model = _model()
    prompt_cache = model.make_cache()
    batch = SimpleNamespace(
        model=model,
        prompt_cache=prompt_cache,
        samplers=[SimpleNamespace(temp=0.0)],
        fallback_sampler=None,
    )
    assert bg._qwen4_sequential_cycle_eligible(
        batch,
        k=1,
        is_greedy=True,
        two_phase_qwen4=True,
    )
    batch.samplers = [lambda values: mx.argmax(values, axis=-1)]
    assert not bg._qwen4_sequential_cycle_eligible(
        batch,
        k=1,
        is_greedy=True,
        two_phase_qwen4=True,
    )
    batch.samplers = [SimpleNamespace(temp=0.5)]
    assert not bg._qwen4_sequential_cycle_eligible(
        batch,
        k=1,
        is_greedy=False,
        two_phase_qwen4=True,
    )


def test_sequential_oracle_recovered_failure_uses_wide_verifier(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_SEQUENTIAL_VERIFY", "1")
    calls = []
    original = bg._call_backbone
    scalar_calls = 0

    def fail_after_second_scalar(model, inputs, cache, n_confirmed=0):
        nonlocal scalar_calls
        result = original(model, inputs, cache, n_confirmed=n_confirmed)
        calls.append((inputs.shape[1], n_confirmed))
        if n_confirmed == 0:
            scalar_calls += 1
            if scalar_calls == 2:
                raise RuntimeError("injected post-forward scalar failure")
        return result

    monkeypatch.setattr(bg, "_call_backbone", fail_after_second_scalar)
    _bg, _model_obj, _batch, state, _expected = _real_cycle(
        1,
        draft_depth=2,
        sized_cache=True,
        expected_pending_kind="verify",
    )
    assert scalar_calls == 2
    assert any(width == 3 and confirmed == 1 for width, confirmed in calls)
    assert state.pending_commit.kind == "verify"


def test_sequential_processor_snapshots_are_identity_aligned():
    class SnapshotOnly:
        def snapshot_state(self):
            return 1

    class RestoreOnly:
        def restore_state(self, _state):
            pass

    with pytest.raises(bg._MtpStepFallback, match="round-trip"):
        bg._snapshot_qwen4_sequential_processors([SnapshotOnly(), RestoreOnly()])

    restored = []

    class RoundTrip:
        def snapshot_state(self):
            return 7

        def restore_state(self, value):
            restored.append(value)

    proc = RoundTrip()
    snapshots = bg._snapshot_qwen4_sequential_processors([proc])
    assert bg._restore_qwen4_sequential_processors([proc], snapshots)
    assert restored == [7]
    assert not bg._restore_qwen4_sequential_processors([RoundTrip()], snapshots)
