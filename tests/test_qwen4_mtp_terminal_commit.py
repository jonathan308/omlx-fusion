# SPDX-License-Identifier: Apache-2.0
"""Exact Qwen4 Lightning-MTP terminal target-cache transactions."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_exact_resident_cache import _request, _scheduler
from test_qwen4_suffix_local_priming import (
    _greedy,
    _model,
    _NoSidecarPrefixCache,
    _suffix_cycle_fixture,
    _target_continuation,
)

from omlx.patches.mlx_lm_mtp import prompt_priming


class _CycleBatch(SimpleNamespace):
    def extract_cache(self, idx):
        # Match production GenerationBatch exactly: resident ownership receives
        # a detached singleton tree even when active decode uses Batch* caches.
        return [cache.extract(idx) for cache in self.prompt_cache]

    def filter(self, keep):
        if not keep:
            self.uids = []


class _BatchGenerator:
    pass


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item)


def _assert_full_cache_parity(actual, expected):
    assert [type(cache) for cache in actual] == [type(cache) for cache in expected]
    families = {"gdn": 0, "ple": 0, "qsa": 0}
    for actual_cache, expected_cache in zip(actual, expected):
        actual_state = list(_arrays(actual_cache.state))
        expected_state = list(_arrays(expected_cache.state))
        assert len(actual_state) == len(expected_state)
        if type(actual_cache).__name__.startswith("QSA"):
            family_by_index = ["qsa"] * len(actual_state)
        else:
            # Qwen4's PLE ArraysCache has GDN conv/recurrent in slots 0/1 and
            # PLE short-conv/history in slots 2/3.
            family_by_index = ["gdn", "gdn", "ple", "ple"][: len(actual_state)]
        mx.eval(*actual_state, *expected_state)
        for family, left, right in zip(
            family_by_index, actual_state, expected_state
        ):
            families[family] += 1
            assert left.shape == right.shape
            assert left.dtype == right.dtype
            if mx.issubdtype(left.dtype, mx.integer):
                assert mx.array_equal(left, right).item()
            else:
                assert mx.allclose(left, right, rtol=1e-3, atol=1e-3).item()
    assert all(count > 0 for count in families.values())


def _assert_cache_arrays_close(actual, expected):
    assert [type(cache) for cache in actual] == [type(cache) for cache in expected]
    compared = 0
    for actual_cache, expected_cache in zip(actual, expected):
        left_values = list(_arrays(actual_cache.state))
        right_values = list(_arrays(expected_cache.state))
        assert len(left_values) == len(right_values)
        mx.eval(*left_values, *right_values)
        for left, right in zip(left_values, right_values):
            compared += 1
            assert left.shape == right.shape
            if mx.issubdtype(left.dtype, mx.integer):
                assert mx.array_equal(left, right).item()
            else:
                assert mx.allclose(left, right, rtol=1e-3, atol=1e-3).item()
    assert compared > 0


def _assert_no_speculative_markers(cache_list):
    pending = list(cache_list)
    while pending:
        cache = pending.pop()
        pending.extend(getattr(cache, "caches", ()) or ())
        for attr in (
            "rollback_state",
            "_qwen4_exp_ple_speculative_state",
            "_mtp_undo",
            "_mtp_draft_stash",
            "_undo",
        ):
            assert getattr(cache, attr, None) is None


def _wrap_sized_recurrent(cache_list, token_count):
    from omlx.cache.type_handlers import SizedArraysCache

    return [
        (
            SizedArraysCache(cache, token_count=token_count)
            if type(cache).__name__ == "ArraysCache"
            else cache
        )
        for cache in cache_list
    ]


def _sized_counts(cache_list):
    counts = []
    pending = list(cache_list)
    while pending:
        cache = pending.pop()
        pending.extend(getattr(cache, "caches", ()) or ())
        if type(cache).__name__ == "SizedArraysCache":
            counts.append(cache._token_count)
    return counts


def _real_cycle(
    accepted: int,
    *,
    boundary: bool = False,
    batched_cache: bool = False,
    sized_cache: bool = False,
    draft_depth: int = 2,
):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    model = _model()
    target_cache, state, history, next_main = _suffix_cycle_fixture(model)
    if sized_cache:
        target_cache = _wrap_sized_recurrent(
            target_cache,
            token_count=len(history),
        )
    if batched_cache:
        from mlx_lm.generate import _merge_caches

        target_cache = _merge_caches([target_cache])
    model._position_ids = None
    model._rope_deltas = None
    oracle_next_main, target_tokens = _target_continuation(model, history, 8)
    assert oracle_next_main == next_main

    assert 0 <= accepted <= draft_depth
    drafts = list(target_tokens[:draft_depth])
    if accepted < draft_depth:
        drafts[accepted] = (drafts[accepted] + 1) % model.args.vocab_size
    state.depth = max(state.depth, draft_depth)
    state.drafts = mx.array(drafts, dtype=mx.uint32)
    state.draft_lps = [
        mx.zeros((model.args.vocab_size,), dtype=mx.float32) for _ in drafts
    ]
    state.draft_accept_lps = list(state.draft_lps)

    tokens = [*history.tolist(), next_main]
    if boundary:
        # The final queue position lands on the next exact block boundary.
        model._omlx_mtp_commit_align = len(tokens) + accepted + 1
    else:
        model._omlx_mtp_commit_align = 0
    batch = _CycleBatch(
        model=model,
        prompt_cache=target_cache,
        tokens=[tokens],
        uids=[1],
        samplers=[None],
        fallback_sampler=_greedy,
        logits_processors=[],
        _token_context=[],
        _num_tokens=[0],
        max_tokens=[512],
        state_machines=[],
        _matcher_states=[],
    )
    batch._omlx_mtp_state = state
    bg._run_verify_cycle_chain(batch, state)
    expected = target_tokens[: accepted + 1]
    assert [entry[0] for entry in state.queue] == expected
    assert state.pending_commit is not None
    assert state.pending_commit.kind == "verify"
    assert state.pending_commit.accepted == accepted
    return bg, model, batch, state, expected


def _ack_next(bg, batch, state, *, terminal: bool, reason: str = "stop"):
    token, _logprobs, source = state.queue.popleft()
    batch.tokens[0].append(token)
    bg._mark_qwen4_pending_emit(state, token, source)
    holder = _BatchGenerator()
    holder._generation_batch = batch
    result = bg._batch_generator_mtp_post_emit(
        holder,
        1,
        terminal=terminal,
        finish_reason=reason,
    )
    return token, result


@pytest.mark.parametrize(
    ("accepted", "terminal_position", "reason"),
    [
        (0, 0, "parser-tool-stop"),
        (1, 0, "stop-string"),
        (1, 1, "stop"),
        (2, 0, "length"),
        (2, 1, "eos"),
        (2, 2, "parser-tool-stop"),
    ],
)
@pytest.mark.parametrize("batched_cache", [False, True])
@pytest.mark.parametrize("sized_cache", [False, True])
def test_real_qwen4_terminal_every_verified_queue_position_is_exact(
    accepted,
    terminal_position,
    reason,
    batched_cache,
    sized_cache,
):
    """Reject/partial/full cycles export exactly the emitted target prefix."""

    bg, model, batch, state, expected = _real_cycle(
        accepted,
        batched_cache=batched_cache,
        sized_cache=sized_cache,
    )
    for position in range(terminal_position):
        token, result = _ack_next(bg, batch, state, terminal=False)
        assert token == expected[position]
        assert result.handled and not result.exact_terminal

    token, result = _ack_next(
        bg,
        batch,
        state,
        terminal=True,
        reason=reason,
    )
    assert token == expected[terminal_position]
    assert result.handled and result.exact_terminal
    assert result.all_tokens == batch.tokens[0]
    assert batch.uids == []
    assert state.queue == deque()
    assert state.pending_commit is None
    assert state.pending_emit is None
    assert prompt_priming.target_cache_offset(result.prompt_cache) == len(
        result.all_tokens
    )
    _assert_no_speculative_markers(result.prompt_cache)

    reference = model.make_cache()
    if sized_cache:
        reference = _wrap_sized_recurrent(
            reference,
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


def test_real_qwen4_nonterminal_queue_drain_commits_once_then_continues():
    bg, _model_obj, batch, state, expected = _real_cycle(1)
    for _position in range(len(expected)):
        _token, result = _ack_next(bg, batch, state, terminal=False)
        assert result.handled and not result.exact_terminal

    assert state.pending_commit is None
    assert state.pending_emit is None
    assert not state.queue
    # Standard speculative pipeline invariant: final correction is emitted but
    # remains the next confirmed verifier input, exactly one token ahead.
    assert bg._qwen4_target_offset(batch.prompt_cache) == len(batch.tokens[0]) - 1
    _assert_no_speculative_markers(batch.prompt_cache)


@pytest.mark.parametrize("accepted", [0, 1, 2])
@pytest.mark.parametrize("sized_cache", [False, True])
def test_real_qwen4_batched_nonterminal_drain_matches_target_oracle(
    accepted,
    sized_cache,
):
    """Production B1 BatchQSA commits the same one-token-skewed target state."""

    bg, model, batch, state, expected = _real_cycle(
        accepted,
        batched_cache=True,
        sized_cache=sized_cache,
    )
    for _ in expected:
        _ack_next(bg, batch, state, terminal=False)

    committed_tokens = batch.tokens[0][:-1]
    actual = batch.extract_cache(0)
    reference = model.make_cache()
    if sized_cache:
        reference = _wrap_sized_recurrent(
            reference,
            token_count=len(committed_tokens),
        )
    model._position_ids = None
    model._rope_deltas = None
    with prompt_priming.suppress_capture():
        output = model(
            mx.array(committed_tokens, dtype=mx.int32)[None],
            cache=reference,
        )
        mx.eval(output.logits)
    _assert_full_cache_parity(actual, reference)
    _assert_no_speculative_markers(actual)
    if sized_cache:
        assert set(_sized_counts(actual)) == {len(committed_tokens)}


def test_qwen4_deferred_adaptive_park_runs_at_final_scheduler_ack():
    """A Qwen4 pending transaction must not swallow the controller exit."""

    bg, model, batch, state, expected = _real_cycle(
        0,
        batched_cache=True,
        sized_cache=True,
    )
    assert len(expected) == 1
    state.park_after_commit = True
    _token, result = _ack_next(bg, batch, state, terminal=False)

    assert result.handled and not result.exact_terminal
    assert not hasattr(batch, "_omlx_mtp_state")
    assert batch._omlx_standard_target_exact_v1 is True
    assert batch._next_tokens is not None
    assert batch._next_logprobs
    assert bg._qwen4_target_offset(batch.prompt_cache) == len(batch.tokens[0])
    _assert_no_speculative_markers(batch.extract_cache(0))

    reference = _wrap_sized_recurrent(
        model.make_cache(),
        token_count=len(batch.tokens[0]),
    )
    model._position_ids = None
    model._rope_deltas = None
    with prompt_priming.suppress_capture():
        output = model(
            mx.array(batch.tokens[0], dtype=mx.int32)[None],
            cache=reference,
        )
        mx.eval(output.logits)
    _assert_full_cache_parity(batch.extract_cache(0), reference)
    assert set(_sized_counts(batch.extract_cache(0))) == {len(batch.tokens[0])}


def test_qwen4_depth5_terminal_reconciles_physical_plus3_sized_metadata():
    """Physical tool stop: full depth-5 verify, terminal at queue position 1."""

    bg, model, batch, state, _expected = _real_cycle(
        5,
        batched_cache=True,
        sized_cache=True,
        draft_depth=5,
    )
    pending = state.pending_commit
    assert pending is not None
    full_offset = pending.target_base_offset + pending.verify_width
    assert set(_sized_counts(batch.prompt_cache)) == {full_offset}

    _ack_next(bg, batch, state, terminal=False)
    _token, result = _ack_next(
        bg,
        batch,
        state,
        terminal=True,
        reason="parser-tool-stop",
    )
    assert result.exact_terminal
    assert full_offset - len(result.all_tokens) == 3
    assert set(_sized_counts(result.prompt_cache)) == {len(result.all_tokens)}

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
    _assert_no_speculative_markers(result.prompt_cache)


@pytest.mark.parametrize("corruption", ["ple", "qsa", "head"])
def test_qwen4_terminal_transaction_corruption_fails_closed(corruption):
    """No incomplete epoch may produce a reusable terminal target cache."""

    bg, _model_obj, batch, state, _expected = _real_cycle(
        1,
        batched_cache=True,
    )
    pending = state.pending_commit
    assert pending is not None
    if corruption == "ple":
        ple_cache, _snapshot = pending.ple_snapshots[0]
        ple_cache._qwen4_exp_ple_speculative_state = None
    elif corruption == "qsa":
        qsa = pending.qsa_snapshots[0].cache
        assert qsa.trim(1) == 1
    else:
        state.hist_offset += 1

    _token, result = _ack_next(
        bg,
        batch,
        state,
        terminal=True,
        reason="stop",
    )
    assert result.handled and not result.exact_terminal
    assert result.prompt_cache is None
    assert result.all_tokens is None
    assert batch.uids == []


def test_qwen4_sized_metadata_mixed_epoch_fails_without_partial_rewrite():
    from omlx.cache.type_handlers import SizedArraysCache
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    model = _model()
    target_cache, _state, history, _next_main = _suffix_cycle_fixture(model)
    recurrent = target_cache[0]
    valid = SizedArraysCache(recurrent.extract(0), token_count=len(history) + 6)
    stale = SizedArraysCache(recurrent.extract(0), token_count=len(history) + 7)
    mixed = [valid, stale, target_cache[1]]

    assert not bg._qwen4_reconcile_sized_recurrent_timeline(
        mixed,
        expected=len(history) + 3,
        allowed_current={len(history) + 6},
    )
    assert valid._token_count == len(history) + 6
    assert stale._token_count == len(history) + 7

    # Raw ArraysCache has no wrapper metadata and remains untouched. Generic
    # models never receive the explicit Qwen4 terminal capability gate.
    assert bg._qwen4_reconcile_sized_recurrent_timeline(
        target_cache,
        expected=len(history),
        allowed_current={len(history)},
    )
    assert not bg._model_qwen4_terminal_commit_enabled(SimpleNamespace())


def test_qwen4_parked_standard_terminal_stamp_requires_exact_response_pair():
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    model = _model()
    batch = SimpleNamespace(
        model=model,
        _omlx_standard_target_exact_v1=True,
    )
    terminal = SimpleNamespace(
        finish_reason="stop",
        prompt_cache=[object()],
        all_tokens=[1, 2, 3],
    )
    live = SimpleNamespace(
        finish_reason=None,
        prompt_cache=None,
        all_tokens=None,
    )
    incomplete = SimpleNamespace(
        finish_reason="length",
        prompt_cache=None,
        all_tokens=[1, 2, 3],
    )

    result = bg._stamp_qwen4_standard_terminal_responses(
        batch,
        [terminal, live, incomplete],
    )
    assert result == [terminal, live, incomplete]
    assert terminal._omlx_qwen4_standard_terminal_v1 is True
    assert not hasattr(live, "_omlx_qwen4_standard_terminal_v1")
    assert not hasattr(incomplete, "_omlx_qwen4_standard_terminal_v1")

    batch._omlx_mtp_state = bg._MtpState(uid=1)
    blocked = SimpleNamespace(
        finish_reason="stop",
        prompt_cache=[object()],
        all_tokens=[1],
    )
    bg._stamp_qwen4_standard_terminal_responses(batch, [blocked])
    assert not hasattr(blocked, "_omlx_qwen4_standard_terminal_v1")


def test_real_qwen4_terminal_target_roundtrips_exact_resident_and_primes_suffix():
    """The proven target-only cache must survive the actual scheduler L0 seam."""

    bg, model, batch, state, expected = _real_cycle(
        2,
        batched_cache=True,
    )
    for _ in expected[:-1]:
        _ack_next(bg, batch, state, terminal=False)
    _token, result = _ack_next(
        bg,
        batch,
        state,
        terminal=True,
        reason="stop",
    )
    assert result.exact_terminal

    scheduler = _scheduler()
    scheduler.model = model
    completed = _request(result.all_tokens)
    completed._mtp_exact_terminal_proved = "qwen4-target-only-v1"
    scheduler._stage_exact_resident_cache(
        completed,
        result.prompt_cache,
        result.all_tokens,
    )
    assert completed._exact_resident_candidate[0] == result.all_tokens
    assert scheduler._publish_exact_resident_cache(completed)

    suffix = [7, 8, 9]
    next_turn = _request([*result.all_tokens, *suffix])
    next_turn.request_id = "qwen4-resident-next-turn"
    assert scheduler._restore_exact_resident_cache(next_turn)
    assert next_turn.prompt_cache is result.prompt_cache
    assert next_turn.cached_tokens == len(result.all_tokens)
    assert next_turn.remaining_tokens == suffix

    prompt_priming.prepare_prefix_context(
        model,
        request_id=next_turn.request_id,
        prompt_tokens=next_turn.prompt_token_ids,
        cached_tokens=next_turn.cached_tokens,
        prefix_cache=_NoSidecarPrefixCache(),
    )
    output = model(
        mx.array(suffix, dtype=mx.int32)[None],
        cache=next_turn.prompt_cache,
    )
    mx.eval(output.logits)
    context = prompt_priming._find_ctx(model)
    assert context is not None
    assert context.request_id == next_turn.request_id
    assert context.prompt_tokens == tuple(next_turn.prompt_token_ids)
    assert context.suffix_local
    assert context.head_hist_offset == len(suffix) - 1
    assert context.target_expected_offset == len(next_turn.prompt_token_ids)


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_real_qwen4_deferred_target_keeps_eager_head_and_target_parity(accepted):
    """Two-phase commit changes scheduling only, never target/head math."""

    bg, model, batch, state, expected = _real_cycle(accepted)
    for _ in expected:
        _ack_next(bg, batch, state, terminal=False)

    # Build an independent eager-control cycle with the same model weights.
    model._position_ids = None
    model._rope_deltas = None
    eager_cache, eager_state, eager_history, eager_next_main = (
        _suffix_cycle_fixture(model)
    )
    _, target_tokens = _target_continuation(model, eager_history, 8)
    eager_drafts = list(target_tokens[:2])
    if accepted < 2:
        eager_drafts[accepted] = (
            eager_drafts[accepted] + 1
        ) % model.args.vocab_size
    eager_state.drafts = mx.array(eager_drafts, dtype=mx.uint32)
    eager_state.draft_lps = [
        mx.zeros((model.args.vocab_size,), dtype=mx.float32) for _ in eager_drafts
    ]
    eager_state.draft_accept_lps = list(eager_state.draft_lps)
    model._omlx_mtp_terminal_commit_v1 = False
    eager_batch = _CycleBatch(
        model=model,
        prompt_cache=eager_cache,
        tokens=[[*eager_history.tolist(), eager_next_main]],
        uids=[2],
        samplers=[None],
        fallback_sampler=_greedy,
        logits_processors=[],
        _token_context=[],
    )
    eager_batch._omlx_mtp_state = eager_state
    bg._run_verify_cycle_chain(eager_batch, eager_state)
    model._omlx_mtp_terminal_commit_v1 = True

    _assert_full_cache_parity(batch.prompt_cache, eager_batch.prompt_cache)
    _assert_cache_arrays_close(state.mtp_cache, eager_state.mtp_cache)
    assert state.hist_offset == eager_state.hist_offset
    assert state.target_expected_offset == eager_state.target_expected_offset
    assert state.next_main.tolist() == eager_state.next_main.tolist()
    assert state.drafts.tolist() == eager_state.drafts.tolist()


def test_real_qwen4_deferred_boundary_and_terminal_tail_are_exact():
    bg, model, batch, state, expected = _real_cycle(
        2,
        boundary=True,
        sized_cache=True,
    )
    pending = state.pending_commit
    assert pending is not None and pending.deferred_boundary
    for _ in range(len(expected)):
        _ack_next(bg, batch, state, terminal=False)

    # Boundary target is now exact and a separately tracked one-token tail was
    # sampled for continued decode.
    assert bg._qwen4_target_offset(batch.prompt_cache) == len(batch.tokens[0])
    assert len(state.queue) == 1
    assert state.pending_commit is not None
    assert state.pending_commit.kind == "tail"
    _tail, result = _ack_next(
        bg,
        batch,
        state,
        terminal=True,
        reason="length",
    )
    assert result.exact_terminal
    assert bg._qwen4_target_offset(result.prompt_cache) == len(result.all_tokens)
    _assert_no_speculative_markers(result.prompt_cache)

    reference = _wrap_sized_recurrent(
        model.make_cache(),
        token_count=len(result.all_tokens),
    )
    model._position_ids = None
    model._rope_deltas = None
    with prompt_priming.suppress_capture():
        output = model(
            mx.array(result.all_tokens, dtype=mx.int32)[None], cache=reference
        )
        mx.eval(output.logits)
    _assert_full_cache_parity(result.prompt_cache, reference)


@pytest.mark.parametrize("terminal_position", [0, 1])
@pytest.mark.parametrize("sized_cache", [False, True])
def test_real_qwen4_separate_init_queue_terminal_is_exact(
    terminal_position,
    sized_cache,
):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    model = _model()
    target_cache, state, history, next_main = _suffix_cycle_fixture(model)
    if sized_cache:
        target_cache = _wrap_sized_recurrent(target_cache, len(history))
    main_id = int(history[-1].item())
    state.queue = deque(
        [
            (main_id, mx.zeros((model.args.vocab_size,)), "init"),
            (next_main, mx.zeros((model.args.vocab_size,)), "init"),
        ]
    )
    state.pending_commit = bg._MtpPendingCommit(
        kind="init",
        target_base_offset=len(history),
        head_base_offset=state.hist_offset - 1,
        verify_width=0,
        accepted=0,
        source_map=("init-resident", "init-tail"),
        token_map=(main_id, next_main),
        final_source="init-tail",
    )
    batch = _CycleBatch(
        model=model,
        prompt_cache=target_cache,
        tokens=[history[:-1].tolist()],
        uids=[1],
    )
    batch._omlx_mtp_state = state
    if terminal_position:
        _ack_next(bg, batch, state, terminal=False)
    _token, result = _ack_next(
        bg,
        batch,
        state,
        terminal=True,
        reason="stop",
    )
    assert result.exact_terminal
    assert bg._qwen4_target_offset(result.prompt_cache) == len(result.all_tokens)
    if sized_cache:
        assert set(_sized_counts(result.prompt_cache)) == {len(result.all_tokens)}
    _assert_no_speculative_markers(result.prompt_cache)


def test_qwen4_pending_verify_blocks_b1_to_batch_and_cancel_never_exports():
    bg, _model_obj, batch, state, _expected = _real_cycle(2)
    batch._omlx_mtp_activation_safe = False
    assert not bg._singleton_mtp_handoff_ready(batch)

    # Cancellation filters/discards the row; without a terminal scheduler ACK
    # there is no cache/token result that either resident tier can publish.
    dropped = bg._drop_mtp_state(batch, "cancel")
    batch.filter([])
    assert dropped is state
    assert batch.uids == []
    assert not hasattr(batch, "_omlx_mtp_state")


def test_qwen4_safe_init_tail_can_handoff_before_b1_to_batch():
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    state = bg._MtpState(
        uid=7,
        queue=deque([(42, mx.zeros((64,)), "init")]),
        pending_commit=bg._MtpPendingCommit(
            kind="init",
            target_base_offset=10,
            head_base_offset=3,
            verify_width=0,
            accepted=0,
            source_map=("init-resident", "init-tail"),
            token_map=(41, 42),
            emitted=1,
        ),
    )
    batch = SimpleNamespace(
        uids=[7],
        _omlx_mtp_state=state,
        _omlx_mtp_activation_safe=False,
    )
    assert bg._singleton_mtp_handoff_ready(batch)


def test_qwen4_init_queue_defers_boundary_until_scheduler_ack():
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    model = _model()
    target_cache, state, history, next_main = _suffix_cycle_fixture(model)
    target_cache = _wrap_sized_recurrent(target_cache, len(history))
    main_id = int(history[-1].item())
    state.queue = deque(
        [
            (main_id, mx.zeros((model.args.vocab_size,)), "init"),
            (next_main, mx.zeros((model.args.vocab_size,)), "init"),
        ]
    )
    state.pending_commit = bg._MtpPendingCommit(
        kind="init",
        target_base_offset=len(history),
        head_base_offset=state.hist_offset - 1,
        verify_width=0,
        accepted=0,
        source_map=("init-resident", "init-tail"),
        token_map=(main_id, next_main),
        deferred_boundary=True,
    )
    batch = _CycleBatch(
        model=model,
        prompt_cache=target_cache,
        tokens=[history[:-1].tolist()],
        uids=[1],
        samplers=[None],
        fallback_sampler=_greedy,
        logits_processors=[],
        _token_context=[],
    )
    batch._omlx_mtp_state = state
    _ack_next(bg, batch, state, terminal=False)
    _ack_next(bg, batch, state, terminal=False)

    assert bg._qwen4_target_offset(batch.prompt_cache) == len(batch.tokens[0])
    assert len(state.queue) == 1
    assert state.pending_commit is not None
    assert state.pending_commit.kind == "tail"
    assert set(_sized_counts(batch.prompt_cache)) == {len(batch.tokens[0])}
