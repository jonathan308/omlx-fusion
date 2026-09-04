# SPDX-License-Identifier: Apache-2.0
"""MTP terminal-cache ownership and exact resident handoff tests."""

from types import SimpleNamespace

import mlx.core as mx

from omlx.patches.mlx_lm_mtp import batch_generator as bg


class _Response(SimpleNamespace):
    pass


class _Batch(SimpleNamespace):
    Response = _Response

    def extract_cache(self, _index):
        return self.prompt_cache

    def filter(self, keep):
        if not keep:
            self.uids = []


def test_unproved_mtp_terminal_never_replays_full_history(monkeypatch):
    state = SimpleNamespace(uid=9)
    calls = []

    def reconcile(batch, observed):
        calls.append((batch, observed))
        raise AssertionError("completed requests must not replay their prompt")

    monkeypatch.setattr(bg, "_reconcile_mtp_to_standard", reconcile)
    batch = _Batch(
        model=SimpleNamespace(),
        uids=[9],
        tokens=[[101]],
        _num_tokens=[0],
        max_tokens=[1],
        state_machines=[SimpleNamespace(match=lambda *_: (None, None, None))],
        _matcher_states=[None],
        _omlx_mtp_state=state,
        prompt_cache=["reconciled-cache"],
    )

    response = bg._emit_response(
        batch,
        token_id=7,
        logprobs_1d=mx.zeros((8,), dtype=mx.float32),
    )[0]

    assert calls == []
    assert response.prompt_cache is None
    assert response.all_tokens is None
    assert not hasattr(response, "_omlx_mtp_standard_terminal_exact")
    assert batch.uids == []


class _OffsetCache:
    def __init__(self, offset):
        self.offset = offset
        self.rollback_state = None
        self._mtp_draft_stash = None
        self._mtp_undo = None
        self._undo = None


def test_exact_generic_terminal_skips_full_replay(monkeypatch):
    state = SimpleNamespace(uid=10)

    def unexpected_reconcile(*_args):
        raise AssertionError("exact target cache must not replay the prompt")

    monkeypatch.setattr(bg, "_reconcile_mtp_to_standard", unexpected_reconcile)
    cache = [_OffsetCache(2)]
    batch = _Batch(
        model=SimpleNamespace(),
        uids=[10],
        tokens=[[101]],
        _num_tokens=[0],
        max_tokens=[1],
        state_machines=[SimpleNamespace(match=lambda *_: (None, None, None))],
        _matcher_states=[None],
        _omlx_mtp_state=state,
        prompt_cache=cache,
    )

    response = bg._emit_response(
        batch,
        token_id=7,
        logprobs_1d=mx.zeros((8,), dtype=mx.float32),
    )[0]

    assert response.prompt_cache is cache
    assert response._omlx_mtp_standard_terminal_exact is True


def test_standard_path_terminal_gets_generic_mtp_proof():
    """A request that never entered MTP is already an exact target terminal."""
    response = SimpleNamespace(
        finish_reason="stop",
        prompt_cache=["target-cache"],
        all_tokens=[1, 2, 3],
    )
    batch = SimpleNamespace(
        _omlx_mtp_state=None,
        _omlx_mtp_batch_state=None,
        _omlx_standard_target_exact_v1=True,
        model=SimpleNamespace(_omlx_mtp_decode_enabled=True),
    )

    stamped = bg._stamp_standard_terminal_responses(batch, [response])

    assert stamped[0]._omlx_mtp_standard_terminal_exact is True


def test_standard_path_does_not_forge_proof_while_mtp_state_is_live():
    response = SimpleNamespace(
        finish_reason="stop",
        prompt_cache=["dirty-cache"],
        all_tokens=[1, 2, 3],
    )
    batch = SimpleNamespace(
        _omlx_mtp_state=SimpleNamespace(uid=1),
        _omlx_mtp_batch_state=None,
        _omlx_standard_target_exact_v1=True,
        model=SimpleNamespace(_omlx_mtp_decode_enabled=True),
    )

    stamped = bg._stamp_standard_terminal_responses(batch, [response])

    assert getattr(stamped[0], "_omlx_mtp_standard_terminal_exact", False) is False
