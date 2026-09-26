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


def test_mtp_terminal_defers_exact_completion_to_post_emit(monkeypatch):
    """A non-exact generic terminal publishes its natural cache untouched.

    The exact completion (one single-token forward at most) is deferred to
    ``BatchGenerator.omlx_mtp_post_emit`` instead of replaying eagerly inside
    ``_emit_response`` (upstream's terminal contract after the v0.7.0rc1
    merge).
    """
    state = SimpleNamespace(uid=9)
    calls = []

    def reconcile(batch, observed):
        calls.append((batch, observed))
        return True

    monkeypatch.setattr(bg, "_reconcile_mtp_to_standard", reconcile)
    batch = _Batch(
        model=SimpleNamespace(),
        uids=[9],
        tokens=[[101]],
        _num_tokens=[0],
        max_tokens=[1],
        _matchers=[SimpleNamespace(advance=lambda *_: False)],
        _omlx_mtp_state=state,
        prompt_cache=["reconciled-cache"],
    )

    response = bg._emit_response(
        batch,
        token_id=7,
        logprobs_1d=mx.zeros((8,), dtype=mx.float32),
    )[0]

    assert calls == []
    assert response.prompt_cache == ["reconciled-cache"]
    assert response.all_tokens == [101, 7]
    assert getattr(response, "_omlx_mtp_standard_terminal_exact", False) is False
    completion = response._omlx_mtp_terminal_completion
    assert isinstance(completion, bg._MtpTerminalCompletion)
    assert completion.uid == 9
    assert completion.tokens == [101, 7]
    assert completion.prompt_cache == ["reconciled-cache"]
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
        _matchers=[SimpleNamespace(advance=lambda *_: False)],
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
