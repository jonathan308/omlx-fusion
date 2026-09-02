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
        return ["reconciled-cache"]

    def filter(self, keep):
        if not keep:
            self.uids = []


def test_mtp_terminal_reconciles_before_publishing_cache(monkeypatch):
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
        state_machines=[SimpleNamespace(match=lambda *_: (None, None, None))],
        _matcher_states=[None],
        _omlx_mtp_state=state,
    )

    response = bg._emit_response(
        batch,
        token_id=7,
        logprobs_1d=mx.zeros((8,), dtype=mx.float32),
    )[0]

    assert calls == [(batch, state)]
    assert response.prompt_cache == ["reconciled-cache"]
    assert response.all_tokens == [101, 7]
    assert response._omlx_mtp_standard_terminal_exact is True
    assert batch.uids == []
