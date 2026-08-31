# SPDX-License-Identifier: Apache-2.0
"""CPU-only lifecycle gates for canonical Qwen4 verify projections."""

from __future__ import annotations

import pytest


def test_verify_qmm_exact_flag_is_thread_local_and_disarm_clears_it():
    from omlx.patches import qwen35_verify_qmm as qmm

    qmm.set_verify_qmm_armed(False)
    assert not qmm._is_armed()
    assert not qmm._is_exact()

    qmm.set_verify_qmm_armed(True, exact=True)
    assert qmm._is_armed()
    assert qmm._is_exact()

    # Even a nonsensical exact=True argument cannot leave exact mode armed
    # after the enclosing target call disarms routing.
    qmm.set_verify_qmm_armed(False, exact=True)
    assert not qmm._is_armed()
    assert not qmm._is_exact()


def test_qwen4_backbone_exception_disarms_exact_verify_gate(monkeypatch):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    calls = []
    monkeypatch.setattr(
        bg,
        "_set_verify_qmm_armed",
        lambda flag, *, exact=False: calls.append((flag, exact)),
    )

    class _FailingQwen4:
        model_type = "qwen4_exp_text"

        def __call__(self, _inputs, **_kwargs):
            raise RuntimeError("controlled target failure")

    with pytest.raises(RuntimeError, match="controlled target failure"):
        bg._call_backbone(
            _FailingQwen4(),
            object(),
            [],
            n_confirmed=1,
        )

    assert calls == [(True, True), (False, False)]


def test_non_qwen_verify_never_arms_exact_projection_mode(monkeypatch):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    calls = []
    monkeypatch.setattr(
        bg,
        "_set_verify_qmm_armed",
        lambda flag, *, exact=False: calls.append((flag, exact)),
    )

    class _FailingGeneric:
        model_type = "qwen3_5"

        def __call__(self, _inputs, **_kwargs):
            raise RuntimeError("controlled target failure")

    with pytest.raises(RuntimeError, match="controlled target failure"):
        bg._call_backbone(
            _FailingGeneric(),
            object(),
            [],
            n_confirmed=1,
        )

    assert calls == [(True, False), (False, False)]
