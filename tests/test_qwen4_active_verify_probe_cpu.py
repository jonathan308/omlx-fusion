# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the Qwen4 live verify decision trace."""

from __future__ import annotations

from types import SimpleNamespace


def _probe_state(active_ids):
    return SimpleNamespace(
        _omlx_active_verify_parity_probe=SimpleNamespace(
            report={
                "cycle": 0,
                "argmax_parity": True,
                "active": {
                    "rows": [
                        {
                            "active": {"top1_id": token},
                            "active_vs_scalar_argmax": True,
                        }
                        for token in active_ids
                    ],
                    "post_cache_vs_scalar": {"bitwise_equal": True},
                },
                "active_pre_vs_fresh_prefix_cache": {"bitwise_equal": True},
            }
        ),
        hist_offset=0,
        queue=[],
        pending_commit=None,
        pending_emit=None,
    )


def _batch():
    return SimpleNamespace(
        prompt_cache=[],
        tokens=[[]],
    )


def test_active_verify_decision_trace_accepts_canonical_rejection(
    monkeypatch,
):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg
    from omlx.patches.mlx_vlm_qwen4_exp_compat import verify_parity

    reports = []
    monkeypatch.setenv(bg._QWEN4_VERIFY_PARITY_PATH_ENV, "/unused/report.jsonl")
    monkeypatch.setattr(
        verify_parity,
        "append_report",
        lambda _path, report: reports.append(report),
    )
    state = _probe_state([5707, 1156])

    bg._finish_qwen4_active_verify_parity(
        _batch(),
        state,
        target_ids=[5707, 1156],
        draft_ids=[4087],
        accepted=0,
        emitted_id=5707,
    )

    assert reports[0]["decision"]["expected_accepted"] == 0
    assert reports[0]["decision"]["expected_emitted_id"] == 5707
    assert reports[0]["decision"]["alignment_valid"] is True
    assert not hasattr(state, "_omlx_active_verify_parity_probe")


def test_active_verify_decision_trace_exposes_shifted_draft_emit(monkeypatch):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg
    from omlx.patches.mlx_vlm_qwen4_exp_compat import verify_parity

    reports = []
    monkeypatch.setenv(bg._QWEN4_VERIFY_PARITY_PATH_ENV, "/unused/report.jsonl")
    monkeypatch.setattr(
        verify_parity,
        "append_report",
        lambda _path, report: reports.append(report),
    )
    state = _probe_state([5707, 1156])

    bg._finish_qwen4_active_verify_parity(
        _batch(),
        state,
        target_ids=[5707, 1156],
        draft_ids=[4087],
        accepted=1,
        emitted_id=4087,
    )

    assert reports[0]["decision"]["expected_accepted"] == 0
    assert reports[0]["decision"]["expected_emitted_id"] == 5707
    assert reports[0]["decision"]["alignment_valid"] is False
