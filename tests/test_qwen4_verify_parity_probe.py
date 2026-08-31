# SPDX-License-Identifier: Apache-2.0
"""Qwen4 multi-token target-verify versus scalar-decode diagnostic gate."""

from __future__ import annotations

import mlx.core as mx

from omlx.patches.mlx_vlm_qwen4_exp_compat.verify_parity import (
    compare_qwen4_verify_window,
)
from test_qwen4_suffix_local_priming import _model


def _fixture():
    model = _model()
    prefix = list(range(2, 12))
    # Width six is Qwen3.8's production depth-five target shape.
    window = list(range(12, 18))
    return model, prefix, window


def test_qwen4_verify_parity_probe_covers_logits_layers_and_cache():
    model, prefix, window = _fixture()
    report = compare_qwen4_verify_window(
        model,
        committed_prefix_tokens=prefix,
        verify_tokens=window,
        prefill_step=4,
    )

    assert report["schema_version"] == 1
    assert report["prefix_tokens"] == len(prefix)
    assert report["verify_width"] == 6
    assert report["verify_token_ids"] == window
    assert len(report["rows"]) == 6
    assert len(report["layers"]) == len(model.model.layers)
    assert report["cache_layout_equal"]
    assert report["cache_leaves"]
    assert report["logits"]["left_shape"] == report["logits"]["right_shape"]
    assert all(row["verify"]["margin"] is not None for row in report["rows"])
    assert all(row["scalar"]["margin"] is not None for row in report["rows"])


def test_qwen4_verify_parity_probe_reports_exact_first_argmax_flip(monkeypatch):
    model, prefix, window = _fixture()
    import mlx_vlm.models.qwen4_exp.language as language

    baseline = compare_qwen4_verify_window(
        model,
        committed_prefix_tokens=prefix,
        verify_tokens=window,
        prefill_step=4,
    )
    scalar_id = baseline["rows"][1]["scalar"]["top1_id"]
    forced_id = (scalar_id + 1) % model.args.vocab_size
    original = language._target_verify_linear

    def force_second_verify_row(linear, x, target_verify):
        output = original(linear, x, target_verify)
        if (
            linear is model.lm_head
            and target_verify
            and x.ndim == 3
            and x.shape[1] == len(window)
        ):
            row = (mx.arange(x.shape[1]) == 1).astype(output.dtype)[None, :, None]
            vocab = (
                mx.arange(output.shape[-1]) == forced_id
            ).astype(output.dtype)[None, None, :]
            output = output + row * vocab * mx.array(10_000, dtype=output.dtype)
        return output

    monkeypatch.setattr(language, "_target_verify_linear", force_second_verify_row)
    report = compare_qwen4_verify_window(
        model,
        committed_prefix_tokens=prefix,
        verify_tokens=window,
        prefill_step=4,
    )

    assert not report["argmax_parity"]
    assert report["first_token_mismatch_row"] == 1
    assert report["rows"][1]["verify"]["top1_id"] == forced_id
    assert report["rows"][1]["scalar"]["top1_id"] == scalar_id
