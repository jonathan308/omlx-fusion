# SPDX-License-Identifier: Apache-2.0
"""Diagnostic scalar-math PLE target-verify contracts."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from test_mlx_vlm_qwen4_exp_compat import _tiny_config


@pytest.mark.parametrize("accepted", [0, 2, 3])
def test_qwen4_tokenwise_ple_matches_scalar_rows_and_rollback(
    monkeypatch,
    accepted,
):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    config = _tiny_config().text_config
    ple = language.Qwen4ExpPLELayer(config, layer_idx=0, ple_layer_index=0)
    actual_cache = language.ArraysCache(size=4)
    scalar_cache = language.ArraysCache(size=4)
    mx.eval(ple.parameters())

    mx.random.seed(20260831)
    width = 4
    hc_hidden = config.hc_count * config.hidden_size
    prefix_hidden = mx.random.normal((1, 3, hc_hidden)).astype(mx.bfloat16)
    prefix_ids = mx.array([[2, 3, 4]], dtype=mx.int32)
    actual_prefix = ple(prefix_hidden, prefix_ids, actual_cache, None)
    scalar_prefix = ple(prefix_hidden, prefix_ids, scalar_cache, None)
    mx.eval(actual_prefix, scalar_prefix)

    verify_hidden = mx.random.normal((1, width, hc_hidden)).astype(mx.bfloat16)
    verify_ids = mx.array([[5, 6, 7, 8]], dtype=mx.int32)
    scalar_rows = []
    scalar_states = []
    for row in range(width):
        scalar_rows.append(
            ple(
                verify_hidden[:, row : row + 1],
                verify_ids[:, row : row + 1],
                scalar_cache,
                None,
            )
        )
        scalar_states.append((scalar_cache[2] + 0, scalar_cache[3] + 0))

    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_PLE_VERIFY", "1")
    actual = ple(
        verify_hidden,
        verify_ids,
        actual_cache,
        None,
        target_verify=True,
    )
    scalar = mx.concatenate(scalar_rows, axis=1)
    snapshot = actual_cache._qwen4_exp_ple_speculative_state
    restored_conv, restored_history = language.LanguageModel._prepare_ple_restore(
        snapshot,
        [accepted],
    )
    expected_conv, expected_history = scalar_states[accepted]
    mx.eval(
        actual,
        scalar,
        restored_conv,
        restored_history,
        expected_conv,
        expected_history,
    )

    assert mx.array_equal(actual, scalar).item()
    assert snapshot.complete
    assert snapshot.input_ids.shape == (1, width)
    assert snapshot.conv_inputs.shape[1] == width
    assert mx.array_equal(restored_conv, expected_conv).item()
    assert mx.array_equal(restored_history, expected_history).item()


def test_qwen4_tokenwise_ple_is_explicitly_opt_in(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    config = _tiny_config().text_config
    ple = language.Qwen4ExpPLELayer(config, layer_idx=0, ple_layer_index=0)
    cache = language.ArraysCache(size=4)
    hidden = mx.zeros(
        (1, 2, config.hc_count * config.hidden_size),
        dtype=mx.bfloat16,
    )
    tokens = mx.array([[2, 3]], dtype=mx.int32)
    calls = []
    original = ple._tokenwise_verify

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(ple, "_tokenwise_verify", tracked)
    monkeypatch.delenv("OMLX_QWEN4_TOKENWISE_PLE_VERIFY", raising=False)
    output = ple(hidden, tokens, cache, None, target_verify=True)
    mx.eval(output)
    assert calls == []
