# SPDX-License-Identifier: Apache-2.0
"""Exact scalar-row Qwen4 GDN target-verification contracts."""

from __future__ import annotations

import mlx.core as mx
import pytest
from test_mlx_vlm_qwen4_exp_compat import _tiny_config

from omlx.cache.type_handlers import SizedArraysCache
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from omlx.patches.mlx_lm_mtp import batch_generator

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import language  # noqa: E402


def _new_cache(token_count: int = 0):
    return SizedArraysCache(language.ArraysCache(size=2), token_count=token_count)


def _assert_cache_arrays_equal(actual, expected):
    mx.eval(actual[0], actual[1], expected[0], expected[1])
    assert mx.array_equal(actual[0], expected[0]).item()
    assert mx.array_equal(actual[1], expected[1]).item()


@pytest.mark.parametrize(
    ("width", "accepted"),
    [(2, 0), (6, 2), (9, 8)],
)
def test_qwen4_tokenwise_gdn_matches_scalar_rows_and_accepted_prefix(
    monkeypatch,
    width,
    accepted,
):
    config = _tiny_config().text_config
    gdn = language.Qwen4ExpGatedDeltaNet(config)
    actual_cache = _new_cache()
    scalar_cache = _new_cache()
    replay_cache = _new_cache()
    mx.eval(gdn.parameters())

    mx.random.seed(20260831 + width)
    prefix = mx.random.normal((1, 3, config.hidden_size)).astype(mx.bfloat16)
    window = mx.random.normal((1, width, config.hidden_size)).astype(mx.bfloat16)
    gdn(prefix, cache=actual_cache)
    gdn(prefix, cache=scalar_cache)
    gdn(prefix, cache=replay_cache)

    scalar_rows = [
        gdn(window[:, row : row + 1], cache=scalar_cache)
        for row in range(width)
    ]
    for row in range(accepted + 1):
        gdn(window[:, row : row + 1], cache=replay_cache)

    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_GDN_VERIFY", "1")
    sink = []
    actual = gdn(
        window,
        cache=actual_cache,
        gdn_sink=sink,
        target_verify=True,
    )
    expected = mx.concatenate(scalar_rows, axis=1)
    mx.eval(actual, expected)

    assert mx.array_equal(actual, expected).item()
    _assert_cache_arrays_equal(actual_cache, scalar_cache)
    assert actual_cache._token_count == 3 + width
    assert len(sink) == 1
    assert len(sink[0]) == 12
    assert sink[0][9].shape == (
        1,
        config.linear_conv_kernel_dim - 1 + width,
        gdn.conv_dim,
    )
    assert sink[0][11].shape == (
        1,
        width,
        config.linear_num_value_heads,
        config.linear_value_head_dim,
        config.linear_key_head_dim,
    )

    # Exercise the unchanged Qwen3.5/Qwen4 full-window rollback ABI.  The
    # recurrent tensors select the exact scalar prefix immediately; Qwen4's
    # outer transaction reconciles the SizedArraysCache timeline separately.
    language.Qwen3_5LanguageModel.rollback_speculative_cache(
        None,
        [actual_cache],
        sink,
        accepted=accepted,
        block_size=width,
    )
    expected_count = 3 + accepted + 1
    assert batch_generator._qwen4_reconcile_sized_recurrent_timeline(
        [actual_cache],
        expected=expected_count,
        allowed_current={3 + width, expected_count},
    )
    assert actual_cache._token_count == expected_count
    _assert_cache_arrays_equal(actual_cache, replay_cache)

    probe = mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
    actual_next = gdn(probe, cache=actual_cache)
    expected_next = gdn(probe, cache=replay_cache)
    mx.eval(actual_next, expected_next)
    assert mx.array_equal(actual_next, expected_next).item()
    _assert_cache_arrays_equal(actual_cache, replay_cache)


@pytest.mark.parametrize(
    ("env", "batch", "width", "target_verify", "has_sink"),
    [
        (None, 1, 2, True, True),
        ("0", 1, 2, True, True),
        ("1", 1, 1, True, True),
        ("1", 1, 10, True, True),
        ("1", 2, 2, True, True),
        ("1", 1, 2, False, False),
        ("1", 1, 2, True, False),
    ],
)
def test_qwen4_tokenwise_gdn_is_strictly_gated(
    monkeypatch,
    env,
    batch,
    width,
    target_verify,
    has_sink,
):
    config = _tiny_config().text_config
    gdn = language.Qwen4ExpGatedDeltaNet(config)
    cache = language.ArraysCache(size=2)
    values = mx.zeros((batch, width, config.hidden_size), dtype=mx.bfloat16)
    sink = [] if has_sink else None
    calls = []
    original = gdn._tokenwise_verify

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(gdn, "_tokenwise_verify", tracked)
    if env is None:
        monkeypatch.delenv("OMLX_QWEN4_TOKENWISE_GDN_VERIFY", raising=False)
    else:
        monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_GDN_VERIFY", env)

    output = gdn(
        values,
        cache=cache,
        gdn_sink=sink,
        target_verify=target_verify,
    )
    mx.eval(output)

    assert calls == []


def test_qwen4_tokenwise_gdn_slices_b1_mask_chronologically(monkeypatch):
    config = _tiny_config().text_config
    gdn = language.Qwen4ExpGatedDeltaNet(config)
    actual_cache = language.ArraysCache(size=2)
    scalar_cache = language.ArraysCache(size=2)
    mx.eval(gdn.parameters())

    mx.random.seed(20260901)
    width = 6
    window = mx.random.normal((1, width, config.hidden_size)).astype(mx.bfloat16)
    mask = mx.array([[True, True, False, True, False, True]])
    expected = mx.concatenate(
        [
            gdn(
                window[:, row : row + 1],
                mask=mask[:, row : row + 1],
                cache=scalar_cache,
            )
            for row in range(width)
        ],
        axis=1,
    )

    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_GDN_VERIFY", "1")
    sink = []
    actual = gdn(
        window,
        mask=mask,
        cache=actual_cache,
        gdn_sink=sink,
        target_verify=True,
    )
    mx.eval(actual, expected)

    assert mx.array_equal(actual, expected).item()
    _assert_cache_arrays_equal(actual_cache, scalar_cache)
    assert mx.array_equal(sink[0][8], mask).item()


def test_qwen4_tokenwise_gdn_preserves_full_model_rollback_contract(monkeypatch):
    config = _tiny_config()
    model = language.LanguageModel(config.text_config, config)
    actual_cache = model.make_cache()
    prefix = mx.array([[2, 3, 4]], dtype=mx.int32)
    window = mx.array([[5, 6, 7, 8, 9, 10]], dtype=mx.int32)
    accepted = 2
    model(prefix, cache=actual_cache)

    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_GDN_VERIFY", "1")
    verified = model(window, cache=actual_cache, return_hidden=True)
    assert len(verified.gdn_states) == 1
    record = verified.gdn_states[0]
    expected_conv = record[9][
        :, accepted + 1 : accepted + config.text_config.linear_conv_kernel_dim
    ]
    expected_state = record[11][:, accepted]
    model.rollback_speculative_cache(
        actual_cache,
        verified.gdn_states,
        accepted=accepted,
        block_size=window.shape[1],
    )

    # Layer zero is the tiny model's GDN+PLE layer.  Its first two slots are
    # the convolution and SSM states governed by the inherited rollback ABI.
    mx.eval(actual_cache[0][0], actual_cache[0][1], expected_conv, expected_state)
    assert mx.array_equal(actual_cache[0][0], expected_conv).item()
    assert mx.array_equal(actual_cache[0][1], expected_state).item()
