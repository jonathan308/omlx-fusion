# SPDX-License-Identifier: Apache-2.0
"""CPU-only exactness and rollback gates for tokenwise Qwen4 QSA verify."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from test_qwen4_qsa_decode_gather import _tiny_text_config


compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import language  # noqa: E402


def _assert_qsa_state_equal(actual, expected):
    assert actual.offset == expected.offset
    assert actual._index_offset == expected._index_offset
    for left, right in zip(actual.state, expected.state):
        if left is None or right is None:
            assert left is right
            continue
        mx.eval(left, right)
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        assert mx.array_equal(left, right).item()


def _prefill(attention, cache, prefix):
    output = attention(prefix, mask="causal", cache=cache)
    mx.eval(output)
    return output


def _scalar_rows(attention, cache, values, positions, count=None):
    count = values.shape[1] if count is None else count
    outputs = []
    for row in range(count):
        outputs.append(
            attention(
                values[:, row : row + 1],
                mask="causal",
                cache=cache,
                position_ids=positions[:, row : row + 1],
                target_verify=False,
            )
        )
    return outputs


def test_tokenwise_qsa_output_and_full_cache_are_scalar_exact(monkeypatch):
    config = _tiny_text_config()
    attention = language.Qwen4ExpAttention(config)
    mx.eval(attention.parameters())
    actual_cache = language.QSAKVCache()
    scalar_cache = language.QSAKVCache()

    mx.random.seed(20260831)
    prefix = mx.random.normal((1, 10, config.hidden_size))
    verify = mx.random.normal((1, 6, config.hidden_size))
    positions = mx.arange(10, 16, dtype=mx.int32)[None]
    _prefill(attention, actual_cache, prefix)
    _prefill(attention, scalar_cache, prefix)

    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_QSA_VERIFY", "1")
    monkeypatch.delenv("OMLX_QWEN4_VERIFY_MATCH_DECODE_QSA", raising=False)
    actual = attention(
        verify,
        mask="causal",
        cache=actual_cache,
        position_ids=positions,
        target_verify=True,
    )
    scalar = mx.concatenate(
        _scalar_rows(attention, scalar_cache, verify, positions),
        axis=1,
    )
    mx.eval(actual, scalar)

    assert mx.array_equal(actual, scalar).item()
    _assert_qsa_state_equal(actual_cache, scalar_cache)


@pytest.mark.parametrize("accepted", [0, 2, 3])
def test_tokenwise_qsa_reject_partial_and_full_rollback_are_scalar_exact(
    monkeypatch,
    accepted,
):
    config = _tiny_text_config()
    attention = language.Qwen4ExpAttention(config)
    mx.eval(attention.parameters())
    verify_cache = language.QSAKVCache()
    canonical_cache = language.QSAKVCache()

    mx.random.seed(314159)
    prefix = mx.random.normal((1, 10, config.hidden_size))
    verify = mx.random.normal((1, 4, config.hidden_size))
    positions = mx.arange(10, 14, dtype=mx.int32)[None]
    _prefill(attention, verify_cache, prefix)
    _prefill(attention, canonical_cache, prefix)

    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_QSA_VERIFY", "1")
    verified = attention(
        verify,
        mask="causal",
        cache=verify_cache,
        position_ids=positions,
        target_verify=True,
    )
    mx.eval(verified)

    retained = accepted + 1
    canonical_rows = _scalar_rows(
        attention,
        canonical_cache,
        verify,
        positions,
        count=retained,
    )
    mx.eval(*canonical_rows)
    rolled = language.Qwen3_5LanguageModel.rollback_speculative_cache(
        None,
        [verify_cache],
        [],
        accepted=accepted,
        block_size=verify.shape[1],
    )

    assert rolled == accepted
    _assert_qsa_state_equal(verify_cache, canonical_cache)
    assert verify_cache._pooled_index_offset <= (
        verify_cache.offset // config.indexer_compress_ratio
    )

    probe = mx.random.normal((1, 1, config.hidden_size))
    probe_position = mx.array([[10 + retained]], dtype=mx.int32)
    actual_next = attention(
        probe,
        cache=verify_cache,
        position_ids=probe_position,
    )
    expected_next = attention(
        probe,
        cache=canonical_cache,
        position_ids=probe_position,
    )
    mx.eval(actual_next, expected_next)
    assert mx.array_equal(actual_next, expected_next).item()
    _assert_qsa_state_equal(verify_cache, canonical_cache)


def test_tokenwise_qsa_slices_positions_and_attention_masks_per_row():
    width = 4
    positions = mx.arange(20, 24, dtype=mx.int32)[None]
    sliced_positions = [positions[:, row : row + 1] for row in range(width)]
    assert [value.item() for value in sliced_positions] == [20, 21, 22, 23]

    mask = mx.arange(width * 24, dtype=mx.int32).reshape(1, 1, width, 24)
    rows = [
        language.Qwen4ExpAttention._slice_tokenwise_verify_mask(
            mask,
            row=row,
            width=width,
            visible_tokens=20 + row + 1,
        )
        for row in range(width)
    ]
    mx.eval(*rows)
    assert [value.shape for value in rows] == [
        (1, 1, 1, 21),
        (1, 1, 1, 22),
        (1, 1, 1, 23),
        (1, 1, 1, 24),
    ]
    for row, value in enumerate(rows):
        assert mx.array_equal(value, mask[..., row : row + 1, : 21 + row]).item()


class _FailingScalarAttention(language.Qwen4ExpAttention):
    def __call__(
        self,
        x,
        mask=None,
        cache=None,
        position_ids=None,
        position_embeddings=None,
        target_verify=False,
    ):
        del mask, position_ids, position_embeddings
        assert not target_verify
        row = len(self.scalar_calls)
        self.scalar_calls.append(row)
        cache.offset += 1
        cache._index_offset += 1
        if row == 2:
            raise RuntimeError("controlled QSA scalar-row failure")
        return mx.zeros((1, 1, x.shape[-1]), dtype=x.dtype)


def test_tokenwise_qsa_failure_restores_partial_suffix_for_wide_fallback():
    attention = _FailingScalarAttention.__new__(_FailingScalarAttention)
    object.__setattr__(attention, "scalar_calls", [])
    start = 10
    capacity = 32
    cache = language.QSAKVCache()
    cache.offset = start
    cache._index_offset = start
    cache._index_keys = mx.zeros((1, capacity, 8), dtype=mx.bfloat16)
    cache._index_position_ids = mx.zeros((1, capacity), dtype=mx.int32)
    values = mx.zeros((1, 6, 32), dtype=mx.bfloat16)
    positions = mx.arange(start, start + 6, dtype=mx.int32)[None]

    output = language.Qwen4ExpAttention._tokenwise_text_verify(
        attention,
        values,
        "causal",
        cache,
        positions,
    )

    assert output is None
    assert attention.scalar_calls == [0, 1, 2]
    assert cache.offset == start
    assert cache._index_offset == start


def test_tokenwise_qsa_is_explicitly_opt_in_and_keeps_cache_proofs(monkeypatch):
    config = _tiny_text_config()
    attention = language.Qwen4ExpAttention(config)
    cache = language.QSAKVCache()
    cache.offset = 10
    values = mx.zeros((1, 6, config.hidden_size), dtype=mx.bfloat16)
    eligible = attention._tokenwise_text_verify_eligible

    monkeypatch.delenv("OMLX_QWEN4_TOKENWISE_QSA_VERIFY", raising=False)
    assert not eligible(values, "causal", cache, None, None, True)

    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_QSA_VERIFY", "1")
    assert not eligible(values, "causal", cache, None, None, True)

    cache._index_offset = 10
    cache._index_keys = mx.zeros((1, 10, 8), dtype=mx.bfloat16)
    cache._index_position_ids = mx.zeros((1, 10), dtype=mx.int32)
    assert eligible(values, "causal", cache, None, None, True)
    assert not eligible(values, "causal", cache, None, None, False)
    assert not eligible(values[:, :1], "causal", cache, None, None, True)
    assert not eligible(values, "unsupported", cache, None, None, True)
    assert not eligible(values, "causal", cache, None, (object(),), True)
