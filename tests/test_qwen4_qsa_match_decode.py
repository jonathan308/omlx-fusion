# SPDX-License-Identifier: Apache-2.0
"""Host gates for aligning Qwen4 verifier QSA with scalar decode."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import language  # noqa: E402


def _attention(*, compress_ratio=4, block_topk=512):
    return SimpleNamespace(
        indexer=SimpleNamespace(
            compress_ratio=compress_ratio,
            block_topk=block_topk,
        ),
        _batch_one_text_position_ids=lambda position_ids, length: (
            language.Qwen4ExpAttention._batch_one_text_position_ids(
                position_ids,
                length,
            )
        ),
    )


def _cache(offset: int, *, index_offset: int | None = None):
    cache = language.QSAKVCache()
    cache.offset = offset
    index_offset = offset if index_offset is None else index_offset
    cache.index_keys = mx.zeros((1, index_offset, 8), dtype=mx.bfloat16)
    cache.index_position_ids = mx.zeros((1, index_offset), dtype=mx.int32)
    return cache


def _x(width: int, batch: int = 1):
    return mx.zeros((batch, width, 32), dtype=mx.bfloat16)


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_opt_in_removes_only_the_verify_token_floor(monkeypatch, value):
    monkeypatch.setenv("OMLX_QWEN4_VERIFY_MATCH_DECODE_QSA", value)
    monkeypatch.setattr(language, "_QSA_VERIFY_MATCH_DECODE_LOGGED", False)

    assert language._qsa_verify_sparse_threshold_met(8_198, 2_049, 512)
    assert language._QSA_VERIFY_MATCH_DECODE_LOGGED is True

    # The sparse-block crossover remains mandatory even with the override.
    assert not language._qsa_verify_sparse_threshold_met(8_198, 512, 512)


@pytest.mark.parametrize("value", [None, "", "0", "false", "off", "invalid"])
def test_default_keeps_32768_floor(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OMLX_QWEN4_VERIFY_MATCH_DECODE_QSA", raising=False)
    else:
        monkeypatch.setenv("OMLX_QWEN4_VERIFY_MATCH_DECODE_QSA", value)

    assert not language._qsa_verify_sparse_threshold_met(8_198, 2_049, 512)
    assert language._qsa_verify_sparse_threshold_met(32_768, 8_192, 512)


def test_verify_matches_scalar_decode_sparse_eligibility_when_enabled(monkeypatch):
    attention = _attention()
    cache = _cache(8_192)
    decode = _x(1)
    verify = _x(6)

    assert language.Qwen4ExpAttention._gathered_text_decode_eligible(
        attention,
        decode,
        None,
        cache,
        None,
        None,
        False,
    )

    monkeypatch.delenv("OMLX_QWEN4_VERIFY_MATCH_DECODE_QSA", raising=False)
    assert not language.Qwen4ExpAttention._gathered_text_verify_eligible(
        attention,
        verify,
        None,
        cache,
        None,
        None,
        True,
    )

    monkeypatch.setenv("OMLX_QWEN4_VERIFY_MATCH_DECODE_QSA", "1")
    assert language.Qwen4ExpAttention._gathered_text_verify_eligible(
        attention,
        verify,
        None,
        cache,
        None,
        None,
        True,
    )


def test_override_retains_verify_shape_text_and_cache_proofs(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_VERIFY_MATCH_DECODE_QSA", "1")
    attention = _attention()
    cache = _cache(8_192)
    verify = _x(6)
    eligible = language.Qwen4ExpAttention._gathered_text_verify_eligible

    assert not eligible(attention, verify, None, cache, None, None, False)
    assert not eligible(attention, _x(1), None, cache, None, None, True)
    assert not eligible(attention, _x(10), None, cache, None, None, True)
    assert not eligible(attention, _x(6, batch=2), None, cache, None, None, True)
    assert not eligible(attention, verify, "non-causal", cache, None, None, True)
    assert not eligible(attention, verify, None, cache, None, (object(),), True)

    mrope = mx.zeros((3, 1, 6), dtype=mx.int32)
    assert not eligible(attention, verify, None, cache, mrope, None, True)

    missing = language.QSAKVCache()
    missing.offset = 8_192
    assert not eligible(attention, verify, None, missing, None, None, True)

    misaligned = _cache(8_192, index_offset=8_191)
    assert not eligible(attention, verify, None, misaligned, None, None, True)


def test_override_cannot_cross_before_scalar_decode_is_sparse(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_VERIFY_MATCH_DECODE_QSA", "1")
    attention = _attention(block_topk=512)
    cache = _cache(2_040)

    assert not language.Qwen4ExpAttention._gathered_text_decode_eligible(
        attention,
        _x(1),
        None,
        cache,
        None,
        None,
        False,
    )
    assert not language.Qwen4ExpAttention._gathered_text_verify_eligible(
        attention,
        _x(6),
        None,
        cache,
        None,
        None,
        True,
    )
