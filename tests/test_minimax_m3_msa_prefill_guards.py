# SPDX-License-Identifier: Apache-2.0
"""Truth table for the MSA prefill long-K-small-L and MIN_L guards.

A small incremental suffix over a long resident KV stalls the custom
blockwise sparse MSA prefill kernel (captured at L=194 and L=483 in the
2026-07-12 ThunderMLX capacity lab), and tiny multi-token chunks turn the
builder into an O(ctx)-per-step prefill. Both shapes must fall through to the
native MiniMax attention path while cold/large prefill chunks stay on the
accelerated MSA path.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest


@pytest.fixture(autouse=True)
def _vendored_language():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    import mlx_vlm.models.minimax_m3_vl.language as language

    return language


def _eligible(language, *, query_len, total_len):
    """Run the gate with every OTHER condition satisfied, so only the L/KV
    shape under test decides."""

    attention = SimpleNamespace(
        sparse_score_type="max",
        sparse_block_size=64,
        sparse_topk_blocks=16,
        index_heads=2,
    )
    queries = mx.zeros((1, 4, query_len, 64))
    keys = mx.zeros((1, 2, total_len, 64))
    return language.MiniMaxAttention._can_use_msa_prefill_attention(
        attention, queries, keys, None, None
    )


def test_guard_defaults_match_the_production_values(_vendored_language):
    assert _vendored_language._MSA_PREFILL_MIN_L == 16
    assert _vendored_language._MSA_PREFILL_LONG_K_SMALL_L_MAX_L == 512
    assert _vendored_language._MSA_PREFILL_LONG_K_SMALL_L_MIN_KV == 32768


def test_decode_steps_stay_off_the_prefill_path(_vendored_language):
    assert _eligible(_vendored_language, query_len=1, total_len=8192) is False


def test_tiny_chunks_fall_back_to_native_attention(_vendored_language):
    """The MIN_L floor: EAGLE3 verify blocks must not score the full KV."""
    assert _eligible(_vendored_language, query_len=15, total_len=8192) is False
    assert _eligible(_vendored_language, query_len=16, total_len=8192) is True


def test_the_captured_stall_shapes_are_excluded(_vendored_language):
    """L=194 and L=483 over a >=32k resident KV wedged the Metal kernel."""
    assert _eligible(_vendored_language, query_len=194, total_len=32768) is False
    assert _eligible(_vendored_language, query_len=483, total_len=40000) is False


def test_the_guard_window_is_inclusive_of_max_l(_vendored_language):
    assert _eligible(_vendored_language, query_len=512, total_len=32768) is False
    assert _eligible(_vendored_language, query_len=513, total_len=32768) is True


def test_small_suffixes_over_short_kv_keep_the_fast_path(_vendored_language):
    """Below MIN_KV there is no stall — the guard must not over-exclude."""
    assert _eligible(_vendored_language, query_len=100, total_len=32767) is True


def test_cold_full_chunks_stay_on_the_msa_path(_vendored_language):
    assert _eligible(_vendored_language, query_len=2048, total_len=65536) is True


def test_a_zero_max_l_disables_the_long_k_guard(_vendored_language, monkeypatch):
    monkeypatch.setattr(_vendored_language, "_MSA_PREFILL_LONG_K_SMALL_L_MAX_L", 0)
    assert _eligible(_vendored_language, query_len=194, total_len=32768) is True
