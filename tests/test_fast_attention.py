# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.fast_attention prefill fast paths."""

import mlx.core as mx
import pytest

from omlx.utils.fast_attention import (
    blocked_sliding_window_attention,
    mixed_head_dim_sdpa,
    window_query_padding,
)


def _reference_window_attention(q, k, v, scale, window, sinks):
    L, S = q.shape[2], k.shape[2]
    prefix = S - L
    qpos = mx.arange(prefix, S)[:, None]
    kpos = mx.arange(S)[None, :]
    mask = (kpos <= qpos) & (kpos > qpos - window)
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, sinks=sinks
    )


@pytest.mark.parametrize("prefix", [0, 50, 127, 128, 300])
@pytest.mark.parametrize("dims", [(64, 64), (192, 128)])
@pytest.mark.parametrize("L", [512, 511, 300])
def test_blocked_sliding_window_matches_masked(prefix, dims, L):
    mx.random.seed(prefix + L)
    qk_dim, v_dim = dims
    H, Hk, window = 8, 2, 128
    S = prefix + L
    q = mx.random.normal((1, H, L, qk_dim))
    k = mx.random.normal((1, Hk, S, qk_dim))
    v = mx.random.normal((1, Hk, S, v_dim))
    sinks = mx.random.normal((H,))
    scale = qk_dim**-0.5
    ref = _reference_window_attention(q, k, v, scale, window, sinks)
    out = blocked_sliding_window_attention(
        q, k, v, scale=scale, window=window, sinks=sinks, block=128
    )
    assert out is not None
    assert out.shape == ref.shape
    assert mx.allclose(out, ref, atol=1e-4, rtol=1e-4).item()


@pytest.mark.parametrize("pad", [0, 37])
def test_blocked_sliding_window_honours_padded_bool_mask(pad):
    mx.random.seed(pad)
    H, Hk, L, window, prefix = 4, 2, 384, 128, 100
    S = prefix + L
    q = mx.random.normal((1, H, L, 64))
    k = mx.random.normal((1, Hk, S, 64))
    v = mx.random.normal((1, Hk, S, 64))
    qpos = mx.arange(prefix, S)[:, None]
    kpos = mx.arange(S)[None, :]
    mask = (kpos <= qpos) & (kpos > qpos - window) & (kpos >= pad)
    mask = mask[None, None]
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)
    out = blocked_sliding_window_attention(
        q, k, v, scale=0.125, window=window, mask=mask, block=128
    )
    assert out is not None
    assert mx.allclose(out, ref, atol=1e-4, rtol=1e-4).item()


def test_blocked_sliding_window_declines_unsupported_layouts():
    q = mx.zeros((2, 4, 512, 64))
    k = mx.zeros((2, 4, 512, 64))
    assert (
        blocked_sliding_window_attention(q, k, k, scale=1.0, window=128) is None
    )  # batched
    q = mx.zeros((1, 4, 200, 64))
    k = mx.zeros((1, 4, 200, 64))
    assert (
        blocked_sliding_window_attention(q, k, k, scale=1.0, window=128) is None
    )  # shorter than two blocks


def test_mixed_head_dim_sdpa_matches_unfused():
    from omlx.utils import fast_attention

    if not fast_attention._nax_available():
        pytest.skip("fused padded-V path is only enabled on NAX GPUs")
    mx.random.seed(0)
    q = mx.random.normal((1, 8, 256, 192)).astype(mx.float16)
    k = mx.random.normal((1, 2, 256, 192)).astype(mx.float16)
    v = mx.random.normal((1, 2, 256, 128)).astype(mx.float16)
    sinks = mx.random.normal((8,)).astype(mx.float16)
    scale = 192**-0.5
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal", sinks=sinks
    )
    out = mixed_head_dim_sdpa(q, k, v, scale=scale, mask="causal", sinks=sinks)
    assert out is not None
    assert out.shape == ref.shape
    assert mx.allclose(out, ref, atol=2e-2, rtol=2e-2).item()


def _mixed_dims_reference(q, k, v, scale, mask):
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    mx.eval(out)
    return out


@pytest.mark.parametrize("mask_kind", ["causal", "array"])
def test_mixed_head_dim_sdpa_long_context_all_routes(monkeypatch, mask_kind):
    """Native, JIT NAX and padded-to-256 routes all match the reference."""
    from omlx.utils import fast_attention

    if not fast_attention._nax_available():
        pytest.skip("fast mixed head-dim paths are only enabled on NAX GPUs")
    mx.random.seed(1)
    L, S = 300, 1100
    q = (0.5 * mx.random.normal((1, 8, L, 192))).astype(mx.bfloat16)
    k = (0.5 * mx.random.normal((1, 2, S, 192))).astype(mx.bfloat16)
    v = (0.5 * mx.random.normal((1, 2, S, 128))).astype(mx.bfloat16)
    scale = 192**-0.5
    if mask_kind == "causal":
        mask = "causal"
    else:
        mask = (mx.arange(L)[:, None] + (S - L)) >= mx.arange(S)[None, :]
        mask = mask[None, None]
    ref = _mixed_dims_reference(q, k, v, scale, mask)

    outs = {}
    outs["auto"] = mixed_head_dim_sdpa(q, k, v, scale=scale, mask=mask)
    monkeypatch.setattr(
        fast_attention, "_native_mixed_dims_supported", lambda *a: False
    )
    outs["jit"] = mixed_head_dim_sdpa(q, k, v, scale=scale, mask=mask)
    monkeypatch.setattr(
        fast_attention, "nax_mixed_head_dim_attention", lambda *a, **k: None
    )
    outs["pad256"] = mixed_head_dim_sdpa(q, k, v, scale=scale, mask=mask)
    for name, out in outs.items():
        assert out is not None, name
        assert out.shape == ref.shape, name
        assert mx.allclose(out, ref, atol=2e-2, rtol=2e-2).item(), name


def test_mixed_head_dim_sdpa_native_probe_is_cached():
    from omlx.utils import fast_attention

    fast_attention._native_mixed_dims_supported.cache_clear()
    first = fast_attention._native_mixed_dims_supported(192, 128)
    assert fast_attention._native_mixed_dims_supported(192, 128) is first
    assert fast_attention._native_mixed_dims_supported.cache_info().hits >= 1


def _copied_blocks_reference(q, k, v, scale, window, sinks, block=128):
    """Blocked window attention with every block materialised contiguously."""
    from omlx.utils import fast_attention

    H, L, D = q.shape[1], q.shape[2], q.shape[3]
    S = k.shape[2]
    prefix = S - L
    pad_q = (-L) % block
    nb = (L + pad_q) // block
    used = min(prefix, window)
    lead = window - used
    span = block + window

    def rows(x):
        x = x[:, :, S - L - used :, :]
        return mx.pad(x, [(0, 0), (0, 0), (lead, pad_q), (0, 0)])

    kr, vr = rows(k), rows(v)
    kb = mx.stack([kr[0, :, b * block : b * block + span] for b in range(nb)])
    vb = mx.stack([vr[0, :, b * block : b * block + span] for b in range(nb)])
    qp = mx.pad(q, [(0, 0), (0, 0), (0, pad_q), (0, 0)])
    qb = mx.stack([qp[0, :, b * block : (b + 1) * block] for b in range(nb)])
    fast_attention._BLOCK_MASK_CACHE[0] = None
    block_mask = fast_attention._window_block_mask(
        nb=nb,
        block=block,
        window=window,
        lead=lead,
        user_mask=None,
        col_start=S - L - used,
        pad_q=pad_q,
    )
    # Same kernel route as the blocked path (the JIT NAX kernel for 192/128
    # on stock MLX), fed contiguous copies instead of strided views.
    out = fast_attention._block_sdpa(
        mx.contiguous(qb),
        mx.contiguous(kb),
        mx.contiguous(vb),
        scale=scale,
        mask=block_mask,
        sinks=sinks,
    )
    out = mx.concatenate([out[b] for b in range(nb)], axis=1)[None]
    return out[:, :, :L]


@pytest.mark.parametrize("prefix,L", [(0, 512), (127, 511), (300, 384)])
def test_blocked_sliding_window_strided_views_match_copied_blocks(prefix, L):
    """The overlapping strided key spans feed the kernel the same values."""
    mx.random.seed(7 + prefix)
    H, Hk, window = 8, 2, 128
    S = prefix + L
    q = mx.random.normal((1, H, L, 192)).astype(mx.bfloat16)
    k = mx.random.normal((1, Hk, S, 192)).astype(mx.bfloat16)
    v = mx.random.normal((1, Hk, S, 128)).astype(mx.bfloat16)
    sinks = mx.random.normal((H,)).astype(mx.bfloat16)
    scale = 192**-0.5
    ref = _copied_blocks_reference(q, k, v, scale, window, sinks)
    out = blocked_sliding_window_attention(
        q, k, v, scale=scale, window=window, sinks=sinks
    )
    assert out is not None
    assert out.shape == ref.shape
    assert mx.array_equal(out, ref).item()


def test_blocked_sliding_window_reuses_the_block_mask_per_mask_array():
    from omlx.utils import fast_attention

    mx.random.seed(3)
    H, Hk, L, window, prefix = 4, 2, 384, 128, 100
    S = prefix + L
    q = mx.random.normal((1, H, L, 64))
    k = mx.random.normal((1, Hk, S, 64))
    v = mx.random.normal((1, Hk, S, 64))
    qpos = mx.arange(prefix, S)[:, None]
    kpos = mx.arange(S)[None, :]
    window_mask = (kpos <= qpos) & (kpos > qpos - window)

    fast_attention._BLOCK_MASK_CACHE[0] = None
    mask_a = (window_mask & (kpos >= 5))[None, None]
    out_a = blocked_sliding_window_attention(
        q, k, v, scale=0.125, window=window, mask=mask_a
    )
    cached = fast_attention._BLOCK_MASK_CACHE[0]
    assert cached is not None and cached[1] is mask_a
    again = blocked_sliding_window_attention(
        q, k, v, scale=0.125, window=window, mask=mask_a
    )
    assert fast_attention._BLOCK_MASK_CACHE[0] is cached
    assert mx.array_equal(out_a, again).item()

    # A different mask array (even after the first is dropped) is re-derived.
    del mask_a, again
    mask_b = (window_mask & (kpos >= 60))[None, None]
    out_b = blocked_sliding_window_attention(
        q, k, v, scale=0.125, window=window, mask=mask_b
    )
    ref_b = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask_b)
    assert fast_attention._BLOCK_MASK_CACHE[0][1] is mask_b
    assert mx.allclose(out_b, ref_b, atol=1e-4, rtol=1e-4).item()
    assert not mx.allclose(out_a, out_b, atol=1e-4, rtol=1e-4).item()


@pytest.mark.parametrize(
    "num_queries,expected", [(4095, 1), (8191, 1), (300, 84), (4096, 0), (255, 0), (8, 0)]
)
def test_window_query_padding(num_queries, expected):
    assert window_query_padding(num_queries) == expected


def test_window_query_padding_is_zero_when_disabled(monkeypatch):
    from omlx.utils import fast_attention

    monkeypatch.setattr(fast_attention, "_ENABLED", False)
    assert window_query_padding(4095) == 0


@pytest.mark.parametrize("prefix,L", [(0, 4095), (127, 300), (500, 511), (0, 1024)])
def test_blocked_sliding_window_uses_caller_padded_queries(prefix, L):
    """Queries padded upstream give bit-identical rows (no query copy here)."""
    mx.random.seed(11 + L)
    H, Hk, window = 8, 2, 128
    S = prefix + L
    q = mx.random.normal((1, H, L, 192)).astype(mx.bfloat16)
    k = mx.random.normal((1, Hk, S, 192)).astype(mx.bfloat16)
    v = mx.random.normal((1, Hk, S, 128)).astype(mx.bfloat16)
    sinks = mx.random.normal((H,)).astype(mx.bfloat16)
    pad = window_query_padding(L)
    junk = mx.random.normal((1, H, pad, 192)).astype(mx.bfloat16)  # any values
    qp = mx.concatenate([q, junk], axis=2)
    kwargs = dict(scale=192**-0.5, window=window, sinks=sinks)
    ref = blocked_sliding_window_attention(q, k, v, **kwargs)
    out = blocked_sliding_window_attention(qp, k, v, query_len=L, **kwargs)
    assert ref is not None and out is not None
    assert out.shape == (1, H, L, 128)
    assert mx.array_equal(out, ref).item()


def test_blocked_sliding_window_declines_wrong_caller_padding():
    q = mx.zeros((1, 4, 4095 + 3, 64))
    k = mx.zeros((1, 2, 4095, 64))
    assert (
        blocked_sliding_window_attention(q, k, k, scale=0.125, window=128, query_len=4095)
        is None
    )
