# SPDX-License-Identifier: Apache-2.0
"""MiMo decode / MTP-verify attention kernels for long KV caches.

``sdpa_rows``: every row of a short forward in one pass over the KV cache,
bit-identical to MLX's vector kernel (the path it replaces: one call for
rows x GQA <= 32, row chunks beyond).  ``sdpa_flash``: split-key matrix
kernel, MLX's float32 math in another summation order.
"""

import zlib

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.mimo_v2 import decode_fast as df
from omlx.patches.mimo_v2 import sdpa_flash as sf
from omlx.patches.mimo_v2 import sdpa_rows as sr

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or not df._nax_available(),
    reason="MiMo fused decode kernels are validated and enabled on M5 (NAX) GPUs",
)

BF16 = mx.bfloat16
D, DV = 192, 128


def _mlx_sdpa(q, k, v, cache=None, scale=1.0, mask=None, sinks=None):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask, sinks=sinks)


def _today(q, k, v, scale, mask, sinks):
    """The decode path's attention before these kernels (B, L, H * Dv)."""
    B, H, L, _ = q.shape
    rep = H // k.shape[1]
    if L * rep > 32:
        return df._sdpa_row_chunks(_mlx_sdpa, q, k, v, None, scale, mask, sinks, 32 // rep)
    return _mlx_sdpa(q, k, v, scale=scale, mask=mask, sinks=sinks).swapaxes(1, 2).reshape(B, L, -1)


def _inputs(B, H, Hk, L, S, mask_kind, with_sinks, tag):
    mx.random.seed(zlib.crc32(repr(tag).encode()))
    q = (mx.random.normal((B, L, H, D)) * 3).astype(BF16).swapaxes(1, 2)
    # KV-cache views: the head stride exceeds the key count (step slack).
    k = mx.random.normal((B, Hk, S + 256, D)).astype(BF16)[:, :, :S]
    v = mx.random.normal((B, Hk, S + 256, DV)).astype(BF16)[:, :, :S]
    sinks = mx.random.normal((H,)).astype(BF16) if with_sinks else None
    from mlx_lm.models.base import create_causal_mask

    if mask_kind == "none":
        mask = None
    elif mask_kind == "causal":
        mask = "causal"
    elif mask_kind == "window":
        mask = create_causal_mask(L, offset=S - L, window_size=S // 3)
    elif mask_kind == "padded":
        mask = create_causal_mask(L, offset=S - L, left_padding=mx.array([0, 700][:B]))
    else:  # additive
        allowed = create_causal_mask(L, offset=S - L)
        bias = (mx.random.normal((L, S)) * 0.5).astype(BF16)
        mask = mx.where(allowed, bias, mx.array(-mx.inf, dtype=BF16))
    return q, k, v, mask, sinks


def _bits(x):
    return np.array(x.view(mx.uint16))


CASES = [
    # (B, H, Hk, L, S, mask, sinks)
    (1, 64, 4, 1, 1100, "none", False),
    (1, 64, 4, 1, 1100, "none", True),
    (1, 64, 4, 2, 1100, "causal", False),
    (1, 64, 4, 3, 1100, "causal", False),
    (1, 64, 4, 3, 1100, "causal", True),
    (1, 64, 4, 4, 17000, "causal", False),
    (1, 64, 4, 3, 17000, "window", True),
    (1, 64, 4, 3, 5000, "additive", False),
    (2, 64, 4, 3, 5000, "padded", False),
    (2, 64, 4, 2, 5000, "padded", True),
    (1, 64, 8, 3, 5000, "causal", True),
]


@pytest.mark.parametrize("case", CASES)
def test_sdpa_rows_is_bit_identical_to_mlx(case):
    B, H, Hk, L, S, mask_kind, with_sinks = case
    q, k, v, mask, sinks = _inputs(B, H, Hk, L, S, mask_kind, with_sinks, case)
    scale = D ** -0.5
    out = sr.sdpa_rows(q, k, v, scale, mask, sinks)
    assert out is not None
    ref = _today(q, k, v, scale, mask, sinks)
    assert out.shape == ref.shape == (B, L, H * DV)
    assert (_bits(out) != _bits(ref)).sum() == 0


def test_sdpa_rows_declines_outside_mlx_2pass_contract():
    scale = D ** -0.5
    # MLX's single-pass kernel below 1024 keys (on Ultra / Max GPUs).
    q, k, v, _, _ = _inputs(1, 64, 4, 3, 1000, "causal", False, "short")
    if sr._device_class() in ("d", "s"):
        assert sr.sdpa_rows(q, k, v, scale, "causal", None) is None
    # Rows whose MLX calls would pick different key-block counts (a causal
    # 3-row verify at exactly 16384 / 65536 keys) keep today's path.
    if sr._device_class() == "d":
        assert sr.plan_blocks(3, 64, 4, 16384, True) is None
        assert sr.plan_blocks(3, 64, 4, 65536, True) is None
        assert sr.plan_blocks(3, 64, 4, 65537, True) == 1024
        assert sr.plan_blocks(1, 64, 4, 16384, False) == 512
    # MLX's single-row GQA-8 variant (no mask, no sinks) has its own arithmetic.
    q = mx.zeros((1, 32, 1, 128), BF16)
    k = mx.zeros((1, 4, 9000, 128), BF16)
    assert sr.sdpa_rows(q, k, k, 1.0, None, None) is None


def test_mlx_blocks_heuristic_matches_mlx_0_32():
    assert sr.mlx_blocks(8192, 16, "d") == 128
    assert sr.mlx_blocks(16384, 16, "d") == 512
    assert sr.mlx_blocks(65535, 32, "d") == 512
    assert sr.mlx_blocks(65536, 16, "d") == 1024
    assert sr.mlx_blocks(9000, 2, "d") == 256
    assert sr.mlx_blocks(40000, 8, "s") == 512
    assert sr.mlx_blocks(40000, 2, "g") == 32


def _ref64(q, k, v, scale, mask, sinks):
    q = np.array(q.astype(mx.float32)).astype(np.float64)
    k = np.array(k.astype(mx.float32)).astype(np.float64)
    v = np.array(v.astype(mx.float32)).astype(np.float64)
    B, H, L, _ = q.shape
    S = k.shape[2]
    rep = H // k.shape[1]
    q = (np.float32(scale) * q.astype(np.float32)).astype(np.float64)
    if isinstance(mask, str):
        allowed = np.broadcast_to(np.arange(S - L, S)[:, None] >= np.arange(S)[None], (B, L, S))
        bias = np.zeros((B, L, S))
    elif mask is None:
        allowed, bias = np.ones((B, L, S), bool), np.zeros((B, L, S))
    else:
        m = np.array(mask if mask.dtype == mx.bool_ else mask.astype(mx.float32))
        m = np.broadcast_to(m.reshape((1,) * (4 - m.ndim) + m.shape), (B, 1, L, S))[:, 0]
        if mask.dtype == mx.bool_:
            allowed, bias = m, np.zeros((B, L, S))
        else:
            allowed, bias = np.isfinite(m), np.where(np.isfinite(m), m, 0)
    sk = None if sinks is None else np.array(sinks.astype(mx.float32)).astype(np.float64)
    out = np.zeros((B, L, H, v.shape[-1]))
    for b in range(B):
        for h in range(H):
            s = np.where(allowed[b], q[b, h] @ k[b, h // rep].T + bias[b], -np.inf)
            if sk is not None:
                s = np.concatenate([np.full((L, 1), sk[h]), s], axis=1)
            p = np.exp(s - s.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            if sk is not None:
                p = p[:, 1:]
            out[b, :, h] = p @ v[b, h // rep]
    return out.reshape(B, L, -1)


@pytest.mark.parametrize("case", CASES)
def test_sdpa_flash_matches_mlx_to_summation_order(case):
    """Same float32 math as MLX's vector kernel: within two bf16 ULPs of it at
    each head vector's scale (MLX's own bf16-rounded block partials put it up
    to an ULP off float64), and at least as close to float64 as MLX."""
    B, H, Hk, L, S, mask_kind, with_sinks = case
    q, k, v, mask, sinks = _inputs(B, H, Hk, L, S, mask_kind, with_sinks, case)
    scale = D ** -0.5
    out = sf.sdpa_flash(q, k, v, scale, mask, sinks)
    assert out is not None and out.shape == (B, L, H * DV)
    ref = _today(q, k, v, scale, mask, sinks)
    r64 = _ref64(q, k, v, scale, mask, sinks)
    new = np.array(out.astype(mx.float32)).astype(np.float64)
    old = np.array(ref.astype(mx.float32)).astype(np.float64)
    assert np.isfinite(new).all()
    vec = np.abs(r64).reshape(B, L, H, DV).max(axis=-1, keepdims=True)
    ulp = np.broadcast_to(2.0 ** (np.floor(np.log2(np.maximum(vec, 1e-30))) - 7), (B, L, H, DV)).reshape(B, L, -1)
    assert (np.abs(new - old) <= ulp * 2.0001).all()
    assert np.abs(new - r64).max() <= np.abs(old - r64).max() + 1e-6


@pytest.mark.parametrize("S", [1100, 17000])
def test_sdpa_flash_verify_rows_equal_one_row_decodes(S):
    """A causal L-row verify computes each row exactly like the one-row
    decode at that row's position (same key splits and tiles)."""
    for L in (2, 3, 4):
        q, k, v, _, sinks = _inputs(1, 64, 4, L, S, "causal", True, ("verify", S, L))
        scale = D ** -0.5
        verify = sf.sdpa_flash(q, k, v, scale, "causal", sinks)
        rows = []
        for r in range(L):
            n = S - L + r + 1
            assert sf.chunk_size(n) == sf.chunk_size(S)
            rows.append(sf.sdpa_flash(q[:, :, r : r + 1], k[:, :, :n], v[:, :, :n], scale, None, sinks))
        decode = mx.concatenate(rows, axis=1)
        assert (_bits(verify) != _bits(decode)).sum() == 0


def test_sdpa_flash_declines_unsupported_shapes():
    q = mx.zeros((1, 12, 1, 192), BF16)  # GQA 3: bands would mix heads
    k = mx.zeros((1, 4, 5000, 192), BF16)
    v = mx.zeros((1, 4, 5000, 128), BF16)
    assert sf.sdpa_flash(q, k, v, 1.0, None, None) is None
    q = mx.zeros((1, 64, 5, 192), BF16)  # more rows than MAX_ROWS
    assert sf.sdpa_flash(q, k, v, 1.0, "causal", None) is None
    q = mx.zeros((1, 64, 1, 192), mx.float32)
    assert sf.sdpa_flash(q, k.astype(mx.float32), v.astype(mx.float32), 1.0, None, None) is None


@pytest.mark.parametrize("kernel", ["rows", "flash"])
def test_kernels_read_strided_views(kernel):
    """Queries / keys / values whose last axis is not contiguous (read with
    their strides, no copy) give the same result as contiguous copies."""
    B, H, Hk, L, S = 1, 64, 4, 3, 5000
    q, k, v, _, sinks = _inputs(B, H, Hk, L, S, "causal", True, ("strided", kernel))
    kt = mx.contiguous(k.swapaxes(2, 3)).swapaxes(2, 3)  # (B, Hk, S, D) view, inner stride S
    vt = mx.contiguous(v.swapaxes(2, 3)).swapaxes(2, 3)
    qt = mx.contiguous(q.swapaxes(2, 3)).swapaxes(2, 3)
    fn = sr.sdpa_rows if kernel == "rows" else sf.sdpa_flash
    scale = D ** -0.5
    a = fn(q, mx.contiguous(k), mx.contiguous(v), scale, "causal", sinks)
    b = fn(qt, kt, vt, scale, "causal", sinks)
    assert a is not None and b is not None
    assert (_bits(a) != _bits(b)).sum() == 0


def test_sdpa_flash_band_split_is_bit_identical(monkeypatch):
    """One-row forwards split each band over two simdgroups (HS 2), verify
    forwards do not (HS 1); the scores are (first half) + (second half) of
    the head dim either way, so the output bits do not depend on the layout."""
    for L, mask_kind in ((1, "none"), (3, "causal")):
        q, k, v, mask, sinks = _inputs(1, 64, 4, L, 5000, mask_kind, True, ("hs", L))
        outs = []
        for hs in ("1", "2"):
            monkeypatch.setenv("OMLX_SDPA_FLASH_HS", hs)
            outs.append(sf.sdpa_flash(q, k, v, D ** -0.5, mask, sinks))
        assert (_bits(outs[0]) != _bits(outs[1])).sum() == 0
