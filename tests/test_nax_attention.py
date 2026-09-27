# SPDX-License-Identifier: Apache-2.0
"""Tests for the JIT NAX mixed head-dim attention (omlx.utils.nax_attention)."""

import mlx.core as mx
import pytest

from omlx.utils import fast_attention, nax_attention
from omlx.utils.nax_attention import nax_mixed_head_dim_attention

requires_nax = pytest.mark.skipif(
    not nax_attention._nax_available(), reason="NAX (M5) GPU required"
)


def _reference(q, k, v, scale, mask=None, sinks=None):
    """fp32 attention: exact softmax (with optional sinks) and fp32 products."""
    B, H, qL, D = q.shape
    Hk, kL = k.shape[1], k.shape[2]
    g = H // Hk
    s = (q.astype(mx.float32).reshape(B, Hk, g, qL, D) * scale) @ k.astype(
        mx.float32
    )[:, :, None].swapaxes(-1, -2)
    if isinstance(mask, str):
        m = (mx.arange(qL)[:, None] + (kL - qL)) >= mx.arange(kL)[None]
        s = mx.where(m, s, -mx.inf)
    elif mask is not None:
        m = mx.broadcast_to(mask, (B, H, qL, kL)).reshape(B, Hk, g, qL, kL)
        s = mx.where(m, s, -mx.inf)
    top = s.max(-1, keepdims=True)
    if sinks is not None:
        sk = sinks.astype(mx.float32).reshape(1, Hk, g, 1, 1)
        top = mx.maximum(top, sk)
        p = mx.exp(s - top)
        den = p.sum(-1, keepdims=True) + mx.exp(sk - top)
    else:
        p = mx.exp(s - top)
        den = p.sum(-1, keepdims=True)
    out = (p @ v.astype(mx.float32)[:, :, None]) / den
    return out.reshape(B, H, qL, -1)


def _inputs(B, H, Hk, qL, kL, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    q = (0.5 * mx.random.normal((B, H, qL, 192))).astype(dtype)
    k = (0.5 * mx.random.normal((B, Hk, kL, 192))).astype(dtype)
    v = (0.5 * mx.random.normal((B, Hk, kL, 128))).astype(dtype)
    return q, k, v


def _max_err(out, ref):
    return mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()


# (B, H, Hk, qL, kL, mask, sinks): unaligned query/key tails, GQA, prefix
# (qL < kL), bool masks, sinks and a batch > 1.
_CASES = [
    (1, 8, 2, 16, 512, "causal", False),
    (1, 8, 2, 2049, 2049, "causal", True),
    (1, 8, 2, 1031, 4096, "causal", False),
    (1, 8, 2, 1024, 1024, None, False),
    (1, 8, 2, 255, 4095, "array", True),
    (2, 8, 4, 300, 700, "array", False),
    (1, 8, 8, 64, 96, "causal", True),
]


@requires_nax
@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_matches_fp32_reference(case, dtype):
    B, H, Hk, qL, kL, mask_kind, has_sinks = case
    q, k, v = _inputs(B, H, Hk, qL, kL, dtype, seed=qL + kL)
    mask = mask_kind
    if mask_kind == "array":
        mask = mx.random.uniform(shape=(B, 1, qL, kL)) > 0.3
        mask[..., 0] = True
    sinks = (2 * mx.random.normal((H,))).astype(dtype) if has_sinks else None
    scale = 192**-0.5
    out = nax_mixed_head_dim_attention(q, k, v, scale=scale, mask=mask, sinks=sinks)
    assert out is not None
    assert out.shape == (B, H, qL, 128) and out.dtype == dtype
    ref = _reference(q, k, v, scale, mask, sinks)
    # bf16/fp16 output rounding plus the tensor-unit P @ V products.
    assert _max_err(out, ref) < (1e-2 if dtype == mx.bfloat16 else 2e-3)


@requires_nax
def test_bit_exact_with_native_mlx_kernel():
    if not fast_attention._native_mixed_dims_supported(192, 128):
        pytest.skip("this MLX build has no native 192/128 NAX attention kernel")
    for i, (B, H, Hk, qL, kL, mask_kind, has_sinks) in enumerate(_CASES):
        q, k, v = _inputs(B, H, Hk, qL, kL, seed=i)
        mask = mask_kind
        if mask_kind == "array":
            mask = mx.random.uniform(shape=(B, 1, qL, kL)) > 0.3
        sinks = mx.random.normal((H,)).astype(mx.bfloat16) if has_sinks else None
        out = nax_mixed_head_dim_attention(
            q, k, v, scale=0.07, mask=mask, sinks=sinks
        )
        native = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=0.07, mask=mask, sinks=sinks
        )
        assert mx.array_equal(out, native).item(), (B, H, Hk, qL, kL, mask_kind)


@requires_nax
def test_strided_inputs_are_read_in_place():
    """KV-cache slices, [B, L, H, D] query rows and overlapping key windows."""
    B, H, Hk, qL, kL = 1, 8, 2, 200, 450
    q, k, v = _inputs(B, H, Hk, qL, kL, seed=3)
    scale = 192**-0.5
    expected = nax_mixed_head_dim_attention(q, k, v, scale=scale, mask="causal")

    # Queries as a transposed projection output, keys/values as slices of a
    # larger preallocated cache (strided heads).
    q_rows = mx.contiguous(q.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    k_cache = mx.zeros((B, Hk, 1024, 192), dtype=k.dtype)
    v_cache = mx.zeros((B, Hk, 1024, 128), dtype=v.dtype)
    k_cache[:, :, :kL] = k
    v_cache[:, :, :kL] = v
    out = nax_mixed_head_dim_attention(
        q_rows, k_cache[:, :, :kL], v_cache[:, :, :kL], scale=scale, mask="causal"
    )
    assert mx.array_equal(out, expected).item()

    # Overlapping per-block key windows (the blocked sliding-window layout).
    block, window, nb = 64, 64, 3
    rows = window + nb * block
    kr = k[:, :, :rows]
    vr = v[:, :, :rows]
    span = block + window
    kb = mx.as_strided(kr, (nb, Hk, span, 192), (block * 192, rows * 192, 192, 1))
    vb = mx.as_strided(vr, (nb, Hk, span, 128), (block * 128, rows * 128, 128, 1))
    qb = q[:, :, : nb * block].reshape(H, nb, block, 192).transpose(1, 0, 2, 3)
    got = nax_mixed_head_dim_attention(qb, kb, vb, scale=scale, mask=None)
    want = nax_mixed_head_dim_attention(
        mx.contiguous(qb), mx.contiguous(kb), mx.contiguous(vb), scale=scale
    )
    assert mx.array_equal(got, want).item()

    # A strided sinks vector is read by value.
    sinks = mx.random.normal((2 * H,)).astype(mx.bfloat16)
    got = nax_mixed_head_dim_attention(q, k, v, scale=scale, sinks=sinks[::2])
    want = nax_mixed_head_dim_attention(
        q, k, v, scale=scale, sinks=mx.contiguous(sinks[::2])
    )
    assert mx.array_equal(got, want).item()


# (B, H, Hk, qL, kL, mask, sinks, dtype) for the key-range pass tests: causal
# chunks at the end of a longer context (the prefill layout), unaligned tails,
# a first chunk (qL == kL, early query blocks end before later passes start),
# bool masks, sinks, fp16 and a batch.
_PASS_CASES = [
    (1, 8, 2, 200, 1300, "causal", False, mx.bfloat16),
    (1, 8, 2, 256, 256, "causal", False, mx.bfloat16),
    (1, 16, 1, 1031, 1031, "causal", True, mx.bfloat16),
    (1, 8, 4, 64, 2000, "causal", False, mx.float16),
    (2, 8, 2, 300, 700, "array", True, mx.bfloat16),
    (1, 8, 2, 128, 1000, None, False, mx.bfloat16),
]


@requires_nax
@pytest.mark.parametrize("case", _PASS_CASES)
@pytest.mark.parametrize("n_pass", [2, 3, 7])
@pytest.mark.parametrize("dsplit", [0, 2])
def test_key_range_passes_are_bit_identical(case, n_pass, dsplit):
    """Splitting the key range over dispatches resumes the exact fp32 state
    (for the one-simdgroup and the head-dim split kernel alike)."""
    B, H, Hk, qL, kL, mask_kind, has_sinks, dtype = case
    pass_keys = kL // n_pass
    q, k, v = _inputs(B, H, Hk, qL, kL, dtype, seed=qL + kL + n_pass)
    mask = mask_kind
    if mask_kind == "array":
        mask = mx.random.uniform(shape=(B, 1, qL, kL)) > 0.3
        mask[..., 0] = True
        mask = mx.broadcast_to(mask, (B, H, qL, kL))
    sinks = (2 * mx.random.normal((H,))).astype(dtype) if has_sinks else None
    scale = 192**-0.5
    one = nax_attention._run(
        q, k, v, scale, mask, sinks, pass_keys=0, dsplit=dsplit
    )
    edges = nax_attention._pass_edges(B * H, (kL + 31) // 32, pass_keys, 0)
    assert len(edges) == n_pass + 1  # really several dispatches
    many = nax_attention._run(
        q, k, v, scale, mask, sinks, pass_keys=pass_keys, min_groups=0, dsplit=dsplit
    )
    assert mx.array_equal(one, many).item()
    ref = _reference(q, k, v, scale, mask, sinks)
    assert _max_err(many, ref) < (1e-2 if dtype == mx.bfloat16 else 2e-3)


def test_pass_edges():
    f = nax_attention._pass_edges
    assert f(8192, 2048, 8192) == [0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048]
    assert f(8192, 256, 8192) == [0, 256]  # one slice of keys: one dispatch
    assert f(8192, 2048, 0) == [0, 2048]  # disabled
    # few threadgroups (a short query tail) run in one wave: one dispatch
    assert f(128, 32768, 8192) == [0, 32768]
    edges = f(4096, 1000, 8192)
    assert edges[0] == 0 and edges[-1] == 1000
    assert all(b > a for a, b in zip(edges, edges[1:]))


@requires_nax
def test_register_resident_queries_are_bit_identical(monkeypatch):
    q, k, v = _inputs(1, 8, 2, 1031, 1500, seed=11)
    scale = 192**-0.5
    run = nax_attention._run
    monkeypatch.setattr(nax_attention, "_QREG", True)
    resident = run(q, k, v, scale, "causal", None, pass_keys=0, dsplit=0)
    monkeypatch.setattr(nax_attention, "_QREG", False)
    reloaded = run(q, k, v, scale, "causal", None, pass_keys=0, dsplit=0)
    assert mx.array_equal(resident, reloaded).item()


def _ulps(out, ref):
    """Max |out - ref| in bf16 ULPs of ref (|ref| >= 2^-10 only)."""
    o, r = out.astype(mx.float32), ref.astype(mx.float32)
    big = mx.abs(r) >= 2.0**-10
    e = mx.floor(mx.log2(mx.maximum(mx.abs(r), 2.0**-126)))
    return mx.where(big, mx.abs(o - r) / mx.power(2.0, e - 7), 0).max().item()


@requires_nax
@pytest.mark.parametrize("case", _PASS_CASES)
def test_head_dim_split_changes_only_summation_order(case):
    """Q @ K.T as two 96-dim fp32 partial sums: last-bit differences only."""
    B, H, Hk, qL, kL, mask_kind, has_sinks, dtype = case
    q, k, v = _inputs(B, H, Hk, qL, kL, dtype, seed=qL * 3 + kL)
    mask = mask_kind
    if mask_kind == "array":
        mask = mx.random.uniform(shape=(B, 1, qL, kL)) > 0.3
        mask[..., 0] = True
    sinks = (2 * mx.random.normal((H,))).astype(dtype) if has_sinks else None
    scale = 192**-0.5
    run = nax_attention._run
    one_sg = run(q, k, v, scale, mask, sinks, pass_keys=0, dsplit=0)
    split = run(q, k, v, scale, mask, sinks, pass_keys=0, dsplit=2)
    assert mx.mean((split == one_sg).astype(mx.float32)).item() > 0.98
    assert _ulps(split, one_sg) <= 2
    ref = _reference(q, k, v, scale, mask, sinks)
    tol = 1e-2 if dtype == mx.bfloat16 else 2e-3
    assert _max_err(split, ref) < tol


@requires_nax
def test_head_dim_split_only_for_passes_by_default(monkeypatch):
    q, k, v = _inputs(1, 8, 2, 200, 1300, seed=12)
    scale = 192**-0.5
    seen = []
    real = nax_attention._run_edges

    def spy(*args):
        seen.append((len(args[6]) - 1, args[7]))
        return real(*args)

    monkeypatch.setattr(nax_attention, "_run_edges", spy)
    monkeypatch.setattr(nax_attention, "_DSPLIT", 1)
    nax_attention._run(q, k, v, scale, "causal", None, pass_keys=0)
    nax_attention._run(q, k, v, scale, "causal", None, pass_keys=400, min_groups=0)
    assert seen[0] == (1, 1)  # one dispatch: one simdgroup per row block
    assert seen[-1] == (3, 2)  # passes: head-dim split


@requires_nax
def test_output_rows_follow_sdpa_layout():
    """The [B, H, L, V] result reshapes to [B, L, H * V] like MLX's SDPA."""
    q, k, v = _inputs(1, 8, 2, 100, 100, seed=5)
    out = nax_mixed_head_dim_attention(q, k, v, scale=0.07, mask="causal")
    flat = out.swapaxes(1, 2).reshape(1, 100, -1)
    ref = _reference(q, k, v, 0.07, "causal").swapaxes(1, 2).reshape(1, 100, -1)
    assert _max_err(flat, ref) < 1e-2


@requires_nax
def test_mask_with_broadcast_key_axis():
    q, k, v = _inputs(1, 4, 2, 64, 96, seed=6)
    row_mask = mx.random.uniform(shape=(1, 1, 64, 1)) > -1.0  # all true
    out = nax_mixed_head_dim_attention(q, k, v, scale=0.07, mask=row_mask)
    ref = nax_mixed_head_dim_attention(q, k, v, scale=0.07, mask=None)
    assert mx.array_equal(out, ref).item()


def test_declines_unsupported_inputs(monkeypatch):
    monkeypatch.setattr(nax_attention, "_nax_available", lambda: True)
    monkeypatch.setattr(nax_attention, "_self_check_passed", lambda: True)
    q, k, v = _inputs(1, 4, 2, 64, 64)
    f = nax_mixed_head_dim_attention
    assert f(q[..., :128], k[..., :128], v, scale=1.0) is None  # 128/128
    assert f(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
             scale=1.0) is None  # fp32
    assert f(q[:, :, :8], k, v, scale=1.0) is None  # decode-shaped
    additive = mx.zeros((64, 64), dtype=mx.bfloat16)
    assert f(q, k, v, scale=1.0, mask=additive) is None
    assert f(q, k, v, scale=1.0, mask=mx.ones((3, 64, 64), dtype=mx.bool_)) is None
    assert f(q, k, v, scale=1.0, mask="window") is None
    assert f(q, k, v, scale=1.0, sinks=mx.zeros((3,))) is None
    assert f(q, k[:, :, :32], v, scale=1.0) is None  # K/V length mismatch
    assert f(q[:, :3], k, v, scale=1.0) is None  # heads not a multiple


def test_kill_switch_and_non_nax(monkeypatch):
    q, k, v = _inputs(1, 4, 2, 64, 64)
    monkeypatch.setattr(nax_attention, "_ENABLED", False)
    assert nax_mixed_head_dim_attention(q, k, v, scale=1.0) is None
    monkeypatch.setattr(nax_attention, "_ENABLED", True)
    monkeypatch.setattr(nax_attention, "_nax_available", lambda: False)
    assert nax_mixed_head_dim_attention(q, k, v, scale=1.0) is None


def test_failed_self_check_disables_route(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("Unable to build metal library from source")

    monkeypatch.setattr(nax_attention, "_nax_available", lambda: True)
    monkeypatch.setattr(nax_attention, "_run_edges", boom)
    nax_attention._self_check_passed.cache_clear()
    try:
        q, k, v = _inputs(1, 4, 2, 64, 64)
        assert nax_mixed_head_dim_attention(q, k, v, scale=1.0) is None
    finally:
        nax_attention._self_check_passed.cache_clear()


@requires_nax
def test_mixed_head_dim_sdpa_prefers_native_then_jit(monkeypatch):
    q, k, v = _inputs(1, 8, 2, 300, 1100, seed=7)
    ref = _reference(q, k, v, 192**-0.5, "causal")
    monkeypatch.setattr(fast_attention, "_native_mixed_dims_supported", lambda *a: False)
    calls = []
    real = fast_attention.nax_mixed_head_dim_attention

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fast_attention, "nax_mixed_head_dim_attention", spy)
    out = fast_attention.mixed_head_dim_sdpa(q, k, v, scale=192**-0.5, mask="causal")
    assert calls and out is not None and _max_err(out, ref) < 1e-2


@requires_nax
def test_mixed_head_dim_sdpa_splits_long_keys_even_with_native(monkeypatch):
    """With MLX's native 192/128 kernel present, long key ranges still take
    the key-range passes (bit-identical, faster); short ones stay native."""
    q, k, v = _inputs(1, 8, 2, 300, 1100, seed=8)
    monkeypatch.setattr(
        fast_attention, "_native_mixed_dims_supported", lambda *a: True
    )
    calls = []
    real = fast_attention.nax_mixed_head_dim_attention

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fast_attention, "nax_mixed_head_dim_attention", spy)
    monkeypatch.setattr(nax_attention, "_PASS_KEYS", 256)
    monkeypatch.setattr(nax_attention, "_PASS_MIN_GROUPS", 1)
    assert nax_attention.uses_key_passes(q, k)
    out = fast_attention.mixed_head_dim_sdpa(q, k, v, scale=192**-0.5, mask="causal")
    assert calls and out is not None
    edges = nax_attention._pass_edges(8 * 5, (1100 + 31) // 32, 256)
    assert len(edges) > 2
    same = nax_attention._run_edges(
        q, k, v, 192**-0.5, "causal", None, [0, edges[-1]], 2
    )
    assert mx.array_equal(out, same).item()

    calls.clear()
    monkeypatch.setattr(nax_attention, "_PASS_KEYS", 0)
    assert not nax_attention.uses_key_passes(q, k)
    monkeypatch.setattr(
        fast_attention, "_native_mixed_dims_supported", lambda *a: False
    )
    fast_attention.mixed_head_dim_sdpa(q, k, v, scale=192**-0.5, mask="causal")
    assert calls  # no native kernel: the JIT kernel still runs (one dispatch)


@requires_nax
@pytest.mark.parametrize("prefix", [0, 100, 300])
def test_blocked_window_attention_uses_jit_kernel(monkeypatch, prefix):
    """MiMo's window layers (192/128, sinks, bool block masks) on stock MLX."""
    monkeypatch.setattr(fast_attention, "_native_mixed_dims_supported", lambda *a: False)
    calls = []
    real = fast_attention.nax_mixed_head_dim_attention

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(fast_attention, "nax_mixed_head_dim_attention", spy)
    H, Hk, L, window = 8, 2, 511, 128
    S = prefix + L
    q, k, v = _inputs(1, H, Hk, L, S, seed=prefix)
    sinks = mx.random.normal((H,)).astype(mx.bfloat16)
    scale = 192**-0.5
    out = fast_attention.blocked_sliding_window_attention(
        q, k, v, scale=scale, window=window, sinks=sinks, block=128
    )
    assert calls == [True]
    qpos = mx.arange(prefix, S)[:, None]
    kpos = mx.arange(S)[None, :]
    mask = ((kpos <= qpos) & (kpos > qpos - window))[None, None]
    ref = _reference(q, k, v, scale, mask, sinks)
    assert out.shape == ref.shape
    assert _max_err(out, ref) < 1e-2
