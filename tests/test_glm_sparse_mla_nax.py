# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 tensor-unit (NAX) sparse MLA prefill attention."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches.glm_moe_dsa import sparse_mla, sparse_mla_nax

pytestmark = pytest.mark.skipif(
    not sparse_mla_nax.nax_sparse_mla_available(),
    reason="needs an M5 (NAX) GPU",
)


def _reference(q_latent, kv_latent, topk, scale):
    """fp32 causal attention of every head over its query's selected rows."""
    _, H, L, D = q_latent.shape
    K = kv_latent.shape[2]
    idx = topk[0, 0].astype(mx.int32)
    valid = (idx >= 0) & (idx < K) & (idx <= (mx.arange(L) + (K - L))[:, None])
    keys = kv_latent[0, 0].astype(mx.float32)[mx.where(valid, idx, 0)]
    q = q_latent[0].swapaxes(0, 1).astype(mx.float32)
    scores = (q @ keys.swapaxes(-1, -2)) * scale
    scores = mx.where(valid[:, None, :], scores, -mx.inf)
    out = mx.softmax(scores, axis=-1) @ keys
    return out.swapaxes(0, 1)[None]


def _inputs(L, K, topk, H=64, q_scale=1.0, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((1, H, L, 512)) * q_scale).astype(dtype)
    kv = mx.random.normal((1, 1, K, 512)).astype(dtype)
    pos = (K - L) + mx.arange(L)[:, None]
    idx = (mx.random.uniform(shape=(L, topk)) * (pos + 1)).astype(mx.int32)
    # Unused slots of every kind: negative, past the query, beyond the cache.
    u = mx.random.uniform(shape=(L, topk))
    idx = mx.where(u < 0.02, -1, idx)
    idx = mx.where((u >= 0.02) & (u < 0.03), pos + 1 + (idx % 7), idx)
    idx = mx.where((u >= 0.03) & (u < 0.035), K + 5, idx)
    return q, kv, idx[None, None]


def _native(q, kv, idx, scale):
    zq = mx.zeros(q.shape[:-1] + (64,), q.dtype)
    zk = mx.zeros(kv.shape[:-1] + (64,), kv.dtype)
    return sparse_mla.sparse_mla_attention(q, zq, kv, zk, idx, scale)


@pytest.mark.parametrize(
    "L,K,topk,q_scale",
    [(64, 256, 128, 1.0), (37, 700, 200, 1.0), (128, 4096, 2051, 1.0), (96, 8192, 2051, 3.0)],
)
def test_matches_fp32_reference_like_native_kernel(L, K, topk, q_scale):
    q, kv, idx = _inputs(L, K, topk, q_scale=q_scale)
    scale = 256**-0.5
    out = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
    ref = _reference(q, kv, idx, scale)
    mx.eval(out, ref)
    assert out.shape == q.shape and out.dtype == q.dtype
    err = mx.abs(out.astype(mx.float32) - ref)
    # Output rounding: half a bf16 ulp of each value, plus fp32 slack.
    bound = mx.abs(ref) * 2.0**-8 + 1e-3
    assert mx.all(err <= bound).item()
    native = _native(q, kv, idx, scale)
    if native is not None:
        n_err = mx.abs(native.astype(mx.float32) - ref)
        # Same arithmetic as the native kernel, different summation order.
        assert mx.mean(err).item() <= 1.05 * mx.mean(n_err).item() + 1e-6
        assert mx.max(err).item() <= 1.05 * mx.max(n_err).item() + 1e-3


def test_fp16_inputs():
    q, kv, idx = _inputs(40, 1024, 300, dtype=mx.float16)
    scale = 256**-0.5
    out = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
    ref = _reference(q, kv, idx, scale)
    assert out.dtype == mx.float16
    err = mx.abs(out.astype(mx.float32) - ref)
    assert mx.all(err <= mx.abs(ref) * 2.0**-10 + 1e-3).item()


def test_uint32_indices_match_int32():
    q, kv, idx = _inputs(32, 512, 128)
    scale = 256**-0.5
    a = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
    b = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx.astype(mx.uint32), scale)
    assert mx.array_equal(a, b).item()


def test_unsupported_shapes_return_none():
    q, kv, idx = _inputs(16, 64, 32)
    scale = 1.0
    f = sparse_mla_nax.sparse_mla_attention_nax
    assert f(q[..., :256], kv[..., :256], idx, scale) is None  # latent width
    assert f(q[:, :16], kv, idx, scale) is None  # heads not a multiple of 32
    assert f(mx.concatenate([q, q]), kv, idx, scale) is None  # batch
    assert f(q, kv, idx[:, :, :8], scale) is None  # rows mismatch
    assert f(q.astype(mx.float32), kv.astype(mx.float32), idx, scale) is None
    assert f(q, kv[:, :, :8], idx, scale) is None  # fewer keys than queries


def test_disabled_by_env(monkeypatch):
    monkeypatch.setattr(sparse_mla_nax, "_ENABLED", False)
    sparse_mla_nax.nax_sparse_mla_available.cache_clear()
    try:
        q, kv, idx = _inputs(16, 64, 32)
        assert sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, 1.0) is None
    finally:
        sparse_mla_nax.nax_sparse_mla_available.cache_clear()


def test_deterministic_across_runs():
    # Large enough to keep many threadgroups in flight; every run must be
    # bit-identical (guards against threadgroup-memory races).
    q, kv, idx = _inputs(256, 8192, 2051, q_scale=3.0, seed=4)
    scale = 256**-0.5
    outs = [sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale) for _ in range(6)]
    mx.eval(outs)
    for o in outs[1:]:
        assert mx.array_equal(outs[0], o).item()
    ref = _reference(q, kv, idx, scale)
    err = mx.abs(outs[0].astype(mx.float32) - ref)
    assert mx.all(err <= mx.abs(ref) * 2.0**-8 + 1e-3).item()


def test_probabilities_keep_fp32_precision():
    """The PV product must use the fp32 probabilities (like the native
    kernel), not a reduced-precision copy: a value column of large
    alternating +-1000 entries cancels in the weighted sum and exposes any
    rounding of the probabilities (~11-bit rounding shows up as >> 1 ulp)."""
    L, K, topk = 32, 4096, 2051
    q, kv, idx = _inputs(L, K, topk, seed=7)
    q = q.astype(mx.float32)
    q[..., 0] = 0.0  # the big column must not influence the scores
    q = q.astype(mx.bfloat16)
    kv = kv.astype(mx.float32)
    sign = mx.where(mx.arange(K) % 2 == 0, 1.0, -1.0)
    kv[0, 0, :, 0] = 1000.0 * sign
    kv = kv.astype(mx.bfloat16)
    scale = 256**-0.5
    out = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
    ref = _reference(q, kv, idx, scale)
    mx.eval(out, ref)
    col = ref[..., 0]
    err = mx.abs(out[..., 0].astype(mx.float32) - col)
    # fp32 probabilities: the error is the bf16 rounding of the result
    # (half an ulp) plus fp32 summation noise of 1000 * sqrt(topk) * 2^-24.
    ulp = mx.power(2.0, mx.floor(mx.log2(mx.maximum(mx.abs(col), 1e-3))) - 7)
    assert mx.all(err <= 0.5 * ulp + 0.02).item()


def _make_sparse_attention():
    from mlx.utils import tree_map

    from omlx.patches import mlx_vlm_glm5_next_compat as compat

    compat.apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models import glm5_next
    from mlx_vlm.models.glm5_next.language import Glm5NextSparseAttention

    text = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=None,
        n_routed_experts=None,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        v_head_dim=16,
        qk_nope_head_dim=16,
        num_experts_per_tok=2,
        first_k_dense_replace=99,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        layer_types=["deepseek_sparse_attention"],
        mlp_layer_types=["dense"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=4,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    mx.random.seed(3)
    attn = Glm5NextSparseAttention(text)
    attn.indexer.index_kpool_compress_gate = mx.random.normal((128, 64)) * 0.1
    attn.indexer.index_kpool_compress_ape = mx.random.normal((4, 128)) * 0.1
    attn.update(tree_map(lambda a: a.astype(mx.bfloat16), attn.parameters()))
    return attn


def _run_attention(attn, chunks, seed=5):
    from mlx_lm.models.cache import KVCache, PoolingCache

    mx.random.seed(seed)
    cache = [KVCache(), PoolingCache(4)]
    outs = []
    for n in chunks:
        x = mx.random.normal((1, n, 64)).astype(mx.bfloat16)
        out = attn(x, None, cache)
        mx.eval(out)
        outs.append(out)
    return outs


def test_call_site_matches_fallback_paths(monkeypatch):
    """The GLM-5.3 call site with the NAX kernel matches the previous paths:
    the native kernel at >= 4096 keys (same latent-space math) and the
    expanded exact-block attention below (a reassociation of the same
    products, so bf16 intermediate rounding differs)."""
    attn = _make_sparse_attention()
    chunks = [2600, 1600]  # sparse at 2600 keys, then at 4200 keys
    monkeypatch.setattr(sparse_mla_nax, "_ENABLED", False)
    sparse_mla_nax.nax_sparse_mla_available.cache_clear()
    try:
        expected = _run_attention(attn, chunks)
    finally:
        monkeypatch.setattr(sparse_mla_nax, "_ENABLED", True)
        sparse_mla_nax.nax_sparse_mla_available.cache_clear()
    calls = []
    orig = sparse_mla_nax.sparse_mla_attention_nax

    def spy(*args, **kwargs):
        out = orig(*args, **kwargs)
        calls.append(out is not None)
        return out

    from mlx_vlm.models.glm5_next import language

    monkeypatch.setattr(language, "sparse_mla_attention_nax", spy)
    got = _run_attention(attn, chunks)
    assert calls == [True, True]
    for e, g, tol in zip(expected, got, (2e-2, 1e-2)):
        assert e.shape == g.shape and e.dtype == g.dtype
        e32, g32 = e.astype(mx.float32), g.astype(mx.float32)
        scale = mx.abs(e32).max().item()
        assert mx.abs(e32 - g32).max().item() <= tol * scale


@pytest.mark.parametrize("mode", ["bf16x3", "half2"])
def test_pv_modes_match_fp32_reference(mode, monkeypatch):
    """Both P @ V operand splits stay at the native kernel's error level."""
    if not sparse_mla_nax.nax_sparse_mla_available():
        pytest.skip("needs NAX")
    monkeypatch.setattr(sparse_mla_nax, "_PV_MODE", mode)
    monkeypatch.setattr(sparse_mla_nax, "_KERNEL", None)
    q, kv, idx = _inputs(96, 8192, 2051, seed=7)
    scale = 576**-0.5
    native = _native(q, kv, idx, scale)
    if native is None:
        pytest.skip("native sparse MLA kernel unavailable")
    native = native.astype(mx.float32)
    out = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale).astype(mx.float32)
    ref = _reference(q, kv, idx, scale)
    err = mx.abs(out - ref).mean().item()
    err_native = mx.abs(native - ref).mean().item()
    assert err <= 1.05 * err_native


def _realistic_inputs(L, K, H=64, dtype=mx.bfloat16, seed=0, topk_blocks=512):
    """Indexer-like top-k rows: 4-token blocks (sinks, a drifting scattered
    set, the most recent blocks) in ascending order, then the 3 causal tail
    slots; unused slots (-1) sort last."""
    import numpy as np

    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    q = mx.random.normal((1, H, L, 512)).astype(dtype)
    kv = mx.random.normal((1, 1, K, 512)).astype(dtype)
    topk = 4 * topk_blocks + 3
    idx = np.full((L, topk), -1, dtype=np.int32)
    cur = None
    for i in range(L):
        p = K - L + i
        nb = (p + 1) // 4
        if nb <= topk_blocks:
            blocks = np.arange(nb)
        else:
            hi = nb - topk_blocks // 8
            n_rand = topk_blocks - topk_blocks // 8 - 4
            if cur is None:
                cur = rng.choice(np.arange(4, hi), n_rand, replace=False)
            cur = cur[(cur < hi) & (rng.random(cur.size) >= 0.05)]
            if cur.size < n_rand:
                pool = np.setdiff1d(np.arange(4, hi), cur)
                cur = np.concatenate([cur, rng.choice(pool, n_rand - cur.size, replace=False)])
            blocks = np.sort(np.concatenate([np.arange(4), cur, np.arange(hi, nb)]))
        rows = (blocks[:, None] * 4 + np.arange(4)[None]).reshape(-1)
        idx[i, : rows.size] = rows
        tc = (p + 1) % 4
        for j in range(3):
            idx[i, topk - 3 + j] = p + 1 - tc + j if j < tc else -1
    return q, kv, mx.array(idx)[None, None]


def _kernel_output(monkeypatch, impl, q, kv, idx, scale):
    monkeypatch.setattr(sparse_mla_nax, "_PV_MODE", "half2")
    monkeypatch.setattr(sparse_mla_nax, "_IMPL", impl)
    monkeypatch.setattr(sparse_mla_nax, "_KERNEL", None)
    out = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
    mx.eval(out)
    return out


def _ulp_distance(a, b):
    import numpy as np

    ai = np.array(a.view(mx.uint16)).astype(np.int32)
    bi = np.array(b.view(mx.uint16)).astype(np.int32)
    ai = np.where(ai & 0x8000, -(ai & 0x7FFF), ai & 0x7FFF)
    bi = np.where(bi & 0x8000, -(bi & 0x7FFF), bi & 0x7FFF)
    return np.abs(ai - bi)


@pytest.mark.parametrize(
    "L,K,dtype",
    [(256, 8192, mx.bfloat16), (97, 3000, mx.bfloat16), (64, 4096, mx.float16)],
)
def test_v2_matches_h2_kernel(L, K, dtype, monkeypatch):
    """The default (v2) kernel reproduces the h2 kernel: same scores, maxima,
    probability pieces and P x V sums; only the order in which the four
    quarter row sums of a tile enter the denominator differs, which moves a
    few outputs by one ulp."""
    import numpy as np

    q, kv, idx = _realistic_inputs(L, K, dtype=dtype, seed=L)
    scale = 256**-0.5
    h2 = _kernel_output(monkeypatch, "h2", q, kv, idx, scale)
    v2 = _kernel_output(monkeypatch, "v2", q, kv, idx, scale)
    assert v2.shape == h2.shape and v2.dtype == h2.dtype
    d = _ulp_distance(v2, h2)
    assert (d == 0).mean() >= 0.999
    # every difference is one ulp of the value (tiny outputs aside)
    ref32 = np.array(h2.astype(mx.float32))
    big = np.abs(ref32) >= np.abs(ref32).max(axis=-1, keepdims=True) * 2.0**-10
    assert d[big].max() <= 1
    ref = _reference(q, kv, idx, scale)
    e_v2 = mx.abs(v2.astype(mx.float32) - ref).mean().item()
    e_h2 = mx.abs(h2.astype(mx.float32) - ref).mean().item()
    assert e_v2 <= e_h2 * 1.001 + 1e-9


def test_dead_tiles_and_ragged_topk(monkeypatch):
    """Whole unused 128-slot tiles in the middle and at the end, and top-k
    widths that are not multiples of the tile or fragment sizes."""
    import numpy as np

    scale = 256**-0.5
    for topk in (17, 100, 129, 400, 2051):
        q, kv, idx = _inputs(48, 3000, topk, seed=topk)
        a = np.array(idx)
        if topk > 256:
            a[..., 128:256] = -1  # a dead tile in the middle
        a[..., ::5, max(0, topk - 70):] = -1  # dead tail tiles for some rows
        a[0, 0, :, 0] = 3000 - 48 + np.arange(48)  # keep one usable slot per row
        idx = mx.array(a)
        out = _kernel_output(monkeypatch, "v2", q, kv, idx, scale)
        ref = _reference(q, kv, idx, scale)
        err = mx.abs(out.astype(mx.float32) - ref)
        assert mx.all(err <= mx.abs(ref) * 2.0**-8 + 1e-3).item(), topk
        h2 = _kernel_output(monkeypatch, "h2", q, kv, idx, scale)
        o32 = np.array(h2.astype(mx.float32))
        big = np.abs(o32) >= np.abs(o32).max(axis=-1, keepdims=True) * 2.0**-10
        assert _ulp_distance(out, h2)[big].max() <= 1, topk


def test_v2_deterministic_many_threadgroups(monkeypatch):
    """Many threadgroups in flight, realistic index rows: every run must be
    bit-identical (a software-pipelined variant with in-flight loads into
    dead registers was not)."""
    q, kv, idx = _realistic_inputs(1024, 8192, seed=11)
    scale = 256**-0.5
    first = _kernel_output(monkeypatch, "v2", q, kv, idx, scale)
    for _ in range(8):
        out = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
        mx.eval(out)
        assert mx.array_equal(out, first).item()
