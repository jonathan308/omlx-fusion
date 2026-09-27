# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 tensor-unit (NAX) DSA indexer scores."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_glm5_next_compat as compat
from omlx.patches.glm_moe_dsa import indexer_nax

pytestmark = pytest.mark.skipif(
    not indexer_nax.nax_indexer_available(),
    reason="needs an M5 (NAX) GPU",
)


@pytest.fixture(autouse=True)
def _apply_glm5_next_compat():
    compat.apply_mlx_vlm_glm5_next_compat_patch()


def _bf16_ulp_distance(a: mx.array, b: mx.array) -> mx.array:
    ai = a.view(mx.int16).astype(mx.int32)
    bi = b.view(mx.int16).astype(mx.int32)
    ai = mx.where(ai < 0, -32768 - ai, ai)
    bi = mx.where(bi < 0, -32768 - bi, bi)
    return mx.abs(ai - bi)


def _reference(q, k, w, before, pool_len, ratio):
    """fp32 scores, heads accumulated in order, masked like the call site."""
    S, H, _ = q.shape
    P = k.shape[0]
    qf, kf, wf = (a.astype(mx.float32) for a in (q, k, w))
    acc = mx.zeros((S, P), mx.float32)
    for h in range(H):
        acc = acc + mx.maximum(qf[:, h] @ kf.T, 0.0) * wf[:, h : h + 1]
    s = mx.arange(S)[:, None]
    p = mx.arange(P)[None]
    valid = (p < pool_len) & ((p + 1) * ratio - 1 <= before + s)
    return acc, valid


def _inputs(S, P, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((S, 32, 128)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((P, 128)) * 0.5).astype(mx.bfloat16)
    w = (mx.random.normal((S, 32)) * 0.1).astype(mx.bfloat16)
    return q, k, w


@pytest.mark.parametrize(
    "S,P,before",
    [(64, 64, 192), (100, 300, 1000), (511, 1024, 3584), (257, 777, 3000), (33, 600, 2400)],
)
def test_scores_match_fp32_reference_and_mask(S, P, before):
    q, k, w = _inputs(S, P)
    pool_len = min(P, (before + S) // 4)
    out = indexer_nax.indexer_scores_nax(q, k, w, before, pool_len, 4)
    acc, valid = _reference(q, k, w, before, pool_len, 4)
    mx.eval(out, acc, valid)
    assert out.shape == (S, P) and out.dtype == mx.bfloat16
    # Masked entries carry exactly the sentinel the old mx.where wrote.
    sentinel = mx.where(valid, out, mx.array(-1e30, mx.bfloat16))
    assert mx.array_equal(out.view(mx.int16), sentinel.view(mx.int16)).item()
    # Live entries: the fp32 sum rounded to bf16, up to summation order.
    ref = acc.astype(mx.bfloat16)
    ulp = mx.where(valid, _bf16_ulp_distance(out, ref), 0)
    scale = mx.abs(mx.where(valid, acc, 0)).max(axis=-1, keepdims=True)
    err = mx.where(valid, mx.abs(out.astype(mx.float32) - acc), 0)
    # Within one bf16 ulp of the value, or tiny relative to the row scale
    # (cancellation near zero).
    ok = (ulp <= 1) | (err <= 1e-4 * scale)
    assert mx.all(ok).item()


def test_scores_match_native_kernel_within_rounding():
    from omlx.custom_kernels.glm_moe_dsa import fast

    if not fast.has_symbol("dsa_indexer_scores"):
        pytest.skip("native DSA indexer kernel unavailable")
    S, P, before = 512, 1024, 3584
    q, k, w = _inputs(S, P, seed=1)
    pool_len = P
    out = indexer_nax.indexer_scores_nax(q, k, w, before, pool_len, 4)
    native = fast.dsa_indexer_scores(
        q[None].transpose(0, 2, 1, 3), k[None, None], w[None], causal=False
    )[0, 0]
    acc, valid = _reference(q, k, w, before, pool_len, 4)
    native = mx.where(valid, native, -1e30)
    mx.eval(out, native, acc)
    scale = mx.abs(acc).max(axis=-1, keepdims=True)
    diff = mx.abs(out.astype(mx.float32) - native.astype(mx.float32))
    ulp = _bf16_ulp_distance(out, native)
    assert mx.all((ulp <= 2) | (diff <= 1e-4 * scale)).item()


def test_unsupported_inputs_return_none():
    q, k, w = _inputs(8, 16)
    assert indexer_nax.indexer_scores_nax(q.astype(mx.float16), k, w, 0, 16, 4) is None
    assert indexer_nax.indexer_scores_nax(q[..., :64], k, w, 0, 16, 4) is None
    assert indexer_nax.indexer_scores_nax(q, k, w[:4], 0, 16, 4) is None


def test_row_cap_is_a_multiple_of_64():
    assert indexer_nax.max_rows_per_call(1) % 64 == 0
    assert indexer_nax.max_rows_per_call(1 << 30) == 64
    assert indexer_nax.max_rows_per_call(16384) == (1 << 27) // 16384


def _make_indexer():
    from mlx.utils import tree_map
    from mlx_vlm.models import glm5_next
    from mlx_vlm.models.glm5_next.language import Glm5NextIndexer

    text = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=None,
        n_routed_experts=None,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        v_head_dim=8,
        qk_nope_head_dim=8,
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
    indexer = Glm5NextIndexer(text)
    indexer.index_kpool_compress_gate = mx.random.normal((128, 64)) * 0.1
    indexer.index_kpool_compress_ape = mx.random.normal((4, 128)) * 0.1
    indexer.update(tree_map(lambda a: a.astype(mx.bfloat16), indexer.parameters()))
    return indexer


def _run_indexer(indexer, chunks, seed=5):
    from mlx_lm.models.cache import KVCache, PoolingCache

    mx.random.seed(seed)
    pool = PoolingCache(4)
    kv = KVCache()
    outs = []
    for n in chunks:
        x = mx.random.normal((1, n, 64)).astype(mx.bfloat16)
        qr = mx.random.normal((1, n, 32)).astype(mx.bfloat16)
        kv.update_and_fetch(
            mx.zeros((1, 1, n, 8), mx.bfloat16), mx.zeros((1, 1, n, 0), mx.bfloat16)
        )
        out = indexer(x, qr, None, cache=pool, kv_cache=kv)
        if out is not None:
            mx.eval(out)
        outs.append(out)
    return outs


def _require_native_scores():
    """The non-NAX indexer path these tests compare against scores with the
    native kernel; without it that path falls back to MLX ops, which round
    the scores differently, so the exactness and near-tie bounds do not
    apply."""
    from omlx.custom_kernels.glm_moe_dsa import fast

    if not fast.has_symbol("dsa_indexer_scores"):
        pytest.skip("native DSA indexer kernel unavailable")


def _native_masked_scores(q, pool_keys, weights, before, pool_len, ratio):
    """The previous call-site computation: native kernel + mx.where mask."""
    from mlx_vlm.models.glm5_next import language

    idx = language.Glm5NextIndexer.__new__(language.Glm5NextIndexer)
    idx.n_heads = q.shape[1]
    idx.head_dim = q.shape[2]
    scores = language.Glm5NextIndexer._native_scores(
        idx, q[None], pool_keys[None], weights[None]
    )[0]
    S, P = scores.shape
    s = mx.arange(S)[:, None]
    p = mx.arange(P)[None]
    valid = (p < pool_len) & ((p + 1) * ratio - 1 <= before + s)
    return mx.where(valid, scores, -1e30)


def test_indexer_fast_path_plumbing_is_exact(monkeypatch):
    """With the old score computation plugged in, the all-rows fast path
    returns bit-identical top-k indices to the 512-row loop."""
    _require_native_scores()
    from mlx_vlm.models.glm5_next import language

    indexer = _make_indexer()
    chunks = [2600, 700, 1500]

    monkeypatch.setattr(language, "nax_indexer_available", lambda: False)
    expected = _run_indexer(indexer, chunks)

    monkeypatch.setattr(language, "nax_indexer_available", lambda: True)
    monkeypatch.setattr(language, "indexer_scores_nax", _native_masked_scores)
    got = _run_indexer(indexer, chunks)

    assert expected[0] is not None
    for e, g in zip(expected, got):
        assert e.shape == g.shape and e.dtype == g.dtype
        assert mx.array_equal(e, g).item()


def test_indexer_nax_selection_matches_up_to_near_ties(monkeypatch):
    _require_native_scores()
    from mlx_vlm.models.glm5_next import language

    indexer = _make_indexer()
    chunks = [2600, 700]

    monkeypatch.setattr(language, "nax_indexer_available", lambda: False)
    expected = _run_indexer(indexer, chunks)
    monkeypatch.setattr(language, "nax_indexer_available", lambda: True)
    got = _run_indexer(indexer, chunks)

    for e, g in zip(expected, got):
        e = mx.sort(e[0, 0], axis=-1)
        g = mx.sort(g[0, 0], axis=-1)
        rows_equal = mx.all(e == g, axis=-1)
        # bf16 scores tie often; the kernels differ only in fp32 summation
        # order, so at most a few rows may pick a different tie member.
        assert mx.mean(rows_equal.astype(mx.float32)).item() > 0.97


def test_indexer_row_chunking_is_exact(monkeypatch):
    """A score-buffer cap that splits the rows gives identical indices."""
    from mlx_vlm.models.glm5_next import language

    indexer = _make_indexer()
    chunks = [2600, 700]
    monkeypatch.setattr(language, "nax_indexer_available", lambda: True)
    whole = _run_indexer(indexer, chunks)
    monkeypatch.setattr(indexer_nax, "_MAX_SCORE_ELEMENTS", 64 * 700)
    split = _run_indexer(indexer, chunks)
    for a, b in zip(whole, split):
        assert mx.array_equal(a, b).item()


def test_topk_differences_are_threshold_near_ties():
    """Rows where the NAX scores select a different pool set than the native
    scores differ only by pools whose native scores sit within one bf16 ulp
    of that row's top-k threshold (fp32 summation order at a near-tie)."""
    from mlx_vlm.models.glm5_next import language
    from omlx.custom_kernels.glm_moe_dsa import fast

    if not fast.has_symbol("dsa_topk_indices"):
        pytest.skip("native top-k kernel unavailable")
    S, P, before = 512, 4096, 16384 - 512
    q, k, w = _inputs(S, P, seed=11)
    native = _native_masked_scores(q, k, w, before, P, 4)
    nax = indexer_nax.indexer_scores_nax(q, k, w, before, P, 4)
    sel_n = language.Glm5NextIndexer._native_topk(native[None], 512)[0]
    sel_x = language.Glm5NextIndexer._native_topk(nax[None], 512)[0]
    vals_n = mx.sort(mx.take_along_axis(native, sel_n, axis=-1), axis=-1)
    vals_x = mx.sort(mx.take_along_axis(native, sel_x, axis=-1), axis=-1)
    mx.eval(vals_n, vals_x)
    # Same multiset of native scores up to one ulp at the threshold.
    ulp = _bf16_ulp_distance(vals_n, vals_x)
    assert mx.max(ulp).item() <= 1
    same = mx.all(mx.sort(sel_n, axis=-1) == mx.sort(sel_x, axis=-1), axis=-1)
    assert mx.mean(same.astype(mx.float32)).item() > 0.9


def test_indexer_without_cache(monkeypatch):
    """Cache-less prefill (positions from 0) takes the NAX path and selects
    the same pools as the native path (up to near ties)."""
    _require_native_scores()
    from mlx_vlm.models.glm5_next import language

    indexer = _make_indexer()
    mx.random.seed(9)
    x = mx.random.normal((1, 2600, 64)).astype(mx.bfloat16)
    qr = mx.random.normal((1, 2600, 32)).astype(mx.bfloat16)
    monkeypatch.setattr(language, "nax_indexer_available", lambda: False)
    expected = indexer(x, qr, None, cache=None, kv_cache=None)
    calls = []
    orig = indexer_nax.indexer_scores_nax

    def spy(*args, **kwargs):
        calls.append(args[3])  # before
        return orig(*args, **kwargs)

    monkeypatch.setattr(language, "nax_indexer_available", lambda: True)
    monkeypatch.setattr(language, "indexer_scores_nax", spy)
    got = indexer(x, qr, None, cache=None, kv_cache=None)
    assert calls == [0]
    assert got.shape == expected.shape and got.dtype == expected.dtype
    e = mx.sort(expected[0, 0], axis=-1)
    g = mx.sort(got[0, 0], axis=-1)
    rows_equal = mx.all(e == g, axis=-1)
    assert mx.mean(rows_equal.astype(mx.float32)).item() > 0.97


def test_nax_indexer_score_from_matches_suffix(monkeypatch):
    """With the dense-prefix bypass the attention layer scores only the rows
    from ``score_from`` on; the NAX path must return exactly the suffix of
    the all-rows selection (rows are scored independently)."""
    from mlx_vlm.models.glm5_next import language

    indexer = _make_indexer()
    mx.random.seed(21)
    x = mx.random.normal((1, 2600, 64)).astype(mx.bfloat16)
    qr = mx.random.normal((1, 2600, 32)).astype(mx.bfloat16)
    calls = []
    orig = indexer_nax.indexer_scores_nax

    def spy(*args, **kwargs):
        calls.append((args[0].shape[0], args[3]))  # (rows, first row position)
        return orig(*args, **kwargs)

    monkeypatch.setattr(language, "nax_indexer_available", lambda: True)
    monkeypatch.setattr(language, "indexer_scores_nax", spy)
    full = indexer(x, qr, None)
    tail = indexer(x, qr, None, score_from=2051)
    mx.eval(full, tail)
    assert calls == [(2600, 0), (549, 2051)]
    assert tail.shape[:3] == (1, 1, 549)
    assert mx.array_equal(tail, full[:, :, 2051:]).item()
    assert indexer(x, qr, None, score_from=2600) is None
