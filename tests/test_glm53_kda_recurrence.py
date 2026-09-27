# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 blocked KDA recurrence vs the stock gated-delta path."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_vlm_glm5_next_compat as compat

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="KDA recurrence kernel needs Metal"
)


@pytest.fixture(autouse=True)
def _apply_glm5_next_compat():
    compat.apply_mlx_vlm_glm5_next_compat_patch()


def _inputs(B, T, H, seed):
    mx.random.seed(seed)
    D = 128

    def l2(x):
        return x * mx.rsqrt((x * x).sum(-1, keepdims=True) + 1e-6)

    q = (l2(mx.random.normal((B, T, H, D))) * D**-0.5).astype(mx.bfloat16)
    k = l2(mx.random.normal((B, T, H, D))).astype(mx.bfloat16)
    v = mx.random.normal((B, T, H, D)).astype(mx.bfloat16)
    a = (2 * mx.random.normal((B, T, H, D))).astype(mx.bfloat16)
    beta = mx.sigmoid(mx.random.normal((B, T, H)).astype(mx.bfloat16))
    a_log = mx.random.uniform(0.3, 2.0, (H,))
    dt_bias = 0.5 * mx.random.normal((H * D,))
    state = 0.1 * mx.random.normal((B, H, D, D))
    return q, k, v, a, beta, a_log, dt_bias, state


def _stock(q, k, v, a, beta, a_log, dt_bias, state, mask=None, ops=False):
    from mlx_vlm.models.glm5_next import gated_delta as G

    H = q.shape[2]
    g = G.compute_g_safe(a_log.reshape(H, 1), a, dt_bias.reshape(H, 128), -5.0)
    if ops:
        return G.gated_delta_ops(q, k, v, g, beta, state, mask)
    return G.gated_delta_kernel(q, k, v, g, beta, state, mask)


@pytest.mark.parametrize(
    "cfg", [(16, 64, 2, True), (16, 32, 1, False), (8, 16, 2, True), (8, 128, 2, False)]
)
def test_recurrence_matches_stock_kernel_and_fp32_reference(cfg):
    from omlx.patches.glm53_kda_recurrence import RecurrenceConfig, kda_recurrence

    B, T, H = 2, 45, 2
    q, k, v, a, beta, a_log, dt_bias, state = _inputs(B, T, H, 5)
    y_ker, s_ker = _stock(q, k, v, a, beta, a_log, dt_bias, state)
    y_ref, s_ref = _stock(q, k, v, a, beta, a_log, dt_bias, state, ops=True)
    y, s = kda_recurrence(
        q, k, v, a, beta, a_log, dt_bias, -5.0, state, config=RecurrenceConfig(*cfg)
    )
    mx.eval(y_ker, s_ker, y_ref, s_ref, y, s)
    assert y.dtype == mx.bfloat16 and s.dtype == mx.float32
    # Same recurrence; only the fp32 summation order of the dots differs.
    assert mx.allclose(s, s_ker, rtol=1e-5, atol=1e-6).item()
    assert mx.allclose(s, s_ref, rtol=1e-4, atol=1e-5).item()
    assert mx.allclose(y, y_ker, rtol=1e-2, atol=1e-3).item()
    assert mx.allclose(y, y_ref, rtol=1e-2, atol=1e-3).item()


def test_recurrence_chunked_equals_one_shot():
    from omlx.patches.glm53_kda_recurrence import kda_recurrence

    q, k, v, a, beta, a_log, dt_bias, state = _inputs(1, 100, 2, 7)
    y_full, s_full = kda_recurrence(q, k, v, a, beta, a_log, dt_bias, -5.0, state)
    ys, s = [], state
    for lo, hi in [(0, 13), (13, 64), (64, 100)]:
        y, s = kda_recurrence(
            q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], a[:, lo:hi], beta[:, lo:hi],
            a_log, dt_bias, -5.0, s,
        )
        ys.append(y)
    y_chunks = mx.concatenate(ys, axis=1)
    mx.eval(y_full, s_full, y_chunks, s)
    assert mx.array_equal(y_full, y_chunks).item()
    assert mx.array_equal(s_full, s).item()


def _attention(num_heads=4, hidden=256, seed=0, bits=None):
    from mlx_vlm.models import glm5_next
    from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention

    config = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=hidden,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=None,
        n_routed_experts=None,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        qk_nope_head_dim=8,
        num_experts_per_tok=2,
        first_k_dense_replace=99,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        index_topk=4,
        index_head_dim=8,
        index_n_heads=2,
        layer_types=["linear_attention"],
        mlp_layer_types=["dense"],
        linear_attn_config={
            "num_heads": num_heads,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    mx.random.seed(seed)
    attn = Glm5NextLinearAttention(config)
    params = []
    for name, p in nn.utils.tree_flatten(attn.parameters()):
        if name.endswith("A_log"):
            params.append((name, mx.random.uniform(0.3, 2.0, p.shape)))
        elif name.endswith("dt_bias"):
            params.append((name, mx.random.normal(p.shape) * 0.5))
        elif name.endswith("o_norm.weight"):
            params.append((name, 1 + 0.1 * mx.random.normal(p.shape)))
        else:
            params.append((name, mx.random.normal(p.shape) * p.shape[-1] ** -0.5))
    attn.load_weights(params)
    if bits:
        nn.quantize(attn, group_size=64, bits=bits,
                    class_predicate=lambda _, m: isinstance(m, nn.Linear))
    # Checkpoint dtype policy: bf16 except the fp32 gate parameters.
    attn.set_dtype(mx.bfloat16)
    fg = attn.forget_gate
    fg.A_log = fg.A_log.astype(mx.float32)
    fg.dt_bias = fg.dt_bias.astype(mx.float32)
    mx.eval(attn.parameters())
    return config, attn


def _run(attn, x, conv0, state0, fused, chunks=None):
    from mlx_vlm.models.cache import ArraysCache

    from omlx.patches import glm53_kda_prework as kda

    enabled = kda._GLM53_KDA_PREFILL_ENABLED
    kda._GLM53_KDA_PREFILL_ENABLED = fused
    try:
        cache = ArraysCache(size=2)
        cache[0] = conv0
        cache[1] = state0
        outs, t = [], 0
        for n in chunks or [x.shape[1]]:
            outs.append(attn(x[:, t : t + n], None, cache))
            t += n
        out = mx.concatenate(outs, axis=1)
        mx.eval(out, cache[0], cache[1])
        return out, cache[0], cache[1]
    finally:
        kda._GLM53_KDA_PREFILL_ENABLED = enabled


def test_fused_prefill_runs_blocked_recurrence(monkeypatch):
    from mlx_vlm.models.glm5_next import gated_delta as G

    from omlx.patches import glm53_kda_prework as kda

    config, attn = _attention()
    H, D = attn.num_heads, attn.head_dim
    mx.random.seed(11)
    x = mx.random.normal((1, 150, config.hidden_size)).astype(mx.bfloat16)
    conv0 = mx.random.normal((1, 3, attn.conv_dim)).astype(mx.bfloat16)
    state0 = 0.05 * mx.random.normal((1, H, D, D))

    calls = []
    real = kda.kda_recurrence

    def counted(*args, **kwargs):
        calls.append(args[0].shape)
        return real(*args, **kwargs)

    monkeypatch.setattr(kda, "kda_recurrence", counted)
    stock = _run(attn, x, conv0, state0, fused=False)
    assert not calls
    fused = _run(attn, x, conv0, state0, fused=True)
    assert calls == [(1, 150, H, D)]
    assert mx.array_equal(stock[1], fused[1]).item()  # conv state
    assert mx.allclose(stock[2], fused[2], rtol=1e-5, atol=1e-6).item()
    assert mx.allclose(stock[0], fused[0], rtol=2e-2, atol=2e-3).item()

    # With the stock recurrence swapped in, the fused route is bit-identical.
    def stock_recurrence(q, k, v, a, beta, a_log, dt_bias, lb, state, config=None):
        g = G.compute_g_safe(a_log.reshape(H, 1), a, dt_bias.reshape(H, D), lb)
        return G.gated_delta_kernel(q, k, v, g, beta, state)

    monkeypatch.setattr(kda, "kda_recurrence", stock_recurrence)
    exact = _run(attn, x, conv0, state0, fused=True)
    for s, e in zip(stock, exact):
        assert mx.array_equal(s, e).item()

    # Chunked prefill carries conv and recurrent state exactly.
    monkeypatch.setattr(kda, "kda_recurrence", real)
    chunked = _run(attn, x, conv0, state0, fused=True, chunks=[64, 86])
    assert mx.array_equal(chunked[1], fused[1]).item()
    assert mx.array_equal(chunked[2], fused[2]).item()


def test_fused_prefill_keeps_stock_recurrence_for_non_fp32_gate(monkeypatch):
    from omlx.patches import glm53_kda_prework as kda

    config, attn = _attention(seed=4)
    attn.forget_gate.dt_bias = attn.forget_gate.dt_bias.astype(mx.bfloat16)
    calls = []
    monkeypatch.setattr(kda, "kda_recurrence", lambda *a, **k: calls.append(1))
    x = mx.random.normal((1, 80, config.hidden_size)).astype(mx.bfloat16)
    out = _run(attn, x, None, None, fused=True)[0]
    assert out.shape == (1, 80, config.hidden_size) and not calls


def test_fused_prefill_leaves_stock_shaped_caches():
    """Decode paths consume cache[0]/cache[1]; they must look like stock's."""
    config, attn = _attention(seed=3)
    H, D = attn.num_heads, attn.head_dim
    x = mx.random.normal((1, 96, config.hidden_size)).astype(mx.bfloat16)
    stock = _run(attn, x, None, None, fused=False)
    fused = _run(attn, x, None, None, fused=True)
    assert fused[1].shape == stock[1].shape == (1, 3, attn.conv_dim)
    assert fused[1].dtype == stock[1].dtype == mx.bfloat16
    assert fused[2].shape == stock[2].shape == (1, H, D, D)
    assert fused[2].dtype == stock[2].dtype == mx.float32
    assert mx.array_equal(fused[1], stock[1]).item()
    assert mx.allclose(fused[2], stock[2], rtol=1e-5, atol=1e-6).item()


def test_fused_prefill_then_fused_decode_is_bitwise_reference_decode():
    """Builds with fused GLM decode kernels: decode from a fused-prefill cache."""
    from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime

    # The MTP runtime (applied process-wide by some earlier tests) replaces
    # the layer calls, which do not route the fused decode kernels.
    if getattr(glm5_next_vlm_runtime, "_APPLIED", False):
        pytest.skip("glm5_next MTP runtime replaced the layer calls in this process")
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.glm5_next import language

    if not hasattr(language.Glm5NextLinearAttention, "_decode_step") or not hasattr(
        language, "_DECODE_FUSION"
    ):
        pytest.skip("fused GLM decode kernels are not part of this build")
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    config, attn = _attention(num_heads=8, hidden=1024, seed=6, bits=8)
    mx.random.seed(12)
    x = (0.5 * mx.random.normal((1, 300, config.hidden_size))).astype(mx.bfloat16)
    prefilled = ArraysCache(size=2)
    mx.eval(attn(x, None, prefilled))
    before = dk.STATS["kda"]
    fused_cache, ref_cache = ArraysCache(size=2), ArraysCache(size=2)
    for c in (fused_cache, ref_cache):
        c[0], c[1] = prefilled[0], prefilled[1]
    saved = language._DECODE_FUSION
    try:
        for width in (1, 3, 8):
            block = (0.5 * mx.random.normal((1, width, config.hidden_size))).astype(
                mx.bfloat16
            )
            language._DECODE_FUSION = True
            fused = attn(block, None, fused_cache)
            language._DECODE_FUSION = False
            ref = attn(block, None, ref_cache)
            mx.eval(fused, ref, fused_cache[0], fused_cache[1], ref_cache[0], ref_cache[1])
            assert mx.array_equal(fused, ref).item(), width
            assert mx.array_equal(fused_cache[0], ref_cache[0]).item(), width
            assert mx.array_equal(fused_cache[1], ref_cache[1]).item(), width
    finally:
        language._DECODE_FUSION = saved
    assert dk.STATS["kda"] > before


def test_prework_reads_qkv_in_place_from_the_fused_projection():
    """A wider input (the whole fused projection) gives the concat's bits."""
    from omlx.patches.glm53_kda_prework import kda_prework_fused

    mx.random.seed(11)
    heads, dim, length = 4, 128, 37
    c_dim = 3 * heads * dim
    fused = (mx.random.normal((1, length, c_dim + 320)) * 0.5).astype(mx.bfloat16)
    conv_state = (mx.random.normal((1, 3, c_dim)) * 0.5).astype(mx.bfloat16)
    conv_w = (mx.random.normal((c_dim, 1, 4)) * 0.3).astype(mx.bfloat16)
    scale = mx.array(dim**-0.5, dtype=mx.float32)
    wide = kda_prework_fused(fused, conv_state, conv_w, scale, length, heads, dim)
    packed = kda_prework_fused(
        mx.contiguous(fused[..., :c_dim]), conv_state, conv_w, scale, length, heads, dim
    )
    for a, b in zip(wide, packed):
        assert a.shape == b.shape
        assert mx.array_equal(a, b).item()
