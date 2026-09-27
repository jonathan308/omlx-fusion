# SPDX-License-Identifier: Apache-2.0
"""MiMo V2 decode fast path: bit-exactness against the reference layer ops."""

import importlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.mimo_v2 import decode_fast as df
from omlx.patches.mimo_v2 import moe_decode as md


# The fused decode kernels are validated (bitwise for one row) and enabled on
# M5 (NAX) GPUs; other GPUs keep the reference path.
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or not df._nax_available(),
    reason="MiMo fused decode kernels are validated and enabled on M5 (NAX) GPUs",
)

BF16 = mx.bfloat16


def _mismatches(a, b):
    a = np.array(a.astype(mx.float32))
    b = np.array(b.astype(mx.float32))
    assert a.shape == b.shape
    return int((a != b).sum())


def _mimo():
    from omlx.patches.mimo_v2 import apply_mimo_v2_patch

    apply_mimo_v2_patch()
    return importlib.import_module("mlx_lm.models.mimo_v2")


@pytest.mark.parametrize("shape", [(1, 1, 4096), (1, 4, 4096), (2, 3, 1024), (1, 2, 4000)])
def test_add_rms_is_bit_exact(shape):
    mx.random.seed(0)
    x = (mx.random.normal(shape) * 3).astype(BF16)
    y = (mx.random.normal(shape) * 2).astype(BF16)
    w = (1 + 0.1 * mx.random.normal(shape[-1:])).astype(BF16)
    h, n, f = df.add_rms(x, y, w, 1e-6, want_f32=True)
    h_ref = x + y
    n_ref = mx.fast.rms_norm(h_ref, w, 1e-6)
    assert _mismatches(h, h_ref) == 0
    assert _mismatches(n, n_ref) == 0
    assert f.dtype == mx.float32 and _mismatches(f, n_ref.astype(mx.float32)) == 0
    h2, n2 = df.add_rms(x, y, w, 1e-6)
    assert _mismatches(h2, h_ref) == 0 and _mismatches(n2, n_ref) == 0


@pytest.mark.parametrize("shape", [(1, 1, 4096), (1, 4, 4096), (2, 3, 1024)])
def test_combine_rms_is_bit_exact(shape):
    mx.random.seed(1)
    h = (mx.random.normal(shape) * 3).astype(BF16)
    y = mx.random.normal(shape[:-1] + (8, shape[-1])).astype(BF16)
    s = mx.softmax(mx.random.normal(shape[:-1] + (8,)), axis=-1)
    w = (1 + 0.1 * mx.random.normal(shape[-1:])).astype(BF16)
    h_out, n_out = df.combine_rms(h, y, s, w, 1e-6)
    h_ref = h + (y * s[..., None]).sum(axis=-2).astype(BF16)
    assert _mismatches(h_out, h_ref) == 0
    assert _mismatches(n_out, mx.fast.rms_norm(h_ref, w, 1e-6)) == 0


@pytest.mark.parametrize("rows", [1, 4, 7])
def test_router_select_matches_reference(rows):
    m = _mimo()
    mx.random.seed(2)
    x = mx.random.normal((1, rows, 512)).astype(BF16)
    weight = (mx.random.normal((256, 512)) * 0.05).astype(BF16)
    bias = mx.random.normal((256,)) * 0.05
    logits = x.astype(mx.float32) @ weight.astype(mx.float32).T
    ref_inds, ref_scores = m.group_expert_select(logits, bias, 8, 1, 1, 1.0, True)
    inds, scores = df.router_select(logits, bias, 8, True, 1.0)
    assert inds.dtype == ref_inds.dtype
    assert np.array_equal(np.array(inds), np.array(ref_inds))
    if rows == 1:
        # Decode: bitwise.
        assert _mismatches(scores, ref_scores) == 0
    else:
        # Verify rows: the reference normalises with mlx's row reduction,
        # whose summation order differs between the stock wheel's
        # precompiled kernel and a source build; identical experts, scores
        # within that summation-order difference.
        np.testing.assert_array_max_ulp(
            np.array(scores.astype(mx.float32)),
            np.array(ref_scores.astype(mx.float32)),
            maxulp=2,
        )


@pytest.mark.parametrize(
    "B,L,offset", [(1, 1, 1234), (1, 4, 77), (2, 3, "array"), (2, 4, "left_padded")]
)
@pytest.mark.parametrize("base", [10000.0, 1e7])
def test_qkv_rope_split_is_bit_exact(B, L, offset, base):
    mx.random.seed(3)
    H, Hkv, D, Dv = 8, 2, 192, 128
    qkv = mx.random.normal((B, L, H * D + Hkv * D + Hkv * Dv)).astype(BF16)
    if offset == "array":
        offset = mx.array([100, 5000])
    elif offset == "left_padded":
        # Batch caches start left-padded rows at negative offsets; MLX's rope
        # adds them to the unsigned row index.
        offset = mx.array([-3, 5000])
    rope = nn.RoPE(64, traditional=False, base=base)
    q = qkv[..., : H * D].reshape(B, L, H, D).swapaxes(1, 2)
    k = qkv[..., H * D : (H + Hkv) * D].reshape(B, L, Hkv, D).swapaxes(1, 2)
    v = qkv[..., (H + Hkv) * D :].reshape(B, L, Hkv, Dv).swapaxes(1, 2) * 0.707
    fq, fk, fv = df.qkv_rope_split(
        qkv,
        offset,
        n_heads=H,
        n_kv_heads=Hkv,
        head_dim=D,
        v_head_dim=Dv,
        rope_dims=64,
        rope_base=base,
        v_scale=0.707,
    )
    assert _mismatches(fq, rope(q, offset=offset)) == 0
    assert _mismatches(fk, rope(k, offset=offset)) == 0
    assert _mismatches(fv, v) == 0


@pytest.mark.parametrize("rows", [1, 2, 4, 7])
def test_router_logits_match_decode_gemv(rows):
    mx.random.seed(4)
    x = mx.random.normal((1, rows, 4096)).astype(BF16)
    weight = (mx.random.normal((256, 4096)) * 0.05).astype(BF16)
    w32 = weight.astype(mx.float32)
    # The reference router at decode: one row, float32 gemv.
    ref = mx.concatenate(
        [x[:, i : i + 1].astype(mx.float32) @ w32.T for i in range(rows)], axis=1
    )
    out = df.router_logits(x, weight)
    assert out.dtype == mx.float32
    assert _mismatches(out, ref) == 0


def _per_row_router(monkeypatch, m):
    """Reference router with every token row through MLX's M=1 gemv.

    The fast path computes each row's router logits exactly like a decode
    step does; MLX's batched float32 matmul (M > 1) sums in another order.
    """
    orig = m.MoEGate.__call__

    def call(self, x):
        if x.shape[-2] * x.shape[0] == 1:
            return orig(self, x)
        w32 = self.weight.astype(mx.float32)
        rows = [
            mx.concatenate(
                [x[b : b + 1, i : i + 1].astype(mx.float32) @ w32.T for i in range(x.shape[1])],
                axis=1,
            )
            for b in range(x.shape[0])
        ]
        return m.group_expert_select(
            mx.concatenate(rows, axis=0),
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )

    monkeypatch.setattr(m.MoEGate, "__call__", call)


_TINY = {
    "model_type": "mimo_v2",
    "vocab_size": 512,
    "hidden_size": 1024,
    "intermediate_size": 512,
    "moe_intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 48,
    "v_head_dim": 32,
    "rope_theta": 1e7,
    "swa_num_attention_heads": 8,
    "swa_num_key_value_heads": 4,
    "swa_head_dim": 48,
    "swa_v_head_dim": 32,
    "swa_rope_theta": 10000.0,
    "sliding_window_size": 32,
    "add_full_attention_sink_bias": False,
    "add_swa_attention_sink_bias": True,
    "hybrid_layer_pattern": [0, 1, 1, 0],
    "moe_layer_freq": [0, 1, 1, 1],
    "n_routed_experts": 64,
    "num_experts_per_tok": 8,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "topk_method": "noaux_tc",
    "partial_rotary_factor": 0.334,
    "attention_bias": False,
    "layernorm_epsilon": 1e-6,
    "max_position_embeddings": 4096,
    "attention_value_scale": 0.707,
}


def _tiny_model(seed=3):
    m = _mimo()
    mx.random.seed(seed)
    model = m.Model(m.ModelArgs.from_dict(dict(_TINY)))
    updates = []
    for key, value in tree_flatten(model.parameters()):
        if key.endswith("gate.weight"):
            updates.append((key, mx.random.normal(value.shape) * 0.05))
        elif key.endswith("e_score_correction_bias"):
            updates.append((key, mx.random.normal(value.shape) * 0.02))
        elif key.endswith("attention_sink_bias"):
            updates.append((key, mx.random.normal(value.shape)))
        elif "norm" in key:
            updates.append((key, 1 + 0.1 * mx.random.normal(value.shape)))
    model.load_weights(updates, strict=False)
    # MiMo-V2.6-Flash layout: 8-bit affine attention/dense, MXFP4 experts.
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda p, mod: isinstance(mod, nn.Linear) and "switch_mlp" not in p,
    )
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        mode="mxfp4",
        class_predicate=lambda p, mod: "switch_mlp" in p and hasattr(mod, "to_quantized"),
    )
    casts = [
        (k, v.astype(BF16))
        for k, v in tree_flatten(model.parameters())
        if v.dtype == mx.float32 and "e_score_correction_bias" not in k
    ]
    model.load_weights(casts, strict=False)
    mx.eval(model.parameters())
    return model


def _clone(caches):
    # Fresh array objects: KV caches update their buffers with in-place slice
    # assignment, which would otherwise write through to the other clone.
    out = []
    for c in caches:
        n = type(c).__new__(type(c))
        n.__dict__.update(
            {k: mx.array(v) if isinstance(v, mx.array) else v for k, v in c.__dict__.items()}
        )
        out.append(n)
    return out


def _forward(model, tokens, cache, fast, monkeypatch):
    monkeypatch.setenv("OMLX_MIMO_DECODE_FAST", "1" if fast else "0")
    out = model(tokens, cache=cache)
    mx.eval(out, [c.state for c in cache])
    return out


def _count_fast_runs(monkeypatch):
    calls = {"n": 0, "experts": 0}
    orig = df.combine_rms
    orig_experts = df._experts

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    def counted_experts(*a, **k):
        calls["experts"] += 1
        return orig_experts(*a, **k)

    monkeypatch.setattr(df, "combine_rms", counted)
    monkeypatch.setattr(df, "_experts", counted_experts)
    return calls


def test_fast_forward_matches_reference(monkeypatch):
    model = _tiny_model()
    _per_row_router(monkeypatch, _mimo())
    calls = _count_fast_runs(monkeypatch)
    tokens = mx.random.randint(0, 512, (1, 80))
    cache = model.make_cache()
    _forward(model, tokens[:, :40], cache, False, monkeypatch)
    pos = 40
    for L in [1, 1, 2, 3, 4, 1, 5, 7, 1, 2]:
        step = tokens[:, pos : pos + L]
        ref_cache, fast_cache = _clone(cache), _clone(cache)
        ref = _forward(model, step, ref_cache, False, monkeypatch)
        before, before_experts = calls["n"], calls["experts"]
        fast = _forward(model, step, fast_cache, True, monkeypatch)
        assert calls["n"] > before, f"fast path did not run at L={L}"
        assert calls["experts"] > before_experts, f"expert kernels did not run at L={L}"
        assert _mismatches(ref, fast) == 0, f"logits differ at L={L}"
        for a, b in zip(ref_cache, fast_cache):
            assert a.offset == b.offset
            assert _mismatches(a.state[0], b.state[0]) == 0
            assert _mismatches(a.state[1], b.state[1]) == 0
        cache = fast_cache
        pos += L
    # The fused q/k/v buffer is shared by the (view) projections.
    attn = model.model.layers[1].self_attn
    fused = attn.__dict__["_omlx_qkv"]
    n_q = attn.q_proj.weight.shape[0]
    assert np.array_equal(np.array(fused.weight[:n_q]), np.array(attn.q_proj.weight))


def test_fast_forward_matches_reference_batch_caches(monkeypatch):
    model = _tiny_model(seed=4)
    _per_row_router(monkeypatch, _mimo())
    calls = _count_fast_runs(monkeypatch)
    tokens = mx.random.randint(0, 512, (1, 48))
    ca, cb = model.make_cache(), model.make_cache()
    _forward(model, tokens[:, :45], ca, False, monkeypatch)
    _forward(model, tokens[:, 3:40], cb, False, monkeypatch)
    batch = [type(a).merge([a, b]) for a, b in zip(ca, cb)]
    for L in [1, 2, 3, 1]:
        step = mx.random.randint(0, 512, (2, L))
        ref_cache, fast_cache = _clone(batch), _clone(batch)
        ref = _forward(model, step, ref_cache, False, monkeypatch)
        before = calls["n"]
        fast = _forward(model, step, fast_cache, True, monkeypatch)
        assert calls["n"] > before
        assert _mismatches(ref, fast) == 0
        batch = fast_cache


def test_fast_path_declines_unsupported_forwards(monkeypatch):
    model = _tiny_model(seed=5)
    inner = model.model
    cache = model.make_cache()
    h = inner.embed_tokens(mx.array([[1] * 8]))
    # 8 rows x top-8 would take SwitchGLU's sorted path: reference loop.
    assert df.run_layers(inner, h, cache, None, None) is None
    h1 = inner.embed_tokens(mx.array([[1]]))
    assert df.run_layers(inner, h1, [None] * len(cache), None, None) is None
    monkeypatch.setenv("OMLX_MIMO_DECODE_FAST", "0")
    assert df.run_layers(inner, h1, cache, None, None) is None
    monkeypatch.setenv("OMLX_MIMO_DECODE_FAST", "1")
    h32 = h1.astype(mx.float32)
    assert df.run_layers(inner, h32, cache, None, None) is None


def test_prepare_rejects_unsupported_rope(monkeypatch):
    model = _tiny_model(seed=6)
    inner = model.model
    inner.layers[0].self_attn.rope = nn.RoPE(16, traditional=True, base=10000.0)
    assert df._prepare(inner) is False
    cache = model.make_cache()
    h1 = inner.embed_tokens(mx.array([[1]]))
    assert df.run_layers(inner, h1, cache, None, None) is None


def test_fast_path_uses_the_model_modules_sdpa(monkeypatch):
    """SDPA patches rebind the model module's global; the fast path must call
    exactly what the reference Attention.__call__ would."""
    m = _mimo()
    model = _tiny_model(seed=7)
    calls = {"n": 0}
    orig = m.scaled_dot_product_attention

    def wrapped(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(m, "scaled_dot_product_attention", wrapped)
    cache = model.make_cache()
    _forward(model, mx.array([[1, 2, 3, 4, 5]]), cache, False, monkeypatch)
    calls["n"] = 0
    _forward(model, mx.array([[6]]), cache, True, monkeypatch)
    assert calls["n"] == len(model.model.layers)


def _mxfp4(e, n, k, seed):
    mx.random.seed(seed)
    w = (mx.random.normal((e, n, k)) * 0.05).astype(BF16)
    wq, sc = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
    return wq, sc


def _gather(x, w, sc, inds):
    return mx.gather_qmm(
        x, w, sc, None, rhs_indices=inds, transpose=True, group_size=32, bits=4, mode="mxfp4"
    )


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("layout", ["split", "fused"])
def test_expert_kernels_match_gather_qmm(rows, layout):
    """gate/up + SwiGLU and down per (row, expert) pair, bit-exact to MLX's
    gather_qmv + mlx-lm swiglu; shared experts are computed once."""
    from mlx_lm.models.activations import swiglu

    E, D, F = 12, 1024, 512
    g, u, d = _mxfp4(E, F, D, 1), _mxfp4(E, F, D, 2), _mxfp4(E, D, F, 3)
    rng = np.random.default_rng(rows)
    inds = mx.array(
        np.stack([rng.choice(E, 8, replace=False) for _ in range(rows)])[None].astype(np.uint32)
    )
    mx.random.seed(rows)
    x = (mx.random.normal((1, rows, D)) * 2).astype(BF16)
    xe = mx.expand_dims(x, (-2, -3))
    ref_act = swiglu(_gather(xe, *g, inds), _gather(xe, *u, inds))
    ref_y = _gather(ref_act, *d, inds).squeeze(-2)
    ref_act = ref_act.squeeze(-2)
    if layout == "fused":
        gu_w = mx.concatenate([g[0], u[0]], axis=1)
        gu_s = mx.concatenate([g[1], u[1]], axis=1)
        act = md.gate_up_swiglu(x, inds, gu_w, gu_s, gu_w, gu_s, n_out=F, up_offset=F)
    else:
        act = md.gate_up_swiglu(x, inds, g[0], g[1], u[0], u[1], n_out=F)
    y = md.down_proj(act, inds, d[0], d[1])
    assert act.shape == (1, rows, 8, F) and y.shape == (1, rows, 8, D)
    assert _mismatches(act, ref_act) == 0
    assert _mismatches(y, ref_y) == 0


def test_expert_kind_detects_layouts():
    model = _tiny_model(seed=8)
    mlp = model.model.layers[1].mlp
    assert df._expert_kind(mlp.switch_mlp) == "split"
    sw = mlp.switch_mlp
    gate, up = sw.gate_proj, sw.up_proj
    gate.weight = mx.concatenate([gate.weight, up.weight], axis=1)
    gate.scales = mx.concatenate([gate.scales, up.scales], axis=1)
    sw.gate_up_proj = gate
    del sw.gate_proj
    del sw.up_proj
    assert df._expert_kind(sw) == "fused"
    # Affine-quantized experts: MLX's own gather path.
    model2 = _tiny_model(seed=9)
    sw2 = model2.model.layers[1].mlp.switch_mlp
    sw2.down_proj.mode = "affine"
    assert df._expert_kind(sw2) is None


def _attention_f32(q, k, v, scale, mask, sinks):
    """float32 reference attention (GQA, causal string / bool mask, sinks)."""
    B, H, L, D = q.shape
    Hk, S = k.shape[1], k.shape[2]
    rep = H // Hk
    q32 = q.astype(mx.float32).reshape(B, Hk, rep, L, D)
    k32 = k.astype(mx.float32)[:, :, None]
    v32 = v.astype(mx.float32)[:, :, None]
    scores = (q32 * scale) @ k32.swapaxes(-1, -2)
    if isinstance(mask, str):
        allowed = mx.arange(S - L, S)[:, None] >= mx.arange(S)[None]
        scores = mx.where(allowed, scores, -mx.inf)
    elif mask is not None:
        m = mask if mask.ndim < 3 else mask.reshape(mask.shape[0], 1, *mask.shape[1:])
        scores = mx.where(m, scores, -mx.inf)
    if sinks is not None:
        s = sinks.astype(mx.float32).reshape(1, Hk, rep, 1, 1)
        s = mx.broadcast_to(s, (*scores.shape[:-1], 1))
        scores = mx.concatenate([s, scores], axis=-1)
    p = mx.softmax(scores, axis=-1)
    if sinks is not None:
        p = p[..., 1:]
    return (p @ v32).reshape(B, H, L, -1)


@pytest.mark.parametrize("L", [3, 4, 7])
@pytest.mark.parametrize("mask_kind", ["causal", "array", "window", "padded"])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_sdpa_row_chunks_match_attention(L, mask_kind, with_sinks):
    """Row-chunked vector SDPA (verify rows x GQA > 32) computes the same
    attention as the full-width call, to bf16 rounding."""
    from mlx_lm.models.base import create_causal_mask, scaled_dot_product_attention

    mx.random.seed(L)
    B, H, Hk, D, Dv, S = (2 if mask_kind == "padded" else 1), 64, 4, 192, 128, 300
    q = mx.random.normal((B, H, L, D)).astype(BF16)
    k = mx.random.normal((B, Hk, S, D)).astype(BF16)
    v = mx.random.normal((B, Hk, S, Dv)).astype(BF16)
    sinks = mx.random.normal((H,)).astype(BF16) if with_sinks else None
    if mask_kind == "causal":
        mask = "causal"
    elif mask_kind == "array":
        mask = create_causal_mask(L, offset=S - L)
    elif mask_kind == "window":
        mask = create_causal_mask(L, offset=S - L, window_size=64)
    else:
        mask = create_causal_mask(L, offset=S - L, left_padding=mx.array([0, 17]))
    scale = D ** -0.5
    out = df._sdpa_row_chunks(
        scaled_dot_product_attention, q, k, v, None, scale, mask, sinks, 32 // (H // Hk)
    )
    assert out.shape == (B, L, H * Dv)
    ref = _attention_f32(q, k, v, scale, mask, sinks).swapaxes(1, 2).reshape(B, L, -1)
    full = scaled_dot_product_attention(q, k, v, None, scale=scale, mask=mask, sinks=sinks)
    full = full.swapaxes(1, 2).reshape(B, L, -1)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    err_full = mx.abs(full.astype(mx.float32) - ref).max().item()
    # Both bf16 results sit within bf16 rounding of the float32 attention.
    assert err <= max(2 * err_full, 1e-2), (err, err_full)


def _gqa16_model(seed=11):
    m = _mimo()
    cfg = dict(_TINY)
    cfg.update(num_attention_heads=16, num_key_value_heads=1, swa_num_attention_heads=16,
               swa_num_key_value_heads=2)
    mx.random.seed(seed)
    model = m.Model(m.ModelArgs.from_dict(cfg))
    updates = []
    for key, value in tree_flatten(model.parameters()):
        if key.endswith("gate.weight"):
            updates.append((key, mx.random.normal(value.shape) * 0.05))
        elif key.endswith("e_score_correction_bias"):
            updates.append((key, mx.random.normal(value.shape) * 0.02))
        elif key.endswith("attention_sink_bias"):
            updates.append((key, mx.random.normal(value.shape)))
        elif "norm" in key:
            updates.append((key, 1 + 0.1 * mx.random.normal(value.shape)))
    model.load_weights(updates, strict=False)
    nn.quantize(model, group_size=64, bits=8,
                class_predicate=lambda p, mod: isinstance(mod, nn.Linear) and "switch_mlp" not in p)
    nn.quantize(model, group_size=32, bits=4, mode="mxfp4",
                class_predicate=lambda p, mod: "switch_mlp" in p and hasattr(mod, "to_quantized"))
    model.load_weights([(k, v.astype(BF16)) for k, v in tree_flatten(model.parameters())
                        if v.dtype == mx.float32 and "e_score_correction_bias" not in k], strict=False)
    mx.eval(model.parameters())
    return model


def _chunked_reference(monkeypatch, m):
    """Reference SDPA with the fast path's verify contract (row chunks)."""
    orig_sdpa = m.scaled_dot_product_attention
    chunked = {"n": 0}

    def ref_sdpa(q, k, v, cache=None, scale=1.0, mask=None, sinks=None):
        rep = q.shape[1] // k.shape[1]
        L = q.shape[2]
        if 1 < L <= 8 and L * rep > 32:
            chunked["n"] += 1
            out = df._sdpa_row_chunks(orig_sdpa, q, k, v, cache, scale, mask, sinks, 32 // rep)
            return out.reshape(q.shape[0], L, q.shape[1], -1).swapaxes(1, 2)
        return orig_sdpa(q, k, v, cache=cache, scale=scale, mask=mask, sinks=sinks)

    monkeypatch.setattr(m, "scaled_dot_product_attention", ref_sdpa)
    return chunked


def test_fast_forward_gqa16_chunked_sdpa_matches_reference(monkeypatch):
    """GQA 16 (MiMo's full-attention layers): verify forwards of 3+ rows run
    the vector SDPA in row chunks; everything else stays bit-exact to the
    reference computed with the same attention and a per-row router."""
    m = _mimo()
    model = _gqa16_model()
    _per_row_router(monkeypatch, m)
    chunked = _chunked_reference(monkeypatch, m)
    tokens = mx.random.randint(0, 512, (1, 60))
    cache = model.make_cache()
    _forward(model, tokens[:, :40], cache, False, monkeypatch)
    pos = 40
    for L in [1, 3, 4, 2, 7]:
        step = tokens[:, pos : pos + L]
        ref_cache, fast_cache = _clone(cache), _clone(cache)
        before = chunked["n"]
        ref = _forward(model, step, ref_cache, False, monkeypatch)
        if L >= 3:
            assert chunked["n"] > before
        fast = _forward(model, step, fast_cache, True, monkeypatch)
        assert _mismatches(ref, fast) == 0, f"logits differ at L={L}"
        for a, b in zip(ref_cache, fast_cache):
            assert _mismatches(a.state[0], b.state[0]) == 0
            assert _mismatches(a.state[1], b.state[1]) == 0
        cache = fast_cache
        pos += L


def test_fast_forward_gqa16_chunked_sdpa_batch_caches(monkeypatch):
    """Row chunks with merged batch caches (left padding, array masks)."""
    m = _mimo()
    model = _gqa16_model(seed=12)
    _per_row_router(monkeypatch, m)
    chunked = _chunked_reference(monkeypatch, m)
    tokens = mx.random.randint(0, 512, (1, 50))
    ca, cb = model.make_cache(), model.make_cache()
    _forward(model, tokens[:, :45], ca, False, monkeypatch)
    _forward(model, tokens[:, 5:36], cb, False, monkeypatch)
    batch = [type(a).merge([a, b]) for a, b in zip(ca, cb)]
    calls = _count_fast_runs(monkeypatch)
    for L in [3, 1, 3, 2]:  # 2 x L rows stay within the fast path's 7
        step = mx.random.randint(0, 512, (2, L))
        ref_cache, fast_cache = _clone(batch), _clone(batch)
        before = chunked["n"]
        ref = _forward(model, step, ref_cache, False, monkeypatch)
        if L >= 3:
            assert chunked["n"] > before
        before_fast = calls["n"]
        fast = _forward(model, step, fast_cache, True, monkeypatch)
        assert calls["n"] > before_fast, f"fast path did not run at L={L}"
        assert _mismatches(ref, fast) == 0, f"logits differ at L={L}"
        batch = fast_cache


def test_fast_path_follows_weight_and_module_changes(monkeypatch):
    """Caches built on the first fast forward (fused q/k/v buffer, float32
    router bias, expert layout) follow later weight loads and regroups."""
    m = _mimo()
    model = _tiny_model(seed=13)
    _per_row_router(monkeypatch, m)
    inner = model.model
    tokens = mx.random.randint(0, 512, (1, 40))
    cache = model.make_cache()
    _forward(model, tokens[:, :30], cache, False, monkeypatch)

    def check(step):
        ref_cache, fast_cache = _clone(cache), _clone(cache)
        ref = _forward(model, step, ref_cache, False, monkeypatch)
        fast = _forward(model, step, fast_cache, True, monkeypatch)
        assert _mismatches(ref, fast) == 0

    check(tokens[:, 30:31])  # arms the fast path
    # 1) new attention weights (e.g. a reload): the stale fused buffer is rebuilt
    attn = inner.layers[1].self_attn
    old_fused = attn.__dict__["_omlx_qkv"]
    attn.q_proj.weight = mx.array(np.array(attn.q_proj.weight)[::-1].copy())
    check(tokens[:, 31:33])
    assert attn.__dict__["_omlx_qkv"] is not old_fused
    # 2) a new router bias
    gate = inner.layers[1].mlp.gate
    gate.e_score_correction_bias = gate.e_score_correction_bias + 0.05
    check(tokens[:, 33:34])
    assert gate.__dict__["_omlx_gate"].source is gate.e_score_correction_bias
    # 3) gate/up regrouped into [gate; up] after the fast path armed
    for layer in inner.layers:
        sw = getattr(layer.mlp, "switch_mlp", None)
        if sw is None:
            continue
        g, u = sw.gate_proj, sw.up_proj
        g.weight = mx.concatenate([g.weight, u.weight], axis=1)
        g.scales = mx.concatenate([g.scales, u.scales], axis=1)
        sw.gate_up_proj = g
        del sw.gate_proj
        del sw.up_proj
        # the reference forward of the fused layout (split after one gather)

        def call(self, x, indices, scores=None, weighted_sum=False):
            # per-expert rows (what MoE combines at decode sizes)
            gu = self.gate_up_proj
            xe = mx.expand_dims(x, (-2, -3))
            y = mx.gather_qmm(xe, gu.weight, gu.scales, gu.get("biases"), rhs_indices=indices,
                              transpose=True, group_size=gu.group_size, bits=gu.bits, mode=gu.mode)
            xg, xu = mx.split(y, 2, axis=-1)
            y = self.down_proj(self.activation(xu, xg), indices)
            return y.squeeze(-2)

        monkeypatch.setattr(type(sw), "__call__", call)
    check(tokens[:, 34:36])
    assert inner.layers[1].mlp.__dict__["_omlx_experts"][0] == "fused"


def test_expert_kind_rejects_wrapped_switch_modules():
    """An expert-offload style wrapper keeps a SwiGLU activation but is not a
    stock SwitchGLU: MLX's own path."""
    model = _tiny_model(seed=14)
    sw = model.model.layers[1].mlp.switch_mlp

    class OffloadSwitchGLU(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.activation = glu.activation
            self.down_proj = glu.down_proj

    assert df._expert_kind(OffloadSwitchGLU(sw)) is None
