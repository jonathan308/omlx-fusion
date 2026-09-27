# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness of GLM-5.3's fused decode/verify kernels.

Every fused path must reproduce the reference op graph it replaces bit for
bit (compared through integer views), both at L == 1 and for each row of a
short verify block.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.patches import mlx_vlm_glm5_next_compat as compat

def _nax() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


# The fused kernels replay the reference graph bit for bit on M5 (NAX) GPUs,
# where they are enabled; other GPUs keep the reference path.
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or not _nax(),
    reason="fused GLM decode kernels are validated and enabled on M5 (NAX) GPUs",
)


@pytest.fixture(autouse=True)
def _apply_glm5_next_compat():
    compat.apply_mlx_vlm_glm5_next_compat_patch()


def _skip_under_mtp_runtime():
    """Model-level fused decode checks need the vendor layer calls.

    ``glm5_next_vlm_runtime.apply()`` (MTP checkpoints) replaces the layer
    ``__call__`` methods process-wide and those bodies do not route the fused
    decode kernels, so once an earlier test in the process applied it these
    checks do not apply.
    """
    from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime

    if getattr(glm5_next_vlm_runtime, "_APPLIED", False):
        pytest.skip("glm5_next MTP runtime replaced the layer calls in this process")


def _language():
    from mlx_vlm.models.glm5_next import language

    return language


@pytest.fixture(autouse=True)
def _all_fused_families(monkeypatch):
    """Exactness tests cover every family, including default-off ones."""
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    monkeypatch.setattr(dk, "DISABLED", set())


def _stats():
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels

    return decode_kernels.STATS


def _bits(a: mx.array) -> mx.array:
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.view(view)


def _mismatches(a: mx.array, b: mx.array) -> int:
    assert a.shape == b.shape and a.dtype == b.dtype
    return int(mx.sum(_bits(a) != _bits(b)).item())


def _rand_affine(lead, out_dims, in_dims, bits, group_size, scale=0.004):
    """Random packed affine weights + bf16 scales/biases (valid for any bits)."""
    words = in_dims * bits // 32
    w = mx.random.randint(0, 2**31 - 1, (*lead, out_dims, words)).astype(mx.uint32)
    w = w * 2 + mx.random.randint(0, 2, w.shape).astype(mx.uint32)
    groups = in_dims // group_size
    s = (mx.random.uniform(0.5, 1.5, (*lead, out_dims, groups)) * scale).astype(
        mx.bfloat16
    )
    b = (
        -mx.random.uniform(0.5, 1.5, (*lead, out_dims, groups)) * scale * 2**bits / 2
    ).astype(mx.bfloat16)
    return w, s, b


def _quantized_linear(out_dims, in_dims, bits, group_size=64):
    layer = nn.QuantizedLinear(64, 64, bias=False, group_size=group_size, bits=bits)
    layer.weight, layer.scales, layer.biases = _rand_affine(
        (), out_dims, in_dims, bits, group_size
    )
    return layer


def _switch_linear(experts, out_dims, in_dims, bits, group_size=64):
    from omlx.patches.deepseek_v4.switch_layers import QuantizedSwitchLinear

    layer = QuantizedSwitchLinear(64, 64, 2, False, group_size, bits)
    layer.weight, layer.scales, layer.biases = _rand_affine(
        (experts,), out_dims, in_dims, bits, group_size
    )
    return layer


# ---------------------------------------------------------------------------
# Hyper-connection collapse + branch RMSNorm
# ---------------------------------------------------------------------------


def _hyper_connection(hidden=4096, seed=0):
    from mlx_vlm.models.deepseek_v4.hyper_connection import HyperConnection

    mx.random.seed(seed)
    cfg = SimpleNamespace(
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-5,
        hidden_size=hidden,
    )
    hc = HyperConnection(cfg)
    hc.fn = mx.random.normal(hc.fn.shape) * 0.01
    hc.base = mx.random.normal(hc.base.shape) * 0.5
    hc.scale = mx.random.uniform(0.5, 1.5, (3,))
    norm = nn.RMSNorm(hidden, eps=1e-5)
    norm.weight = mx.random.uniform(0.5, 1.5, (hidden,)).astype(mx.bfloat16)
    hc.eval()
    norm.eval()
    mx.eval(hc.parameters(), norm.parameters())
    return hc, norm


@pytest.mark.parametrize("hidden", [4096, 1024])
@pytest.mark.parametrize("length", [1, 2, 4, 8])
def test_decode_hc_pre_is_bitwise_reference(hidden, length):
    language = _language()
    hc, norm = _hyper_connection(hidden, seed=length)
    for trial in range(3):
        x = (mx.random.normal((1, length, 4, hidden)) * (1 + 2 * trial)).astype(
            mx.bfloat16
        )
        collapsed, post, comb = hc(x)
        reference = norm(collapsed)
        fused = language._decode_hc_pre(hc, norm, x)
        assert fused is not None
        normalized, fused_post, fused_comb = fused
        assert _mismatches(normalized, reference) == 0
        assert _mismatches(fused_post, post) == 0
        assert _mismatches(fused_comb, comb) == 0
        # Each verify row equals the one-token decode of that row.
        for row in range(length):
            single = language._decode_hc_pre(hc, norm, x[:, row : row + 1])
            assert _mismatches(single[0], normalized[:, row : row + 1]) == 0
            assert _mismatches(single[2], fused_comb[:, row : row + 1]) == 0


def _check_one_token_hc_expand(hidden):
    """Bitwise check of the fused one-token expand; returns engaged calls."""
    from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand

    language = _language()
    hc, norm = _hyper_connection(hidden, seed=hidden)
    engaged = 0
    for trial in range(8):
        residual = (mx.random.normal((1, 1, 4, hidden)) * (1 + 2 * trial)).astype(
            mx.bfloat16
        )
        _, post, comb = hc(residual)
        x = (mx.random.normal((1, 1, hidden)) * (0.5 + trial)).astype(mx.bfloat16)
        before = _stats()["hc_expand"]
        fused = language._decode_hc_expand(x, residual, post, comb)
        engaged += _stats()["hc_expand"] - before
        reference = hc_expand(x, residual, post, comb)
        assert _mismatches(fused, reference) == 0
        compiled = mx.compile(lambda *a: language._decode_hc_expand(*a))(
            x, residual, post, comb
        )
        assert _mismatches(compiled, reference) == 0
    return engaged


def _run_with_tf32(snippet: str) -> str:
    """Run ``snippet`` with this module imported as ``t`` and MLX TF32 on.

    The test session disables TF32 (conftest), which moves MLX's fp32 GEMMs
    off the NAX units; the production default keeps them there.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    code = (
        "import sys; sys.path[:0] = [%r, %r]\n"
        "from omlx.patches import mlx_vlm_glm5_next_compat as compat\n"
        "compat.apply_mlx_vlm_glm5_next_compat_patch()\n"
        "import test_glm5_next_decode_kernels as t\n" % (str(here), str(here.parent))
    ) + snippet
    env = dict(os.environ, MLX_ENABLE_TF32="1", OMLX_GLM5_DECODE_DISABLE="")
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return done.stdout


@pytest.mark.parametrize("hidden", [4096, 1024])
def test_one_token_hc_expand_declines_without_nax_tf32(hidden):
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    if dk.nax_relaxed_fp32_matmul():
        pytest.skip("TF32 NAX matmuls are enabled in this session")
    assert _check_one_token_hc_expand(hidden) == 0


def test_one_token_hc_expand_is_bitwise_reference_with_nax_tf32():
    out = _run_with_tf32(
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "if dk.nax_relaxed_fp32_matmul():\n"
        "    for hidden in (4096, 1024):\n"
        "        assert t._check_one_token_hc_expand(hidden) == 8\n"
        "    print('checked')\n"
        "else:\n"
        "    print('no-nax')\n"
    )
    if "no-nax" in out:
        pytest.skip("this GPU runs fp32 GEMMs without NAX")
    assert "checked" in out


def _check_hc_deferred_chain(hidden):
    """Chained one-token HC pres with each expand folded into the next
    (``_decode_hc_pre_deferred``) against the reference HyperConnection,
    RMSNorm and hc_expand; returns the number of checked half-layers."""
    from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand

    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    language = _language()
    checked = 0
    for seed in range(6):
        layers = [_hyper_connection(hidden, seed=100 * seed + i) for i in range(4)]
        mx.random.seed(seed)
        h_ref = (mx.random.normal((1, 1, 4, hidden)) * (1 + seed)).astype(mx.bfloat16)
        x = h_ref
        for step, (hc, norm) in enumerate(layers):
            if seed % 2:
                hc.base = hc.base * 10  # sharper sinkhorn / sigmoid inputs
            assert dk.hc_defer_supported(hc, norm, mx.bfloat16, hidden)
            collapsed, post, comb = hc(h_ref)
            reference = norm(collapsed)
            before = _stats()["hc_pre_fused"]
            xn, h, f_post, f_comb, mm = language._decode_hc_pre_deferred(hc, norm, x)
            assert _stats()["hc_pre_fused"] == before + 1
            assert _mismatches(h, h_ref) == 0, (seed, step)
            assert _mismatches(xn, reference) == 0, (seed, step)
            assert _mismatches(f_post, post) == 0, (seed, step)
            assert _mismatches(f_comb, comb) == 0, (seed, step)
            y = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(mx.bfloat16)
            h_ref = hc_expand(y, h_ref, post, comb)
            x = language._HCDeferred(y, h, f_post, f_comb, mm)
            checked += 1
        assert _mismatches(x.materialize(), h_ref) == 0
    return checked


def test_hc_deferred_chain_declines_without_nax_tf32():
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    if dk.nax_relaxed_fp32_matmul():
        pytest.skip("TF32 NAX matmuls are enabled in this session")
    hc, norm = _hyper_connection(1024)
    assert not dk.hc_defer_supported(hc, norm, mx.bfloat16, 1024)


def test_hc_deferred_chain_is_bitwise_reference_with_nax_tf32():
    out = _run_with_tf32(
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "if dk.nax_relaxed_fp32_matmul():\n"
        "    for hidden in (4096, 1024, 2048):\n"
        "        assert t._check_hc_deferred_chain(hidden) == 24\n"
        "    print('checked')\n"
        "else:\n"
        "    print('no-nax')\n"
    )
    if "no-nax" in out:
        pytest.skip("this GPU runs fp32 GEMMs without NAX")
    assert "checked" in out


def test_hc_defer_declines_uncovered_connections():
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    hc, norm = _hyper_connection(1024)
    assert not dk.hc_defer_supported(hc, norm, mx.float32, 1024)
    hc3, norm3 = _hyper_connection(768)
    assert not dk.hc_defer_supported(hc3, norm3, mx.bfloat16, 768)
    dk.DISABLED.add("hc_defer")
    assert not dk.hc_defer_supported(hc, norm, mx.bfloat16, 1024)


def test_decode_hc_pre_declines_uncovered_inputs():
    language = _language()
    hc, norm = _hyper_connection(1024)
    assert language._decode_hc_pre(hc, norm, mx.zeros((2, 1, 4, 1024), mx.bfloat16)) is None
    assert language._decode_hc_pre(hc, norm, mx.zeros((1, 9, 4, 1024), mx.bfloat16)) is None
    hc.train()
    assert language._decode_hc_pre(hc, norm, mx.zeros((1, 1, 4, 1024), mx.bfloat16)) is None


# ---------------------------------------------------------------------------
# MoE experts
# ---------------------------------------------------------------------------


def _moe(experts=16, hidden=1024, inter=512, top_k=8, shared_bits=8, seed=0):
    language = _language()
    mx.random.seed(seed)
    cfg = SimpleNamespace(
        hidden_size=hidden, moe_intermediate_size=inter, n_routed_experts=experts,
        swiglu_limit=10.0, num_experts_per_tok=top_k, norm_topk_prob=True,
        n_group=1, topk_group=1, routed_scaling_factor=2.5,
        n_shared_experts=1 if shared_bits else None, intermediate_size=4 * hidden,
    )
    moe = language.Glm5NextMoE(cfg)
    sw = moe.switch_mlp
    sw.gate_proj = _switch_linear(experts, inter, hidden, 4)
    sw.up_proj = _switch_linear(experts, inter, hidden, 4)
    sw.down_proj = _switch_linear(experts, hidden, inter, 4)
    if shared_bits:
        sh = moe.shared_experts
        sh.gate_proj = _quantized_linear(inter, hidden, shared_bits)
        sh.up_proj = _quantized_linear(inter, hidden, shared_bits)
        sh.down_proj = _quantized_linear(hidden, inter, shared_bits)
    moe.gate.weight = mx.random.normal((experts, hidden)) * 0.02
    moe.gate.e_score_correction_bias = mx.random.normal((experts,)) * 0.01
    moe.eval()
    mx.eval(moe.parameters())
    return moe


@pytest.mark.parametrize("slot_major", [True, False])
@pytest.mark.parametrize("shared_bits", [8, 4, 0])
@pytest.mark.parametrize("length", [1, 2, 4, 7])
def test_decode_experts_are_bitwise_reference(length, shared_bits, slot_major, monkeypatch):
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    language = _language()
    monkeypatch.setattr(dk, "DISABLED", set() if slot_major else {"moe_slot_major"})
    moe = _moe(shared_bits=shared_bits, seed=length)
    for trial in range(2):
        x = (mx.random.normal((1, length, 1024)) * (0.5 + trial)).astype(mx.bfloat16)
        indices, scores = moe.gate(x)
        wide_before = _stats()["moe_shared_wide"]
        down_before = _stats()["moe_down_shared_wide"]
        fused = moe._decode_experts(x, indices, scores)
        assert fused is not None
        wide_used = _stats()["moe_shared_wide"] - wide_before
        assert wide_used == (1 if shared_bits and length > 1 else 0)
        assert _stats()["moe_down_shared_wide"] - down_before == wide_used
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = moe(x)
        compiled = mx.compile(moe)(x) if length == 1 else reference
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        assert _mismatches(fused, reference) == 0
        assert _mismatches(fused, compiled) == 0
        assert _mismatches(moe(x), reference) == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_token_moe_selects_routes_inside_gate_up(seed, monkeypatch):
    """One token: the gate/up kernel replays the router's top-k selection
    (router logits -> gate/up -> down; the shared expert's gate/up as its own
    dispatch), bitwise like the one-launch and router-select-kernel paths
    and the reference MoE, including exact score ties."""
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    language = _language()
    moe = _moe(experts=288, hidden=4096, inter=2048, shared_bits=8, seed=seed)
    if seed == 2:
        weight = moe.gate.weight
        bias = moe.gate.e_score_correction_bias
        for e in (7, 70, 140, 280):  # exact duplicates of expert 200
            weight[e] = weight[200]
            bias[e] = bias[200]
        moe.gate.weight, moe.gate.e_score_correction_bias = weight, bias
    for trial in range(4):
        x = (mx.random.normal((1, 1, 4096)) * (0.3 + trial)).astype(mx.bfloat16)
        before = _stats()["router_select_fused"]
        split = _stats()["moe_shared_split"]
        fused = moe(x)
        assert _stats()["router_select_fused"] == before + 1
        assert _stats()["moe_shared_split"] == split + 1
        monkeypatch.setattr(dk, "DISABLED", {"moe_shared_split"})
        one_launch = moe(x)
        monkeypatch.setattr(dk, "DISABLED", {"router_select_fused"})
        two_step = moe(x)
        monkeypatch.setattr(dk, "DISABLED", set())
        assert _mismatches(fused, one_launch) == 0
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = moe(x)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        assert _mismatches(fused, two_step) == 0
        assert _mismatches(fused, reference) == 0


@pytest.mark.parametrize("bits", [8, 4])
def test_one_token_dense_mlp_gate_up_is_bitwise_reference(bits, monkeypatch):
    """GLM-5.3's dense MLP layers: gate/up + clamped SwiGLU in one dispatch
    for one token, bitwise like the eager and the compiled reference."""
    language = _language()
    cfg = SimpleNamespace(hidden_size=1024, intermediate_size=2048, swiglu_limit=10.0)
    mlp = language.Glm5NextMLP(cfg)
    mlp.gate_proj = _quantized_linear(2048, 1024, bits)
    mlp.up_proj = _quantized_linear(2048, 1024, bits)
    mlp.down_proj = _quantized_linear(1024, 2048, bits)
    mlp.eval()
    mx.eval(mlp.parameters())
    for trial in range(4):
        x = (mx.random.normal((1, 1, 1024)) * (0.5 + 3 * trial)).astype(mx.bfloat16)
        before = _stats()["mlp_gate_up"]
        fused = mlp(x)
        assert _stats()["mlp_gate_up"] == before + 1
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = mlp(x)
        compiled = mx.compile(mlp)(x)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        assert _mismatches(fused, reference) == 0
        assert _mismatches(fused, compiled) == 0


@pytest.mark.parametrize("bits", [8, 5, 4, 6])
@pytest.mark.parametrize("n_k", [(512, 256), (256, 512), (128, 1024)])
def test_one_token_mla_head_qmv_is_bitwise_reference(bits, n_k):
    """embed_q / unembed_out (QuantizedMultiLinear, 64 heads) for one token."""
    from mlx_lm.models.mla import MultiLinear

    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    N, K = n_k
    for gs in (64, 32):
        mx.random.seed(bits * 31 + N + gs)
        layer = MultiLinear(K, N, 64)
        layer.weight = (mx.random.normal(layer.weight.shape) * 0.05).astype(mx.bfloat16)
        layer = layer.to_quantized(gs, bits)
        for trial in range(3):
            x = (mx.random.normal((1, 64, 1, K)) * (1 + 2 * trial)).astype(mx.bfloat16)
            reference = layer(x)
            for nsg in (2, 8):
                fused = dk.mla_head_qmv(x, layer, nsg=nsg)
                assert fused is not None
                assert _mismatches(fused, reference) == 0, (gs, trial, nsg)


def test_mla_head_qmv_declines_qmv_quad_shapes():
    from mlx_lm.models.mla import MultiLinear

    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    layer = MultiLinear(128, 512, 8).to_quantized(64, 8)
    assert dk.mla_head_qmv(mx.zeros((1, 8, 1, 128), mx.bfloat16), layer) is None
    layer = MultiLinear(256, 512, 8).to_quantized(64, 8)
    assert dk.mla_head_qmv(mx.zeros((1, 8, 2, 256), mx.bfloat16), layer) is None
    assert dk.mla_head_qmv(mx.zeros((1, 8, 1, 256), mx.bfloat16), MultiLinear(256, 512, 8)) is None


@pytest.mark.parametrize("kv_len", [1, 300, 4099])
@pytest.mark.parametrize("width", [1, 7, 2051])
def test_dsa_gather_selected_is_bitwise_reference(kv_len, width):
    """The one-token sparse attention's clipped take_along_axis and mask."""
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    mx.random.seed(kv_len + width)
    cache = (mx.random.normal((1, 1, kv_len + 64, 512)) * 3).astype(mx.bfloat16)
    kv = cache[:, :, :kv_len, :]
    idx = mx.random.randint(-3, kv_len + 3, (1, 1, 1, width)).astype(mx.int32)
    idx = mx.where(idx >= kv_len, -1, idx)
    clamped = mx.clip(idx[:, :, 0, :], 0, kv_len - 1)[..., None]
    reference = mx.take_along_axis(
        kv, mx.broadcast_to(clamped, clamped.shape[:-1] + (512,)), axis=2
    )
    reference_mask = (idx >= 0)[:, :, 0, :][:, :, None, :]
    out, valid = dk.dsa_gather_selected(kv, idx[:, :, 0, :])
    assert _mismatches(out, reference) == 0
    assert valid.dtype == mx.bool_ and valid.shape == reference_mask.shape
    assert mx.array_equal(valid, reference_mask).item()


def test_decode_experts_leave_sorted_route_counts_to_switch_glu():
    moe = _moe()
    x = mx.random.normal((1, 8, 1024)).astype(mx.bfloat16)  # 64 routes -> sorted
    indices, scores = moe.gate(x)
    assert moe._decode_experts(x, indices, scores) is None


def test_decode_experts_compile_inside_ffn_graph(monkeypatch):
    language = _language()
    moe = _moe(seed=5)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    fused = mx.compile(moe)(x)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    reference = moe(x)
    assert _mismatches(fused, reference) == 0


# ---------------------------------------------------------------------------
# DSA indexer decode scores and selection
# ---------------------------------------------------------------------------


def _indexer(hidden=256, seed=0):
    language = _language()
    mx.random.seed(seed)
    cfg = SimpleNamespace(
        hidden_size=hidden, index_n_heads=32, index_head_dim=128, index_topk=2048,
        index_kpool=4, index_kpool_always_select_tail=True, q_lora_rank=128,
    )
    indexer = language.Glm5NextIndexer(cfg)
    indexer.index_kpool_compress_ape = mx.random.normal((4, 128)) * 0.1
    indexer.index_kpool_compress_gate = mx.random.normal((128, hidden)) * 0.05
    indexer.set_dtype(mx.bfloat16)
    indexer.eval()
    mx.eval(indexer.parameters())
    return indexer


def _native_indexer_available():
    from omlx.custom_kernels.glm_moe_dsa import fast

    return fast.has_symbol("dsa_indexer_scores") and fast.has_symbol("dsa_topk_indices")


@pytest.mark.parametrize("length", [1, 2, 4, 5, 8])
def test_dsa_decode_scores_match_native_steel_tile(length):
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    indexer = _indexer(seed=length)
    for pool in (513, 1024, 1090):
        q = mx.random.normal((1, length, 32, 128)).astype(mx.bfloat16)
        keys = (mx.random.normal((1, pool, 128)) * 0.7).astype(mx.bfloat16)
        weights = (mx.random.normal((1, length, 32)) * 0.1).astype(mx.bfloat16)
        pool_len = pool - 2
        first = pool_len * 4 - length + 1
        reference = indexer._native_scores(q, keys, weights)
        idx = mx.arange(pool)
        query_pos = first + mx.arange(length)
        valid = (idx[None, None] < pool_len) & (
            ((idx + 1) * 4 - 1)[None, None] <= query_pos[None, :, None]
        )
        reference = mx.where(valid, reference, -1e30)
        scores = dk.dsa_decode_scores(q, keys, weights, first, pool_len, 4)
        assert _mismatches(scores, reference) == 0


def _make_pool_caches():
    from omlx.patches.deepseek_v4 import apply_pooling_cache_support

    apply_pooling_cache_support()
    from mlx_lm.models.cache import KVCache, PoolingCache

    return PoolingCache(4), KVCache()


@pytest.mark.parametrize("pool", [512, 1025, 1026, 1500, 2048])
def test_dsa_topk_rows_matches_native_topk(pool):
    """Decode/verify indexer top-k (bitonic sort) against the native
    radix-select kernel: same indices in the same order, with exact score
    ties, -1e30 masked blocks and a NaN."""
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    for rows in (1, 4, 8):
        for trial in range(4):
            mx.random.seed(pool + 10 * rows + trial)
            s = mx.random.normal((1, rows, pool))
            if trial == 1:
                s = mx.round(s * 4) / 4
            if trial == 2:
                s = mx.where(mx.random.uniform(shape=s.shape) < 0.3, -1e30, s)
            s = s.astype(mx.bfloat16)
            if trial == 3:
                s = s.at[0, 0, 7].add(float("nan"))
            expected = fast.dsa_topk_indices(s[:, None], 512)[:, 0]
            got = dk.dsa_topk_rows(s, 512)
            assert got is not None and got.dtype == expected.dtype
            assert mx.array_equal(got, expected).item(), (rows, trial)


def test_indexer_fast_selection_matches_general_path():
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    indexer = _indexer(seed=11)
    hidden = 256
    mx.random.seed(12)
    prompt = (mx.random.normal((1, 2105, hidden)) * 0.5).astype(mx.bfloat16)
    qr_prompt = (mx.random.normal((1, 2105, 128)) * 0.5).astype(mx.bfloat16)
    pools = []
    for _ in range(2):
        pool, kv = _make_pool_caches()
        for start in range(0, 2105, 512):
            indexer.fast_decode = False
            out = indexer(
                prompt[:, start : start + 512], qr_prompt[:, start : start + 512],
                None, cache=pool, kv_cache=kv,
            )
            if out is not None:
                mx.eval(out)
        pools.append((pool, kv))
    (fast_pool, fast_kv), (ref_pool, ref_kv) = pools
    for step, width in enumerate([1, 1, 3, 1, 4, 8, 1, 2]):
        x = (mx.random.normal((1, width, hidden)) * 0.5).astype(mx.bfloat16)
        qr = (mx.random.normal((1, width, 128)) * 0.5).astype(mx.bfloat16)
        indexer.fast_decode = True
        fast = indexer(x, qr, None, cache=fast_pool, kv_cache=fast_kv)
        indexer.fast_decode = False
        reference = indexer(x, qr, None, cache=ref_pool, kv_cache=ref_kv)
        assert fast is not None and reference is not None
        assert fast.shape == reference.shape
        assert fast.dtype == reference.dtype
        assert int(mx.sum(fast != reference).item()) == 0, f"step {step} width {width}"


# ---------------------------------------------------------------------------
# End to end: a small quantized GLM-5.3 whose shapes engage every fused path
# ---------------------------------------------------------------------------


def _fused_shape_model(seed, heads=16, quantize_mla=False):
    from mlx_lm.models.mla import MultiLinear
    from mlx_vlm.models import glm5_next

    from omlx.patches.deepseek_v4.switch_layers import SwitchLinear

    quantized = (nn.Linear, SwitchLinear) + ((MultiLinear,) if quantize_mla else ())

    language = _language()
    text = glm5_next.TextConfig(
        model_type="glm5_next_text", vocab_size=256, hidden_size=1024,
        intermediate_size=2048, moe_intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=heads, num_key_value_heads=heads, n_shared_experts=1,
        n_routed_experts=128, routed_scaling_factor=2.5, kv_lora_rank=512,
        q_lora_rank=256, qk_rope_head_dim=0, v_head_dim=64, qk_nope_head_dim=64,
        num_experts_per_tok=8, first_k_dense_replace=1, max_position_embeddings=8192,
        rms_norm_eps=1e-5, index_topk=2048, index_head_dim=128, index_n_heads=32,
        layer_types=[
            "linear_attention", "deepseek_sparse_attention",
            "linear_attention", "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        linear_attn_config={
            "num_heads": 8, "head_dim": 128, "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=4, hc_mult=4, hc_sinkhorn_iters=20,
    )
    mx.random.seed(seed)
    model = language.LanguageModel(text)
    nn.quantize(
        model, group_size=64, bits=4,
        class_predicate=lambda _, m: isinstance(m, quantized),
    )
    params = []
    for name, value in nn.utils.tree_flatten(model.parameters()):
        if value.dtype == mx.uint32:
            continue
        if language.glm5_next_cast_predicate(name):
            value = (mx.random.normal(value.shape) * 0.05).astype(mx.bfloat16)
            if name.endswith("scales"):
                value = mx.abs(value) * 0.1 + 0.002
            if "norm" in name and name.endswith("weight"):
                value = (1.0 + value).astype(mx.bfloat16)
        else:
            value = mx.random.normal(value.shape) * 0.05
            if name.endswith("hc.scale") or name.endswith("hc.base"):
                value = value * 10
        params.append((name, value))
    model.load_weights(params, strict=False)
    model.eval()
    mx.eval(model.parameters())
    return model


def _check_small_model(seed=41, prompt_len=2101, heads=16, quantize_mla=False):
    """Fused vs reference logits of a small model, bitwise; returns families used.

    Prompts beyond index_topk (2048) run the sparse DSA paths, shorter ones
    the dense latent attention.
    """
    language = _language()
    fused_model = _fused_shape_model(seed, heads, quantize_mla)
    reference_model = _fused_shape_model(seed, heads, quantize_mla)
    prompt = mx.random.randint(0, 256, (1, prompt_len)).astype(mx.int32)
    caches = []
    for model in (fused_model, reference_model):
        cache = model.make_cache()
        for start in range(0, prompt.shape[1], 512):
            logits = model(prompt[:, start : start + 512], cache=cache).logits
            mx.eval(logits)
        caches.append(cache)
    fused_cache, reference_cache = caches
    next_ids = mx.argmax(logits[:, -1:], axis=-1).astype(mx.int32)
    before = dict(_stats())
    try:
        for step, width in enumerate([1, 1, 4, 1, 7, 2, 1, 8, 3]):
            block = mx.concatenate(
                [next_ids, (next_ids + mx.arange(1, width)[None]) % 256], axis=1
            )[:, :width]
            language._DECODE_FUSION = True
            fused = fused_model(block, cache=fused_cache).logits
            mx.eval(fused)
            language._DECODE_FUSION = False
            reference = reference_model(block, cache=reference_cache).logits
            mx.eval(reference)
            assert mx.all(mx.isfinite(reference)).item()
            assert _mismatches(fused, reference) == 0, f"step {step} width {width}"
            next_ids = mx.argmax(reference[:, -1:], axis=-1).astype(mx.int32)
    finally:
        language._DECODE_FUSION = True
    return {k for k, v in _stats().items() if v > before.get(k, 0)}


_ALWAYS_FUSED = {
    "hc_mix", "moe_gate_up", "moe_down", "dsa_scores", "kda", "router", "latent_attn",
    "multi_qmv", "router_select_fused", "moe_shared_split", "dsa_topk",
}


def test_small_model_decode_and_verify_logits_are_bitwise_reference():
    _skip_under_mtp_runtime()
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    used = _check_small_model()
    assert _ALWAYS_FUSED | {"latent_sparse_rows"} <= used, used


def test_small_model_default_families_are_bitwise_reference(monkeypatch):
    """The production family set (fused latent attention off): one-token
    sparse steps gather the selected latent rows in one dispatch."""
    _skip_under_mtp_runtime()
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    monkeypatch.setattr(dk, "DISABLED", set(dk.DEFAULT_DISABLED))
    used = _check_small_model()
    assert "dsa_gather" in used and "latent_attn" not in used, used
    # The routed gate/up kernel selects the routes and keeps the shared expert.
    assert "router_select_fused" in used and "moe_shared_split" not in used, used


def test_small_model_dense_attention_is_bitwise_reference():
    _skip_under_mtp_runtime()
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    used = _check_small_model(seed=43, prompt_len=300)
    assert {
        "hc_mix", "kda", "router", "moe_gate_up", "multi_qmv", "router_select_fused",
        "moe_shared_split",
    } <= used, used
    assert ("latent_attn" in used) == dk.nax_available(), used


def test_small_model_quantized_mla_is_bitwise_reference():
    """Quantized MLA projections (as in the checkpoint): unembed_out (K =
    kv_lora_rank) runs mla_head_qmv for one token; logits stay bitwise."""
    _skip_under_mtp_runtime()
    used = _check_small_model(seed=47, prompt_len=300, quantize_mla=True)
    assert "mla_head_qmv" in used, used


def test_small_model_bitwise_reference_with_nax_tf32():
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    out = _run_with_tf32(
        "print(sorted(t._check_small_model() | t._check_small_model(43, 300)))\n"
    )
    used = set(eval(out.strip().splitlines()[-1]))
    assert _ALWAYS_FUSED <= used, used
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    for family in ("hc_expand", "router_rows", "hc_pre_fused", "hc_post_mm"):
        if family in used:
            continue
        # Only acceptable where MLX itself would not use NAX relaxed fp32.
        assert not _run_with_tf32(
            "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
            "print(dk.nax_relaxed_fp32_matmul())\n"
        ).strip().endswith("True")
    del dk


def test_small_model_mtp_runtime_loop_defers_hc_bitwise():
    """The MTP runtime's replacement model loop (plain decode) folds the HC
    expands into the next layer like the vendor loop, bit for bit."""
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    out = _run_with_tf32(
        "from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime as rt\n"
        "assert rt.apply()\n"
        "from mlx_vlm.models.glm5_next import language as g5\n"
        "assert g5.Glm5NextModel._omlx_mtp_call_patched\n"
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "used = t._check_small_model(43, 300) | t._check_small_model()\n"
        "print(dk.nax_relaxed_fp32_matmul(), sorted(used))\n"
    )
    last = out.strip().splitlines()[-1]
    if last.startswith("True"):
        assert "'hc_pre_fused'" in last and "'hc_post_mm'" in last, last


# ---------------------------------------------------------------------------
# KDA linear attention layer body
# ---------------------------------------------------------------------------


def _kda_layer(heads=8, hidden=1024, gate_bits=8, seed=0, v_bits=8):
    from mlx_vlm.models import glm5_next

    language = _language()
    mx.random.seed(seed)
    config = SimpleNamespace(
        hidden_size=hidden, linear_num_heads=heads, linear_head_dim=128,
        linear_conv_kernel_dim=4, linear_lower_bound=-5.0, rms_norm_eps=1e-5,
    )
    layer = language.Glm5NextLinearAttention(config)
    qkv = heads * 128
    for name, (out_dims, in_dims) in {
        "q_proj": (qkv, hidden), "k_proj": (qkv, hidden), "v_proj": (qkv, hidden),
        "g_a_proj": (128, hidden), "b_proj": (heads, hidden),
    }.items():
        bits = v_bits if name == "v_proj" else 8
        setattr(layer, name, _quantized_linear(out_dims, in_dims, bits))
    fg = layer.forget_gate
    fg.f_a_proj = _quantized_linear(128, hidden, 8)
    fg.f_b_proj = _quantized_linear(qkv, 128, gate_bits)
    layer.g_b_proj = _quantized_linear(qkv, 128, gate_bits)
    fg.A_log = mx.random.normal((heads,)) * 0.5
    fg.dt_bias = mx.random.normal((qkv,)) * 0.5
    layer.conv1d.weight = (mx.random.normal((3 * qkv, 4, 1)) * 0.5).astype(mx.bfloat16)
    layer.o_norm.weight = mx.random.uniform(0.5, 1.5, (128,)).astype(mx.bfloat16)
    layer.o_proj = nn.Identity()  # compare the o_proj input directly
    layer.eval()
    mx.eval(layer.parameters())
    del glm5_next
    return layer


def _arrays_cache():
    from mlx_vlm.models.cache import ArraysCache

    return ArraysCache(size=2)


@pytest.mark.parametrize("gate_bits", [8, 5])
def test_kda_decode_step_is_bitwise_reference(gate_bits, monkeypatch):
    _skip_under_mtp_runtime()
    language = _language()
    layer = _kda_layer(gate_bits=gate_bits, seed=gate_bits)
    fused_cache, reference_cache = _arrays_cache(), _arrays_cache()
    prompt = (mx.random.normal((1, 12, 1024)) * 0.8).astype(mx.bfloat16)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    for cache in (fused_cache, reference_cache):
        mx.eval(layer(prompt, cache=cache))
    for step, width in enumerate([1, 1, 3, 8, 2, 1]):
        x = (mx.random.normal((1, width, 1024)) * (0.5 + step % 3)).astype(mx.bfloat16)
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = layer(x, cache=reference_cache)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        before = _stats()["kda"]
        fused = layer(x, cache=fused_cache)
        assert _stats()["kda"] == before + 1
        mx.eval(reference, fused, fused_cache.cache, reference_cache.cache)
        assert _mismatches(fused, reference) == 0, f"step {step} width {width}"
        assert _mismatches(fused_cache[0], reference_cache[0]) == 0
        assert _mismatches(fused_cache[1], reference_cache[1]) == 0


def test_eager_sigmoid_probe_reproduces_mx_sigmoid():
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    for dtype in (mx.bfloat16, mx.float32):
        precise = dk.eager_sigmoid_precise(dtype)
        assert precise in (True, False), dtype
        x = (mx.random.normal((4096,)) * 6).astype(dtype)
        kernel = mx.fast.metal_kernel(
            name="glm5_sigmoid_probe", input_names=["x"],
            output_names=["default_out", "precise_out"],
            header=dk._QMV_HEADER, source=dk._SIGMOID_PROBE_SOURCE,
        )
        default, exact = kernel(
            inputs=[x], template=[("T", dtype)], grid=(x.size, 1, 1),
            threadgroup=(256, 1, 1), output_shapes=[x.shape] * 2,
            output_dtypes=[dtype] * 2,
        )
        assert _mismatches(exact if precise else default, mx.sigmoid(x)) == 0


@pytest.mark.parametrize("gate_bits", [8, 5])
@pytest.mark.parametrize("seed", range(20, 30))
def test_kda_decode_step_seed_sweep_is_bitwise_reference(seed, gate_bits, monkeypatch):
    """The reference's beta and output-gate sigmoids are eager mx.sigmoid
    kernels, whose exp differs between MLX builds (precise in the release
    wheel's precompiled kernels); several of these seeds differed in a few
    outputs (and then in the recurrent state) when the kernel always used
    the runtime-compiled exp."""
    _skip_under_mtp_runtime()
    language = _language()
    layer = _kda_layer(seed=seed, gate_bits=gate_bits)
    fused_cache, reference_cache = _arrays_cache(), _arrays_cache()
    prompt = (mx.random.normal((1, 12, 1024)) * 0.8).astype(mx.bfloat16)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    for cache in (fused_cache, reference_cache):
        mx.eval(layer(prompt, cache=cache))
    for step, width in enumerate([1, 1, 4, 1, 8, 1, 1, 2, 1, 1]):
        x = (mx.random.normal((1, width, 1024)) * (0.5 + step % 3)).astype(mx.bfloat16)
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = layer(x, cache=reference_cache)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        gate5 = _stats()["kda_gate5"]
        fused = layer(x, cache=fused_cache)
        # 5-bit gate rows are replayed in the kernel for one token only.
        assert _stats()["kda_gate5"] - gate5 == int(gate_bits == 5 and width == 1)
        mx.eval(reference, fused, fused_cache.cache, reference_cache.cache)
        assert _mismatches(fused, reference) == 0, f"step {step} width {width}"
        assert _mismatches(fused_cache[1], reference_cache[1]) == 0


@pytest.mark.parametrize("v_bits", [5, 4])
def test_kda_decode_step_with_mixed_projection_bits(v_bits, monkeypatch):
    """GLM-5.3 layer 40 quantizes v_proj to 5 bits and q/k/gates to 8: the
    decode path runs one projection matmul per quantization instead of the
    reference layer body."""
    _skip_under_mtp_runtime()
    language = _language()
    layer = _kda_layer(seed=20 + v_bits, v_bits=v_bits)
    fused_cache, reference_cache = _arrays_cache(), _arrays_cache()
    prompt = (mx.random.normal((1, 12, 1024)) * 0.8).astype(mx.bfloat16)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    for cache in (fused_cache, reference_cache):
        mx.eval(layer(prompt, cache=cache))
    for step, width in enumerate([1, 1, 4, 1, 8]):
        x = (mx.random.normal((1, width, 1024)) * (0.5 + step % 3)).astype(mx.bfloat16)
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = layer(x, cache=reference_cache)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        before = _stats()["kda"]
        fused = layer(x, cache=fused_cache)
        assert _stats()["kda"] == before + 1
        mx.eval(reference, fused, fused_cache.cache, reference_cache.cache)
        assert _mismatches(fused, reference) == 0, f"step {step} width {width}"
        assert _mismatches(fused_cache[0], reference_cache[0]) == 0
        assert _mismatches(fused_cache[1], reference_cache[1]) == 0
    assert not layer._fused_ready and layer._decode_groups


@pytest.mark.parametrize("width", [1, 4])
def test_kda_decode_step_from_empty_cache(width, monkeypatch):
    _skip_under_mtp_runtime()
    language = _language()
    layer = _kda_layer(seed=30 + width)
    x = (mx.random.normal((1, width, 1024)) * 0.7).astype(mx.bfloat16)
    fused_cache, reference_cache = _arrays_cache(), _arrays_cache()
    before = _stats()["kda"]
    fused = layer(x, cache=fused_cache)
    assert _stats()["kda"] == before + 1
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    reference = layer(x, cache=reference_cache)
    mx.eval(fused, reference)
    assert _mismatches(fused, reference) == 0
    assert _mismatches(fused_cache[0], reference_cache[0]) == 0
    assert _mismatches(fused_cache[1], reference_cache[1]) == 0


# ---------------------------------------------------------------------------
# MoE router (one token)
# ---------------------------------------------------------------------------


def _router(experts=288, hidden=4096, seed=0):
    language = _language()
    mx.random.seed(seed)
    cfg = SimpleNamespace(
        num_experts_per_tok=8, norm_topk_prob=True, n_group=1, topk_group=1,
        routed_scaling_factor=2.5, n_routed_experts=experts, hidden_size=hidden,
    )
    gate = language.Glm5NextMoEGate(cfg)
    gate.weight = mx.random.normal((experts, hidden)) * 0.02
    gate.e_score_correction_bias = mx.random.normal((experts,)) * 0.01
    return gate


@pytest.mark.parametrize("experts,hidden,bias", [(288, 4096, 0.0), (288, 4096, 14.0), (64, 512, 0.0)])
def test_router_is_bitwise_reference(experts, hidden, bias, monkeypatch):
    """One-token router vs group_expert_select, bitwise over many draws.

    The reference takes the sigmoid with MLX's eager kernel (precise exp on
    release wheels, where a runtime-compiled exp differs in the last bit for
    a few percent of logits and so, after normalization, in ~5% of routes);
    ``bias`` 14 is the checkpoint's e_score_correction_bias level.
    """
    language = _language()
    gate = _router(experts, hidden, seed=experts)
    gate.e_score_correction_bias = gate.e_score_correction_bias + bias
    for trial in range(96):
        x = (mx.random.normal((1, 1, hidden)) * (0.3 + trial % 6)).astype(mx.bfloat16)
        before = _stats()["router"]
        indices, scores = gate(x)
        assert _stats()["router"] == before + 1
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        ref_indices, ref_scores = gate(x)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        assert indices.dtype == ref_indices.dtype and indices.shape == ref_indices.shape
        assert mx.array_equal(indices, ref_indices).item()
        assert _mismatches(scores, ref_scores) == 0, trial


def test_router_breaks_exact_ties_like_argpartition(monkeypatch):
    language = _language()
    gate = _router(seed=3)
    weight = gate.weight
    bias = gate.e_score_correction_bias
    # Experts 7, 70, 140 and 280 become exact duplicates of expert 200.
    for e in (7, 70, 140, 280):
        weight[e] = weight[200]
        bias[e] = bias[200]
    bias[200] = bias[200] + 1.0  # push the tied group into the top-k
    for e in (7, 70, 140, 280):
        bias[e] = bias[200]
    gate.weight, gate.e_score_correction_bias = weight, bias
    x = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
    indices, scores = gate(x)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    ref_indices, ref_scores = gate(x)
    tied = [int(i) for i in ref_indices[0, 0].tolist() if i in (7, 70, 140, 200, 280)]
    assert tied == sorted(tied) and len(tied) == 5
    assert mx.array_equal(indices, ref_indices).item()
    assert _mismatches(scores, ref_scores) == 0


def test_router_orders_nan_scores_like_argpartition(monkeypatch):
    language = _language()
    gate = _router(64, 512, seed=9)
    bias = gate.e_score_correction_bias
    for e in range(64):
        if e not in (5, 33, 60):
            bias[e] = float("nan")
    gate.e_score_correction_bias = bias
    x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
    before = _stats()["router"]
    indices, scores = gate(x)
    assert _stats()["router"] == before + 1
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    ref_indices, ref_scores = gate(x)
    assert sorted(ref_indices[0, 0].tolist()[:3]) == [5, 33, 60]
    assert mx.array_equal(indices, ref_indices).item()
    assert _mismatches(scores, ref_scores) == 0


def test_router_declines_other_gemv_configurations():
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    # K >= 16 * E selects MLX's split-K (bn=8) gemv, which is not replicated.
    x = mx.zeros((1, 1024), mx.bfloat16)
    assert dk.moe_router(x, mx.zeros((16, 1024)), mx.zeros((16,)), 8, 2.5, True) is None


def _check_router_rows(experts=288, hidden=4096):
    """Verify-block routers (2..8 rows), bitwise; returns engaged calls."""
    language = _language()
    gate = _router(experts, hidden, seed=7)
    engaged = 0
    for trial, rows in enumerate([2, 3, 4, 5, 8, 4]):
        x = (mx.random.normal((1, rows, hidden)) * (0.3 + trial)).astype(mx.bfloat16)
        before = _stats()["router_rows"]
        indices, scores = gate(x)
        engaged += _stats()["router_rows"] - before
        language._DECODE_FUSION = False
        try:
            ref_indices, ref_scores = gate(x)
        finally:
            language._DECODE_FUSION = True
        assert mx.array_equal(indices, ref_indices).item(), rows
        assert _mismatches(scores, ref_scores) == 0, rows
    return engaged


def test_router_rows_declines_without_nax_tf32():
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    if dk.nax_relaxed_fp32_matmul():
        pytest.skip("TF32 NAX matmuls are enabled in this session")
    assert _check_router_rows() == 0


def test_router_rows_bitwise_reference_with_nax_tf32():
    out = _run_with_tf32(
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "if dk.nax_relaxed_fp32_matmul():\n"
        "    for args in ((), (128, 1024)):\n"
        "        n = t._check_router_rows(*args)\n"
        "        declined = [k for k, ok in dk._ROUTER_ROWS_CHECKED.items() if not ok]\n"
        "        # Every call is bitwise the reference (checked above); a call may\n"
        "        # skip the kernel only because its configuration was declined by\n"
        "        # the first-use check (a build whose TF32 GEMM rounds that shape\n"
        "        # differently, e.g. the stock wheel for some row counts).\n"
        "        assert n == 6 or (0 < len(declined) and n >= 1), (n, declined)\n"
        "    print('checked')\n"
        "else:\n"
        "    print('no-nax')\n"
    )
    if "no-nax" in out:
        pytest.skip("this GPU runs fp32 GEMMs without NAX")
    assert "checked" in out


# ---------------------------------------------------------------------------
# Latent MLA attention (64 heads, as in GLM-5.3)
# ---------------------------------------------------------------------------


def _latent_ready():
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    if not dk.nax_available():
        pytest.skip("latent attention replicas need NAX GEMMs")
    return dk


@pytest.mark.parametrize("keys", [2, 200, 256, 1029, 2051])
def test_latent_attention_dense_decode_is_bitwise_sdpa(keys):
    dk = _latent_ready()
    mx.random.seed(keys)
    q = (mx.random.normal((1, 64, 1, 512)) * 0.6).astype(mx.bfloat16)
    kv = (mx.random.normal((1, 1, keys, 512)) * 0.8).astype(mx.bfloat16)
    reference = mx.fast.scaled_dot_product_attention(q, kv, kv, scale=256**-0.5)
    fused = dk.latent_attention(q, kv, 256**-0.5)
    assert fused is not None
    assert _mismatches(fused, reference) == 0


@pytest.mark.parametrize("width,cache_len", [(2051, 4100), (515, 3000)])
def test_latent_attention_sparse_decode_is_bitwise_gathered_sdpa(width, cache_len):
    dk = _latent_ready()
    mx.random.seed(width)
    q = (mx.random.normal((1, 64, 1, 512)) * 0.6).astype(mx.bfloat16)
    kv = (mx.random.normal((1, 1, cache_len, 512)) * 0.8).astype(mx.bfloat16)
    idx = mx.random.randint(-1, cache_len + 3, (width,)).astype(mx.int32)
    clamped = mx.clip(idx, 0, cache_len - 1)
    gathered = mx.take_along_axis(
        kv, mx.broadcast_to(clamped[None, None, :, None], (1, 1, width, 512)), axis=2
    )
    mask = (idx >= 0).reshape(1, 1, 1, width)
    reference = mx.fast.scaled_dot_product_attention(
        q, gathered, gathered, scale=256**-0.5, mask=mask
    )
    fused = dk.latent_attention(q, kv, 256**-0.5, indices=idx)
    assert _mismatches(fused, reference) == 0


@pytest.mark.parametrize("length", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("keys", [30, 1030, 2040])
def test_latent_attention_dense_verify_is_bitwise_masked_sdpa(length, keys):
    dk = _latent_ready()
    mx.random.seed(length * keys)
    q = (mx.random.normal((1, 64, length, 512)) * 0.6).astype(mx.bfloat16)
    kv = (mx.random.normal((1, 1, keys, 512)) * 0.8).astype(mx.bfloat16)
    offset = keys - length
    mask = mx.arange(offset, offset + length)[:, None] >= mx.arange(keys)[None]
    reference = mx.fast.scaled_dot_product_attention(
        q, kv, kv, scale=256**-0.5, mask=mask
    )
    assert _mismatches(dk.latent_attention(q, kv, 256**-0.5, mask=mask), reference) == 0
    causal = mx.fast.scaled_dot_product_attention(
        q, kv, kv, scale=256**-0.5, mask="causal"
    )
    assert _mismatches(dk.latent_attention(q, kv, 256**-0.5, causal=True), causal) == 0


@pytest.mark.parametrize("length", [2, 4, 5, 8])
@pytest.mark.parametrize("width,cache_len", [(2051, 4100), (515, 3000), (64, 200)])
def test_latent_attention_sparse_verify_is_bitwise_gathered_sdpa(length, width, cache_len):
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    mx.random.seed(length * width)
    q = (mx.random.normal((1, 64, length, 512)) * 0.6).astype(mx.bfloat16)
    kv = (mx.random.normal((1, 1, cache_len, 512)) * 0.8).astype(mx.bfloat16)
    sel = mx.random.randint(-1, cache_len + 3, (1, length, width)).astype(mx.int32)
    clamped = mx.clip(sel, 0, cache_len - 1)
    gathered = mx.take_along_axis(
        mx.broadcast_to(kv[:, 0, None], (1, length, cache_len, 512)),
        mx.broadcast_to(clamped[..., None], (1, length, width, 512)),
        axis=2,
    )
    reference = mx.fast.scaled_dot_product_attention(
        q.transpose(0, 2, 1, 3).reshape(length, 64, 1, 512),
        gathered.reshape(length, 1, width, 512),
        gathered.reshape(length, 1, width, 512),
        scale=256**-0.5,
        mask=(sel >= 0).reshape(length, 1, 1, width),
    ).reshape(1, length, 64, 512).transpose(0, 2, 1, 3)
    fused = dk.latent_attention_sparse_rows(q, kv, sel[0], 256**-0.5)
    assert _mismatches(fused, reference) == 0


# ---------------------------------------------------------------------------
# Projections sharing one input (MLA q_a / kv_a / indexer wk, weights_proj)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_multi_qmv_is_bitwise_separate_projections(tokens, bits):
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    language = _language()
    mx.random.seed(tokens * 10 + bits)
    k = 1024
    layers = [_quantized_linear(n, k, bits) for n in (512, 136, 128, 32)]
    x = (mx.random.normal((1, tokens, k)) * 0.7).astype(mx.bfloat16)
    for count in (1, 2, 4):
        group = layers[:count]
        fused = dk.multi_qmv(x.reshape(tokens, k), group)
        assert fused is not None
        for layer, out in zip(group, fused):
            reference = language.linear_forward(layer, x).reshape(tokens, -1)
            assert _mismatches(out, reference) == 0, (count, layer.weight.shape)


def test_multi_qmv_declines_uncovered_projections():
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    mx.random.seed(3)
    x = (mx.random.normal((1, 1024)) * 0.7).astype(mx.bfloat16)
    eight, six = _quantized_linear(256, 1024, 8), _quantized_linear(256, 1024, 6)
    assert dk.multi_qmv(x, [eight, six]) is None  # mixed bits
    assert dk.multi_qmv(x, [eight, _quantized_linear(12, 1024, 8)]) is None
    biased = nn.QuantizedLinear(1024, 256, bias=True, bits=8)
    assert dk.multi_qmv(x, [eight, biased]) is None
    assert dk.multi_qmv(x, [eight, nn.Linear(1024, 256, bias=False)]) is None
    assert dk.multi_qmv(mx.zeros((9, 1024), mx.bfloat16), [eight]) is None
    # one token: qmv_fast shapes only (4-bit needs K % 512 == 0)
    four = _quantized_linear(256, 1280, 4)
    assert dk.multi_qmv(mx.zeros((1, 1280), mx.bfloat16), [four, four]) is None


def test_multi_linear_groups_projections_by_quantization():
    language = _language()
    mx.random.seed(5)
    x = (mx.random.normal((1, 3, 1024)) * 0.7).astype(mx.bfloat16)
    layers = [
        _quantized_linear(512, 1024, 6),
        _quantized_linear(256, 1024, 8),
        _quantized_linear(128, 1024, 6),
        _quantized_linear(32, 1024, 8),
    ]
    before = _stats()["multi_qmv"]
    outs = language._multi_linear(x, layers)
    assert _stats()["multi_qmv"] == before + 2
    for layer, out in zip(layers, outs):
        assert _mismatches(out, language.linear_forward(layer, x)) == 0
    assert language._multi_linear(x, layers[:2]) is None


def test_disabled_families_take_the_reference_path(monkeypatch):
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    language = _language()
    mx.random.seed(11)
    x = (mx.random.normal((1, 2, 1024)) * 0.7).astype(mx.bfloat16)
    layers = [_quantized_linear(256, 1024, 8), _quantized_linear(128, 1024, 8)]
    monkeypatch.setattr(dk, "DISABLED", {"multi_qmv"})
    before = _stats()["multi_qmv"]
    assert dk.multi_qmv(x.reshape(2, 1024), layers) is None
    outs = language._multi_linear(x, layers)
    assert _stats()["multi_qmv"] == before
    for layer, out in zip(layers, outs):
        assert _mismatches(out, language.linear_forward(layer, x)) == 0


@pytest.mark.parametrize("prefetch", [1, 3, 8, 16])
def test_latent_attention_prefetch_depths_are_bitwise_sdpa(prefetch, monkeypatch):
    dk = _latent_ready()
    monkeypatch.setattr(dk, "_LATENT_PREFETCH", prefetch)
    mx.random.seed(prefetch)
    q = (mx.random.normal((1, 64, 1, 512)) * 0.6).astype(mx.bfloat16)
    kv = (mx.random.normal((1, 1, 3000, 512)) * 0.8).astype(mx.bfloat16)
    for width in (2051, 100):
        idx = mx.random.randint(-1, 3003, (width,)).astype(mx.int32)
        clamped = mx.clip(idx, 0, 2999)
        gathered = mx.take_along_axis(
            kv, mx.broadcast_to(clamped[None, None, :, None], (1, 1, width, 512)), axis=2
        )
        reference = mx.fast.scaled_dot_product_attention(
            q, gathered, gathered, scale=256**-0.5, mask=(idx >= 0).reshape(1, 1, 1, width)
        )
        fused = dk.latent_attention(q, kv, 256**-0.5, indices=idx)
        assert _mismatches(fused, reference) == 0, width
    q4 = (mx.random.normal((1, 64, 4, 512)) * 0.6).astype(mx.bfloat16)
    dense = kv[:, :, :1500]
    causal = mx.fast.scaled_dot_product_attention(
        q4, dense, dense, scale=256**-0.5, mask="causal"
    )
    assert _mismatches(dk.latent_attention(q4, dense, 256**-0.5, causal=True), causal) == 0


def _latent_production_cases():
    """Every compile-time latent configuration GLM-5.3 decode/verify uses."""
    mx.random.seed(21)
    kv = (mx.random.normal((1, 1, 4100, 512)) * 0.8).astype(mx.bfloat16)
    q1 = (mx.random.normal((1, 64, 1, 512)) * 0.6).astype(mx.bfloat16)
    cases = []
    for width in (2051, 100):
        idx = mx.random.randint(-1, 4103, (width,)).astype(mx.int32)
        rows = mx.clip(idx, 0, 4099)[None, None, :, None]
        g = mx.take_along_axis(kv, mx.broadcast_to(rows, (1, 1, width, 512)), axis=2)
        mask = (idx >= 0).reshape(1, 1, 1, width)
        cases.append((
            f"sparse {width}",
            mx.fast.scaled_dot_product_attention(q1, g, g, scale=256**-0.5, mask=mask),
            dict(q=q1, keys=kv, indices=idx),
        ))
    for keys in (100, 1000, 1500):
        d = kv[:, :, :keys]
        cases.append((
            f"dense {keys}",
            mx.fast.scaled_dot_product_attention(q1, d, d, scale=256**-0.5),
            dict(q=q1, keys=d),
        ))
    for length in range(2, 9):
        qL = (mx.random.normal((1, 64, length, 512)) * 0.6).astype(mx.bfloat16)
        for keys in (120, 1500):
            d = kv[:, :, :keys]
            m = mx.arange(keys - length, keys)[:, None] >= mx.arange(keys)[None]
            cases.append((
                f"verify {length}x{keys} mask",
                mx.fast.scaled_dot_product_attention(qL, d, d, scale=256**-0.5, mask=m),
                dict(q=qL, keys=d, mask=m),
            ))
            cases.append((
                f"verify {length}x{keys} causal",
                mx.fast.scaled_dot_product_attention(qL, d, d, scale=256**-0.5, mask="causal"),
                dict(q=qL, keys=d, causal=True),
            ))
    return cases


def test_latent_attention_production_configurations_are_bitwise_sdpa(monkeypatch):
    dk = _latent_ready()
    monkeypatch.setattr(dk, "_LATENT_CHECKED", {})
    for name, reference, kwargs in _latent_production_cases():
        q, keys = kwargs.pop("q"), kwargs.pop("keys")
        before = _stats()["latent_attn"]
        fused = dk.latent_attention(q, keys, 256**-0.5, **kwargs)
        assert fused is not None and _stats()["latent_attn"] == before + 1, name
        assert _mismatches(fused, reference) == 0, name
    assert all(dk._LATENT_CHECKED.values())


def test_latent_attention_first_use_check_rejects_wrong_kernels(monkeypatch):
    dk = _latent_ready()
    monkeypatch.setattr(dk, "_LATENT_CHECKED", {})
    real = dk._latent_kernels

    def broken(gather, mask_kind):
        scores, softmax, values = real(gather, mask_kind)

        def wrong_values(**kw):
            return [o + 1 for o in values(**kw)]

        return scores, softmax, wrong_values

    monkeypatch.setattr(dk, "_latent_kernels", broken)
    mx.random.seed(2)
    q = (mx.random.normal((1, 64, 1, 512)) * 0.6).astype(mx.bfloat16)
    kv = (mx.random.normal((1, 1, 300, 512)) * 0.8).astype(mx.bfloat16)
    assert dk.latent_attention(q, kv, 256**-0.5) is None
    assert dk.latent_attention(q, kv, 256**-0.5) is None  # cached verdict
    assert list(dk._LATENT_CHECKED.values()) == [False]


def test_router_rows_first_use_check_rejects_wrong_kernels():
    out = _run_with_tf32(
        "import mlx.core as mx\n"
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "if not dk.nax_relaxed_fp32_matmul():\n"
        "    print('no-nax')\n"
        "else:\n"
        "    real = dk._router_select_kernel()\n"
        "    def wrong(**kw):\n"
        "        idx, sc = real(**kw)\n"
        "        return idx, sc * 1.5\n"
        "    dk._router_select_kernel = lambda: wrong\n"
        "    gate = t._router(64, 512, seed=4)\n"
        "    x = mx.random.normal((4, 512)).astype(mx.bfloat16)\n"
        "    args = (gate.weight, gate.e_score_correction_bias, 8, 2.5, True)\n"
        "    assert dk.moe_router_rows(x, *args) is None\n"
        "    assert dk.moe_router_rows(x, *args) is None\n"
        "    assert list(dk._ROUTER_ROWS_CHECKED.values()) == [False]\n"
        "    print('checked')\n"
    )
    if "no-nax" in out:
        pytest.skip("this GPU runs fp32 GEMMs without NAX")
    assert "checked" in out


@pytest.mark.parametrize("tokens", [2, 5, 8])
def test_down_combine_folds_glm_shared_down_bitwise(tokens):
    """GLM-5.3 shapes: routed down 4-bit [E, 4096, 2048], shared down 8-bit."""
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    language = _language()
    mx.random.seed(tokens)
    routed = _switch_linear(10, 4096, 2048, 4)
    shared = _quantized_linear(4096, 2048, 8)
    act = (mx.random.normal((tokens, 8, 2048)) * 0.3).astype(mx.bfloat16)
    shared_act = (mx.random.normal((tokens, 2048)) * 0.3).astype(mx.bfloat16)
    idx = mx.stack([mx.random.permutation(10)[:8] for _ in range(tokens)]).astype(mx.uint32)
    scores = mx.random.uniform(0.05, 0.4, (tokens, 8))
    shared_y = language.linear_forward(shared, shared_act.reshape(1, tokens, -1)).reshape(tokens, -1)
    reference = dk.moe_down_combine(act, idx, scores, routed, shared_y=shared_y)
    fused = dk.moe_down_combine(act, idx, scores, routed, shared, shared_act=shared_act)
    assert fused is not None and reference is not None
    assert _mismatches(fused, reference) == 0


def test_latent_attention_declines_gemv_routed_scores():
    """Fewer than 16 query rows (heads x tokens) or a single key: MLX computes
    the scores with gemv_wide / gemv, which the NAX replica does not follow."""
    dk = _latent_ready()
    mx.random.seed(8)
    kv = (mx.random.normal((1, 1, 300, 512)) * 0.8).astype(mx.bfloat16)
    for heads, length in ((8, 1), (4, 2), (15, 1)):
        q = (mx.random.normal((1, heads, length, 512)) * 0.6).astype(mx.bfloat16)
        assert dk.latent_attention(q, kv, 256**-0.5, causal=length > 1) is None
    q = (mx.random.normal((1, 8, 2, 512)) * 0.6).astype(mx.bfloat16)
    assert dk.latent_attention(q, kv, 256**-0.5, causal=True) is not None  # 16 rows
    q64 = (mx.random.normal((1, 64, 1, 512)) * 0.6).astype(mx.bfloat16)
    assert dk.latent_attention(q64, kv[:, :, :1], 256**-0.5) is None


def test_kda_fast_prefill_then_fused_decode_is_bitwise_reference():
    """Caches written by the fused KDA prefill (perf/glm-gdn-prefill, >= 64-token
    chunks) feed the fused decode/verify path exactly like the stock ones."""
    _language()
    try:
        from mlx_vlm.models.glm5_next import kda_prefill
    except ImportError:
        pytest.skip("this build has no fused KDA prefill")
    if not getattr(kda_prefill, "ENABLED", False):
        pytest.skip("fused KDA prefill disabled")
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    used = _check_small_model() | _check_small_model(43, 300)
    assert getattr(kda_prefill, "_ENGAGED_LOGGED", True)
    assert _ALWAYS_FUSED <= used, used


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("seed", [41, 44, 45, 48])
@pytest.mark.parametrize("prompt_len", [300, 2101])
def test_small_model_seed_sweep_is_bitwise_reference(seed, prompt_len, heads):
    """Includes seed 48 / 300 tokens / 8 heads, where a latent-attention
    replica of the NAX scores GEMM differed from MLX's gemv_wide route
    (fewer than 16 query rows) in a few elements."""
    if prompt_len > 2048 and not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    _check_small_model(seed, prompt_len, heads)


def _fuse_gate_up(switch_mlp):
    """The MoE gate/up fusion's layout: gate_up_proj = [gate; up] rows per
    expert (omlx.patches.moe_gate_up_fusion._fuse_one)."""
    gate, up = switch_mlp.gate_proj, switch_mlp.up_proj
    fused = {f: mx.concatenate([gate[f], up[f]], axis=1) for f in ("weight", "scales", "biases")}
    mx.eval(list(fused.values()))
    for field, value in fused.items():
        setattr(gate, field, value)
    switch_mlp.gate_up_proj = gate
    del switch_mlp.gate_proj
    del switch_mlp.up_proj


@pytest.mark.parametrize("length", [1, 2, 5, 8])
def test_decode_experts_read_fused_gate_up_layout(length, monkeypatch):
    language = _language()
    moe = _moe(seed=10 + length)
    x = (mx.random.normal((1, length, 1024)) * 0.7).astype(mx.bfloat16)
    indices, scores = moe.gate(x)
    split = moe._decode_experts(x, indices, scores)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    reference = moe(x)
    monkeypatch.setattr(language, "_DECODE_FUSION", True)
    if split is None:  # 64 routes: sorted, left to SwitchGLU
        assert length == 8
    else:
        assert _mismatches(split, reference) == 0
    _fuse_gate_up(moe.switch_mlp)
    fused = moe._decode_experts(x, indices, scores)
    assert (fused is None) == (split is None)
    if fused is not None:
        assert _mismatches(fused, split) == 0
    if hasattr(type(moe.switch_mlp), "projections"):
        # This build's SwitchGLU runs the fused layout (MoE gate/up fusion).
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        fused_reference = moe(x)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        assert _mismatches(fused_reference, reference) == 0
        assert _mismatches(moe(x), reference) == 0


def test_latent_attention_kernels_are_off_by_default():
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "OMLX_GLM5_DECODE_DISABLE"}
    code = (
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "print(sorted(dk.DISABLED))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300
    ).stdout
    assert out.strip().splitlines()[-1] == "['latent_attn', 'latent_sparse_rows', 'moe_shared_split']"


def test_router_rows_first_use_check_inside_compile_uses_reference():
    out = _run_with_tf32(
        "import mlx.core as mx\n"
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "if not dk.nax_relaxed_fp32_matmul():\n"
        "    print('no-nax')\n"
        "else:\n"
        "    language = t._language()\n"
        "    gate = t._router(64, 512, seed=6)\n"
        "    x = mx.random.normal((1, 4, 512)).astype(mx.bfloat16)\n"
        "    traced = mx.compile(lambda v: gate(v))(x)\n"
        "    assert dk._ROUTER_ROWS_CHECKED == {}  # no check inside compile\n"
        "    eager = gate(x)\n"
        "    assert list(dk._ROUTER_ROWS_CHECKED.values()) == [True]\n"
        "    mx.eval(traced)\n"
        "    language._DECODE_FUSION = False\n"
        "    ref_eager = gate(x)\n"
        "    assert mx.array_equal(eager[0], ref_eager[0]).item()\n"
        "    assert mx.array_equal(eager[1].view(mx.uint32), ref_eager[1].view(mx.uint32)).item()\n"
        "    print('checked')\n"
    )
    if "no-nax" in out:
        pytest.skip("this GPU runs fp32 GEMMs without NAX")
    assert "checked" in out


def test_upstream_kda_prefill_then_fused_decode_is_bitwise_reference(monkeypatch):
    """Caches written by upstream's fused KDA prefill (glm53_kda_prework,
    >= 64-row chunks; the test prompts' 512-token chunks) feed the fused
    decode and verify paths exactly like the stock prefill's."""
    _skip_under_mtp_runtime()
    try:
        from omlx.patches import glm53_kda_prework as prework
    except ImportError:
        pytest.skip("this build has no glm53 fused KDA prefill")
    _language()
    if not getattr(prework, "_GLM53_KDA_PREFILL_ENABLED", False):
        pytest.skip("glm53 fused KDA prefill disabled")
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")

    monkeypatch.setattr(prework, "_GLM53_KDA_ENGAGED_LOGGED", False)
    used = _check_small_model() | _check_small_model(43, 300)
    assert prework._GLM53_KDA_ENGAGED_LOGGED
    assert _ALWAYS_FUSED <= used, used


@pytest.mark.parametrize("every", [1, 3])
def test_decode_early_eval_only_schedules(every, monkeypatch):
    """One-token decode forwards evaluate every few layers while the graph is
    still being built; the logits and caches are those of the lazy forward."""
    _skip_under_mtp_runtime()
    language = _language()
    model = _fused_shape_model(seed=45)
    prompt = mx.random.randint(0, 256, (1, 300)).astype(mx.int32)
    caches = []
    for _ in range(2):
        cache = model.make_cache()
        mx.eval(model(prompt, cache=cache).logits)
        caches.append(cache)
    calls = []
    real_async_eval = mx.async_eval

    def counting_async_eval(*args):
        calls.append(len(args))
        return real_async_eval(*args)

    token = mx.array([[17]], dtype=mx.int32)
    for step in range(3):
        monkeypatch.setattr(language, "_DECODE_EVAL_EVERY", 0)
        lazy = model(token, cache=caches[0]).logits
        mx.eval(lazy)
        monkeypatch.setattr(language, "_DECODE_EVAL_EVERY", every)
        monkeypatch.setattr(mx, "async_eval", counting_async_eval)
        early = model(token, cache=caches[1]).logits
        monkeypatch.setattr(mx, "async_eval", real_async_eval)
        mx.eval(early)
        assert _mismatches(early, lazy) == 0, f"step {step}"
        token = mx.argmax(lazy[:, -1:], axis=-1).astype(mx.int32)
    # 4 layers: evaluations after layers `every`, 2 * every, ... (not the last).
    assert len(calls) == 3 * len(range(every, 4, every))
    for a, b in zip(caches[0], caches[1]):
        for x, y in zip(a.state, b.state):
            if isinstance(x, mx.array):
                assert _mismatches(x, y) == 0


def test_compiled_decode_releases_the_weights_when_the_model_is_dropped():
    """One-token steps compile each layer's FFN half around multi-output fused
    kernels (router logits, route-selecting gate/up, HC pre/post). MLX 0.32.2
    leaks such intermediates of a compiled trace with everything they reference
    (ml-explore/mlx#4453): with the weights as trace constants, dropping the
    model left MoE layers' routed gate/up experts allocated (~73 MB per layer
    here, ~2.5 GB on GLM-5.3). Runs on a worker thread whose final
    ``mx.clear_streams()`` drops its compile cache, like an engine thread."""
    import gc
    import threading

    result = {}

    def work():
        gc.collect()
        mx.clear_cache()
        base = mx.get_active_memory()
        # Pausing the collector makes the groups MLX 0.32.2 leaks deterministic
        # here (one to three MoE layers' gate/up experts without the fix).
        gc.disable()
        try:
            run()
        finally:
            gc.enable()
        gc.collect()
        mx.synchronize()
        mx.clear_streams()
        gc.collect()
        mx.clear_cache()
        result["leak"] = mx.get_active_memory() - base

    def run():
        model = _fused_shape_model(7)
        cache = model.make_cache()
        prompt = mx.random.randint(0, 256, (1, 64)).astype(mx.int32)
        logits = model(prompt, cache=cache).logits
        token = mx.argmax(logits[:, -1:], axis=-1).astype(mx.int32)
        before = dict(_stats())
        for _ in range(3):
            logits = model(token, cache=cache).logits
            token = mx.argmax(logits[:, -1:], axis=-1).astype(mx.int32)
            mx.eval(token)
        result["router_select_fused"] = _stats()["router_select_fused"] - before.get(
            "router_select_fused", 0
        )

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert result["router_select_fused"] > 0  # the compiled fused MoE path ran
    assert result["leak"] < (1 << 20), f"{result['leak']} bytes still active"
