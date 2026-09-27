# SPDX-License-Identifier: Apache-2.0
"""MiMo's MoE combines expert outputs with oMLX's fused weighted-sum kernel."""

import mlx.core as mx
import mlx.nn as nn
import pytest


def _mimo_module():
    import importlib

    from omlx.patches.mimo_v2 import apply_mimo_v2_patch

    apply_mimo_v2_patch()
    return importlib.import_module("mlx_lm.models.mimo_v2")


def _small_moe(mimo, T, hidden=128, inter=64, experts=16, top_k=8):
    cfg = mimo.ModelArgs.from_dict(
        {
            "model_type": "mimo_v2",
            "vocab_size": 1000,
            "hidden_size": hidden,
            "intermediate_size": 256,
            "moe_intermediate_size": inter,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "v_head_dim": 24,
            "rope_theta": 1000.0,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 32,
            "swa_v_head_dim": 24,
            "swa_rope_theta": 1000.0,
            "sliding_window_size": 32,
            "add_full_attention_sink_bias": False,
            "add_swa_attention_sink_bias": True,
            "hybrid_layer_pattern": [0, 1],
            "moe_layer_freq": [0, 1],
            "n_routed_experts": experts,
            "num_experts_per_tok": top_k,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "topk_method": "noaux_tc",
            "partial_rotary_factor": 0.5,
            "attention_bias": False,
            "layernorm_epsilon": 1e-5,
            "max_position_embeddings": 1000,
            "attention_value_scale": 0.707,
        }
    )
    moe = mimo.MoE(cfg)
    mx.random.seed(0)
    moe.gate.weight = mx.random.normal(moe.gate.weight.shape) * 0.1
    moe.gate.e_score_correction_bias = mx.zeros_like(moe.gate.e_score_correction_bias)
    nn.quantize(moe.switch_mlp, group_size=64, bits=4)
    x = mx.random.normal((1, T, hidden)).astype(mx.bfloat16)
    mx.eval(moe.parameters(), x)
    return moe, x


def _reference(moe, x):
    inds, scores = moe.gate(x)
    y = moe.switch_mlp(x, inds)
    if y.ndim == x.ndim + 1:
        y = (y * scores[..., None]).sum(axis=-2)
    return y.astype(x.dtype)


@pytest.mark.parametrize("T", [4, 96])
def test_mimo_moe_fused_combine_matches_reference(T):
    mimo = _mimo_module()
    if mimo._FusedSwitchGLU is None:
        pytest.skip("oMLX GLM MoE kernels unavailable")
    moe, x = _small_moe(mimo, T)
    assert moe._fused_combine
    out = moe(x)
    ref = _reference(moe, x)
    mx.eval(out, ref)
    assert out.shape == ref.shape == x.shape
    assert out.dtype == x.dtype
    assert mx.allclose(out.astype(mx.float32), ref.astype(mx.float32), atol=2e-2, rtol=2e-2).item()


def test_mimo_moe_unsupported_top_k_uses_plain_switch_glu():
    mimo = _mimo_module()
    moe, x = _small_moe(mimo, 96, top_k=2)
    assert not moe._fused_combine
    out = moe(x)
    ref = _reference(moe, x)
    mx.eval(out, ref)
    assert mx.allclose(out.astype(mx.float32), ref.astype(mx.float32), atol=2e-2, rtol=2e-2).item()


def test_mimo_moe_falls_back_without_fused_switch_glu(monkeypatch):
    mimo = _mimo_module()
    monkeypatch.setattr(mimo, "_FusedSwitchGLU", None)
    moe, x = _small_moe(mimo, 96)
    assert not moe._fused_combine
    out = moe(x)
    mx.eval(out)
    assert out.shape == x.shape
