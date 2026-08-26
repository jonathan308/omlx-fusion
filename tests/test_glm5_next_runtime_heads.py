from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.glm5_next.dsa import Glm5NextDsaConfig
from omlx.patches.glm5_next.moe import (
    GLM5_NEXT_BLOCK_FP8_RUNTIME_READY,
    make_sparse_moe_class,
)
from omlx.patches.glm5_next.mtp import (
    GLM5_NEXT_MTP_RUNTIME_READY,
    make_mtp_block_class,
    mtp_partial_rollback,
    sanitize_mtp_weights,
)


def _tiny_config():
    return SimpleNamespace(
        hidden_size=32,
        rms_norm_eps=1e-5,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        swiglu_limit=10.0,
    )


def _tiny_dsa_config():
    return Glm5NextDsaConfig(
        hidden_size=32,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        index_kpool=2,
    )


def _flat_parameters(module):
    return dict(tree_flatten(module.parameters()))


def _copy_router(source, target):
    target.gate.weight = source.gate.weight
    target.gate.e_score_correction_bias = source.gate.e_score_correction_bias


def _install_fp8_reference_pair(fp8_model, dense_model):
    for group_name in ("switch_mlp", "shared_experts"):
        fp8_group = getattr(fp8_model, group_name)
        dense_group = getattr(dense_model, group_name)
        for projection in ("gate_proj", "up_proj", "down_proj"):
            fp8_layer = getattr(fp8_group, projection)
            dense_layer = getattr(dense_group, projection)
            codes = mx.to_fp8(fp8_layer.weight.astype(mx.float32))
            if group_name == "switch_mlp":
                scales = (
                    mx.arange(codes.shape[0], dtype=mx.float32)[:, None, None] * 0.125
                    + 0.75
                )
            else:
                scales = mx.array([[1.25]], dtype=mx.float32)
            fp8_layer.weight = codes
            fp8_layer.weight_scale_inv = scales
            decoded = mx.from_fp8(codes, dtype=mx.float32) * scales
            dense_layer.weight = decoded


def test_fp32_sigmoid_top8_equation_on_tiny_top2_geometry():
    model = make_sparse_moe_class(validate_official=False)(_tiny_config())
    router_weight = mx.arange(4 * 32).reshape(4, 32).astype(mx.float16) / 500
    correction = mx.array([0.2, -0.1, 0.05, 0.0], dtype=mx.float32)
    model.gate.weight = router_weight
    model.gate.e_score_correction_bias = correction
    hidden = mx.arange(64).reshape(1, 2, 32).astype(mx.float16) / 64

    indices, weights = model.gate(hidden)
    mx.eval(indices, weights)
    logits = (
        np.asarray(hidden, dtype=np.float32)
        @ np.asarray(router_weight, dtype=np.float32).T
    )
    scores = 1.0 / (1.0 + np.exp(-logits))
    selected = np.argpartition(-(scores + np.asarray(correction)), kth=1, axis=-1)[
        ..., :2
    ]
    expected = np.take_along_axis(scores, selected, axis=-1)
    expected = expected / (expected.sum(axis=-1, keepdims=True) + 1e-20) * 2.5
    np.testing.assert_array_equal(np.asarray(indices), selected)
    np.testing.assert_allclose(np.asarray(weights), expected, rtol=1e-6, atol=1e-6)


def test_selected_expert_block_fp8_sidecars_execute_without_dense_bank_expansion():
    moe_class = make_sparse_moe_class(validate_official=False)
    fp8_model = moe_class(_tiny_config())
    dense_reference = moe_class(_tiny_config())
    _copy_router(fp8_model, dense_reference)
    _install_fp8_reference_pair(fp8_model, dense_reference)
    hidden = mx.random.normal((1, 3, 32)).astype(mx.float16)

    actual = fp8_model(hidden)
    expected = dense_reference(hidden)
    mx.eval(actual, expected)
    assert actual.shape == (1, 3, 32)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=4e-3, atol=4e-3
    )


@pytest.mark.parametrize("bits", [4, 8])
def test_converted_affine_switchglu_triples_have_exact_load_names_and_execute(bits):
    model = make_sparse_moe_class(validate_official=False)(_tiny_config())
    nn.quantize(
        model,
        group_size=32,
        bits=bits,
        mode="affine",
        class_predicate=lambda path, module: (
            hasattr(module, "to_quantized")
            and ("switch_mlp" in path or "shared_experts" in path)
        ),
    )
    parameters = _flat_parameters(model)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"switch_mlp.{projection}"
        assert prefix + ".weight" in parameters
        assert prefix + ".scales" in parameters
        assert prefix + ".biases" in parameters
        assert prefix + ".weight_scale_inv" not in parameters
    assert parameters["switch_mlp.gate_proj.weight"].shape == (
        4,
        32,
        bits * 32 // 32,
    )
    output = model(mx.random.normal((1, 2, 32)).astype(mx.float16))
    mx.eval(output)
    assert output.shape == (1, 2, 32)


def test_depth_one_mtp_parameter_tree_executes_dsa_moe_and_two_array_cache():
    mtp_class = make_mtp_block_class(validate_official=False)
    head = mtp_class(_tiny_config(), dsa_config=_tiny_dsa_config())
    parameters = _flat_parameters(head)
    required = {
        "enorm.weight",
        "hnorm.weight",
        "eh_proj.weight",
        "norm.weight",
        "block.input_layernorm.weight",
        "block.self_attn.q_a_proj.weight",
        "block.self_attn.indexer.wk.weight",
        "block.self_attn.indexer.index_kpool_compress_gate",
        "block.post_attention_layernorm.weight",
        "block.mlp.gate.weight",
        "block.mlp.switch_mlp.gate_proj.weight",
    }
    assert required <= set(parameters)

    hidden = mx.random.normal((1, 3, 32)).astype(mx.float16)
    embedding = mx.random.normal((1, 3, 32)).astype(mx.float16)
    mask = mx.ones((1, 3), dtype=mx.bool_)
    cache = head.make_cache()
    output = head(hidden, embedding, mask, cache)
    mx.eval(output, *cache.state)
    assert output.shape == (1, 3, 32)
    assert len(cache.state) == 2
    assert cache.state[0].shape == (1, 3, 4)
    assert cache.state[1].shape == (1, 3, 9)

    assert mtp_partial_rollback(cache, accepted=1, num_drafts=3) is True
    assert cache.offset == 1
    assert cache.state[0].shape[1] == cache.state[1].shape[1] == 1


@dataclass(frozen=True)
class _Shape:
    shape: tuple[int, ...]


def test_mtp_sanitize_preserves_converted_affine_load_tree_names():
    prefix = "model.language_model.layers.45."
    weights = {
        prefix + "eh_proj.weight": _Shape((4096, 8192)),
        prefix + "enorm.weight": _Shape((4096,)),
        prefix + "hnorm.weight": _Shape((4096,)),
        prefix + "shared_head.norm.weight": _Shape((4096,)),
        prefix + "input_layernorm.weight": _Shape((4096,)),
        prefix + "post_attention_layernorm.weight": _Shape((4096,)),
        prefix + "self_attn.q_a_layernorm.weight": _Shape((1536,)),
        prefix + "self_attn.kv_a_layernorm.weight": _Shape((512,)),
        prefix + "self_attn.indexer.wk.weight": _Shape((128, 4096)),
        prefix + "self_attn.indexer.weights_proj.weight": _Shape((32, 4096)),
        prefix + "self_attn.indexer.wq_b.weight": _Shape((4096, 1536)),
        prefix + "self_attn.indexer.k_norm.weight": _Shape((128,)),
        prefix + "self_attn.indexer.k_norm.bias": _Shape((128,)),
        prefix + "self_attn.indexer.index_kpool_compress_ape": _Shape((4, 128)),
        prefix + "self_attn.indexer.index_kpool_compress_gate": _Shape((128, 4096)),
    }
    for suffix in ("weight", "scales", "biases"):
        weights[prefix + f"mlp.switch_mlp.gate_proj.{suffix}"] = _Shape((1,))
    output = sanitize_mtp_weights(weights)
    for suffix in ("weight", "scales", "biases"):
        assert f"mtp.0.block.mlp.switch_mlp.gate_proj.{suffix}" in output


def test_runtime_readiness_flags_are_affirmative_only_with_executable_paths():
    assert GLM5_NEXT_BLOCK_FP8_RUNTIME_READY is True
    assert GLM5_NEXT_MTP_RUNTIME_READY is True
