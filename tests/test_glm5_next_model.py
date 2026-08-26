from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from omlx.model_discovery import detect_model_type
from omlx.patches.glm5_next import apply_glm5_next_patch
from omlx.patches.glm5_next.kda import KDAConfig
from omlx.patches.glm5_next.mhc import MHCConfig
from omlx.patches.glm5_next.model import (
    DSA_LAYERS,
    KDA_LAYERS,
    MAIN_LAYER_COUNT,
    DecoderLayer,
    Glm5NextDsaCache,
    Glm5NextKDACache,
    Glm5NextMTPBlock,
    Model,
    ModelArgs,
    TextModel,
    TextModelArgs,
    layer_kind,
    make_layer_cache,
    native_vision_ready,
    require_runtime_ready,
    runtime_gaps,
    sanitize_weight_name,
)
from omlx.patches.glm5_next.mtp import make_mtp_block_class, make_mtp_cache
from omlx.patches.glm5_next.vision import Glm5NextVisionUnsupportedError


def _text_config():
    return {
        "model_type": "glm5_next_text",
        "vocab_size": 154_880,
        "hidden_size": 4_096,
        "intermediate_size": 12_288,
        "num_hidden_layers": 45,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "first_k_dense_replace": 3,
        "moe_intermediate_size": 2_048,
        "n_routed_experts": 288,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "moe_router_dtype": "float32",
        "n_group": 1,
        "topk_group": 1,
        "mhc": True,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        "mla_use_nope": True,
        "q_lora_rank": 1_536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2_048,
        "index_kpool": 4,
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "index_share_for_mtp_iteration": True,
        "num_nextn_predict_layers": 1,
        "layer_types": [
            "deepseek_sparse_attention" if index in DSA_LAYERS else "linear_attention"
            for index in range(45)
        ],
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
        "linear_attn_config": {
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": list(KDA_LAYERS),
            "full_attn_layers": list(DSA_LAYERS),
        },
    }


def _vision_config():
    return {
        "model_type": "glm5_next_vision",
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_heads": 16,
        "image_size": 448,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 4096,
        "projection_intermediate_size": 10240,
        "in_channels": 3,
        "attention_bias": True,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-5,
    }


def _outer_config():
    return {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": _text_config(),
        "vision_config": _vision_config(),
        "tie_word_embeddings": False,
    }


def test_exact_main_graph_schedule_and_depth_one_mtp_contract():
    args = TextModelArgs.from_dict(_text_config())
    assert MAIN_LAYER_COUNT == 45
    assert len(KDA_LAYERS) == 34
    assert len(DSA_LAYERS) == 11
    assert tuple(range(3, 45, 4)) == DSA_LAYERS
    assert [layer_kind(index) for index in range(45)].count("kda") == 34
    assert [layer_kind(index) for index in range(45)].count("dsa") == 11
    assert args.hc_mult == 4
    assert args.n_routed_experts == 288
    assert args.num_experts_per_tok == 8
    assert args.num_nextn_predict_layers == 1


def test_model_layer_class_constructs_only_with_tiny_fake_geometry():
    fake = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        rms_norm_eps=1e-5,
        swiglu_limit=10.0,
        first_k_dense_replace=3,
        kda_config=lambda: KDAConfig(
            hidden_size=8, num_heads=2, head_dim=4, conv_kernel_size=3
        ),
        mhc_config=lambda: MHCConfig(hidden_size=8, streams=2, sinkhorn_iters=2),
    )
    layer = DecoderLayer(fake, 0)
    assert layer.is_linear is True
    assert layer.self_attn.config.hidden_size == 8
    assert layer.hc_attn.config.streams == 2


def test_registration_is_glm5_next_only_and_idempotent():
    old_glm52 = sys.modules.get("mlx_lm.models.glm_moe_dsa")
    apply_glm5_next_patch()
    registered = sys.modules["mlx_lm.models.glm5_next"]
    assert registered.GLM5_NEXT_STRICT_GRAPH is True
    assert registered.GLM5_NEXT_NATIVE_TEXT_READY is True
    assert registered.__name__ == "mlx_lm.models.glm5_next"
    assert apply_glm5_next_patch() is False
    assert sys.modules.get("mlx_lm.models.glm_moe_dsa") is old_glm52


def test_multimodal_outer_config_routes_until_vlm_adapter_readiness_signal(
    tmp_path, monkeypatch
):
    (tmp_path / "config.json").write_text(json.dumps(_outer_config()))
    assert detect_model_type(tmp_path) == "vlm"
    args = ModelArgs.from_dict(_outer_config())
    assert args.model_type == "glm5_next"
    assert args.vision_config["model_type"] == "glm5_next_vision"
    assert native_vision_ready() is True

    from omlx.patches import glm5_next
    from omlx.patches.glm5_next import vision

    monkeypatch.setattr(glm5_next, "GLM5_NEXT_VLM_ADAPTER_READY", False)
    assert detect_model_type(tmp_path) == "llm"
    monkeypatch.setattr(vision, "GLM5_NEXT_VISION_RUNTIME_READY", True)
    monkeypatch.setattr(vision, "vision_runtime_gaps", lambda: [])
    monkeypatch.setattr(glm5_next, "GLM5_NEXT_VLM_ADAPTER_READY", True)
    assert native_vision_ready() is True
    assert detect_model_type(tmp_path) == "vlm"


def test_cache_composition_is_34_recurrent_plus_11_two_array_dsa():
    caches = [make_layer_cache(index) for index in range(45)]
    assert len(caches) == 45
    assert sum(isinstance(cache, Glm5NextKDACache) for cache in caches) == 34
    assert sum(isinstance(cache, Glm5NextDsaCache) for cache in caches) == 11
    assert all(caches[index].empty() for index in range(45))
    assert len(Glm5NextDsaCache().state) == 2
    assert len(Glm5NextKDACache().state) == 4
    assert isinstance(make_mtp_cache(), Glm5NextDsaCache)
    assert Glm5NextMTPBlock is make_mtp_block_class()


def test_source_weight_sanitizer_mapping_never_uses_glm52_aliases():
    assert (
        sanitize_weight_name("model.language_model.layers.0.self_attn.q_proj.weight")
        == "language_model.model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        sanitize_weight_name("model.language_model.layers.0.hc_attn_fn")
        == "language_model.model.layers.0.hc_attn.fn"
    )
    assert sanitize_weight_name("model.visual.blocks.0.norm1.weight") is None
    assert sanitize_weight_name("mtp.0.block.self_attn.q_a_proj.weight") == (
        "language_model.mtp.0.block.self_attn.q_a_proj.weight"
    )
    with pytest.raises(ValueError, match="glm_moe_dsa weight aliases are forbidden"):
        sanitize_weight_name("model.layers.0.self_attn.q_proj.weight")


def test_converted_affine_sanitizer_retains_packed_expert_sidecars():
    shell = Model.__new__(Model)
    shell._converted_affine = True
    marker = object()
    prefix = "model.language_model.layers.3.mlp.switch_mlp.gate_proj"
    sanitized = shell.sanitize(
        {
            prefix + ".weight": marker,
            prefix + ".scales": marker,
            prefix + ".biases": marker,
            "model.visual.blocks.0.norm1.weight": marker,
        }
    )
    target = "language_model.model.layers.3.mlp.switch_mlp.gate_proj"
    assert sanitized[target + ".weight"] is marker
    assert sanitized[target + ".scales"] is marker
    assert sanitized[target + ".biases"] is marker
    assert not any("visual" in key for key in sanitized)


def test_media_rejection_happens_before_any_model_execution():
    shell = Model.__new__(Model)
    with pytest.raises(
        Glm5NextVisionUnsupportedError, match="image input was detected"
    ):
        shell(None, pixel_values=object())
    with pytest.raises(
        Glm5NextVisionUnsupportedError, match="video input was detected"
    ):
        shell(None, video_grid_thw=object())


def test_quant_and_cast_predicates_preserve_fp32_state_contracts():
    quant = TextModel.quant_predicate.fget(None)
    cast = TextModel.cast_predicate.fget(None)
    assert quant("model.layers.0.self_attn.A_log", object()) is False
    assert quant("model.layers.0.hc_attn.base", object()) is False
    assert quant("model.layers.3.mlp.gate", object()) is False
    assert quant("model.layers.3.self_attn.indexer.wq_b", object()) is False
    assert quant("model.layers.0.self_attn.q_conv1d", object()) is False
    assert quant("model.layers.3.input_layernorm", object()) is False
    assert quant("model.layers.0.self_attn.q_proj", object()) is True
    assert quant("model.layers.3.mlp.switch_mlp.gate_proj", object()) is True
    assert cast("model.layers.0.self_attn.dt_bias") is False
    assert cast("model.layers.3.self_attn.indexer.wk.weight") is False
    assert cast("model.layers.3.self_attn.q_a_proj.weight") is True


def test_text_runtime_readiness_is_genuine_without_allocating_official_model():
    assert runtime_gaps() == ()
    assert require_runtime_ready() is None


def test_config_variants_fail_closed_without_allocating_model():
    bad = _text_config()
    bad["layer_types"][3] = "linear_attention"
    with pytest.raises(ValueError, match="34 KDA and 11 DSA"):
        TextModelArgs.from_dict(bad)
    bad = _text_config()
    bad["n_routed_experts"] = 256
    with pytest.raises(ValueError, match="n_routed_experts"):
        TextModelArgs.from_dict(bad)
    bad_outer = _outer_config()
    bad_outer["model_type"] = "glm_moe_dsa"
    with pytest.raises(ValueError, match="model_type must be 'glm5_next'"):
        ModelArgs.from_dict(bad_outer)
