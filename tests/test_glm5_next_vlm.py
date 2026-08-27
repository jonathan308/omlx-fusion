# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from omlx.model_discovery import detect_model_type
from omlx.patches.glm5_next import (
    GLM5_NEXT_VLM_ADAPTER_READY,
    apply_glm5_next_vlm_patch,
)
from omlx.patches.glm5_next.vision import (
    IMAGE_TOKEN_ID,
    VIDEO_END_TOKEN_ID,
    VIDEO_START_TOKEN_ID,
    make_vision_model_class,
)


def _text_config():
    from omlx.patches.glm5_next.model import DSA_LAYERS, KDA_LAYERS

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
            "deepseek_sparse_attention" if i in DSA_LAYERS else "linear_attention"
            for i in range(45)
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


def _config():
    return {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": _text_config(),
        "vision_config": _vision_config(),
        "tie_word_embeddings": False,
    }


def test_exact_mlx_vlm_namespace_and_loader_resolution_never_alias_glm52():
    old_glm52 = sys.modules.get("mlx_vlm.models.glm_moe_dsa")
    assert apply_glm5_next_vlm_patch() in (True, False)
    package = importlib.import_module("mlx_vlm.models.glm5_next")
    config_module = importlib.import_module("mlx_vlm.models.glm5_next.config")
    language_module = importlib.import_module("mlx_vlm.models.glm5_next.language")
    vision_module = importlib.import_module("mlx_vlm.models.glm5_next.vision")
    outer_module = importlib.import_module("mlx_vlm.models.glm5_next.glm5_next")

    assert package.GLM5_NEXT_NATIVE_VLM is True
    assert config_module.ModelConfig is package.ModelConfig
    assert language_module.LanguageModel is package.LanguageModel
    assert vision_module.VisionModel is package.VisionModel
    assert outer_module.Model is package.Model
    assert sys.modules.get("mlx_vlm.models.glm_moe_dsa") is old_glm52

    import mlx_vlm.utils as vlm_utils

    assert getattr(vlm_utils.load_model, "_glm5_next_mlx_sanitize", False) is True

    from mlx_vlm.utils import get_model_and_args

    resolved, model_type = get_model_and_args(_config())
    assert resolved is package
    assert model_type == "glm5_next"

    from mlx_vlm.prompt_utils import get_message_json

    formatted = get_message_json(
        "glm5_next", "describe", "user", num_images=1, skip_image_token=False
    )
    assert formatted["content"][0]["type"] == "image"
    assert formatted["content"][1]["text"] == "describe"


def test_config_and_discovery_route_to_native_vlm_without_text_fallback(tmp_path):
    from mlx_vlm.models.glm5_next import ModelConfig

    parsed = ModelConfig.from_dict(_config())
    assert parsed.model_type == "glm5_next"
    assert parsed.text_config.model_type == "glm5_next_text"
    assert parsed.vision_config.model_type == "glm5_next_vision"
    assert GLM5_NEXT_VLM_ADAPTER_READY is True

    (tmp_path / "config.json").write_text(json.dumps(_config()))
    assert detect_model_type(tmp_path) == "vlm"

    broken = _config()
    broken["model_type"] = "glm_moe_dsa"
    with pytest.raises(ValueError, match="model_type must be 'glm5_next'"):
        ModelConfig.from_dict(broken)


def test_image_and_video_features_are_injected_end_to_end_with_tiny_arrays():
    mx = pytest.importorskip("mlx.core")
    from mlx_vlm.models.glm5_next import Model

    class FakeTower:
        dtype = mx.float32
        inject_media_features = make_vision_model_class().inject_media_features

        def encode_image(self, _pixels, _grid):
            return (mx.array([[10.0, 11.0, 12.0, 13.0]]),)

        def encode_video(self, _pixels, _grid):
            return (mx.array([[20.0, 21.0, 22.0, 23.0]]),)

    class FakeEmbeddings:
        def __call__(self, input_ids):
            return mx.broadcast_to(
                input_ids[..., None].astype(mx.float32), (*input_ids.shape, 4)
            )

    class FakeLanguageModel:
        def __init__(self):
            self.model = SimpleNamespace(embed_tokens=FakeEmbeddings())
            self.last_inputs_embeds = None

        def __call__(self, _ids, inputs_embeds=None, **_kwargs):
            self.last_inputs_embeds = inputs_embeds
            return mx.sum(inputs_embeds, axis=-1)

    shell = Model.__new__(Model)
    shell.vision_tower = FakeTower()
    shell.language_model = FakeLanguageModel()
    input_ids = mx.array(
        [[7, IMAGE_TOKEN_ID, VIDEO_START_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_END_TOKEN_ID]]
    )
    logits = shell(
        input_ids,
        pixel_values=mx.zeros((4, 1176)),
        image_grid_thw=mx.array([[1, 2, 2]]),
        pixel_values_videos=mx.zeros((4, 1176)),
        video_grid_thw=mx.array([[1, 2, 2]]),
        mm_token_type_ids=mx.array([[0, 1, 0, 2, 0]]),
    )
    embeds = shell.language_model.last_inputs_embeds
    assert embeds.shape == (1, 5, 4)
    assert embeds[0, 1].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert embeds[0, 3].tolist() == [20.0, 21.0, 22.0, 23.0]
    assert logits[0, 1].item() == 46.0
    assert logits[0, 3].item() == 86.0


def test_sanitizer_preserves_converted_vision_tree_and_native_text_mapping():
    from mlx_vlm.models.glm5_next import Model

    shell = Model.__new__(Model)
    shell._converted_affine = True
    text = object()
    vision_weight, vision_scale, vision_bias = object(), object(), object()
    output = shell.sanitize(
        {
            "model.language_model.embed_tokens.weight": text,
            "model.visual.blocks.0.attn.qkv.weight": vision_weight,
            "model.visual.blocks.0.attn.qkv.scales": vision_scale,
            "model.visual.blocks.0.attn.qkv.biases": vision_bias,
        }
    )
    assert output["language_model.model.embed_tokens.weight"] is text
    assert output["vision_tower.blocks.0.attn.qkv.weight"] is vision_weight
    assert output["vision_tower.blocks.0.attn.qkv.scales"] is vision_scale
    assert output["vision_tower.blocks.0.attn.qkv.biases"] is vision_bias
    assert not any(key.startswith("model.visual.") for key in output)


def test_cached_image_features_skip_tower_but_still_inject():
    mx = pytest.importorskip("mlx.core")
    from mlx_vlm.models.glm5_next import Model

    class FailTower:
        inject_media_features = make_vision_model_class().inject_media_features

        def encode_image(self, *_args, **_kwargs):
            raise AssertionError("cached feature path reran the vision tower")

    shell = Model.__new__(Model)
    shell.vision_tower = FailTower()
    shell.language_model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=lambda ids: mx.zeros((*ids.shape, 4), dtype=mx.float32)
        )
    )
    ids = mx.array([[7, IMAGE_TOKEN_ID]])
    result = shell.get_input_embeddings(
        ids,
        mx.zeros((4, 1176)),
        image_grid_thw=mx.array([[1, 2, 2]]),
        cached_image_features=mx.array([[1.0, 2.0, 3.0, 4.0]]),
    )
    assert result.inputs_embeds[0, 1].tolist() == [1.0, 2.0, 3.0, 4.0]


def _nvfp4_quantization():
    """The exact ModelOpt NVFP4 runtime contract of the Fusion checkpoint."""

    return {
        "bits": 4,
        "group_size": 16,
        "mode": "nvfp4",
        "layout": "glm5-next-modelopt-nvfp4-v1",
        "modelopt_global_scale": True,
        "scope": "glm5_next_routed_experts",
        "source_layout": "modelopt-0.45-per-expert",
    }


def _expected_nvfp4_triples():
    expected = set()
    bases = [
        f"language_model.model.layers.{index}.mlp.switch_mlp.{projection}"
        for index in range(3, 45)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    bases += [
        f"language_model.mtp.0.block.mlp.switch_mlp.{projection}"
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    assert len(bases) == 129
    for base in bases:
        expected |= {base + ".weight", base + ".scales", base + ".global_scale"}
    return expected


def _flat_parameter_paths(model):
    from mlx.utils import tree_flatten

    return {path for path, _value in tree_flatten(model.parameters())}


def test_vlm_binds_all_129_exact_nvfp4_global_scale_triples():
    pytest.importorskip("mlx.core")
    from mlx_vlm.models.glm5_next import Model, ModelConfig

    config = ModelConfig.from_dict(
        {**_config(), "quantization": _nvfp4_quantization()}
    )
    model = Model(config)
    assert model._nvfp4 is True
    # NVFP4 is not an affine checkpoint; sanitize must not take affine paths.
    assert model._converted_affine is False

    expected = _expected_nvfp4_triples()
    assert len(expected) == 129 * 3
    missing = expected - _flat_parameter_paths(model)
    assert not missing, f"VLM tree lacks bound NVFP4 tensors: {sorted(missing)[:3]}"

    switch = model.language_model.model.layers[3].mlp.switch_mlp.gate_proj
    assert type(switch).__name__ == "ScaledNVFP4SwitchLinear"
    assert switch.weight.shape == (288, 2048, 512)
    assert switch.scales.shape == (288, 2048, 256)
    assert switch.global_scale.shape == (288,)
    mtp_switch = model.language_model.mtp[0].block.mlp.switch_mlp.down_proj
    assert type(mtp_switch).__name__ == "ScaledNVFP4SwitchLinear"
    assert mtp_switch.weight.shape == (288, 4096, 256)
    # Dense MLPs and shared experts stay ordinary under routed-only scope.
    dense = model.language_model.model.layers[0].mlp.gate_proj
    assert type(dense).__name__ != "ScaledNVFP4Linear"
    shared = model.language_model.model.layers[3].mlp.shared_experts.gate_proj
    assert type(shared).__name__ != "ScaledNVFP4Linear"
    # Routed-only NVFP4 stores shared experts as dense BF16 weight-only
    # tensors; no block-FP8 sidecar may remain in the bound tree.
    assert "weight_scale_inv" not in shared
    mtp_shared = model.language_model.mtp[0].block.mlp.shared_experts.up_proj
    assert "weight_scale_inv" not in mtp_shared


def test_text_and_vlm_outer_models_share_one_nvfp4_binding_invariant():
    pytest.importorskip("mlx.core")
    from omlx.patches.glm5_next.model import Model as TextOuter, ModelArgs
    from mlx_vlm.models.glm5_next import Model as VlmOuter, ModelConfig

    quantization = _nvfp4_quantization()
    text = TextOuter(
        ModelArgs.from_dict({**_config(), "quantization": quantization})
    )
    vlm = VlmOuter(ModelConfig.from_dict({**_config(), "quantization": quantization}))
    assert text._nvfp4 is True and vlm._nvfp4 is True
    assert text._converted_affine is False and vlm._converted_affine is False

    for index in range(45):
        text_mlp = text.language_model.model.layers[index].mlp
        vlm_mlp = vlm.language_model.model.layers[index].mlp
        assert type(text_mlp) is type(vlm_mlp)
        if index < 3:
            continue
        for projection in ("gate_proj", "up_proj", "down_proj"):
            text_switch = getattr(text_mlp.switch_mlp, projection)
            vlm_switch = getattr(vlm_mlp.switch_mlp, projection)
            assert type(text_switch) is type(vlm_switch)
            assert type(vlm_switch).__name__ == "ScaledNVFP4SwitchLinear"
    text_mtp = text.language_model.mtp[0].block.mlp.switch_mlp.gate_proj
    vlm_mtp = vlm.language_model.mtp[0].block.mlp.switch_mlp.gate_proj
    assert type(text_mtp) is type(vlm_mtp)


def test_vlm_nvfp4_binding_fails_closed_on_partial_bind(monkeypatch):
    pytest.importorskip("mlx.core")
    from omlx.patches.glm5_next import nvfp4
    from mlx_vlm.models.glm5_next import Model, ModelConfig

    config = ModelConfig.from_dict(
        {**_config(), "quantization": _nvfp4_quantization()}
    )

    def partial_bind(_model, _quantization):
        return nvfp4.NVFP4AdapterResult(True, 128)

    monkeypatch.setattr(nvfp4, "configure_glm5_next_nvfp4", partial_bind)
    with pytest.raises(nvfp4.Glm5NextNVFP4Error, match="128/129"):
        Model(config)
