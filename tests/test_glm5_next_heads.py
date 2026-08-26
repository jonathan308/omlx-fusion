from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import pytest

from omlx.patches.glm5_next.moe import (
    OFFICIAL_EXPERTS,
    sanitize_moe_weights,
    validate_moe_config,
    validate_moe_weight_layout,
)
from omlx.patches.glm5_next.mtp import (
    partition_mtp_cache,
    sanitize_mtp_weights,
    validate_mtp_config,
    validate_mtp_weight_layout,
)
from omlx.patches.glm5_next.vision import (
    Glm5NextVisionUnsupportedError,
    MediaKind,
    _required_vision_shapes,
    classify_media_inputs,
    reject_unsupported_media,
    sanitize_vision_weights,
    validate_vision_config,
    validate_vision_weight_layout,
)


@dataclass(frozen=True)
class _Tensor:
    shape: tuple[int, ...]
    label: str = ""


def _stack(values):
    return _Tensor((len(values), *_tensor_shape(values[0])))


def _tensor_shape(value):
    return tuple(value.shape)


def _text_config():
    return {
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
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
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "num_hidden_layers": 45,
        "num_nextn_predict_layers": 1,
        "index_share_for_mtp_iteration": True,
    }


def _moe_weights(prefix="model.language_model.layers.3.mlp"):
    weights = {
        f"{prefix}.gate.weight": _Tensor((288, 4096)),
        f"{prefix}.gate.e_score_correction_bias": _Tensor((288,)),
    }
    shapes = {
        "gate_proj": ((2048, 4096), (16, 32)),
        "up_proj": ((2048, 4096), (16, 32)),
        "down_proj": ((4096, 2048), (32, 16)),
    }
    for expert in range(OFFICIAL_EXPERTS):
        for projection, (shape, scale_shape) in shapes.items():
            key = f"{prefix}.experts.{expert}.{projection}.weight"
            weights[key] = _Tensor(shape, f"expert-{expert}")
            weights[f"{key}_scale_inv"] = _Tensor(scale_shape)
    for projection, (shape, scale_shape) in shapes.items():
        key = f"{prefix}.shared_experts.{projection}.weight"
        weights[key] = _Tensor(shape)
        weights[f"{key}_scale_inv"] = _Tensor(scale_shape)
    return weights


def _mtp_weights():
    prefix = "model.language_model.layers.45."
    shapes = {
        "eh_proj.weight": (4096, 8192),
        "enorm.weight": (4096,),
        "hnorm.weight": (4096,),
        "shared_head.norm.weight": (4096,),
        "input_layernorm.weight": (4096,),
        "post_attention_layernorm.weight": (4096,),
        "self_attn.q_a_layernorm.weight": (1536,),
        "self_attn.kv_a_layernorm.weight": (512,),
        "self_attn.indexer.wk.weight": (128, 4096),
        "self_attn.indexer.weights_proj.weight": (32, 4096),
        "self_attn.indexer.wq_b.weight": (4096, 1536),
        "self_attn.indexer.k_norm.weight": (128,),
        "self_attn.indexer.k_norm.bias": (128,),
        "self_attn.indexer.index_kpool_compress_ape": (4, 128),
        "self_attn.indexer.index_kpool_compress_gate": (128, 4096),
    }
    return {prefix + key: _Tensor(shape) for key, shape in shapes.items()}


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


def test_head_modules_do_not_import_mlx_or_allocate_at_import_time():
    code = """
import sys
import omlx.patches.glm5_next.moe
import omlx.patches.glm5_next.mtp
import omlx.patches.glm5_next.vision
assert 'mlx' not in sys.modules
assert 'mlx.core' not in sys.modules
assert 'mlx_lm' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_exact_moe_config_and_fp32_router_are_required():
    validate_moe_config({"text_config": _text_config()})
    bad = _text_config()
    bad["moe_router_dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="moe_router_dtype"):
        validate_moe_config(bad)


def test_official_fp8_experts_pack_with_scale_metadata_and_preserve_mtp():
    prefix = "model.language_model.layers.3.mlp"
    weights = _moe_weights(prefix)
    marker = _Tensor((4096, 8192), "mtp")
    weights["model.language_model.layers.45.eh_proj.weight"] = marker
    packed = sanitize_moe_weights(weights, stack_fn=_stack)
    validate_moe_weight_layout(packed, prefix)
    assert packed[f"{prefix}.switch_mlp.gate_proj.weight"].shape == (288, 2048, 4096)
    assert packed[f"{prefix}.switch_mlp.gate_proj.weight_scale_inv"].shape == (
        288,
        16,
        32,
    )
    assert packed[f"{prefix}.switch_mlp.down_proj.weight_scale_inv"].shape == (
        288,
        32,
        16,
    )
    assert packed["model.language_model.layers.45.eh_proj.weight"] is marker
    assert not any(f"{prefix}.experts." in key for key in packed)


def test_fp8_expert_sidecars_are_mandatory():
    weights = _moe_weights()
    weights.pop(
        "model.language_model.layers.3.mlp.experts.287.up_proj.weight_scale_inv"
    )
    with pytest.raises(ValueError, match="missing checkpoint tensor"):
        sanitize_moe_weights(weights, stack_fn=_stack)


def test_converted_affine_q4_switchglu_triples_are_preserved():
    prefix = "model.language_model.layers.3.mlp"
    weights = {
        f"{prefix}.gate.weight": _Tensor((288, 4096)),
        f"{prefix}.gate.e_score_correction_bias": _Tensor((288,)),
    }
    projections = {
        "gate_proj": ((288, 2048, 512), (288, 2048, 64)),
        "up_proj": ((288, 2048, 512), (288, 2048, 64)),
        "down_proj": ((288, 4096, 256), (288, 4096, 32)),
    }
    for projection, (weight_shape, sidecar_shape) in projections.items():
        base = f"{prefix}.switch_mlp.{projection}"
        weights[f"{base}.weight"] = _Tensor(weight_shape)
        weights[f"{base}.scales"] = _Tensor(sidecar_shape)
        weights[f"{base}.biases"] = _Tensor(sidecar_shape)
        shared = f"{prefix}.shared_experts.{projection}"
        weights[f"{shared}.weight"] = _Tensor(weight_shape[1:])
        weights[f"{shared}.scales"] = _Tensor(sidecar_shape[1:])
        weights[f"{shared}.biases"] = _Tensor(sidecar_shape[1:])
    sanitized = sanitize_moe_weights(weights, stack_fn=_stack)
    validate_moe_weight_layout(sanitized, prefix)
    assert sanitized == weights


def test_layer_45_mtp_is_preserved_at_exact_depth_and_remapped():
    config = _text_config()
    validate_mtp_config(config)
    weights = _mtp_weights()
    weights["model.language_model.layers.0.input_layernorm.weight"] = _Tensor((4096,))
    validate_mtp_weight_layout(weights)
    out = sanitize_mtp_weights(weights)
    assert "model.language_model.layers.45.eh_proj.weight" not in out
    assert out["mtp.0.eh_proj.weight"].shape == (4096, 8192)
    assert out["mtp.0.norm.weight"].shape == (4096,)
    assert out["mtp.0.block.self_attn.indexer.wk.weight"].shape == (128, 4096)
    assert "model.language_model.layers.0.input_layernorm.weight" in out


def test_converted_affine_mtp_tree_is_idempotent():
    weights = {
        "mtp.0.eh_proj.weight": _Tensor((4096, 1024)),
        "mtp.0.eh_proj.scales": _Tensor((4096, 128)),
        "mtp.0.eh_proj.biases": _Tensor((4096, 128)),
        "mtp.0.enorm.weight": _Tensor((4096,)),
        "mtp.0.hnorm.weight": _Tensor((4096,)),
        "mtp.0.norm.weight": _Tensor((4096,)),
        "mtp.0.block.self_attn.indexer.wk.weight": _Tensor((128, 4096)),
        "mtp.0.block.self_attn.indexer.weights_proj.weight": _Tensor((32, 4096)),
    }
    assert sanitize_mtp_weights(weights) == weights


def test_layer_45_cache_boundary_is_exactly_latent_plus_indexer():
    latent, indexer = object(), object()
    assert partition_mtp_cache([latent, indexer]) == ((latent, indexer),)
    with pytest.raises(ValueError, match="depth must remain exactly 1"):
        partition_mtp_cache([latent, indexer], depth=2)
    with pytest.raises(ValueError, match="exactly two slots"):
        partition_mtp_cache([latent])


def test_vision_config_and_all_24_blocks_form_one_strict_boundary():
    validate_vision_config({"vision_config": _vision_config()})
    weights = {
        key: _Tensor(shape)
        for key, shape in _required_vision_shapes("model.visual.").items()
    }
    weights["model.language_model.norm.weight"] = _Tensor((4096,))
    validate_vision_weight_layout(weights)
    out = sanitize_vision_weights(weights)
    assert out.keys() == weights.keys()
    assert out["model.visual.blocks.23.attn.qkv.weight"].shape == (3072, 1024)


def test_vision_weights_never_silently_drop_a_partial_tower():
    weights = {
        key: _Tensor(shape)
        for key, shape in _required_vision_shapes("model.visual.").items()
    }
    weights.pop("model.visual.blocks.23.norm2.weight")
    with pytest.raises(ValueError, match="missing checkpoint tensor"):
        sanitize_vision_weights(weights)


def test_media_is_classified_and_rejected_fail_closed():
    assert classify_media_inputs({"input_ids": object()}) is MediaKind.TEXT
    assert classify_media_inputs(pixel_values=object()) is MediaKind.IMAGE
    assert classify_media_inputs(pixel_values_videos=object()) is MediaKind.VIDEO
    assert (
        classify_media_inputs(pixel_values=object(), video_grid_thw=object())
        is MediaKind.IMAGE_AND_VIDEO
    )
    reject_unsupported_media({"input_ids": object()})
    with pytest.raises(
        Glm5NextVisionUnsupportedError, match="image input was detected"
    ):
        reject_unsupported_media(pixel_values=object())
