from __future__ import annotations

import copy

import pytest

from omlx.patches.qwen4_exp.config import (
    ModelArgs,
    Qwen4ExpConfigError,
    TextModelArgs,
)
from omlx.utils.model_loading import _is_mtp_compatible


def _official_text_config() -> dict:
    return {
        "model_type": "qwen4_exp_text",
        "vocab_size": 248320,
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "linear_num_value_heads": 48,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "output_gate_type": "sigmoid",
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "hc_count": 4,
        "hc_lowrank": 320,
        "ple_layer_ids": [2],
        "ple_embed_dim": 2560,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 20000000,
        "split_ngram_parts": 128,
        "make_ngram_vocab_size_divisible_by": 128,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "partial_rotary_factor": 0.25,
        "full_attention_interval": 4,
        "layer_types": [
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(48)
        ],
        "rope_parameters": {
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
        },
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "mtp": {
            "hybrid": True,
            "layer_types": ["full_attention"],
            "num_hidden_layers": 1,
            "rope_theta": 10000000,
        },
        "eos_token_id": 248044,
    }


def test_official_config_is_native_qwen4_exp_and_depth_one_mtp():
    text = TextModelArgs.from_dict(_official_text_config())
    outer = ModelArgs.from_dict(
        {
            "model_type": "qwen4_exp",
            "text_config": _official_text_config(),
            "vision_config": {"model_type": "qwen4_exp"},
            "qwen4_exp_artifact": {
                "layout": "qwen4-exp-fused-gate-up-q8-v3"
            },
        }
    )

    assert outer.model_type == "qwen4_exp"
    assert outer.qwen4_exp_artifact == {
        "layout": "qwen4-exp-fused-gate-up-q8-v3"
    }
    assert text.qsa_layer_indices == tuple(range(3, 48, 4))
    assert text.mtp_num_hidden_layers == 1
    assert _is_mtp_compatible({"text_config": _official_text_config()}, "qwen4_exp")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "qwen3_5_moe"),
        ("output_gate_type", "silu"),
        ("linear_num_value_heads", 32),
        ("hc_count", 2),
        ("split_ngram_parts", 64),
        ("mtp_num_hidden_layers", 5),
    ],
)
def test_nearby_architectures_fail_closed(field, value):
    config = _official_text_config()
    config[field] = value

    with pytest.raises(Qwen4ExpConfigError, match=field):
        TextModelArgs.from_dict(config)


def test_qwen3_alias_and_flattened_outer_configs_are_rejected():
    with pytest.raises(Qwen4ExpConfigError, match="qwen4_exp"):
        ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": {}})
    with pytest.raises(Qwen4ExpConfigError, match="nested text_config"):
        ModelArgs.from_dict({"model_type": "qwen4_exp", "hidden_size": 2560})


def test_mtp_must_remain_official_hybrid_depth_one():
    config = _official_text_config()
    config["mtp"] = copy.deepcopy(config["mtp"])
    config["mtp"]["layer_types"] = ["linear_attention"]

    with pytest.raises(Qwen4ExpConfigError, match="depth-1 hybrid"):
        TextModelArgs.from_dict(config)
