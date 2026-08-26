from __future__ import annotations

import json
import struct

import pytest

from omlx.patches.glm5_next.contract import (
    OFFICIAL_REVISION,
    Glm5NextContractError,
    validate_config,
    validate_source_contract,
)
from omlx.patches.glm5_next.convert import iter_tensor_headers


def _official_config() -> dict:
    dsa = list(range(3, 45, 4))
    linear = [i for i in range(45) if i not in dsa]
    return {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "glm5_next_text",
            "dtype": "bfloat16",
            "vocab_size": 154880,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 45,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "max_position_embeddings": 1048576,
            "first_k_dense_replace": 3,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 288,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "moe_router_dtype": "float32",
            "num_nextn_predict_layers": 1,
            "mhc": True,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 20,
            "mla_use_nope": True,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 2048,
            "index_kpool": 4,
            "index_kpool_compress": True,
            "index_kpool_always_select_tail": True,
            "index_share_for_mtp_iteration": True,
            "layer_types": [
                "deepseek_sparse_attention" if i in dsa else "linear_attention"
                for i in range(45)
            ],
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
            "linear_attn_config": {
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "kda_layers": linear,
                "full_attn_layers": dsa,
            },
        },
        "vision_config": {
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
        },
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [f"module.{i}" for i in range(1509)],
        },
    }


def _write_safetensors(path, tensors):
    offset = 0
    header = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        payload.extend(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_official_config_contract_accepts_new_architecture():
    validate_config(_official_config())


def test_old_glm_moe_dsa_alias_is_rejected():
    config = _official_config()
    config["model_type"] = "glm_moe_dsa"
    with pytest.raises(Glm5NextContractError, match="model_type changed"):
        validate_config(config)


def test_hybrid_schedule_and_no_rope_contract_are_strict():
    config = _official_config()
    config["text_config"]["qk_rope_head_dim"] = 64
    with pytest.raises(Glm5NextContractError, match="qk_rope_head_dim changed"):
        validate_config(config)


def test_revision_is_pinned_before_reading_source(tmp_path):
    with pytest.raises(Glm5NextContractError, match="revision must be pinned"):
        validate_source_contract(tmp_path, source_revision="main")
    assert len(OFFICIAL_REVISION) == 40


def test_streaming_header_reader_never_materializes_payload_collection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shard = "model-00001-of-00001.safetensors"
    _write_safetensors(
        source / shard,
        [
            ("model.weight", "F8_E4M3", [4], b"\x00" * 4),
            ("model.weight_scale_inv", "F32", [1], b"\x00" * 4),
        ],
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {
                    "model.weight": shard,
                    "model.weight_scale_inv": shard,
                },
            }
        )
    )
    headers = list(iter_tensor_headers(source))
    assert [(item.name, item.dtype, item.source_bytes) for item in headers] == [
        ("model.weight", "F8_E4M3", 4),
        ("model.weight_scale_inv", "F32", 4),
    ]


def test_streaming_reader_fails_on_index_header_mismatch(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shard = "model-00001-of-00001.safetensors"
    _write_safetensors(source / shard, [("wrong.weight", "F32", [1], b"\x00" * 4)])
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"right.weight": shard}})
    )
    with pytest.raises(Glm5NextContractError, match="index/header mismatch"):
        list(iter_tensor_headers(source))
