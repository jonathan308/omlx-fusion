"""Pinned, fail-closed source contract for zai-org/GLM-5.3-Flash."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping

OFFICIAL_REPOSITORY: Final = "zai-org/GLM-5.3-Flash"
OFFICIAL_REVISION: Final = "84c6a6aa9497188e15a635ba793b0f95a79b1033"
OFFICIAL_ARCHITECTURE: Final = "Glm5NextForConditionalGeneration"
OFFICIAL_SOURCE_SHARDS: Final = 62
OFFICIAL_TENSOR_COUNT: Final = 76_108
OFFICIAL_TENSOR_BYTES: Final = 328_326_771_576
OFFICIAL_TOTAL_PARAMETERS: Final = 320_000_000_000
OFFICIAL_ACTIVE_PARAMETERS: Final = 18_000_000_000

_MAIN_DSA_LAYERS: Final = frozenset(range(3, 45, 4))
_MAIN_LINEAR_LAYERS: Final = frozenset(range(45)) - _MAIN_DSA_LAYERS
_MTP_LAYER: Final = 45
_LAYER_RE: Final = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_EXPERT_RE: Final = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
)


class Glm5NextContractError(ValueError):
    """The source does not match the pinned official checkpoint contract."""


@dataclass(frozen=True, slots=True)
class Glm5NextSourceContract:
    revision: str
    shard_count: int
    tensor_count: int
    tensor_bytes: int
    linear_attention_layers: tuple[int, ...]
    dsa_layers: tuple[int, ...]
    mtp_layer: int
    expert_count: int
    tokenizer_class: str


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except FileNotFoundError as exc:
        raise Glm5NextContractError(f"missing required source file: {path.name}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Glm5NextContractError(f"invalid JSON source file: {path.name}") from exc
    if not isinstance(value, dict):
        raise Glm5NextContractError(f"JSON root must be an object: {path.name}")
    return value


def _exact(container: Mapping[str, Any], field: str, expected: Any, where: str) -> None:
    actual = container.get(field)
    if actual != expected:
        raise Glm5NextContractError(
            f"{where}.{field} changed: expected {expected!r}, found {actual!r}"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate architecture, hybrid schedule, MoE, MTP, vision, and FP8 ABI."""

    _exact(config, "model_type", "glm5_next", "config")
    _exact(config, "architectures", [OFFICIAL_ARCHITECTURE], "config")
    _exact(config, "tie_word_embeddings", False, "config")

    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise Glm5NextContractError("config.text_config must be an object")
    exact_text = {
        "model_type": "glm5_next_text",
        "dtype": "bfloat16",
        "vocab_size": 154_880,
        "hidden_size": 4_096,
        "intermediate_size": 12_288,
        "num_hidden_layers": 45,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "max_position_embeddings": 1_048_576,
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
        "num_nextn_predict_layers": 1,
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
    }
    for field, expected in exact_text.items():
        _exact(text, field, expected, "config.text_config")

    expected_layers = [
        "deepseek_sparse_attention" if i in _MAIN_DSA_LAYERS else "linear_attention"
        for i in range(45)
    ]
    _exact(text, "layer_types", expected_layers, "config.text_config")
    _exact(
        text,
        "mlp_layer_types",
        ["dense"] * 3 + ["sparse"] * 42,
        "config.text_config",
    )

    linear = text.get("linear_attn_config")
    if not isinstance(linear, Mapping):
        raise Glm5NextContractError("config.text_config.linear_attn_config must be an object")
    for field, expected in {
        "num_heads": 64,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": sorted(_MAIN_LINEAR_LAYERS),
        "full_attn_layers": sorted(_MAIN_DSA_LAYERS),
    }.items():
        _exact(linear, field, expected, "config.text_config.linear_attn_config")

    vision = config.get("vision_config")
    if not isinstance(vision, Mapping):
        raise Glm5NextContractError("config.vision_config must be an object")
    for field, expected in {
        "model_type": "glm5_next_vision",
        "depth": 24,
        "hidden_size": 1_024,
        "intermediate_size": 4_096,
        "num_heads": 16,
        "image_size": 448,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 4_096,
    }.items():
        _exact(vision, field, expected, "config.vision_config")

    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping):
        raise Glm5NextContractError("official source requires quantization_config")
    for field, expected in {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }.items():
        _exact(quant, field, expected, "config.quantization_config")
    excluded = quant.get("modules_to_not_convert")
    if not isinstance(excluded, list) or len(excluded) != 1_509:
        raise Glm5NextContractError(
            "config.quantization_config.modules_to_not_convert must contain 1509 entries"
        )


def _safe_shard_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise Glm5NextContractError("weight_map shard names must be non-empty strings")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise Glm5NextContractError(f"unsafe shard path in weight_map: {value!r}")
    return value


def validate_weight_index(index: Mapping[str, Any]) -> None:
    """Validate the exact pinned index and all architecture-defining families."""

    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, Mapping) or not isinstance(weight_map, Mapping):
        raise Glm5NextContractError("index requires metadata and weight_map objects")
    _exact(metadata, "total_size", OFFICIAL_TENSOR_BYTES, "index.metadata")
    if len(weight_map) != OFFICIAL_TENSOR_COUNT:
        raise Glm5NextContractError(
            f"expected {OFFICIAL_TENSOR_COUNT} tensors, found {len(weight_map)}"
        )
    if not all(isinstance(name, str) for name in weight_map):
        raise Glm5NextContractError("weight_map tensor names must be strings")

    shards = {_safe_shard_name(value) for value in weight_map.values()}
    expected_shards = {
        f"model-{ordinal:05d}-of-{OFFICIAL_SOURCE_SHARDS:05d}.safetensors"
        for ordinal in range(1, OFFICIAL_SOURCE_SHARDS + 1)
    }
    if shards != expected_shards:
        raise Glm5NextContractError("source shard set differs from pinned 62-shard layout")

    names = set(weight_map)
    allowed = ("lm_head.", "model.language_model.", "model.visual.")
    unexpected = next((name for name in names if not name.startswith(allowed)), None)
    if unexpected is not None:
        raise Glm5NextContractError(f"unexpected weight namespace: {unexpected}")
    if any(name.startswith("model.layers.") for name in names):
        raise Glm5NextContractError("glm_moe_dsa weight aliases are forbidden")

    layers = {
        int(match.group(1))
        for name in names
        if (match := _LAYER_RE.match(name)) is not None
    }
    if layers != set(range(46)):
        raise Glm5NextContractError("checkpoint must contain main layers 0..44 and MTP layer 45")

    indexer_layers = {
        int(match.group(1))
        for name in names
        if ".self_attn.indexer." in name
        and (match := _LAYER_RE.match(name)) is not None
    }
    if indexer_layers != set(_MAIN_DSA_LAYERS) | {_MTP_LAYER}:
        raise Glm5NextContractError("DSA indexer layer set changed")
    kda_layers = {
        int(match.group(1))
        for name in names
        if name.endswith(".self_attn.A_log")
        and (match := _LAYER_RE.match(name)) is not None
    }
    if kda_layers != set(_MAIN_LINEAR_LAYERS):
        raise Glm5NextContractError("KDA linear-attention layer set changed")

    expert_ids: dict[int, set[int]] = {}
    for name in names:
        match = _EXPERT_RE.match(name)
        if match is not None:
            expert_ids.setdefault(int(match.group(1)), set()).add(int(match.group(2)))
    if set(expert_ids) != set(range(3, 46)):
        raise Glm5NextContractError("MoE layer set changed")
    expected_experts = set(range(288))
    if any(ids != expected_experts for ids in expert_ids.values()):
        raise Glm5NextContractError("each sparse/MTP layer must contain experts 0..287")

    required = {
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.layers.0.hc_attn_base",
        "model.language_model.layers.0.self_attn.q_conv1d.weight",
        "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_gate",
        "model.language_model.layers.45.eh_proj.weight",
        "model.language_model.layers.45.enorm.weight",
        "model.language_model.layers.45.hnorm.weight",
        "model.language_model.layers.45.shared_head.norm.weight",
        "model.visual.patch_embed.proj.weight",
        "model.visual.merger.proj.weight",
    }
    missing = sorted(required - names)
    if missing:
        raise Glm5NextContractError(f"missing required architecture weights: {missing}")

    scales = {name for name in names if name.endswith(".weight_scale_inv")}
    unpaired = sorted(
        name for name in scales if name.removesuffix("_scale_inv") not in names
    )
    if unpaired:
        raise Glm5NextContractError(f"unpaired FP8 scale tensor: {unpaired[0]}")


def validate_source_contract(
    source: Path | str, *, source_revision: str
) -> Glm5NextSourceContract:
    source = Path(source)
    if source_revision != OFFICIAL_REVISION:
        raise Glm5NextContractError(
            f"revision must be pinned to {OFFICIAL_REVISION}, found {source_revision}"
        )
    config = _load_object(source / "config.json")
    index = _load_object(source / "model.safetensors.index.json")
    tokenizer = _load_object(source / "tokenizer_config.json")
    validate_config(config)
    validate_weight_index(index)
    _exact(tokenizer, "tokenizer_class", "TokenizersBackend", "tokenizer_config")
    _exact(tokenizer, "model_max_length", 1_048_576, "tokenizer_config")
    _exact(tokenizer, "padding_side", "left", "tokenizer_config")
    return Glm5NextSourceContract(
        revision=source_revision,
        shard_count=OFFICIAL_SOURCE_SHARDS,
        tensor_count=OFFICIAL_TENSOR_COUNT,
        tensor_bytes=OFFICIAL_TENSOR_BYTES,
        linear_attention_layers=tuple(sorted(_MAIN_LINEAR_LAYERS)),
        dsa_layers=tuple(sorted(_MAIN_DSA_LAYERS)),
        mtp_layer=_MTP_LAYER,
        expert_count=288,
        tokenizer_class="TokenizersBackend",
    )
