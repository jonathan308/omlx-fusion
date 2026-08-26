# SPDX-License-Identifier: Apache-2.0
"""Strict configuration contract for Qwen3.8 Flash Next."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any


class Qwen4ExpConfigError(ValueError):
    """The checkpoint does not match Fusion's native Qwen4-Exp contract."""


def _known_kwargs(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


def _official_layer_types() -> list[str]:
    return [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(48)
    ]


@dataclass
class TextModelArgs:
    model_type: str = "qwen4_exp_text"
    vocab_size: int = 248_320
    hidden_size: int = 2_560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    max_position_embeddings: int = 262_144
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False

    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    mamba_ssm_dtype: str = "float32"
    output_gate_type: str = "sigmoid"

    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    # Artifact-only execution layout.  Official BF16 checkpoints leave this
    # false and use the published packed tensor sanitizer.  Fusion v3 compute
    # artifacts set it after the outer artifact marker has been validated so
    # gate+up remain one packed projection at runtime.
    fused_moe_gate_up: bool = False

    hc_count: int = 4
    hc_lowrank: int = 320

    ple_layer_ids: list[int] = field(default_factory=lambda: [2])
    ple_embed_dim: int = 2_560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    split_ngram_parts: int = 128
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    eos_token_id: int | list[int] | None = 248_044

    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2_048
    indexer_compress_ratio: int = 4
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    rope_scaling: dict[str, Any] | None = None
    full_attention_interval: int = 4
    layer_types: list[str] = field(default_factory=_official_layer_types)
    rope_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000.0,
        }
    )

    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    mtp: dict[str, Any] = field(
        default_factory=lambda: {
            "hybrid": True,
            "layer_types": ["full_attention"],
            "num_hidden_layers": 1,
            "rope_theta": 10_000_000,
        }
    )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TextModelArgs:
        instance = cls(**_known_kwargs(cls, values))
        instance.validate_architecture()
        return instance

    def validate_architecture(self) -> None:
        exact = {
            "model_type": "qwen4_exp_text",
            "vocab_size": 248_320,
            "hidden_size": 2_560,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
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
            "ple_embed_dim": 2_560,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20_000_000,
            "split_ngram_parts": 128,
            "make_ngram_vocab_size_divisible_by": 128,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2_048,
            "indexer_compress_ratio": 4,
            "partial_rotary_factor": 0.25,
            "full_attention_interval": 4,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        }
        mismatches = [
            f"{name}={getattr(self, name)!r} (expected {expected!r})"
            for name, expected in exact.items()
            if getattr(self, name) != expected
        ]
        if self.layer_types != _official_layer_types():
            mismatches.append("layer_types must be linear,linear,linear,full x12")
        if self.attention_bias:
            mismatches.append("attention_bias must be false")
        if self.hidden_act != "silu":
            mismatches.append("hidden_act must be 'silu'")
        rope_factor = self.rope_parameters.get(
            "partial_rotary_factor", self.partial_rotary_factor
        )
        if float(rope_factor) != 0.25:
            mismatches.append("rope_parameters.partial_rotary_factor must be 0.25")
        if float(self.rope_parameters.get("rope_theta", self.rope_theta)) != float(
            self.rope_theta
        ):
            mismatches.append("rope_parameters.rope_theta must match rope_theta")
        mtp = self.mtp or {}
        if not (
            mtp.get("hybrid") is True
            and mtp.get("layer_types") == ["full_attention"]
            and int(mtp.get("num_hidden_layers", 0) or 0) == 1
        ):
            mismatches.append("mtp must be the official depth-1 hybrid QSA+MoE head")
        if mismatches:
            raise Qwen4ExpConfigError(
                "Unsupported Qwen3.8 Flash Next architecture: " + "; ".join(mismatches)
            )

    @property
    def qsa_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, kind in enumerate(self.layer_types)
            if kind == "full_attention"
        )


@dataclass
class ModelArgs:
    model_type: str
    text_config: dict[str, Any]
    vision_config: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    quantization: dict[str, Any] | None = None
    quantization_config: dict[str, Any] | None = None
    qwen4_exp_artifact: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ModelArgs:
        if values.get("model_type") != "qwen4_exp":
            raise Qwen4ExpConfigError(
                f"Expected model_type='qwen4_exp', got {values.get('model_type')!r}"
            )
        text_config = values.get("text_config")
        if not isinstance(text_config, Mapping):
            raise Qwen4ExpConfigError("qwen4_exp requires a nested text_config")
        # Validate before MLX creates any model arrays.
        TextModelArgs.from_dict(text_config)
        return cls(
            model_type="qwen4_exp",
            text_config=dict(text_config),
            vision_config=(
                dict(values["vision_config"])
                if isinstance(values.get("vision_config"), Mapping)
                else None
            ),
            tie_word_embeddings=bool(values.get("tie_word_embeddings", False)),
            quantization=(
                dict(values["quantization"])
                if isinstance(values.get("quantization"), Mapping)
                else None
            ),
            quantization_config=(
                dict(values["quantization_config"])
                if isinstance(values.get("quantization_config"), Mapping)
                else None
            ),
            qwen4_exp_artifact=(
                dict(values["qwen4_exp_artifact"])
                if isinstance(values.get("qwen4_exp_artifact"), Mapping)
                else None
            ),
        )


__all__ = ["ModelArgs", "Qwen4ExpConfigError", "TextModelArgs"]
