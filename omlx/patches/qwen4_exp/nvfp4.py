# SPDX-License-Identifier: Apache-2.0
"""Exact routed-expert loader for Qwen3.8 Flash Next ModelOpt NVFP4.

The validated RadixArk candidate stores each of 512 experts separately with
packed E2M1 weights, E4M3 group-16 scales, and one FP32 ``weight_scale_2``.
MLX already implements the native NVFP4 matmul; this module only repacks the
byte carriers into SwitchLinear banks and applies the retained ModelOpt scale
after the gather-QMM. Non-expert compute remains BF16 and PLE is supplied by
the separate verified SSD artifact.
"""

from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

_LAYOUT: Final = "qwen4-exp-modelopt-nvfp4-v1"
_LAYERS: Final = 48
_EXPERTS: Final = 512
_GROUP_SIZE: Final = 16
_BITS: Final = 4
_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<field>weight|weight_scale|weight_scale_2|input_scale)$"
)


class Qwen4ExpNVFP4Error(ValueError):
    """The candidate does not match the pinned Qwen NVFP4 contract."""


def is_supported_config(config: Any) -> bool:
    if not isinstance(config, dict) or config.get("model_type") != "qwen4_exp":
        return False
    text = config.get("text_config")
    artifact = config.get("qwen4_exp_artifact")
    quantization = config.get("quantization")
    return (
        isinstance(text, dict)
        and text.get("model_type") == "qwen4_exp_text"
        and text.get("num_hidden_layers") == _LAYERS
        and text.get("num_experts") == _EXPERTS
        and isinstance(artifact, dict)
        and artifact.get("layout") == _LAYOUT
        and isinstance(quantization, dict)
        and quantization.get("bits") == _BITS
        and quantization.get("group_size") == _GROUP_SIZE
        and quantization.get("mode") == "nvfp4"
        and quantization.get("scope") == "qwen4_exp_routed_experts"
        and quantization.get("modelopt_global_scale") is True
    )


def require_supported_config(config: Any) -> None:
    if not is_supported_config(config):
        raise Qwen4ExpNVFP4Error(
            "unsupported Qwen4-Exp NVFP4 config; exact routed-expert, "
            "group-16, scale_2 layout is required"
        )


@lru_cache(maxsize=1)
def _switch_class():
    import mlx.core as mx
    import mlx.nn as nn

    class ScaledNVFP4SwitchLinear(nn.Module):
        def __init__(self, input_dims: int, output_dims: int, num_experts: int):
            super().__init__()
            if input_dims % _GROUP_SIZE:
                raise Qwen4ExpNVFP4Error("NVFP4 input width is not group aligned")
            self.weight = mx.zeros(
                (num_experts, output_dims, input_dims // 8), dtype=mx.uint32
            )
            self.scales = mx.zeros(
                (num_experts, output_dims, input_dims // _GROUP_SIZE),
                dtype=mx.uint8,
            )
            self.global_scale = mx.ones((num_experts,), dtype=mx.float32)
            self.group_size = _GROUP_SIZE
            self.bits = _BITS
            self.mode = "nvfp4"
            self.freeze()

        @property
        def input_dims(self):
            return self.scales.shape[-1] * _GROUP_SIZE

        @property
        def output_dims(self):
            return self.weight.shape[-2]

        @property
        def num_experts(self):
            return self.weight.shape[0]

        def __call__(self, x, indices, sorted_indices=False):
            output = mx.gather_qmm(
                x,
                self["weight"],
                self["scales"],
                rhs_indices=indices,
                transpose=True,
                group_size=_GROUP_SIZE,
                bits=_BITS,
                mode="nvfp4",
                sorted_indices=sorted_indices,
            )
            selected = self["global_scale"][indices].astype(output.dtype)
            return output * selected[..., None, None]

    return ScaledNVFP4SwitchLinear


def make_scaled_switch(input_dims: int, output_dims: int, num_experts: int):
    return _switch_class()(input_dims, output_dims, num_experts)


def replace_routed_modules(model: Any) -> int:
    layers = model.language_model.model.layers
    if len(layers) != _LAYERS:
        raise Qwen4ExpNVFP4Error("Qwen4-Exp NVFP4 requires exactly 48 layers")
    replaced = 0
    for layer in layers:
        switch = layer.mlp.switch_mlp
        switch.gate_proj = make_scaled_switch(2560, 640, _EXPERTS)
        switch.up_proj = make_scaled_switch(2560, 640, _EXPERTS)
        switch.down_proj = make_scaled_switch(640, 2560, _EXPERTS)
        replaced += 3
    return replaced


def _expected_shape(projection: str, field: str) -> tuple[int, ...]:
    if projection in {"gate_proj", "up_proj"}:
        output_dims, packed_input, groups = 640, 1280, 160
    else:
        output_dims, packed_input, groups = 2560, 320, 40
    if field == "weight":
        return output_dims, packed_input
    if field == "weight_scale":
        return output_dims, groups
    return ()


def transform_weights_exact(weights: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx

    families: dict[tuple[int, str], dict[int, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    output: dict[str, Any] = {}
    for name, value in weights.items():
        match = _EXPERT_RE.match(name)
        if match is None:
            if "ple_embedding.ngram_embedding.shard_" in name:
                raise Qwen4ExpNVFP4Error(
                    "FP8 PLE table must be excluded; use the BF16 SSD artifact"
                )
            output[name] = value
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        projection = match.group("projection")
        field = match.group("field")
        if not 0 <= layer < _LAYERS or not 0 <= expert < _EXPERTS:
            raise Qwen4ExpNVFP4Error(f"expert placement changed: {name}")
        families[(layer, projection)][expert][field] = value

    expected_families = {
        (layer, projection)
        for layer in range(_LAYERS)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    if set(families) != expected_families:
        missing = sorted(expected_families - set(families))
        raise Qwen4ExpNVFP4Error(f"missing routed expert families: {missing[:3]}")

    for (layer, projection), experts in sorted(families.items()):
        if set(experts) != set(range(_EXPERTS)):
            raise Qwen4ExpNVFP4Error(
                f"layer {layer} {projection} does not contain 512 experts"
            )
        fields = ("weight", "weight_scale", "weight_scale_2", "input_scale")
        for expert, tensors in experts.items():
            if set(tensors) != set(fields):
                raise Qwen4ExpNVFP4Error(
                    f"incomplete layer {layer} expert {expert} {projection}"
                )
            for field in fields:
                value = tensors[field]
                if tuple(value.shape) != _expected_shape(projection, field):
                    raise Qwen4ExpNVFP4Error(
                        f"invalid {projection}.{field} shape: {value.shape}"
                    )
            if experts[expert]["weight"].dtype != mx.uint8:
                raise Qwen4ExpNVFP4Error("NVFP4 packed weights must be U8")
            if experts[expert]["weight_scale"].dtype != mx.uint8:
                raise Qwen4ExpNVFP4Error("NVFP4 group scales must be F8/U8")
            if experts[expert]["weight_scale_2"].dtype != mx.float32:
                raise Qwen4ExpNVFP4Error("NVFP4 scale_2 must be FP32")

        prefix = f"model.language_model.layers.{layer}.mlp.switch_mlp.{projection}"
        packed = mx.stack([experts[index]["weight"] for index in range(_EXPERTS)])
        output[f"{prefix}.weight"] = packed.view(mx.uint32)
        output[f"{prefix}.scales"] = mx.stack(
            [experts[index]["weight_scale"] for index in range(_EXPERTS)]
        )
        output[f"{prefix}.global_scale"] = mx.stack(
            [experts[index]["weight_scale_2"] for index in range(_EXPERTS)]
        ).reshape(_EXPERTS)
    return output


def load(model_name: str | Path):
    """Load a filtered RadixArk artifact through native mlx-lm classes."""

    import mlx.core as mx
    import mlx_lm.utils as lm_utils

    model_path = Path(model_name)
    config = json.loads((model_path / "config.json").read_text())
    require_supported_config(config)
    weights = {}
    for filename in sorted(glob.glob(str(model_path / "*.safetensors"))):
        weights.update(mx.load(filename))
    model_class, args_class = lm_utils._get_classes(config=config)
    model = model_class(args_class.from_dict(config))
    count = replace_routed_modules(model)
    if count != _LAYERS * 3:
        raise Qwen4ExpNVFP4Error("routed module replacement count changed")
    weights = transform_weights_exact(weights)
    weights = model.sanitize(weights)
    model.eval()
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    tokenizer = lm_utils.load_tokenizer(
        model_path, eos_token_ids=config.get("eos_token_id")
    )
    return model, tokenizer


__all__ = [
    "Qwen4ExpNVFP4Error",
    "is_supported_config",
    "load",
    "make_scaled_switch",
    "replace_routed_modules",
    "require_supported_config",
    "transform_weights_exact",
]
