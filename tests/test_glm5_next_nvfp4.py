from __future__ import annotations

import json
import struct
from pathlib import Path

import ml_dtypes
import mlx.core as mx
import numpy as np
import pytest
from safetensors import safe_open

from omlx.patches.glm5_next.convert import INDEX_NAME
from omlx.patches.glm5_next.convert_nvfp4 import (
    MANIFEST_NAME,
    Glm5NextNVFP4ConversionError,
    build_nvfp4_conversion_plan,
    convert_glm53_flash_nvfp4,
)
from omlx.patches.glm5_next.moe import validate_moe_weight_layout
from omlx.patches.glm5_next.nvfp4 import (
    GLM5_NEXT_NVFP4_RUNTIME_READY,
    NVFP4_LAYOUT,
    Glm5NextNVFP4Error,
    NVFP4Tensor,
    dequantize_modelopt_nvfp4,
    is_glm5_next_nvfp4_config,
    make_scaled_nvfp4_switch_linear,
    quantize_modelopt_nvfp4,
)


class _Shape:
    def __init__(self, *shape: int):
        self.shape = shape


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, (dtype, array) in sorted(tensors.items()):
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _fp8_code(value: float) -> int:
    return int(np.asarray(value, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8))


def _tiny_source(root: Path, *, bad_expert_width: bool = False) -> None:
    root.mkdir()
    width = 24 if bad_expert_width else 32
    one = _fp8_code(1.0)
    tensors: dict[str, tuple[str, np.ndarray]] = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for expert in range(288):
            base = (
                f"model.language_model.layers.3.mlp.experts.{expert}."
                f"{projection}.weight"
            )
            tensors[base] = (
                "F8_E4M3",
                np.full((2, width), one, dtype=np.uint8),
            )
            tensors[base + "_scale_inv"] = (
                "F32",
                np.full((1, 1), 1.0 + expert / 32.0, dtype=np.float32),
            )

    # Eligible dense and shared MLP weights.
    tensors["model.language_model.layers.0.mlp.gate_proj.weight"] = (
        "F32",
        np.linspace(-2, 2, 64, dtype=np.float32).reshape(2, 32),
    )
    tensors["model.language_model.layers.3.mlp.shared_experts.gate_proj.weight"] = (
        "F32",
        np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32),
    )

    # Precision-preserved text, router, MTP, and multimodal families.
    tensors["model.language_model.layers.0.self_attn.in_proj_qkv.weight"] = (
        "F32",
        np.ones((2, 32), dtype=np.float32),
    )
    tensors["model.language_model.layers.0.hc_attn_base"] = (
        "F32",
        np.ones((2,), dtype=np.float32),
    )
    tensors["model.language_model.layers.3.mlp.gate.weight"] = (
        "F32",
        np.ones((288, 32), dtype=np.float32),
    )
    tensors["model.language_model.layers.3.mlp.gate.e_score_correction_bias"] = (
        "F32",
        np.zeros((288,), dtype=np.float32),
    )
    tensors["model.language_model.layers.45.eh_proj.weight"] = (
        "F32",
        np.arange(64, dtype=np.float32).reshape(2, 32),
    )
    mtp_expert = "model.language_model.layers.45.mlp.experts.0.gate_proj.weight"
    tensors[mtp_expert] = (
        "F8_E4M3",
        np.full((2, 32), one, dtype=np.uint8),
    )
    tensors[mtp_expert + "_scale_inv"] = (
        "F32",
        np.ones((1, 1), dtype=np.float32),
    )
    for layer in range(24):
        tensors[f"model.visual.blocks.{layer}.norm1.weight"] = (
            "F32",
            np.full((2,), layer + 1, dtype=np.float32),
        )
    tensors["model.visual.patch_embed.proj.weight"] = (
        "F32",
        np.arange(64, dtype=np.float32).reshape(2, 32),
    )

    shard = "model-00001-of-00001.safetensors"
    _write_safetensors(root / shard, tensors)
    (root / INDEX_NAME).write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(array.nbytes for _, array in tensors.values())
                },
                "weight_map": {name: shard for name in tensors},
            }
        )
    )
    (root / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
    (root / "tokenizer_config.json").write_text('{"tokenizer_class":"synthetic"}')
    (root / "processor_config.json").write_text('{"processor_class":"synthetic"}')


def _independent_reference(value: np.ndarray) -> NVFP4Tensor:
    """Small, deliberately independent brute-force ModelOpt reference."""

    value = np.asarray(value, dtype=np.float32)
    global_scale = np.asarray(np.max(np.abs(value)) / (6.0 * 448.0), dtype=np.float32)
    blocks = value.reshape(*value.shape[:-1], -1, 16)
    amax = np.max(np.abs(blocks), axis=-1)
    raw_scales = np.where(amax == 0, 1.0, amax / (6.0 * global_scale))
    scales = np.clip(raw_scales, 2**-9, 448).astype(ml_dtypes.float8_e4m3fn)
    decoded_scales = scales.astype(np.float32)
    normalized = blocks / (decoded_scales[..., None] * global_scale)

    positive = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    magnitude = np.abs(normalized)[..., None]
    error = np.abs(magnitude - positive)
    # np.argmin picks the lower/even code at ties except 0.75, 1.75, 3.5;
    # select the even upper code at those three midpoints.
    codes = np.argmin(error, axis=-1).astype(np.uint8)
    for boundary, upper in ((0.75, 2), (1.75, 4), (3.5, 6)):
        codes = np.where(magnitude[..., 0] == boundary, upper, codes)
    codes |= (normalized < 0).astype(np.uint8) << np.uint8(3)
    codes = codes.reshape(value.shape)
    packed_u8 = (codes[..., 1::2] << np.uint8(4)) | codes[..., 0::2]
    return NVFP4Tensor(
        np.ascontiguousarray(packed_u8).view(np.uint32),
        scales.view(np.uint8),
        global_scale,
        tuple(value.shape),
    )


def test_modelopt_nvfp4_matches_independent_numeric_reference():
    rng = np.random.default_rng(723)
    value = rng.normal(size=(5, 64)).astype(np.float32) * 2.75
    # Exercise all midpoint/ties-to-even cases explicitly.
    value[0, :7] = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    actual = quantize_modelopt_nvfp4(value)
    expected = _independent_reference(value)
    np.testing.assert_array_equal(actual.weight, expected.weight)
    np.testing.assert_array_equal(actual.scales, expected.scales)
    np.testing.assert_array_equal(actual.global_scale, expected.global_scale)
    np.testing.assert_array_equal(
        dequantize_modelopt_nvfp4(actual),
        dequantize_modelopt_nvfp4(expected),
    )


def test_moe_validator_accepts_exact_nvfp4_runtime_tree():
    prefix = "model.language_model.layers.3.mlp"
    weights = {
        f"{prefix}.gate.weight": _Shape(288, 4096),
        f"{prefix}.gate.e_score_correction_bias": _Shape(288),
    }
    for projection, shape in (
        ("gate_proj", (2048, 4096)),
        ("up_proj", (2048, 4096)),
        ("down_proj", (4096, 2048)),
    ):
        rows, columns = shape
        routed = f"{prefix}.switch_mlp.{projection}"
        weights[f"{routed}.weight"] = _Shape(288, rows, columns // 8)
        weights[f"{routed}.scales"] = _Shape(288, rows, columns // 16)
        weights[f"{routed}.global_scale"] = _Shape(288)
        shared = f"{prefix}.shared_experts.{projection}"
        weights[f"{shared}.weight"] = _Shape(rows, columns // 8)
        weights[f"{shared}.scales"] = _Shape(rows, columns // 16)
        weights[f"{shared}.global_scale"] = _Shape()

    validate_moe_weight_layout(weights, prefix)

    routed_only = dict(weights)
    for projection, shape in (
        ("gate_proj", (2048, 4096)),
        ("up_proj", (2048, 4096)),
        ("down_proj", (4096, 2048)),
    ):
        shared = f"{prefix}.shared_experts.{projection}"
        routed_only[f"{shared}.weight"] = _Shape(*shape)
        routed_only.pop(f"{shared}.scales")
        routed_only.pop(f"{shared}.global_scale")
    validate_moe_weight_layout(routed_only, prefix)

    del weights[f"{prefix}.shared_experts.down_proj.global_scale"]
    with pytest.raises(ValueError, match="missing tensor"):
        validate_moe_weight_layout(weights, prefix)


def test_routed_only_modelopt_config_is_recognized_strictly():
    config = {
        "bits": 4,
        "group_size": 16,
        "mode": "nvfp4",
        "layout": NVFP4_LAYOUT,
        "modelopt_global_scale": True,
        "scope": "glm5_next_routed_experts",
        "source_layout": "modelopt-0.45-per-expert",
    }
    assert is_glm5_next_nvfp4_config(config)
    config.pop("source_layout")
    assert not is_glm5_next_nvfp4_config(config)


def test_converter_emits_exact_names_scales_and_preserves_vision_mtp(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "nvfp4"
    _tiny_source(source)
    plan = build_nvfp4_conversion_plan(
        source, output, validate_official=False, source_revision="synthetic"
    )
    result = convert_glm53_flash_nvfp4(
        source, output, validate_official=False, source_revision="synthetic"
    )
    index = json.loads((output / INDEX_NAME).read_text())
    names = set(index["weight_map"])
    expert = "model.language_model.layers.3.mlp.switch_mlp.gate_proj"
    dense = "model.language_model.layers.0.mlp.gate_proj"
    shared = "model.language_model.layers.3.mlp.shared_experts.gate_proj"
    for base in (expert, dense, shared):
        assert {
            base + ".weight",
            base + ".scales",
            base + ".global_scale",
        } <= names
        assert base + ".biases" not in names
    filename = index["weight_map"][expert + ".weight"]
    with safe_open(str(output / filename), framework="np") as handle:
        assert handle.get_slice(expert + ".weight").get_shape() == [288, 2, 4]
        assert handle.get_slice(expert + ".scales").get_shape() == [288, 2, 2]
        scales = handle.get_tensor(expert + ".global_scale")
        assert scales.dtype == np.float32
        assert scales.shape == (288,)

    # Sensitive text and full multimodal/MTP trees retain names and precision.
    sensitive = "model.language_model.layers.0.self_attn.in_proj_qkv.weight"
    assert sensitive in names
    assert sensitive.removesuffix(".weight") + ".scales" not in names
    assert "model.language_model.layers.45.eh_proj.weight" in names
    mtp_expert = "model.language_model.layers.45.mlp.experts.0.gate_proj.weight"
    assert mtp_expert in names
    assert mtp_expert + "_scale_inv" in names
    assert ".layers.45.mlp.switch_mlp." not in "\n".join(names)
    assert "model.visual.patch_embed.proj.weight" in names
    assert all(f"model.visual.blocks.{i}.norm1.weight" in names for i in range(24))
    vision_file = index["weight_map"]["model.visual.patch_embed.proj.weight"]
    with safe_open(str(output / vision_file), framework="np") as handle:
        expected = np.arange(64, dtype=np.float32).reshape(2, 32)
        np.testing.assert_array_equal(
            handle.get_tensor("model.visual.patch_embed.proj.weight"), expected
        )

    config = json.loads((output / "config.json").read_text())
    assert config["quantization"] == {
        "bits": 4,
        "group_size": 16,
        "layout": NVFP4_LAYOUT,
        "mode": "nvfp4",
        "modelopt_global_scale": True,
        "scope": "glm5_next_mlp",
    }
    assert result.projected_output_bytes == plan.projected_output_bytes
    assert result.stats.expert_tensors_packed == 288 * 3
    assert result.stats.max_dequantized_tensor_bytes <= 2 * 32 * 4
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["complete"] is True
    assert len(manifest["output_index_sha256"]) == 64


def test_conversion_is_atomic_and_resumable(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "nvfp4"
    _tiny_source(source)
    first = convert_glm53_flash_nvfp4(
        source, output, validate_official=False, source_revision="synthetic"
    )
    before = {
        path.name: path.stat().st_mtime_ns for path in output.glob("*.safetensors")
    }

    def forbidden(_: np.ndarray) -> NVFP4Tensor:
        raise AssertionError("verified conversion units must be resumed")

    second = convert_glm53_flash_nvfp4(
        source,
        output,
        validate_official=False,
        source_revision="synthetic",
        quantizer=forbidden,
    )
    assert second.stats.units_resumed == 4
    assert second.stats.source_tensors_read == 0
    assert before == {
        path.name: path.stat().st_mtime_ns for path in output.glob("*.safetensors")
    }
    assert first.projected_output_bytes == second.projected_output_bytes
    assert not list(output.glob("*.tmp"))


def test_interrupted_expert_bank_resumes_from_atomic_boundary(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "nvfp4"
    _tiny_source(source)
    calls = 0

    def interrupt(value: np.ndarray) -> NVFP4Tensor:
        nonlocal calls
        calls += 1
        if calls == 4:  # two dense tensors, expert 0, then fail on expert 1
            raise RuntimeError("synthetic interruption")
        return quantize_modelopt_nvfp4(value)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        convert_glm53_flash_nvfp4(
            source,
            output,
            validate_official=False,
            source_revision="synthetic",
            quantizer=interrupt,
        )
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert len(manifest["units"]) == 1
    assert not (output / "experts-layer-03-gate_proj-nvfp4.safetensors").exists()
    assert not list(output.glob("*.tmp"))

    resumed = convert_glm53_flash_nvfp4(
        source, output, validate_official=False, source_revision="synthetic"
    )
    assert resumed.stats.units_resumed == 1
    assert resumed.stats.units_completed == 3


def test_native_mlx_switch_execution_consumes_exact_modelopt_carriers():
    assert GLM5_NEXT_NVFP4_RUNTIME_READY is True
    rng = np.random.default_rng(91)
    experts = rng.normal(size=(3, 8, 32)).astype(np.float32)
    quantized = [quantize_modelopt_nvfp4(experts[i]) for i in range(3)]
    layer = make_scaled_nvfp4_switch_linear(32, 8, 3)
    layer.weight = mx.array(np.stack([item.weight for item in quantized]))
    layer.scales = mx.array(np.stack([item.scales for item in quantized]))
    layer.global_scale = mx.array(
        np.stack([item.global_scale for item in quantized]).astype(np.float32)
    )

    inputs = rng.normal(size=(2, 32)).astype(np.float32)
    indices_np = np.array([[0, 2], [1, 0]], dtype=np.uint32)
    inputs_mx = mx.expand_dims(mx.array(inputs), (-2, -3))
    actual = layer(inputs_mx, mx.array(indices_np), sorted_indices=False).squeeze(-2)
    mx.eval(actual)
    dense = np.stack([dequantize_modelopt_nvfp4(item) for item in quantized])
    expected = np.stack(
        [
            np.stack([inputs[token] @ dense[expert].T for expert in indices_np[token]])
            for token in range(2)
        ]
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)


def test_unsupported_nvfp4_shapes_fail_closed_before_output(tmp_path: Path):
    with pytest.raises(Glm5NextNVFP4Error, match="divisible by 16"):
        quantize_modelopt_nvfp4(np.ones((2, 24), dtype=np.float32))

    source, output = tmp_path / "source", tmp_path / "nvfp4"
    _tiny_source(source, bad_expert_width=True)
    with pytest.raises(Glm5NextNVFP4ConversionError, match="aligned paired block-FP8"):
        convert_glm53_flash_nvfp4(
            source, output, validate_official=False, source_revision="synthetic"
        )
    assert not output.exists()
