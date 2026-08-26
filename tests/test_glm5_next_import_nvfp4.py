from __future__ import annotations

import json
import struct
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from safetensors import safe_open

import omlx.patches.glm5_next.import_nvfp4 as importer
from omlx.patches.glm5_next.convert import INDEX_NAME
from omlx.patches.glm5_next.import_nvfp4 import (
    MANIFEST_NAME,
    Glm5NextNVFP4ImportError,
    build_import_plan,
    import_glm53_nvfp4,
)
from omlx.patches.glm5_next.nvfp4 import NVFP4_LAYOUT


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, (dtype, array) in sorted(tensors.items()):
        raw = np.asarray(array).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _quant_config() -> dict:
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "dynamic": False,
                    "symmetric": True,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
        "quant_algo": "NVFP4",
        "kv_cache_scheme": None,
        "producer": {"name": "modelopt", "version": "0.45.0"},
        "quant_method": "modelopt",
        "ignore": list(importer._IGNORE),
    }


def _packed_codes(offset: int) -> np.ndarray:
    codes = (np.arange(64, dtype=np.uint8).reshape(2, 32) + offset) & 0x0F
    return (codes[..., 1::2] << np.uint8(4)) | codes[..., 0::2]


def _tiny_source(root: Path, *, bad_scale: bool = False) -> None:
    root.mkdir()
    tensors: dict[str, tuple[str, np.ndarray]] = {}
    for expert in range(3):
        for projection_index, projection in enumerate(
            ("gate_proj", "up_proj", "down_proj")
        ):
            base = f"model.language_model.layers.3.mlp.experts.{expert}.{projection}"
            tensors[base + ".weight"] = (
                "U8",
                _packed_codes(expert + projection_index),
            )
            scale_shape = (2, 1) if bad_scale and expert == 2 else (2, 2)
            scale = np.full(
                scale_shape,
                1.0 + expert / 4.0,
                dtype=ml_dtypes.float8_e4m3fn,
            ).view(np.uint8)
            tensors[base + ".weight_scale"] = ("F8_E4M3", scale)
            tensors[base + ".weight_scale_2"] = (
                "F32",
                np.asarray(0.25 + expert / 8.0, dtype=np.float32),
            )

    tensors["model.language_model.layers.3.mlp.gate.weight"] = (
        "F32",
        np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    tensors["model.language_model.layers.45.eh_proj.weight"] = (
        "BF16",
        np.arange(8, dtype=np.float32)
        .astype(ml_dtypes.bfloat16)
        .view(np.uint16)
        .reshape(2, 4),
    )
    tensors["model.visual.blocks.0.norm1.weight"] = (
        "BF16",
        np.asarray([1.0, 2.0], dtype=ml_dtypes.bfloat16).view(np.uint16),
    )

    shard = "model-00001-of-00001.safetensors"
    _write_safetensors(root / shard, tensors)
    total = sum(array.nbytes for _, array in tensors.values())
    (root / INDEX_NAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": total},
                "weight_map": {name: shard for name in tensors},
            }
        )
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "architectures": ["Glm5NextForConditionalGeneration"],
                "tie_word_embeddings": False,
                "quantization_config": _quant_config(),
            }
        )
    )
    (root / "README.md").write_text(
        "---\nlicense: mit\nbase_model: zai-org/GLM-5.3-Flash\n"
        "base_model_relation: quantized\nquantized_by: LibertAIDAI\n---\n"
    )
    (root / "tokenizer_config.json").write_text("{}")


def _decode_source(
    weight: np.ndarray, scales: np.ndarray, scale_2: float
) -> np.ndarray:
    codes = np.empty((*weight.shape[:-1], weight.shape[-1] * 2), dtype=np.uint8)
    codes[..., 0::2] = weight & np.uint8(0x0F)
    codes[..., 1::2] = weight >> np.uint8(4)
    magnitudes = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    values = magnitudes[codes & 7]
    values = np.where((codes & 8) != 0, -values, values)
    groups = values.reshape(*values.shape[:-1], -1, 16)
    decoded_scales = scales.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    return (groups * decoded_scales[..., None] * np.float32(scale_2)).reshape(
        values.shape
    )


def _decode_output(weight: np.ndarray, scales: np.ndarray, global_scale: float):
    packed = np.ascontiguousarray(weight).view(np.uint8)
    return _decode_source(packed, scales, global_scale)


def test_lossless_import_names_projection_and_config(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    _tiny_source(source)
    plan = build_import_plan(source, output, validate_pinned=False)
    result = import_glm53_nvfp4(source, output, validate_pinned=False)
    index = json.loads((output / INDEX_NAME).read_text())
    names = set(index["weight_map"])
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.language_model.layers.3.mlp.switch_mlp.{projection}"
        assert {
            base + ".weight",
            base + ".scales",
            base + ".global_scale",
        } <= names
        assert not any(
            f"model.language_model.layers.3.mlp.experts.{expert}.{projection}" in name
            for expert in range(3)
            for name in names
        )
    assert "model.language_model.layers.3.mlp.gate.weight" in names
    assert "model.language_model.layers.45.eh_proj.weight" in names
    assert "model.visual.blocks.0.norm1.weight" in names
    assert plan.source_payload_bytes == plan.projected_output_bytes
    assert result.projected_output_bytes == plan.projected_output_bytes
    assert result.stats.expert_matrices_repacked == 9
    assert result.stats.max_copy_chunk_bytes <= 8 * 1024 * 1024

    config = json.loads((output / "config.json").read_text())
    assert config["quantization"] == {
        "bits": 4,
        "group_size": 16,
        "mode": "nvfp4",
        "layout": NVFP4_LAYOUT,
        "modelopt_global_scale": True,
        "scope": "glm5_next_routed_experts",
        "source_layout": "modelopt-0.45-per-expert",
    }
    assert config["glm5_next_artifact"]["lossless_repack"] is True
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["complete"] is True
    assert len(manifest["output_index_sha256"]) == 64


def test_packed_rows_decode_identically_and_nonexperts_are_byte_exact(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    _tiny_source(source)
    import_glm53_nvfp4(source, output, validate_pinned=False)
    index = json.loads((output / INDEX_NAME).read_text())["weight_map"]
    base = "model.language_model.layers.3.mlp.switch_mlp.gate_proj"
    with safe_open(str(output / index[base + ".weight"]), framework="np") as handle:
        weight = handle.get_tensor(base + ".weight")
        scales = handle.get_tensor(base + ".scales")
        globals_ = handle.get_tensor(base + ".global_scale")
    assert weight.dtype == np.uint32 and weight.shape == (3, 2, 4)
    assert scales.dtype == np.uint8 and scales.shape == (3, 2, 2)
    assert globals_.dtype == np.float32 and globals_.shape == (3,)

    source_weight = _packed_codes(0)
    source_scales = np.full((2, 2), 1.0, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)
    np.testing.assert_array_equal(weight[0].view(np.uint8), source_weight)
    np.testing.assert_array_equal(scales[0], source_scales)
    np.testing.assert_array_equal(
        _decode_output(weight[0], scales[0], globals_[0]),
        _decode_source(source_weight, source_scales, 0.25),
    )

    # BF16 payload bits, not merely decoded values, survive the normal path.
    key = "model.language_model.layers.45.eh_proj.weight"
    with safe_open(str(output / index[key]), framework="np") as handle:
        output_bits = handle.get_tensor(key).view(np.uint16)
    expected_bits = (
        np.arange(8, dtype=np.float32)
        .astype(ml_dtypes.bfloat16)
        .view(np.uint16)
        .reshape(2, 4)
    )
    np.testing.assert_array_equal(output_bits, expected_bits)


def test_import_is_checksum_resumable(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    _tiny_source(source)
    first = import_glm53_nvfp4(source, output, validate_pinned=False)
    before = {
        path.name: path.stat().st_mtime_ns for path in output.glob("*.safetensors")
    }

    def forbidden(_unit: str, _expert: int) -> None:
        raise AssertionError("verified units must resume without expert reads")

    second = import_glm53_nvfp4(
        source, output, validate_pinned=False, unit_hook=forbidden
    )
    assert second.stats.units_resumed == 4
    assert second.stats.tensors_copied == 0
    assert first.projected_output_bytes == second.projected_output_bytes
    assert before == {
        path.name: path.stat().st_mtime_ns for path in output.glob("*.safetensors")
    }


def test_interrupted_bank_is_atomic_and_resumes_normal_unit(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    _tiny_source(source)

    def interrupt(unit: str, expert: int) -> None:
        if unit.endswith("down_proj") and expert == 1:
            raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        import_glm53_nvfp4(source, output, validate_pinned=False, unit_hook=interrupt)
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert len(manifest["units"]) == 1
    assert not (output / "experts-layer-03-down_proj-nvfp4.safetensors").exists()
    assert not list(output.glob("*.tmp"))

    resumed = import_glm53_nvfp4(source, output, validate_pinned=False)
    assert resumed.stats.units_resumed == 1
    assert resumed.stats.units_completed == 3


def test_pins_repository_revision_producer_base_and_ignore(tmp_path: Path):
    source = tmp_path / "source"
    _tiny_source(source)
    importer._validate_readme(source)
    with pytest.raises(Glm5NextNVFP4ImportError, match="repository must be pinned"):
        build_import_plan(
            source,
            tmp_path / "output",
            repository="someone/else",
            validate_pinned=False,
        )
    with pytest.raises(Glm5NextNVFP4ImportError, match="revision must be pinned"):
        build_import_plan(
            source,
            tmp_path / "output",
            revision="main",
            validate_pinned=False,
        )

    config = json.loads((source / "config.json").read_text())
    config["quantization_config"]["producer"]["version"] = "0.46.0"
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(Glm5NextNVFP4ImportError, match="producer changed"):
        build_import_plan(source, tmp_path / "output", validate_pinned=False)

    config["quantization_config"] = _quant_config()
    config["quantization_config"]["ignore"] = list(importer._IGNORE[:-1])
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(Glm5NextNVFP4ImportError, match="ignore changed"):
        build_import_plan(source, tmp_path / "output", validate_pinned=False)

    (source / "README.md").write_text(
        "---\nlicense: mit\nbase_model: someone/else\n"
        "base_model_relation: quantized\nquantized_by: LibertAIDAI\n---\n"
    )
    with pytest.raises(Glm5NextNVFP4ImportError, match="base/provenance changed"):
        importer._validate_readme(source)


def test_invalid_modelopt_carrier_fails_before_destination(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    _tiny_source(source, bad_scale=True)
    with pytest.raises(Glm5NextNVFP4ImportError, match="invalid E4M3 group scale"):
        import_glm53_nvfp4(source, output, validate_pinned=False)
    assert not output.exists()
