from __future__ import annotations

import json
import struct
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from safetensors import safe_open

from omlx.patches.glm5_next.convert import (
    INDEX_NAME,
    MANIFEST_NAME,
    Glm5NextConversionError,
    convert_glm53_flash,
    dequantize_fp8_blocks,
)


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


def _fake_affine(array: np.ndarray, group_size: int, bits: int, dtype: str):
    assert dtype == "F32"
    assert array.shape[-1] % group_size == 0
    q_shape = (*array.shape[:-1], array.shape[-1] * bits // 32)
    group_shape = (*array.shape[:-1], array.shape[-1] // group_size)
    return (
        np.zeros(q_shape, dtype=np.uint32),
        np.ones(group_shape, dtype=np.float32),
        np.zeros(group_shape, dtype=np.float32),
    )


def _fp8_code(value: float) -> int:
    return int(np.asarray(value, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8))


def _tiny_source(root: Path) -> None:
    root.mkdir()
    tensors: dict[str, tuple[str, np.ndarray]] = {}
    one = _fp8_code(1.0)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for expert in range(288):
            base = (
                f"model.language_model.layers.3.mlp.experts.{expert}."
                f"{projection}.weight"
            )
            tensors[base] = ("F8_E4M3", np.full((2, 32), one, dtype=np.uint8))
            tensors[base + "_scale_inv"] = (
                "F32",
                np.full((1, 1), expert + 1, dtype=np.float32),
            )
    tensors["model.language_model.layers.3.mlp.gate.weight"] = (
        "F32",
        np.ones((288, 32), dtype=np.float32),
    )
    tensors["model.language_model.layers.3.mlp.gate.e_score_correction_bias"] = (
        "F32",
        np.zeros((288,), dtype=np.float32),
    )
    tensors["model.language_model.layers.0.self_attn.A_log"] = (
        "F32",
        np.ones((1,), dtype=np.float32),
    )
    tensors["model.language_model.layers.0.hc_attn_base"] = (
        "F32",
        np.ones((1,), dtype=np.float32),
    )
    tensors["model.language_model.layers.45.eh_proj.weight"] = (
        "F32",
        np.ones((2, 32), dtype=np.float32),
    )
    for layer in range(24):
        tensors[f"model.visual.blocks.{layer}.norm1.weight"] = (
            "F32",
            np.ones((1,), dtype=np.float32),
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


def _index(root: Path) -> dict:
    return json.loads((root / INDEX_NAME).read_text())


def test_official_fp8_inverse_scale_grid_is_numerically_exact():
    values = np.array(
        [[1, 2, 3, 4], [0.5, -1, -2, 6], [8, 10, 12, 14], [1, 1, 1, 1]],
        dtype=ml_dtypes.float8_e4m3fn,
    )
    codes = values.view(np.uint8)
    inverse = np.array([[1, 2], [3, 4]], dtype=np.float32)
    actual = dequantize_fp8_blocks(codes, inverse, block_size=2)
    expected = values.astype(np.float32) * np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("bits", "packed_columns"), [(8, 8), (4, 4)])
def test_q8_q4_pack_all_experts_and_preserve_multimodal_mtp(
    tmp_path: Path, bits: int, packed_columns: int
):
    source = tmp_path / "source"
    output = tmp_path / f"q{bits}"
    _tiny_source(source)
    result = convert_glm53_flash(
        source,
        output,
        bits=bits,
        group_size=32,
        quantizer=_fake_affine,
        validate_official=False,
        source_revision="synthetic",
    )
    names = set(_index(output)["weight_map"])
    for projection in ("gate_proj", "up_proj", "down_proj"):
        weight = f"model.language_model.layers.3.mlp.switch_mlp.{projection}.weight"
        assert {
            weight,
            weight.removesuffix(".weight") + ".scales",
            weight.removesuffix(".weight") + ".biases",
        } <= names
        filename = _index(output)["weight_map"][weight]
        with safe_open(str(output / filename), framework="np") as handle:
            assert handle.get_slice(weight).get_shape() == [288, 2, packed_columns]
    assert "model.language_model.layers.3.mlp.gate.weight" in names
    with safe_open(
        str(
            output
            / _index(output)["weight_map"][
                "model.language_model.layers.3.mlp.gate.weight"
            ]
        ),
        framework="np",
    ) as handle:
        assert (
            str(
                handle.get_tensor("model.language_model.layers.3.mlp.gate.weight").dtype
            )
            == "float32"
        )
    assert "mtp.0.eh_proj.weight" in names
    assert all(
        f"model.visual.blocks.{layer}.norm1.weight" in names for layer in range(24)
    )
    assert (output / "tokenizer_config.json").is_file()
    assert (output / "processor_config.json").is_file()
    assert result.stats.max_dequantized_tensor_bytes == 2 * 32 * 4
    assert result.stats.expert_tensors_packed == 288 * 3
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["complete"] is True
    assert len(manifest["output_index_sha256"]) == 64


def test_atomic_resume_verifies_sha_and_does_not_requantize(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    _tiny_source(source)
    first = convert_glm53_flash(
        source,
        output,
        group_size=32,
        quantizer=_fake_affine,
        validate_official=False,
        source_revision="synthetic",
    )
    before = {
        path.name: path.stat().st_mtime_ns for path in output.glob("*.safetensors")
    }

    def forbidden(*args):
        raise AssertionError("verified units must be resumed")

    second = convert_glm53_flash(
        source,
        output,
        group_size=32,
        quantizer=forbidden,
        validate_official=False,
        source_revision="synthetic",
    )
    assert second.stats.units_resumed == 4
    assert second.stats.source_tensors_read == 0
    assert before == {
        path.name: path.stat().st_mtime_ns for path in output.glob("*.safetensors")
    }
    assert first.projected_output_bytes == second.projected_output_bytes
    assert not list(output.glob("*.tmp"))


def test_interrupted_expert_pack_resumes_at_atomic_unit_boundary(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    _tiny_source(source)
    calls = 0

    def interrupt(array, group_size, bits, dtype):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return _fake_affine(array, group_size, bits, dtype)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        convert_glm53_flash(
            source,
            output,
            group_size=32,
            quantizer=interrupt,
            validate_official=False,
            source_revision="synthetic",
        )
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert len(manifest["units"]) == 1
    assert not list(output.glob("*.tmp"))

    resumed = convert_glm53_flash(
        source,
        output,
        group_size=32,
        quantizer=_fake_affine,
        validate_official=False,
        source_revision="synthetic",
    )
    assert resumed.stats.units_resumed == 1
    assert resumed.stats.units_completed == 3


def test_rejects_any_unmapped_tensor_before_writing_payload(tmp_path: Path):
    source = tmp_path / "source"
    _tiny_source(source)
    index = json.loads((source / INDEX_NAME).read_text())
    shard = next(iter(index["weight_map"].values()))
    # An index/header mismatch is also fail-closed and precedes any output payload.
    index["weight_map"]["alien.weight"] = shard
    (source / INDEX_NAME).write_text(json.dumps(index))
    with pytest.raises(Glm5NextConversionError, match="index/header"):
        convert_glm53_flash(
            source,
            tmp_path / "output",
            group_size=32,
            quantizer=_fake_affine,
            validate_official=False,
        )
