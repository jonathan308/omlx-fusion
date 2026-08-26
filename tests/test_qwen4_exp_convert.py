from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from omlx.patches.qwen4_exp.convert import (
    COMPUTE_DIRNAME,
    INDEX_NAME,
    MANIFEST_NAME,
    Qwen4ExpConversionError,
    _atomic_safetensors,
    _EncodedTensor,
    _safetensors_header,
    convert_qwen38_flash_next,
)
from omlx.patches.qwen4_exp.ple import PLE_PREFIX, PLE_TABLE_PREFIX


def _fake_q8(array: np.ndarray, group_size: int, bits: int, source_dtype: str):
    assert bits == 8
    assert source_dtype in {"BF16", "F32"}
    assert array.shape[-1] % group_size == 0
    packed_shape = (*array.shape[:-1], array.shape[-1] // 4)
    group_shape = (*array.shape[:-1], array.shape[-1] // group_size)
    scale_dtype = np.uint16 if source_dtype == "BF16" else np.float32
    return (
        np.zeros(packed_shape, dtype=np.uint32),
        np.ones(group_shape, dtype=scale_dtype),
        np.zeros(group_shape, dtype=scale_dtype),
    )


def _write_source(root: Path, *, bf16: bool = False) -> dict[str, str]:
    root.mkdir()
    config = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {"model_type": "qwen4_exp_text", "hidden_size": 2560},
        "arbitrary_official_field": {"keep": True},
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer_config.json").write_text('{"tokenizer_class":"Qwen2Tokenizer"}')

    auxiliary = {
        f"{PLE_PREFIX}.layer_multipliers": np.arange(3, dtype=np.int64),
        f"{PLE_PREFIX}.ngram_heads_offsets": np.arange(16, dtype=np.int64),
        f"{PLE_PREFIX}.ngram_heads_vocab_sizes": np.arange(16, dtype=np.int64) + 3,
    }
    first = {
        f"{PLE_TABLE_PREFIX}.shard_0.weight": np.arange(640, dtype=np.float32).reshape(
            20, 32
        ),
        "model.language_model.layers.0.linear_attn.in_proj_q.weight": np.ones(
            (3, 32), dtype=np.float32
        ),
        "model.language_model.layers.0.linear_attn.A_log": np.ones(
            (3,), dtype=np.float32
        ),
        "mtp.fc_embedding.weight": np.ones((4, 32), dtype=np.float32),
        **auxiliary,
    }
    second = {
        f"{PLE_TABLE_PREFIX}.shard_1.weight": np.arange(640, dtype=np.float32).reshape(
            20, 32
        ),
        f"{PLE_PREFIX}.conv1d.weight": np.ones((2, 32), dtype=np.float32),
        "model.language_model.layers.0.input_norm.weight": np.ones(
            (32,), dtype=np.float32
        ),
        "model.language_model.layers.0.linear_attn.dt_bias": np.ones(
            (3,), dtype=np.float32
        ),
        "mtp.fc_hidden.weight": np.ones((4, 32), dtype=np.float32),
        "mtp.layers.0.mlp.experts.gate_up_proj": np.ones((2, 4, 32), dtype=np.float32),
        "mtp.layers.0.mlp.experts.down_proj": np.ones((2, 32, 32), dtype=np.float32),
    }
    weight_map: dict[str, str] = {}
    for ordinal, tensors in enumerate((first, second), start=1):
        filename = f"model-{ordinal:05d}-of-00002.safetensors"
        if bf16:
            encoded = {}
            for name, tensor in tensors.items():
                if np.issubdtype(tensor.dtype, np.floating):
                    raw = (tensor.astype(np.float32).view(np.uint32) >> 16).astype(
                        np.uint16
                    )
                    encoded[name] = _EncodedTensor(raw, "BF16")
                else:
                    encoded[name] = _EncodedTensor(tensor, "I64")
            _atomic_safetensors(root / filename, encoded)
        else:
            save_file(tensors, str(root / filename), metadata={"format": "pt"})
        weight_map.update({name: filename for name in tensors})
    (root / INDEX_NAME).write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    return weight_map


def _convert(
    source: Path,
    destination: Path,
    *,
    ple_quantization="bf16",
    ple_source_dtype="F32",
    **kwargs,
):
    return convert_qwen38_flash_next(
        source,
        destination,
        expected_source_shards=2,
        expected_ple_shards=2,
        ple_rows_per_shard=20,
        ple_head_dim=32,
        ple_source_dtype=ple_source_dtype,
        ple_quantization=ple_quantization,
        quantizer=_fake_q8,
        source_revision="synthetic-source-sha",
        **kwargs,
    )


def _index(path: Path) -> dict:
    return json.loads((path / INDEX_NAME).read_text())


def test_streams_split_artifacts_without_ple_leak_and_preserves_mtp(tmp_path):
    source = tmp_path / "source"
    source_weight_map = _write_source(source)
    result = _convert(source, tmp_path / "output")

    compute = _index(result.compute_dir)
    ple = _index(result.ple_dir)
    compute_names = set(compute["weight_map"])
    ple_names = set(ple["weight_map"])

    assert not any(
        name.startswith(f"{PLE_TABLE_PREFIX}.shard_") for name in compute_names
    )
    assert len(ple_names) == 2 + 3
    assert all(
        name.startswith(f"{PLE_TABLE_PREFIX}.shard_")
        or name
        in {
            f"{PLE_PREFIX}.layer_multipliers",
            f"{PLE_PREFIX}.ngram_heads_offsets",
            f"{PLE_PREFIX}.ngram_heads_vocab_sizes",
        }
        for name in ple_names
    )

    source_mtp = {name for name in source_weight_map if name.startswith("mtp.")}
    ordinary_mtp = {
        name
        for name in source_mtp
        if not name.endswith((".experts.gate_up_proj", ".experts.down_proj"))
    }
    assert ordinary_mtp <= compute_names
    assert "mtp.fc_embedding.scales" in compute_names
    assert "mtp.fc_embedding.biases" in compute_names
    assert "mtp.layers.0.mlp.experts.gate_up_proj" not in compute_names
    assert "mtp.layers.0.mlp.experts.down_proj" not in compute_names
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"mtp.layers.0.mlp.switch_mlp.{projection}"
        assert {f"{base}.weight", f"{base}.scales", f"{base}.biases"} <= compute_names

    # Numerically sensitive/state tensors and both conv/norm paths remain dense.
    for name in (
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.dt_bias",
        f"{PLE_PREFIX}.conv1d.weight",
        "model.language_model.layers.0.input_norm.weight",
    ):
        assert name in compute_names
        assert name.removesuffix(".weight") + ".scales" not in compute_names

    config = json.loads((result.compute_dir / "config.json").read_text())
    assert config["arbitrary_official_field"] == {"keep": True}
    assert config["quantization"] == {"bits": 8, "group_size": 32, "mode": "affine"}
    assert (result.compute_dir / "tokenizer_config.json").is_file()
    assert (result.ple_dir / "tokenizer_config.json").is_file()
    assert result.stats.max_source_tensor_bytes == 2 * 32 * 32 * 4
    assert result.stats.source_tensors_read == len(source_weight_map) - 2
    assert result.stats.ple_table_tensors_stream_copied == 2
    assert not list(result.root.rglob("*.tmp"))


def test_output_shards_are_valid_and_indexed_atomically(tmp_path):
    source = tmp_path / "source"
    _write_source(source)
    result = _convert(source, tmp_path / "output")

    for artifact in (result.compute_dir, result.ple_dir):
        index = _index(artifact)
        by_file: dict[str, set[str]] = {}
        for name, filename in index["weight_map"].items():
            by_file.setdefault(filename, set()).add(name)
        for filename, expected in by_file.items():
            with safe_open(str(artifact / filename), framework="np") as handle:
                assert set(handle.keys()) == expected
    manifest = json.loads((result.root / MANIFEST_NAME).read_text())
    assert manifest["complete"] is True
    assert manifest["source_revision"] == "synthetic-source-sha"
    assert len(manifest["source_index_sha256"]) == 64
    assert manifest["layout_version"] == "qwen4-exp-split-q8-v2"
    assert manifest["ple_quantization"] == "bf16"


def test_optional_q8_ple_has_exact_three_tensor_layout_and_no_compute_leak(tmp_path):
    source = tmp_path / "source"
    _write_source(source)
    result = _convert(source, tmp_path / "output", ple_quantization="q8")
    compute_names = set(_index(result.compute_dir)["weight_map"])
    ple = _index(result.ple_dir)
    ple_names = set(ple["weight_map"])
    assert result.ple_dir.name == "ple-q8"
    assert not any(
        name.startswith(f"{PLE_TABLE_PREFIX}.shard_") for name in compute_names
    )
    assert len(ple_names) == 2 * 3 + 3
    for shard in range(2):
        base = f"{PLE_TABLE_PREFIX}.shard_{shard}"
        assert {f"{base}.weight", f"{base}.scales", f"{base}.biases"} <= ple_names
    assert ple["metadata"]["qwen4_exp_ple_bits"] == 8
    assert result.stats.ple_table_tensors_stream_copied == 0


def test_bf16_words_convert_without_numpy_bfloat16_dependency(tmp_path):
    source = tmp_path / "source"
    _write_source(source, bf16=True)
    result = _convert(
        source,
        tmp_path / "output",
        ple_quantization="q8",
        ple_source_dtype="BF16",
    )
    ple = _index(result.ple_dir)
    table_scale = f"{PLE_TABLE_PREFIX}.shard_0.scales"
    scale_file = result.ple_dir / ple["weight_map"][table_scale]
    header, _ = _safetensors_header(scale_file)
    assert header[table_scale]["dtype"] == "BF16"
    assert result.stats.max_source_tensor_bytes == 2 * 32 * 32 * 2
    assert ple["metadata"]["qwen4_exp_ple_scale_dtype"] == "BF16"


def test_resume_skips_verified_completed_shards(tmp_path):
    source = tmp_path / "source"
    _write_source(source)
    destination = tmp_path / "output"
    first = _convert(source, destination)
    before = (
        (destination / COMPUTE_DIRNAME / "model-00001-of-00002.safetensors")
        .stat()
        .st_mtime_ns
    )

    calls = 0

    def should_not_run(array, group_size, bits, source_dtype):
        nonlocal calls
        calls += 1
        return _fake_q8(array, group_size, bits, source_dtype)

    second = convert_qwen38_flash_next(
        source,
        destination,
        expected_source_shards=2,
        expected_ple_shards=2,
        ple_rows_per_shard=20,
        ple_head_dim=32,
        ple_source_dtype="F32",
        quantizer=should_not_run,
        source_revision="synthetic-source-sha",
    )
    assert first.root == second.root
    assert calls == 0
    assert second.stats.source_shards_resumed == 2
    assert second.stats.source_tensors_read == 0
    assert (
        before
        == (destination / COMPUTE_DIRNAME / "model-00001-of-00002.safetensors")
        .stat()
        .st_mtime_ns
    )


def test_partial_failure_resumes_at_source_shard_boundary(tmp_path):
    source = tmp_path / "source"
    _write_source(source)
    destination = tmp_path / "output"
    calls = 0

    def fail_on_second_shard(array, group_size, bits, source_dtype):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("synthetic interruption")
        return _fake_q8(array, group_size, bits, source_dtype)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        convert_qwen38_flash_next(
            source,
            destination,
            expected_source_shards=2,
            expected_ple_shards=2,
            ple_rows_per_shard=20,
            ple_head_dim=32,
            ple_source_dtype="F32",
            quantizer=fail_on_second_shard,
            source_revision="synthetic-source-sha",
        )
    manifest = json.loads((destination / MANIFEST_NAME).read_text())
    assert len(manifest["source_shards"]) == 1

    resumed = _convert(source, destination)
    assert resumed.stats.source_shards_resumed == 1
    assert resumed.stats.source_shards_completed == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config, index: config.update(model_type="qwen3_5"), "model_type"),
        (
            lambda config, index: index["weight_map"].pop("mtp.fc_hidden.weight"),
            "MTP projection",
        ),
        (
            lambda config, index: config.update(quantization={"bits": 8}),
            "unquantized",
        ),
    ],
)
def test_fail_closed_before_reading_weights(tmp_path, mutation, message):
    source = tmp_path / "source"
    _write_source(source)
    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / INDEX_NAME).read_text())
    mutation(config, index)
    (source / "config.json").write_text(json.dumps(config))
    (source / INDEX_NAME).write_text(json.dumps(index))
    with pytest.raises(Qwen4ExpConversionError, match=message):
        _convert(source, tmp_path / "output")
    assert not (tmp_path / "output" / MANIFEST_NAME).exists()


def test_fail_closed_on_ple_shape_and_manifest_identity_drift(tmp_path):
    source = tmp_path / "source"
    _write_source(source)
    destination = tmp_path / "output"
    _convert(source, destination)

    with pytest.raises(Qwen4ExpConversionError, match="manifest mismatch"):
        convert_qwen38_flash_next(
            source,
            destination,
            expected_source_shards=2,
            expected_ple_shards=2,
            ple_rows_per_shard=20,
            ple_head_dim=64,
            ple_source_dtype="F32",
            quantizer=_fake_q8,
            source_revision="synthetic-source-sha",
        )
