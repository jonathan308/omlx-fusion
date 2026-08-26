# SPDX-License-Identifier: Apache-2.0
"""Bounded, resumable Qwen3.8-Flash-Next BF16 -> MLX Q8 conversion.

The official checkpoint is roughly 360 GB.  This converter deliberately works
one published safetensors shard (and one source tensor) at a time; it never
constructs the model and never collects the source checkpoint in memory.

Two artifacts are produced:

* ``compute-q8`` contains the ordinary model, including every official
  ``mtp.*`` tensor.  Eligible matrix weights use MLX affine Q8.
* ``ple-bf16`` is the default correctness artifact: the 128 PLE embedding
  shards are copied byte-for-byte into the SSD pool without materialization.
* ``ple-q8`` is an explicit optional affine-Q8 PLE artifact in the exact
  layout consumed by :mod:`.ple`.

The PLE projections/normalization/conv remain in the compute artifact.  A
global PLE table tensor can therefore never leak into resident model loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from .ple import (
    OFFICIAL_PLE_LAYOUT,
    OFFICIAL_PLE_Q8_LAYOUT,
    PLE_MLX_Q8_BITS,
    PLE_MLX_Q8_GROUP_SIZE,
    PLE_PREFIX,
    PLE_SHARD_COUNT,
    PLE_TABLE_PREFIX,
    mlx_q8_index_metadata,
)

CONVERTER_VERSION: Final = 1
LAYOUT_VERSION: Final = "qwen4-exp-split-q8-v1"
DEFAULT_SOURCE_SHARDS: Final = 131
COMPUTE_DIRNAME: Final = "compute-q8"
PLE_BF16_DIRNAME: Final = "ple-bf16"
PLE_Q8_DIRNAME: Final = "ple-q8"
# Compatibility name for callers using the default artifact.
PLE_DIRNAME: Final = PLE_BF16_DIRNAME
MANIFEST_NAME: Final = "conversion-manifest.json"
INDEX_NAME: Final = "model.safetensors.index.json"

_PLE_AUXILIARY: Final = frozenset(
    {
        f"{PLE_PREFIX}.layer_multipliers",
        f"{PLE_PREFIX}.ngram_heads_offsets",
        f"{PLE_PREFIX}.ngram_heads_vocab_sizes",
    }
)
_TOKENIZER_FILES: Final = frozenset(
    {
        "added_tokens.json",
        "chat_template.json",
        "chat_template.jinja",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)
_MTP_LAYER_RE: Final = re.compile(r"^mtp\.layers\.(\d+)\.")


class Qwen4ExpConversionError(ValueError):
    """The source or destination violates the converter contract."""


Quantizer = Callable[
    [np.ndarray, int, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]
]


@dataclass(frozen=True, slots=True)
class _EncodedTensor:
    data: np.ndarray
    dtype_name: str

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def nbytes(self) -> int:
        return self.data.nbytes


_SOURCE_DTYPES: Final[dict[str, tuple[np.dtype[Any], int]]] = {
    "BOOL": (np.dtype("?"), 1),
    "I8": (np.dtype("i1"), 1),
    "U8": (np.dtype("u1"), 1),
    "I16": (np.dtype("<i2"), 2),
    "U16": (np.dtype("<u2"), 2),
    "I32": (np.dtype("<i4"), 4),
    "U32": (np.dtype("<u4"), 4),
    "I64": (np.dtype("<i8"), 8),
    "U64": (np.dtype("<u8"), 8),
    "F16": (np.dtype("<f2"), 2),
    # BF16 is deliberately represented by its raw words.
    "BF16": (np.dtype("<u2"), 2),
    "F32": (np.dtype("<f4"), 4),
    "F64": (np.dtype("<f8"), 8),
}

_NUMPY_TO_SAFETENSORS: Final[dict[str, str]] = {
    "bool": "BOOL",
    "int8": "I8",
    "uint8": "U8",
    "int16": "I16",
    "uint16": "U16",
    "int32": "I32",
    "uint32": "U32",
    "int64": "I64",
    "uint64": "U64",
    "float16": "F16",
    "float32": "F32",
    "float64": "F64",
}


@dataclass(slots=True)
class ConversionStats:
    source_shards_completed: int = 0
    source_shards_resumed: int = 0
    source_tensors_read: int = 0
    max_source_tensor_bytes: int = 0
    compute_tensors: int = 0
    ple_tensors: int = 0
    ple_table_tensors_stream_copied: int = 0


@dataclass(frozen=True, slots=True)
class ConversionResult:
    root: Path
    compute_dir: Path
    ple_dir: Path
    manifest_path: Path
    stats: ConversionStats = field(compare=False)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_file(root: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename:
        raise Qwen4ExpConversionError("weight_map contains a non-string filename")
    pure = PurePosixPath(filename)
    if pure.is_absolute() or ".." in pure.parts or "\\" in filename:
        raise Qwen4ExpConversionError(f"unsafe source shard path: {filename!r}")
    path = root.joinpath(*pure.parts)
    try:
        mode = path.stat().st_mode
    except FileNotFoundError as exc:
        raise Qwen4ExpConversionError(f"missing source shard: {filename}") from exc
    if not stat.S_ISREG(mode):
        raise Qwen4ExpConversionError(f"source shard is not a file: {filename}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except FileNotFoundError as exc:
        raise Qwen4ExpConversionError(f"missing required file: {path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Qwen4ExpConversionError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Qwen4ExpConversionError(f"JSON root must be an object: {path}")
    return value


def _validate_source(
    source: Path,
    *,
    expected_source_shards: int | None,
    expected_ple_shards: int,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    config = _load_json(source / "config.json")
    text_config = config.get("text_config")
    architectures = config.get("architectures")
    if config.get("model_type") != "qwen4_exp":
        raise Qwen4ExpConversionError("source model_type must be qwen4_exp")
    if (
        not isinstance(text_config, dict)
        or text_config.get("model_type") != "qwen4_exp_text"
    ):
        raise Qwen4ExpConversionError(
            "source text_config.model_type must be qwen4_exp_text"
        )
    if architectures != ["Qwen4ExpForConditionalGeneration"]:
        raise Qwen4ExpConversionError("unexpected Qwen4-Exp architecture declaration")
    if "quantization" in config or "quantization_config" in config:
        raise Qwen4ExpConversionError(
            "source must be the unquantized official artifact"
        )

    index = _load_json(source / INDEX_NAME)
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict) or not raw_weight_map:
        raise Qwen4ExpConversionError("source index requires a non-empty weight_map")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_weight_map.items()
    ):
        raise Qwen4ExpConversionError("source weight_map must map strings to strings")
    weight_map: dict[str, str] = dict(raw_weight_map)
    shard_names = sorted(set(weight_map.values()))
    if (
        expected_source_shards is not None
        and len(shard_names) != expected_source_shards
    ):
        raise Qwen4ExpConversionError(
            f"expected {expected_source_shards} source shards, found {len(shard_names)}"
        )
    for shard_name in shard_names:
        _safe_relative_file(source, shard_name)

    ple_names = {
        f"{PLE_TABLE_PREFIX}.shard_{index}.weight"
        for index in range(expected_ple_shards)
    }
    indexed_ple = {
        name for name in weight_map if name.startswith(f"{PLE_TABLE_PREFIX}.shard_")
    }
    missing_ple = ple_names.difference(indexed_ple)
    unexpected_ple = indexed_ple.difference(ple_names)
    if missing_ple or unexpected_ple:
        raise Qwen4ExpConversionError(
            "source PLE table is not the complete expected shard set "
            f"(missing={len(missing_ple)}, unexpected={len(unexpected_ple)})"
        )
    missing_aux = _PLE_AUXILIARY.difference(weight_map)
    if missing_aux:
        raise Qwen4ExpConversionError("source is missing PLE hash metadata tensors")

    mtp_names = {name for name in weight_map if name.startswith("mtp.")}
    required_mtp = {"mtp.fc_embedding.weight", "mtp.fc_hidden.weight"}
    if not required_mtp.issubset(mtp_names):
        raise Qwen4ExpConversionError(
            "source is missing official MTP projection tensors"
        )
    mtp_layers = {
        int(match.group(1))
        for name in mtp_names
        if (match := _MTP_LAYER_RE.match(name)) is not None
    }
    if mtp_layers != {0}:
        raise Qwen4ExpConversionError(
            f"Qwen3.8-Flash-Next requires exactly MTP layer 0, found {sorted(mtp_layers)}"
        )
    return config, weight_map, shard_names


def _is_ple_table(name: str) -> bool:
    return name.startswith(f"{PLE_TABLE_PREFIX}.shard_")


def _must_remain_dense(name: str) -> bool:
    lower = name.lower()
    return (
        name in _PLE_AUXILIARY
        or "a_log" in lower
        or "dt_bias" in lower
        or "norm" in lower
        or "conv1d" in lower
        or lower.endswith(".bias")
        or "ngram_heads_" in lower
        or "layer_multipliers" in lower
    )


def _eligible_q8(name: str, tensor: np.ndarray, *, group_size: int) -> bool:
    return (
        name.endswith(".weight")
        and tensor.ndim >= 2
        and tensor.shape[-1] % group_size == 0
        and not _must_remain_dense(name)
    )


def _mlx_affine_q8(
    tensor: np.ndarray, group_size: int, bits: int, source_dtype: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bits != 8:
        raise Qwen4ExpConversionError("Qwen4-Exp streaming converter only emits Q8")
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - platform/package guard
        raise RuntimeError("MLX is required to convert Qwen4-Exp weights") from exc
    if source_dtype == "BF16":
        # ``safetensors.numpy`` cannot decode BF16 without ml_dtypes.  The raw
        # uint16 words are losslessly reinterpreted inside MLX instead.
        source = mx.array(tensor).view(mx.bfloat16)
    else:
        source = mx.array(tensor)
    weight, scales, biases = mx.quantize(
        source, group_size=group_size, bits=bits, mode="affine"
    )
    mx.eval(weight, scales, biases)
    if source_dtype == "BF16":
        # NumPy cannot represent MLX BF16 on the pinned runtime. Preserve the
        # exact words and let the safetensors writer label them BF16.
        scale_array = np.asarray(scales.view(mx.uint16))
        bias_array = np.asarray(biases.view(mx.uint16))
    else:
        scale_array = np.asarray(scales)
        bias_array = np.asarray(biases)
    return np.asarray(weight), scale_array, bias_array


def _quantized_tensors(
    name: str,
    tensor: _EncodedTensor,
    *,
    quantizer: Quantizer,
    group_size: int,
    bits: int,
) -> dict[str, _EncodedTensor]:
    if not name.endswith(".weight"):
        raise Qwen4ExpConversionError(f"cannot quantize non-weight tensor {name}")
    weight, scales, biases = quantizer(tensor.data, group_size, bits, tensor.dtype_name)
    base = name[: -len(".weight")]
    scale_dtype = (
        tensor.dtype_name if tensor.dtype_name in {"BF16", "F16", "F32"} else "F32"
    )
    return {
        f"{base}.weight": _EncodedTensor(np.asarray(weight), "U32"),
        f"{base}.scales": _EncodedTensor(np.asarray(scales), scale_dtype),
        f"{base}.biases": _EncodedTensor(np.asarray(biases), scale_dtype),
    }


def _atomic_safetensors(path: Path, tensors: Mapping[str, _EncodedTensor]) -> str:
    if not tensors:
        raise Qwen4ExpConversionError(f"refusing to write empty shard {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        header: dict[str, Any] = {"__metadata__": {"format": "mlx"}}
        cursor = 0
        ordered = sorted(tensors.items())
        for name, tensor in ordered:
            array = np.ascontiguousarray(tensor.data)
            expected_dtype = _SOURCE_DTYPES.get(tensor.dtype_name)
            if expected_dtype is None or array.dtype != expected_dtype[0]:
                raise Qwen4ExpConversionError(
                    f"encoded tensor {name} has {array.dtype}, expected {tensor.dtype_name}"
                )
            header[name] = {
                "dtype": tensor.dtype_name,
                "shape": list(array.shape),
                "data_offsets": [cursor, cursor + array.nbytes],
            }
            cursor += array.nbytes
        encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded_header += b" " * (-len(encoded_header) % 8)
        with temporary.open("xb", buffering=0) as stream:
            stream.write(struct.pack("<Q", len(encoded_header)))
            stream.write(encoded_header)
            for _, tensor in ordered:
                array = np.ascontiguousarray(tensor.data)
                raw = memoryview(array).cast("B")
                for start in range(0, len(raw), 8 * 1024 * 1024):
                    stream.write(raw[start : start + 8 * 1024 * 1024])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb", buffering=0) as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise Qwen4ExpConversionError(f"truncated safetensors file: {path.name}")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_length <= 16 * 1024 * 1024:
            raise Qwen4ExpConversionError(f"invalid safetensors header: {path.name}")
        try:
            header = json.loads(stream.read(header_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise Qwen4ExpConversionError(
                f"invalid safetensors JSON header: {path.name}"
            ) from exc
    if not isinstance(header, dict):
        raise Qwen4ExpConversionError(f"invalid safetensors header root: {path.name}")
    return header, 8 + header_length


def _read_source_tensor(
    path: Path,
    header: Mapping[str, Any],
    data_start: int,
    name: str,
) -> _EncodedTensor:
    descriptor = header.get(name)
    if not isinstance(descriptor, dict):
        raise Qwen4ExpConversionError(f"source header is missing tensor {name}")
    dtype_name = descriptor.get("dtype")
    shape_value = descriptor.get("shape")
    offsets = descriptor.get("data_offsets")
    if (
        not isinstance(dtype_name, str)
        or dtype_name not in _SOURCE_DTYPES
        or not isinstance(shape_value, list)
        or not all(isinstance(value, int) and value >= 0 for value in shape_value)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) for value in offsets)
    ):
        raise Qwen4ExpConversionError(f"malformed tensor descriptor {name}")
    shape = tuple(shape_value)
    start, stop = offsets
    numpy_dtype, item_size = _SOURCE_DTYPES[dtype_name]
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * item_size
    if stop - start != expected_bytes:
        raise Qwen4ExpConversionError(f"invalid byte length for tensor {name}")
    array = np.empty(shape, dtype=numpy_dtype)
    raw = memoryview(array).cast("B")
    with path.open("rb", buffering=0) as stream:
        stream.seek(data_start + start)
        position = 0
        while position < expected_bytes:
            count = stream.readinto(raw[position : position + 8 * 1024 * 1024])
            if not count:
                raise Qwen4ExpConversionError(f"truncated source tensor {name}")
            position += count
    return _EncodedTensor(array, dtype_name)


def _atomic_raw_safetensors_subset(
    source: Path,
    destination: Path,
    names: Sequence[str],
) -> tuple[str, int]:
    """Extract tensors by streaming their encoded bytes, including BF16.

    This is the default PLE path: a ~800 MB embedding shard is never turned
    into a NumPy or MLX array merely to preserve it exactly.
    """

    source_header, source_data_start = _safetensors_header(source)
    output_header: dict[str, Any] = {"__metadata__": {"format": "mlx"}}
    copies: list[tuple[int, int]] = []
    cursor = 0
    for name in sorted(names):
        descriptor = source_header.get(name)
        if not isinstance(descriptor, dict):
            raise Qwen4ExpConversionError(
                f"source header does not contain indexed tensor {name}"
            )
        offsets = descriptor.get("data_offsets")
        shape = descriptor.get("shape")
        dtype = descriptor.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or not isinstance(shape, list)
            or not isinstance(dtype, str)
        ):
            raise Qwen4ExpConversionError(f"malformed source tensor {name}")
        start, stop = offsets
        if not 0 <= start <= stop:
            raise Qwen4ExpConversionError(f"invalid source offsets for {name}")
        length = stop - start
        output_header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + length],
        }
        copies.append((source_data_start + start, length))
        cursor += length

    encoded_header = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    encoded_header += b" " * (-len(encoded_header) % 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with (
            source.open("rb", buffering=0) as input_stream,
            temporary.open("xb", buffering=0) as output_stream,
        ):
            output_stream.write(struct.pack("<Q", len(encoded_header)))
            output_stream.write(encoded_header)
            for offset, remaining in copies:
                input_stream.seek(offset)
                while remaining:
                    chunk = input_stream.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise Qwen4ExpConversionError(
                            f"truncated tensor data in {source.name}"
                        )
                    output_stream.write(chunk)
                    remaining -= len(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(destination), cursor


def _output_valid(root: Path, record: Mapping[str, Any]) -> bool:
    for key in ("compute", "ple"):
        item = record.get(key)
        if item is None:
            continue
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            return False
        path = root / item["file"]
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            return False
    return True


def _copy_metadata_files(source: Path, compute_dir: Path, ple_dir: Path) -> None:
    for item in source.iterdir():
        if not item.is_file() or (
            item.name not in _TOKENIZER_FILES and item.name != "config.json"
        ):
            continue
        for destination in (compute_dir / item.name, ple_dir / item.name):
            temporary = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                shutil.copy2(item, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)


def _write_compute_config(
    source_config: Mapping[str, Any], compute_dir: Path, *, ple_dirname: str
) -> None:
    config = dict(source_config)
    config["quantization"] = {
        "bits": PLE_MLX_Q8_BITS,
        "group_size": PLE_MLX_Q8_GROUP_SIZE,
        "mode": "affine",
    }
    config["qwen4_exp_artifact"] = {
        "layout": LAYOUT_VERSION,
        "ple_artifact": f"../{ple_dirname}",
        "ple_residency": "ssd_mmap",
    }
    _atomic_bytes(compute_dir / "config.json", _json_bytes(config))


def _index_from_records(
    records: Mapping[str, Any], key: str, *, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    weight_map: dict[str, str] = {}
    total_size = 0
    for record in records.values():
        output = record.get(key)
        if not isinstance(output, dict):
            continue
        filename = Path(output["file"]).name
        for name in output["tensors"]:
            if name in weight_map:
                raise Qwen4ExpConversionError(f"duplicate output tensor {name}")
            weight_map[name] = filename
        total_size += int(output["tensor_bytes"])
    return {
        "metadata": {**metadata, "total_size": total_size},
        "weight_map": weight_map,
    }


def convert_qwen38_flash_next(
    source_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    expected_source_shards: int | None = DEFAULT_SOURCE_SHARDS,
    expected_ple_shards: int = PLE_SHARD_COUNT,
    ple_rows_per_shard: int = OFFICIAL_PLE_LAYOUT.rows_per_shard,
    ple_head_dim: int = OFFICIAL_PLE_LAYOUT.head_dim,
    ple_source_dtype: str = "BF16",
    ple_quantization: str = "bf16",
    quantizer: Quantizer = _mlx_affine_q8,
    source_revision: str | None = None,
) -> ConversionResult:
    """Convert an official BF16 artifact without loading or constructing it."""

    if ple_quantization not in {"bf16", "q8"}:
        raise Qwen4ExpConversionError("ple_quantization must be 'bf16' or 'q8'")
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if source == output or source in output.parents:
        raise Qwen4ExpConversionError("output must not be the source or its parent")
    config, weight_map, source_shards = _validate_source(
        source,
        expected_source_shards=expected_source_shards,
        expected_ple_shards=expected_ple_shards,
    )
    output.mkdir(parents=True, exist_ok=True)
    compute_dir = output / COMPUTE_DIRNAME
    ple_dirname = PLE_BF16_DIRNAME if ple_quantization == "bf16" else PLE_Q8_DIRNAME
    ple_dir = output / ple_dirname
    compute_dir.mkdir(exist_ok=True)
    ple_dir.mkdir(exist_ok=True)
    manifest_path = output / MANIFEST_NAME
    source_index_sha = _sha256(source / INDEX_NAME)
    identity = {
        "converter_version": CONVERTER_VERSION,
        "layout_version": LAYOUT_VERSION,
        "source_index_sha256": source_index_sha,
        "source_revision": source_revision,
        "source_shard_count": len(source_shards),
        "q8_bits": PLE_MLX_Q8_BITS,
        "q8_group_size": PLE_MLX_Q8_GROUP_SIZE,
        "q8_mode": "affine",
        "ple_shard_count": expected_ple_shards,
        "ple_rows_per_shard": ple_rows_per_shard,
        "ple_head_dim": ple_head_dim,
        "ple_source_dtype": ple_source_dtype,
        "ple_quantization": ple_quantization,
    }
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        for key, expected in identity.items():
            if manifest.get(key) != expected:
                raise Qwen4ExpConversionError(
                    f"existing conversion manifest mismatch for {key}; use a fresh destination"
                )
        records = manifest.get("source_shards")
        if not isinstance(records, dict):
            raise Qwen4ExpConversionError(
                "conversion manifest source_shards is invalid"
            )
    else:
        records: dict[str, Any] = {}
        manifest = {**identity, "source": str(source), "source_shards": records}
        _atomic_bytes(manifest_path, _json_bytes(manifest))

    names_by_shard: dict[str, list[str]] = {name: [] for name in source_shards}
    for tensor_name, shard_name in weight_map.items():
        names_by_shard[shard_name].append(tensor_name)

    stats = ConversionStats()
    for ordinal, shard_name in enumerate(source_shards, start=1):
        source_path = _safe_relative_file(source, shard_name)
        source_sha = _sha256(source_path)
        old_record = records.get(shard_name)
        if (
            isinstance(old_record, dict)
            and old_record.get("source_sha256") == source_sha
            and _output_valid(output, old_record)
        ):
            stats.source_shards_resumed += 1
            continue

        compute_tensors: dict[str, _EncodedTensor] = {}
        ple_tensors: dict[str, _EncodedTensor] = {}
        ple_raw_names: list[str] = []
        expected_names = sorted(names_by_shard[shard_name])
        source_header, source_data_start = _safetensors_header(source_path)
        actual_names = sorted(name for name in source_header if name != "__metadata__")
        if actual_names != expected_names:
            raise Qwen4ExpConversionError(
                f"index/header tensor mismatch in {shard_name}"
            )
        for name in expected_names:
            if _is_ple_table(name) and ple_quantization == "bf16":
                descriptor = source_header.get(name)
                if not isinstance(descriptor, dict):
                    raise Qwen4ExpConversionError(f"missing PLE tensor header: {name}")
                shape = tuple(descriptor.get("shape", ()))
                dtype = descriptor.get("dtype")
                if shape != (ple_rows_per_shard, ple_head_dim):
                    raise Qwen4ExpConversionError(
                        f"PLE table tensor has invalid shape: {name} {shape}; "
                        f"expected {(ple_rows_per_shard, ple_head_dim)}"
                    )
                if dtype != ple_source_dtype:
                    raise Qwen4ExpConversionError(
                        f"PLE table tensor {name} must be {ple_source_dtype}, found {dtype}"
                    )
                ple_raw_names.append(name)
                stats.ple_table_tensors_stream_copied += 1
                continue
            tensor = _read_source_tensor(
                source_path, source_header, source_data_start, name
            )
            stats.source_tensors_read += 1
            stats.max_source_tensor_bytes = max(
                stats.max_source_tensor_bytes, tensor.nbytes
            )
            if _is_ple_table(name):
                descriptor = source_header.get(name)
                if (
                    not isinstance(descriptor, dict)
                    or descriptor.get("dtype") != ple_source_dtype
                ):
                    raise Qwen4ExpConversionError(
                        f"PLE table tensor {name} must be {ple_source_dtype}"
                    )
                if tensor.shape != (ple_rows_per_shard, ple_head_dim):
                    raise Qwen4ExpConversionError(
                        f"PLE table tensor has invalid shape: {name} {tensor.shape}; "
                        f"expected {(ple_rows_per_shard, ple_head_dim)}"
                    )
                if not _eligible_q8(
                    name, tensor.data, group_size=PLE_MLX_Q8_GROUP_SIZE
                ):
                    raise Qwen4ExpConversionError(
                        f"PLE table tensor cannot use MLX Q8 contract: {name} {tensor.shape}"
                    )
                ple_tensors.update(
                    _quantized_tensors(
                        name,
                        tensor,
                        quantizer=quantizer,
                        group_size=PLE_MLX_Q8_GROUP_SIZE,
                        bits=PLE_MLX_Q8_BITS,
                    )
                )
            else:
                if _eligible_q8(name, tensor.data, group_size=PLE_MLX_Q8_GROUP_SIZE):
                    compute_tensors.update(
                        _quantized_tensors(
                            name,
                            tensor,
                            quantizer=quantizer,
                            group_size=PLE_MLX_Q8_GROUP_SIZE,
                            bits=PLE_MLX_Q8_BITS,
                        )
                    )
                else:
                    compute_tensors[name] = tensor
                if name in _PLE_AUXILIARY:
                    if ple_quantization == "bf16":
                        ple_raw_names.append(name)
                    else:
                        ple_tensors[name] = tensor

        output_name = f"model-{ordinal:05d}-of-{len(source_shards):05d}.safetensors"
        record: dict[str, Any] = {"source_sha256": source_sha}
        if compute_tensors:
            relative = f"{COMPUTE_DIRNAME}/{output_name}"
            checksum = _atomic_safetensors(output / relative, compute_tensors)
            record["compute"] = {
                "file": relative,
                "sha256": checksum,
                "tensors": sorted(compute_tensors),
                "tensor_bytes": sum(value.nbytes for value in compute_tensors.values()),
            }
            stats.compute_tensors += len(compute_tensors)
        if ple_tensors or ple_raw_names:
            relative = f"{ple_dirname}/{output_name}"
            if ple_quantization == "bf16":
                checksum, tensor_bytes = _atomic_raw_safetensors_subset(
                    source_path, output / relative, ple_raw_names
                )
                output_tensor_names = sorted(ple_raw_names)
            else:
                checksum = _atomic_safetensors(output / relative, ple_tensors)
                tensor_bytes = sum(value.nbytes for value in ple_tensors.values())
                output_tensor_names = sorted(ple_tensors)
            record["ple"] = {
                "file": relative,
                "sha256": checksum,
                "tensors": output_tensor_names,
                "tensor_bytes": tensor_bytes,
            }
            stats.ple_tensors += len(output_tensor_names)
        records[shard_name] = record
        manifest["source_shards"] = records
        _atomic_bytes(manifest_path, _json_bytes(manifest))
        stats.source_shards_completed += 1

    if set(records) != set(source_shards):
        raise Qwen4ExpConversionError("conversion ended without every source shard")

    compute_index = _index_from_records(
        records,
        "compute",
        metadata={
            "format": "mlx",
            "qwen4_exp_layout": LAYOUT_VERSION,
            "qwen4_exp_source_index_sha256": source_index_sha,
            "qwen4_exp_mtp_depth": 1,
        },
    )
    ple_metadata = {
        "format": "mlx",
        "qwen4_exp_ple_format": "bf16-ssd-mmap-v1",
        "qwen4_exp_ple_dtype": ple_source_dtype,
        "qwen4_exp_ple_layer": 1,
        "qwen4_exp_ple_shard_count": expected_ple_shards,
        "qwen4_exp_ple_rows_per_shard": ple_rows_per_shard,
        "qwen4_exp_ple_head_dim": ple_head_dim,
        "qwen4_exp_source_index_sha256": source_index_sha,
    }
    if ple_quantization == "q8":
        q8_metadata = mlx_q8_index_metadata(OFFICIAL_PLE_Q8_LAYOUT)
        if (
            expected_ple_shards != PLE_SHARD_COUNT
            or ple_rows_per_shard != OFFICIAL_PLE_Q8_LAYOUT.rows_per_shard
            or ple_head_dim != OFFICIAL_PLE_Q8_LAYOUT.head_dim
        ):
            # Bounded synthetic tests use a tiny structural facsimile. The CLI
            # path always emits the untouched official metadata contract.
            q8_metadata.update(
                {
                    "qwen4_exp_ple_shard_count": expected_ple_shards,
                    "qwen4_exp_ple_rows_per_shard": ple_rows_per_shard,
                    "qwen4_exp_ple_head_dim": ple_head_dim,
                }
            )
        ple_metadata.update(q8_metadata)
    ple_index = _index_from_records(records, "ple", metadata=ple_metadata)
    source_mtp_names = {name for name in weight_map if name.startswith("mtp.")}
    missing_mtp = source_mtp_names.difference(compute_index["weight_map"])
    if missing_mtp:
        preview = ", ".join(sorted(missing_mtp)[:3])
        raise Qwen4ExpConversionError(
            f"conversion dropped {len(missing_mtp)} MTP tensor(s): {preview}"
        )
    if any(_is_ple_table(name) for name in compute_index["weight_map"]):
        raise Qwen4ExpConversionError("PLE table leaked into compute artifact")
    table_tensors_per_shard = 1 if ple_quantization == "bf16" else 3
    expected_ple_output = expected_ple_shards * table_tensors_per_shard + len(
        _PLE_AUXILIARY
    )
    if len(ple_index["weight_map"]) != expected_ple_output:
        raise Qwen4ExpConversionError(
            f"PLE Q8 artifact has {len(ple_index['weight_map'])} tensors, expected {expected_ple_output}"
        )

    _atomic_bytes(compute_dir / INDEX_NAME, _json_bytes(compute_index))
    _atomic_bytes(ple_dir / INDEX_NAME, _json_bytes(ple_index))
    _copy_metadata_files(source, compute_dir, ple_dir)
    _write_compute_config(config, compute_dir, ple_dirname=ple_dirname)
    manifest["complete"] = True
    manifest["compute_index_sha256"] = _sha256(compute_dir / INDEX_NAME)
    manifest["ple_index_sha256"] = _sha256(ple_dir / INDEX_NAME)
    _atomic_bytes(manifest_path, _json_bytes(manifest))
    return ConversionResult(output, compute_dir, ple_dir, manifest_path, stats)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream official Qwen3.8-Flash-Next BF16 into split MLX Q8 artifacts"
    )
    parser.add_argument("source", help="official 131-shard BF16 model directory")
    parser.add_argument("output", help="new output directory")
    parser.add_argument(
        "--source-revision",
        help="optional Hugging Face commit SHA recorded in the manifest",
    )
    parser.add_argument(
        "--ple-q8",
        action="store_true",
        help="quantize the SSD PLE table to affine Q8 (default preserves BF16)",
    )
    args = parser.parse_args(argv)
    result = convert_qwen38_flash_next(
        args.source,
        args.output,
        source_revision=args.source_revision,
        ple_quantization="q8" if args.ple_q8 else "bf16",
    )
    print(result.compute_dir)
    print(result.ple_dir)
    print(result.manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
