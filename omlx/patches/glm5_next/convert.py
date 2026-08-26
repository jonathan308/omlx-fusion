# SPDX-License-Identifier: Apache-2.0
"""Bounded GLM-5.3-Flash block-FP8 -> MLX affine conversion.

The converter never constructs the model and never holds a source shard (or
an expert bank) in memory. Ordinary tensors are processed one at a time. The
288 routed experts are quantized one expert at a time and streamed into the
packed ``switch_mlp`` layout consumed by the GLM5-Next MoE primitive.
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
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from .contract import (
    OFFICIAL_REVISION,
    Glm5NextContractError,
    Glm5NextSourceContract,
    validate_source_contract,
)

CONVERTER_VERSION: Final = 2
LAYOUT_VERSION: Final = "glm5-next-multimodal-affine-v2"
INDEX_NAME: Final = "model.safetensors.index.json"
MANIFEST_NAME: Final = "conversion-manifest.json"
DEFAULT_GROUP_SIZE: Final = 64
FP8_BLOCK_SIZE: Final = 128
OFFICIAL_EXPERTS: Final = 288

_EXPERT_RE = re.compile(
    r"^(?P<prefix>model\.language_model\.layers\.(?P<layer>\d+)\.mlp)\."
    r"experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_ALLOWED_ROOTS = ("lm_head.", "model.language_model.", "model.visual.")
_COPY_FILES: Final = frozenset(
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


class Glm5NextConversionError(Glm5NextContractError):
    """The source, conversion, or destination violates the exact ABI."""


class Glm5NextUnsupportedMathError(RuntimeError):
    """Compatibility alias retained for callers of the former planner."""


@dataclass(frozen=True, slots=True)
class TensorHeader:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    source_bytes: int
    data_start: int = 0
    data_offset: int = 0


@dataclass(frozen=True, slots=True)
class Glm5NextConversionPlan:
    source: Path
    destination: Path
    contract: Glm5NextSourceContract
    source_payload_bytes: int
    dense_bf16_upper_bound_bytes: int
    largest_tensor_bytes: int
    unsupported_math: tuple[str, ...] = ()
    bits: int = 8
    group_size: int = DEFAULT_GROUP_SIZE
    projected_output_bytes: int = 0


@dataclass(slots=True)
class ConversionStats:
    units_completed: int = 0
    units_resumed: int = 0
    source_tensors_read: int = 0
    max_source_tensor_bytes: int = 0
    max_dequantized_tensor_bytes: int = 0
    expert_tensors_packed: int = 0


@dataclass(frozen=True, slots=True)
class ConversionResult:
    output_dir: Path
    index_path: Path
    manifest_path: Path
    bits: int
    projected_output_bytes: int
    stats: ConversionStats = field(compare=False)


@dataclass(frozen=True, slots=True)
class _EncodedTensor:
    data: np.ndarray
    dtype_name: str

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


@dataclass(frozen=True, slots=True)
class _OutputSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return _elements(self.shape) * _DTYPE_BYTES[self.dtype]


Quantizer = Callable[
    [np.ndarray, int, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]
]

_SOURCE_DTYPES: Final[dict[str, tuple[np.dtype[Any], int]]] = {
    "BOOL": (np.dtype("?"), 1),
    "I8": (np.dtype("i1"), 1),
    "U8": (np.dtype("u1"), 1),
    "F8_E4M3": (np.dtype("u1"), 1),
    "F8_E4M3FN": (np.dtype("u1"), 1),
    "I16": (np.dtype("<i2"), 2),
    "U16": (np.dtype("<u2"), 2),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
    "I32": (np.dtype("<i4"), 4),
    "U32": (np.dtype("<u4"), 4),
    "F32": (np.dtype("<f4"), 4),
    "I64": (np.dtype("<i8"), 8),
    "U64": (np.dtype("<u8"), 8),
    "F64": (np.dtype("<f8"), 8),
}
_DTYPE_BYTES: Final = {name: item[1] for name, item in _SOURCE_DTYPES.items()}
_FLOAT_DTYPES: Final = frozenset({"F8_E4M3", "F8_E4M3FN", "F16", "BF16", "F32", "F64"})


def _elements(shape: Sequence[int]) -> int:
    value = 1
    for dimension in shape:
        value *= int(dimension)
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Glm5NextConversionError(f"cannot read JSON file: {path.name}") from exc
    if not isinstance(value, dict):
        raise Glm5NextConversionError(f"JSON root must be an object: {path.name}")
    return value


def _load_index(source: Path) -> Mapping[str, Any]:
    value = _load_json(source / INDEX_NAME)
    if not isinstance(value.get("weight_map"), Mapping):
        raise Glm5NextConversionError(f"invalid {INDEX_NAME}")
    return value


def _safe_source_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise Glm5NextConversionError("weight_map contains an invalid shard name")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise Glm5NextConversionError(f"unsafe source shard path: {value!r}")
    path = root.joinpath(*pure.parts)
    try:
        mode = path.stat().st_mode
    except FileNotFoundError as exc:
        raise Glm5NextConversionError(f"missing source shard: {value}") from exc
    if not stat.S_ISREG(mode):
        raise Glm5NextConversionError(f"source shard is not a file: {value}")
    return path


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    try:
        with path.open("rb", buffering=0) as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise Glm5NextConversionError(f"truncated safetensors: {path.name}")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= 256 * 1024 * 1024:
                raise Glm5NextConversionError(
                    f"invalid safetensors header: {path.name}"
                )
            value = json.loads(stream.read(length))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Glm5NextConversionError(f"invalid safetensors file: {path.name}") from exc
    if not isinstance(value, dict):
        raise Glm5NextConversionError(f"invalid safetensors header root: {path.name}")
    return value, 8 + length


def _catalog(source: Path) -> tuple[dict[str, TensorHeader], list[str], dict[str, str]]:
    index = _load_index(source)
    raw_map = index["weight_map"]
    if not raw_map or not all(
        isinstance(n, str) and isinstance(s, str) for n, s in raw_map.items()
    ):
        raise Glm5NextConversionError("weight_map must be a non-empty string mapping")
    weight_map = dict(raw_map)
    shard_names = sorted(set(weight_map.values()))
    result: dict[str, TensorHeader] = {}
    for shard_name in shard_names:
        path = _safe_source_file(source, shard_name)
        header, data_start = _read_safetensors_header(path)
        actual = {name for name in header if name != "__metadata__"}
        expected = {name for name, item in weight_map.items() if item == shard_name}
        if actual != expected:
            raise Glm5NextConversionError(f"index/header mismatch in {shard_name}")
        for name in sorted(actual):
            descriptor = header[name]
            if not isinstance(descriptor, Mapping):
                raise Glm5NextConversionError(f"invalid tensor descriptor: {name}")
            dtype, shape, offsets = (
                descriptor.get("dtype"),
                descriptor.get("shape"),
                descriptor.get("data_offsets"),
            )
            if (
                dtype not in _SOURCE_DTYPES
                or not isinstance(shape, list)
                or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(item, int) for item in offsets)
                or not 0 <= offsets[0] <= offsets[1]
            ):
                raise Glm5NextConversionError(f"unsupported tensor descriptor: {name}")
            typed_shape = tuple(shape)
            expected_bytes = _elements(typed_shape) * _DTYPE_BYTES[dtype]
            if offsets[1] - offsets[0] != expected_bytes:
                raise Glm5NextConversionError(f"invalid tensor byte length: {name}")
            result[name] = TensorHeader(
                name,
                shard_name,
                dtype,
                typed_shape,
                expected_bytes,
                data_start,
                offsets[0],
            )
    return result, shard_names, weight_map


def iter_tensor_headers(source: Path | str) -> Iterator[TensorHeader]:
    catalog, shards, _ = _catalog(Path(source))
    for shard in shards:
        yield from (tensor for tensor in catalog.values() if tensor.shard == shard)


def _read_tensor(source: Path, tensor: TensorHeader) -> _EncodedTensor:
    array = np.empty(tensor.shape, dtype=_SOURCE_DTYPES[tensor.dtype][0])
    target = memoryview(array).cast("B")
    with _safe_source_file(source, tensor.shard).open("rb", buffering=0) as stream:
        stream.seek(tensor.data_start + tensor.data_offset)
        position = 0
        while position < tensor.source_bytes:
            count = stream.readinto(target[position : position + 8 * 1024 * 1024])
            if not count:
                raise Glm5NextConversionError(f"truncated source tensor: {tensor.name}")
            position += count
    return _EncodedTensor(array, tensor.dtype)


def dequantize_fp8_blocks(
    codes: np.ndarray, inverse_scales: np.ndarray, *, block_size: int = FP8_BLOCK_SIZE
) -> np.ndarray:
    """Decode official E4M3FN codes and multiply the 2-D inverse-scale grid."""
    if codes.dtype != np.uint8 or codes.ndim < 2:
        raise Glm5NextConversionError("FP8 codes must be uint8 with rank >= 2")
    expected = (
        *codes.shape[:-2],
        (codes.shape[-2] + block_size - 1) // block_size,
        (codes.shape[-1] + block_size - 1) // block_size,
    )
    if tuple(inverse_scales.shape) != expected:
        raise Glm5NextConversionError(
            f"FP8 inverse-scale shape changed: expected {expected}, found {tuple(inverse_scales.shape)}"
        )
    try:
        import ml_dtypes
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ml_dtypes is required for exact E4M3FN decoding") from exc
    decoded = codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    rows = np.arange(codes.shape[-2]) // block_size
    columns = np.arange(codes.shape[-1]) // block_size
    expanded = np.take(
        np.take(inverse_scales.astype(np.float32), rows, axis=-2), columns, axis=-1
    )
    return decoded * expanded


def _decode_scale(encoded: _EncodedTensor) -> np.ndarray:
    if encoded.dtype_name == "BF16":
        try:
            import ml_dtypes
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("ml_dtypes is required to decode BF16") from exc
        return encoded.data.view(ml_dtypes.bfloat16).astype(np.float32)
    if encoded.dtype_name not in {"F16", "F32", "F64"}:
        raise Glm5NextConversionError("FP8 inverse scales must be BF16/F16/F32/F64")
    return encoded.data.astype(np.float32)


def _dequantize_pair(
    source: Path, weight: TensorHeader, scale: TensorHeader, stats: ConversionStats
) -> np.ndarray:
    codes, inverse = _read_tensor(source, weight), _read_tensor(source, scale)
    stats.source_tensors_read += 2
    stats.max_source_tensor_bytes = max(
        stats.max_source_tensor_bytes, codes.nbytes, inverse.nbytes
    )
    result = dequantize_fp8_blocks(codes.data, _decode_scale(inverse))
    stats.max_dequantized_tensor_bytes = max(
        stats.max_dequantized_tensor_bytes, int(result.nbytes)
    )
    return result


def _mlx_affine(
    tensor: np.ndarray, group_size: int, bits: int, source_dtype: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bits not in {4, 8}:
        raise Glm5NextConversionError("only affine Q8 and Q4 are supported")
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MLX is required for GLM-5.3 conversion") from exc
    source = (
        mx.array(tensor).view(mx.bfloat16)
        if source_dtype == "BF16"
        else mx.array(tensor)
    )
    weight, scales, biases = mx.quantize(
        source, group_size=group_size, bits=bits, mode="affine"
    )
    mx.eval(weight, scales, biases)
    if source_dtype == "BF16":
        return (
            np.asarray(weight),
            np.asarray(scales.view(mx.uint16)),
            np.asarray(biases.view(mx.uint16)),
        )
    return np.asarray(weight), np.asarray(scales), np.asarray(biases)


def _must_remain_dense(name: str) -> bool:
    lower = name.lower()
    return (
        ".indexer." in lower
        or ".hc_" in lower
        or ".mlp.gate." in lower
        or "a_log" in lower
        or "dt_bias" in lower
        or "conv1d" in lower
        or "norm" in lower
        or lower.endswith(".bias")
        or "index_kpool" in lower
    )


def _router(name: str) -> bool:
    return name.endswith(".mlp.gate.weight") or name.endswith(
        ".mlp.gate.e_score_correction_bias"
    )


def _eligible(name: str, tensor: TensorHeader, group_size: int) -> bool:
    return (
        name.endswith(".weight")
        and len(tensor.shape) >= 2
        and tensor.shape[-1] % group_size == 0
        and tensor.dtype in _FLOAT_DTYPES
        and not _must_remain_dense(name)
    )


def _target_name(name: str) -> str:
    if not name.startswith(_ALLOWED_ROOTS):
        raise Glm5NextConversionError(f"unmapped source tensor: {name}")
    prefix = "model.language_model.layers.45."
    if not name.startswith(prefix):
        return name
    suffix = name[len(prefix) :]
    special = {
        "eh_proj.weight": "eh_proj.weight",
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "shared_head.norm.weight": "norm.weight",
    }
    return "mtp.0." + special.get(suffix, "block." + suffix)


def _quant_specs(
    name: str, header: TensorHeader, bits: int, group_size: int
) -> list[_OutputSpec]:
    if header.shape[-1] % group_size:
        raise Glm5NextConversionError(f"tensor is not group aligned: {name}")
    base = _target_name(name).removesuffix(".weight")
    packed = (*header.shape[:-1], header.shape[-1] * bits // 32)
    groups = (*header.shape[:-1], header.shape[-1] // group_size)
    scale_dtype = "BF16" if header.dtype == "BF16" else "F32"
    return [
        _OutputSpec(base + ".weight", "U32", packed),
        _OutputSpec(base + ".scales", scale_dtype, groups),
        _OutputSpec(base + ".biases", scale_dtype, groups),
    ]


def _dense_spec(name: str, header: TensorHeader) -> _OutputSpec:
    dtype = "F32" if _router(name) or header.dtype.startswith("F8_") else header.dtype
    return _OutputSpec(_target_name(name), dtype, header.shape)


def _encoded_header(specs: Sequence[_OutputSpec]) -> tuple[bytes, int]:
    header: dict[str, Any] = {"__metadata__": {"format": "mlx"}}
    cursor, seen = 0, set()
    for spec in specs:
        if spec.name in seen:
            raise Glm5NextConversionError(f"duplicate output tensor: {spec.name}")
        seen.add(spec.name)
        header[spec.name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [cursor, cursor + spec.nbytes],
        }
        cursor += spec.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return encoded, cursor


def _write_array(stream: Any, spec: _OutputSpec, encoded: _EncodedTensor) -> None:
    array = np.ascontiguousarray(encoded.data)
    if encoded.dtype_name != spec.dtype or tuple(array.shape) != spec.shape:
        raise Glm5NextConversionError(
            f"producer changed {spec.name}: expected {spec.dtype}{spec.shape}, found {encoded.dtype_name}{tuple(array.shape)}"
        )
    expected_dtype = _SOURCE_DTYPES[spec.dtype][0]
    if array.dtype != expected_dtype:
        raise Glm5NextConversionError(
            f"producer dtype mismatch for {spec.name}: {array.dtype} != {expected_dtype}"
        )
    raw = memoryview(array).cast("B")
    for start in range(0, len(raw), 8 * 1024 * 1024):
        stream.write(raw[start : start + 8 * 1024 * 1024])


def _cast_router_f32(encoded: _EncodedTensor) -> _EncodedTensor:
    if encoded.dtype_name == "F32":
        return encoded
    if encoded.dtype_name == "BF16":
        import ml_dtypes

        value = encoded.data.view(ml_dtypes.bfloat16).astype(np.float32)
    else:
        value = encoded.data.astype(np.float32)
    return _EncodedTensor(value, "F32")


def _normal_items(
    names: Sequence[str],
    catalog: Mapping[str, TensorHeader],
    bits: int,
    group_size: int,
) -> list[tuple[str, list[_OutputSpec]]]:
    items: list[tuple[str, list[_OutputSpec]]] = []
    for name in sorted(names):
        if name.endswith("_scale_inv") or _EXPERT_RE.match(name):
            continue
        header = catalog[name]
        if header.dtype.startswith("F8_"):
            if name + "_scale_inv" not in catalog:
                raise Glm5NextConversionError(f"unpaired FP8 tensor: {name}")
            if _must_remain_dense(name):
                raise Glm5NextConversionError(
                    f"precision-sensitive tensor may not be source FP8: {name}"
                )
        specs = (
            _quant_specs(name, header, bits, group_size)
            if _eligible(name, header, group_size)
            else [_dense_spec(name, header)]
        )
        items.append((name, specs))
    return items


def _write_normal_shard(
    path: Path,
    items: Sequence[tuple[str, list[_OutputSpec]]],
    *,
    source: Path,
    catalog: Mapping[str, TensorHeader],
    bits: int,
    group_size: int,
    quantizer: Quantizer,
    stats: ConversionStats,
) -> tuple[str, list[str], int]:
    specs = [spec for _, item_specs in items for spec in item_specs]
    if not specs:
        raise Glm5NextConversionError(f"refusing to write empty shard: {path.name}")
    header_bytes, tensor_bytes = _encoded_header(specs)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb", buffering=0) as stream:
            stream.write(struct.pack("<Q", len(header_bytes)))
            stream.write(header_bytes)
            for name, item_specs in items:
                header = catalog[name]
                if header.dtype.startswith("F8_"):
                    source_tensor = _EncodedTensor(
                        _dequantize_pair(
                            source, header, catalog[name + "_scale_inv"], stats
                        ),
                        "F32",
                    )
                else:
                    source_tensor = _read_tensor(source, header)
                    stats.source_tensors_read += 1
                    stats.max_source_tensor_bytes = max(
                        stats.max_source_tensor_bytes, source_tensor.nbytes
                    )
                if len(item_specs) == 3:
                    q, scales, biases = quantizer(
                        source_tensor.data, group_size, bits, source_tensor.dtype_name
                    )
                    produced = (
                        _EncodedTensor(np.asarray(q), "U32"),
                        _EncodedTensor(np.asarray(scales), item_specs[1].dtype),
                        _EncodedTensor(np.asarray(biases), item_specs[2].dtype),
                    )
                else:
                    produced = (
                        _cast_router_f32(source_tensor)
                        if _router(name)
                        else source_tensor,
                    )
                for spec, tensor in zip(item_specs, produced, strict=True):
                    _write_array(stream, spec, tensor)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path), [spec.name for spec in specs], tensor_bytes


def _expert_groups(
    catalog: Mapping[str, TensorHeader],
) -> dict[tuple[str, str], dict[int, str]]:
    groups: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for name in catalog:
        match = _EXPERT_RE.match(name)
        if match is None:
            continue
        key, expert = (
            (match.group("prefix"), match.group("projection")),
            int(match.group("expert")),
        )
        if expert in groups[key]:
            raise Glm5NextConversionError(f"duplicate expert {expert} in {key[0]}")
        groups[key][expert] = name
    expected = set(range(OFFICIAL_EXPERTS))
    for (prefix, projection), names in groups.items():
        if set(names) != expected:
            raise Glm5NextConversionError(
                f"{prefix}.{projection} must contain exactly experts 0..287"
            )
        if len({catalog[name].shape for name in names.values()}) != 1:
            raise Glm5NextConversionError(
                f"expert shapes differ in {prefix}.{projection}"
            )
        for name in names.values():
            if (
                not catalog[name].dtype.startswith("F8_")
                or name + "_scale_inv" not in catalog
            ):
                raise Glm5NextConversionError(f"expert is not paired block-FP8: {name}")
    by_prefix: dict[str, set[str]] = defaultdict(set)
    for prefix, projection in groups:
        by_prefix[prefix].add(projection)
    for prefix, projections in by_prefix.items():
        if projections != {"gate_proj", "up_proj", "down_proj"}:
            raise Glm5NextConversionError(f"incomplete expert projections at {prefix}")
    return dict(groups)


def _validate_fp8_layout(catalog: Mapping[str, TensorHeader], group_size: int) -> None:
    """Validate every official block grid before creating destination state."""
    for name, weight in catalog.items():
        if not weight.dtype.startswith("F8_"):
            continue
        if len(weight.shape) < 2:
            raise Glm5NextConversionError(
                f"block-FP8 tensor must have rank >= 2: {name}"
            )
        scale_name = name + "_scale_inv"
        scale = catalog.get(scale_name)
        expected = (
            *weight.shape[:-2],
            (weight.shape[-2] + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE,
            (weight.shape[-1] + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE,
        )
        if (
            scale is None
            or scale.shape != expected
            or scale.dtype not in {"BF16", "F16", "F32", "F64"}
        ):
            found = None if scale is None else (scale.dtype, scale.shape)
            raise Glm5NextConversionError(
                f"invalid block-128 inverse-scale layout for {name}: "
                f"expected floating {expected}, found {found}"
            )
        if _EXPERT_RE.match(name) and weight.shape[-1] % group_size:
            raise Glm5NextConversionError(f"expert is not affine-group aligned: {name}")


def _expert_target(prefix: str, projection: str) -> str:
    return _target_name(f"{prefix}.switch_mlp.{projection}.weight")


def _write_expert_pack(
    path: Path,
    prefix: str,
    projection: str,
    names: Mapping[int, str],
    *,
    source: Path,
    catalog: Mapping[str, TensorHeader],
    bits: int,
    group_size: int,
    quantizer: Quantizer,
    stats: ConversionStats,
) -> tuple[str, list[str], int]:
    first = catalog[names[0]]
    if first.shape[-1] % group_size:
        raise Glm5NextConversionError(f"expert is not group aligned: {names[0]}")
    base = _expert_target(prefix, projection).removesuffix(".weight")
    q_shape = (OFFICIAL_EXPERTS, *first.shape[:-1], first.shape[-1] * bits // 32)
    group_shape = (OFFICIAL_EXPERTS, *first.shape[:-1], first.shape[-1] // group_size)
    specs = [
        _OutputSpec(base + ".weight", "U32", q_shape),
        _OutputSpec(base + ".scales", "F32", group_shape),
        _OutputSpec(base + ".biases", "F32", group_shape),
    ]
    header_bytes, tensor_bytes = _encoded_header(specs)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    scale_tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.scales.tmp")
    bias_tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.biases.tmp")
    try:
        with (
            temporary.open("xb", buffering=0) as output,
            scale_tmp.open("xb", buffering=0) as scale_stream,
            bias_tmp.open("xb", buffering=0) as bias_stream,
        ):
            output.write(struct.pack("<Q", len(header_bytes)))
            output.write(header_bytes)
            one_q = _OutputSpec(specs[0].name, "U32", q_shape[1:])
            one_scale = _OutputSpec(specs[1].name, "F32", group_shape[1:])
            one_bias = _OutputSpec(specs[2].name, "F32", group_shape[1:])
            for expert in range(OFFICIAL_EXPERTS):
                name = names[expert]
                dense = _dequantize_pair(
                    source, catalog[name], catalog[name + "_scale_inv"], stats
                )
                q, scales, biases = quantizer(dense, group_size, bits, "F32")
                _write_array(output, one_q, _EncodedTensor(np.asarray(q), "U32"))
                _write_array(
                    scale_stream, one_scale, _EncodedTensor(np.asarray(scales), "F32")
                )
                _write_array(
                    bias_stream, one_bias, _EncodedTensor(np.asarray(biases), "F32")
                )
                stats.expert_tensors_packed += 1
            for sidecar in (scale_stream, bias_stream):
                sidecar.flush()
                os.fsync(sidecar.fileno())
            for side_path in (scale_tmp, bias_tmp):
                with side_path.open("rb", buffering=0) as side:
                    while chunk := side.read(8 * 1024 * 1024):
                        output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        scale_tmp.unlink(missing_ok=True)
        bias_tmp.unlink(missing_ok=True)
    return _sha256(path), [spec.name for spec in specs], tensor_bytes


def _copy_metadata(
    source: Path, output: Path, config: Mapping[str, Any], bits: int, group_size: int
) -> None:
    for item in source.iterdir():
        if not item.is_file() or item.name not in _COPY_FILES:
            continue
        destination = output / item.name
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copy2(item, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    converted = dict(config)
    converted.pop("quantization_config", None)
    converted["quantization"] = {
        "bits": bits,
        "group_size": group_size,
        "mode": "affine",
    }
    converted["glm5_next_artifact"] = {
        "layout": LAYOUT_VERSION,
        "multimodal": True,
        "vision_layers": 24,
        "main_layers": 45,
        "mtp_layer": 45,
        "experts": OFFICIAL_EXPERTS,
        "fp32_router": True,
    }
    _atomic_bytes(output / "config.json", _json_bytes(converted))


def _strict_abi_validation(catalog: Mapping[str, TensorHeader]) -> None:
    """Exercise converter-facing validators without allocating model arrays."""
    from .kda import validate_kda_weights
    from .mhc import validate_mhc_weights
    from .moe import sanitize_moe_weights
    from .mtp import sanitize_mtp_weights
    from .vision import sanitize_vision_weights

    validate_kda_weights(catalog)
    validate_mhc_weights(catalog)

    @dataclass(frozen=True)
    class _Virtual:
        shape: tuple[int, ...]
        dtype: str

    def stack(values: list[Any]) -> _Virtual:
        if not values or len({tuple(value.shape) for value in values}) != 1:
            raise Glm5NextConversionError("MoE virtual stack shape changed")
        return _Virtual((len(values), *values[0].shape), values[0].dtype)

    sanitized = sanitize_moe_weights(catalog, stack_fn=stack)
    sanitized = sanitize_mtp_weights(sanitized)
    sanitize_vision_weights(sanitized)
    from .dsa import ALL_DSA_LAYERS, Glm5NextDsaConfig, dsa_weight_shapes

    shapes = dsa_weight_shapes(Glm5NextDsaConfig.official())
    for layer in ALL_DSA_LAYERS:
        prefix = f"model.language_model.layers.{layer}.self_attn."
        actual = {
            name[len(prefix) :].removesuffix("_scale_inv")
            for name in catalog
            if name.startswith(prefix)
        }
        if actual != set(shapes):
            raise Glm5NextConversionError(
                f"DSA weight names changed at layer {layer}: "
                f"missing={sorted(set(shapes) - actual)}, "
                f"extra={sorted(actual - set(shapes))}"
            )
        for suffix, expected in shapes.items():
            name = prefix + suffix
            if name not in catalog or catalog[name].shape != expected:
                raise Glm5NextConversionError(f"DSA weight ABI changed: {name}")


def _projected_bytes(
    catalog: Mapping[str, TensorHeader], *, bits: int, group_size: int
) -> int:
    total = 0
    for name, header in catalog.items():
        if name.endswith("_scale_inv") or _EXPERT_RE.match(name):
            continue
        if _eligible(name, header, group_size):
            total += sum(
                spec.nbytes for spec in _quant_specs(name, header, bits, group_size)
            )
        else:
            total += _dense_spec(name, header).nbytes
    for experts in _expert_groups(catalog).values():
        header = catalog[experts[0]]
        elements = OFFICIAL_EXPERTS * _elements(header.shape)
        groups = (
            OFFICIAL_EXPERTS
            * _elements(header.shape[:-1])
            * (header.shape[-1] // group_size)
        )
        total += elements * bits // 8 + groups * 8
    return total


def build_conversion_plan(
    source: Path | str,
    destination: Path | str,
    *,
    source_revision: str = OFFICIAL_REVISION,
    bits: int = 8,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> Glm5NextConversionPlan:
    source = Path(source)
    contract = validate_source_contract(source, source_revision=source_revision)
    catalog, _, _ = _catalog(source)
    payload = sum(item.source_bytes for item in catalog.values())
    if payload != contract.tensor_bytes:
        raise Glm5NextContractError(
            f"header payload size changed: expected {contract.tensor_bytes}, found {payload}"
        )
    bf16 = sum(
        _elements(item.shape) * 2 if item.dtype in _FLOAT_DTYPES else item.source_bytes
        for item in catalog.values()
    )
    return Glm5NextConversionPlan(
        source,
        Path(destination),
        contract,
        payload,
        bf16,
        max(item.source_bytes for item in catalog.values()),
        bits=bits,
        group_size=group_size,
        projected_output_bytes=_projected_bytes(
            catalog, bits=bits, group_size=group_size
        ),
    )


def _record_valid(output: Path, record: Mapping[str, Any]) -> bool:
    filename = record.get("file")
    return (
        isinstance(filename, str)
        and (output / filename).is_file()
        and _sha256(output / filename) == record.get("sha256")
    )


def _expert_source_identity(
    names: Mapping[int, str],
    catalog: Mapping[str, TensorHeader],
    source_shas: Mapping[str, str],
) -> dict[str, str]:
    shards = {catalog[name].shard for name in names.values()} | {
        catalog[name + "_scale_inv"].shard for name in names.values()
    }
    return {shard: source_shas[shard] for shard in sorted(shards)}


def convert_glm53_flash(
    source: Path | str,
    destination: Path | str,
    *,
    source_revision: str = OFFICIAL_REVISION,
    bits: int = 8,
    group_size: int = DEFAULT_GROUP_SIZE,
    quantizer: Quantizer = _mlx_affine,
    validate_official: bool = True,
) -> ConversionResult:
    """Create a resumable multimodal Q8 (default) or Q4 MLX artifact."""
    if bits not in {4, 8}:
        raise Glm5NextConversionError("bits must be 8 or 4")
    if group_size <= 0 or group_size % 32:
        raise Glm5NextConversionError("group_size must be a positive multiple of 32")
    source_path, output = Path(source).resolve(), Path(destination).resolve()
    if source_path == output or source_path in output.parents:
        raise Glm5NextConversionError("destination must be outside the source tree")
    if validate_official:
        validate_source_contract(source_path, source_revision=source_revision)
    config = _load_json(source_path / "config.json")
    catalog, shard_names, weight_map = _catalog(source_path)
    _validate_fp8_layout(catalog, group_size)
    if validate_official:
        _strict_abi_validation(catalog)
    groups = _expert_groups(catalog)
    if not groups:
        raise Glm5NextConversionError("checkpoint contains no routed expert families")
    for name in catalog:
        if name.endswith("_scale_inv"):
            weight_name = name.removesuffix("_scale_inv")
            if weight_name not in catalog or not catalog[weight_name].dtype.startswith(
                "F8_"
            ):
                raise Glm5NextConversionError(f"unmapped inverse scale tensor: {name}")
        elif not name.startswith(_ALLOWED_ROOTS):
            raise Glm5NextConversionError(f"unmapped source tensor: {name}")

    # The index SHA fixes tensor placement and metadata; per-shard hashes also
    # bind resume records to payload bytes.  Hash each 62-shard source file
    # once, then reuse those identities for every cross-shard expert pack.
    source_shas = {
        shard: _sha256(_safe_source_file(source_path, shard)) for shard in shard_names
    }

    output.mkdir(parents=True, exist_ok=True)
    manifest_path, index_sha = output / MANIFEST_NAME, _sha256(source_path / INDEX_NAME)
    identity = {
        "converter_version": CONVERTER_VERSION,
        "layout_version": LAYOUT_VERSION,
        "source_revision": source_revision,
        "source_index_sha256": index_sha,
        "source_shard_count": len(shard_names),
        "bits": bits,
        "group_size": group_size,
        "mode": "affine",
        "multimodal": True,
        "vision_layers": 24,
        "main_layers": 45,
        "mtp_layer": 45,
        "experts": OFFICIAL_EXPERTS,
    }
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        for key, expected in identity.items():
            if manifest.get(key) != expected:
                raise Glm5NextConversionError(
                    f"existing conversion manifest mismatch for {key}; use a fresh destination"
                )
        records = manifest.get("units")
        if not isinstance(records, dict):
            raise Glm5NextConversionError("conversion manifest units are invalid")
    else:
        records: dict[str, Any] = {}
        manifest = {**identity, "source": str(source_path), "units": records}
        _atomic_bytes(manifest_path, _json_bytes(manifest))

    # A rerun may be repairing a checksum-invalid unit from an otherwise
    # complete artifact.  Never leave a stale truthful-looking completion bit
    # while that repair is in progress.
    manifest["complete"] = False
    manifest.pop("output_index_sha256", None)
    _atomic_bytes(manifest_path, _json_bytes(manifest))

    stats = ConversionStats()
    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        names_by_shard[shard].append(name)

    def complete(unit: str, record: dict[str, Any]) -> None:
        records[unit] = record
        manifest["units"] = records
        _atomic_bytes(manifest_path, _json_bytes(manifest))
        stats.units_completed += 1

    normal_unit_count = 0
    for ordinal, shard_name in enumerate(shard_names, start=1):
        items = _normal_items(names_by_shard[shard_name], catalog, bits, group_size)
        if not items:
            continue
        normal_unit_count += 1
        unit = "source:" + shard_name
        old = records.get(unit)
        if (
            isinstance(old, dict)
            and old.get("source_sha256") == source_shas[shard_name]
            and _record_valid(output, old)
        ):
            stats.units_resumed += 1
            continue
        filename = f"model-{ordinal:05d}-of-{len(shard_names):05d}.safetensors"
        checksum, tensors, tensor_bytes = _write_normal_shard(
            output / filename,
            items,
            source=source_path,
            catalog=catalog,
            bits=bits,
            group_size=group_size,
            quantizer=quantizer,
            stats=stats,
        )
        complete(
            unit,
            {
                "file": filename,
                "sha256": checksum,
                "source_sha256": source_shas[shard_name],
                "tensors": tensors,
                "tensor_bytes": tensor_bytes,
            },
        )

    for (prefix, projection), names in sorted(groups.items()):
        unit = f"experts:{prefix}:{projection}"
        old = records.get(unit)
        expert_sources = _expert_source_identity(names, catalog, source_shas)
        if (
            isinstance(old, dict)
            and old.get("source_shards_sha256") == expert_sources
            and _record_valid(output, old)
        ):
            stats.units_resumed += 1
            continue
        layer_match = _LAYER_RE.match(prefix + ".")
        layer = int(layer_match.group(1)) if layer_match else -1
        filename = f"experts-layer-{layer:02d}-{projection}.safetensors"
        checksum, tensors, tensor_bytes = _write_expert_pack(
            output / filename,
            prefix,
            projection,
            names,
            source=source_path,
            catalog=catalog,
            bits=bits,
            group_size=group_size,
            quantizer=quantizer,
            stats=stats,
        )
        complete(
            unit,
            {
                "file": filename,
                "sha256": checksum,
                "source_shards_sha256": expert_sources,
                "tensors": tensors,
                "tensor_bytes": tensor_bytes,
            },
        )

    if len(records) != normal_unit_count + len(groups):
        raise Glm5NextConversionError("conversion ended without every output unit")
    weight_out: dict[str, str] = {}
    total_size = 0
    for record in records.values():
        for name in record["tensors"]:
            if name in weight_out:
                raise Glm5NextConversionError(
                    f"duplicate indexed output tensor: {name}"
                )
            weight_out[name] = record["file"]
        total_size += int(record["tensor_bytes"])

    source_vision = {
        n
        for n in catalog
        if n.startswith("model.visual.") and not n.endswith("_scale_inv")
    }
    source_mtp = {
        n
        for n in catalog
        if n.startswith("model.language_model.layers.45.")
        and not n.endswith("_scale_inv")
        and _EXPERT_RE.match(n) is None
    }
    for name in source_vision | source_mtp:
        target = _target_name(name)
        if target not in weight_out:
            raise Glm5NextConversionError(f"conversion dropped tensor: {name}")
    if validate_official:
        vision_blocks = {
            int(n.split(".")[3])
            for n in source_vision
            if n.startswith("model.visual.blocks.") and n.split(".")[3].isdigit()
        }
        if vision_blocks != set(range(24)):
            raise Glm5NextConversionError("vision output must preserve blocks 0..23")

    index = {
        "metadata": {
            "format": "mlx",
            "total_size": total_size,
            "glm5_next_layout": LAYOUT_VERSION,
            "glm5_next_bits": bits,
            "glm5_next_group_size": group_size,
            "glm5_next_multimodal": True,
            "glm5_next_experts": OFFICIAL_EXPERTS,
            "glm5_next_source_index_sha256": index_sha,
        },
        "weight_map": weight_out,
    }
    index_path = output / INDEX_NAME
    _atomic_bytes(index_path, _json_bytes(index))
    _copy_metadata(source_path, output, config, bits, group_size)
    projected = _projected_bytes(catalog, bits=bits, group_size=group_size)
    manifest["complete"] = True
    manifest["projected_output_bytes"] = projected
    manifest["output_index_sha256"] = _sha256(index_path)
    _atomic_bytes(manifest_path, _json_bytes(manifest))
    return ConversionResult(output, index_path, manifest_path, bits, projected, stats)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream official GLM-5.3-Flash FP8 into multimodal MLX Q8/Q4"
    )
    parser.add_argument("source", type=Path, help="official 62-shard model directory")
    parser.add_argument("destination", type=Path, help="new MLX artifact directory")
    parser.add_argument(
        "--bits",
        type=int,
        choices=(8, 4),
        default=8,
        help="affine weight precision (default: Q8)",
    )
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--source-revision", default=OFFICIAL_REVISION)
    args = parser.parse_args(argv)
    result = convert_glm53_flash(
        args.source,
        args.destination,
        source_revision=args.source_revision,
        bits=args.bits,
        group_size=args.group_size,
    )
    print(
        f"{result.output_dir} ({result.projected_output_bytes / (1024**3):.1f} GiB projected, Q{result.bits})"
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ConversionResult",
    "ConversionStats",
    "Glm5NextConversionError",
    "Glm5NextConversionPlan",
    "Glm5NextUnsupportedMathError",
    "TensorHeader",
    "build_conversion_plan",
    "convert_glm53_flash",
    "dequantize_fp8_blocks",
    "iter_tensor_headers",
]
