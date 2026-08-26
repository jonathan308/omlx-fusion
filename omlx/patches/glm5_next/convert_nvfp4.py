# SPDX-License-Identifier: Apache-2.0
"""Streaming official GLM-5.3 block-FP8 to exact ModelOpt NVFP4.

The NVFP4 math and its licensing/source attribution live in :mod:`.nvfp4`.
This converter deliberately shares only the read-only safetensors catalogue
helpers with :mod:`.convert`; it neither calls nor modifies the active affine
Q4/Q8 conversion path.

Only the dominant routed experts, shared experts, and the first three dense
MLPs are quantized.  KDA/DSA attention, mHC, router, normalization,
convolution, embeddings/LM head, vision, and layer-45 MTP tensors retain their
source precision and names.  Every tensor is processed independently and a
routed bank holds at most one dequantized expert at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from .contract import OFFICIAL_REVISION, validate_source_contract
from .convert import (
    INDEX_NAME,
    TensorHeader,
    _atomic_bytes,
    _catalog,
    _decode_scale,
    _elements,
    _json_bytes,
    _load_json,
    _read_tensor,
    _safe_source_file,
    _sha256,
    _strict_abi_validation,
    dequantize_fp8_blocks,
)
from .nvfp4 import (
    NVFP4_GROUP_SIZE,
    NVFP4_LAYOUT,
    NVFP4Tensor,
    quantize_modelopt_nvfp4,
)

CONVERTER_VERSION: Final = 1
MANIFEST_NAME: Final = "conversion-manifest-nvfp4.json"
OFFICIAL_EXPERTS: Final = 288
_FLOAT_DTYPES: Final = frozenset({"F8_E4M3", "F8_E4M3FN", "F16", "BF16", "F32", "F64"})
_DTYPE_BYTES: Final = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_NUMPY_DTYPES: Final = {
    "BOOL": np.dtype("?"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "F8_E4M3": np.dtype("u1"),
    "F8_E4M3FN": np.dtype("u1"),
    "I16": np.dtype("<i2"),
    "U16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "F32": np.dtype("<f4"),
    "I64": np.dtype("<i8"),
    "U64": np.dtype("<u8"),
    "F64": np.dtype("<f8"),
}
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
_EXPERT_RE = re.compile(
    r"^(?P<prefix>model\.language_model\.layers\.(?P<layer>\d+)\.mlp)\."
    r"experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_DENSE_MLP_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>[0-2])\.mlp\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_SHARED_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.shared_experts\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


class Glm5NextNVFP4ConversionError(ValueError):
    """The source/destination violates the fail-closed NVFP4 contract."""


@dataclass(slots=True)
class NVFP4ConversionStats:
    units_completed: int = 0
    units_resumed: int = 0
    source_tensors_read: int = 0
    expert_tensors_packed: int = 0
    max_source_tensor_bytes: int = 0
    max_dequantized_tensor_bytes: int = 0


@dataclass(frozen=True, slots=True)
class NVFP4ConversionPlan:
    source: Path
    destination: Path
    source_payload_bytes: int
    projected_output_bytes: int
    quantized_parameter_count: int
    preserved_parameter_count: int


@dataclass(frozen=True, slots=True)
class NVFP4ConversionResult:
    output_dir: Path
    index_path: Path
    manifest_path: Path
    projected_output_bytes: int
    stats: NVFP4ConversionStats = field(compare=False)


@dataclass(frozen=True, slots=True)
class _Spec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return _elements(self.shape) * _DTYPE_BYTES[self.dtype]


Quantizer = Callable[[np.ndarray], NVFP4Tensor]


def _is_expert(name: str) -> bool:
    return _EXPERT_RE.match(name) is not None


def _is_mtp_expert(name: str) -> bool:
    match = _EXPERT_RE.match(name)
    return match is not None and int(match.group("layer")) == 45


def _is_main_expert(name: str) -> bool:
    return _is_expert(name) and not _is_mtp_expert(name)


def _eligible_dense(name: str, header: TensorHeader) -> bool:
    """Select only text MLP matrices whose runtime wrappers are installed."""

    return (
        header.dtype in _FLOAT_DTYPES
        and len(header.shape) >= 2
        and header.shape[-1] % NVFP4_GROUP_SIZE == 0
        and (
            _DENSE_MLP_RE.match(name) is not None
            or _SHARED_EXPERT_RE.match(name) is not None
        )
    )


def _quant_specs(name: str, shape: tuple[int, ...]) -> tuple[_Spec, _Spec, _Spec]:
    if shape[-1] % NVFP4_GROUP_SIZE:
        raise Glm5NextNVFP4ConversionError(f"NVFP4 tensor is not group aligned: {name}")
    base = name.removesuffix(".weight")
    return (
        _Spec(base + ".weight", "U32", (*shape[:-1], shape[-1] // 8)),
        _Spec(base + ".scales", "U8", (*shape[:-1], shape[-1] // 16)),
        _Spec(base + ".global_scale", "F32", ()),
    )


def _expert_specs(prefix: str, projection: str, shape: tuple[int, ...]):
    base = f"{prefix}.switch_mlp.{projection}"
    return (
        _Spec(
            base + ".weight",
            "U32",
            (OFFICIAL_EXPERTS, *shape[:-1], shape[-1] // 8),
        ),
        _Spec(
            base + ".scales",
            "U8",
            (OFFICIAL_EXPERTS, *shape[:-1], shape[-1] // 16),
        ),
        _Spec(base + ".global_scale", "F32", (OFFICIAL_EXPERTS,)),
    )


def _encoded_header(specs: Sequence[_Spec]) -> tuple[bytes, int]:
    cursor = 0
    header: dict[str, Any] = {"__metadata__": {"format": "mlx"}}
    for spec in specs:
        if spec.name in header:
            raise Glm5NextNVFP4ConversionError(f"duplicate output tensor: {spec.name}")
        header[spec.name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [cursor, cursor + spec.nbytes],
        }
        cursor += spec.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return encoded, cursor


def _write_array(stream: Any, spec: _Spec, array: np.ndarray) -> None:
    # ``np.ascontiguousarray`` promotes a scalar from shape ``()`` to ``(1,)``;
    # preserve the safetensors scalar ABI for ModelOpt's global scale.
    value = np.asarray(array)
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    if value.dtype != _NUMPY_DTYPES[spec.dtype] or tuple(value.shape) != spec.shape:
        raise Glm5NextNVFP4ConversionError(
            f"producer changed {spec.name}: expected {spec.dtype}{spec.shape}, "
            f"found {value.dtype}{tuple(value.shape)}"
        )
    raw = memoryview(value).cast("B")
    for start in range(0, len(raw), 8 * 1024 * 1024):
        stream.write(raw[start : start + 8 * 1024 * 1024])


def _read_dense(
    source: Path,
    header: TensorHeader,
    catalog: Mapping[str, TensorHeader],
    stats: NVFP4ConversionStats,
) -> np.ndarray:
    if header.dtype.startswith("F8_"):
        scale_name = header.name + "_scale_inv"
        scale = catalog.get(scale_name)
        if scale is None:
            raise Glm5NextNVFP4ConversionError(
                f"block-FP8 tensor has no inverse scale: {header.name}"
            )
        codes = _read_tensor(source, header)
        inverse = _read_tensor(source, scale)
        stats.source_tensors_read += 2
        stats.max_source_tensor_bytes = max(
            stats.max_source_tensor_bytes, int(codes.nbytes), int(inverse.nbytes)
        )
        dense = dequantize_fp8_blocks(codes.data, _decode_scale(inverse))
        stats.max_dequantized_tensor_bytes = max(
            stats.max_dequantized_tensor_bytes, int(dense.nbytes)
        )
        return dense

    encoded = _read_tensor(source, header)
    stats.source_tensors_read += 1
    stats.max_source_tensor_bytes = max(
        stats.max_source_tensor_bytes, int(encoded.nbytes)
    )
    if header.dtype == "BF16":
        try:
            import ml_dtypes
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("ml_dtypes is required to decode BF16") from exc
        dense = encoded.data.view(ml_dtypes.bfloat16).astype(np.float32)
    else:
        dense = encoded.data.astype(np.float32)
    stats.max_dequantized_tensor_bytes = max(
        stats.max_dequantized_tensor_bytes, int(dense.nbytes)
    )
    return dense


def _validate_quantized(value: NVFP4Tensor, specs: Sequence[_Spec]) -> None:
    arrays = (value.weight, value.scales, value.global_scale)
    for spec, array in zip(specs, arrays, strict=True):
        expected_dtype = _NUMPY_DTYPES[spec.dtype]
        if (
            np.asarray(array).dtype != expected_dtype
            or tuple(np.asarray(array).shape) != spec.shape
        ):
            raise Glm5NextNVFP4ConversionError(
                f"NVFP4 quantizer ABI changed for {spec.name}: "
                f"{np.asarray(array).dtype}{np.asarray(array).shape}"
            )


def _normal_items(
    names: Sequence[str], catalog: Mapping[str, TensorHeader]
) -> list[tuple[str, tuple[_Spec, ...]]]:
    items: list[tuple[str, tuple[_Spec, ...]]] = []
    for name in sorted(names):
        is_mtp_scale = name.endswith("_scale_inv") and _is_mtp_expert(
            name.removesuffix("_scale_inv")
        )
        if (name.endswith("_scale_inv") and not is_mtp_scale) or _is_main_expert(name):
            continue
        header = catalog[name]
        if _eligible_dense(name, header):
            specs = _quant_specs(name, header.shape)
        else:
            if header.dtype.startswith("F8_") and not _is_mtp_expert(name):
                raise Glm5NextNVFP4ConversionError(
                    "precision-preserved tensor unexpectedly uses source FP8 and "
                    f"cannot be copied without a block-FP8 runtime: {name}"
                )
            specs = (_Spec(name, header.dtype, header.shape),)
        items.append((name, specs))
    return items


def _write_normal_shard(
    path: Path,
    items: Sequence[tuple[str, tuple[_Spec, ...]]],
    *,
    source: Path,
    catalog: Mapping[str, TensorHeader],
    quantizer: Quantizer,
    stats: NVFP4ConversionStats,
) -> tuple[str, list[str], int]:
    specs = [spec for _, group in items for spec in group]
    header_bytes, tensor_bytes = _encoded_header(specs)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb", buffering=0) as stream:
            stream.write(struct.pack("<Q", len(header_bytes)))
            stream.write(header_bytes)
            for name, output_specs in items:
                header = catalog[name]
                if len(output_specs) == 1:
                    value = _read_tensor(source, header)
                    stats.source_tensors_read += 1
                    stats.max_source_tensor_bytes = max(
                        stats.max_source_tensor_bytes, int(value.nbytes)
                    )
                    _write_array(stream, output_specs[0], value.data)
                    continue
                quantized = quantizer(_read_dense(source, header, catalog, stats))
                _validate_quantized(quantized, output_specs)
                for spec, value in zip(
                    output_specs,
                    (quantized.weight, quantized.scales, quantized.global_scale),
                    strict=True,
                ):
                    _write_array(stream, spec, value)
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
        if match is None or int(match.group("layer")) == 45:
            continue
        key = (match.group("prefix"), match.group("projection"))
        expert = int(match.group("expert"))
        if expert in groups[key]:
            raise Glm5NextNVFP4ConversionError(f"duplicate expert {expert}: {key}")
        groups[key][expert] = name
    expected = set(range(OFFICIAL_EXPERTS))
    for (prefix, projection), names in groups.items():
        if set(names) != expected:
            raise Glm5NextNVFP4ConversionError(
                f"{prefix}.{projection} must contain experts 0..287"
            )
        shapes = {catalog[name].shape for name in names.values()}
        if len(shapes) != 1:
            raise Glm5NextNVFP4ConversionError(f"expert shapes differ: {prefix}")
        for name in names.values():
            header = catalog[name]
            scale = catalog.get(name + "_scale_inv")
            if (
                not header.dtype.startswith("F8_")
                or scale is None
                or header.shape[-1] % NVFP4_GROUP_SIZE
            ):
                raise Glm5NextNVFP4ConversionError(
                    f"expert is not aligned paired block-FP8: {name}"
                )
    prefixes: dict[str, set[str]] = defaultdict(set)
    for prefix, projection in groups:
        prefixes[prefix].add(projection)
    for prefix, projections in prefixes.items():
        if projections != {"gate_proj", "up_proj", "down_proj"}:
            raise Glm5NextNVFP4ConversionError(f"incomplete expert family: {prefix}")
    return dict(groups)


def _write_expert_bank(
    path: Path,
    prefix: str,
    projection: str,
    names: Mapping[int, str],
    *,
    source: Path,
    catalog: Mapping[str, TensorHeader],
    quantizer: Quantizer,
    stats: NVFP4ConversionStats,
) -> tuple[str, list[str], int]:
    shape = catalog[names[0]].shape
    specs = _expert_specs(prefix, projection, shape)
    header_bytes, tensor_bytes = _encoded_header(specs)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    scale_tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.scales.tmp")
    global_tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.global.tmp")
    one_specs = (
        _Spec(specs[0].name, "U32", specs[0].shape[1:]),
        _Spec(specs[1].name, "U8", specs[1].shape[1:]),
        _Spec(specs[2].name, "F32", ()),
    )
    try:
        with (
            temporary.open("xb", buffering=0) as weight_stream,
            scale_tmp.open("xb", buffering=0) as scale_stream,
            global_tmp.open("xb", buffering=0) as global_stream,
        ):
            weight_stream.write(struct.pack("<Q", len(header_bytes)))
            weight_stream.write(header_bytes)
            for expert in range(OFFICIAL_EXPERTS):
                name = names[expert]
                quantized = quantizer(
                    _read_dense(source, catalog[name], catalog, stats)
                )
                _validate_quantized(quantized, one_specs)
                _write_array(weight_stream, one_specs[0], quantized.weight)
                _write_array(scale_stream, one_specs[1], quantized.scales)
                _write_array(global_stream, one_specs[2], quantized.global_scale)
                stats.expert_tensors_packed += 1
            for stream in (scale_stream, global_stream):
                stream.flush()
                os.fsync(stream.fileno())
            for sidecar in (scale_tmp, global_tmp):
                with sidecar.open("rb", buffering=0) as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        weight_stream.write(chunk)
            weight_stream.flush()
            os.fsync(weight_stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        scale_tmp.unlink(missing_ok=True)
        global_tmp.unlink(missing_ok=True)
    return _sha256(path), [spec.name for spec in specs], tensor_bytes


def _projected_bytes(catalog: Mapping[str, TensorHeader]) -> tuple[int, int, int]:
    total = quantized = preserved = 0
    groups = _expert_groups(catalog)
    for name, header in catalog.items():
        is_mtp_scale = name.endswith("_scale_inv") and _is_mtp_expert(
            name.removesuffix("_scale_inv")
        )
        if (name.endswith("_scale_inv") and not is_mtp_scale) or _is_main_expert(name):
            continue
        elements = _elements(header.shape)
        if _eligible_dense(name, header):
            total += sum(spec.nbytes for spec in _quant_specs(name, header.shape))
            quantized += elements
        else:
            total += header.source_bytes
            preserved += elements
    for (prefix, projection), names in groups.items():
        shape = catalog[names[0]].shape
        total += sum(spec.nbytes for spec in _expert_specs(prefix, projection, shape))
        quantized += OFFICIAL_EXPERTS * _elements(shape)
    return total, quantized, preserved


def build_nvfp4_conversion_plan(
    source: Path | str,
    destination: Path | str,
    *,
    source_revision: str = OFFICIAL_REVISION,
    validate_official: bool = True,
) -> NVFP4ConversionPlan:
    source_path = Path(source).resolve()
    if validate_official:
        validate_source_contract(source_path, source_revision=source_revision)
    catalog, shard_names, weight_map = _catalog(source_path)
    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        names_by_shard[shard].append(name)
    # Planning is also fail-closed: exercise both expert geometry and the
    # preserved-vs-quantized policy before reporting a usable size.
    _expert_groups(catalog)
    for shard in shard_names:
        _normal_items(names_by_shard[shard], catalog)
    projected, quantized, preserved = _projected_bytes(catalog)
    return NVFP4ConversionPlan(
        source=source_path,
        destination=Path(destination).resolve(),
        source_payload_bytes=sum(item.source_bytes for item in catalog.values()),
        projected_output_bytes=projected,
        quantized_parameter_count=quantized,
        preserved_parameter_count=preserved,
    )


def _record_valid(output: Path, record: Mapping[str, Any]) -> bool:
    filename = record.get("file")
    return (
        isinstance(filename, str)
        and (output / filename).is_file()
        and _sha256(output / filename) == record.get("sha256")
    )


def _copy_metadata(source: Path, output: Path, config: Mapping[str, Any]) -> None:
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
        "bits": 4,
        "group_size": NVFP4_GROUP_SIZE,
        "mode": "nvfp4",
        "layout": NVFP4_LAYOUT,
        "modelopt_global_scale": True,
        "scope": "glm5_next_mlp",
    }
    converted["glm5_next_artifact"] = {
        "layout": NVFP4_LAYOUT,
        "source": "official-block-fp8-e4m3-block128",
        "main_layers": 45,
        "mtp_layer": 45,
        "experts": OFFICIAL_EXPERTS,
        "vision_preserved": True,
        "mtp_preserved": True,
        "sensitive_text_preserved": True,
    }
    _atomic_bytes(output / "config.json", _json_bytes(converted))


def convert_glm53_flash_nvfp4(
    source: Path | str,
    destination: Path | str,
    *,
    source_revision: str = OFFICIAL_REVISION,
    quantizer: Quantizer = quantize_modelopt_nvfp4,
    validate_official: bool = True,
) -> NVFP4ConversionResult:
    """Create or resume an exact two-level NVFP4 GLM-5.3 artifact."""

    source_path = Path(source).resolve()
    output = Path(destination).resolve()
    if source_path == output or source_path in output.parents:
        raise Glm5NextNVFP4ConversionError(
            "destination must be outside the source tree"
        )
    if validate_official:
        validate_source_contract(source_path, source_revision=source_revision)
    config = _load_json(source_path / "config.json")
    catalog, shard_names, weight_map = _catalog(source_path)
    if validate_official:
        _strict_abi_validation(catalog)
    groups = _expert_groups(catalog)
    if not groups:
        raise Glm5NextNVFP4ConversionError("checkpoint has no routed expert families")

    for name in catalog:
        if name.endswith("_scale_inv"):
            weight = catalog.get(name.removesuffix("_scale_inv"))
            if weight is None or not weight.dtype.startswith("F8_"):
                raise Glm5NextNVFP4ConversionError(f"orphan inverse scale: {name}")
        elif not name.startswith(
            ("lm_head.", "model.language_model.", "model.visual.")
        ):
            raise Glm5NextNVFP4ConversionError(f"unmapped source tensor: {name}")

    # Validate the full output policy before creating destination state.
    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        names_by_shard[shard].append(name)
    normal_by_shard = {
        shard: _normal_items(names_by_shard[shard], catalog) for shard in shard_names
    }
    projected, _, _ = _projected_bytes(catalog)
    source_shas = {
        shard: _sha256(_safe_source_file(source_path, shard)) for shard in shard_names
    }

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST_NAME
    identity = {
        "converter_version": CONVERTER_VERSION,
        "layout_version": NVFP4_LAYOUT,
        "source_revision": source_revision,
        "source_index_sha256": _sha256(source_path / INDEX_NAME),
        "source_shard_count": len(shard_names),
        "bits": 4,
        "group_size": NVFP4_GROUP_SIZE,
        "mode": "nvfp4",
        "modelopt_global_scale": True,
        "scope": "glm5_next_mlp",
        "experts": OFFICIAL_EXPERTS,
        "vision_preserved": True,
        "mtp_preserved": True,
    }
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        for key, expected in identity.items():
            if manifest.get(key) != expected:
                raise Glm5NextNVFP4ConversionError(
                    f"existing NVFP4 manifest mismatch for {key}; use a fresh destination"
                )
        records = manifest.get("units")
        if not isinstance(records, dict):
            raise Glm5NextNVFP4ConversionError("manifest units are invalid")
    else:
        records: dict[str, Any] = {}
        manifest = {**identity, "source": str(source_path), "units": records}
        _atomic_bytes(manifest_path, _json_bytes(manifest))
    manifest["complete"] = False
    manifest.pop("output_index_sha256", None)
    _atomic_bytes(manifest_path, _json_bytes(manifest))

    stats = NVFP4ConversionStats()

    def complete(unit: str, record: dict[str, Any]) -> None:
        records[unit] = record
        manifest["units"] = records
        _atomic_bytes(manifest_path, _json_bytes(manifest))
        stats.units_completed += 1

    normal_count = 0
    for ordinal, shard in enumerate(shard_names, start=1):
        items = normal_by_shard[shard]
        if not items:
            continue
        normal_count += 1
        unit = "source:" + shard
        old = records.get(unit)
        if (
            isinstance(old, dict)
            and old.get("source_sha256") == source_shas[shard]
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
            quantizer=quantizer,
            stats=stats,
        )
        complete(
            unit,
            {
                "file": filename,
                "sha256": checksum,
                "source_sha256": source_shas[shard],
                "tensors": tensors,
                "tensor_bytes": tensor_bytes,
            },
        )

    for (prefix, projection), names in sorted(groups.items()):
        unit = f"experts:{prefix}:{projection}"
        shards = sorted(
            {catalog[name].shard for name in names.values()}
            | {catalog[name + "_scale_inv"].shard for name in names.values()}
        )
        identities = {shard: source_shas[shard] for shard in shards}
        old = records.get(unit)
        if (
            isinstance(old, dict)
            and old.get("source_shards_sha256") == identities
            and _record_valid(output, old)
        ):
            stats.units_resumed += 1
            continue
        layer = int(prefix.split(".layers.", 1)[1].split(".", 1)[0])
        filename = f"experts-layer-{layer:02d}-{projection}-nvfp4.safetensors"
        checksum, tensors, tensor_bytes = _write_expert_bank(
            output / filename,
            prefix,
            projection,
            names,
            source=source_path,
            catalog=catalog,
            quantizer=quantizer,
            stats=stats,
        )
        complete(
            unit,
            {
                "file": filename,
                "sha256": checksum,
                "source_shards_sha256": identities,
                "tensors": tensors,
                "tensor_bytes": tensor_bytes,
            },
        )

    if len(records) != normal_count + len(groups):
        raise Glm5NextNVFP4ConversionError("conversion ended without every output unit")
    weight_out: dict[str, str] = {}
    total_size = 0
    for record in records.values():
        for name in record["tensors"]:
            if name in weight_out:
                raise Glm5NextNVFP4ConversionError(f"duplicate indexed tensor: {name}")
            weight_out[name] = record["file"]
        total_size += int(record["tensor_bytes"])

    source_vision = {
        name
        for name in catalog
        if name.startswith("model.visual.") and not name.endswith("_scale_inv")
    }
    source_mtp = {
        name
        for name in catalog
        if name.startswith("model.language_model.layers.45.")
        and not name.endswith("_scale_inv")
    }
    for name in source_vision | source_mtp:
        if name not in weight_out:
            raise Glm5NextNVFP4ConversionError(
                f"conversion dropped preserved tensor: {name}"
            )
    if validate_official:
        blocks = {
            int(name.split(".")[3])
            for name in source_vision
            if name.startswith("model.visual.blocks.") and name.split(".")[3].isdigit()
        }
        if blocks != set(range(24)):
            raise Glm5NextNVFP4ConversionError("vision blocks 0..23 were not preserved")

    index = {
        "metadata": {
            "format": "mlx",
            "total_size": total_size,
            "glm5_next_layout": NVFP4_LAYOUT,
            "glm5_next_bits": 4,
            "glm5_next_group_size": NVFP4_GROUP_SIZE,
            "glm5_next_modelopt_global_scale": True,
            "glm5_next_experts": OFFICIAL_EXPERTS,
            "glm5_next_source_index_sha256": identity["source_index_sha256"],
        },
        "weight_map": weight_out,
    }
    index_path = output / INDEX_NAME
    _atomic_bytes(index_path, _json_bytes(index))
    _copy_metadata(source_path, output, config)
    manifest["complete"] = True
    manifest["projected_output_bytes"] = projected
    manifest["output_index_sha256"] = _sha256(index_path)
    _atomic_bytes(manifest_path, _json_bytes(manifest))
    return NVFP4ConversionResult(output, index_path, manifest_path, projected, stats)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream official GLM-5.3-Flash block-FP8 into exact ModelOpt NVFP4"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-revision", default=OFFICIAL_REVISION)
    args = parser.parse_args(argv)
    result = convert_glm53_flash_nvfp4(
        args.source, args.destination, source_revision=args.source_revision
    )
    print(
        f"{result.output_dir} "
        f"({result.projected_output_bytes / (1024**3):.1f} GiB projected, NVFP4)"
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "Glm5NextNVFP4ConversionError",
    "MANIFEST_NAME",
    "NVFP4ConversionPlan",
    "NVFP4ConversionResult",
    "NVFP4ConversionStats",
    "build_nvfp4_conversion_plan",
    "convert_glm53_flash_nvfp4",
]
