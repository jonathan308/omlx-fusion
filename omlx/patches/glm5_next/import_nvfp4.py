# SPDX-License-Identifier: Apache-2.0
"""Losslessly import the pinned LibertAI GLM-5.3 ModelOpt NVFP4 artifact.

Source contract:

* repository ``LibertAIDAI/GLM-5.3-Flash-NVFP4``;
* revision ``9e0d74e3cef17f634e84fb8e2223707e02616290``;
* NVIDIA ModelOpt 0.45.0, weight-only NVFP4 routed experts;
* base model ``zai-org/GLM-5.3-Flash``.

The importer performs no floating-point conversion and no quantization.  A
ModelOpt expert stores two E2M1 nibbles per ``U8`` byte; MLX stores the same
byte stream viewed four bytes at a time as ``U32``.  ModelOpt's E4M3 group
scales are copied into MLX ``U8`` carriers and ``weight_scale_2`` is retained
as our per-expert ``global_scale``.  Every non-expert tensor payload is copied
byte-for-byte in bounded chunks.
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

from .contract import validate_config as validate_official_architecture
from .convert import (
    INDEX_NAME,
    TensorHeader,
    _atomic_bytes,
    _catalog,
    _json_bytes,
    _load_json,
    _read_tensor,
    _safe_source_file,
    _sha256,
)
from .nvfp4 import NVFP4_LAYOUT

SOURCE_REPOSITORY: Final = "LibertAIDAI/GLM-5.3-Flash-NVFP4"
SOURCE_REVISION: Final = "9e0d74e3cef17f634e84fb8e2223707e02616290"
SOURCE_BASE_MODEL: Final = "zai-org/GLM-5.3-Flash"
SOURCE_SHARDS: Final = 120
SOURCE_TENSORS: Final = 113_074
SOURCE_TENSOR_BYTES: Final = 194_644_803_576
SOURCE_EXPERT_MATRICES: Final = 37_152
SOURCE_NONEXPERT_TENSORS: Final = 1_618
SOURCE_EXPERT_LAYERS: Final = tuple(range(3, 46))
SOURCE_EXPERTS: Final = 288
SOURCE_PROJECTIONS: Final = ("gate_proj", "up_proj", "down_proj")
MANIFEST_NAME: Final = "import-manifest-nvfp4.json"
IMPORTER_VERSION: Final = 1
OUTPUT_SCOPE: Final = "glm5_next_routed_experts"

_IGNORE: Final = (
    "lm_head",
    "model.language_model.embed_tokens",
    "*.self_attn.in_proj_qkvbfg_a",
    "*.self_attn.q_proj",
    "*.self_attn.k_proj",
    "*.self_attn.v_proj",
    "*.self_attn.b_proj",
    "*.self_attn.f_a_proj",
    "*.self_attn.g_a_proj",
    "*.self_attn.f_b_proj",
    "*.self_attn.g_b_proj",
    "*.self_attn.fused_qkv_a_proj",
    "*.self_attn.q_a_proj",
    "*.self_attn.kv_a_proj_with_mqa",
    "*.self_attn.q_b_proj",
    "*.self_attn.kv_b_proj",
    "*.self_attn.indexer.wk_weights_proj",
    "*.self_attn.indexer.wk",
    "*.self_attn.indexer.weights_proj",
    "*.self_attn.indexer.wq_b",
    "*.self_attn.o_proj",
    "*.mlp.gate",
    "*.mlp.gate_up_proj",
    "*.mlp.down_proj",
    "*.mlp.shared_experts.gate_up_proj",
    "*.mlp.shared_experts.down_proj",
    "*.mlp.shared_experts.gate_proj",
    "*.mlp.shared_experts.up_proj",
    "*.mlp.gate_proj",
    "*.mlp.up_proj",
    "*.eh_proj",
    "model.visual.*",
)
_COPY_FILES: Final = frozenset(
    {
        "LICENSE",
        "README.md",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
_EXPERT_FIELD_RE = re.compile(
    r"^(?P<prefix>model\.language_model\.layers\.(?P<layer>\d+)\.mlp)\."
    r"experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<field>weight|weight_scale|weight_scale_2)$"
)
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


class Glm5NextNVFP4ImportError(ValueError):
    """The pinned source or destination violates the lossless import ABI."""


@dataclass(slots=True)
class ImportStats:
    units_completed: int = 0
    units_resumed: int = 0
    tensors_copied: int = 0
    expert_matrices_repacked: int = 0
    max_copy_chunk_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ImportPlan:
    source: Path
    destination: Path
    source_payload_bytes: int
    projected_output_bytes: int
    expert_matrices: int
    nonexpert_tensors: int


@dataclass(frozen=True, slots=True)
class ImportResult:
    output_dir: Path
    index_path: Path
    manifest_path: Path
    projected_output_bytes: int
    stats: ImportStats = field(compare=False)


@dataclass(frozen=True, slots=True)
class _Spec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count * _DTYPE_BYTES[self.dtype]


@dataclass(frozen=True, slots=True)
class _ExpertTriple:
    weight: str
    scale: str
    scale_2: str


UnitHook = Callable[[str, int], None]


def _validate_repository(repository: str, revision: str) -> None:
    if repository != SOURCE_REPOSITORY:
        raise Glm5NextNVFP4ImportError(
            f"repository must be pinned to {SOURCE_REPOSITORY}, found {repository!r}"
        )
    if revision != SOURCE_REVISION:
        raise Glm5NextNVFP4ImportError(
            f"revision must be pinned to {SOURCE_REVISION}, found {revision!r}"
        )


def _validate_readme(source: Path) -> None:
    try:
        text = (source / "README.md").read_text()
    except (FileNotFoundError, UnicodeDecodeError) as exc:
        raise Glm5NextNVFP4ImportError(
            "pinned source requires a readable README.md"
        ) from exc
    required = {
        "license: mit",
        f"base_model: {SOURCE_BASE_MODEL}",
        "base_model_relation: quantized",
        "quantized_by: LibertAIDAI",
    }
    frontmatter = text.split("---", 2)
    if len(frontmatter) < 3:
        raise Glm5NextNVFP4ImportError("README.md lacks pinned YAML frontmatter")
    lines = {line.strip() for line in frontmatter[1].splitlines()}
    missing = sorted(required - lines)
    if missing:
        raise Glm5NextNVFP4ImportError(
            "README.md base/provenance changed: " + ", ".join(missing)
        )


def _validate_quantization_config(config: Mapping[str, Any]) -> None:
    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping):
        raise Glm5NextNVFP4ImportError("config requires quantization_config")
    expected_top = {
        "quant_algo": "NVFP4",
        "kv_cache_scheme": None,
        "producer": {"name": "modelopt", "version": "0.45.0"},
        "quant_method": "modelopt",
        "ignore": list(_IGNORE),
    }
    for name, expected in expected_top.items():
        if quant.get(name) != expected:
            raise Glm5NextNVFP4ImportError(f"config.quantization_config.{name} changed")
    groups = quant.get("config_groups")
    expected_groups = {
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
    }
    if groups != expected_groups:
        raise Glm5NextNVFP4ImportError(
            "config.quantization_config.config_groups changed"
        )
    if set(quant) != {
        "config_groups",
        "quant_algo",
        "kv_cache_scheme",
        "producer",
        "quant_method",
        "ignore",
    }:
        raise Glm5NextNVFP4ImportError("quantization_config fields changed")


def _validate_architecture(config: Mapping[str, Any]) -> None:
    # Reuse the pinned official architectural validator after substituting only
    # its source-format stanza.  No architecture field is relaxed here.
    candidate = dict(config)
    candidate["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [""] * 1_509,
    }
    try:
        validate_official_architecture(candidate)
    except ValueError as exc:
        raise Glm5NextNVFP4ImportError(
            "pinned GLM-5.3 architecture/config changed"
        ) from exc


def _validate_pinned_index(
    catalog: Mapping[str, TensorHeader],
    shards: Sequence[str],
    weight_map: Mapping[str, str],
) -> None:
    if len(catalog) != SOURCE_TENSORS:
        raise Glm5NextNVFP4ImportError(
            f"expected {SOURCE_TENSORS} tensors, found {len(catalog)}"
        )
    expected_shards = {
        f"model-{ordinal:05d}-of-{SOURCE_SHARDS:05d}.safetensors"
        for ordinal in range(1, SOURCE_SHARDS + 1)
    }
    if set(shards) != expected_shards or set(weight_map.values()) != expected_shards:
        raise Glm5NextNVFP4ImportError("source shard set changed from pinned 120 files")
    payload = sum(item.source_bytes for item in catalog.values())
    if payload != SOURCE_TENSOR_BYTES:
        raise Glm5NextNVFP4ImportError(
            f"source payload changed: expected {SOURCE_TENSOR_BYTES}, found {payload}"
        )


def _expert_families(
    catalog: Mapping[str, TensorHeader], *, pinned: bool
) -> dict[tuple[str, str], dict[int, _ExpertTriple]]:
    fields: dict[tuple[str, str, int], dict[str, str]] = defaultdict(dict)
    for name in catalog:
        match = _EXPERT_FIELD_RE.match(name)
        if match is None:
            if name.endswith(("weight_scale", "weight_scale_2")):
                raise Glm5NextNVFP4ImportError(
                    f"quantization sidecar is outside routed experts: {name}"
                )
            continue
        key = (
            match.group("prefix"),
            match.group("projection"),
            int(match.group("expert")),
        )
        field_name = match.group("field")
        if field_name in fields[key]:
            raise Glm5NextNVFP4ImportError(f"duplicate expert field: {name}")
        fields[key][field_name] = name

    families: dict[tuple[str, str], dict[int, _ExpertTriple]] = defaultdict(dict)
    for (prefix, projection, expert), names in fields.items():
        if set(names) != {"weight", "weight_scale", "weight_scale_2"}:
            raise Glm5NextNVFP4ImportError(
                f"incomplete ModelOpt expert {prefix}.{expert}.{projection}"
            )
        triple = _ExpertTriple(
            names["weight"], names["weight_scale"], names["weight_scale_2"]
        )
        weight = catalog[triple.weight]
        scale = catalog[triple.scale]
        scale_2 = catalog[triple.scale_2]
        if weight.dtype != "U8" or len(weight.shape) != 2 or weight.shape[-1] % 4:
            raise Glm5NextNVFP4ImportError(
                f"invalid packed U8 expert weight: {triple.weight}"
            )
        logical_width = weight.shape[-1] * 2
        expected_scale = (*weight.shape[:-1], logical_width // 16)
        allowed_scale_dtypes = {"F8_E4M3"} if pinned else {"F8_E4M3", "F8_E4M3FN"}
        if scale.dtype not in allowed_scale_dtypes or scale.shape != expected_scale:
            raise Glm5NextNVFP4ImportError(f"invalid E4M3 group scale: {triple.scale}")
        if pinned:
            expected_weight_shape = (
                (4_096, 1_024) if projection == "down_proj" else (2_048, 2_048)
            )
            expected_pinned_scale = (
                (4_096, 128) if projection == "down_proj" else (2_048, 256)
            )
            if (
                weight.shape != expected_weight_shape
                or scale.shape != expected_pinned_scale
            ):
                raise Glm5NextNVFP4ImportError(
                    f"pinned expert carrier shape changed: {triple.weight}"
                )
        if scale_2.dtype != "F32" or scale_2.shape != ():
            raise Glm5NextNVFP4ImportError(
                f"invalid FP32 scalar scale_2: {triple.scale_2}"
            )
        families[(prefix, projection)][expert] = triple

    for (prefix, projection), experts in families.items():
        expected_experts = (
            set(range(SOURCE_EXPERTS)) if pinned else set(range(len(experts)))
        )
        if set(experts) != expected_experts:
            raise Glm5NextNVFP4ImportError(
                f"{prefix}.{projection} expert ids are not contiguous from zero"
            )
        geometries = {
            (
                catalog[item.weight].shape,
                catalog[item.scale].shape,
                catalog[item.weight].dtype,
                catalog[item.scale].dtype,
            )
            for item in experts.values()
        }
        if len(geometries) != 1:
            raise Glm5NextNVFP4ImportError(
                f"expert carrier geometry differs in {prefix}.{projection}"
            )

    projections_by_prefix: dict[str, set[str]] = defaultdict(set)
    for prefix, projection in families:
        projections_by_prefix[prefix].add(projection)
    for prefix, projections in projections_by_prefix.items():
        if projections != set(SOURCE_PROJECTIONS):
            raise Glm5NextNVFP4ImportError(f"incomplete projection family at {prefix}")

    if pinned:
        expected_prefixes = {
            f"model.language_model.layers.{layer}.mlp" for layer in SOURCE_EXPERT_LAYERS
        }
        if set(projections_by_prefix) != expected_prefixes:
            raise Glm5NextNVFP4ImportError("routed-expert layer set changed")
        matrices = sum(len(experts) for experts in families.values())
        if matrices != SOURCE_EXPERT_MATRICES:
            raise Glm5NextNVFP4ImportError(
                f"expected {SOURCE_EXPERT_MATRICES} expert matrices, found {matrices}"
            )
    if not families:
        raise Glm5NextNVFP4ImportError("source contains no ModelOpt expert families")
    return dict(families)


def _source_contract(
    source: Path,
    *,
    repository: str,
    revision: str,
    validate_pinned: bool,
) -> tuple[
    dict[str, Any],
    dict[str, TensorHeader],
    list[str],
    dict[str, str],
    dict[tuple[str, str], dict[int, _ExpertTriple]],
]:
    _validate_repository(repository, revision)
    config = _load_json(source / "config.json")
    _validate_quantization_config(config)
    if validate_pinned:
        _validate_readme(source)
        _validate_architecture(config)
    index = _load_json(source / INDEX_NAME)
    metadata = index.get("metadata")
    if not isinstance(metadata, Mapping):
        raise Glm5NextNVFP4ImportError("index metadata must be an object")
    if validate_pinned and metadata.get("total_size") != SOURCE_TENSOR_BYTES:
        raise Glm5NextNVFP4ImportError("index metadata.total_size changed")
    catalog, shards, weight_map = _catalog(source)
    if validate_pinned:
        _validate_pinned_index(catalog, shards, weight_map)
    families = _expert_families(catalog, pinned=validate_pinned)
    expert_names = {
        name
        for experts in families.values()
        for triple in experts.values()
        for name in (triple.weight, triple.scale, triple.scale_2)
    }
    if validate_pinned and len(catalog) - len(expert_names) != SOURCE_NONEXPERT_TENSORS:
        raise Glm5NextNVFP4ImportError(
            "pinned source must contain exactly 1,618 non-expert tensors"
        )
    unexpected = [
        name
        for name in catalog
        if name not in expert_names
        and not name.startswith(("lm_head.", "model.language_model.", "model.visual."))
    ]
    if unexpected:
        raise Glm5NextNVFP4ImportError(f"unmapped source tensor: {unexpected[0]}")
    return config, catalog, shards, dict(weight_map), families


def _expert_specs(
    prefix: str,
    projection: str,
    count: int,
    weight: TensorHeader,
    scale: TensorHeader,
) -> tuple[_Spec, _Spec, _Spec]:
    base = f"{prefix}.switch_mlp.{projection}"
    return (
        _Spec(
            base + ".weight",
            "U32",
            (count, *weight.shape[:-1], weight.shape[-1] // 4),
        ),
        _Spec(base + ".scales", "U8", (count, *scale.shape)),
        _Spec(base + ".global_scale", "F32", (count,)),
    )


def _encoded_header(specs: Sequence[_Spec]) -> tuple[bytes, int]:
    header: dict[str, Any] = {"__metadata__": {"format": "mlx"}}
    cursor = 0
    for spec in specs:
        header[spec.name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [cursor, cursor + spec.nbytes],
        }
        cursor += spec.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return encoded, cursor


def _copy_tensor_payload(
    source: Path,
    header: TensorHeader,
    output: Any,
    stats: ImportStats,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    with _safe_source_file(source, header.shard).open("rb", buffering=0) as stream:
        stream.seek(header.data_start + header.data_offset)
        remaining = header.source_bytes
        while remaining:
            chunk = stream.read(min(chunk_size, remaining))
            if not chunk:
                raise Glm5NextNVFP4ImportError(
                    f"truncated source tensor payload: {header.name}"
                )
            output.write(chunk)
            remaining -= len(chunk)
            stats.max_copy_chunk_bytes = max(stats.max_copy_chunk_bytes, len(chunk))
    stats.tensors_copied += 1


def _write_nonexpert_shard(
    path: Path,
    names: Sequence[str],
    *,
    source: Path,
    catalog: Mapping[str, TensorHeader],
    stats: ImportStats,
) -> tuple[str, list[str], int]:
    specs = [_Spec(name, catalog[name].dtype, catalog[name].shape) for name in names]
    header_bytes, tensor_bytes = _encoded_header(specs)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb", buffering=0) as output:
            output.write(struct.pack("<Q", len(header_bytes)))
            output.write(header_bytes)
            for name in names:
                _copy_tensor_payload(source, catalog[name], output, stats)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path), list(names), tensor_bytes


def _write_expert_bank(
    path: Path,
    unit: str,
    prefix: str,
    projection: str,
    experts: Mapping[int, _ExpertTriple],
    *,
    source: Path,
    catalog: Mapping[str, TensorHeader],
    stats: ImportStats,
    unit_hook: UnitHook | None,
) -> tuple[str, list[str], int]:
    count = len(experts)
    first = experts[0]
    specs = _expert_specs(
        prefix, projection, count, catalog[first.weight], catalog[first.scale]
    )
    header_bytes, tensor_bytes = _encoded_header(specs)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    scale_tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.scales.tmp")
    global_tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.global.tmp")
    try:
        with (
            temporary.open("xb", buffering=0) as weights,
            scale_tmp.open("xb", buffering=0) as scales,
            global_tmp.open("xb", buffering=0) as globals_,
        ):
            weights.write(struct.pack("<Q", len(header_bytes)))
            weights.write(header_bytes)
            for expert in range(count):
                if unit_hook is not None:
                    unit_hook(unit, expert)
                triple = experts[expert]
                # U8 -> U32 is a pure view: copy the identical packed bytes.
                _copy_tensor_payload(source, catalog[triple.weight], weights, stats)
                # F8_E4M3 -> U8 is likewise an identical one-byte carrier view.
                _copy_tensor_payload(source, catalog[triple.scale], scales, stats)
                scale_2 = _read_tensor(source, catalog[triple.scale_2])
                scalar = np.asarray(scale_2.data, dtype=np.float32)
                if scalar.shape != () or not np.isfinite(scalar) or scalar <= 0:
                    raise Glm5NextNVFP4ImportError(
                        f"invalid ModelOpt global scale value: {triple.scale_2}"
                    )
                _copy_tensor_payload(source, catalog[triple.scale_2], globals_, stats)
                stats.expert_matrices_repacked += 1
            for stream in (scales, globals_):
                stream.flush()
                os.fsync(stream.fileno())
            for sidecar in (scale_tmp, global_tmp):
                with sidecar.open("rb", buffering=0) as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        weights.write(chunk)
            weights.flush()
            os.fsync(weights.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        scale_tmp.unlink(missing_ok=True)
        global_tmp.unlink(missing_ok=True)
    return _sha256(path), [spec.name for spec in specs], tensor_bytes


def _projected_size(
    catalog: Mapping[str, TensorHeader],
    families: Mapping[tuple[str, str], Mapping[int, _ExpertTriple]],
) -> tuple[int, int]:
    expert_names = {
        name
        for experts in families.values()
        for triple in experts.values()
        for name in (triple.weight, triple.scale, triple.scale_2)
    }
    nonexpert = [name for name in catalog if name not in expert_names]
    total = sum(catalog[name].source_bytes for name in nonexpert)
    for (prefix, projection), experts in families.items():
        first = experts[0]
        total += sum(
            spec.nbytes
            for spec in _expert_specs(
                prefix,
                projection,
                len(experts),
                catalog[first.weight],
                catalog[first.scale],
            )
        )
    return total, len(nonexpert)


def build_import_plan(
    source: Path | str,
    destination: Path | str,
    *,
    repository: str = SOURCE_REPOSITORY,
    revision: str = SOURCE_REVISION,
    validate_pinned: bool = True,
) -> ImportPlan:
    source_path = Path(source).resolve()
    _, catalog, _, _, families = _source_contract(
        source_path,
        repository=repository,
        revision=revision,
        validate_pinned=validate_pinned,
    )
    projected, nonexpert = _projected_size(catalog, families)
    return ImportPlan(
        source_path,
        Path(destination).resolve(),
        sum(item.source_bytes for item in catalog.values()),
        projected,
        sum(len(experts) for experts in families.values()),
        nonexpert,
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
        target = output / item.name
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copy2(item, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    converted = dict(config)
    converted.pop("quantization_config", None)
    converted["quantization"] = {
        "bits": 4,
        "group_size": 16,
        "mode": "nvfp4",
        "layout": NVFP4_LAYOUT,
        "modelopt_global_scale": True,
        "scope": OUTPUT_SCOPE,
        "source_layout": "modelopt-0.45-per-expert",
    }
    converted["glm5_next_artifact"] = {
        "layout": NVFP4_LAYOUT,
        "imported_from": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "base_model": SOURCE_BASE_MODEL,
        "experts": SOURCE_EXPERTS,
        "expert_layers": list(SOURCE_EXPERT_LAYERS),
        "expert_matrices": SOURCE_EXPERT_MATRICES,
        "lossless_repack": True,
        "nonexpert_payloads_preserved": True,
    }
    _atomic_bytes(output / "config.json", _json_bytes(converted))


def import_glm53_nvfp4(
    source: Path | str,
    destination: Path | str,
    *,
    repository: str = SOURCE_REPOSITORY,
    revision: str = SOURCE_REVISION,
    validate_pinned: bool = True,
    unit_hook: UnitHook | None = None,
) -> ImportResult:
    """Losslessly repack the pinned per-expert artifact into MLX expert banks."""

    source_path = Path(source).resolve()
    output = Path(destination).resolve()
    if source_path == output or source_path in output.parents:
        raise Glm5NextNVFP4ImportError("destination must be outside the source tree")
    config, catalog, shards, weight_map, families = _source_contract(
        source_path,
        repository=repository,
        revision=revision,
        validate_pinned=validate_pinned,
    )
    projected, _ = _projected_size(catalog, families)
    if validate_pinned and projected != SOURCE_TENSOR_BYTES:
        raise Glm5NextNVFP4ImportError(
            f"lossless projected size changed: {projected} != {SOURCE_TENSOR_BYTES}"
        )

    source_shas = {
        shard: _sha256(_safe_source_file(source_path, shard)) for shard in shards
    }
    expert_names = {
        name
        for experts in families.values()
        for triple in experts.values()
        for name in (triple.weight, triple.scale, triple.scale_2)
    }
    nonexpert_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        if name not in expert_names:
            nonexpert_by_shard[shard].append(name)
    for names in nonexpert_by_shard.values():
        names.sort()

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST_NAME
    identity = {
        "importer_version": IMPORTER_VERSION,
        "layout_version": NVFP4_LAYOUT,
        "source_repository": repository,
        "source_revision": revision,
        "source_index_sha256": _sha256(source_path / INDEX_NAME),
        "source_shard_count": len(shards),
        "source_payload_bytes": sum(item.source_bytes for item in catalog.values()),
        "mode": "nvfp4",
        "group_size": 16,
        "bits": 4,
        "scope": OUTPUT_SCOPE,
        "lossless_repack": True,
    }
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        for name, expected in identity.items():
            if manifest.get(name) != expected:
                raise Glm5NextNVFP4ImportError(
                    f"existing import manifest mismatch for {name}; use a fresh destination"
                )
        records = manifest.get("units")
        if not isinstance(records, dict):
            raise Glm5NextNVFP4ImportError("import manifest units are invalid")
    else:
        records: dict[str, Any] = {}
        manifest = {**identity, "source": str(source_path), "units": records}
        _atomic_bytes(manifest_path, _json_bytes(manifest))
    manifest["complete"] = False
    manifest.pop("output_index_sha256", None)
    _atomic_bytes(manifest_path, _json_bytes(manifest))

    stats = ImportStats()

    def complete(unit: str, record: dict[str, Any]) -> None:
        records[unit] = record
        manifest["units"] = records
        _atomic_bytes(manifest_path, _json_bytes(manifest))
        stats.units_completed += 1

    normal_count = 0
    for ordinal, shard in enumerate(shards, start=1):
        names = nonexpert_by_shard.get(shard, [])
        if not names:
            continue
        normal_count += 1
        unit = "nonexpert:" + shard
        old = records.get(unit)
        if (
            isinstance(old, dict)
            and old.get("source_sha256") == source_shas[shard]
            and _record_valid(output, old)
        ):
            stats.units_resumed += 1
            continue
        filename = f"nonexpert-{ordinal:05d}-of-{len(shards):05d}.safetensors"
        checksum, tensors, tensor_bytes = _write_nonexpert_shard(
            output / filename,
            names,
            source=source_path,
            catalog=catalog,
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

    for (prefix, projection), experts in sorted(families.items()):
        unit = f"experts:{prefix}:{projection}"
        source_set = {
            catalog[name].shard
            for triple in experts.values()
            for name in (triple.weight, triple.scale, triple.scale_2)
        }
        source_identity = {shard: source_shas[shard] for shard in sorted(source_set)}
        old = records.get(unit)
        if (
            isinstance(old, dict)
            and old.get("source_shards_sha256") == source_identity
            and _record_valid(output, old)
        ):
            stats.units_resumed += 1
            continue
        layer = int(prefix.split(".layers.", 1)[1].split(".", 1)[0])
        filename = f"experts-layer-{layer:02d}-{projection}-nvfp4.safetensors"
        checksum, tensors, tensor_bytes = _write_expert_bank(
            output / filename,
            unit,
            prefix,
            projection,
            experts,
            source=source_path,
            catalog=catalog,
            stats=stats,
            unit_hook=unit_hook,
        )
        complete(
            unit,
            {
                "file": filename,
                "sha256": checksum,
                "source_shards_sha256": source_identity,
                "tensors": tensors,
                "tensor_bytes": tensor_bytes,
            },
        )

    if len(records) != normal_count + len(families):
        raise Glm5NextNVFP4ImportError("import ended without every output unit")
    weight_out: dict[str, str] = {}
    total_size = 0
    for record in records.values():
        for name in record["tensors"]:
            if name in weight_out:
                raise Glm5NextNVFP4ImportError(f"duplicate output tensor: {name}")
            weight_out[name] = record["file"]
        total_size += int(record["tensor_bytes"])
    if total_size != projected:
        raise Glm5NextNVFP4ImportError(
            f"indexed payload differs from projection: {total_size} != {projected}"
        )

    index = {
        "metadata": {
            "format": "mlx",
            "total_size": total_size,
            "glm5_next_layout": NVFP4_LAYOUT,
            "glm5_next_import_source": SOURCE_REPOSITORY,
            "glm5_next_import_revision": SOURCE_REVISION,
            "glm5_next_lossless_repack": True,
            "glm5_next_expert_matrices": sum(
                len(experts) for experts in families.values()
            ),
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
    return ImportResult(output, index_path, manifest_path, projected, stats)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Losslessly repack pinned GLM-5.3 ModelOpt NVFP4 experts for MLX"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repository", default=SOURCE_REPOSITORY)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    args = parser.parse_args(argv)
    result = import_glm53_nvfp4(
        args.source,
        args.destination,
        repository=args.repository,
        revision=args.revision,
    )
    print(
        f"{result.output_dir} "
        f"({result.projected_output_bytes / (1024**3):.1f} GiB projected, lossless NVFP4)"
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "Glm5NextNVFP4ImportError",
    "ImportPlan",
    "ImportResult",
    "ImportStats",
    "MANIFEST_NAME",
    "OUTPUT_SCOPE",
    "SOURCE_BASE_MODEL",
    "SOURCE_EXPERT_MATRICES",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "SOURCE_TENSOR_BYTES",
    "build_import_plan",
    "import_glm53_nvfp4",
]
