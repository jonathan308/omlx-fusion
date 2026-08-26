"""Streaming inspection foundation for GLM-5.3-Flash conversion.

No model is instantiated and safetensors payloads are never collected in
memory.  Full conversion deliberately remains disabled until native KDA,
mHC, no-RoPE DSA, multimodal merging, and the GLM5-Next MTP ABI exist.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .contract import (
    OFFICIAL_REVISION,
    Glm5NextContractError,
    Glm5NextSourceContract,
    validate_source_contract,
)

_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
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
_FLOAT_DTYPES = {"F8_E4M3", "F8_E4M3FN", "F8_E5M2", "F16", "BF16", "F32", "F64"}


class Glm5NextUnsupportedMathError(RuntimeError):
    """Conversion stopped because its required runtime math is unavailable."""


@dataclass(frozen=True, slots=True)
class TensorHeader:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    source_bytes: int


@dataclass(frozen=True, slots=True)
class Glm5NextConversionPlan:
    source: Path
    destination: Path
    contract: Glm5NextSourceContract
    source_payload_bytes: int
    dense_bf16_upper_bound_bytes: int
    largest_tensor_bytes: int
    unsupported_math: tuple[str, ...]


def _load_index(source: Path) -> Mapping[str, Any]:
    try:
        value = json.loads((source / "model.safetensors.index.json").read_bytes())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Glm5NextContractError("cannot read model.safetensors.index.json") from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("weight_map"), Mapping):
        raise Glm5NextContractError("invalid model.safetensors.index.json")
    return value


def _read_safetensors_header(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("rb", buffering=0) as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise Glm5NextContractError(f"truncated safetensors header: {path.name}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > 256 * 1024 * 1024:
                raise Glm5NextContractError(f"invalid safetensors header size: {path.name}")
            raw_header = stream.read(header_length)
        value = json.loads(raw_header)
    except FileNotFoundError as exc:
        raise Glm5NextContractError(f"missing source shard: {path.name}") from exc
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Glm5NextContractError(f"invalid safetensors header: {path.name}") from exc
    if not isinstance(value, Mapping):
        raise Glm5NextContractError(f"safetensors header is not an object: {path.name}")
    return value


def iter_tensor_headers(source: Path | str) -> Iterator[TensorHeader]:
    """Read exactly one shard header at a time and yield payload descriptors."""

    source = Path(source)
    index = _load_index(source)
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    seen: set[str] = set()
    for shard in shards:
        if not isinstance(shard, str):
            raise Glm5NextContractError("non-string shard in weight_map")
        header = _read_safetensors_header(source / shard)
        for name, descriptor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(descriptor, Mapping):
                raise Glm5NextContractError(f"invalid tensor descriptor: {name}")
            dtype = descriptor.get("dtype")
            shape = descriptor.get("shape")
            offsets = descriptor.get("data_offsets")
            if dtype not in _DTYPE_BYTES or not isinstance(shape, list):
                raise Glm5NextContractError(f"unsupported tensor descriptor: {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                raise Glm5NextContractError(f"invalid tensor offsets: {name}")
            typed_shape = tuple(shape)
            if not all(isinstance(dim, int) and dim >= 0 for dim in typed_shape):
                raise Glm5NextContractError(f"invalid tensor shape: {name}")
            expected_name = weight_map.get(name)
            if expected_name != shard:
                raise Glm5NextContractError(f"index/header mismatch for tensor: {name}")
            seen.add(name)
            yield TensorHeader(
                name=name,
                shard=shard,
                dtype=dtype,
                shape=typed_shape,
                source_bytes=offsets[1] - offsets[0],
            )
    if seen != set(weight_map):
        missing = next(iter(set(weight_map) - seen), "unknown")
        raise Glm5NextContractError(f"indexed tensor missing from shard headers: {missing}")


def build_conversion_plan(
    source: Path | str,
    destination: Path | str,
    *,
    source_revision: str = OFFICIAL_REVISION,
) -> Glm5NextConversionPlan:
    source = Path(source)
    destination = Path(destination)
    contract = validate_source_contract(source, source_revision=source_revision)
    payload = 0
    bf16_upper = 0
    largest = 0
    for tensor in iter_tensor_headers(source):
        payload += tensor.source_bytes
        largest = max(largest, tensor.source_bytes)
        elements = 1
        for dimension in tensor.shape:
            elements *= dimension
        if tensor.dtype in _FLOAT_DTYPES:
            bf16_upper += elements * 2
        else:
            bf16_upper += tensor.source_bytes
    if payload != contract.tensor_bytes:
        raise Glm5NextContractError(
            f"header payload size changed: expected {contract.tensor_bytes}, found {payload}"
        )
    unsupported = (
        "KDA recurrent linear attention with short convolution on 34/45 layers",
        "manifold-constrained hyper-connections (mHC, multiplier 4)",
        "no-RoPE DSA/MLA with compressed k-pool indexer on 11/45 layers",
        "288-expert sigmoid/noaux_tc MoE with FP32 routing",
        "native image/video token merge and 24-layer vision encoder",
        "GLM5-Next layer-45 MTP/shared-head state ABI",
    )
    return Glm5NextConversionPlan(
        source=source,
        destination=destination,
        contract=contract,
        source_payload_bytes=payload,
        dense_bf16_upper_bound_bytes=bf16_upper,
        largest_tensor_bytes=largest,
        unsupported_math=unsupported,
    )


def convert_glm53_flash(
    source: Path | str,
    destination: Path | str,
    *,
    source_revision: str = OFFICIAL_REVISION,
) -> Glm5NextConversionPlan:
    """Validate and inventory the source, then stop before writing an artifact."""

    plan = build_conversion_plan(
        source, destination, source_revision=source_revision
    )
    raise Glm5NextUnsupportedMathError(
        "GLM-5.3-Flash conversion is intentionally disabled until these exact "
        "runtime contracts are implemented: " + "; ".join(plan.unsupported_math)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-revision", default=OFFICIAL_REVISION)
    args = parser.parse_args(argv)
    convert_glm53_flash(
        args.source, args.destination, source_revision=args.source_revision
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
