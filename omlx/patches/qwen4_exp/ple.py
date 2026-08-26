# SPDX-License-Identifier: Apache-2.0
"""Read-only, SSD-backed Qwen3.8-Flash-Next PLE embeddings.

The published checkpoint stores the 51.2B-parameter PLE table as 128
contiguous row shards.  Loading those tensors through the ordinary model
loader materializes roughly 95 GiB of BF16 weights.  This module instead
validates the safetensors index and headers up front, maps each table tensor
read-only on first use, and copies only the pages containing requested rows.

This is deliberately specific to Qwen4-Exp's *actual* checkpoint contract:
the PLE module is on zero-indexed decoder layer 1 even though ``ple_layer_ids``
is one-indexed in config, and it has eight bigram plus eight trigram heads.
It is not mlx-lm speculative n-gram decoding and it must not be used as a
generic embedding loader.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import stat
import struct
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import Executor, Future
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

PLE_PREFIX: Final = "model.language_model.layers.1.ple.ple_embedding"
PLE_TABLE_PREFIX: Final = f"{PLE_PREFIX}.ngram_embedding"
PLE_SHARD_COUNT: Final = 128
PLE_NGRAM_SIZE: Final = 3
PLE_HEADS_PER_NGRAM: Final = 8
PLE_HEAD_COUNT: Final = 16
PLE_MLX_Q8_DTYPE: Final = "MLX_Q8_AFFINE"
PLE_MLX_Q8_BITS: Final = 8
PLE_MLX_Q8_GROUP_SIZE: Final = 32
PLE_MLX_Q8_MODE: Final = "affine"
PLE_MLX_Q8_FORMAT: Final = "mlx-affine-q8-v1"

# These flat model.safetensors.index.json metadata fields are part of the Q8
# artifact contract.  They make a mixed compute-Q4 / PLE-Q8 conversion
# unambiguous without abusing the model-wide ``quantization`` config.
PLE_MLX_Q8_METADATA: Final[dict[str, object]] = {
    "qwen4_exp_ple_format": PLE_MLX_Q8_FORMAT,
    "qwen4_exp_ple_bits": PLE_MLX_Q8_BITS,
    "qwen4_exp_ple_group_size": PLE_MLX_Q8_GROUP_SIZE,
    "qwen4_exp_ple_mode": PLE_MLX_Q8_MODE,
    "qwen4_exp_ple_scale_dtype": "BF16",
    "qwen4_exp_ple_layer": 1,
    "qwen4_exp_ple_shard_count": PLE_SHARD_COUNT,
}

_PLE_Q8_LAYOUT_METADATA_FIELDS: Final[dict[str, str]] = {
    "unigram_vocab_size": "qwen4_exp_ple_unigram_vocab_size",
    "ngram_vocab_size_base": "qwen4_exp_ple_ngram_vocab_size_base",
    "head_dim": "qwen4_exp_ple_head_dim",
    "eos_token_id": "qwen4_exp_ple_eos_token_id",
    "seed": "qwen4_exp_ple_seed",
    "make_vocab_divisible_by": "qwen4_exp_ple_make_vocab_divisible_by",
}

_MAX_HEADER_BYTES: Final = 16 * 1024 * 1024
_MASK64: Final = (1 << 64) - 1
_SPLITMIX_GAMMA: Final = 0x9E3779B97F4A7C15
_SPLITMIX_M1: Final = 0xBF58476D1CE4E5B9
_SPLITMIX_M2: Final = 0x94D049BB133111EB
_PRIME_1: Final = 10007

_DTYPES: Final[dict[str, tuple[np.dtype[Any], int]]] = {
    # NumPy has no portable bfloat16 scalar.  It is kept as raw uint16 in
    # the page cache and decoded only for selected rows.
    "BF16": (np.dtype("<u2"), 2),
    "F16": (np.dtype("<f2"), 2),
    "F32": (np.dtype("<f4"), 4),
    "I64": (np.dtype("<i8"), 8),
    "U32": (np.dtype("<u4"), 4),
}


class PLEArtifactError(ValueError):
    """The PLE checkpoint is incomplete, unsafe, or structurally invalid."""


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(
    unigram_vocab_size: int,
    *,
    seed: int,
    ple_layer_index: int = 0,
) -> NDArray[np.int64]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    values = []
    for index in range(PLE_NGRAM_SIZE):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        values.append(2 * (_splitmix64(value) % half_bound) + 1)
    return np.asarray(values, dtype=np.int64)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


@dataclass(frozen=True, slots=True)
class PLELayout:
    """Static layout used to prove that an artifact matches Qwen4-Exp PLE.

    Non-default values exist so unit tests and future official checkpoints can
    exercise the reader without manufacturing a 95-GiB table.  The structural
    invariants (layer 1, 128 shards, 8+8 heads, and trigram maximum) cannot be
    overridden.
    """

    unigram_vocab_size: int = 248_320
    ngram_vocab_size_base: int = 20_000_000
    head_dim: int = 160
    eos_token_id: int = 248_044
    seed: int = 1234
    make_vocab_divisible_by: int = 128
    table_dtype: str = "BF16"

    def __post_init__(self) -> None:
        if self.unigram_vocab_size <= 0:
            raise ValueError("unigram_vocab_size must be positive")
        if self.ngram_vocab_size_base < 2:
            raise ValueError("ngram_vocab_size_base must be at least 2")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if not 0 <= self.eos_token_id < self.unigram_vocab_size:
            raise ValueError("eos_token_id must be in the unigram vocabulary")
        if self.make_vocab_divisible_by <= 0:
            raise ValueError("make_vocab_divisible_by must be positive")
        if self.table_dtype not in {"BF16", "F16", "F32", PLE_MLX_Q8_DTYPE}:
            raise ValueError("PLE table dtype must be BF16, F16, F32, or MLX_Q8_AFFINE")
        if (
            self.table_dtype == PLE_MLX_Q8_DTYPE
            and self.head_dim % PLE_MLX_Q8_GROUP_SIZE
        ):
            raise ValueError("MLX affine Q8 PLE head_dim must divide into groups of 32")
        if self.padded_vocab_size % PLE_SHARD_COUNT:
            raise ValueError(
                "padded PLE vocabulary must divide exactly across 128 shards"
            )

    @property
    def head_vocab_sizes(self) -> NDArray[np.int64]:
        return np.asarray(
            [
                _find_nth_prime_after(self.ngram_vocab_size_base - 1, index + 1)
                for index in range(PLE_HEAD_COUNT)
            ],
            dtype=np.int64,
        )

    @property
    def head_offsets(self) -> NDArray[np.int64]:
        sizes = self.head_vocab_sizes
        return np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(sizes[:-1], dtype=np.int64))
        )

    @property
    def layer_multipliers(self) -> NDArray[np.int64]:
        return _build_layer_multipliers(
            self.unigram_vocab_size,
            seed=self.seed,
            ple_layer_index=0,
        )

    @property
    def padded_vocab_size(self) -> int:
        total = int(self.head_vocab_sizes.sum())
        divisor = self.make_vocab_divisible_by
        return ((total + divisor - 1) // divisor) * divisor

    @property
    def rows_per_shard(self) -> int:
        return self.padded_vocab_size // PLE_SHARD_COUNT

    @property
    def embedding_dim(self) -> int:
        return PLE_HEAD_COUNT * self.head_dim


OFFICIAL_PLE_LAYOUT: Final = PLELayout()
OFFICIAL_PLE_Q8_LAYOUT: Final = PLELayout(table_dtype=PLE_MLX_Q8_DTYPE)


def mlx_q8_index_metadata(
    layout: PLELayout = OFFICIAL_PLE_Q8_LAYOUT,
) -> dict[str, object]:
    """Return the exact index metadata a Q8 PLE converter must emit."""

    if layout.table_dtype != PLE_MLX_Q8_DTYPE:
        raise ValueError("Q8 metadata requires an MLX_Q8_AFFINE layout")
    return {
        **PLE_MLX_Q8_METADATA,
        **{
            metadata_key: getattr(layout, field_name)
            for field_name, metadata_key in _PLE_Q8_LAYOUT_METADATA_FIELDS.items()
        },
        "qwen4_exp_ple_rows_per_shard": layout.rows_per_shard,
    }


def _layout_from_q8_metadata(metadata: dict[str, Any]) -> PLELayout:
    values: dict[str, int] = {}
    for field_name, metadata_key in _PLE_Q8_LAYOUT_METADATA_FIELDS.items():
        value = metadata.get(metadata_key)
        if type(value) is not int:
            raise PLEArtifactError(
                f"invalid MLX Q8 PLE metadata {metadata_key}: expected an integer"
            )
        values[field_name] = value
    try:
        return PLELayout(**values, table_dtype=PLE_MLX_Q8_DTYPE)
    except ValueError as exc:
        raise PLEArtifactError(f"invalid MLX Q8 PLE layout metadata: {exc}") from exc


def _layout_from_index_metadata(metadata: Any) -> PLELayout:
    """Select dense BF16 or affine Q8 from an explicit format marker."""

    if not isinstance(metadata, dict):
        return OFFICIAL_PLE_LAYOUT
    artifact_format = metadata.get("qwen4_exp_ple_format")
    if artifact_format == PLE_MLX_Q8_FORMAT:
        return _layout_from_q8_metadata(metadata)
    if artifact_format in (None, "bf16-ssd-mmap-v1"):
        return OFFICIAL_PLE_LAYOUT
    raise PLEArtifactError(
        f"unsupported Qwen4-Exp PLE artifact format: {artifact_format!r}"
    )


@dataclass(frozen=True, slots=True)
class _FileFingerprint:
    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileFingerprint:
        return cls(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@dataclass(frozen=True, slots=True)
class _TensorDescriptor:
    path: Path
    name: str
    dtype_name: str
    shape: tuple[int, ...]
    byte_start: int
    byte_length: int
    fingerprint: _FileFingerprint

    @property
    def numpy_dtype(self) -> np.dtype[Any]:
        try:
            return _DTYPES[self.dtype_name][0]
        except KeyError as exc:  # pragma: no cover - constructor validates it
            raise PLEArtifactError(f"unsupported dtype {self.dtype_name}") from exc


class _MappedTensor:
    """A tensor-range mmap that does not fault data in until rows are read."""

    def __init__(self, descriptor: _TensorDescriptor) -> None:
        self.descriptor = descriptor
        self._mapping: mmap.mmap | None = None
        self._mapping_delta = 0
        self._lock = threading.Lock()

    def _ensure_open(self) -> mmap.mmap:
        with self._lock:
            if self._mapping is not None:
                return self._mapping

            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            descriptor = self.descriptor
            file_descriptor = os.open(descriptor.path, flags)
            try:
                actual = _FileFingerprint.from_stat(os.fstat(file_descriptor))
                if actual != descriptor.fingerprint:
                    raise PLEArtifactError(
                        f"safetensors file changed after validation: {descriptor.path}"
                    )
                granularity = mmap.ALLOCATIONGRANULARITY
                mapping_start = descriptor.byte_start // granularity * granularity
                self._mapping_delta = descriptor.byte_start - mapping_start
                mapping_length = self._mapping_delta + descriptor.byte_length
                self._mapping = mmap.mmap(
                    file_descriptor,
                    mapping_length,
                    access=mmap.ACCESS_READ,
                    offset=mapping_start,
                )
            finally:
                os.close(file_descriptor)

            if hasattr(self._mapping, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                # The hint is an optimization.  Some macOS/filesystem
                # combinations expose madvise but reject MADV_RANDOM.
                with suppress(OSError, ValueError):
                    self._mapping.madvise(mmap.MADV_RANDOM)
            return self._mapping

    def read_rows(self, start: int, stop: int) -> NDArray[Any]:
        rows, columns = self.descriptor.shape
        if not 0 <= start <= stop <= rows:
            raise IndexError(f"row range [{start}, {stop}) is outside [0, {rows})")
        mapping = self._ensure_open()
        array = np.ndarray(
            (rows, columns),
            dtype=self.descriptor.numpy_dtype,
            buffer=mapping,
            offset=self._mapping_delta,
        )
        # A copy is intentional: callers never retain an mmap view, and the
        # explicit cache remains byte-bounded and safe to close independently.
        return array[start:stop].copy()

    def read_all(self) -> NDArray[Any]:
        if len(self.descriptor.shape) != 1:
            raise PLEArtifactError(f"expected a vector tensor: {self.descriptor.name}")
        mapping = self._ensure_open()
        return np.ndarray(
            self.descriptor.shape,
            dtype=self.descriptor.numpy_dtype,
            buffer=mapping,
            offset=self._mapping_delta,
        ).copy()

    def close(self) -> None:
        with self._lock:
            if self._mapping is not None:
                self._mapping.close()
                self._mapping = None


@dataclass(frozen=True, slots=True)
class _Q8Page:
    weight: NDArray[np.uint32]
    scales: NDArray[np.uint16]
    biases: NDArray[np.uint16]

    @property
    def nbytes(self) -> int:
        return self.weight.nbytes + self.scales.nbytes + self.biases.nbytes


@dataclass(slots=True)
class _Q8MappedShard:
    """The three tensors emitted by MLX ``QuantizedEmbedding`` for one shard."""

    weight: _MappedTensor
    scales: _MappedTensor
    biases: _MappedTensor

    def read_rows(self, start: int, stop: int) -> _Q8Page:
        return _Q8Page(
            weight=self.weight.read_rows(start, stop),
            scales=self.scales.read_rows(start, stop),
            biases=self.biases.read_rows(start, stop),
        )

    def close(self) -> None:
        self.weight.close()
        self.scales.close()
        self.biases.close()


_Page = NDArray[Any] | _Q8Page


class _BoundedPageCache:
    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_cache_bytes cannot be negative")
        self.max_bytes = max_bytes
        self._size = 0
        self._pages: OrderedDict[tuple[int, int], _Page] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[int, int]) -> _Page | None:
        with self._lock:
            value = self._pages.get(key)
            if value is None:
                self.misses += 1
                return None
            self._pages.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: tuple[int, int], value: _Page) -> None:
        if self.max_bytes == 0 or value.nbytes > self.max_bytes:
            return
        with self._lock:
            old = self._pages.pop(key, None)
            if old is not None:
                self._size -= old.nbytes
            self._pages[key] = value
            self._size += value.nbytes
            while self._size > self.max_bytes:
                _, evicted = self._pages.popitem(last=False)
                self._size -= evicted.nbytes

    def clear(self) -> None:
        with self._lock:
            self._pages.clear()
            self._size = 0

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._size

    @property
    def page_count(self) -> int:
        with self._lock:
            return len(self._pages)


def _checked_model_file(model_dir: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename:
        raise PLEArtifactError("safetensors index contains a non-string filename")
    pure = PurePosixPath(filename)
    if pure.is_absolute() or ".." in pure.parts or "\\" in filename:
        raise PLEArtifactError(f"unsafe safetensors filename in index: {filename!r}")
    path = model_dir.joinpath(*pure.parts)
    try:
        info = path.stat()
    except FileNotFoundError as exc:
        raise PLEArtifactError(f"missing safetensors shard: {filename}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise PLEArtifactError(f"safetensors shard is not a regular file: {filename}")
    return path


def _parse_safetensors_header(
    path: Path,
) -> tuple[dict[str, Any], int, _FileFingerprint]:
    info = path.stat()
    fingerprint = _FileFingerprint.from_stat(info)
    with path.open("rb", buffering=0) as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise PLEArtifactError(f"truncated safetensors prefix: {path.name}")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_length <= _MAX_HEADER_BYTES:
            raise PLEArtifactError(
                f"invalid safetensors header length {header_length}: {path.name}"
            )
        if 8 + header_length > info.st_size:
            raise PLEArtifactError(f"truncated safetensors header: {path.name}")
        try:
            header = json.loads(stream.read(header_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PLEArtifactError(f"invalid safetensors JSON: {path.name}") from exc
    if not isinstance(header, dict):
        raise PLEArtifactError(f"safetensors header must be an object: {path.name}")

    ranges: list[tuple[int, int, str]] = []
    data_bytes = info.st_size - 8 - header_length
    for name, value in header.items():
        if name == "__metadata__":
            # ``mx.save_safetensors`` emits JSON null when no file metadata is
            # supplied; Hugging Face writers generally emit a string map.
            if value is not None and not isinstance(value, dict):
                raise PLEArtifactError(f"invalid __metadata__ in {path.name}")
            continue
        if not isinstance(name, str) or not isinstance(value, dict):
            raise PLEArtifactError(f"invalid tensor entry in {path.name}")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        dtype = value.get("dtype")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(item, int) and item >= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
        ):
            raise PLEArtifactError(
                f"malformed tensor descriptor {name!r} in {path.name}"
            )
        start, stop = offsets
        if not 0 <= start <= stop <= data_bytes:
            raise PLEArtifactError(f"out-of-range tensor {name!r} in {path.name}")
        ranges.append((start, stop, name))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:], strict=False):
        if previous[1] > current[0]:
            raise PLEArtifactError(
                f"overlapping tensors {previous[2]!r} and {current[2]!r} in {path.name}"
            )
    return header, 8 + header_length, fingerprint


def _tensor_descriptor(
    *,
    path: Path,
    name: str,
    header: dict[str, Any],
    data_start: int,
    fingerprint: _FileFingerprint,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
) -> _TensorDescriptor:
    value = header.get(name)
    if not isinstance(value, dict):
        raise PLEArtifactError(f"index points to absent tensor {name!r} in {path.name}")
    dtype_name = value.get("dtype")
    shape_value = value.get("shape")
    offsets = value.get("data_offsets")
    shape = tuple(shape_value) if isinstance(shape_value, list) else ()
    if dtype_name != expected_dtype:
        raise PLEArtifactError(
            f"{name} must have dtype {expected_dtype}, found {dtype_name!r}"
        )
    if shape != expected_shape:
        raise PLEArtifactError(
            f"{name} must have shape {expected_shape}, found {shape}"
        )
    try:
        _, item_size = _DTYPES[expected_dtype]
    except KeyError as exc:
        raise PLEArtifactError(f"unsupported expected dtype {expected_dtype}") from exc
    expected_bytes = math.prod(expected_shape) * item_size
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise PLEArtifactError(f"invalid data offsets for {name}")
    byte_length = offsets[1] - offsets[0]
    if byte_length != expected_bytes:
        raise PLEArtifactError(
            f"{name} byte length is {byte_length}, expected {expected_bytes}"
        )
    return _TensorDescriptor(
        path=path,
        name=name,
        dtype_name=expected_dtype,
        shape=expected_shape,
        byte_start=data_start + offsets[0],
        byte_length=byte_length,
        fingerprint=fingerprint,
    )


class Qwen4ExpPLESSDPool:
    """Bounded, mmap-backed lookup for the Qwen4-Exp PLE table.

    ``gather`` accepts global table row IDs. ``hash_ids`` implements the
    checkpoint's eight bigram and eight trigram hashes, and ``lookup`` joins
    both operations into the 2560-wide PLE input expected by the projection.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike[str],
        *,
        layout: PLELayout | None = None,
        index_name: str = "model.safetensors.index.json",
        rows_per_page: int = 256,
        max_cache_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if rows_per_page <= 0:
            raise ValueError("rows_per_page must be positive")
        self.model_dir = Path(model_dir)
        self.rows_per_page = rows_per_page
        self._cache = _BoundedPageCache(max_cache_bytes)
        self._closed = False
        self._state = threading.Condition()
        self._active_reads = 0

        index_path = self.model_dir / index_name
        try:
            index = json.loads(index_path.read_bytes())
        except FileNotFoundError as exc:
            raise PLEArtifactError(f"missing safetensors index: {index_path}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PLEArtifactError(f"invalid safetensors index: {index_path}") from exc
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            raise PLEArtifactError(
                "safetensors index must contain an object weight_map"
            )
        weight_map: dict[str, Any] = index["weight_map"]
        metadata = index.get("metadata")
        if layout is None:
            layout = _layout_from_index_metadata(metadata)
        self.layout = layout

        table_names = [
            f"{PLE_TABLE_PREFIX}.shard_{shard}.weight"
            for shard in range(PLE_SHARD_COUNT)
        ]
        q8_scales_names = [
            f"{PLE_TABLE_PREFIX}.shard_{shard}.scales"
            for shard in range(PLE_SHARD_COUNT)
        ]
        q8_biases_names = [
            f"{PLE_TABLE_PREFIX}.shard_{shard}.biases"
            for shard in range(PLE_SHARD_COUNT)
        ]
        q8_storage = layout.table_dtype == PLE_MLX_Q8_DTYPE
        expected_table_names = list(table_names)
        if q8_storage:
            expected_table_names.extend(q8_scales_names)
            expected_table_names.extend(q8_biases_names)
        indexed_table_names = {
            name
            for name in weight_map
            if isinstance(name, str) and name.startswith(f"{PLE_TABLE_PREFIX}.shard_")
        }
        unexpected_table_names = indexed_table_names.difference(expected_table_names)
        if unexpected_table_names:
            preview = ", ".join(sorted(unexpected_table_names)[:3])
            raise PLEArtifactError(f"unexpected Qwen4-Exp PLE table tensors: {preview}")
        auxiliary = {
            f"{PLE_PREFIX}.layer_multipliers": ("I64", (PLE_NGRAM_SIZE,)),
            f"{PLE_PREFIX}.ngram_heads_offsets": ("I64", (PLE_HEAD_COUNT,)),
            f"{PLE_PREFIX}.ngram_heads_vocab_sizes": ("I64", (PLE_HEAD_COUNT,)),
        }
        required_names = [*expected_table_names, *auxiliary]
        missing = [name for name in required_names if name not in weight_map]
        if missing:
            preview = ", ".join(missing[:3])
            raise PLEArtifactError(
                f"incomplete Qwen4-Exp PLE artifact: missing {len(missing)} tensor(s): {preview}"
            )

        if q8_storage:
            if not isinstance(metadata, dict):
                raise PLEArtifactError("MLX Q8 PLE artifact requires index metadata")
            expected_metadata = mlx_q8_index_metadata(layout)
            unexpected_metadata = {
                key
                for key in metadata
                if isinstance(key, str)
                and key.startswith("qwen4_exp_ple_")
                and key not in expected_metadata
            }
            if unexpected_metadata:
                preview = ", ".join(sorted(unexpected_metadata)[:3])
                raise PLEArtifactError(
                    f"unexpected MLX Q8 PLE metadata fields: {preview}"
                )
            for key, expected in expected_metadata.items():
                actual = metadata.get(key)
                if type(actual) is not type(expected) or actual != expected:
                    raise PLEArtifactError(
                        f"invalid MLX Q8 PLE metadata {key}: "
                        f"expected {expected!r}, found {actual!r}"
                    )

        file_cache: dict[Path, tuple[dict[str, Any], int, _FileFingerprint]] = {}

        def descriptor(
            name: str,
            dtype_name: str,
            shape: tuple[int, ...],
        ) -> _TensorDescriptor:
            path = _checked_model_file(self.model_dir, weight_map[name])
            parsed = file_cache.get(path)
            if parsed is None:
                parsed = _parse_safetensors_header(path)
                file_cache[path] = parsed
            header, data_start, fingerprint = parsed
            return _tensor_descriptor(
                path=path,
                name=name,
                header=header,
                data_start=data_start,
                fingerprint=fingerprint,
                expected_dtype=dtype_name,
                expected_shape=shape,
            )

        if q8_storage:
            packed_columns = layout.head_dim * PLE_MLX_Q8_BITS // 32
            group_columns = layout.head_dim // PLE_MLX_Q8_GROUP_SIZE
            self._shards = [
                _Q8MappedShard(
                    weight=_MappedTensor(
                        descriptor(
                            table_names[shard],
                            "U32",
                            (layout.rows_per_shard, packed_columns),
                        )
                    ),
                    scales=_MappedTensor(
                        descriptor(
                            q8_scales_names[shard],
                            "BF16",
                            (layout.rows_per_shard, group_columns),
                        )
                    ),
                    biases=_MappedTensor(
                        descriptor(
                            q8_biases_names[shard],
                            "BF16",
                            (layout.rows_per_shard, group_columns),
                        )
                    ),
                )
                for shard in range(PLE_SHARD_COUNT)
            ]
        else:
            table_shape = (layout.rows_per_shard, layout.head_dim)
            self._shards = [
                _MappedTensor(descriptor(name, layout.table_dtype, table_shape))
                for name in table_names
            ]
        aux_tensors = {
            name: _MappedTensor(descriptor(name, dtype_name, shape))
            for name, (dtype_name, shape) in auxiliary.items()
        }
        try:
            self.layer_multipliers = aux_tensors[
                f"{PLE_PREFIX}.layer_multipliers"
            ].read_all()
            self.head_offsets = aux_tensors[
                f"{PLE_PREFIX}.ngram_heads_offsets"
            ].read_all()
            self.head_vocab_sizes = aux_tensors[
                f"{PLE_PREFIX}.ngram_heads_vocab_sizes"
            ].read_all()
        finally:
            for tensor in aux_tensors.values():
                tensor.close()

        expected_values = {
            "layer_multipliers": layout.layer_multipliers,
            "ngram_heads_offsets": layout.head_offsets,
            "ngram_heads_vocab_sizes": layout.head_vocab_sizes,
        }
        actual_values = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.head_offsets,
            "ngram_heads_vocab_sizes": self.head_vocab_sizes,
        }
        for label, expected in expected_values.items():
            if not np.array_equal(actual_values[label], expected):
                self.close()
                raise PLEArtifactError(
                    f"checkpoint {label} does not match the Qwen4-Exp hash contract"
                )

        last_used_row = int(self.head_offsets[-1] + self.head_vocab_sizes[-1])
        if last_used_row > layout.padded_vocab_size:
            self.close()
            raise PLEArtifactError("PLE head ranges exceed the table")

    def _check_open(self) -> None:
        with self._state:
            if self._closed:
                raise RuntimeError("PLE SSD pool is closed")

    def _begin_read(self) -> None:
        with self._state:
            if self._closed:
                raise RuntimeError("PLE SSD pool is closed")
            self._active_reads += 1

    def _end_read(self) -> None:
        with self._state:
            self._active_reads -= 1
            if self._active_reads == 0:
                self._state.notify_all()

    def _load_page(self, shard_index: int, page_index: int) -> _Page:
        self._begin_read()
        try:
            key = (shard_index, page_index)
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            start = page_index * self.rows_per_page
            stop = min(start + self.rows_per_page, self.layout.rows_per_shard)
            page = self._shards[shard_index].read_rows(start, stop)
            self._cache.put(key, page)
            return page
        finally:
            self._end_read()

    @staticmethod
    def _decode_rows(rows: NDArray[Any], dtype_name: str) -> NDArray[np.float32]:
        if dtype_name == "BF16":
            words = rows.astype(np.uint32, copy=False) << np.uint32(16)
            return words.view(np.float32)
        return rows.astype(np.float32, copy=False)

    def _normalize_row_ids(
        self, row_ids: ArrayLike
    ) -> tuple[tuple[int, ...], NDArray[np.int64]]:
        ids = np.asarray(row_ids)
        if ids.dtype.kind not in "iu":
            raise TypeError("PLE row IDs must be integers")
        flat = ids.astype(np.int64, copy=False).reshape(-1)
        if flat.size and (
            flat.min() < 0 or flat.max() >= self.layout.padded_vocab_size
        ):
            raise IndexError(
                f"PLE row IDs must be in [0, {self.layout.padded_vocab_size})"
            )
        return ids.shape, flat

    def _group_row_ids(
        self, flat: NDArray[np.int64]
    ) -> dict[tuple[int, int], list[tuple[int, int]]]:
        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for output_index, global_row in enumerate(flat):
            shard_index, local_row = divmod(int(global_row), self.layout.rows_per_shard)
            page_index, row_in_page = divmod(local_row, self.rows_per_page)
            groups[(shard_index, page_index)].append((output_index, row_in_page))
        return groups

    @staticmethod
    def _positions(
        positions: list[tuple[int, int]],
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        return (
            np.fromiter(
                (item[0] for item in positions), dtype=np.intp, count=len(positions)
            ),
            np.fromiter(
                (item[1] for item in positions), dtype=np.intp, count=len(positions)
            ),
        )

    def _decode_q8_rows(
        self, page: _Q8Page, positions: NDArray[np.intp]
    ) -> NDArray[np.float32]:
        packed = page.weight[positions]
        shifts = np.asarray((0, 8, 16, 24), dtype=np.uint32)
        codes = ((packed[..., None] >> shifts) & np.uint32(0xFF)).reshape(
            packed.shape[0], self.layout.head_dim
        )
        scales = self._decode_rows(page.scales[positions], "BF16")
        biases = self._decode_rows(page.biases[positions], "BF16")
        groups = self.layout.head_dim // PLE_MLX_Q8_GROUP_SIZE
        decoded = codes.astype(np.float32).reshape(
            packed.shape[0], groups, PLE_MLX_Q8_GROUP_SIZE
        )
        decoded *= scales[..., None]
        decoded += biases[..., None]
        return decoded.reshape(packed.shape[0], self.layout.head_dim)

    def gather_raw(self, row_ids: ArrayLike) -> NDArray[Any]:
        """Gather rows in their storage dtype without mapping a full shard.

        BF16 is returned as its raw ``uint16`` bit pattern.  An MLX caller can
        preserve BF16 without a float32 staging allocation with
        ``mx.array(rows).view(mx.bfloat16)``.
        """

        self._check_open()
        if self.layout.table_dtype == PLE_MLX_Q8_DTYPE:
            raise TypeError(
                "MLX affine Q8 has three storage tensors; use gather() to "
                "dequantize selected rows"
            )
        original_shape, flat = self._normalize_row_ids(row_ids)
        output = np.empty(
            (flat.size, self.layout.head_dim),
            dtype=_DTYPES[self.layout.table_dtype][0],
        )
        for (shard_index, page_index), positions in self._group_row_ids(flat).items():
            page = self._load_page(shard_index, page_index)
            if not isinstance(page, np.ndarray):  # pragma: no cover - invariant
                raise PLEArtifactError("dense PLE shard returned Q8 storage")
            output_positions, page_positions = self._positions(positions)
            output[output_positions] = page[page_positions]
        return output.reshape((*original_shape, self.layout.head_dim))

    def gather(self, row_ids: ArrayLike) -> NDArray[np.float32]:
        """Gather selected rows and decode them to portable float32."""

        if self.layout.table_dtype == PLE_MLX_Q8_DTYPE:
            self._check_open()
            original_shape, flat = self._normalize_row_ids(row_ids)
            output = np.empty((flat.size, self.layout.head_dim), dtype=np.float32)
            for (shard_index, page_index), positions in self._group_row_ids(
                flat
            ).items():
                page = self._load_page(shard_index, page_index)
                if not isinstance(page, _Q8Page):  # pragma: no cover - invariant
                    raise PLEArtifactError("Q8 PLE shard returned dense storage")
                output_positions, page_positions = self._positions(positions)
                output[output_positions] = self._decode_q8_rows(page, page_positions)
            return output.reshape((*original_shape, self.layout.head_dim))
        return self._decode_rows(self.gather_raw(row_ids), self.layout.table_dtype)

    def prefetch(
        self,
        row_ids: ArrayLike,
        *,
        executor: Executor | None = None,
    ) -> tuple[Future[None], ...]:
        """Warm required pages, synchronously or through a caller-owned executor.

        The caller owns executor lifetime and scheduling policy.  This seam is
        sufficient for decode/prefill overlap without creating hidden worker
        threads in the model loader.
        """

        self._check_open()
        _, flat = self._normalize_row_ids(row_ids)
        pages = {
            (
                int(row) // self.layout.rows_per_shard,
                (int(row) % self.layout.rows_per_shard) // self.rows_per_page,
            )
            for row in flat
        }
        if executor is None:
            for shard_index, page_index in pages:
                self._load_page(shard_index, page_index)
            return ()
        return tuple(
            executor.submit(self._prefetch_page, shard_index, page_index)
            for shard_index, page_index in pages
        )

    def _prefetch_page(self, shard_index: int, page_index: int) -> None:
        self._load_page(shard_index, page_index)

    def hash_ids(
        self,
        input_ids: ArrayLike,
        *,
        previous_context: ArrayLike | None = None,
    ) -> NDArray[np.int64]:
        """Compute official 8-bigram + 8-trigram global table row IDs."""

        self._check_open()
        tokens = np.asarray(input_ids)
        if tokens.dtype.kind not in "iu" or tokens.ndim != 2:
            raise ValueError("input_ids must be a rank-2 integer array")
        tokens = tokens.astype(np.int64, copy=False)
        if tokens.size and (
            tokens.min() < 0 or tokens.max() >= self.layout.unigram_vocab_size
        ):
            raise ValueError("input_ids contain a token outside the unigram vocabulary")
        batch_size, sequence_length = tokens.shape
        if sequence_length == 0:
            return np.empty((batch_size, 0, PLE_HEAD_COUNT), dtype=np.int64)
        context_length = PLE_NGRAM_SIZE - 1
        if previous_context is None:
            context = np.full(
                (batch_size, context_length),
                self.layout.eos_token_id,
                dtype=np.int64,
            )
        else:
            context = np.asarray(previous_context)
            if context.dtype.kind not in "iu" or context.shape != (
                batch_size,
                context_length,
            ):
                raise ValueError(
                    f"previous_context must have shape {(batch_size, context_length)}"
                )
            context = context.astype(np.int64, copy=False)
            if context.size and (
                context.min() < 0 or context.max() >= self.layout.unigram_vocab_size
            ):
                raise ValueError(
                    "previous_context contains a token outside the unigram vocabulary"
                )

        history = np.concatenate((context, tokens), axis=1)
        shifted = [self._shift_right_ignore_eos(history, shift) for shift in range(3)]
        blocks = []
        for ngram in range(2, PLE_NGRAM_SIZE + 1):
            start = (ngram - 2) * PLE_HEADS_PER_NGRAM
            stop = start + PLE_HEADS_PER_NGRAM
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = np.bitwise_xor(
                    mixed,
                    shifted[position] * self.layer_multipliers[position],
                )
            ids = np.remainder(mixed[..., None], self.head_vocab_sizes[start:stop])
            blocks.append(ids + self.head_offsets[start:stop])
        return np.concatenate(blocks, axis=-1)[:, -sequence_length:]

    def _shift_right_ignore_eos(
        self, token_ids: NDArray[np.int64], shift: int
    ) -> NDArray[np.int64]:
        if shift == 0:
            return token_ids
        batch_size, sequence_length = token_ids.shape
        positions = np.arange(sequence_length, dtype=np.int64)
        eos_positions = np.where(token_ids == self.layout.eos_token_id, positions, -1)
        previous_eos_inclusive = np.maximum.accumulate(eos_positions, axis=1)
        previous_eos = np.concatenate(
            (
                np.full((batch_size, 1), -1, dtype=np.int64),
                previous_eos_inclusive[:, :-1],
            ),
            axis=1,
        )
        segment_start = previous_eos + 1
        position_in_segment = positions[None, :] - segment_start
        source_positions = positions - shift
        gather_positions = np.maximum(source_positions, 0)
        shifted = token_ids[:, gather_positions]
        valid = (position_in_segment >= shift) & (source_positions[None, :] >= 0)
        return np.where(valid, shifted, self.layout.eos_token_id)

    def lookup(
        self,
        input_ids: ArrayLike,
        *,
        previous_context: ArrayLike | None = None,
    ) -> NDArray[np.float32]:
        """Return concatenated PLE head vectors for a token batch."""

        row_ids = self.hash_ids(input_ids, previous_context=previous_context)
        rows = self.gather(row_ids)
        return rows.reshape((*row_ids.shape[:-1], self.layout.embedding_dim))

    def lookup_raw(
        self,
        input_ids: ArrayLike,
        *,
        previous_context: ArrayLike | None = None,
    ) -> NDArray[Any]:
        """Return selected PLE vectors without widening BF16 to float32."""

        row_ids = self.hash_ids(input_ids, previous_context=previous_context)
        rows = self.gather_raw(row_ids)
        return rows.reshape((*row_ids.shape[:-1], self.layout.embedding_dim))

    @property
    def cache_info(self) -> dict[str, int]:
        return {
            "hits": self._cache.hits,
            "misses": self._cache.misses,
            "pages": self._cache.page_count,
            "bytes": self._cache.size_bytes,
            "max_bytes": self._cache.max_bytes,
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        state = getattr(self, "_state", None)
        if state is None:
            return
        with state:
            if self._closed:
                return
            self._closed = True
            while self._active_reads:
                state.wait()
        self._cache.clear()
        for shard in getattr(self, "_shards", ()):
            shard.close()

    def __enter__(self) -> Qwen4ExpPLESSDPool:
        self._check_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


__all__ = [
    "OFFICIAL_PLE_LAYOUT",
    "OFFICIAL_PLE_Q8_LAYOUT",
    "PLE_MLX_Q8_BITS",
    "PLE_MLX_Q8_DTYPE",
    "PLE_MLX_Q8_FORMAT",
    "PLE_MLX_Q8_GROUP_SIZE",
    "PLE_MLX_Q8_METADATA",
    "PLE_MLX_Q8_MODE",
    "PLEArtifactError",
    "PLELayout",
    "PLE_PREFIX",
    "PLE_SHARD_COUNT",
    "PLE_TABLE_PREFIX",
    "Qwen4ExpPLESSDPool",
    "mlx_q8_index_metadata",
]
