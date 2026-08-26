# SPDX-License-Identifier: Apache-2.0
"""Contract tests for Qwen4-Exp's SSD-backed PLE table."""

import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from omlx.patches.qwen4_exp import ple as ple_module
from omlx.patches.qwen4_exp.ple import (
    OFFICIAL_PLE_LAYOUT,
    OFFICIAL_PLE_Q8_LAYOUT,
    PLE_MLX_Q8_BITS,
    PLE_MLX_Q8_DTYPE,
    PLE_MLX_Q8_FORMAT,
    PLE_MLX_Q8_GROUP_SIZE,
    PLE_MLX_Q8_METADATA,
    PLE_MLX_Q8_MODE,
    PLE_PREFIX,
    PLE_SHARD_COUNT,
    PLE_TABLE_PREFIX,
    PLEArtifactError,
    PLELayout,
    Qwen4ExpPLESSDPool,
    mlx_q8_index_metadata,
)

TINY_LAYOUT = PLELayout(
    unigram_vocab_size=64,
    ngram_vocab_size_base=3,
    head_dim=2,
    eos_token_id=63,
    seed=1234,
    make_vocab_divisible_by=128,
    table_dtype="F32",
)

TINY_Q8_LAYOUT = PLELayout(
    unigram_vocab_size=64,
    ngram_vocab_size_base=3,
    head_dim=32,
    eos_token_id=63,
    seed=1234,
    make_vocab_divisible_by=128,
    table_dtype=PLE_MLX_Q8_DTYPE,
)


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]
) -> None:
    # This is the default emitted by mx.save_safetensors when no file-level
    # metadata is supplied.
    header = {"__metadata__": None}
    offset = 0
    for name, (dtype_name, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype_name,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for _, (_, _, raw) in tensors.items():
            stream.write(raw)


def _build_artifact(
    model_dir: Path,
    *,
    layout: PLELayout = TINY_LAYOUT,
    drop: str | None = None,
    extra_table_name: str | None = None,
    table_shape_override: tuple[int, int] | None = None,
    table_dtype_override: str | None = None,
) -> dict[int, np.ndarray]:
    model_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    expected = {}
    table_dtype = table_dtype_override or layout.table_dtype
    for shard_index in range(PLE_SHARD_COUNT):
        name = f"{PLE_TABLE_PREFIX}.shard_{shard_index}.weight"
        shape = table_shape_override or (layout.rows_per_shard, layout.head_dim)
        global_start = shard_index * layout.rows_per_shard
        values = np.stack(
            [
                np.arange(global_start, global_start + shape[0], dtype=np.float32),
                np.arange(global_start, global_start + shape[0], dtype=np.float32)
                + 0.25,
            ],
            axis=1,
        )
        if shape[1] != 2:
            values = np.resize(values, shape).astype(np.float32)
        if table_dtype == "BF16":
            raw = (values.view(np.uint32) >> np.uint32(16)).astype("<u2").tobytes()
        elif table_dtype == "F16":
            raw = values.astype("<f2").tobytes()
        else:
            raw = values.astype("<f4").tobytes()
        tensors[name] = (table_dtype, shape, raw)
        expected[shard_index] = values

    auxiliaries = {
        f"{PLE_PREFIX}.layer_multipliers": layout.layer_multipliers,
        f"{PLE_PREFIX}.ngram_heads_offsets": layout.head_offsets,
        f"{PLE_PREFIX}.ngram_heads_vocab_sizes": layout.head_vocab_sizes,
    }
    for name, values in auxiliaries.items():
        tensors[name] = ("I64", values.shape, values.astype("<i8").tobytes())
    if drop is not None:
        tensors.pop(drop)
    if extra_table_name is not None:
        tensors[extra_table_name] = ("F32", (1, 2), np.zeros((1, 2), "<f4").tobytes())

    shard_path = model_dir / "model.safetensors"
    _write_safetensors(shard_path, tensors)
    weight_map = {name: shard_path.name for name in tensors}
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard_path.stat().st_size},
                "weight_map": weight_map,
            }
        )
    )
    return expected


def _float32_to_bf16_bytes(values: np.ndarray) -> bytes:
    return (
        (values.astype(np.float32).view(np.uint32) >> np.uint32(16))
        .astype("<u2")
        .tobytes()
    )


def _build_q8_artifact(
    model_dir: Path,
    *,
    drop: str | None = None,
    metadata_overrides: dict[str, object] | None = None,
    omit_metadata: bool = False,
    scales_shape_override: tuple[int, int] | None = None,
    weight_shape_override: tuple[int, int] | None = None,
    weight_dtype_override: str | None = None,
    biases_dtype_override: str | None = None,
) -> dict[int, np.ndarray]:
    """Write the exact three-tensor MLX affine-Q8 embedding representation."""

    layout = TINY_Q8_LAYOUT
    model_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    expected = {}
    packed_columns = layout.head_dim * PLE_MLX_Q8_BITS // 32
    group_columns = layout.head_dim // PLE_MLX_Q8_GROUP_SIZE
    shifts = np.asarray((0, 8, 16, 24), dtype=np.uint32)

    for shard_index in range(PLE_SHARD_COUNT):
        global_start = shard_index * layout.rows_per_shard
        row_numbers = np.arange(
            global_start, global_start + layout.rows_per_shard, dtype=np.uint32
        )
        codes = (
            row_numbers[:, None] * np.uint32(17)
            + np.arange(layout.head_dim, dtype=np.uint32)[None, :] * np.uint32(7)
        ) & np.uint32(0xFF)
        packed = np.bitwise_or.reduce(
            codes.reshape(layout.rows_per_shard, packed_columns, 4)
            << shifts[None, None, :],
            axis=-1,
        ).astype("<u4")
        scales = (
            0.25 * (1 + (row_numbers[:, None] + np.arange(group_columns)[None, :]) % 4)
        ).astype(np.float32)
        biases = (
            -1.0
            + 0.5 * ((row_numbers[:, None] + np.arange(group_columns)[None, :]) % 3)
        ).astype(np.float32)
        expected[shard_index] = (
            codes.astype(np.float32).reshape(
                layout.rows_per_shard, group_columns, PLE_MLX_Q8_GROUP_SIZE
            )
            * scales[..., None]
            + biases[..., None]
        ).reshape(layout.rows_per_shard, layout.head_dim)

        prefix = f"{PLE_TABLE_PREFIX}.shard_{shard_index}"
        weight_shape = weight_shape_override or packed.shape
        weight_raw = np.resize(packed, weight_shape).astype("<u4").tobytes()
        tensors[f"{prefix}.weight"] = (
            weight_dtype_override or "U32",
            weight_shape,
            weight_raw,
        )
        scales_shape = scales_shape_override or scales.shape
        scales_raw = np.resize(scales, scales_shape)
        tensors[f"{prefix}.scales"] = (
            "BF16",
            scales_shape,
            _float32_to_bf16_bytes(scales_raw),
        )
        tensors[f"{prefix}.biases"] = (
            biases_dtype_override or "BF16",
            biases.shape,
            _float32_to_bf16_bytes(biases),
        )

    auxiliaries = {
        f"{PLE_PREFIX}.layer_multipliers": layout.layer_multipliers,
        f"{PLE_PREFIX}.ngram_heads_offsets": layout.head_offsets,
        f"{PLE_PREFIX}.ngram_heads_vocab_sizes": layout.head_vocab_sizes,
    }
    for name, values in auxiliaries.items():
        tensors[name] = ("I64", values.shape, values.astype("<i8").tobytes())
    if drop is not None:
        tensors.pop(drop)

    shard_path = model_dir / "model.safetensors"
    _write_safetensors(shard_path, tensors)
    weight_map = {name: shard_path.name for name in tensors}
    index: dict[str, object] = {"weight_map": weight_map}
    if not omit_metadata:
        metadata = {
            **mlx_q8_index_metadata(layout),
            "total_size": shard_path.stat().st_size,
        }
        if metadata_overrides:
            metadata.update(metadata_overrides)
        index["metadata"] = metadata
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    return expected


def _reference_hash(
    input_ids: np.ndarray,
    previous_context: np.ndarray,
    layout: PLELayout,
) -> np.ndarray:
    history = np.concatenate((previous_context, input_ids), axis=1)
    shifted = []
    for shift in range(3):
        rows = []
        for row in history.tolist():
            shifted_row = []
            segment = []
            for token in row:
                segment.append(token)
                if shift == 0:
                    shifted_row.append(token)
                elif len(segment) > shift:
                    shifted_row.append(segment[-shift - 1])
                else:
                    shifted_row.append(layout.eos_token_id)
                if token == layout.eos_token_id:
                    segment = []
            rows.append(shifted_row)
        shifted.append(np.asarray(rows, dtype=np.int64))

    result = []
    for batch_index in range(input_ids.shape[0]):
        sequence = []
        for sequence_index in range(
            history.shape[1] - input_ids.shape[1], history.shape[1]
        ):
            heads = []
            for ngram in (2, 3):
                mixed = int(shifted[0][batch_index, sequence_index]) * int(
                    layout.layer_multipliers[0]
                )
                for position in range(1, ngram):
                    mixed ^= int(shifted[position][batch_index, sequence_index]) * int(
                        layout.layer_multipliers[position]
                    )
                start = (ngram - 2) * 8
                for head_index in range(start, start + 8):
                    heads.append(
                        mixed % int(layout.head_vocab_sizes[head_index])
                        + int(layout.head_offsets[head_index])
                    )
            sequence.append(heads)
        result.append(sequence)
    return np.asarray(result, dtype=np.int64)


def test_official_layout_and_exact_checkpoint_binding() -> None:
    assert PLE_PREFIX == "model.language_model.layers.1.ple.ple_embedding"
    assert PLE_SHARD_COUNT == 128
    assert OFFICIAL_PLE_LAYOUT.rows_per_shard == 2_500_012
    assert OFFICIAL_PLE_LAYOUT.head_dim == 160
    assert OFFICIAL_PLE_LAYOUT.embedding_dim == 2560
    assert OFFICIAL_PLE_LAYOUT.padded_vocab_size == 320_001_536
    assert OFFICIAL_PLE_LAYOUT.table_dtype == "BF16"
    assert OFFICIAL_PLE_LAYOUT.layer_multipliers.tolist() == [
        23_703_573_157_769,
        20_109_073_645_365,
        8_052_911_324_071,
    ]
    assert OFFICIAL_PLE_LAYOUT.head_offsets.tolist() == [
        0,
        20_000_003,
        40_000_026,
        60_000_059,
        80_000_106,
        100_000_165,
        120_000_228,
        140_000_297,
        160_000_374,
        180_000_455,
        200_000_548,
        220_000_655,
        240_000_802,
        260_000_955,
        280_001_114,
        300_001_275,
    ]


def test_official_mlx_q8_layout_contract() -> None:
    assert OFFICIAL_PLE_Q8_LAYOUT.table_dtype == PLE_MLX_Q8_DTYPE
    assert OFFICIAL_PLE_Q8_LAYOUT.rows_per_shard == 2_500_012
    assert OFFICIAL_PLE_Q8_LAYOUT.head_dim == 160
    assert PLE_MLX_Q8_BITS == 8
    assert PLE_MLX_Q8_GROUP_SIZE == 32
    assert PLE_MLX_Q8_MODE == "affine"
    assert PLE_MLX_Q8_FORMAT == "mlx-affine-q8-v1"
    assert PLE_MLX_Q8_METADATA["qwen4_exp_ple_scale_dtype"] == "BF16"
    # MLX packs four 8-bit values per uint32 and stores one scale/bias per
    # group of 32: [rows, 40] + [rows, 5] + [rows, 5].
    assert OFFICIAL_PLE_Q8_LAYOUT.head_dim * PLE_MLX_Q8_BITS // 32 == 40
    assert OFFICIAL_PLE_Q8_LAYOUT.head_dim // PLE_MLX_Q8_GROUP_SIZE == 5


def test_q8_lookup_matches_known_affine_reference(tmp_path: Path) -> None:
    expected = _build_q8_artifact(tmp_path)
    with Qwen4ExpPLESSDPool(tmp_path, layout=TINY_Q8_LAYOUT, rows_per_page=2) as pool:
        ids = np.asarray(
            [0, 3, 4, 17, TINY_Q8_LAYOUT.padded_vocab_size - 1],
            dtype=np.int64,
        )
        got = pool.gather(ids)
        wanted = np.stack(
            [
                expected[0][0],
                expected[0][3],
                expected[1][0],
                expected[4][1],
                expected[127][-1],
            ]
        )
        assert np.array_equal(got, wanted)
        with pytest.raises(TypeError, match="three storage tensors"):
            pool.gather_raw(ids)


def test_constructor_auto_detects_q8_layout_from_exact_metadata(tmp_path: Path) -> None:
    expected = _build_q8_artifact(tmp_path)
    with Qwen4ExpPLESSDPool(tmp_path) as pool:
        assert pool.layout == TINY_Q8_LAYOUT
        assert np.array_equal(
            pool.gather([0, 4]), np.stack([expected[0][0], expected[1][0]])
        )


def test_q8_token_lookup_uses_only_hashed_rows(tmp_path: Path) -> None:
    expected = _build_q8_artifact(tmp_path)
    tokens = np.asarray([[1, 2, 3], [4, 63, 5]], dtype=np.int64)
    with Qwen4ExpPLESSDPool(tmp_path, layout=TINY_Q8_LAYOUT) as pool:
        row_ids = pool.hash_ids(tokens)
        got = pool.lookup(tokens)
    wanted_rows = []
    for global_row in row_ids.reshape(-1):
        shard, local = divmod(int(global_row), TINY_Q8_LAYOUT.rows_per_shard)
        wanted_rows.append(expected[shard][local])
    wanted = np.stack(wanted_rows).reshape(2, 3, TINY_Q8_LAYOUT.embedding_dim)
    assert np.array_equal(got, wanted)


def test_q8_reads_only_selected_pages_and_bounds_combined_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_q8_artifact(tmp_path)
    calls = []
    original = ple_module._MappedTensor.read_rows

    def tracked_read_rows(self, start, stop):
        calls.append((self.descriptor.name, start, stop))
        return original(self, start, stop)

    monkeypatch.setattr(ple_module._MappedTensor, "read_rows", tracked_read_rows)
    # Two rows: packed uint32[8] + BF16 scales[1] + BF16 biases[1].
    page_bytes = 2 * (8 * 4 + 1 * 2 + 1 * 2)
    with Qwen4ExpPLESSDPool(
        tmp_path,
        layout=TINY_Q8_LAYOUT,
        rows_per_page=2,
        max_cache_bytes=page_bytes,
    ) as pool:
        assert all(
            shard.weight._mapping is None
            and shard.scales._mapping is None
            and shard.biases._mapping is None
            for shard in pool._shards
        )
        pool.gather([0, 1, 4, TINY_Q8_LAYOUT.padded_vocab_size - 1])
        assert calls
        assert all(stop - start <= 2 for _, start, stop in calls)
        assert {name.rsplit(".", 1)[-1] for name, _, _ in calls} == {
            "weight",
            "scales",
            "biases",
        }
        assert pool.cache_info["bytes"] <= page_bytes
        assert pool.cache_info["pages"] <= 1


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"qwen4_exp_ple_bits": 4}, "qwen4_exp_ple_bits"),
        ({"qwen4_exp_ple_group_size": 64}, "qwen4_exp_ple_group_size"),
        ({"qwen4_exp_ple_mode": "mxfp8"}, "qwen4_exp_ple_mode"),
        ({"qwen4_exp_ple_scale_dtype": "F16"}, "qwen4_exp_ple_scale_dtype"),
        ({"qwen4_exp_ple_layer": 2}, "qwen4_exp_ple_layer"),
        ({"qwen4_exp_ple_shard_count": 127}, "qwen4_exp_ple_shard_count"),
        ({"qwen4_exp_ple_rows_per_shard": 5}, "rows_per_shard"),
        ({"qwen4_exp_ple_head_dim": 64}, "head_dim"),
        ({"qwen4_exp_ple_seed": None}, "qwen4_exp_ple_seed"),
        ({"qwen4_exp_ple_bits": True}, "qwen4_exp_ple_bits"),
        ({"qwen4_exp_ple_unknown": 1}, "unexpected MLX Q8 PLE metadata"),
    ],
)
def test_q8_malformed_metadata_fails_closed(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    _build_q8_artifact(tmp_path, metadata_overrides=overrides)
    with pytest.raises(PLEArtifactError, match=match):
        Qwen4ExpPLESSDPool(tmp_path, layout=TINY_Q8_LAYOUT)


def test_q8_missing_metadata_fails_closed(tmp_path: Path) -> None:
    _build_q8_artifact(tmp_path, omit_metadata=True)
    with pytest.raises(PLEArtifactError, match="requires index metadata"):
        Qwen4ExpPLESSDPool(tmp_path, layout=TINY_Q8_LAYOUT)


def test_q8_missing_bias_fails_closed(tmp_path: Path) -> None:
    missing = f"{PLE_TABLE_PREFIX}.shard_73.biases"
    _build_q8_artifact(tmp_path, drop=missing)
    with pytest.raises(PLEArtifactError, match="missing 1 tensor"):
        Qwen4ExpPLESSDPool(tmp_path, layout=TINY_Q8_LAYOUT)


def test_q8_scale_shape_fails_closed(tmp_path: Path) -> None:
    _build_q8_artifact(tmp_path, scales_shape_override=(4, 2))
    with pytest.raises(PLEArtifactError, match="must have shape"):
        Qwen4ExpPLESSDPool(tmp_path, layout=TINY_Q8_LAYOUT)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"weight_shape_override": (4, 7)}, "must have shape"),
        ({"weight_dtype_override": "F32"}, "must have dtype U32"),
        ({"biases_dtype_override": "F16"}, "must have dtype BF16"),
    ],
)
def test_q8_packed_weight_and_affine_tensors_are_strict(
    tmp_path: Path, kwargs: dict[str, object], match: str
) -> None:
    _build_q8_artifact(tmp_path, **kwargs)
    with pytest.raises(PLEArtifactError, match=match):
        Qwen4ExpPLESSDPool(tmp_path, layout=TINY_Q8_LAYOUT)


def test_constructor_validates_all_128_shards_without_mapping_table(
    tmp_path: Path,
) -> None:
    _build_artifact(tmp_path)
    pool = Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT)
    try:
        assert len(pool._shards) == 128
        assert all(shard._mapping is None for shard in pool._shards)
        assert np.array_equal(pool.layer_multipliers, TINY_LAYOUT.layer_multipliers)
        assert np.array_equal(pool.head_offsets, TINY_LAYOUT.head_offsets)
        assert np.array_equal(pool.head_vocab_sizes, TINY_LAYOUT.head_vocab_sizes)
    finally:
        pool.close()


def test_gather_reads_only_selected_pages_and_bounds_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _build_artifact(tmp_path)
    calls = []
    original = ple_module._MappedTensor.read_rows

    def tracked_read_rows(self, start, stop):
        calls.append((self.descriptor.name, start, stop))
        return original(self, start, stop)

    monkeypatch.setattr(ple_module._MappedTensor, "read_rows", tracked_read_rows)
    page_bytes = 2 * TINY_LAYOUT.head_dim * np.dtype("f4").itemsize
    with Qwen4ExpPLESSDPool(
        tmp_path,
        layout=TINY_LAYOUT,
        rows_per_page=2,
        max_cache_bytes=page_bytes,
    ) as pool:
        rows = np.asarray([0, 1, 4, TINY_LAYOUT.padded_vocab_size - 1])
        got = pool.gather(rows)
        wanted = np.stack(
            [
                expected[0][0],
                expected[0][1],
                expected[1][0],
                expected[127][-1],
            ]
        )
        assert np.array_equal(got, wanted)
        assert calls
        assert all(stop - start <= 2 for _, start, stop in calls)
        assert pool.cache_info["bytes"] <= page_bytes
        assert pool.cache_info["pages"] <= 1


def test_cache_hits_and_prefetch_executor_seam(tmp_path: Path) -> None:
    _build_artifact(tmp_path)
    with Qwen4ExpPLESSDPool(
        tmp_path, layout=TINY_LAYOUT, rows_per_page=2, max_cache_bytes=1024
    ) as pool:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = pool.prefetch([0, 1, 8], executor=executor)
            assert futures
            for future in futures:
                future.result()
        before = pool.cache_info.copy()
        pool.gather([0, 1, 8])
        assert pool.cache_info["hits"] > before["hits"]
        pool.clear_cache()
        assert pool.cache_info["pages"] == 0
        assert pool.cache_info["bytes"] == 0


def test_hash_ids_matches_independent_bigram_trigram_reference(tmp_path: Path) -> None:
    _build_artifact(tmp_path)
    input_ids = np.asarray([[1, 2, 63, 3], [5, 6, 7, 8]], dtype=np.int64)
    context = np.asarray([[9, 10], [63, 11]], dtype=np.int64)
    with Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT) as pool:
        got = pool.hash_ids(input_ids, previous_context=context)
    expected = _reference_hash(input_ids, context, TINY_LAYOUT)
    assert got.shape == (2, 4, 16)
    assert np.array_equal(got, expected)
    assert np.all(got[..., :8] < TINY_LAYOUT.head_offsets[8])
    assert np.all(got[..., 8:] >= TINY_LAYOUT.head_offsets[8])


def test_lookup_flattens_16_selected_heads_only(tmp_path: Path) -> None:
    _build_artifact(tmp_path)
    tokens = np.asarray([[1, 2, 3]], dtype=np.int64)
    with Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT) as pool:
        row_ids = pool.hash_ids(tokens)
        expected = pool.gather(row_ids).reshape(1, 3, 32)
        got = pool.lookup(tokens)
        empty = pool.lookup(np.empty((2, 0), dtype=np.int64))
    assert np.array_equal(got, expected)
    assert got.shape == (1, 3, TINY_LAYOUT.embedding_dim)
    assert empty.shape == (2, 0, TINY_LAYOUT.embedding_dim)


def test_bfloat16_rows_decode_after_selection(tmp_path: Path) -> None:
    layout = PLELayout(
        unigram_vocab_size=64,
        ngram_vocab_size_base=3,
        head_dim=2,
        eos_token_id=63,
        table_dtype="BF16",
    )
    expected = _build_artifact(tmp_path, layout=layout)
    with Qwen4ExpPLESSDPool(tmp_path, layout=layout, rows_per_page=1) as pool:
        ids = [0, layout.rows_per_shard, layout.padded_vocab_size - 1]
        raw = pool.gather_raw(ids)
        got = pool.gather(ids)
    wanted = np.stack([expected[0][0], expected[1][0], expected[127][-1]])
    wanted = ((wanted.view(np.uint32) >> np.uint32(16)) << np.uint32(16)).view(
        np.float32
    )
    assert raw.dtype == np.dtype("<u2")
    assert np.array_equal(
        (raw.astype(np.uint32) << np.uint32(16)).view(np.float32), wanted
    )
    assert np.array_equal(got, wanted)


def test_missing_shard_fails_closed(tmp_path: Path) -> None:
    missing = f"{PLE_TABLE_PREFIX}.shard_73.weight"
    _build_artifact(tmp_path, drop=missing)
    with pytest.raises(PLEArtifactError, match="missing 1 tensor"):
        Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT)


def test_unexpected_shard_number_fails_closed(tmp_path: Path) -> None:
    _build_artifact(
        tmp_path,
        extra_table_name=f"{PLE_TABLE_PREFIX}.shard_128.weight",
    )
    with pytest.raises(PLEArtifactError, match="unexpected Qwen4-Exp PLE"):
        Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT)


@pytest.mark.parametrize(
    ("shape", "dtype_name", "match"),
    [
        ((3, 2), None, "must have shape"),
        (None, "F16", "must have dtype F32"),
    ],
)
def test_table_shape_and_dtype_are_strict(
    tmp_path: Path,
    shape: tuple[int, int] | None,
    dtype_name: str | None,
    match: str,
) -> None:
    _build_artifact(
        tmp_path,
        table_shape_override=shape,
        table_dtype_override=dtype_name,
    )
    with pytest.raises(PLEArtifactError, match=match):
        Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT)


def test_hash_metadata_mismatch_fails_closed(tmp_path: Path) -> None:
    _build_artifact(tmp_path)
    path = tmp_path / "model.safetensors"
    raw = bytearray(path.read_bytes())
    header_size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_size])
    name = f"{PLE_PREFIX}.layer_multipliers"
    start = 8 + header_size + header[name]["data_offsets"][0]
    raw[start : start + 8] = np.asarray([2], dtype="<i8").tobytes()
    path.write_bytes(raw)
    with pytest.raises(PLEArtifactError, match="layer_multipliers"):
        Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT)


def test_file_change_before_first_lookup_fails_closed(tmp_path: Path) -> None:
    _build_artifact(tmp_path)
    pool = Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT)
    path = tmp_path / "model.safetensors"
    with path.open("ab") as stream:
        stream.write(b"changed")
    try:
        with pytest.raises(PLEArtifactError, match="changed after validation"):
            pool.gather([0])
    finally:
        pool.close()


def test_closed_pool_and_invalid_row_ids_fail_closed(tmp_path: Path) -> None:
    _build_artifact(tmp_path)
    pool = Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT)
    with pytest.raises(IndexError):
        pool.gather([-1])
    with pytest.raises(IndexError):
        pool.prefetch([TINY_LAYOUT.padded_vocab_size])
    with pytest.raises(TypeError):
        pool.gather([1.5])
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.gather([0])
