# SPDX-License-Identifier: Apache-2.0
"""PLE mmap residency, registry, cache-clear, and teardown contracts."""

import asyncio
import gc
import json
import mmap
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import omlx.admin.routes as admin_routes
from omlx.patches.qwen4_exp import model as model_module
from omlx.patches.qwen4_exp import ple as ple_module
from omlx.patches.qwen4_exp.model import Model, Qwen4ExpPLELayer
from omlx.patches.qwen4_exp.ple import (
    PLE_RESIDENCY_ENV,
    PLE_RESIDENCY_MEMORY,
    PLE_RESIDENCY_METADATA_KEY,
    PLE_RESIDENCY_SSD_MMAP,
    PLEArtifactError,
    Qwen4ExpPLESSDPool,
    active_ple_pool_info,
    clear_active_ple_page_caches,
    drop_active_ple_resident_pages,
)
from tests.test_qwen4_exp_ple import TINY_LAYOUT, _build_artifact


def _set_index_residency(model_dir: Path, value: object) -> None:
    path = model_dir / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index.setdefault("metadata", {})[PLE_RESIDENCY_METADATA_KEY] = value
    path.write_text(json.dumps(index))


def test_ssd_mmap_is_default_and_keeps_all_table_mappings_lazy(tmp_path: Path) -> None:
    _build_artifact(tmp_path)
    with Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT) as pool:
        assert pool.residency_policy == PLE_RESIDENCY_SSD_MMAP
        assert pool.cache_info["mapped_bytes"] == 0
        assert pool.cache_info["prefetched_bytes"] == 0
        assert all(shard._mapping is None for shard in pool._shards)


def test_memory_policy_maps_every_shard_and_issues_willneed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_artifact(tmp_path)
    calls: list[tuple[str, int]] = []

    def record_advice(self, advice: int) -> int:
        calls.append((self.descriptor.name, advice))
        return self.descriptor.byte_length

    monkeypatch.setattr(ple_module._MappedTensor, "advise", record_advice)
    with Qwen4ExpPLESSDPool(
        tmp_path,
        layout=TINY_LAYOUT,
        residency_policy="file-cache-pinned",
    ) as pool:
        assert pool.residency_policy == PLE_RESIDENCY_MEMORY
        assert all(shard._mapping is not None for shard in pool._shards)
        assert len(calls) == 128
        if hasattr(mmap, "MADV_WILLNEED"):
            assert {advice for _, advice in calls} == {mmap.MADV_WILLNEED}
            assert pool.cache_info["prefetched_bytes"] == sum(
                shard.descriptor.byte_length for shard in pool._shards
            )
        assert pool.cache_info["mapped_bytes"] > 0


def test_policy_precedence_explicit_then_environment_then_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_artifact(tmp_path)
    _set_index_residency(tmp_path, "memory")
    monkeypatch.setenv(PLE_RESIDENCY_ENV, "ssd_mmap")

    with Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT) as pool:
        assert pool.residency_policy == PLE_RESIDENCY_SSD_MMAP
    with Qwen4ExpPLESSDPool(
        tmp_path, layout=TINY_LAYOUT, residency_policy="memory"
    ) as pool:
        assert pool.residency_policy == PLE_RESIDENCY_MEMORY

    monkeypatch.delenv(PLE_RESIDENCY_ENV)
    with Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT) as pool:
        assert pool.residency_policy == PLE_RESIDENCY_MEMORY


def test_invalid_policy_fails_before_table_mapping(tmp_path: Path) -> None:
    _build_artifact(tmp_path)
    with pytest.raises(PLEArtifactError, match="expected 'ssd_mmap' or 'memory'"):
        Qwen4ExpPLESSDPool(
            tmp_path, layout=TINY_LAYOUT, residency_policy="copy-into-mlx"
        )


def test_model_text_config_exposes_explicit_residency_seam() -> None:
    text_args = SimpleNamespace()
    args = SimpleNamespace(
        model_type="qwen4_exp",
        text_config={"ple_residency_policy": "memory"},
        quantization=None,
        quantization_config=None,
    )
    with (
        patch.object(model_module.TextModelArgs, "from_dict", return_value=text_args),
        patch.object(model_module, "TextModel", return_value=SimpleNamespace()),
    ):
        Model(args)
    assert text_args.ple_residency_policy == "memory"


def test_registry_telemetry_and_hot_page_clear(tmp_path: Path) -> None:
    _build_artifact(tmp_path)
    baseline = active_ple_pool_info()["pools"]
    pool = Qwen4ExpPLESSDPool(
        tmp_path, layout=TINY_LAYOUT, rows_per_page=2, max_cache_bytes=4096
    )
    try:
        pool.gather([0])
        pool.gather([0])
        info = active_ple_pool_info()
        assert info["pools"] == baseline + 1
        assert info["hits"] >= 1
        assert info["misses"] >= 1
        assert info["pages"] >= 1
        assert info["bytes"] > 0

        cleared = clear_active_ple_page_caches()
        assert cleared["pools"] == baseline + 1
        assert cleared["pages"] >= 1
        assert cleared["bytes"] > 0
        assert pool.cache_info["pages"] == 0
    finally:
        pool.close()
        pool.close()
    assert active_ple_pool_info()["pools"] == baseline


def test_ssd_drop_discards_pages_but_never_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_artifact(tmp_path)
    pool = Qwen4ExpPLESSDPool(
        tmp_path, layout=TINY_LAYOUT, rows_per_page=2, max_cache_bytes=4096
    )
    pool.gather([0])
    mapped = pool._shards[0]
    calls: list[int] = []

    def record_advice(self, advice: int) -> int:
        calls.append(advice)
        return self.descriptor.byte_length

    monkeypatch.setattr(ple_module._MappedTensor, "advise", record_advice)
    try:
        report = drop_active_ple_resident_pages()
        assert report["pools"] >= 1
        assert report["pages"] >= 1
        assert report["bytes"] > 0
        if hasattr(mmap, "MADV_DONTNEED"):
            assert mmap.MADV_DONTNEED in calls
        assert mapped._mapping is not None
        assert (tmp_path / "model.safetensors").is_file()
        assert (tmp_path / "model.safetensors.index.json").is_file()
    finally:
        pool.close()


def test_ssd_admin_clear_reports_ple_counts_without_deleting_weights(
    tmp_path: Path,
) -> None:
    _build_artifact(tmp_path)
    pool = Qwen4ExpPLESSDPool(
        tmp_path, layout=TINY_LAYOUT, rows_per_page=2, max_cache_bytes=4096
    )
    pool.gather([0])
    empty_engine_pool = type(
        "Pool",
        (),
        {"get_status": lambda self: {"models": []}, "_entries": {}},
    )()
    try:
        with (
            patch.object(
                admin_routes, "_get_engine_pool", return_value=empty_engine_pool
            ),
            patch.object(admin_routes, "_get_global_settings", return_value=None),
        ):
            result = asyncio.run(admin_routes.clear_ssd_cache(is_admin=True))
        assert result["ple_pools"] >= 1
        assert result["ple_cleared_pages"] >= 1
        assert result["ple_cleared_bytes"] > 0
        assert (tmp_path / "model.safetensors").exists()
    finally:
        pool.close()


def test_model_close_and_layer_finalizer_release_pool_idempotently(
    tmp_path: Path,
) -> None:
    _build_artifact(tmp_path)
    baseline = active_ple_pool_info()["pools"]
    pool = Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT)

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    object.__setattr__(layer, "_pool", pool)
    object.__setattr__(layer, "_pool_finalizer", weakref.finalize(layer, pool.close))
    model = Model.__new__(Model)
    object.__setattr__(
        model,
        "language_model",
        SimpleNamespace(
            model=SimpleNamespace(layers=[None, SimpleNamespace(ple=layer)])
        ),
    )

    model.close()
    model.close()
    assert active_ple_pool_info()["pools"] == baseline

    # Exercise the same finalizer seam independently of explicit Model.close.
    second = Qwen4ExpPLESSDPool(tmp_path, layout=TINY_LAYOUT)
    owner = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    object.__setattr__(owner, "_pool", second)
    close = MagicMock(wraps=second.close)
    object.__setattr__(owner, "_pool_finalizer", weakref.finalize(owner, close))
    del owner
    gc.collect()
    close.assert_called_once_with()
    assert active_ple_pool_info()["pools"] == baseline
