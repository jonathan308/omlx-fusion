# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the admin SSD-cache filesystem fallback."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import omlx.admin.routes as admin_routes
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager


def _settings(cache_dir: Path):
    cache = SimpleNamespace(get_ssd_cache_dir=lambda _base_path: cache_dir)
    return SimpleNamespace(cache=cache, base_path=cache_dir.parent)


@pytest.mark.asyncio
async def test_live_manager_skips_unsynchronized_filesystem_fallback(
    tmp_path: Path, monkeypatch
):
    """A post-clear temp or final file belongs to the new write generation."""
    cache_dir = tmp_path / "cache"
    subdir = cache_dir / "a"
    subdir.mkdir(parents=True)
    final_path = subdir / ("a" * 64 + ".safetensors")
    temp_path = subdir / ("b" * 64 + "_tmp.safetensors")
    final_path.write_bytes(b"orphan-final")
    temp_path.write_bytes(b"live-writer-temp")

    manager = SimpleNamespace(clear=MagicMock(return_value=0))
    scheduler = SimpleNamespace(paged_ssd_cache_manager=manager)
    monkeypatch.setattr(
        admin_routes,
        "_iter_loaded_schedulers",
        lambda: iter([("loaded-model", scheduler)]),
    )
    monkeypatch.setattr(
        admin_routes,
        "_get_global_settings",
        lambda: _settings(cache_dir),
    )

    result = await admin_routes.clear_ssd_cache(is_admin=True)

    manager.clear.assert_called_once_with()
    assert final_path.read_bytes() == b"orphan-final"
    assert temp_path.read_bytes() == b"live-writer-temp"
    assert result == {"status": "ok", "total_deleted": 0}


@pytest.mark.asyncio
async def test_live_managers_clear_at_one_lifecycle_boundary(
    tmp_path: Path, monkeypatch
):
    """Admin uses clear_many once instead of releasing managers sequentially."""
    cache_dir = tmp_path / "cache"
    clear_many = MagicMock(return_value=4)
    manager_a = SimpleNamespace(clear=MagicMock(), clear_many=clear_many)
    manager_b = SimpleNamespace(clear=MagicMock())
    schedulers = [
        ("model-a", SimpleNamespace(paged_ssd_cache_manager=manager_a)),
        ("model-b", SimpleNamespace(paged_ssd_cache_manager=manager_b)),
    ]
    monkeypatch.setattr(
        admin_routes,
        "_iter_loaded_schedulers",
        lambda: iter(schedulers),
    )
    monkeypatch.setattr(
        admin_routes,
        "_get_global_settings",
        lambda: _settings(cache_dir),
    )

    result = await admin_routes.clear_ssd_cache(is_admin=True)

    clear_many.assert_called_once_with(
        [manager_a, manager_b],
        sweep_untracked=True,
    )
    manager_a.clear.assert_not_called()
    manager_b.clear.assert_not_called()
    assert result == {"status": "ok", "total_deleted": 4}


@pytest.mark.asyncio
async def test_live_manager_clear_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch
):
    """A failed synchronized clear must not fall through to a raw sweep."""
    cache_dir = tmp_path / "cache"
    subdir = cache_dir / "e"
    subdir.mkdir(parents=True)
    orphan_path = subdir / ("e" * 64 + ".safetensors")
    orphan_path.write_bytes(b"must-remain-after-failed-clear")

    manager = SimpleNamespace(clear=MagicMock(side_effect=RuntimeError("boom")))
    scheduler = SimpleNamespace(paged_ssd_cache_manager=manager)
    monkeypatch.setattr(
        admin_routes,
        "_iter_loaded_schedulers",
        lambda: iter([("broken-model", scheduler)]),
    )
    monkeypatch.setattr(
        admin_routes,
        "_get_global_settings",
        lambda: _settings(cache_dir),
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_routes.clear_ssd_cache(is_admin=True)

    assert exc_info.value.status_code == 500
    assert "incomplete" in str(exc_info.value.detail)
    assert orphan_path.read_bytes() == b"must-remain-after-failed-clear"


@pytest.mark.asyncio
async def test_live_manager_sweeps_untracked_files_inside_barrier(
    tmp_path: Path, monkeypatch
):
    """Crash-left final and temp files are removed by the fenced root sweep."""
    cache_dir = tmp_path / "cache"
    manager = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=1024**3,
    )
    subdir = cache_dir / "f"
    orphan_final = subdir / ("f" * 64 + ".safetensors")
    orphan_temp = subdir / ("0" * 64 + "_tmp.safetensors")
    orphan_final.write_bytes(b"orphan-final")
    orphan_temp.write_bytes(b"orphan-temp")
    scheduler = SimpleNamespace(paged_ssd_cache_manager=manager)
    monkeypatch.setattr(
        admin_routes,
        "_iter_loaded_schedulers",
        lambda: iter([("loaded-model", scheduler)]),
    )
    monkeypatch.setattr(
        admin_routes,
        "_get_global_settings",
        lambda: _settings(cache_dir),
    )

    try:
        result = await admin_routes.clear_ssd_cache(is_admin=True)
        assert not orphan_final.exists()
        assert not orphan_temp.exists()
        assert result == {"status": "ok", "total_deleted": 2}
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_no_live_manager_removes_orphaned_writer_temp(
    tmp_path: Path, monkeypatch
):
    """Without a live writer, phase 2 still cleans crash-left staging files."""
    cache_dir = tmp_path / "cache"
    subdir = cache_dir / "c"
    subdir.mkdir(parents=True)
    temp_path = subdir / ("c" * 64 + "_tmp.safetensors")
    final_path = subdir / ("d" * 64 + ".safetensors")
    temp_path.write_bytes(b"orphan-temp")
    final_path.write_bytes(b"orphan-final")

    monkeypatch.setattr(
        admin_routes,
        "_iter_loaded_schedulers",
        lambda: iter(()),
    )
    monkeypatch.setattr(
        admin_routes,
        "_get_global_settings",
        lambda: _settings(cache_dir),
    )

    result = await admin_routes.clear_ssd_cache(is_admin=True)

    assert not temp_path.exists()
    assert not final_path.exists()
    assert result == {"status": "ok", "total_deleted": 2}
