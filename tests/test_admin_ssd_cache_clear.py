# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the admin SSD-cache filesystem fallback."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import omlx.admin.routes as admin_routes


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
