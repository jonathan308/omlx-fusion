# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /admin/api/ssd-cache/max-size.

Live SSD-cap tuning: clamped into the 50-400 GiB rails, pushed into every
loaded model's PagedSSDCacheManager without a reload, and persisted through
the global settings file.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.admin.routes as admin_routes
import omlx.server  # noqa: F401 — triggers set_admin_getters
from omlx.admin.routes import SSDCacheMaxSizeRequest

MODEL_ID = "test-model"
GiB = 1024**3


def _run(request):
    return asyncio.run(admin_routes.set_ssd_cache_max_size(request, is_admin=True))


def _pool(models, entries=None, scheduler_config=None):
    pool = MagicMock(spec=[])
    pool.get_status = MagicMock(return_value={"models": models})
    pool._entries = entries or {}
    pool._scheduler_config = scheduler_config or SimpleNamespace(
        paged_ssd_cache_max_size=0
    )
    return pool


def _loaded_entry(set_max_size_mock):
    """Build the entry.engine._engine.engine.scheduler chain a loaded model has."""
    scheduler = SimpleNamespace(
        paged_ssd_cache_manager=SimpleNamespace(set_max_size=set_max_size_mock),
    )
    core = SimpleNamespace(scheduler=scheduler)
    return SimpleNamespace(engine=SimpleNamespace(_engine=SimpleNamespace(engine=core)))


def _settings():
    return SimpleNamespace(
        cache=SimpleNamespace(ssd_cache_max_size="auto"),
        save=MagicMock(),
    )


class _RouteEnv:
    """Patch the pool + settings getters the route resolves lazily."""

    def __init__(self, pool, settings):
        self._patches = [
            patch.object(admin_routes, "_get_engine_pool", return_value=pool),
            patch.object(admin_routes, "_get_global_settings", return_value=settings),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False


def test_an_unparseable_size_is_a_400():
    with pytest.raises(HTTPException) as excinfo:
        _run(SSDCacheMaxSizeRequest(max_size="lots"))
    assert excinfo.value.status_code == 400


def test_a_nonpositive_size_is_a_400():
    with pytest.raises(HTTPException) as excinfo:
        _run(SSDCacheMaxSizeRequest(max_size="0"))
    assert excinfo.value.status_code == 400


def test_loaded_managers_are_retuned_without_a_reload():
    set_max = MagicMock(return_value=250 * GiB)
    pool = _pool(
        models=[{"id": MODEL_ID, "loaded": True}],
        entries={MODEL_ID: _loaded_entry(set_max)},
    )
    settings = _settings()
    with _RouteEnv(pool, settings):
        result = _run(SSDCacheMaxSizeRequest(max_size="250GB"))

    set_max.assert_called_once_with(250 * GiB)
    assert result["models_updated"] == 1
    assert result["effective_bytes"] == 250 * GiB
    assert result["clamped"] is False
    # Future loads inherit the retune through the pool's scheduler config.
    assert pool._scheduler_config.paged_ssd_cache_max_size == 250 * GiB
    # ... and restarts inherit it through the settings file.
    assert settings.cache.ssd_cache_max_size == "250GB"
    settings.save.assert_called_once_with()
    assert result["persisted"] is True


def test_out_of_rail_values_clamp_instead_of_rejecting():
    pool = _pool(models=[])
    settings = _settings()
    with _RouteEnv(pool, settings):
        low = _run(SSDCacheMaxSizeRequest(max_size="10GB"))
        high = _run(SSDCacheMaxSizeRequest(max_size="1TB", persist=False))

    assert low["effective_bytes"] == 50 * GiB and low["clamped"] is True
    assert settings.cache.ssd_cache_max_size == "50GB"
    assert high["effective_bytes"] == 400 * GiB and high["clamped"] is True


def test_persist_false_leaves_the_settings_file_alone():
    settings = _settings()
    with _RouteEnv(_pool(models=[]), settings):
        result = _run(SSDCacheMaxSizeRequest(max_size="200GB", persist=False))

    assert result["persisted"] is False
    assert settings.cache.ssd_cache_max_size == "auto"
    settings.save.assert_not_called()


def test_a_persistence_failure_still_reports_the_live_retune():
    settings = _settings()
    settings.save.side_effect = OSError("read-only filesystem")
    with _RouteEnv(_pool(models=[]), settings):
        result = _run(SSDCacheMaxSizeRequest(max_size="200GB"))

    assert result["status"] == "ok"
    assert result["persisted"] is False
    assert result["effective_bytes"] == 200 * GiB


def test_no_loaded_models_still_updates_the_pool_config():
    pool = _pool(models=[])
    with _RouteEnv(pool, _settings()):
        result = _run(SSDCacheMaxSizeRequest(max_size="150GB"))

    assert result["models_updated"] == 0
    assert pool._scheduler_config.paged_ssd_cache_max_size == 150 * GiB


def test_a_manager_without_live_tuning_is_skipped():
    scheduler = SimpleNamespace(paged_ssd_cache_manager=SimpleNamespace())
    core = SimpleNamespace(scheduler=scheduler)
    entry = SimpleNamespace(
        engine=SimpleNamespace(_engine=SimpleNamespace(engine=core))
    )
    pool = _pool(models=[{"id": MODEL_ID, "loaded": True}], entries={MODEL_ID: entry})
    with _RouteEnv(pool, _settings()):
        result = _run(SSDCacheMaxSizeRequest(max_size="150GB"))

    assert result["models_updated"] == 0
    assert result["status"] == "ok"
