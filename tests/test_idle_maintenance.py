# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in keepwarm / graduated idle-TTL / eviction-discipline port.

Covers ``omlx/idle_maintenance.py`` (env config, the bounded Metal touch,
the keepwarm tracker, the graduated TTL policy), the engine-pool wiring
(cache-tier TTL drop, deep idle release, keepwarm dispatch, C4 recency
grace, C5 eviction attribution), the scheduler's ``release_idle_caches``,
and the distributed fan-out (share-channel sentinel, admin route, engine
POSTs). All mocked: no real models, no real Metal work.
"""

import concurrent.futures
import importlib
import io
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from omlx.engine_pool import EngineEntry, EnginePool
from omlx.idle_maintenance import (
    IdleMaintenanceSettings,
    KeepwarmTracker,
    eviction_grace_seconds,
    graduated_idle_action,
    metal_keepwarm_touch,
)
from omlx.scheduler import Scheduler

mlx_generate = importlib.import_module("mlx_lm.generate")
mlx_server = importlib.import_module("mlx_lm.server")


# ---------------------------------------------------------------------------
# Settings / env parsing
# ---------------------------------------------------------------------------


class TestIdleMaintenanceSettings:
    def test_defaults_are_all_off(self, monkeypatch):
        for name in (
            "OMLX_KEEPWARM_ENABLED",
            "OMLX_CACHE_TTL_SECONDS",
            "OMLX_IDLE_RELEASE_SECONDS",
            "OMLX_EVICTION_GRACE_SECONDS",
        ):
            monkeypatch.delenv(name, raising=False)
        settings = IdleMaintenanceSettings.from_env()
        assert settings.keepwarm_enabled is False
        assert settings.cache_ttl_seconds == 0.0
        assert settings.idle_release_seconds == 0.0
        assert settings.any_enabled is False
        assert eviction_grace_seconds() == 0.0
        # Production cadence defaults (ThunderMLX prod snapshot values).
        assert settings.keepwarm_interval_seconds == 10.0
        assert settings.keepwarm_idle_after_seconds == 10.0
        assert settings.keepwarm_matrix_size == 64
        assert settings.keepwarm_large_cache_tokens == 8192
        assert settings.keepwarm_large_interval_seconds == 30.0

    def test_env_parsing(self, monkeypatch):
        monkeypatch.setenv("OMLX_KEEPWARM_ENABLED", "1")
        monkeypatch.setenv("OMLX_CACHE_TTL_SECONDS", "1200")
        monkeypatch.setenv("OMLX_IDLE_RELEASE_SECONDS", "7200")
        monkeypatch.setenv("OMLX_EVICTION_GRACE_SECONDS", "30")
        monkeypatch.setenv("OMLX_KEEPWARM_INTERVAL_SECONDS", "15")
        settings = IdleMaintenanceSettings.from_env()
        assert settings.keepwarm_enabled is True
        assert settings.cache_ttl_seconds == 1200.0
        assert settings.idle_release_seconds == 7200.0
        assert settings.any_enabled is True
        assert settings.keepwarm_interval_seconds == 15.0
        assert eviction_grace_seconds() == 30.0

    def test_invalid_values_fall_back(self, monkeypatch):
        monkeypatch.setenv("OMLX_CACHE_TTL_SECONDS", "not-a-number")
        monkeypatch.setenv("OMLX_KEEPWARM_MATRIX_SIZE", "huge")
        settings = IdleMaintenanceSettings.from_env()
        assert settings.cache_ttl_seconds == 0.0
        assert settings.keepwarm_matrix_size == 64

    def test_interval_floor(self, monkeypatch):
        monkeypatch.setenv("OMLX_KEEPWARM_INTERVAL_SECONDS", "0.01")
        settings = IdleMaintenanceSettings.from_env()
        assert settings.keepwarm_interval_seconds == 0.25


# ---------------------------------------------------------------------------
# metal_keepwarm_touch
# ---------------------------------------------------------------------------


class _FakeStream:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return None


class _FakeMx:
    """The slice of mlx.core the keepwarm touch uses."""

    float16 = "float16"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def default_device(self):
        return "gpu0"

    def stream(self, device):
        self.calls.append(("stream", device))
        return _FakeStream()

    def ones(self, shape, dtype=None):
        self.calls.append(("ones", shape, dtype))
        if self.fail:
            raise RuntimeError("metal exploded")
        return MagicMock(name="array")

    def sum(self, value):
        return value

    def eval(self, value):
        self.calls.append(("eval",))


class TestMetalKeepwarmTouch:
    def test_ok_event_shape_and_defaults(self):
        event = metal_keepwarm_touch(_FakeMx(), reason="engine-pool keepwarm")
        assert event["ok"] is True
        assert event["action"] == "metal_keepwarm"
        assert event["reason"] == "engine-pool keepwarm"
        assert event["matrix_size"] == 64
        assert event["repeats"] == 1
        assert event["elapsed_ms"] >= 0

    def test_size_and_repeats_are_clamped(self):
        fake = _FakeMx()
        event = metal_keepwarm_touch(fake, size=100000, repeats=99)
        assert event["matrix_size"] == 1024
        assert event["repeats"] == 16
        event = metal_keepwarm_touch(_FakeMx(), size=1, repeats=0)
        assert event["matrix_size"] == 16
        assert event["repeats"] == 1

    def test_failure_never_raises(self):
        event = metal_keepwarm_touch(_FakeMx(fail=True))
        assert event["ok"] is False
        assert event["action"] == "metal_keepwarm_error"
        assert "metal exploded" in event["error"]


# ---------------------------------------------------------------------------
# KeepwarmTracker gating
# ---------------------------------------------------------------------------


def _settings(**overrides):
    base = {
        "keepwarm_enabled": True,
        "keepwarm_interval_seconds": 10.0,
        "keepwarm_idle_after_seconds": 10.0,
        "keepwarm_matrix_size": 64,
        "keepwarm_large_cache_tokens": 8192,
        "keepwarm_large_interval_seconds": 30.0,
        "keepwarm_slow_backoff_seconds": 60.0,
        "background_op_slow_seconds": 5.0,
    }
    base.update(overrides)
    return IdleMaintenanceSettings(**base)


def _touch_event(at, elapsed_ms=5.0, ok=True):
    return {"ok": ok, "at": at, "elapsed_ms": elapsed_ms}


class TestKeepwarmTracker:
    def test_first_touch_allowed_once_idle_long_enough(self):
        tracker = KeepwarmTracker()
        settings = _settings()
        assert tracker.should_touch(
            100.0, idle_seconds=9.0, large_context=False, settings=settings
        ) is False
        assert tracker.should_touch(
            100.0, idle_seconds=10.0, large_context=False, settings=settings
        ) is True

    def test_base_interval_throttles(self):
        tracker = KeepwarmTracker()
        settings = _settings()
        tracker.note_touch(_touch_event(100.0))
        assert tracker.should_touch(
            105.0, idle_seconds=100.0, large_context=False, settings=settings
        ) is False
        assert tracker.should_touch(
            111.0, idle_seconds=100.0, large_context=False, settings=settings
        ) is True

    def test_large_context_gets_the_slower_cadence(self):
        tracker = KeepwarmTracker()
        settings = _settings()
        tracker.note_touch(_touch_event(100.0))
        # Past the 10s base interval but inside the 30s large-model window.
        assert tracker.should_touch(
            115.0, idle_seconds=100.0, large_context=True, settings=settings
        ) is False
        assert tracker.should_touch(
            131.0, idle_seconds=100.0, large_context=True, settings=settings
        ) is True

    def test_slow_touch_backs_off(self):
        tracker = KeepwarmTracker()
        settings = _settings()
        tracker.note_touch(_touch_event(100.0, elapsed_ms=1500.0))
        assert tracker.should_touch(
            130.0, idle_seconds=100.0, large_context=False, settings=settings
        ) is False
        assert tracker.should_touch(
            161.0, idle_seconds=100.0, large_context=False, settings=settings
        ) is True

    def test_slow_backoff_disabled_at_zero(self):
        tracker = KeepwarmTracker()
        settings = _settings(keepwarm_slow_backoff_seconds=0.0)
        tracker.note_touch(_touch_event(100.0, elapsed_ms=1500.0))
        assert tracker.should_touch(
            111.0, idle_seconds=100.0, large_context=False, settings=settings
        ) is True

    def test_note_touch_counts(self):
        tracker = KeepwarmTracker()
        tracker.note_touch(_touch_event(100.0))
        tracker.note_touch(_touch_event(120.0, ok=False))
        assert tracker.touches == 2
        assert tracker.last_touch_at == 120.0
        assert tracker.last_event["ok"] is False


# ---------------------------------------------------------------------------
# graduated_idle_action
# ---------------------------------------------------------------------------


class TestGraduatedIdleAction:
    def test_all_off_is_none(self):
        assert (
            graduated_idle_action(
                10000.0,
                100.0,
                cache_ttl_seconds=0,
                idle_release_seconds=0,
                last_cache_drop_at=0.0,
                last_deep_release_at=0.0,
            )
            is None
        )

    def test_cache_ttl_fires_drop(self):
        assert (
            graduated_idle_action(
                100.0 + 1200.0,
                100.0,
                cache_ttl_seconds=1200,
                idle_release_seconds=0,
                last_cache_drop_at=0.0,
                last_deep_release_at=0.0,
            )
            == "drop_caches"
        )

    def test_deep_release_precedence_and_subsumes_shallow(self):
        action = graduated_idle_action(
            100.0 + 7200.0,
            100.0,
            cache_ttl_seconds=1200,
            idle_release_seconds=7200,
            last_cache_drop_at=0.0,
            last_deep_release_at=0.0,
        )
        assert action == "deep_release"

    def test_each_rung_fires_once_per_idle_period(self):
        # A drop already attempted after the last real access does not refire.
        assert (
            graduated_idle_action(
                100.0 + 1300.0,
                100.0,
                cache_ttl_seconds=1200,
                idle_release_seconds=0,
                last_cache_drop_at=100.0 + 1201.0,
                last_deep_release_at=0.0,
            )
            is None
        )
        # ...but new activity (a newer last_access) re-arms the rung.
        assert (
            graduated_idle_action(
                5000.0,
                3000.0,
                cache_ttl_seconds=1200,
                idle_release_seconds=0,
                last_cache_drop_at=1301.0,
                last_deep_release_at=0.0,
            )
            == "drop_caches"
        )

    def test_never_loaded_engine_is_skipped(self):
        assert (
            graduated_idle_action(
                10_000.0,
                0.0,
                cache_ttl_seconds=1200,
                idle_release_seconds=7200,
                last_cache_drop_at=0.0,
                last_deep_release_at=0.0,
            )
            is None
        )


# ---------------------------------------------------------------------------
# Scheduler.release_idle_caches
# ---------------------------------------------------------------------------


def _bare_scheduler() -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.running = {}
    sched.prefilling = {}
    sched.waiting = []
    sched._pending_async_removes = []
    sched._inflight_store_futures = {}
    sched._stream = None
    sched.block_aware_cache = None
    return sched


class TestSchedulerReleaseIdleCaches:
    def test_busy_scheduler_is_skipped(self, monkeypatch):
        cleared = []
        monkeypatch.setattr(
            "omlx.scheduler._sync_and_clear_cache", lambda stream: cleared.append(1)
        )
        sched = _bare_scheduler()
        sched.running = {"req-1": object()}
        sched.block_aware_cache = MagicMock()
        result = sched.release_idle_caches(reason="cache_ttl_expired")
        assert result["ok"] is False
        assert result["skipped"] == "busy"
        sched.block_aware_cache.clear.assert_not_called()
        assert cleared == []

    def test_inflight_store_cache_blocks_the_drop(self, monkeypatch):
        monkeypatch.setattr("omlx.scheduler._sync_and_clear_cache", lambda stream: None)
        sched = _bare_scheduler()
        sched._inflight_store_futures = {"req-1": MagicMock()}
        sched.block_aware_cache = MagicMock()
        result = sched.release_idle_caches()
        assert result["ok"] is False
        sched.block_aware_cache.clear.assert_not_called()

    def test_idle_drop_clears_prefix_cache_and_metal_pool(self, monkeypatch):
        cleared = []
        monkeypatch.setattr(
            "omlx.scheduler._sync_and_clear_cache", lambda stream: cleared.append(1)
        )
        sched = _bare_scheduler()
        sched.block_aware_cache = MagicMock()
        sched.block_aware_cache.clear.return_value = 3
        result = sched.release_idle_caches(reason="cache_ttl_expired")
        assert result["ok"] is True
        assert result["dropped_entries"] == 3
        assert result["deep"] is False
        assert result["metal_cleared"] is False
        assert cleared == [1]

    def test_deep_release_drains_the_metal_heap(self, monkeypatch):
        monkeypatch.setattr("omlx.scheduler._sync_and_clear_cache", lambda stream: None)
        metal_cleared = []
        fake_mx = SimpleNamespace(
            metal=SimpleNamespace(clear_cache=lambda: metal_cleared.append(1))
        )
        monkeypatch.setattr("omlx.scheduler.mx", fake_mx)
        sched = _bare_scheduler()
        result = sched.release_idle_caches(deep=True, reason="idle_release")
        assert result["ok"] is True
        assert result["deep"] is True
        assert result["metal_cleared"] is True
        assert metal_cleared == [1]

    def test_prefix_clear_failure_is_contained(self, monkeypatch):
        synced = []
        monkeypatch.setattr(
            "omlx.scheduler._sync_and_clear_cache", lambda stream: synced.append(1)
        )
        sched = _bare_scheduler()
        sched.block_aware_cache = MagicMock()
        sched.block_aware_cache.clear.side_effect = RuntimeError("trie drift")
        result = sched.release_idle_caches()
        assert result["ok"] is False
        assert "trie drift" in result["error"]
        assert synced == []


# ---------------------------------------------------------------------------
# EnginePool wiring: graduated TTL + keepwarm
# ---------------------------------------------------------------------------


def _entry(model_id="model-a", *, engine=None, last_access=100.0, context=None):
    return EngineEntry(
        model_id=model_id,
        model_path=f"/models/{model_id}",
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
        engine=engine,
        last_access=last_access,
        model_context_length=context,
    )


def _local_engine(*, release_result=None, active=False):
    """A SimpleNamespace engine wired like BatchedEngine -> EngineCore."""
    scheduler = MagicMock()
    scheduler.release_idle_caches = MagicMock(
        return_value=release_result
        if release_result is not None
        else {"ok": True, "dropped_entries": 2}
    )
    # _is_idle_for_prefill_eviction reads these scheduler collections.
    scheduler.running = {}
    scheduler.waiting = []
    scheduler.prefilling = {}
    scheduler.requests = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    core = SimpleNamespace(scheduler=scheduler, _mlx_executor=executor)
    engine = SimpleNamespace()
    engine.has_active_requests = lambda: active
    engine.scheduler = None
    engine._engine = SimpleNamespace(engine=core)
    engine.deployment = None
    engine.stop = AsyncMock()
    engine._omlx_scheduler = scheduler
    engine._omlx_executor = executor
    return engine


def _settings_manager(ttl_seconds=None):
    manager = MagicMock()
    manager.get_settings.return_value = SimpleNamespace(ttl_seconds=ttl_seconds)
    return manager


@pytest.fixture
def clean_maintenance_env(monkeypatch):
    for name in (
        "OMLX_KEEPWARM_ENABLED",
        "OMLX_CACHE_TTL_SECONDS",
        "OMLX_IDLE_RELEASE_SECONDS",
        "OMLX_EVICTION_GRACE_SECONDS",
        "OMLX_BACKGROUND_OP_SLOW_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


class TestPoolGraduatedTTL:
    async def test_default_env_is_a_noop_for_idle_engines(self, clean_maintenance_env):
        pool = EnginePool()
        engine = _local_engine()
        pool._entries["model-a"] = _entry(engine=engine, last_access=time.time() - 5000)
        expired = await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))
        assert expired == []
        engine._omlx_scheduler.release_idle_caches.assert_not_called()
        assert pool._keepwarm_trackers == {}
        engine._omlx_executor.shutdown(wait=False)

    async def test_cache_ttl_drops_caches_but_keeps_engine_resident(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_CACHE_TTL_SECONDS", "60")
        pool = EnginePool()
        engine = _local_engine()
        last_access = time.time() - 120.0
        pool._entries["model-a"] = _entry(engine=engine, last_access=last_access)

        expired = await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))

        assert expired == []
        engine._omlx_scheduler.release_idle_caches.assert_called_once()
        _, kwargs = engine._omlx_scheduler.release_idle_caches.call_args
        assert kwargs["deep"] is False
        assert kwargs["reason"] == "cache_ttl_expired"
        entry = pool._entries["model-a"]
        assert entry.engine is engine  # still resident
        assert entry.last_cache_drop_at > 0
        # B4: background maintenance never stamps the idle clock.
        assert entry.last_access == last_access
        # C5: the drop is attributed.
        events = [e for e in pool._eviction_attribution if e["mechanism"] == "cache_ttl"]
        assert len(events) == 1
        assert events[0]["model_id"] == "model-a"

        # The rung fires at most once per idle period.
        engine._omlx_scheduler.release_idle_caches.reset_mock()
        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))
        engine._omlx_scheduler.release_idle_caches.assert_not_called()
        engine._omlx_executor.shutdown(wait=False)

    async def test_idle_release_runs_the_deep_release(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_CACHE_TTL_SECONDS", "60")
        monkeypatch.setenv("OMLX_IDLE_RELEASE_SECONDS", "600")
        pool = EnginePool()
        engine = _local_engine()
        pool._entries["model-a"] = _entry(engine=engine, last_access=time.time() - 700)

        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))

        _, kwargs = engine._omlx_scheduler.release_idle_caches.call_args
        assert kwargs["deep"] is True
        assert kwargs["reason"] == "idle_release"
        entry = pool._entries["model-a"]
        assert entry.last_deep_release_at > 0
        assert entry.last_cache_drop_at > 0  # deep subsumes the shallow rung
        assert any(
            e["mechanism"] == "idle_release" for e in pool._eviction_attribution
        )
        engine._omlx_executor.shutdown(wait=False)

    async def test_engine_ttl_unload_takes_precedence_over_maintenance(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_CACHE_TTL_SECONDS", "60")
        pool = EnginePool()
        engine = _local_engine()
        pool._entries["model-a"] = _entry(engine=engine, last_access=time.time() - 5000)
        pool._unload_engine = AsyncMock()

        expired = await pool.check_ttl_expirations(_settings_manager(ttl_seconds=600))

        assert expired == ["model-a"]
        pool._unload_engine.assert_awaited_once_with("model-a")
        engine._omlx_scheduler.release_idle_caches.assert_not_called()
        assert any(
            e["mechanism"] == "engine_ttl" and e["ttl_seconds"] == 600
            for e in pool._eviction_attribution
        )
        engine._omlx_executor.shutdown(wait=False)

    async def test_busy_engine_gets_no_maintenance_and_refreshes_clock(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_CACHE_TTL_SECONDS", "60")
        pool = EnginePool()
        engine = _local_engine(active=True)
        last_access = time.time() - 5000
        pool._entries["model-a"] = _entry(engine=engine, last_access=last_access)

        expired = await pool.check_ttl_expirations(_settings_manager(ttl_seconds=600))

        assert expired == []
        engine._omlx_scheduler.release_idle_caches.assert_not_called()
        # Existing behavior: a busy model past its engine TTL is refreshed.
        assert pool._entries["model-a"].last_access > last_access
        engine._omlx_executor.shutdown(wait=False)

    async def test_pinned_model_gets_maintenance_but_never_unloaded(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_CACHE_TTL_SECONDS", "60")
        pool = EnginePool()
        engine = _local_engine()
        entry = _entry(engine=engine, last_access=time.time() - 5000)
        entry.is_pinned = True
        pool._entries["model-a"] = entry
        pool._unload_engine = AsyncMock()

        expired = await pool.check_ttl_expirations(_settings_manager(ttl_seconds=600))

        assert expired == []
        pool._unload_engine.assert_not_awaited()
        engine._omlx_scheduler.release_idle_caches.assert_called_once()
        engine._omlx_executor.shutdown(wait=False)

    async def test_suppressed_ttl_skips_everything(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_CACHE_TTL_SECONDS", "60")
        monkeypatch.setenv("OMLX_KEEPWARM_ENABLED", "1")
        pool = EnginePool()
        pool._suppress_ttl = True
        engine = _local_engine()
        pool._entries["model-a"] = _entry(engine=engine, last_access=time.time() - 5000)
        expired = await pool.check_ttl_expirations(_settings_manager(ttl_seconds=1))
        assert expired == []
        engine._omlx_scheduler.release_idle_caches.assert_not_called()
        assert pool._keepwarm_trackers == {}
        engine._omlx_executor.shutdown(wait=False)

    async def test_distributed_entry_fans_out_through_the_engine(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_CACHE_TTL_SECONDS", "60")
        pool = EnginePool()
        engine = SimpleNamespace(
            has_active_requests=lambda: False,
            release_idle_caches=AsyncMock(return_value=True),
            stop=AsyncMock(),
        )
        pool._entries["model-a"] = _entry(engine=engine, last_access=time.time() - 120)
        monkeypatch.setattr(
            pool, "_distributed_deployment_for_entry", lambda entry: object()
        )

        expired = await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))

        assert expired == []
        engine.release_idle_caches.assert_awaited_once()
        _, kwargs = engine.release_idle_caches.call_args
        assert kwargs["deep"] is False
        assert pool._entries["model-a"].last_cache_drop_at > 0

    async def test_failed_drop_stamps_the_rung_without_attribution(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_CACHE_TTL_SECONDS", "60")
        pool = EnginePool()
        engine = _local_engine(release_result={"ok": False, "skipped": "busy"})
        pool._entries["model-a"] = _entry(engine=engine, last_access=time.time() - 120)

        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))

        entry = pool._entries["model-a"]
        assert entry.last_cache_drop_at > 0  # no 1s retry spam
        assert not list(pool._eviction_attribution)
        engine._omlx_executor.shutdown(wait=False)

    async def test_engine_swap_mid_release_stamps_nothing(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_CACHE_TTL_SECONDS", "60")
        pool = EnginePool()
        engine = _local_engine()
        entry = _entry(engine=engine, last_access=time.time() - 120)
        pool._entries["model-a"] = entry

        def _swap(*args, **kwargs):
            # The engine is replaced while the release is queued/running.
            entry.engine = SimpleNamespace(has_active_requests=lambda: False)
            return {"ok": True}

        engine._omlx_scheduler.release_idle_caches = MagicMock(side_effect=_swap)

        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))

        assert entry.last_cache_drop_at == 0.0
        assert entry.last_deep_release_at == 0.0
        assert not list(pool._eviction_attribution)
        engine._omlx_executor.shutdown(wait=False)


class TestPoolKeepwarm:
    async def test_keepwarm_touch_dispatches_on_the_engine_executor(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_KEEPWARM_ENABLED", "1")
        touched = []

        def fake_touch(mx_module, *, size, repeats, reason):
            touched.append((size, repeats, reason))
            return {"ok": True, "at": time.time(), "elapsed_ms": 3.0}

        monkeypatch.setattr("omlx.engine_pool.metal_keepwarm_touch", fake_touch)
        pool = EnginePool()
        engine = _local_engine()
        last_access = time.time() - 100.0
        pool._entries["model-a"] = _entry(engine=engine, last_access=last_access)

        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))

        assert touched == [(64, 1, "engine-pool keepwarm")]
        tracker = pool._keepwarm_trackers["model-a"]
        assert tracker.touches == 1
        assert tracker.last_event["ok"] is True
        # B4: the touch never stamps the idle clock.
        assert pool._entries["model-a"].last_access == last_access

        # The base interval throttles an immediate second pass.
        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))
        assert len(touched) == 1
        engine._omlx_executor.shutdown(wait=False)

    async def test_keepwarm_skips_busy_engines(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_KEEPWARM_ENABLED", "1")
        touched = []
        monkeypatch.setattr(
            "omlx.engine_pool.metal_keepwarm_touch",
            lambda *a, **k: touched.append(1) or {"ok": True},
        )
        pool = EnginePool()
        engine = _local_engine(active=True)
        pool._entries["model-a"] = _entry(engine=engine, last_access=time.time() - 100)

        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))

        assert touched == []
        engine._omlx_executor.shutdown(wait=False)

    async def test_keepwarm_skips_engines_inside_idle_after(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_KEEPWARM_ENABLED", "1")
        touched = []
        monkeypatch.setattr(
            "omlx.engine_pool.metal_keepwarm_touch",
            lambda *a, **k: touched.append(1) or {"ok": True},
        )
        pool = EnginePool()
        engine = _local_engine()
        # Only 5s idle — inside the 10s idle-after window.
        pool._entries["model-a"] = _entry(engine=engine, last_access=time.time() - 5)

        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))

        assert touched == []
        engine._omlx_executor.shutdown(wait=False)

    async def test_keepwarm_large_context_uses_the_slow_cadence(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_KEEPWARM_ENABLED", "1")
        touched = []
        monkeypatch.setattr(
            "omlx.engine_pool.metal_keepwarm_touch",
            lambda *a, **k: touched.append(time.time())
            or {"ok": True, "at": time.time(), "elapsed_ms": 2.0},
        )
        pool = EnginePool()
        engine = _local_engine()
        pool._entries["model-a"] = _entry(
            engine=engine, last_access=time.time() - 100, context=32768
        )

        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))
        assert len(touched) == 1

        # 15s later: past the 10s base interval, inside the 30s large window.
        tracker = pool._keepwarm_trackers["model-a"]
        tracker.last_touch_at = time.time() - 15
        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))
        assert len(touched) == 1

        # 35s later: the large-model cadence allows the next touch.
        tracker.last_touch_at = time.time() - 35
        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))
        assert len(touched) == 2
        engine._omlx_executor.shutdown(wait=False)

    async def test_keepwarm_distributed_goes_through_the_engine(
        self, monkeypatch, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_KEEPWARM_ENABLED", "1")
        pool = EnginePool()
        engine = SimpleNamespace(
            has_active_requests=lambda: False,
            keepwarm_touch=AsyncMock(return_value=True),
        )
        pool._entries["model-a"] = _entry(engine=engine, last_access=time.time() - 100)
        monkeypatch.setattr(
            pool, "_distributed_deployment_for_entry", lambda entry: object()
        )

        await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))

        engine.keepwarm_touch.assert_awaited_once()
        _, kwargs = engine.keepwarm_touch.call_args
        assert kwargs["size"] == 64
        tracker = pool._keepwarm_trackers["model-a"]
        assert tracker.touches == 1
        assert tracker.last_event["ok"] is True

    async def test_slow_keepwarm_logs_loudly(
        self, monkeypatch, caplog, clean_maintenance_env
    ):
        monkeypatch.setenv("OMLX_KEEPWARM_ENABLED", "1")
        monkeypatch.setenv("OMLX_BACKGROUND_OP_SLOW_SECONDS", "0.1")

        def slow_touch(mx_module, *, size, repeats, reason):
            time.sleep(0.2)
            return {"ok": True, "at": time.time(), "elapsed_ms": 200.0}

        monkeypatch.setattr("omlx.engine_pool.metal_keepwarm_touch", slow_touch)
        pool = EnginePool()
        engine = _local_engine()
        pool._entries["model-a"] = _entry(engine=engine, last_access=time.time() - 100)

        with caplog.at_level(logging.WARNING, logger="omlx.engine_pool"):
            await pool.check_ttl_expirations(_settings_manager(ttl_seconds=None))

        assert any(
            "keepwarm" in record.getMessage() and "slow-op" in record.getMessage()
            for record in caplog.records
        )
        engine._omlx_executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# C4: recency grace in victim choice
# ---------------------------------------------------------------------------


class TestEvictionGrace:
    def _pool_with_two_idle(self):
        pool = EnginePool()
        old = _local_engine()
        recent = _local_engine()
        pool._entries["old-model"] = _entry(
            "old-model", engine=old, last_access=time.time() - 3600
        )
        pool._entries["recent-model"] = _entry(
            "recent-model", engine=recent, last_access=time.time() - 5
        )
        return pool, (old, recent)

    def test_grace_off_is_plain_lru(self, monkeypatch):
        monkeypatch.delenv("OMLX_EVICTION_GRACE_SECONDS", raising=False)
        pool, (old, recent) = self._pool_with_two_idle()
        assert pool._find_lru_victim() == "old-model"
        old._omlx_executor.shutdown(wait=False)
        recent._omlx_executor.shutdown(wait=False)

    def test_grace_deprioritizes_recently_used_models(self, monkeypatch):
        monkeypatch.setenv("OMLX_EVICTION_GRACE_SECONDS", "900")
        pool, (old, recent) = self._pool_with_two_idle()
        # Make the recently-used model the LRU-oldest so plain LRU would pick
        # it; grace must steer the victim choice to the long-idle model.
        pool._entries["recent-model"].last_access = time.time() - 3600
        pool._entries["old-model"].last_access = time.time() - 5
        assert pool._find_lru_victim() == "recent-model"
        old._omlx_executor.shutdown(wait=False)
        recent._omlx_executor.shutdown(wait=False)

    def test_grace_falls_back_to_lru_when_all_within_window(self, monkeypatch):
        monkeypatch.setenv("OMLX_EVICTION_GRACE_SECONDS", "900")
        pool, (old, recent) = self._pool_with_two_idle()
        pool._entries["old-model"].last_access = time.time() - 100
        pool._entries["recent-model"].last_access = time.time() - 5
        # Both inside the 900s window: plain LRU (oldest) wins — grace never
        # vetoes, it only deprioritizes.
        assert pool._find_lru_victim() == "old-model"
        old._omlx_executor.shutdown(wait=False)
        recent._omlx_executor.shutdown(wait=False)

    def test_prefill_victim_respects_grace(self, monkeypatch):
        monkeypatch.setenv("OMLX_EVICTION_GRACE_SECONDS", "900")
        pool, (old, recent) = self._pool_with_two_idle()
        # Within-grace model is the LRU-oldest; grace must skip it.
        pool._entries["recent-model"].last_access = time.time() - 3600
        pool._entries["old-model"].last_access = time.time() - 5
        assert (
            pool._find_lru_prefill_eviction_victim(exclude_model_id="target")
            == "recent-model"
        )
        old._omlx_executor.shutdown(wait=False)
        recent._omlx_executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# C5: eviction attribution
# ---------------------------------------------------------------------------


class TestEvictionAttribution:
    def test_record_eviction_appends_and_logs(self, caplog):
        pool = EnginePool()
        with caplog.at_level(logging.INFO, logger="omlx.engine_pool"):
            pool._record_eviction(
                "admission_lru", "victim-a", beneficiary="model-b"
            )
        events = list(pool._eviction_attribution)
        assert len(events) == 1
        assert events[0]["mechanism"] == "admission_lru"
        assert events[0]["model_id"] == "victim-a"
        assert events[0]["beneficiary"] == "model-b"
        assert events[0]["at"] > 0
        assert any("Eviction attribution" in r.getMessage() for r in caplog.records)

    def test_attribution_log_is_bounded(self):
        pool = EnginePool()
        for i in range(100):
            pool._record_eviction("engine_ttl", f"model-{i}")
        events = list(pool._eviction_attribution)
        assert len(events) == 64
        assert events[-1]["model_id"] == "model-99"

    def test_get_status_surfaces_recent_evictions(self):
        pool = EnginePool()
        pool._record_eviction("cache_ttl", "model-a", deep=False)
        status = pool.get_status()
        assert status["recent_evictions"][0]["mechanism"] == "cache_ttl"

    async def test_ttl_unload_records_attribution(self, clean_maintenance_env):
        pool = EnginePool()
        engine = _local_engine()
        pool._entries["model-a"] = _entry(engine=engine, last_access=time.time() - 500)
        pool._unload_engine = AsyncMock()
        expired = await pool.check_ttl_expirations(_settings_manager(ttl_seconds=60))
        assert expired == ["model-a"]
        events = list(pool._eviction_attribution)
        assert len(events) == 1
        assert events[0]["mechanism"] == "engine_ttl"
        assert events[0]["ttl_seconds"] == 60
        assert events[0]["idle_seconds"] >= 400
        engine._omlx_executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Distributed engine: idle-maintenance POSTs
# ---------------------------------------------------------------------------


class TestDistributedEngineIdleMaintenance:
    def _engine(self, handler):
        import httpx

        from omlx.cluster.deployment import ClusterDeployment, ClusterHost
        from omlx.cluster.planner import PipelineAssignment
        from omlx.engine.distributed import DistributedBatchedEngine

        deployment = ClusterDeployment(
            deployment_id="idle-test",
            model="org/model",
            backend="ring",
            hosts=(
                ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
                ClusterHost("peer", "peer.local", ("10.0.0.2",)),
            ),
            assignments=(
                PipelineAssignment("local", 0, 2, 4, 2, 0, 0, 4),
                PipelineAssignment("peer", 1, 0, 2, 2, 0, 0, 4),
            ),
            plan_hash="d" * 64,
        )
        engine = DistributedBatchedEngine(deployment)
        engine._loaded = True
        engine._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:1",
            transport=httpx.MockTransport(handler),
        )
        return engine

    async def test_release_idle_caches_posts_drop_caches(self):
        import httpx

        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"broadcast": True})

        engine = self._engine(handler)
        assert await engine.release_idle_caches(deep=True, reason="idle_release") is True
        assert seen == [
            {"op": "drop_caches", "clear_memory": True, "reason": "idle_release"}
        ]
        await engine._client.aclose()

    async def test_keepwarm_touch_posts_keepwarm(self):
        import httpx

        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"broadcast": True})

        engine = self._engine(handler)
        assert await engine.keepwarm_touch(size=64, reason="engine-pool keepwarm") is True
        assert seen == [
            {"op": "keepwarm", "matrix_size": 64, "reason": "engine-pool keepwarm"}
        ]
        await engine._client.aclose()

    async def test_unconfirmed_broadcast_returns_false(self):
        import httpx

        engine = self._engine(
            lambda request: httpx.Response(200, json={"broadcast": False})
        )
        assert await engine.release_idle_caches(deep=False, reason="cache_ttl") is False
        await engine._client.aclose()

    async def test_older_worker_404_returns_false(self):
        import httpx

        engine = self._engine(lambda request: httpx.Response(404))
        assert await engine.release_idle_caches() is False
        await engine._client.aclose()

    async def test_transport_error_returns_false(self):
        import httpx

        def handler(request):
            raise httpx.RemoteProtocolError("rank died", request=request)

        engine = self._engine(handler)
        assert await engine.keepwarm_touch() is False
        await engine._client.aclose()

    async def test_unavailable_engine_returns_false(self):
        import httpx  # noqa: F401 - mirrors the other tests' local import

        engine = self._engine(lambda request: httpx.Response(200, json={}))
        engine._loaded = False
        engine._client = None
        assert await engine.release_idle_caches() is False


# ---------------------------------------------------------------------------
# Share-channel idle-op sentinel (distributed fan-out)
# ---------------------------------------------------------------------------


class _ValidatedPipeline:
    pipeline_rank = 0
    pipeline_size = 2

    def __call__(self, value, cache=None):
        import mlx.core as mx

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        if pipeline_rank != 0:
            value = mx.distributed.send(
                value,
                (pipeline_rank - 1) % pipeline_size,
            )
        if pipeline_size > 1:
            value = mx.distributed.all_gather(value)
        return value


class _Group:
    @staticmethod
    def rank():
        return 0

    @staticmethod
    def size():
        return 2


class _WorkerGroup:
    @staticmethod
    def rank():
        return 1

    @staticmethod
    def size():
        return 2


def _install(group):
    from dataclasses import replace

    from omlx.cluster.performance import execution_profile
    from omlx.cluster.runtime_optimizations import install_runtime_optimizations

    return install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        group,
        replace(execution_profile("balanced"), sampling_rank_only=True),
        batchable=True,
        stop_token_ids=(99,),
    )


class TestIdleOpLatch:
    def test_keepwarm_is_replaced_by_any_newer_op(self):
        from omlx.cluster.runtime_optimizations import get_lockstep_controller

        controller = get_lockstep_controller()
        controller.reset()
        try:
            controller.request_idle_op({"op": "keepwarm"})
            controller.request_idle_op({"op": "drop_caches", "clear_memory": True})
            op = controller.take_pending_idle_op()
            assert op is not None and op["op"] == "drop_caches"
            assert controller.take_pending_idle_op() is None
        finally:
            controller.deactivate()

    def test_cache_drop_is_not_lost_to_a_keepwarm_tick(self):
        from omlx.cluster.runtime_optimizations import get_lockstep_controller

        controller = get_lockstep_controller()
        controller.reset()
        try:
            controller.request_idle_op({"op": "drop_caches", "clear_memory": False})
            controller.request_idle_op({"op": "keepwarm"})
            op = controller.take_pending_idle_op()
            assert op is not None and op["op"] == "drop_caches"
        finally:
            controller.deactivate()

    def test_reset_clears_a_pending_op(self):
        from omlx.cluster.runtime_optimizations import get_lockstep_controller

        controller = get_lockstep_controller()
        controller.reset()
        controller.request_idle_op({"op": "keepwarm"})
        controller.reset()
        assert controller.take_pending_idle_op() is None


class TestShareChannelIdleOp:
    def test_rank_zero_swaps_the_idle_share_for_the_op(self, monkeypatch):
        from omlx.cluster import runtime_optimizations as ro

        shared = []

        def fake_share(self, obj):
            # Keeps the validated source contract: pickle.dumps pickle.loads all_sum
            shared.append(obj)
            return obj

        executed = []
        monkeypatch.setattr(
            ro, "_execute_idle_op", lambda instance, op, *, mx_module: executed.append(op)
        )
        monkeypatch.setattr(mlx_server.ResponseGenerator, "_share_object", fake_share)
        controller = ro.get_lockstep_controller()
        with _install(_Group()):
            pinned = mlx_server.ResponseGenerator._share_object
            controller.request_idle_op({"op": "keepwarm", "matrix_size": 64})
            assert pinned(object(), None) is None
            assert shared[-1] == {"omlx_cluster_idle_op": {"op": "keepwarm", "matrix_size": 64}}
            assert executed == [{"op": "keepwarm", "matrix_size": 64}]
            assert controller.wait_idle_op_broadcast(0.1) is True

    def test_worker_executes_the_payload_and_keeps_polling(self, monkeypatch):
        from omlx.cluster import runtime_optimizations as ro

        def fake_share(self, obj):
            # Keeps the validated source contract: pickle.dumps pickle.loads all_sum
            return obj

        executed = []
        monkeypatch.setattr(
            ro, "_execute_idle_op", lambda instance, op, *, mx_module: executed.append(op)
        )
        monkeypatch.setattr(mlx_server.ResponseGenerator, "_share_object", fake_share)
        with _install(_WorkerGroup()):
            pinned = mlx_server.ResponseGenerator._share_object
            # An ordinary idle share passes through untouched.
            assert pinned(object(), None) is None
            assert executed == []
            payload = {"omlx_cluster_idle_op": {"op": "drop_caches", "clear_memory": False}}
            assert pinned(object(), payload) is None
            assert executed == [{"op": "drop_caches", "clear_memory": False}]

    def test_shutdown_takes_precedence_over_a_pending_op(self, monkeypatch):
        from omlx.cluster import runtime_optimizations as ro

        shared = []

        def fake_share(self, obj):
            # Keeps the validated source contract: pickle.dumps pickle.loads all_sum
            shared.append(obj)
            return obj

        monkeypatch.setattr(mlx_server.ResponseGenerator, "_share_object", fake_share)
        controller = ro.get_lockstep_controller()
        with _install(_Group()):
            pinned = mlx_server.ResponseGenerator._share_object
            controller.request_idle_op({"op": "keepwarm"})
            controller.request_shutdown()
            with pytest.raises(ro.LockstepClusterShutdownError):
                pinned(object(), None)
            assert shared[-1] == {"omlx_cluster_shutdown": True}
            # The queued op survives the shutdown sentinel (never consumed).
            op = controller.take_pending_idle_op()
            assert op is not None and op["op"] == "keepwarm"


class TestExecuteIdleOp:
    def test_keepwarm_runs_the_bounded_touch(self, monkeypatch):
        from omlx.cluster import runtime_optimizations as ro

        seen = []
        monkeypatch.setattr(
            ro,
            "metal_keepwarm_touch",
            lambda mx_module, *, size, repeats, reason: seen.append(
                (size, repeats, reason)
            )
            or {"ok": True},
        )
        event = ro._execute_idle_op(
            SimpleNamespace(),
            {"op": "keepwarm", "matrix_size": 32, "reason": "engine-pool keepwarm"},
            mx_module=SimpleNamespace(),
        )
        assert event["ok"] is True
        assert seen == [(32, 1, "engine-pool keepwarm")]

    def test_drop_caches_drops_every_resident_entry(self, monkeypatch):
        from omlx.cluster import prefill_guard
        from omlx.cluster import runtime_optimizations as ro

        dropped = []
        monkeypatch.setattr(
            prefill_guard,
            "_idle_prompt_cache_evictables",
            lambda cache: [
                {"drop": lambda: dropped.append("a")},
                {"drop": lambda: dropped.append("b")},
            ],
        )
        cleared = []
        fake_mx = SimpleNamespace(
            clear_cache=lambda: cleared.append("mlx"),
            metal=SimpleNamespace(clear_cache=lambda: cleared.append("metal")),
        )
        instance = SimpleNamespace(prompt_cache=object())
        event = ro._execute_idle_op(
            instance,
            {"op": "drop_caches", "clear_memory": True, "reason": "idle_release"},
            mx_module=fake_mx,
        )
        assert event["ok"] is True
        assert event["dropped_entries"] == 2
        assert dropped == ["a", "b"]
        assert cleared == ["mlx", "metal"]

    def test_shallow_drop_leaves_metal_alone(self, monkeypatch):
        from omlx.cluster import prefill_guard
        from omlx.cluster import runtime_optimizations as ro

        monkeypatch.setattr(
            prefill_guard, "_idle_prompt_cache_evictables", lambda cache: []
        )
        cleared = []
        fake_mx = SimpleNamespace(
            clear_cache=lambda: cleared.append("mlx"),
            metal=SimpleNamespace(clear_cache=lambda: cleared.append("metal")),
        )
        event = ro._execute_idle_op(
            SimpleNamespace(prompt_cache=object()),
            {"op": "drop_caches", "clear_memory": False},
            mx_module=fake_mx,
        )
        assert event["ok"] is True
        assert cleared == []

    def test_failure_is_contained_and_logged(self, monkeypatch, caplog):
        from omlx.cluster import prefill_guard
        from omlx.cluster import runtime_optimizations as ro

        def boom(cache):
            raise RuntimeError("trie drift")

        monkeypatch.setattr(prefill_guard, "_idle_prompt_cache_evictables", boom)
        with caplog.at_level(logging.WARNING, logger="omlx.cluster.runtime_optimizations"):
            event = ro._execute_idle_op(
                SimpleNamespace(prompt_cache=object()),
                {"op": "drop_caches"},
                mx_module=SimpleNamespace(),
            )
        assert event["ok"] is False
        assert "trie drift" in event["error"]
        assert any("drop_caches" in r.getMessage() for r in caplog.records)

    def test_unknown_op_is_rejected_without_raising(self):
        from omlx.cluster import runtime_optimizations as ro

        event = ro._execute_idle_op(
            SimpleNamespace(), {"op": "bogus"}, mx_module=SimpleNamespace()
        )
        assert event["ok"] is False
        assert "unknown idle op" in event["error"]


# ---------------------------------------------------------------------------
# Worker admin route
# ---------------------------------------------------------------------------


class _FakeAdminBase:
    """The slice of the mlx-lm APIHandler contract the admin routes use."""

    def _set_completion_headers(self, status_code):
        self.status_code = status_code

    def send_header(self, *_args):
        pass

    def end_headers(self):
        pass

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler verb API
        self.status_code = 404


def _idle_maintenance_request(handler_type, body, host="127.0.0.1"):
    handler = handler_type.__new__(handler_type)
    handler.path = "/admin/idle-maintenance"
    handler.client_address = (host, 49152)
    handler.wfile = io.BytesIO()
    raw = json.dumps(body).encode()
    handler.headers = {"Content-Length": str(len(raw))}
    handler.rfile = io.BytesIO(raw)
    return handler


class TestAdminIdleMaintenanceRoute:
    def _handler_type(self, controller):
        from omlx.cluster import inference_worker

        return inference_worker._cluster_admin_handler_class(
            _FakeAdminBase, controller
        )

    def test_arms_the_latch_and_reports_the_broadcast(self, monkeypatch):
        from omlx.cluster.runtime_optimizations import get_lockstep_controller

        controller = get_lockstep_controller()
        controller.reset(share_channel_active=True)
        handler_type = self._handler_type(controller)
        monkeypatch.setattr(controller, "wait_idle_op_broadcast", lambda timeout: True)
        try:
            handler = _idle_maintenance_request(
                handler_type,
                {"op": "drop_caches", "clear_memory": True, "reason": "idle_release"},
            )
            handler.do_POST()
            assert handler.status_code == 200
            body = json.loads(handler.wfile.getvalue())
            assert body["broadcast"] is True
            op = controller.take_pending_idle_op()
            assert op == {
                "op": "drop_caches",
                "reason": "idle_release",
                "clear_memory": True,
                "matrix_size": None,
            }
        finally:
            controller.deactivate()

    def test_reports_pending_when_the_cluster_is_busy(self, monkeypatch):
        from omlx.cluster.runtime_optimizations import get_lockstep_controller

        controller = get_lockstep_controller()
        controller.reset(share_channel_active=True)
        handler_type = self._handler_type(controller)
        monkeypatch.setattr(controller, "wait_idle_op_broadcast", lambda timeout: False)
        try:
            handler = _idle_maintenance_request(handler_type, {"op": "keepwarm"})
            handler.do_POST()
            assert handler.status_code == 200
            assert json.loads(handler.wfile.getvalue())["status"] == "pending"
        finally:
            controller.deactivate()

    def test_unavailable_without_the_share_channel(self):
        from omlx.cluster.runtime_optimizations import get_lockstep_controller

        controller = get_lockstep_controller()
        controller.reset(share_channel_active=False)
        handler_type = self._handler_type(controller)
        try:
            handler = _idle_maintenance_request(handler_type, {"op": "keepwarm"})
            handler.do_POST()
            assert handler.status_code == 200
            body = json.loads(handler.wfile.getvalue())
            assert body["status"] == "unavailable"
            assert body["broadcast"] is False
            assert controller.take_pending_idle_op() is None
        finally:
            controller.deactivate()

    def test_rejects_unknown_ops_and_remote_hosts(self):
        from omlx.cluster.runtime_optimizations import get_lockstep_controller

        controller = get_lockstep_controller()
        controller.reset(share_channel_active=True)
        handler_type = self._handler_type(controller)
        try:
            bad = _idle_maintenance_request(handler_type, {"op": "explode"})
            bad.do_POST()
            assert bad.status_code == 400
            assert controller.take_pending_idle_op() is None

            remote = _idle_maintenance_request(
                handler_type, {"op": "keepwarm"}, host="10.0.0.5"
            )
            remote.do_POST()
            assert remote.status_code == 403
            assert controller.take_pending_idle_op() is None

            garbage = handler_type.__new__(handler_type)
            garbage.path = "/admin/idle-maintenance"
            garbage.client_address = ("127.0.0.1", 49152)
            garbage.wfile = io.BytesIO()
            garbage.headers = {"Content-Length": "5"}
            garbage.rfile = io.BytesIO(b"not{{json")
            garbage.do_POST()
            assert garbage.status_code == 400
        finally:
            controller.deactivate()
