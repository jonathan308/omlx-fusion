# SPDX-License-Identifier: Apache-2.0
"""Tests for the model-switch arbiter (omlx/gateway.py).

Ports of the ThunderMLX model_gateway.py semantics: grace windows with
bounded wait-out, busy-refusal, sticky default model, passive-probe
no-wakeup, wakeup attribution, and stale-port reclamation. All model
loads/engines are mocked; no real models are loaded.
"""

import asyncio
import json
import logging
import time
from unittest.mock import MagicMock

import pytest

from omlx.engine_pool import EnginePool
from omlx.exceptions import ModelBusyError, ModelNotFoundError
from omlx.gateway import (
    EVENT_RING_SIZE,
    GatewayAttributionMiddleware,
    GatewayConfig,
    ModelSwitchArbiter,
    ModelSwitchDeferredError,
    estimate_input_tokens,
    request_attribution,
)

GB = 1024**3


def _make_pool(ceiling: int = 0) -> EnginePool:
    """EnginePool with a stubbed admission ceiling (0 = no limit)."""
    pool = EnginePool()
    pool._get_final_ceiling = lambda c=int(ceiling): c
    return pool


@pytest.fixture
def model_dir(tmp_path):
    """Two tiny fake models on disk."""
    for name in ("model-a", "model-b"):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (path / "model.safetensors").write_bytes(b"0" * 1024)
    return tmp_path


def _discover(pool: EnginePool, model_dir) -> None:
    pool.discover_models(str(model_dir))
    for entry in pool._entries.values():
        entry.engine_type = "batched"
        entry.model_type = "llm"


def _mark_loaded(pool: EnginePool, model_id: str, *, last_access: float | None = None):
    """Simulate a loaded, idle engine without loading anything."""
    entry = pool.get_entry(model_id)
    engine = MagicMock()
    # _entry_has_active_requests uses `is True`; a MagicMock return is falsy.
    entry.engine = engine
    entry.last_access = time.time() if last_access is None else last_access
    return entry


@pytest.fixture
def pressured_pool(model_dir, monkeypatch):
    """Pool where loading model-b requires evicting model-a.

    ceiling=1000, model-a committed at 600, model-b sized 600: projected
    1200 > evict_target 1000 -> eviction pressure.
    """
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    pool = _make_pool(ceiling=1000)
    _discover(pool, model_dir)
    pool._current_model_memory = 600
    pool.get_entry("model-a").estimated_size = 600
    pool.get_entry("model-b").estimated_size = 600
    return pool


def _arbiter(**overrides) -> ModelSwitchArbiter:
    config = GatewayConfig(enabled=True, **overrides)
    return ModelSwitchArbiter(config)


# =============================================================================
# GatewayConfig / env killswitches
# =============================================================================


class TestGatewayConfig:
    def test_defaults_disabled(self, monkeypatch):
        for var in (
            "OMLX_GATEWAY_ENABLED",
            "OMLX_GATEWAY_STICKY_EMPTY_MODEL",
            "OMLX_GATEWAY_SWITCH_GRACE_S",
            "OMLX_GATEWAY_PROBE_NO_WAKE",
        ):
            monkeypatch.delenv(var, raising=False)
        config = GatewayConfig.from_env()
        assert config.enabled is False
        assert config.sticky_empty_model is True
        assert config.switch_grace_seconds == 30.0
        assert config.probe_no_wake is True

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("OMLX_GATEWAY_ENABLED", "1")
        monkeypatch.setenv("OMLX_GATEWAY_STICKY_EMPTY_MODEL", "0")
        monkeypatch.setenv("OMLX_GATEWAY_SWITCH_GRACE_S", "900")
        monkeypatch.setenv("OMLX_GATEWAY_PROBE_NO_WAKE", "0")
        config = GatewayConfig.from_env()
        assert config.enabled is True
        assert config.sticky_empty_model is False
        assert config.switch_grace_seconds == 900.0
        assert config.probe_no_wake is False

    def test_invalid_grace_falls_back(self, monkeypatch):
        monkeypatch.setenv("OMLX_GATEWAY_SWITCH_GRACE_S", "not-a-number")
        assert GatewayConfig.from_env().switch_grace_seconds == 30.0


# =============================================================================
# EnginePool advisory helpers
# =============================================================================


class TestPoolAdvisoryHelpers:
    def test_eviction_pressure_requires_pressure(self, pressured_pool):
        pool = pressured_pool
        assert pool.eviction_pressure_for_load("model-b") is True
        # Fit alongside: tiny model projects under the target.
        pool.get_entry("model-b").estimated_size = 100
        pool._current_model_memory = 100
        assert pool.eviction_pressure_for_load("model-b") is False

    def test_eviction_pressure_skips_loaded_and_unknown(self, pressured_pool):
        pool = pressured_pool
        _mark_loaded(pool, "model-b")
        assert pool.eviction_pressure_for_load("model-b") is False
        assert pool.eviction_pressure_for_load("nope") is False

    def test_eviction_pressure_no_ceiling(self, model_dir, monkeypatch):
        monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
        monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
        pool = _make_pool(ceiling=0)
        _discover(pool, model_dir)
        assert pool.eviction_pressure_for_load("model-b") is False

    def test_residency_snapshot_loaded_only(self, pressured_pool):
        pool = pressured_pool
        entry = _mark_loaded(pool, "model-a", last_access=123.0)
        entry.in_use = 1
        snapshot = pool.residency_snapshot()
        assert set(snapshot) == {"model-a"}
        info = snapshot["model-a"]
        assert info["busy"] is True
        assert info["pinned"] is False
        assert info["last_access"] == 123.0
        assert info["distributed"] is False

    def test_entry_is_distributed(self, pressured_pool):
        pool = pressured_pool
        entry = pool.get_entry("model-b")
        assert pool.entry_is_distributed(entry) is False
        entry.source_type = "cluster"
        assert pool.entry_is_distributed(entry) is True

    def test_entry_is_distributed_via_registry(self, pressured_pool):
        pool = pressured_pool
        entry = pool.get_entry("model-b")

        class _Registry:
            def get_for_model(self, model_path):
                return object()  # any deployment record marks it distributed

        pool._cluster_registry = _Registry()
        assert pool.entry_is_distributed(entry) is True


# =============================================================================
# A4: validate-before-destructive-switch (pool-level guarantee)
# =============================================================================


class TestValidateBeforeSwitch:
    def test_unknown_model_never_evicts_incumbent(self, pressured_pool):
        pool = pressured_pool
        entry_a = _mark_loaded(pool, "model-a")
        with pytest.raises(ModelNotFoundError):
            asyncio.run(pool.get_engine("typo-model"))
        # The incumbent is untouched: no eviction ran before validation.
        assert pool.get_entry("model-a").engine is entry_a.engine

    def test_gate_noop_for_unknown_model(self, pressured_pool):
        arbiter = _arbiter()
        asyncio.run(arbiter.before_model_load(pressured_pool, "typo-model"))
        assert list(arbiter._events) == []


# =============================================================================
# A1/A2: the switch gate
# =============================================================================


class TestSwitchGate:
    def test_loaded_model_bypasses_gate(self, pressured_pool):
        pool = pressured_pool
        _mark_loaded(pool, "model-b")
        arbiter = _arbiter()
        asyncio.run(arbiter.before_model_load(pool, "model-b"))
        assert list(arbiter._events) == []

    def test_cold_load_without_pressure_records_wake(self, pressured_pool, caplog):
        pool = pressured_pool
        pool._current_model_memory = 0  # plenty of room -> not destructive
        arbiter = _arbiter()
        token = request_attribution.set(
            {"path": "/v1/chat/completions", "client": "10.0.0.2:5100", "user_agent": "zcode/1.0"}
        )
        try:
            with caplog.at_level(logging.INFO, logger="omlx.gateway"):
                asyncio.run(arbiter.before_model_load(pool, "model-b"))
        finally:
            request_attribution.reset(token)
        assert "model-wakeup-attribution" in caplog.text
        assert "10.0.0.2:5100" in caplog.text
        assert "zcode/1.0" in caplog.text
        wakes = [e for e in arbiter._events if e["action"] == "model_wake"]
        assert len(wakes) == 1
        assert wakes[0]["destructive"] is False
        assert wakes[0]["client"] == "10.0.0.2:5100"

    def test_switch_outside_grace_proceeds_immediately(self, pressured_pool):
        pool = pressured_pool
        _mark_loaded(pool, "model-a", last_access=time.time() - 120)
        arbiter = _arbiter(switch_grace_seconds=30.0, defer_poll_seconds=0.005)
        start = time.monotonic()
        asyncio.run(arbiter.before_model_load(pool, "model-b"))
        assert time.monotonic() - start < 5
        switches = [e for e in arbiter._events if e["action"] == "model_switch"]
        assert len(switches) == 1
        assert switches[0]["waited"] is False
        assert switches[0]["evicts"] == ["model-a"]

    def test_switch_waits_out_grace(self, pressured_pool):
        pool = pressured_pool
        _mark_loaded(pool, "model-a", last_access=time.time())
        arbiter = _arbiter(
            switch_grace_seconds=0.05,
            defer_poll_seconds=0.005,
            defer_buffer_seconds=1.0,
        )
        asyncio.run(arbiter.before_model_load(pool, "model-b"))
        switches = [e for e in arbiter._events if e["action"] == "model_switch"]
        assert len(switches) == 1
        assert switches[0]["waited"] is True

    def test_sustained_traffic_defers_then_refuses(self, pressured_pool):
        """New traffic keeps re-arming the grace window past the deadline."""
        pool = pressured_pool
        entry_a = _mark_loaded(pool, "model-a", last_access=time.time())
        arbiter = _arbiter(
            switch_grace_seconds=0.05,
            defer_poll_seconds=0.005,
            defer_buffer_seconds=0.05,
            deferral_log_throttle_seconds=0.0,
        )

        async def scenario():
            async def restamp():
                for _ in range(200):
                    entry_a.last_access = time.time()
                    await asyncio.sleep(0.005)

            stamper = asyncio.create_task(restamp())
            try:
                await arbiter.before_model_load(pool, "model-b")
            finally:
                stamper.cancel()

        with pytest.raises(ModelSwitchDeferredError) as exc_info:
            asyncio.run(scenario())
        assert exc_info.value.model_id == "model-b"
        assert exc_info.value.blockers == ["model-a"]
        assert exc_info.value.retry_after > 0
        actions = [e["action"] for e in arbiter._events]
        assert "model_switch_grace_wait" in actions
        assert "model_switch_grace_timeout" in actions
        assert "model_switch" not in actions

    def test_busy_incumbent_refuses_immediately(self, pressured_pool):
        pool = pressured_pool
        entry_a = _mark_loaded(pool, "model-a", last_access=time.time())
        entry_a.in_use = 1  # in-flight lease
        arbiter = _arbiter(defer_poll_seconds=0.005)
        start = time.monotonic()
        with pytest.raises(ModelBusyError):
            asyncio.run(arbiter.before_model_load(pool, "model-b"))
        assert time.monotonic() - start < 5  # no grace wait for busy models
        events = [e for e in arbiter._events if e["action"] == "model_switch_refused_busy"]
        assert len(events) == 1
        assert events[0]["blockers"] == ["model-a"]

    def test_pinned_incumbent_is_not_a_blocker(self, pressured_pool):
        """Pinned models are never evicted, so the gate leaves the load to
        the pool's own admission error instead of refusing on their account."""
        pool = pressured_pool
        entry_a = _mark_loaded(pool, "model-a", last_access=time.time())
        entry_a.is_pinned = True
        arbiter = _arbiter(defer_poll_seconds=0.005)
        asyncio.run(arbiter.before_model_load(pool, "model-b"))
        assert any(e["action"] == "model_switch" for e in arbiter._events)

    def test_cluster_target_bypasses_gate(self, pressured_pool):
        pool = pressured_pool
        _mark_loaded(pool, "model-a", last_access=time.time())  # in grace
        entry_a = pool.get_entry("model-a")
        entry_a.in_use = 1  # even busy: cluster requests never wait/refuse
        pool.get_entry("model-b").source_type = "cluster"
        arbiter = _arbiter(defer_poll_seconds=0.005)
        asyncio.run(arbiter.before_model_load(pool, "model-b"))
        assert list(arbiter._events) == []

    def test_registry_deployment_target_bypasses_gate(self, pressured_pool):
        pool = pressured_pool
        _mark_loaded(pool, "model-a", last_access=time.time())

        class _Registry:
            def get_for_model(self, model_path):
                return object()

        pool._cluster_registry = _Registry()
        arbiter = _arbiter(defer_poll_seconds=0.005)
        asyncio.run(arbiter.before_model_load(pool, "model-b"))
        assert list(arbiter._events) == []

    def test_deferral_events_throttled(self, pressured_pool):
        pool = pressured_pool
        _mark_loaded(pool, "model-a", last_access=time.time())
        arbiter = _arbiter(
            switch_grace_seconds=0.05,
            defer_poll_seconds=0.005,
            defer_buffer_seconds=1.0,
            deferral_log_throttle_seconds=60.0,
        )
        asyncio.run(arbiter.before_model_load(pool, "model-b"))
        waits = [e for e in arbiter._events if e["action"] == "model_switch_grace_wait"]
        # The loop polled ~10 times but recorded at most one wait event.
        assert len(waits) <= 1


# =============================================================================
# A3: sticky default model
# =============================================================================


class TestStickyDefault:
    def test_returns_last_routed_while_loaded(self, pressured_pool):
        pool = pressured_pool
        _mark_loaded(pool, "model-a")
        arbiter = _arbiter()
        arbiter.note_explicit_model("model-a")
        assert arbiter.sticky_default_model(pool) == "model-a"

    def test_none_when_not_loaded(self, pressured_pool):
        arbiter = _arbiter()
        arbiter.note_explicit_model("model-a")
        assert arbiter.sticky_default_model(pressured_pool) is None

    def test_none_when_disabled(self, pressured_pool):
        pool = pressured_pool
        _mark_loaded(pool, "model-a")
        arbiter = _arbiter(sticky_empty_model=False)
        arbiter.note_explicit_model("model-a")
        assert arbiter.sticky_default_model(pool) is None

    def test_skips_non_llm_engines(self, pressured_pool):
        pool = pressured_pool
        entry = _mark_loaded(pool, "model-a")
        entry.engine_type = "embedding"
        arbiter = _arbiter()
        arbiter.note_explicit_model("model-a")
        assert arbiter.sticky_default_model(pool) is None

    def test_note_explicit_ignores_empty(self):
        arbiter = _arbiter()
        arbiter.note_explicit_model("")
        arbiter.note_explicit_model(None)
        assert arbiter._last_routed_model is None
        arbiter.note_explicit_model("model-a")
        arbiter.note_explicit_model("   ")
        assert arbiter._last_routed_model == "model-a"


# =============================================================================
# A5: passive-probe no-wakeup (count_tokens estimate)
# =============================================================================


class TestProbeNoWake:
    def test_estimate_for_unloaded_model(self, pressured_pool):
        arbiter = _arbiter()
        estimate = arbiter.count_tokens_no_wake(pressured_pool, "model-b", "x" * 400)
        assert estimate == 100
        events = [e for e in arbiter._events if e["action"] == "passive_probe_no_wakeup"]
        assert len(events) == 1

    def test_loaded_model_uses_exact_path(self, pressured_pool):
        pool = pressured_pool
        _mark_loaded(pool, "model-a")
        arbiter = _arbiter()
        assert arbiter.count_tokens_no_wake(pool, "model-a", "x" * 400) is None

    def test_unknown_model_uses_normal_path(self, pressured_pool):
        arbiter = _arbiter()
        assert arbiter.count_tokens_no_wake(pressured_pool, "typo", "x" * 400) is None

    def test_disabled(self, pressured_pool):
        arbiter = _arbiter(probe_no_wake=False)
        assert arbiter.count_tokens_no_wake(pressured_pool, "model-b", "x" * 400) is None

    def test_estimate_input_tokens_bytes_over_four(self):
        assert estimate_input_tokens("") == 0
        assert estimate_input_tokens("abcd") == 1
        assert estimate_input_tokens("é" * 4) == 2  # 8 UTF-8 bytes
        assert estimate_input_tokens("hi") == 1  # floor of 1


# =============================================================================
# A6: event ring, attribution middleware, status
# =============================================================================


class TestAttribution:
    def test_event_ring_caps_at_64(self):
        arbiter = _arbiter()
        for i in range(EVENT_RING_SIZE + 20):
            arbiter.record_event("probe", i=i)
        assert len(arbiter._events) == EVENT_RING_SIZE
        # Newest first.
        assert arbiter._events[0]["i"] == EVENT_RING_SIZE + 19

    def test_middleware_sets_attribution(self):
        captured = {}

        async def app(scope, receive, send):
            captured.update(request_attribution.get() or {})

        middleware = GatewayAttributionMiddleware(app)
        scope = {
            "type": "http",
            "path": "/v1/chat/completions",
            "client": ("192.168.1.5", 60210),
            "headers": [(b"user-agent", b"claude-cli/2.0")],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        asyncio.run(middleware(scope, receive, lambda m: asyncio.sleep(0)))
        assert captured["path"] == "/v1/chat/completions"
        assert captured["client"] == "192.168.1.5:60210"
        assert captured["user_agent"] == "claude-cli/2.0"
        # Context resets after the request.
        assert request_attribution.get() is None

    def test_middleware_passes_non_http_scope(self):
        seen = []

        async def app(scope, receive, send):
            seen.append(scope["type"])

        middleware = GatewayAttributionMiddleware(app)
        asyncio.run(middleware({"type": "lifespan"}, None, None))
        assert seen == ["lifespan"]

    def test_get_status(self, pressured_pool):
        pool = pressured_pool
        _mark_loaded(pool, "model-a")
        arbiter = _arbiter()
        arbiter.note_explicit_model("model-a")
        arbiter.record_event("probe")
        status = arbiter.get_status(pool)
        assert status["enabled"] is True
        assert status["config"]["switch_grace_seconds"] == 30.0
        assert status["last_routed_model"] == "model-a"
        assert status["last_event"]["action"] == "probe"
        assert "model-a" in status["residency"]

