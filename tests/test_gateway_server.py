# SPDX-License-Identifier: Apache-2.0
"""Server-integration tests for the model-switch arbiter (omlx/gateway.py).

Covers the wiring in omlx/server.py: sticky-default routing in get_engine,
explicit-model stickiness stamping, 409 mappings for gate refusals, the
/v1/gateway/status surface, and the count_tokens no-wake probe path.
Everything is mocked; no real models are loaded.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import omlx.server as server
from omlx.api.anthropic_models import AnthropicMessage, TokenCountRequest
from omlx.engine import BaseEngine
from omlx.engine_pool import EnginePool
from omlx.gateway import GatewayConfig, ModelSwitchArbiter


def _make_pool(ceiling: int = 0) -> EnginePool:
    pool = EnginePool()
    pool._get_final_ceiling = lambda c=int(ceiling): c
    return pool


@pytest.fixture
def model_dir(tmp_path):
    for name in ("model-a", "model-b"):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (path / "model.safetensors").write_bytes(b"0" * 1024)
    return tmp_path


@pytest.fixture
def pool(model_dir, monkeypatch):
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    pool = _make_pool(ceiling=1000)
    pool.discover_models(str(model_dir))
    for entry in pool._entries.values():
        entry.engine_type = "batched"
        entry.model_type = "llm"
        entry.estimated_size = 600
    pool._current_model_memory = 600
    return pool


def _mark_loaded(pool: EnginePool, model_id: str, *, last_access: float | None = None):
    entry = pool.get_entry(model_id)
    entry.engine = MagicMock()  # idle: has_active_requests() is not `True`
    entry.last_access = time.time() if last_access is None else last_access
    return entry


def _mock_llm_engine() -> MagicMock:
    return MagicMock(spec=BaseEngine)


@pytest.fixture
def server_state(pool):
    """Point _server_state at the fake pool and restore afterwards."""
    saved = {
        "engine_pool": server._server_state.engine_pool,
        "gateway_arbiter": server._server_state.gateway_arbiter,
        "default_model": server._server_state.default_model,
        "settings_manager": server._server_state.settings_manager,
        "global_settings": server._server_state.global_settings,
        "api_key": server._server_state.api_key,
        "oq_manager": server._server_state.oq_manager,
    }
    server._server_state.engine_pool = pool
    server._server_state.gateway_arbiter = None
    server._server_state.default_model = "model-a"
    server._server_state.settings_manager = None
    server._server_state.global_settings = None
    server._server_state.api_key = None
    server._server_state.oq_manager = None
    yield pool
    for key, value in saved.items():
        setattr(server._server_state, key, value)


def _enabled_arbiter(**overrides) -> ModelSwitchArbiter:
    return ModelSwitchArbiter(GatewayConfig(enabled=True, **overrides))


# =============================================================================
# A3: sticky default routing in server.get_engine
# =============================================================================


class TestServerStickyRouting:
    def test_empty_model_follows_loaded_session_model(self, server_state):
        pool = server_state
        _mark_loaded(pool, "model-b")
        arbiter = _enabled_arbiter()
        arbiter.note_explicit_model("model-b")
        server._server_state.gateway_arbiter = arbiter

        captured = AsyncMock(return_value=_mock_llm_engine())
        pool.get_engine = captured
        asyncio.run(server.get_engine(None))
        # Routed to the sticky model, not the configured default model-a.
        assert captured.call_args.args[0] == "model-b"

    def test_empty_model_falls_back_to_default_when_nothing_loaded(self, server_state):
        pool = server_state
        arbiter = _enabled_arbiter()
        server._server_state.gateway_arbiter = arbiter

        captured = AsyncMock(return_value=_mock_llm_engine())
        pool.get_engine = captured
        asyncio.run(server.get_engine(None))
        assert captured.call_args.args[0] == "model-a"

    def test_sticky_unloaded_falls_back_to_default(self, server_state):
        pool = server_state
        arbiter = _enabled_arbiter()
        arbiter.note_explicit_model("model-b")  # not loaded -> no opinion
        server._server_state.gateway_arbiter = arbiter

        captured = AsyncMock(return_value=_mock_llm_engine())
        pool.get_engine = captured
        asyncio.run(server.get_engine(None))
        assert captured.call_args.args[0] == "model-a"

    def test_explicit_request_updates_stickiness(self, server_state):
        pool = server_state
        # Outside the grace window so the gate lets the switch through.
        _mark_loaded(pool, "model-a", last_access=time.time() - 120)
        arbiter = _enabled_arbiter()
        server._server_state.gateway_arbiter = arbiter

        pool.get_engine = AsyncMock(return_value=_mock_llm_engine())
        asyncio.run(server.get_engine("model-b"))
        assert arbiter._last_routed_model == "model-b"

    def test_defaulted_request_never_updates_stickiness(self, server_state):
        pool = server_state
        _mark_loaded(pool, "model-a")
        arbiter = _enabled_arbiter()
        arbiter.note_explicit_model("model-a")
        server._server_state.gateway_arbiter = arbiter

        pool.get_engine = AsyncMock(return_value=_mock_llm_engine())
        asyncio.run(server.get_engine(None))
        # The defaulted side-call used model-a but did not re-stamp.
        assert arbiter._last_routed_model == "model-a"

    def test_empty_string_model_treated_as_empty(self, server_state):
        pool = server_state
        _mark_loaded(pool, "model-b")
        arbiter = _enabled_arbiter()
        arbiter.note_explicit_model("model-b")
        server._server_state.gateway_arbiter = arbiter

        captured = AsyncMock(return_value=_mock_llm_engine())
        pool.get_engine = captured
        asyncio.run(server.get_engine(""))
        assert captured.call_args.args[0] == "model-b"

    def test_arbiter_disabled_preserves_empty_string_404(self, server_state):
        """Without the arbiter, "" follows the historical resolve-and-404 path."""
        pool = server_state
        server._server_state.gateway_arbiter = None
        pool.get_engine = AsyncMock(side_effect=server.ModelNotFoundError("", []))
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(server.get_engine(""))
        assert exc_info.value.status_code == 404

    def test_arbiter_disabled_no_sticky(self, server_state):
        pool = server_state
        _mark_loaded(pool, "model-b")
        server._server_state.gateway_arbiter = None
        captured = AsyncMock(return_value=_mock_llm_engine())
        pool.get_engine = captured
        asyncio.run(server.get_engine(None))
        assert captured.call_args.args[0] == "model-a"  # configured default


# =============================================================================
# A2: gate refusals surface as HTTP 409
# =============================================================================


class TestServerGateRefusals:
    def test_busy_refusal_maps_to_409(self, server_state):
        pool = server_state
        entry_a = _mark_loaded(pool, "model-a", last_access=time.time())
        entry_a.in_use = 1
        arbiter = _enabled_arbiter(defer_poll_seconds=0.005)
        server._server_state.gateway_arbiter = arbiter

        pool.get_engine = AsyncMock(return_value=_mock_llm_engine())
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(server.get_engine("model-b"))
        assert exc_info.value.status_code == 409
        assert "busy" in exc_info.value.detail.lower()
        pool.get_engine.assert_not_called()  # refused before any eviction

    def test_grace_timeout_maps_to_409_with_retry_after(self, server_state):
        pool = server_state
        entry_a = _mark_loaded(pool, "model-a", last_access=time.time())
        arbiter = _enabled_arbiter(
            switch_grace_seconds=0.05,
            defer_poll_seconds=0.005,
            defer_buffer_seconds=0.05,
            deferral_log_throttle_seconds=0.0,
        )
        server._server_state.gateway_arbiter = arbiter
        pool.get_engine = AsyncMock(return_value=_mock_llm_engine())

        async def scenario():
            async def restamp():
                for _ in range(200):
                    entry_a.last_access = time.time()
                    await asyncio.sleep(0.005)

            stamper = asyncio.create_task(restamp())
            try:
                await server.get_engine("model-b")
            finally:
                stamper.cancel()

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(scenario())
        assert exc_info.value.status_code == 409
        assert exc_info.value.headers["Retry-After"] != ""
        assert int(exc_info.value.headers["Retry-After"]) >= 1

    def test_grace_expiry_lets_load_through(self, server_state):
        pool = server_state
        _mark_loaded(pool, "model-a", last_access=time.time())
        arbiter = _enabled_arbiter(
            switch_grace_seconds=0.02,
            defer_poll_seconds=0.005,
            defer_buffer_seconds=1.0,
        )
        server._server_state.gateway_arbiter = arbiter
        captured = AsyncMock(return_value=_mock_llm_engine())
        pool.get_engine = captured
        asyncio.run(server.get_engine("model-b"))
        assert captured.call_args.args[0] == "model-b"
        assert any(e["action"] == "model_switch" for e in arbiter._events)


# =============================================================================
# Status endpoint
# =============================================================================


class TestGatewayStatusEndpoint:
    def test_disabled(self, server_state):
        server._server_state.gateway_arbiter = None
        assert asyncio.run(server.gateway_status()) == {"enabled": False}

    def test_enabled(self, server_state):
        pool = server_state
        _mark_loaded(pool, "model-a")
        arbiter = _enabled_arbiter()
        arbiter.record_event("probe")
        server._server_state.gateway_arbiter = arbiter
        status = asyncio.run(server.gateway_status())
        assert status["enabled"] is True
        assert status["events"][0]["action"] == "probe"
        assert "model-a" in status["residency"]


# =============================================================================
# A5: count_tokens no-wake probe path
# =============================================================================


def _count_request(model: str = "model-b") -> TokenCountRequest:
    return TokenCountRequest(
        model=model,
        messages=[AnthropicMessage(role="user", content="hello world")],
    )


class TestCountTokensNoWake:
    def test_unloaded_model_gets_estimate_without_loading(self, server_state, monkeypatch):
        pool = server_state
        arbiter = _enabled_arbiter()
        server._server_state.gateway_arbiter = arbiter

        must_not_load = AsyncMock(side_effect=AssertionError("must not load"))
        monkeypatch.setattr(server, "get_engine_for_model", must_not_load)
        request = _count_request()
        response = asyncio.run(server.count_anthropic_tokens(request))
        expected = max(1, len(request.model_dump_json().encode("utf-8")) // 4)
        assert response.input_tokens == expected
        must_not_load.assert_not_called()
        assert pool.get_entry("model-b").engine is None
        assert any(
            e["action"] == "passive_probe_no_wakeup" for e in arbiter._events
        )

    def test_loaded_model_uses_exact_path(self, server_state, monkeypatch):
        pool = server_state
        _mark_loaded(pool, "model-b")
        arbiter = _enabled_arbiter()
        server._server_state.gateway_arbiter = arbiter

        sentinel = RuntimeError("exact-path-reached")
        monkeypatch.setattr(
            server, "get_engine_for_model", AsyncMock(side_effect=sentinel)
        )
        with pytest.raises(RuntimeError, match="exact-path-reached"):
            asyncio.run(server.count_anthropic_tokens(_count_request()))
        # The no-wake path did not intercept: no estimate event recorded.
        assert not any(
            e["action"] == "passive_probe_no_wakeup" for e in arbiter._events
        )

    def test_arbiter_disabled_preserves_loading_behavior(self, server_state, monkeypatch):
        server._server_state.gateway_arbiter = None
        sentinel = RuntimeError("load-path-reached")
        monkeypatch.setattr(
            server, "get_engine_for_model", AsyncMock(side_effect=sentinel)
        )
        with pytest.raises(RuntimeError, match="load-path-reached"):
            asyncio.run(server.count_anthropic_tokens(_count_request()))


class TestPassiveProbesNeverLoad:
    """A5: health/status/model-list probes must not trigger a load or wake."""

    def test_health_and_models_probes_do_not_touch_engines(
        self, server_state, monkeypatch
    ):
        pool = server_state
        server._server_state.gateway_arbiter = _enabled_arbiter()
        spy = AsyncMock(side_effect=AssertionError("probe triggered a load"))
        monkeypatch.setattr(EnginePool, "get_engine", spy)
        monkeypatch.setattr(EnginePool, "acquire", spy)

        from fastapi import Response

        asyncio.run(server.health(Response()))
        models = asyncio.run(server.list_models())
        assert {m.id for m in models.data} == {"model-a", "model-b"}
        spy.assert_not_called()
        assert pool.get_entry("model-a").engine is None
        assert pool.get_entry("model-b").engine is None
