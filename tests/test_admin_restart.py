# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin server-restart route.

Covers the supervisor-gating contract: the endpoint refuses with 503 when
``OMLX_SUPERVISED`` is not set in the environment (plain ``omlx serve``)
and accepts with 202 + schedules a SIGTERM when running under the menu bar
or tracked dev launchd supervisor.
"""

from __future__ import annotations

import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes


ROOT = Path(__file__).resolve().parents[1]
DEV_SCRIPT = ROOT / "scripts/omlx-dev"
DASHBOARD_JS = ROOT / "omlx/admin/static/js/dashboard.js"


@pytest.fixture
def client(monkeypatch):
    """Build a TestClient with auth bypassed for the restart route."""
    async def _fake_require_admin():
        return True

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app)


class TestRestartServerRoute:
    def test_returns_503_when_unsupervised(self, client, monkeypatch):
        """No OMLX_SUPERVISED env var = no supervisor = no respawn path."""
        monkeypatch.delenv("OMLX_SUPERVISED", raising=False)

        r = client.post("/admin/api/server/restart")
        assert r.status_code == 503
        body = r.json()
        assert "detail" in body
        assert "supervisor" in body["detail"].lower()

    def test_returns_202_when_supervised(self, client, monkeypatch):
        """With OMLX_SUPERVISED set, the handler returns 202 immediately.

        ``_schedule_self_terminate`` is replaced with a spy so the test
        process never actually receives SIGTERM. Patching the seam (not
        ``asyncio.get_running_loop``) keeps FastAPI's TestClient portal
        intact.
        """
        monkeypatch.setenv("OMLX_SUPERVISED", "menubar")

        with patch("omlx.admin.routes._schedule_self_terminate") as spy:
            r = client.post("/admin/api/server/restart")

        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "restarting"
        assert body["supervisor"] == "menubar"
        assert body["expected_downtime_seconds"] > 0
        # The handler must schedule the SIGTERM (not invoke it synchronously)
        # and pass a positive delay so FastAPI can flush the 202 first.
        spy.assert_called_once()
        ((delay,), _kwargs) = spy.call_args
        assert delay > 0

    def test_supervisor_label_round_trips(self, client, monkeypatch):
        """Whatever supervisor identifier is set in env comes back in
        the response — useful for the dashboard and for diagnosing
        which supervisor is responsible for the respawn."""
        monkeypatch.setenv("OMLX_SUPERVISED", "launchd")

        with patch("omlx.admin.routes._schedule_self_terminate"):
            r = client.post("/admin/api/server/restart")

        assert r.status_code == 202
        assert r.json()["supervisor"] == "launchd"

    def test_supervisor_identity_round_trips_when_available(
        self, client, monkeypatch
    ):
        """Dev launchd jobs expose the exact job label for diagnostics."""
        monkeypatch.setenv("OMLX_SUPERVISED", "launchd")
        monkeypatch.setenv("OMLX_SUPERVISOR_ID", "com.omlx.fusion.dev")

        with patch("omlx.admin.routes._schedule_self_terminate"):
            r = client.post("/admin/api/server/restart")

        assert r.status_code == 202
        assert r.json()["supervisor_id"] == "com.omlx.fusion.dev"

    def test_unsupervised_does_not_schedule_termination(self, client, monkeypatch):
        """503 path must not schedule a SIGTERM — otherwise plain
        ``omlx serve`` instances would die with no respawn after a
        single accidental click against an unsupervised server."""
        monkeypatch.delenv("OMLX_SUPERVISED", raising=False)

        with patch("omlx.admin.routes._schedule_self_terminate") as spy:
            r = client.post("/admin/api/server/restart")

        assert r.status_code == 503
        spy.assert_not_called()


def test_self_termination_is_deferred_until_after_handler_can_respond():
    """Registering the signal must not terminate the process synchronously."""
    loop = MagicMock()
    scheduled: dict[str, object] = {}

    def capture(delay, callback):
        scheduled.update(delay=delay, callback=callback)

    loop.call_later.side_effect = capture
    with (
        patch("omlx.admin.routes.asyncio.get_running_loop", return_value=loop),
        patch("omlx.admin.routes.os.getpid", return_value=4242),
        patch("omlx.admin.routes.os.kill") as kill,
    ):
        admin_routes._schedule_self_terminate(0.5)
        kill.assert_not_called()
        assert scheduled["delay"] == 0.5
        scheduled["callback"]()

    kill.assert_called_once_with(4242, signal.SIGTERM)


def test_dev_launch_agent_advertises_exact_restartable_identity():
    """The dev job's marker and respawn policy are one atomic contract."""
    script = DEV_SCRIPT.read_text(encoding="utf-8")
    launch_agent = script.split("write_launch_agent() {", 1)[1].split(
        "run_server() {", 1
    )[0]
    stop_server = script.split("stop_server() {", 1)[1].split(
        "status_server() {", 1
    )[0]

    assert "<key>OMLX_SUPERVISED</key>\n    <string>launchd</string>" in launch_agent
    assert (
        "<key>OMLX_SUPERVISOR_ID</key>\n    <string>${label}</string>"
        in launch_agent
    )
    assert "<key>KeepAlive</key>\n  <true/>" in launch_agent
    # KeepAlive must not defeat an explicit Stop: bootout removes the job.
    assert 'launchctl bootout "${launch_target}"' in stop_server


def test_web_restart_persists_form_before_post_and_stops_on_save_failure():
    """Unsaved restart-scoped settings cannot be lost by the Restart button."""
    script = DASHBOARD_JS.read_text(encoding="utf-8")
    restart = script.split("async restartServerStart() {", 1)[1].split(
        "_restartServerPoll() {", 1
    )[0]

    save_index = restart.index("const saved = await this.saveGlobalSettings();")
    guard_index = restart.index("if (!saved)", save_index)
    guard_return_index = restart.index("return;", guard_index)
    post_index = restart.index("fetch('/admin/api/server/restart'", guard_index)
    assert save_index < guard_index < guard_return_index < post_index
    assert "message: this.saveError" in restart[guard_index:post_index]

    save = script.split("async saveGlobalSettings() {", 1)[1].split(
        "// Sub key management", 1
    )[0]
    assert "let saved = false;" in save
    assert "saved = true;" in save
    assert "return saved;" in save


def test_restart_transport_fallback_requires_down_then_up_or_times_out():
    """An ambiguous dropped 202 polls safely instead of claiming success."""
    script = DASHBOARD_JS.read_text(encoding="utf-8")
    restart = script.split("async restartServerStart() {", 1)[1].split(
        "_restartServerPoll() {", 1
    )[0]
    transport_error = restart.split("} catch (err) {", 1)[1].split(
        "if (response.status === 503)", 1
    )[0]
    assert "status: 'waiting'" in transport_error
    assert "this._restartServerPoll();" in transport_error

    poll = script.split("_restartServerPoll() {", 1)[1].split(
        "get llmModels()", 1
    )[0]
    assert "const deadline = Date.now() + 60000" in poll
    assert "let sawDownAt = 0" in poll
    assert "if (!alive)" in poll
    assert "if (!sawDownAt)" in poll
    assert "settings.server.restart_status_timeout" in poll
