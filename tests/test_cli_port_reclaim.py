# SPDX-License-Identifier: Apache-2.0
"""Tests for oMLX serve stale-port reclamation (A17 port).

Port of ThunderMLX start_gateway.sh's verified-stale-process reclamation:
the port owner is only signalled when identity-verified as another `omlx
serve` process; a foreign listener refuses with exit code 2.
"""

from unittest.mock import MagicMock

import pytest

# =============================================================================
# A17: stale-process reclamation with port-owner identity check
# =============================================================================


class TestStalePortReclaim:
    def _patch_subprocess(self, monkeypatch, listener_pid, owner):
        import subprocess as real_subprocess

        calls = {"kill": []}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "lsof":
                return MagicMock(stdout=f"{listener_pid}\n" if listener_pid else "")
            if cmd[0] == "ps":
                return MagicMock(stdout=owner)
            raise AssertionError(f"unexpected command: {cmd}")

        monkeypatch.setattr(real_subprocess, "run", fake_run)
        monkeypatch.setattr("os.kill", lambda pid, sig: calls["kill"].append((pid, sig)))
        return calls

    def test_free_port_is_noop(self, monkeypatch):
        from omlx.cli import _reclaim_stale_serve_port

        calls = self._patch_subprocess(monkeypatch, None, "")
        _reclaim_stale_serve_port(8000)
        assert calls["kill"] == []

    def test_replaces_stale_omlx_process(self, monkeypatch):
        import signal
        import subprocess as real_subprocess

        from omlx.cli import _reclaim_stale_serve_port

        state = {"alive": True}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "lsof":
                return MagicMock(stdout="4242\n" if state["alive"] else "")
            if cmd[0] == "ps":
                return MagicMock(stdout="/venv/bin/python -m omlx serve --port 8000")
            raise AssertionError(cmd)

        kills = []

        def fake_kill(pid, sig):
            kills.append((pid, sig))
            if sig == signal.SIGTERM:
                state["alive"] = False

        monkeypatch.setattr(real_subprocess, "run", fake_run)
        monkeypatch.setattr("os.kill", fake_kill)
        _reclaim_stale_serve_port(8000)
        assert kills == [(4242, signal.SIGTERM)]

    def test_escalates_to_sigkill(self, monkeypatch):
        import signal
        import subprocess as real_subprocess

        from omlx.cli import _reclaim_stale_serve_port

        def fake_run(cmd, **kwargs):
            if cmd[0] == "lsof":
                return MagicMock(stdout="4242\n")  # never goes away
            if cmd[0] == "ps":
                return MagicMock(stdout="omlx serve")
            raise AssertionError(cmd)

        kills = []
        monkeypatch.setattr(real_subprocess, "run", fake_run)
        monkeypatch.setattr("os.kill", lambda pid, sig: kills.append((pid, sig)))
        monkeypatch.setattr("time.sleep", lambda _: None)
        _reclaim_stale_serve_port(8000, term_grace_seconds=0.02)
        signals = [sig for _, sig in kills]
        assert signal.SIGTERM in signals
        assert signal.SIGKILL in signals

    def test_refuses_foreign_port_owner(self, monkeypatch):
        from omlx.cli import _reclaim_stale_serve_port

        calls = self._patch_subprocess(monkeypatch, 4242, "nginx: master process")
        with pytest.raises(SystemExit) as exc_info:
            _reclaim_stale_serve_port(8000)
        assert exc_info.value.code == 2
        assert calls["kill"] == []

    def test_lsof_failure_is_noop(self, monkeypatch):
        import subprocess as real_subprocess

        from omlx.cli import _reclaim_stale_serve_port

        def fake_run(cmd, **kwargs):
            raise OSError("lsof not found")

        monkeypatch.setattr(real_subprocess, "run", fake_run)
        _reclaim_stale_serve_port(8000)  # must not raise
