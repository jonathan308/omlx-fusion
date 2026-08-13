# SPDX-License-Identifier: Apache-2.0
"""Agent-traffic wedge test: the agent decode regime, certified without a server.

The caller and the clock are injected, so the cycle/rotation/wedge contract
is exercised with no network at all.
"""

import importlib.util
import json
from pathlib import Path

import httpx


def _load_script(name):
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent_traffic_test = _load_script("agent_traffic_test")


def _ok_caller(messages, max_tokens, temperature, tools):
    return {
        "seconds": 0.1,
        "chunks": 3,
        "text": f"reply-{len(messages)}",
        "finish": "stop",
        "tool_chunks": 0,
    }


def test_dry_run_prints_the_plan_and_touches_nothing(capsys, tmp_path):
    rc = agent_traffic_test.main(
        ["--model", "m", "--cycles", "2", "--dry-run"],
        caller=_ok_caller,
        sleep=lambda seconds: None,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "2 cycles x 7 steps" in out
    assert "tool: max_tokens=512" in out


def test_a_clean_run_logs_every_turn(tmp_path, capsys):
    log = tmp_path / "turns.jsonl"
    rc = agent_traffic_test.main(
        [
            "--model", "m",
            "--cycles", "2",
            "--think-gap", "0",
            "--out", str(log),
        ],
        caller=_ok_caller,
        sleep=lambda seconds: None,
    )

    assert rc == 0
    assert "2/2 cycles clean" in capsys.readouterr().out
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 14  # 2 cycles x 7 steps
    assert {record["status"] for record in records} == {"ok"}
    assert [record["kind"] for record in records[:7]] == [
        "short", "tool", "followup", "analysis", "rapid1", "rapid2", "rapid3",
    ]
    # Context accumulates within a session (user + assistant per turn).
    assert records[1]["ctx_msgs"] == records[0]["ctx_msgs"] + 2


def test_a_read_timeout_is_a_wedge_with_exit_75(tmp_path):
    def wedged(messages, max_tokens, temperature, tools):
        raise httpx.ReadTimeout("no tokens")

    log = tmp_path / "turns.jsonl"
    rc = agent_traffic_test.main(
        ["--model", "m", "--cycles", "1", "--think-gap", "0", "--out", str(log)],
        caller=wedged,
        sleep=lambda seconds: None,
    )

    assert rc == 75
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "wedge"


def test_a_lost_stream_is_a_wedge(tmp_path):
    def dropped(messages, max_tokens, temperature, tools):
        raise httpx.RemoteProtocolError("peer closed")

    rc = agent_traffic_test.main(
        [
            "--model", "m", "--cycles", "1", "--think-gap", "0",
            "--out", str(tmp_path / "turns.jsonl"),
        ],
        caller=dropped,
        sleep=lambda seconds: None,
    )
    assert rc == 75


def test_an_unexpected_error_is_logged_and_aborts(tmp_path):
    def broken(messages, max_tokens, temperature, tools):
        raise RuntimeError("boom")

    log = tmp_path / "turns.jsonl"
    rc = agent_traffic_test.main(
        ["--model", "m", "--cycles", "1", "--think-gap", "0", "--out", str(log)],
        caller=broken,
        sleep=lambda seconds: None,
    )

    assert rc == 75
    record = json.loads(log.read_text().splitlines()[0])
    assert record["status"] == "error"
    assert "boom" in record["error"]


def test_sessions_rotate_after_the_configured_turn_count():
    import io

    session = agent_traffic_test.AgentTrafficSession(session_turns=3)
    log = io.StringIO()
    for _ in range(3):
        assert session.turn(
            _ok_caller, log, kind="short", user_msg="hi",
            max_tokens=8, tools=None, temperature=0.2,
        )
    assert session.context_size == 1  # rotated back to the system prompt
    assert session.turn_no == 3  # turn numbering survives rotation


def test_the_api_key_prefers_the_environment(monkeypatch, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"auth": {"api_key": "from-file"}}))
    monkeypatch.setenv("OMLX_API_KEY", "from-env")
    assert agent_traffic_test.api_key(settings) == "from-env"
    monkeypatch.delenv("OMLX_API_KEY")
    assert agent_traffic_test.api_key(settings) == "from-file"
    assert agent_traffic_test.api_key(tmp_path / "missing.json") == ""


def test_stream_chat_parses_sse_and_reports_finish(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield from [
                'data: {"choices": [{"delta": {"content": "Hel"}}]}',
                'data: {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}',
                "data: [DONE]",
            ]

    def fake_stream(method, url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _Response()

    monkeypatch.setattr(agent_traffic_test.httpx, "stream", fake_stream)

    result = agent_traffic_test.stream_chat(
        base_url="http://omlx.test",
        key="secret",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0.2,
        tools=agent_traffic_test.TOOLS,
    )

    assert captured["url"] == "http://omlx.test/v1/chat/completions"
    assert captured["json"]["stream"] is True
    assert captured["json"]["tools"] == agent_traffic_test.TOOLS
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert result["text"] == "Hello"
    assert result["chunks"] == 2
    assert result["finish"] == "stop"
