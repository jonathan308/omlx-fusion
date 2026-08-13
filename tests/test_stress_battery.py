# SPDX-License-Identifier: Apache-2.0
"""The stress battery certifies a server through a scripted transport.

No server, no sockets: the fake transport answers like a healthy oMLX (or a
wedged one) and every phase verdict is asserted from the battery's results.
"""

import importlib.util
import re
import threading
from pathlib import Path


def _load_script(name):
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stress_battery = _load_script("stress_battery")


class FakeTransport:
    """Answers like a healthy server; ``fail_health_after`` wedges /health."""

    def __init__(self, *, fail_health_after=None, probe_failures=()):
        self.base_url = "http://fake"
        self._health_calls = 0
        self._fail_health_after = fail_health_after
        self._probe_failures = set(probe_failures)
        self._lock = threading.Lock()
        self.chats = 0
        self.streams = 0

    def get(self, path, *, timeout=5.0):
        with self._lock:
            self._health_calls += 1
            calls = self._health_calls
        if self._fail_health_after is not None and calls > self._fail_health_after:
            raise ConnectionError("server is gone")
        if path == "/health":
            return {
                "status": "healthy",
                "engine_pool": {"model_count": 1, "loaded_count": 1},
            }
        return {"metrics": {"requests": 1}}

    def chat(self, messages, *, max_tokens, temperature=0.2, tools=None, read_timeout=600.0):
        with self._lock:
            self.chats += 1
        content = messages[-1]["content"]
        match = re.search(r"Reply with the word (\S+) only", content)
        if match and match.group(1) in self._probe_failures:
            raise TimeoutError("slot never freed")
        text = match.group(1) if match else "summary"
        return {"text": text, "seconds": 0.01, "usage": {}}

    def stream(self, messages, *, max_tokens, temperature=0.2, keep_chunks=None, read_timeout=600.0):
        with self._lock:
            self.streams += 1
        chunks = keep_chunks if keep_chunks is not None else max_tokens
        return {
            "text": "x" * chunks,
            "chunks": chunks,
            "ttft": 0.01,
            "seconds": max(chunks / 100.0, 0.01),  # 100 chunks/s: above the floor
            "abandoned": keep_chunks is not None,
        }


def _fast_args(extra=()):
    return stress_battery._arguments(
        [
            "--model", "m",
            "--hammer-cycles", "2",
            "--decode-tokens", "100",
            "--min-decode-tps", "5",
            "--agent-cycles", "1",
            "--prefill-records", "10",
            "--concurrent", "3",
            "--idle-seconds", "0.02",
            "--cache-settle-seconds", "0.05",
            *extra,
        ]
    )


def _run(transport, args):
    return stress_battery.Battery(transport, args, sleep=lambda seconds: None).run()


def _results(battery_results):
    return {entry["phase"]: entry["status"] for entry in battery_results}


def test_dry_run_lists_every_phase(capsys):
    rc = stress_battery.main(["--model", "m", "--dry-run"], transport=FakeTransport())

    assert rc == 0
    out = capsys.readouterr().out
    for phase in ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P9"):
        assert f"{phase}:" in out
    assert "P8" not in out  # the ssh vm_stat probe was dropped, documented


def test_a_server_that_is_down_at_start_aborts_with_90():
    class Down:
        base_url = "http://down"

        def get(self, path, *, timeout=5.0):
            raise ConnectionError("refused")

    battery = stress_battery.Battery(Down(), _fast_args(), sleep=lambda s: None)
    assert battery.run() == 90
    assert battery.results == []


def test_a_healthy_server_is_certified_without_a_cache_dir():
    battery = stress_battery.Battery(
        FakeTransport(), _fast_args(), sleep=lambda seconds: None
    )
    assert battery.run() == 0
    verdicts = _results(battery.results)
    assert verdicts["P0"] == "skip"  # no --cache-dir: stated, not failed
    for phase in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P9"):
        assert verdicts[phase] == "pass", phase


def test_p0_passes_when_the_cache_tier_grows(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    class CacheWriting(FakeTransport):
        def chat(self, messages, **kwargs):
            (cache_dir / f"artifact-{self.chats}").write_text("kv")
            return super().chat(messages, **kwargs)

    battery = stress_battery.Battery(
        CacheWriting(),
        _fast_args(["--cache-dir", str(cache_dir)]),
        sleep=lambda seconds: None,
    )
    battery.p0_cache_artifact()
    assert battery.results[0]["status"] == "pass"


def test_p0_fails_when_no_artifact_appears(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    battery = stress_battery.Battery(
        FakeTransport(),
        _fast_args(["--cache-dir", str(cache_dir)]),
        sleep=lambda seconds: None,
    )
    battery.p0_cache_artifact()
    assert battery.results[0]["status"] == "fail"


def test_p2_fails_when_the_slot_stays_busy():
    transport = FakeTransport(probe_failures=("FREE-0",))
    battery = stress_battery.Battery(
        transport, _fast_args(), sleep=lambda seconds: None
    )
    battery.p2_stop_hammer()
    assert battery.results[0]["status"] == "fail"
    assert "1 slot(s) still busy" in battery.results[0]["detail"]


def test_p3_fails_below_the_decode_floor():
    class Slow(FakeTransport):
        def stream(self, messages, **kwargs):
            result = super().stream(messages, **kwargs)
            result["seconds"] = result["chunks"]  # 1 chunk/s
            return result

    battery = stress_battery.Battery(
        Slow(), _fast_args(), sleep=lambda seconds: None
    )
    battery.p3_decode_marathon()
    assert battery.results[0]["status"] == "fail"


def test_p5_fails_when_the_mid_prefill_stop_sticks():
    class Stuck(FakeTransport):
        def chat(self, messages, **kwargs):
            content = messages[-1]["content"]
            if "SLOT" in content:
                raise TimeoutError("still prefilling")
            return super().chat(messages, **kwargs)

    battery = stress_battery.Battery(
        Stuck(), _fast_args(), sleep=lambda seconds: None
    )
    battery.p5_prefill_stop()
    assert battery.results[0]["status"] == "fail"


def test_p7_fails_when_health_dies_mid_window():
    transport = FakeTransport(fail_health_after=2)
    battery = stress_battery.Battery(
        transport, _fast_args(), sleep=lambda seconds: None
    )
    battery.p7_idle_survival()
    assert battery.results[0]["status"] == "fail"
    assert "/health failed" in battery.results[0]["detail"]


def test_p6_reports_each_unanswered_request():
    class Flaky(FakeTransport):
        def chat(self, messages, **kwargs):
            content = messages[-1]["content"]
            if "QUEUE-2" in content:
                raise TimeoutError("lost")
            return super().chat(messages, **kwargs)

    battery = stress_battery.Battery(
        Flaky(), _fast_args(), sleep=lambda seconds: None
    )
    battery.p6_concurrency()
    assert battery.results[0]["status"] == "fail"
    assert "2/3 concurrent" in battery.results[0]["detail"]


def test_main_uses_the_injected_transport(capsys):
    rc = stress_battery.main(
        [
            "--model", "m",
            "--hammer-cycles", "1",
            "--decode-tokens", "50",
            "--min-decode-tps", "5",
            "--agent-cycles", "1",
            "--prefill-records", "10",
            "--concurrent", "2",
            "--idle-seconds", "0.01",
        ],
        transport=FakeTransport(),
        sleep=lambda seconds: None,
    )

    assert rc == 0
    assert "STRESS-CERTIFIED" in capsys.readouterr().out
