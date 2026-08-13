# SPDX-License-Identifier: Apache-2.0
"""Stall-watchdog timing, fatal-path, idle-heartbeat and forensics tests."""

import time

from omlx.cluster import stall_watchdog
from omlx.cluster.stall_watchdog import StallWatchdog


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Marker:
    def __init__(self) -> None:
        self.updates: list[tuple[str, dict]] = []

    def update(self, phase, **extra):
        self.updates.append((phase, extra))


def _watchdog(monkeypatch, clock, **env):
    defaults = {
        "OMLX_RANK_STALL_PREFILL_S": "240",
        "OMLX_RANK_STALL_DECODE_S": "180",
        "OMLX_RANK_STALL_MAX_GENERATION_S": "7200",
        "OMLX_RANK_STALL_PREFILL_MIN_TPS": "60",
        "OMLX_RANK_STALL_PREFILL_MARGIN_S": "120",
        "OMLX_RANK_STALL_CAPTURE_S": "60",
        "OMLX_RANK_IDLE_HEARTBEAT_S": "60",
    }
    defaults.update(env)
    for name, value in defaults.items():
        monkeypatch.setenv(name, value)
    marker = _Marker()
    events: list[dict] = []
    fatal: list[str] = []
    captures: list[dict] = []
    watchdog = StallWatchdog(
        rank=0,
        marker=marker,
        emit_event=events.append,
        on_fatal=lambda reason: fatal.append(reason),
        capture=lambda **kwargs: captures.append(kwargs),
        state_dir="/tmp/state",
        monotonic=clock,
    )
    return watchdog, marker, events, fatal, captures


def test_prefill_window_scales_with_the_prompt_budget(monkeypatch):
    clock = _Clock()
    watchdog, *_ = _watchdog(monkeypatch, clock)

    assert watchdog.effective_prefill_timeout(0) == 240.0
    assert watchdog.effective_prefill_timeout(100) == 240.0
    # 300k tokens at the conservative 60 t/s floor plus the margin.
    assert watchdog.effective_prefill_timeout(300_000) == 300_000 / 60 + 120


def test_prefill_floor_stays_above_the_jaccl_progress_guard(monkeypatch):
    """The watchdog must never fire before the JACCL guard would.

    The patched wheel's progress guard claims a dead-partner stall at
    JACCL_PROGRESS_TIMEOUT_MS (wheel default 30 s). The prefill floor is 240 s
    so a slow-but-live prefill is never killed prematurely, and a genuinely
    dead partner is always the guard's teardown first.
    """

    clock = _Clock()
    watchdog, *_ = _watchdog(monkeypatch, clock)

    assert watchdog.prefill_fatal_s > (
        stall_watchdog._JACCL_PROGRESS_GUARD_WHEEL_DEFAULT_S
    )
    # Even a scaled window starts from that floor, never below it.
    assert watchdog.effective_prefill_timeout(1) >= watchdog.prefill_fatal_s


def test_continuous_activity_never_fires_regardless_of_elapsed(monkeypatch):
    clock = _Clock()
    watchdog, _, _, fatal, captures = _watchdog(monkeypatch, clock)

    watchdog.note_request_started()
    watchdog.note_prefill_budget(50_000)
    # Ten minutes of healthy prefill chunk ticks: no stall ever accumulates.
    for _ in range(120):
        clock.advance(5)
        watchdog.note_activity()
        assert watchdog.watch_once(clock()) is None

    assert fatal == []
    assert captures == []


def test_prefill_chunk_silence_fires_capture_then_fatal(monkeypatch):
    clock = _Clock()
    watchdog, marker, events, fatal, captures = _watchdog(monkeypatch, clock)

    watchdog.note_request_started()
    watchdog.note_prefill_budget(1000)
    clock.advance(30)
    assert watchdog.watch_once(clock()) is None
    clock.advance(31)  # 61 s of silence: forensics fire, fatality does not.
    assert watchdog.watch_once(clock()) == "capture"
    assert len(captures) == 1
    assert captures[0]["rank"] == 0
    assert captures[0]["stall_seconds"] > 60
    clock.advance(100)  # 161 s: still inside the 240 s prefill window.
    assert watchdog.watch_once(clock()) is None
    clock.advance(80)  # 241 s of chunk silence: fatal.
    assert watchdog.watch_once(clock()) == "fatal"

    assert len(fatal) == 1
    assert "prefill" in fatal[0]
    assert marker.updates[-1][0] == "stall"
    assert "error" in marker.updates[-1][1]
    assert events[-1]["type"] == "rank_stall"
    assert events[-1]["rank"] == 0
    # One capture per stall episode, even across many later evaluations.
    assert len(captures) == 1


def test_decode_window_applies_once_a_token_exists(monkeypatch):
    clock = _Clock()
    watchdog, _, _, fatal, _ = _watchdog(monkeypatch, clock)

    watchdog.note_request_started()
    watchdog.note_token()
    clock.advance(181)  # past the 180 s decode window
    assert watchdog.watch_once(clock()) == "fatal"
    assert fatal and "decode" in fatal[0]


def test_max_generation_backstop_bounds_a_live_episode(monkeypatch):
    clock = _Clock()
    watchdog, _, _, fatal, _ = _watchdog(monkeypatch, clock)

    watchdog.note_request_started()
    # Steady activity past the two-hour cap: never a stall, still fatal.
    for _ in range(721):
        clock.advance(10)
        watchdog.note_activity()
    assert watchdog.watch_once(clock()) == "fatal"
    assert fatal and "max duration" in fatal[0]


def test_finishing_the_request_disarms_everything(monkeypatch):
    """A clean exit leaves no capture and no fatal behind."""

    clock = _Clock()
    watchdog, _, _, fatal, captures = _watchdog(monkeypatch, clock)

    watchdog.note_request_started()
    watchdog.note_activity()
    clock.advance(5)
    watchdog.note_request_finished()
    clock.advance(10000)  # far past every window
    assert watchdog.watch_once(clock()) is None
    assert fatal == []
    assert captures == []


def test_new_episode_rearms_the_capture(monkeypatch):
    clock = _Clock()
    watchdog, _, _, fatal, captures = _watchdog(monkeypatch, clock)

    watchdog.note_request_started()
    clock.advance(61)
    assert watchdog.watch_once(clock()) == "capture"
    watchdog.note_activity()
    watchdog.note_request_finished()

    clock.advance(10)
    watchdog.note_request_started()
    clock.advance(61)
    assert watchdog.watch_once(clock()) == "capture"
    assert len(captures) == 2
    assert fatal == []


def test_idle_heartbeat_ticks_only_when_idle_and_due(monkeypatch):
    clock = _Clock()
    watchdog, marker, _, _, _ = _watchdog(monkeypatch, clock)

    watchdog.note_channel_share()
    assert [
        update for update in marker.updates if "idle_heartbeat" in update[1]
    ], "first idle channel observation should tick"

    watchdog.note_channel_share()  # 0 s later: not due
    beats = [u for u in marker.updates if "idle_heartbeat" in u[1]]
    assert len(beats) == 1

    clock.advance(61)
    watchdog.note_channel_share()  # due again
    beats = [u for u in marker.updates if "idle_heartbeat" in u[1]]
    assert len(beats) == 2
    assert beats[-1][1]["idle_heartbeat"]["count"] == 2

    watchdog.note_request_started()  # busy: the channel is not the signal
    clock.advance(120)
    watchdog.note_channel_share()
    assert len([u for u in marker.updates if "idle_heartbeat" in u[1]]) == 2


def test_idle_heartbeat_is_rank_zero_only(monkeypatch):
    clock = _Clock()
    marker = _Marker()
    watchdog = StallWatchdog(
        rank=1,
        marker=marker,
        monotonic=clock,
    )

    watchdog.note_channel_share()
    clock.advance(3600)
    watchdog.note_channel_share()

    assert marker.updates == []


def test_fatal_path_runs_without_optional_collaborators(monkeypatch):
    clock = _Clock()
    watchdog, _, _, fatal, _ = _watchdog(monkeypatch, clock)
    watchdog._marker = None
    watchdog._emit_event = None
    watchdog._capture = None

    watchdog.note_request_started()
    clock.advance(300)

    assert watchdog.watch_once(clock()) == "fatal"
    assert len(fatal) == 1


def test_watch_thread_evaluates_and_stops(monkeypatch):
    clock = _Clock()
    watchdog, _, _, fatal, _ = _watchdog(monkeypatch, clock)

    watchdog.start(poll_interval=0.01)
    watchdog.note_request_started()
    clock.advance(300)
    deadline = time.monotonic() + 5
    while not fatal and time.monotonic() < deadline:
        time.sleep(0.01)
    watchdog.stop()

    assert len(fatal) == 1
    assert watchdog._thread is None
