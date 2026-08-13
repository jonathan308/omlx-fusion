# SPDX-License-Identifier: Apache-2.0
"""Restart-budget, liveness-guard, warmup and restart-supervisor tests."""

import json
import time
from types import SimpleNamespace

import pytest

from omlx.cluster.launch import DistributedLaunchError
from omlx.cluster.supervision import (
    FAILURE,
    GUARD_TEARDOWN,
    OPERATOR_STOP,
    UNEXPECTED_CLEAN_EXIT,
    ApiLivenessGuard,
    ClusterRestartSupervisor,
    RestartBudget,
    RestartStateStore,
    WarmupFailedError,
    classify_launcher_exit,
    run_startup_warmup,
)

# ---------------------------------------------------------------------------
# Death classification — structured signals, never log-grep
# ---------------------------------------------------------------------------


def test_operator_stop_is_never_restarted():
    assert (
        classify_launcher_exit(
            returncode=1,
            failure_event={"type": "rank_stall", "reason": "wedged"},
            rank_failure_phases={1: "stall"},
            stop_requested=True,
        )
        == OPERATOR_STOP
    )


def test_guard_events_and_marker_phases_classify_as_guard_teardowns():
    assert (
        classify_launcher_exit(
            returncode=0,
            failure_event={"type": "peer_lost", "reason": "rank 1 gone"},
            rank_failure_phases={},
            stop_requested=False,
        )
        == GUARD_TEARDOWN
    )
    assert (
        classify_launcher_exit(
            returncode=0,
            failure_event={"type": "rank_stall", "reason": "no progress"},
            rank_failure_phases={},
            stop_requested=False,
        )
        == GUARD_TEARDOWN
    )
    # The event can be lost (launcher reaped first); the marker phase remains.
    assert (
        classify_launcher_exit(
            returncode=75,
            failure_event=None,
            rank_failure_phases={1: "stall"},
            stop_requested=False,
        )
        == GUARD_TEARDOWN
    )
    assert (
        classify_launcher_exit(
            returncode=0,
            failure_event=None,
            rank_failure_phases={0: "launcher_lost"},
            stop_requested=False,
        )
        == GUARD_TEARDOWN
    )


def test_zero_exit_without_a_stop_is_a_crash_not_an_operator_stop():
    assert (
        classify_launcher_exit(
            returncode=0,
            failure_event=None,
            rank_failure_phases={},
            stop_requested=False,
        )
        == UNEXPECTED_CLEAN_EXIT
    )
    assert (
        classify_launcher_exit(
            returncode=1,
            failure_event=None,
            rank_failure_phases={0: "failed"},
            stop_requested=False,
        )
        == FAILURE
    )


# ---------------------------------------------------------------------------
# Restart budget: increments, breakers, backoff schedule
# ---------------------------------------------------------------------------


def _budget(**overrides) -> RestartBudget:
    kwargs = {
        "max_guard_teardowns": 6,
        "max_quick_failures": 3,
        "quick_window_s": 120.0,
        "backoff_initial_s": 15.0,
        "backoff_max_s": 300.0,
    }
    kwargs.update(overrides)
    return RestartBudget(**kwargs)


def _fast_budget(**overrides) -> RestartBudget:
    """Budget with negligible backoffs so retry loops stay instant."""

    kwargs = {"backoff_initial_s": 0.01, "backoff_max_s": 0.02}
    kwargs.update(overrides)
    return _budget(**kwargs)


def test_quick_failures_trip_the_breaker_on_the_third_fast_death():
    budget = _budget()

    assert budget.record(FAILURE, runtime_s=5).restart is True
    assert budget.record(FAILURE, runtime_s=5).restart is True
    decision = budget.record(FAILURE, runtime_s=5)

    assert decision.restart is False
    assert "quick startup failures" in decision.reason
    assert decision.quick_failures == 3


def test_guard_teardowns_have_their_own_budget_and_do_not_consume_quick_ones():
    budget = _budget()

    for _ in range(5):
        decision = budget.record(GUARD_TEARDOWN, runtime_s=5)
        assert decision.restart is True
    assert budget.quick_failures == 0

    decision = budget.record(GUARD_TEARDOWN, runtime_s=5)
    assert decision.restart is False
    assert "flapping" in decision.reason


def test_a_healthy_run_resets_every_budget_and_the_backoff():
    budget = _budget()
    budget.record(FAILURE, runtime_s=5)
    budget.record(FAILURE, runtime_s=5)
    assert budget.quick_failures == 2

    decision = budget.record(FAILURE, runtime_s=500)

    assert decision.restart is True
    assert budget.quick_failures == 0
    assert budget.guard_teardowns == 0
    assert decision.backoff_s == 15.0


def test_backoff_doubles_to_the_cap():
    budget = _budget(max_quick_failures=10)

    seen = [budget.record(FAILURE, runtime_s=5).backoff_s for _ in range(6)]

    assert seen == [15.0, 30.0, 60.0, 120.0, 240.0, 300.0]


def test_operator_stop_does_not_consume_budget():
    budget = _budget()
    decision = budget.record(OPERATOR_STOP, runtime_s=5)
    assert decision.restart is False
    assert budget.quick_failures == 0
    assert budget.guard_teardowns == 0


# ---------------------------------------------------------------------------
# Restart-state persistence
# ---------------------------------------------------------------------------


def test_restart_state_survives_a_supervisor_restart(tmp_path):
    store = RestartStateStore(tmp_path / "cluster" / "restart-state.json")

    first = _budget()
    first.record(FAILURE, runtime_s=5)
    first.record(FAILURE, runtime_s=5)
    store.save("dep", "hash-1", first.snapshot())

    # A new supervisor process loads the same budgets: the third fast death
    # trips the breaker instead of starting a fresh storm.
    reloaded = _budget()
    entry = store.load("dep", "hash-1")
    assert entry is not None
    reloaded.quick_failures = entry["quick_failures"]
    reloaded.backoff_s = entry["backoff_s"]

    assert reloaded.quick_failures == 2
    assert reloaded.backoff_s == 60.0
    assert reloaded.record(FAILURE, runtime_s=5).restart is False


def test_restart_state_is_scoped_to_the_plan(tmp_path):
    store = RestartStateStore(tmp_path / "restart-state.json")
    store.save(
        "dep",
        "hash-1",
        {"quick_failures": 2, "guard_teardowns": 1, "backoff_s": 30.0},
    )

    assert store.load("dep", "hash-1")["quick_failures"] == 2
    # A re-planned deployment is a different machine: fresh budgets.
    assert store.load("dep", "hash-2") is None
    assert store.load("other-dep", "hash-1") is None


def test_restart_state_store_is_fail_open_on_corruption(tmp_path):
    path = tmp_path / "restart-state.json"
    path.write_text("{not json")
    store = RestartStateStore(path)

    assert store.load("dep", "hash") is None
    store.save("dep", "hash", {"quick_failures": 1})
    assert store.load("dep", "hash")["quick_failures"] == 1


def test_restart_state_roundtrip_file_permissions(tmp_path):
    store = RestartStateStore(tmp_path / "cluster" / "restart-state.json")
    store.save(
        "dep",
        "hash",
        {"quick_failures": 1, "guard_teardowns": 0, "backoff_s": 15.0},
    )

    payload = json.loads(store.path.read_text())
    assert payload["schema_version"] == 1
    assert payload["deployments"]["dep"]["quick_failures"] == 1
    assert payload["deployments"]["dep"]["plan_hash"] == "hash"
    assert store.path.stat().st_mode & 0o777 == 0o600


# ---------------------------------------------------------------------------
# Warmup on start
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: dict, status: int = 200) -> None:
        self._body = json.dumps(body).encode()
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_warmup_posts_one_tiny_generation():
    calls: list[dict] = []

    def opener(request, timeout):
        calls.append(
            {
                "url": request.full_url,
                "payload": json.loads(request.data.decode()),
                "timeout": timeout,
            }
        )
        return _FakeResponse(
            {
                "choices": [{"text": "ok", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4},
            }
        )

    result = run_startup_warmup(
        "http://127.0.0.1:43210/",
        max_tokens=4,
        timeout_s=12,
        opener=opener,
    )

    assert result["ok"] is True
    assert result["completion_tokens"] == 4
    assert calls[0]["url"] == "http://127.0.0.1:43210/v1/completions"
    assert calls[0]["payload"]["max_tokens"] == 4
    assert calls[0]["payload"]["stream"] is False
    assert calls[0]["timeout"] == 12


def test_warmup_failure_raises_for_budget_routing():
    def failing_opener(_request, timeout):
        raise TimeoutError("endpoint wedged")

    with pytest.raises(WarmupFailedError, match="warmup generation failed"):
        run_startup_warmup(
            "http://127.0.0.1:1", timeout_s=1, opener=failing_opener
        )

    with pytest.raises(WarmupFailedError, match="no choices"):
        run_startup_warmup(
            "http://127.0.0.1:1",
            timeout_s=1,
            opener=lambda _r, timeout: _FakeResponse({"choices": []}),
        )


# ---------------------------------------------------------------------------
# API liveness guard
# ---------------------------------------------------------------------------


def _guard(tmp_path, *, prober, marker=None, **overrides):
    marker_path = tmp_path / "marker.json"
    if marker is not None:
        marker_path.write_text(json.dumps(marker))
    kwargs = {
        "endpoint": "http://127.0.0.1:43210",
        "booted_at": 0.0,
        "marker_path": marker_path,
        "on_recycle": lambda _reason: None,
        "interval_s": 15.0,
        "start_grace_s": 180.0,
        "down_guard_s": 300.0,
        "probe_timeout_s": 20.0,
        "prober": prober,
    }
    kwargs.update(overrides)
    return ApiLivenessGuard(**kwargs)


def _running_marker(*, prompt_tokens=128, completion_tokens=0,
                    prefill_processed=0, prefill_total=0, active=1):
    return {
        "metrics": {
            "active_requests": active,
            "last_request": {
                "status": "running",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prefill_progress": {
                    "active": True,
                    "processed": prefill_processed,
                    "total": prefill_total,
                },
            },
        }
    }


def test_unreachable_api_inside_start_grace_is_ignored(tmp_path):
    guard = _guard(tmp_path, prober=lambda *_a: False)

    assert guard.check_once(100.0) is None
    assert guard._down_since is None


def test_unreachable_api_recycles_only_after_the_down_budget(tmp_path):
    guard = _guard(tmp_path, prober=lambda *_a: False)

    assert guard.check_once(200.0) is None  # past grace: down timer starts
    assert guard.check_once(200.0 + 299.0) is None
    reason = guard.check_once(200.0 + 301.0)
    assert reason and "unreachable" in reason


def test_api_recovery_resets_the_down_timer(tmp_path):
    state = {"alive": False}
    guard = _guard(tmp_path, prober=lambda *_a: state["alive"])

    assert guard.check_once(200.0) is None
    assert guard._down_since == 200.0
    state["alive"] = True
    assert guard.check_once(400.0) is None
    assert guard._down_since is None
    state["alive"] = False
    assert guard.check_once(500.0) is None
    assert guard._down_since == 500.0


def test_probe_timeout_is_generous_enough_for_a_busy_server(tmp_path):
    seen_timeouts: list[float] = []

    def prober(_endpoint, timeout_s):
        seen_timeouts.append(timeout_s)
        return True

    guard = _guard(tmp_path, prober=prober)
    guard.check_once(1000.0)

    # 20 s, not 5 s: a busy server is not a dead one.
    assert seen_timeouts == [20.0]


def test_no_progress_recycler_scales_with_prompt_size(tmp_path):
    guard = _guard(
        tmp_path,
        prober=lambda *_a: True,
        marker=_running_marker(prompt_tokens=300_000),
    )

    # clamp(base 120, margin 120 + 300000/1000, max 900) == 420
    assert guard.no_progress_limit_s(300_000) == 420.0
    assert guard.no_progress_limit_s(0) == 120.0
    assert guard.no_progress_limit_s(10_000_000) == 900.0

    assert guard.check_once(200.0) is None  # first sighting starts the timer
    assert guard.check_once(200.0 + 419.0) is None
    reason = guard.check_once(200.0 + 421.0)
    assert reason and "no prefill/decode progress" in reason


def test_no_progress_recycler_ignores_stale_health_during_start_grace(tmp_path):
    """The previous process's stale active request must not recycle a boot."""

    guard = _guard(
        tmp_path,
        prober=lambda *_a: True,
        marker=_running_marker(),
    )

    assert guard.check_once(15.0) is None
    assert guard.check_once(150.0) is None
    assert guard._zero_since is None


def test_no_progress_recycler_ignores_progressing_requests(tmp_path):
    for marker in (
        _running_marker(completion_tokens=3),
        _running_marker(prefill_processed=10, prefill_total=100),
        _running_marker(active=0),
        {"metrics": {"active_requests": 0, "last_request": None}},
        {},
    ):
        guard = _guard(tmp_path, prober=lambda *_a: True, marker=marker)
        assert guard.check_once(1000.0) is None
        assert guard.check_once(100000.0) is None


def test_liveness_guard_thread_recycles_and_exits(tmp_path):
    clock = [0.0]
    recycled: list[str] = []
    (tmp_path / "marker.json").write_text(json.dumps({}))
    guard = ApiLivenessGuard(
        endpoint="http://127.0.0.1:1",
        booted_at=0.0,
        marker_path=tmp_path / "marker.json",
        on_recycle=recycled.append,
        interval_s=0.01,
        start_grace_s=0.1,
        down_guard_s=0.2,
        probe_timeout_s=1.0,
        monotonic=lambda: clock[0],
        prober=lambda *_a: False,
    )

    guard.start()
    deadline = time.monotonic() + 5
    while not recycled and time.monotonic() < deadline:
        clock[0] += 0.05
        time.sleep(0.01)
    guard.stop()

    assert recycled and "unreachable" in recycled[0]


# ---------------------------------------------------------------------------
# The restart supervisor, with a fully fake job supervisor
# ---------------------------------------------------------------------------


class _FakeJobSupervisor:
    """Stands in for DistributedJobSupervisor; scripted outcomes."""

    def __init__(self, script, shared):
        self._script = script
        self._shared = shared
        self.endpoint = "http://127.0.0.1:43210"
        self.failure_event = None
        self.ready_event = {"type": "ready"}
        self._returncode: int | None = None
        self.stopped = False

    def start(self):
        self._shared["boots"] += 1
        outcome = self._script.pop(0) if self._script else "ready"
        if outcome == "fail":
            self._returncode = 1
            raise DistributedLaunchError("boot failed")
        if outcome == "stall_death":
            # Boot "succeeds", then the rank stalls and dies a guard death.
            self._returncode = 0
            self.failure_event = {"type": "rank_stall", "reason": "wedged"}
            return
        if outcome == "fail_death":
            # Boot "succeeds", then the launcher dies an ordinary fast death.
            self._returncode = 1
            return
        self._returncode = None

    def stop(self):
        self.stopped = True
        if self._returncode is None:
            self._returncode = 0

    def status(self):
        return SimpleNamespace(returncode=self._returncode)

    def rank_failure_phases(self):
        return {}


class _FakeGuard:
    def __init__(self, **_kwargs):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self, **_kwargs):
        self.stopped = True


def _supervisor(tmp_path, script, **overrides):
    shared = {"boots": 0}
    deployment = SimpleNamespace(deployment_id="dep-1", plan_hash="hash-1")

    def factory():
        return _FakeJobSupervisor(script, shared)

    kwargs = {
        "supervisor_factory": factory,
        "store": RestartStateStore(tmp_path / "restart-state.json"),
        "warmup_runner": lambda endpoint: {"ok": True},
        "guard_factory": lambda **kw: _FakeGuard(**kw),
        "budget_factory": _fast_budget,
        "death_poll_s": 0.01,
    }
    kwargs.update(overrides)
    return ClusterRestartSupervisor(deployment, **kwargs), shared


def test_supervisor_retries_a_failed_boot_within_budget(tmp_path):
    supervisor_obj, shared = _supervisor(tmp_path, ["fail", "ready"])

    ready = supervisor_obj.start()

    assert ready == {"type": "ready"}
    assert shared["boots"] == 2
    assert supervisor_obj.status()["quick_failures"] == 1
    supervisor_obj.stop()


def test_supervisor_stops_after_the_quick_failure_breaker_trips(tmp_path):
    supervisor_obj, shared = _supervisor(tmp_path, ["fail", "fail", "fail"])

    with pytest.raises(DistributedLaunchError, match="quick startup failures"):
        supervisor_obj.start()

    assert shared["boots"] == 3
    assert supervisor_obj.broken_reason is not None


def test_supervisor_budget_persists_across_instances(tmp_path):
    store = RestartStateStore(tmp_path / "restart-state.json")
    first, shared = _supervisor(
        tmp_path,
        ["fail", "fail"],
        store=store,
        budget_factory=lambda: _fast_budget(max_quick_failures=2),
    )
    with pytest.raises(DistributedLaunchError):
        first.start()
    assert shared["boots"] == 2

    # A new supervisor (the server restarted) resumes the spent budget: one
    # more fast death trips the breaker immediately instead of a fresh storm.
    second, shared2 = _supervisor(
        tmp_path,
        ["fail", "ready"],
        store=store,
        budget_factory=lambda: _fast_budget(max_quick_failures=2),
    )
    with pytest.raises(DistributedLaunchError, match="quick startup failures"):
        second.start()
    assert shared2["boots"] == 1


def test_warmup_failure_is_a_failed_boot_and_retries(tmp_path):
    attempts: list[str] = []

    def flaky_warmup(endpoint):
        attempts.append(endpoint)
        if len(attempts) == 1:
            raise WarmupFailedError("wedged first generation")
        return {"ok": True}

    supervisor_obj, shared = _supervisor(
        tmp_path, ["ready", "ready"], warmup_runner=flaky_warmup
    )

    assert supervisor_obj.start() == {"type": "ready"}
    # The sacrificial generation ran before "servable", failed once, and the
    # boot was retried through the budget.
    assert len(attempts) == 2
    assert shared["boots"] == 2
    assert supervisor_obj.status()["quick_failures"] == 1
    supervisor_obj.stop()


def test_warmup_failures_trip_the_breaker(tmp_path):
    def bad_warmup(_endpoint):
        raise WarmupFailedError("wedged first generation")

    supervisor_obj, shared = _supervisor(
        tmp_path,
        ["ready", "ready", "ready", "ready"],
        warmup_runner=bad_warmup,
    )

    with pytest.raises(DistributedLaunchError, match="quick startup failures"):
        supervisor_obj.start()
    assert shared["boots"] == 3


def test_warmup_can_be_skipped_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_WARMUP_ON_START", "0")

    def forbidden_warmup(_endpoint):
        raise AssertionError("warmup must not run when disabled")

    supervisor_obj, shared = _supervisor(
        tmp_path, ["ready"], warmup_runner=forbidden_warmup
    )

    assert supervisor_obj.start() == {"type": "ready"}
    assert shared["boots"] == 1
    supervisor_obj.stop()


def test_background_guard_death_relaunches_under_the_guard_budget(tmp_path):
    supervisor_obj, shared = _supervisor(tmp_path, ["stall_death", "ready"])
    downed: list[object] = []
    readied: list[str] = []
    supervisor_obj._on_down = downed.append
    supervisor_obj._on_ready = lambda _s, endpoint: readied.append(endpoint)
    supervisor_obj.start()

    deadline = time.monotonic() + 5
    while shared["boots"] < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    supervisor_obj.stop()

    assert shared["boots"] == 2
    assert len(readied) == 2
    status = supervisor_obj.status()
    assert status["guard_teardowns"] == 1
    assert status["quick_failures"] == 0
    assert status["restarts"] == 1


def test_breaker_trip_on_background_deaths_calls_on_broken(tmp_path):
    supervisor_obj, shared = _supervisor(
        tmp_path,
        ["fail_death", "fail_death", "ready"],
        budget_factory=lambda: _fast_budget(max_quick_failures=2),
    )
    broken: list[str] = []
    supervisor_obj._on_broken = broken.append
    supervisor_obj.start()

    deadline = time.monotonic() + 5
    while not broken and time.monotonic() < deadline:
        time.sleep(0.02)
    supervisor_obj.stop()

    assert broken and "quick startup failures" in broken[0]
    assert supervisor_obj.status()["broken_reason"] == broken[0]
    assert shared["boots"] == 2


def test_operator_stop_during_a_backoff_wait_ends_the_loop(tmp_path):
    supervisor_obj, shared = _supervisor(
        tmp_path,
        ["stall_death", "ready"],
        budget_factory=_budget,  # real 15 s backoff: stop must interrupt it
    )
    supervisor_obj.start()

    deadline = time.monotonic() + 5
    while shared["boots"] < 2 and time.monotonic() < deadline:
        # The death is noticed immediately; the reboot is behind a 15 s
        # backoff that stop() must cut short.
        time.sleep(0.01)
        if supervisor_obj._last_decision is not None:
            break
    supervisor_obj.stop()
    boots_at_stop = shared["boots"]
    time.sleep(0.2)

    assert supervisor_obj._last_decision is not None
    assert shared["boots"] == boots_at_stop == 1


def test_recycle_request_relaunches_a_live_boot(tmp_path):
    supervisor_obj, shared = _supervisor(tmp_path, ["ready", "ready"])
    supervisor_obj.start()
    first = supervisor_obj._current

    supervisor_obj.request_restart("API down for 301s")
    deadline = time.monotonic() + 5
    while shared["boots"] < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    supervisor_obj.stop()

    assert shared["boots"] == 2
    assert first.stopped is True
    # A liveness recycle is a guard-budget death, not a quick failure.
    assert supervisor_obj.status()["quick_failures"] == 0
    assert supervisor_obj.status()["guard_teardowns"] == 1
