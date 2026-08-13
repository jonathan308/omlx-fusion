# SPDX-License-Identifier: Apache-2.0
"""Coordinator-side supervision and self-heal for a distributed deployment.

Ported from ThunderMLX's ``auto_restart.sh`` supervisor loop, its API
liveness guard, and ``m3_warmup.py``, adapted to oMLX's architecture. What
this module provides, per boot of the rank group:

1. **A restart loop with a guard-teardown budget and a quick-failure circuit
   breaker.** When the supervised launcher dies, the deployment is relaunched
   — but a rank that tore itself down through the progress guard, the stall
   watchdog, or a peer-lost sweep (a *guard teardown*) is a memory-safe
   wedge self-heal and gets its own budget (default 6, beyond which the link
   is flapping and an operator must look), while ordinary fast deaths consume
   a quick-failure budget (3 boots inside 120 s stops the loop to protect
   JACCL/Metal memory). Relaunches back off exponentially (15 s doubling to a
   300 s cap). Deaths are classified from oMLX's structured signals — the
   parsed ``OMLX_CLUSTER_EVENT:`` events and the per-rank failure markers —
   never by grepping logs. A zero exit without an operator stop is *not* a
   stop: the launcher swallows rank exits into fast zero-exits, so it
   restarts like any other fast death.
2. **Restart state that survives the supervisor.** oMLX has no external
   supervisor process: this object lives inside the oMLX server and dies with
   it. The budgets therefore persist per deployment in the cluster registry
   directory (``~/.omlx/cluster/restart-state.json``), so a crash-looping
   deployment whose server also restarts does not get its budgets reset and
   storm forever. The limit is documented and deliberate: persistence is
   per-coordinator-Mac, and a *new plan hash* starts with fresh budgets — a
   re-planned deployment is a different machine.
3. **An API liveness guard on the rank-0 endpoint.** Probes ``/health`` with
   a 20 s timeout — heavy decode starves the event loop for seconds at a
   time, and a short probe misclassifies a busy server as dead (a production
   false-kill that cost a working cluster). A start grace ignores *everything*
   the probe could learn from a dying previous process, and a no-progress
   recycler scaled by prompt size triggers a budgeted restart — never a blind
   kill.
4. **Warmup on start.** After the ranks report ready, one sacrificial tiny
   generation runs through the private endpoint before the deployment is
   marked servable: the first real generation kept absorbing the cold-start
   wedge hazard, so the warmup takes it inside the managed loop, where a
   failure is just a failed boot routed through the restart budget.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .launch import DistributedJobSupervisor, DistributedLaunchError
from .liveness import read_marker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Death classification
# ---------------------------------------------------------------------------

OPERATOR_STOP = "operator_stop"
GUARD_TEARDOWN = "guard_teardown"
FAILURE = "failure"
UNEXPECTED_CLEAN_EXIT = "unexpected_clean_exit"

# Structured evidence that a rank tore itself down on purpose: the graceful
# exit released its Metal state first, so the death is memory-safe by
# construction and earns the separate guard-teardown budget instead of the
# quick-failure one.
_GUARD_EVENT_TYPES = frozenset({"peer_lost", "launcher_lost", "rank_stall"})
_GUARD_MARKER_PHASES = frozenset({"peer_lost", "launcher_lost", "stall"})


def classify_launcher_exit(
    *,
    returncode: int | None,
    failure_event: dict[str, Any] | None,
    rank_failure_phases: dict[int, str],
    stop_requested: bool,
) -> str:
    """Classify a rank-group death from structured signals, never log-grep.

    ThunderMLX recovered the same distinction by tailing ``startup.log`` for
    "exited with code 75"; oMLX ranks publish their exit reason as events and
    marker phases, which is strictly more reliable. A guard teardown — the
    JACCL progress guard's exit-75 surfacing as a fast zero-exit, a peer-loss
    sweep, or the stall watchdog — restarts under its own budget because the
    rank already released its wired memory on the way out.
    """

    if stop_requested:
        return OPERATOR_STOP
    event_type = str((failure_event or {}).get("type") or "")
    if event_type in _GUARD_EVENT_TYPES:
        return GUARD_TEARDOWN
    if any(phase in _GUARD_MARKER_PHASES for phase in rank_failure_phases.values()):
        return GUARD_TEARDOWN
    if returncode == 0:
        # A zero exit without an operator stop is the launcher swallowing a
        # rank's exit code — a crash or guard sweep, not a stop request.
        return UNEXPECTED_CLEAN_EXIT
    return FAILURE


# ---------------------------------------------------------------------------
# Restart budget: guard-teardown allowance + quick-failure breaker + backoff
# ---------------------------------------------------------------------------

_MAX_GUARD_TEARDOWNS_ENV = "OMLX_CLUSTER_RESTART_MAX_GUARD_TEARDOWNS"
_MAX_QUICK_FAILURES_ENV = "OMLX_CLUSTER_RESTART_MAX_QUICK_FAILURES"
_QUICK_WINDOW_ENV = "OMLX_CLUSTER_RESTART_QUICK_WINDOW_S"
_BACKOFF_INITIAL_ENV = "OMLX_CLUSTER_RESTART_BACKOFF_INITIAL_S"
_BACKOFF_MAX_ENV = "OMLX_CLUSTER_RESTART_BACKOFF_MAX_S"

_DEFAULT_MAX_GUARD_TEARDOWNS = 6
_DEFAULT_MAX_QUICK_FAILURES = 3
_DEFAULT_QUICK_WINDOW_S = 120.0
_DEFAULT_BACKOFF_INITIAL_S = 15.0
_DEFAULT_BACKOFF_MAX_S = 300.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw:
        with suppress(ValueError):
            value = float(raw)
            if value >= 0:
                return value
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw:
        with suppress(ValueError):
            value = int(raw)
            if value >= 0:
                return value
    return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return default


@dataclass(frozen=True)
class RestartDecision:
    """What the budget decided about one death."""

    restart: bool
    backoff_s: float
    reason: str
    quick_failures: int
    guard_teardowns: int


class RestartBudget:
    """The exit-75 budget and quick-failure breaker as pure arithmetic.

    The rules, faithfully ported from the production loop:

    - A boot that survived at least ``quick_window_s`` was healthy: every
      counter and the backoff reset, and its eventual death restarts at the
      initial backoff no matter how it died.
    - Inside the window, a guard teardown consumes only the guard budget
      (default 6): they are memory-safe self-heals, but past the budget the
      link is flapping and the loop stops for an operator.
    - Any other fast death consumes the quick-failure budget (default 3):
      past it the loop stops to protect JACCL/Metal memory from a boot storm.
    - Each restart sleeps the current backoff, which then doubles up to the
      cap.
    """

    def __init__(
        self,
        *,
        max_guard_teardowns: int = _DEFAULT_MAX_GUARD_TEARDOWNS,
        max_quick_failures: int = _DEFAULT_MAX_QUICK_FAILURES,
        quick_window_s: float = _DEFAULT_QUICK_WINDOW_S,
        backoff_initial_s: float = _DEFAULT_BACKOFF_INITIAL_S,
        backoff_max_s: float = _DEFAULT_BACKOFF_MAX_S,
        quick_failures: int = 0,
        guard_teardowns: int = 0,
        backoff_s: float | None = None,
    ) -> None:
        self.max_guard_teardowns = max(1, int(max_guard_teardowns))
        self.max_quick_failures = max(1, int(max_quick_failures))
        self.quick_window_s = max(0.0, float(quick_window_s))
        self.backoff_initial_s = max(0.0, float(backoff_initial_s))
        self.backoff_max_s = max(self.backoff_initial_s, float(backoff_max_s))
        self.quick_failures = max(0, int(quick_failures))
        self.guard_teardowns = max(0, int(guard_teardowns))
        self.backoff_s = (
            self.backoff_initial_s if backoff_s is None else max(0.0, backoff_s)
        )

    @classmethod
    def from_environment(cls) -> RestartBudget:
        return cls(
            max_guard_teardowns=_env_int(
                _MAX_GUARD_TEARDOWNS_ENV, _DEFAULT_MAX_GUARD_TEARDOWNS
            ),
            max_quick_failures=_env_int(
                _MAX_QUICK_FAILURES_ENV, _DEFAULT_MAX_QUICK_FAILURES
            ),
            quick_window_s=_env_float(_QUICK_WINDOW_ENV, _DEFAULT_QUICK_WINDOW_S),
            backoff_initial_s=_env_float(
                _BACKOFF_INITIAL_ENV, _DEFAULT_BACKOFF_INITIAL_S
            ),
            backoff_max_s=_env_float(_BACKOFF_MAX_ENV, _DEFAULT_BACKOFF_MAX_S),
        )

    def _take_backoff(self) -> float:
        value = self.backoff_s
        self.backoff_s = min(self.backoff_s * 2, self.backoff_max_s)
        return value

    def record(self, kind: str, runtime_s: float) -> RestartDecision:
        """Fold one death into the budgets and decide whether to relaunch."""

        if kind == OPERATOR_STOP:
            return RestartDecision(
                False, 0.0, "operator stop requested", 0, 0
            )
        if runtime_s >= self.quick_window_s:
            # A long run resets everything — including the backoff. The death
            # of a proven-healthy boot is a fresh incident, not a storm.
            self.quick_failures = 0
            self.guard_teardowns = 0
            self.backoff_s = self.backoff_initial_s
            backoff = self._take_backoff()
            return RestartDecision(
                True,
                backoff,
                f"deployment died after a healthy {runtime_s:.0f}s run; "
                f"restarting with reset budgets",
                self.quick_failures,
                self.guard_teardowns,
            )
        if kind == GUARD_TEARDOWN:
            self.guard_teardowns += 1
        else:
            # FAILURE and UNEXPECTED_CLEAN_EXIT both consume the boot budget.
            self.quick_failures += 1
        if self.quick_failures >= self.max_quick_failures:
            return RestartDecision(
                False,
                0.0,
                f"{self.quick_failures} quick startup failures inside "
                f"{self.quick_window_s:.0f}s; stopping restarts to protect "
                "JACCL/Metal memory — check the rank logs before activating "
                "again",
                self.quick_failures,
                self.guard_teardowns,
            )
        if self.guard_teardowns >= self.max_guard_teardowns:
            return RestartDecision(
                False,
                0.0,
                f"{self.guard_teardowns} guard teardowns inside the failure "
                "window; the link between the ranks is flapping — stopping "
                "restarts so an operator can inspect both Macs",
                self.quick_failures,
                self.guard_teardowns,
            )
        backoff = self._take_backoff()
        return RestartDecision(
            True,
            backoff,
            f"restarting in {backoff:.0f}s (kind={kind}, quick failures "
            f"{self.quick_failures}/{self.max_quick_failures}, guard "
            f"teardowns {self.guard_teardowns}/{self.max_guard_teardowns})",
            self.quick_failures,
            self.guard_teardowns,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "quick_failures": self.quick_failures,
            "guard_teardowns": self.guard_teardowns,
            "backoff_s": self.backoff_s,
        }


# ---------------------------------------------------------------------------
# Restart-state persistence
# ---------------------------------------------------------------------------


class RestartStateStore:
    """Atomic per-deployment restart budgets in the cluster registry dir.

    The supervisor is in-process and dies with the oMLX server; without this
    file a crash-looping deployment whose server restarts would get fresh
    budgets every boot and storm forever. Fail-open throughout: a corrupt or
    unreadable budget file must never block a launch — the in-memory budget
    still bounds the current process.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @classmethod
    def default(cls) -> RestartStateStore:
        try:
            from .registry import get_cluster_registry

            base = Path(get_cluster_registry().base_path)
        except Exception:
            base = Path.home() / ".omlx"
        return cls(base / "cluster" / "restart-state.json")

    def _read_all(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("restart state unreadable, starting fresh: %s", exc)
            return {}
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return {}
        deployments = payload.get("deployments")
        return deployments if isinstance(deployments, dict) else {}

    def load(self, deployment_id: str, plan_hash: str) -> dict[str, Any] | None:
        entry = self._read_all().get(deployment_id)
        if not isinstance(entry, dict):
            return None
        if entry.get("plan_hash") != plan_hash:
            # A re-planned deployment is a different machine: fresh budgets.
            return None
        return entry

    def save(self, deployment_id: str, plan_hash: str, state: dict[str, Any]) -> None:
        deployments = self._read_all()
        deployments[deployment_id] = {
            **state,
            "plan_hash": plan_hash,
            "updated_at": time.time(),
        }
        payload = {"schema_version": 1, "deployments": deployments}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            with suppress(OSError):
                os.chmod(self.path, 0o600)
        except OSError as exc:
            logger.warning("could not persist restart state: %s", exc)

    def clear(self, deployment_id: str) -> None:
        deployments = self._read_all()
        if deployment_id not in deployments:
            return
        del deployments[deployment_id]
        payload = {"schema_version": 1, "deployments": deployments}
        try:
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            logger.warning("could not clear restart state: %s", exc)


# ---------------------------------------------------------------------------
# Warmup on start: one sacrificial generation before "servable"
# ---------------------------------------------------------------------------

_WARMUP_ON_START_ENV = "OMLX_CLUSTER_WARMUP_ON_START"
_WARMUP_TIMEOUT_ENV = "OMLX_CLUSTER_WARMUP_TIMEOUT_S"
_WARMUP_MAX_TOKENS_ENV = "OMLX_CLUSTER_WARMUP_MAX_TOKENS"
_WARMUP_PROMPT_ENV = "OMLX_CLUSTER_WARMUP_PROMPT"

_DEFAULT_WARMUP_TIMEOUT_S = 300.0
_DEFAULT_WARMUP_MAX_TOKENS = 8
_DEFAULT_WARMUP_PROMPT = "Warm up the endpoint. Reply with one short sentence."


class WarmupFailedError(RuntimeError):
    """The sacrificial first generation did not complete."""


def run_startup_warmup(
    endpoint: str,
    *,
    max_tokens: int = _DEFAULT_WARMUP_MAX_TOKENS,
    timeout_s: float = _DEFAULT_WARMUP_TIMEOUT_S,
    prompt: str = _DEFAULT_WARMUP_PROMPT,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Drive one tiny generation through the rank-0 private endpoint.

    The first real request after a cold start pays JIT compilation, cache
    allocation and the first cross-rank collectives at once — historically the
    wedge hazard window. Taking that hit here, inside the managed supervision
    loop, turns a user-visible wedge into a retried boot. Bounded by
    ``timeout_s``; any failure raises :class:`WarmupFailedError` so the caller
    routes the boot through the restart budget instead of marking it servable.
    """

    payload = json.dumps(
        {
            "model": "default_model",
            "prompt": prompt,
            "max_tokens": max(1, int(max_tokens)),
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=max(1.0, float(timeout_s))) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise WarmupFailedError(
            f"warmup generation failed: {type(exc).__name__}: {exc}"
        ) from exc
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices:
        raise WarmupFailedError("warmup generation returned no choices")
    usage = body.get("usage") or {}
    return {
        "ok": True,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
    }


# ---------------------------------------------------------------------------
# API liveness guard
# ---------------------------------------------------------------------------

_API_GUARD_INTERVAL_ENV = "OMLX_CLUSTER_API_GUARD_INTERVAL_S"
_API_GUARD_START_GRACE_ENV = "OMLX_CLUSTER_API_GUARD_START_GRACE_S"
_API_DOWN_GUARD_ENV = "OMLX_CLUSTER_API_DOWN_GUARD_S"
_API_PROBE_TIMEOUT_ENV = "OMLX_CLUSTER_API_PROBE_TIMEOUT_S"
_NO_PROGRESS_BASE_ENV = "OMLX_CLUSTER_NO_PROGRESS_BASE_S"
_NO_PROGRESS_MARGIN_ENV = "OMLX_CLUSTER_NO_PROGRESS_MARGIN_S"
_NO_PROGRESS_CONTEXT_TPS_ENV = "OMLX_CLUSTER_NO_PROGRESS_CONTEXT_TPS"
_NO_PROGRESS_MAX_ENV = "OMLX_CLUSTER_NO_PROGRESS_MAX_S"

_DEFAULT_API_GUARD_INTERVAL_S = 15.0
_DEFAULT_API_GUARD_START_GRACE_S = 180.0
# The production override, kept as the default here: a shorter down-budget
# SIGTERM'd a live, working server whose /health was starved by decode GIL
# pressure while requests were still completing.
_DEFAULT_API_DOWN_GUARD_S = 300.0
# Busy is not dead: heavy decode starves the event loop for seconds at a
# time, so the probe timeout must be generous or the guard kills live servers.
_DEFAULT_API_PROBE_TIMEOUT_S = 20.0
_DEFAULT_NO_PROGRESS_BASE_S = 120.0
_DEFAULT_NO_PROGRESS_MARGIN_S = 120.0
_DEFAULT_NO_PROGRESS_CONTEXT_TPS = 1000.0
_DEFAULT_NO_PROGRESS_MAX_S = 900.0


def _default_prober(endpoint: str, timeout_s: float) -> bool:
    request = urllib.request.Request(endpoint.rstrip("/") + "/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, timeout_s)) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


class ApiLivenessGuard:
    """Watch one boot's rank-0 endpoint and request a recycle when it dies.

    Two independent triggers, both ending in a *budgeted* restart request —
    never a blind kill:

    - **API down.** The endpoint stops answering at all. Ignored during the
      start grace (a booting server is allowed to be unreachable); past it, a
      down-timer must exceed the down budget. The 20 s probe timeout is the
      load-bearing detail: a shorter probe misclassifies a busy server as
      dead.
    - **No progress.** The API answers, a request is active, and it has
      produced no token and no prefill progress for a limit scaled by its
      prompt size (``clamp(base, margin + prompt/context_tps, max)``) — a
      300k prompt gets ~420 s before "no first token" is declared. The start
      grace suppresses this too: during boot handover the dying previous
      process can still answer with *its* stale active request, and acting on
      that once killed a healthy 15 s-old boot.

    One guard object watches exactly one boot; restarts construct a fresh one,
    so no timer can leak across a relaunch.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        booted_at: float,
        marker_path: Path,
        on_recycle: Callable[[str], None],
        interval_s: float | None = None,
        start_grace_s: float | None = None,
        down_guard_s: float | None = None,
        probe_timeout_s: float | None = None,
        no_progress_base_s: float | None = None,
        no_progress_margin_s: float | None = None,
        no_progress_context_tps: float | None = None,
        no_progress_max_s: float | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        prober: Callable[[str, float], bool] | None = None,
        marker_reader: Callable[[Path], dict[str, Any] | None] = read_marker,
    ) -> None:
        self._endpoint = endpoint
        self._booted_at = booted_at
        self._marker_path = Path(marker_path)
        self._on_recycle = on_recycle
        self._interval_s = (
            _env_float(_API_GUARD_INTERVAL_ENV, _DEFAULT_API_GUARD_INTERVAL_S)
            if interval_s is None
            else interval_s
        )
        self._start_grace_s = (
            _env_float(_API_GUARD_START_GRACE_ENV, _DEFAULT_API_GUARD_START_GRACE_S)
            if start_grace_s is None
            else start_grace_s
        )
        self._down_guard_s = (
            _env_float(_API_DOWN_GUARD_ENV, _DEFAULT_API_DOWN_GUARD_S)
            if down_guard_s is None
            else down_guard_s
        )
        self._probe_timeout_s = (
            _env_float(_API_PROBE_TIMEOUT_ENV, _DEFAULT_API_PROBE_TIMEOUT_S)
            if probe_timeout_s is None
            else probe_timeout_s
        )
        self._no_progress_base_s = (
            _env_float(_NO_PROGRESS_BASE_ENV, _DEFAULT_NO_PROGRESS_BASE_S)
            if no_progress_base_s is None
            else no_progress_base_s
        )
        self._no_progress_margin_s = (
            _env_float(_NO_PROGRESS_MARGIN_ENV, _DEFAULT_NO_PROGRESS_MARGIN_S)
            if no_progress_margin_s is None
            else no_progress_margin_s
        )
        self._no_progress_context_tps = (
            _env_float(_NO_PROGRESS_CONTEXT_TPS_ENV, _DEFAULT_NO_PROGRESS_CONTEXT_TPS)
            if no_progress_context_tps is None
            else no_progress_context_tps
        )
        self._no_progress_max_s = (
            _env_float(_NO_PROGRESS_MAX_ENV, _DEFAULT_NO_PROGRESS_MAX_S)
            if no_progress_max_s is None
            else no_progress_max_s
        )
        self._monotonic = monotonic
        self._prober = prober if prober is not None else _default_prober
        self._marker_reader = marker_reader
        self._down_since: float | None = None
        self._zero_since: float | None = None
        self._zero_signature: tuple[int, int] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def no_progress_limit_s(self, prompt_tokens: int) -> float:
        """The no-first-token budget for a prompt of this many tokens."""

        margin = max(0.0, self._no_progress_margin_s)
        tps = max(1.0, self._no_progress_context_tps)
        maximum = max(self._no_progress_base_s, self._no_progress_max_s)
        return max(
            self._no_progress_base_s,
            min(maximum, margin + max(0, prompt_tokens) / tps),
        )

    def check_once(self, now: float) -> str | None:
        """One guard evaluation; returns a recycle reason or None."""

        age = now - self._booted_at
        if not self._prober(self._endpoint, self._probe_timeout_s):
            self._zero_since = None
            self._zero_signature = None
            if age < self._start_grace_s:
                return None
            if self._down_since is None:
                self._down_since = now
                return None
            down_for = now - self._down_since
            if down_for >= self._down_guard_s:
                return (
                    f"rank-zero API unreachable for {down_for:.0f}s while "
                    f"the launcher is still alive (probe timeout "
                    f"{self._probe_timeout_s:.0f}s)"
                )
            return None
        self._down_since = None
        if age < self._start_grace_s:
            # The probe may be answering with the *previous* process's stale
            # health during boot handover. No-progress evidence that old must
            # never recycle this boot.
            return None
        marker = self._marker_reader(self._marker_path)
        zero_prompt = self._zero_progress_prompt_tokens(marker)
        if zero_prompt is None:
            self._zero_since = None
            self._zero_signature = None
            return None
        signature = zero_prompt
        if signature != self._zero_signature:
            self._zero_signature = signature
            self._zero_since = now
            return None
        assert self._zero_since is not None
        silent_for = now - self._zero_since
        limit = self.no_progress_limit_s(zero_prompt[0])
        if silent_for >= limit:
            return (
                f"active request made no prefill/decode progress for "
                f"{silent_for:.0f}s (limit {limit:.0f}s for a "
                f"{zero_prompt[0]}-token prompt)"
            )
        return None

    @staticmethod
    def _zero_progress_prompt_tokens(
        marker: dict[str, Any] | None,
    ) -> tuple[int, int] | None:
        """(prompt tokens, active count) of a stuck request, or None.

        The marker's telemetry snapshot describes the most recently active
        request. "Stuck" is the strict ThunderMLX shape: the API accepted the
        request, and it has produced zero tokens and zero prefill work —
        accepted-but-never-started. Mid-prefill chunk silence is the rank-side
        stall watchdog's jurisdiction, not this guard's.
        """

        if not isinstance(marker, dict):
            return None
        metrics = marker.get("metrics")
        if not isinstance(metrics, dict):
            return None
        if int(metrics.get("active_requests") or 0) <= 0:
            return None
        last = metrics.get("last_request")
        if not isinstance(last, dict) or last.get("status") != "running":
            return None
        if int(last.get("completion_tokens") or 0) > 0:
            return None
        prefill = last.get("prefill_progress")
        if not isinstance(prefill, dict):
            return None
        if int(prefill.get("processed") or 0) > 0 or int(prefill.get("total") or 0) > 0:
            return None
        prompt_tokens = int(last.get("prompt_tokens") or 0)
        return prompt_tokens, int(metrics.get("active_requests") or 0)

    def run(self) -> None:
        while not self._stop.wait(max(0.1, self._interval_s)):
            try:
                reason = self.check_once(self._monotonic())
            except Exception:
                # A guard that kills itself on a bad probe leaves the boot
                # unwatched; log and keep the cadence.
                logger.exception("API liveness guard evaluation failed")
                continue
            if reason:
                logger.warning("API liveness guard requesting recycle: %s", reason)
                self._on_recycle(reason)
                return

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run, name="omlx-cluster-api-liveness", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# The restart supervisor
# ---------------------------------------------------------------------------


class ClusterRestartSupervisor:
    """Own one deployment's boots: restart loop, guards, warmup, budgets.

    Composes :class:`DistributedJobSupervisor` (one per boot attempt) rather
    than subclassing it: the job supervisor enforces a single boot's
    readiness/teardown deadlines; this object decides whether there *is*
    another boot. All restarts run on a single supervision thread — the death
    watcher and the liveness guard only feed it signals — so a death and a
    recycle request can never race each other into a double relaunch.

    Architectural limit, stated plainly: this supervisor is in-process and
    dies with the oMLX server. The persisted budgets
    (:class:`RestartStateStore`) are what keep a crash-loop bounded across a
    server restart; there is no external watchdog watching this watcher.
    """

    def __init__(
        self,
        deployment: Any,
        *,
        supervisor_factory: Callable[[], DistributedJobSupervisor] | None = None,
        launch_kwargs: dict[str, Any] | None = None,
        state_dir: str = "~/.omlx/cluster/runtime",
        store: RestartStateStore | None = None,
        budget_factory: Callable[[], RestartBudget] | None = None,
        warmup_runner: Callable[[str], dict[str, Any]] | None = None,
        guard_factory: Callable[..., ApiLivenessGuard] | None = None,
        on_ready: Callable[[Any, str], None] | None = None,
        on_down: Callable[[Any], None] | None = None,
        on_broken: Callable[[str], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        death_poll_s: float = 2.0,
    ) -> None:
        self.deployment = deployment
        if supervisor_factory is None:
            kwargs = dict(launch_kwargs or {})

            def supervisor_factory() -> DistributedJobSupervisor:
                return DistributedJobSupervisor(deployment, **kwargs)

        self._supervisor_factory = supervisor_factory
        self._state_dir = state_dir
        # None means "the default location", not "no persistence": the whole
        # point of the store is that an in-process supervisor's budgets
        # outlive it, so opting out must be an explicit act by a test.
        self._store = RestartStateStore.default() if store is None else store
        self._budget_factory = budget_factory or RestartBudget.from_environment
        self._warmup_runner = warmup_runner
        self._guard_factory = guard_factory
        self._on_ready = on_ready
        self._on_down = on_down
        self._on_broken = on_broken
        self._monotonic = monotonic
        self._death_poll_s = max(0.05, float(death_poll_s))
        self._warmup_enabled = _env_flag(_WARMUP_ON_START_ENV, True)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._poke = threading.Event()
        self._current: Any = None
        self._generation = 0
        self._booted_at = 0.0
        self._budget: RestartBudget | None = None
        self._guard: ApiLivenessGuard | None = None
        self._watch_thread: threading.Thread | None = None
        self._recycle_reason: str | None = None
        self._broken_reason: str | None = None
        self._restarts = 0
        self._last_decision: RestartDecision | None = None

    # -- introspection -------------------------------------------------------

    @property
    def endpoint(self) -> str | None:
        current = self._current
        return current.endpoint if current is not None else None

    @property
    def broken_reason(self) -> str | None:
        return self._broken_reason

    def status(self) -> dict[str, Any]:
        budget = self._budget
        return {
            "broken_reason": self._broken_reason,
            "restarts": self._restarts,
            "quick_failures": budget.quick_failures if budget else 0,
            "guard_teardowns": budget.guard_teardowns if budget else 0,
            "backoff_s": budget.backoff_s if budget else None,
            "last_decision": (
                self._last_decision.reason if self._last_decision else None
            ),
        }

    # -- budget persistence ----------------------------------------------------

    def _load_budget(self) -> RestartBudget:
        budget = self._budget_factory()
        if self._store is not None:
            entry = self._store.load(
                self.deployment.deployment_id, self.deployment.plan_hash
            )
            if entry:
                # The whole point of the store: a supervisor process restart
                # must not hand a storm fresh budgets.
                budget.quick_failures = int(entry.get("quick_failures") or 0)
                budget.guard_teardowns = int(entry.get("guard_teardowns") or 0)
                budget.backoff_s = float(
                    entry.get("backoff_s") or budget.backoff_initial_s
                )
        return budget

    def _save_budget(self) -> None:
        if self._store is not None and self._budget is not None:
            self._store.save(
                self.deployment.deployment_id,
                self.deployment.plan_hash,
                self._budget.snapshot(),
            )

    def _record_death(
        self,
        supervisor: Any,
        *,
        runtime_s: float,
        kind: str | None = None,
        detail: str | None = None,
    ) -> RestartDecision:
        if kind is None:
            kind = classify_launcher_exit(
                returncode=supervisor.status().returncode,
                failure_event=getattr(supervisor, "failure_event", None),
                rank_failure_phases=supervisor.rank_failure_phases(),
                stop_requested=self._stop.is_set(),
            )
        assert self._budget is not None
        decision = self._budget.record(kind, runtime_s)
        self._last_decision = decision
        self._save_budget()
        logger.warning(
            "cluster deployment %s died (kind=%s, runtime=%.0fs%s): %s",
            self.deployment.deployment_id,
            kind,
            runtime_s,
            f", {detail}" if detail else "",
            decision.reason,
        )
        return decision

    # -- boot attempts ----------------------------------------------------------

    def _run_warmup(self, supervisor: Any) -> None:
        if not self._warmup_enabled:
            return
        runner = self._warmup_runner or self._default_warmup_runner()
        endpoint = supervisor.endpoint
        if endpoint is None:
            raise WarmupFailedError("rank-zero endpoint was not created")
        runner(endpoint)

    @staticmethod
    def _default_warmup_runner() -> Callable[[str], dict[str, Any]]:
        max_tokens = _env_int(_WARMUP_MAX_TOKENS_ENV, _DEFAULT_WARMUP_MAX_TOKENS)
        timeout_s = _env_float(_WARMUP_TIMEOUT_ENV, _DEFAULT_WARMUP_TIMEOUT_S)
        prompt = os.environ.get(_WARMUP_PROMPT_ENV, _DEFAULT_WARMUP_PROMPT)

        def run(endpoint: str) -> dict[str, Any]:
            return run_startup_warmup(
                endpoint,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                prompt=prompt,
            )

        return run

    def _boot_until_ready(self) -> Any:
        """Attempt boots until one is ready and warmed, or budgets say stop."""

        while not self._stop.is_set():
            attempt_started = self._monotonic()
            supervisor = self._supervisor_factory()
            with self._lock:
                self._current = supervisor
                self._generation += 1
            try:
                supervisor.start()
                self._run_warmup(supervisor)
            except Exception as exc:
                runtime_s = self._monotonic() - attempt_started
                kind = FAILURE if isinstance(exc, WarmupFailedError) else None
                decision = self._record_death(
                    supervisor,
                    runtime_s=runtime_s,
                    kind=kind,
                    detail=str(exc)[:500],
                )
                self._safe_stop(supervisor)
                if self._stop.is_set():
                    raise DistributedLaunchError(
                        "cluster start was interrupted"
                    ) from exc
                if not decision.restart:
                    self._broken_reason = decision.reason
                    raise DistributedLaunchError(decision.reason) from exc
                if self._stop.wait(decision.backoff_s):
                    raise DistributedLaunchError(
                        "cluster start was interrupted"
                    ) from exc
                continue
            self._booted_at = attempt_started
            if self._on_ready is not None and supervisor.endpoint is not None:
                self._on_ready(supervisor, supervisor.endpoint)
            # A recycle requested while this boot was still loading belongs to
            # the previous boot's evidence; never let it fire on a fresh one.
            self._take_recycle_reason()
            self._poke.clear()
            self._start_guard(supervisor, attempt_started)
            return supervisor
        raise DistributedLaunchError("cluster start was interrupted")

    @staticmethod
    def _safe_stop(supervisor: Any) -> None:
        with suppress(Exception):
            supervisor.stop()

    # -- liveness guard ------------------------------------------------------------

    def _start_guard(self, supervisor: Any, booted_at: float) -> None:
        endpoint = supervisor.endpoint
        if endpoint is None:
            return
        self._stop_guard()
        factory = self._guard_factory
        if factory is None:
            marker_path = (
                Path(self._state_dir).expanduser()
                / f"{self.deployment.deployment_id}-rank-0.json"
            )

            def factory(**kwargs: Any) -> ApiLivenessGuard:
                return ApiLivenessGuard(marker_path=marker_path, **kwargs)

        self._guard = factory(
            endpoint=endpoint,
            booted_at=booted_at,
            on_recycle=self.request_restart,
            monotonic=self._monotonic,
        )
        self._guard.start()

    def _stop_guard(self) -> None:
        guard, self._guard = self._guard, None
        if guard is not None:
            guard.stop()

    # -- the supervision loop ---------------------------------------------------

    def request_restart(self, reason: str) -> None:
        """Ask the supervision thread to recycle the live boot. Never blocks."""

        with self._lock:
            if self._stop.is_set():
                return
            if self._recycle_reason is None:
                self._recycle_reason = reason
        self._poke.set()

    def _take_recycle_reason(self) -> str | None:
        with self._lock:
            reason, self._recycle_reason = self._recycle_reason, None
            return reason

    def start(self) -> dict[str, Any]:
        """Boot (with retries) until the deployment is ready and warmed."""

        self._stop.clear()
        self._broken_reason = None
        self._budget = self._load_budget()
        supervisor = self._boot_until_ready()
        self._watch_thread = threading.Thread(
            target=self._supervise_loop,
            name="omlx-cluster-restart-supervisor",
            daemon=True,
        )
        self._watch_thread.start()
        return dict(getattr(supervisor, "ready_event", None) or {})

    def stop(self) -> None:
        """Halt supervision: no more restarts, guards, or warmup attempts.

        Deliberately does *not* stop the current job supervisor: the owning
        engine calls its ``stop()`` directly (a long-standing, test-pinned
        contract), and this object must never double-terminate a rank group.
        """

        self._stop.set()
        self._poke.set()
        self._stop_guard()
        thread = self._watch_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def _supervise_loop(self) -> None:
        """Single-writer restart loop: deaths and recycles are handled here."""

        while not self._stop.is_set():
            supervisor = self._current
            if supervisor is None:
                return
            generation = self._generation
            returncode: int | None = None
            while not self._stop.is_set():
                if self._poke.wait(self._death_poll_s):
                    self._poke.clear()
                    break
                returncode = supervisor.status().returncode
                if returncode is not None:
                    break
            if self._stop.is_set():
                return
            with self._lock:
                if generation != self._generation:
                    continue
            recycle_reason = None
            if returncode is None:
                recycle_reason = self._take_recycle_reason()
                if recycle_reason is None:
                    continue
            self._stop_guard()
            if self._on_down is not None:
                self._on_down(supervisor)
            runtime_s = self._monotonic() - self._booted_at
            decision = self._record_death(
                supervisor,
                runtime_s=runtime_s,
                kind=GUARD_TEARDOWN if recycle_reason else None,
                detail=recycle_reason,
            )
            self._safe_stop(supervisor)
            if not decision.restart:
                self._broken_reason = decision.reason
                logger.error(
                    "cluster supervision halted for %s: %s",
                    self.deployment.deployment_id,
                    decision.reason,
                )
                if self._on_broken is not None:
                    self._on_broken(decision.reason)
                return
            self._restarts += 1
            logger.warning(
                "cluster supervision relaunching %s (restart #%d): %s",
                self.deployment.deployment_id,
                self._restarts,
                recycle_reason or f"launcher exited with code {returncode}",
            )
            if self._stop.wait(decision.backoff_s):
                return
            try:
                self._boot_until_ready()
            except DistributedLaunchError as exc:
                if self._broken_reason is None:
                    self._broken_reason = str(exc)
                if not self._stop.is_set() and self._on_broken is not None:
                    self._on_broken(self._broken_reason)
                return
