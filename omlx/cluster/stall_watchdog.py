# SPDX-License-Identifier: Apache-2.0
"""In-generation stall watchdog for one distributed rank.

Ported from ThunderMLX's ``run_with_watchdog.py``, adapted to oMLX's rank
machinery. The failure it exists for: when a partner rank dies or the Metal
queue wedges mid-request, a collective has no timeout, so this rank would
otherwise hang forever holding tens of GiB of wired unified memory. The
watchdog proves liveness from *activity inside the generation path* — a
completed batch step, a prefill chunk, a decoded token — and ends the rank
through the graceful exit (Metal release first) when activity stops.

Three independent clocks, mirroring the production-tuned original:

1. **Prefill chunk-silence window.** Before the first token, liveness means a
   prefill chunk completed recently. The window scales with the announced
   prompt budget (``max(base, budget / min_tps + margin)``) so a 300k-token
   prompt gets room while a small one keeps fast detection. The floor
   (``OMLX_RANK_STALL_PREFILL_S``, default 240 s) stays comfortably above the
   JACCL wheel's progress-guard window (``JACCL_PROGRESS_TIMEOUT_MS``, wheel
   default 30 s): a genuinely dead partner is always claimed by the guard
   first, and the watchdog only ever fires on a wedge the guard cannot see —
   a non-collective stall, or the ring backend, which has no guard at all.
   It must never kill a slow-but-live prefill earlier than that.
2. **Decode liveness window.** Once any token has been produced in this
   episode, silence past ``OMLX_RANK_STALL_DECODE_S`` (default 180 s) is a
   wedge. Liveness ticks are batch-step completions, not client-visible text:
   a thinking model can be silent to the client while both ranks decode in
   lockstep, and killing that live path is what strands wired memory.
3. **Max-generation backstop.** ``OMLX_RANK_STALL_MAX_GENERATION_S`` (default
   7200 s) bounds any single generation episode regardless of activity, so a
   disconnected client or a pathological run cannot hold the distributed
   deployment forever.

At stall > ``OMLX_RANK_STALL_CAPTURE_S`` (default 60 s) — long before any
fatal window — the watchdog fires the forensics capture once per episode:
every previous wedge autopsy died with the process, so evidence is taken
while the ranks are still spinning.

The idle heartbeat is observability, pinned to the token-broadcast channel:
MLX-LM's generation loop shares ``None`` over the channel every ~100 ms while
idle, which is what actually keeps the JACCL progress guard fed on an idle
cluster. The watchdog counts those shares and, every
``OMLX_RANK_IDLE_HEARTBEAT_S`` (default 60 s) of idleness, records the
channel's health in the rank marker so the liveness layer can distinguish
"idle and healthy" from "channel gone quiet".
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

_PREFILL_FATAL_ENV = "OMLX_RANK_STALL_PREFILL_S"
_DECODE_FATAL_ENV = "OMLX_RANK_STALL_DECODE_S"
_MAX_GENERATION_ENV = "OMLX_RANK_STALL_MAX_GENERATION_S"
_PREFILL_MIN_TPS_ENV = "OMLX_RANK_STALL_PREFILL_MIN_TPS"
_PREFILL_MARGIN_ENV = "OMLX_RANK_STALL_PREFILL_MARGIN_S"
_CAPTURE_ENV = "OMLX_RANK_STALL_CAPTURE_S"
_IDLE_HEARTBEAT_ENV = "OMLX_RANK_IDLE_HEARTBEAT_S"

# Production defaults from the ThunderMLX two-Mac deployment.
_DEFAULT_PREFILL_FATAL_S = 240.0
_DEFAULT_DECODE_FATAL_S = 180.0
_DEFAULT_MAX_GENERATION_S = 7200.0
_DEFAULT_PREFILL_MIN_TPS = 60.0
_DEFAULT_PREFILL_MARGIN_S = 120.0
_DEFAULT_CAPTURE_S = 60.0
_DEFAULT_IDLE_HEARTBEAT_S = 60.0

# The JACCL progress guard (patched MLX wheel) exits a rank whose collective
# makes no completion progress for JACCL_PROGRESS_TIMEOUT_MS; the wheel
# default is 30 s. Documented here because the prefill floor is chosen
# relative to it: the watchdog must never fire before the guard would on a
# dead-partner stall, or it becomes the premature killer the scaling above
# exists to prevent.
_JACCL_PROGRESS_GUARD_WHEEL_DEFAULT_S = 30.0

_WATCH_POLL_S = 5.0


def _env_float(name: str, default: float) -> float:
    """A non-negative float from the environment, or the default."""

    raw = os.environ.get(name, "").strip()
    if raw:
        with suppress(ValueError):
            value = float(raw)
            if value >= 0:
                return value
    return default


class StallWatchdog:
    """Watch one rank's generation activity and end a wedged rank gracefully.

    All hooks are cheap, lock-protected timestamp writes called from the
    serving path (telemetry batch steps, request queue, prefill budgeting).
    The watcher thread only reads. ``on_fatal`` is expected to be the rank's
    graceful exit (Metal release, then self-SIGTERM with a dead-man fallback)
    and to never return; it is injected so this module stays free of the
    worker's exit machinery and the fatal path stays testable.
    """

    def __init__(
        self,
        *,
        rank: int,
        marker: Any = None,
        emit_event: Callable[[dict[str, Any]], None] | None = None,
        on_fatal: Callable[[str], None] | None = None,
        capture: Callable[..., Any] | None = None,
        state_dir: str = "~/.omlx/cluster/runtime",
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rank = rank
        self._marker = marker
        self._emit_event = emit_event
        self._on_fatal = on_fatal
        self._capture = capture
        self._state_dir = state_dir
        self._monotonic = monotonic
        self.prefill_fatal_s = _env_float(_PREFILL_FATAL_ENV, _DEFAULT_PREFILL_FATAL_S)
        self.decode_fatal_s = _env_float(_DECODE_FATAL_ENV, _DEFAULT_DECODE_FATAL_S)
        self.max_generation_s = _env_float(
            _MAX_GENERATION_ENV, _DEFAULT_MAX_GENERATION_S
        )
        self.prefill_min_tps = _env_float(
            _PREFILL_MIN_TPS_ENV, _DEFAULT_PREFILL_MIN_TPS
        )
        self.prefill_margin_s = _env_float(
            _PREFILL_MARGIN_ENV, _DEFAULT_PREFILL_MARGIN_S
        )
        self.capture_s = _env_float(_CAPTURE_ENV, _DEFAULT_CAPTURE_S)
        self.idle_heartbeat_s = _env_float(
            _IDLE_HEARTBEAT_ENV, _DEFAULT_IDLE_HEARTBEAT_S
        )
        self._lock = threading.Lock()
        self._active_requests = 0
        self._episode_started_at: float | None = None
        self._last_activity_at: float | None = None
        self._tokens_seen = 0
        self._prefill_token_budget = 0
        self._capture_fired = False
        self._fatal_fired = False
        self._idle_beats = 0
        self._last_idle_beat_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- activity hooks, called from the serving path ----------------------

    def note_request_started(self) -> None:
        now = self._monotonic()
        with self._lock:
            self._active_requests += 1
            if self._episode_started_at is None:
                # A new episode: every per-episode clock and the forensics
                # latch reset here so one capture fires per stall episode.
                self._episode_started_at = now
                self._last_activity_at = now
                self._tokens_seen = 0
                self._prefill_token_budget = 0
                self._capture_fired = False

    def note_request_finished(self) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            if self._active_requests == 0:
                self._episode_started_at = None
                self._last_activity_at = None
                self._tokens_seen = 0
                self._prefill_token_budget = 0

    def note_activity(self) -> None:
        """One batch step, prefill chunk or decode drain completed."""

        with self._lock:
            if self._active_requests > 0:
                self._last_activity_at = self._monotonic()

    def note_token(self) -> None:
        """A token was produced; the episode is now in the decode window."""

        now = self._monotonic()
        with self._lock:
            if self._active_requests > 0:
                self._tokens_seen += 1
                self._last_activity_at = now

    def note_prefill_budget(self, tokens: int) -> None:
        """Size the prefill fatal window to the announced prompt work."""

        with self._lock:
            if self._active_requests > 0:
                self._prefill_token_budget = max(
                    self._prefill_token_budget, max(0, int(tokens))
                )

    def note_channel_share(self) -> None:
        """Record one pass over the rank-0 token-broadcast channel.

        MLX-LM's idle loop already drives this channel every ~100 ms; the
        heartbeat does not add traffic, it makes the traffic's presence (or
        absence) visible. Rank 0 only, and only while idle — an active
        request's own collectives are the liveness signal then.
        """

        if self._rank != 0 or self.idle_heartbeat_s <= 0:
            return
        now = self._monotonic()
        with self._lock:
            if self._active_requests > 0:
                return
            if now - self._last_idle_beat_at < self.idle_heartbeat_s:
                return
            self._last_idle_beat_at = now
            self._idle_beats += 1
            beats = self._idle_beats
        if self._marker is not None:
            with suppress(Exception):
                # Fail-soft like every marker write: visibility must never
                # take the rank down.
                self._marker.update(
                    "ready",
                    idle_heartbeat={
                        "count": beats,
                        "interval_seconds": self.idle_heartbeat_s,
                    },
                )

    # -- decisions ----------------------------------------------------------

    def effective_prefill_timeout(self, budget_tokens: int) -> float:
        """The prefill chunk-silence fatal window for this much prompt.

        Scaled by the announced budget at a conservative tokens/second floor
        so the window tracks real prefill work: a single huge prompt step
        blocks inside the pipeline recv for the whole chunk, and a fixed
        window would fire mid-chunk — killing a live prefill before the
        JACCL progress guard has even given up on the partner.
        """

        base = self.prefill_fatal_s
        if budget_tokens > 0 and self.prefill_min_tps > 0:
            return max(
                base,
                budget_tokens / self.prefill_min_tps + self.prefill_margin_s,
            )
        return base

    def _fatal(self, reason: str) -> None:
        """Record the wedge as structured evidence, then exit gracefully.

        The ``rank_stall`` event and the ``stall`` marker phase are what the
        coordinator's restart supervision classifies as a guard teardown — a
        memory-safe self-heal with its own restart budget — rather than an
        ordinary boot failure.

        Latched: ``on_fatal`` is the graceful exit, which *returns* (the
        unwind runs on the main thread under a dead-man timer). Without the
        latch the next watchdog poll would fire a second exit — a second
        dead-man, a second Metal release — while the first is still unwinding.
        """

        with self._lock:
            if self._fatal_fired:
                return
            self._fatal_fired = True
        if self._marker is not None:
            with suppress(Exception):
                self._marker.update("stall", error=reason)
        if self._emit_event is not None:
            with suppress(Exception):
                self._emit_event(
                    {"type": "rank_stall", "rank": self._rank, "reason": reason}
                )
        if self._on_fatal is not None:
            self._on_fatal(reason)

    def _fire_capture(self, *, stall_s: float, now: float) -> None:
        """Take live-wedge evidence on a helper thread; never blocks here."""

        capture = self._capture
        if capture is None:
            return

        def run() -> None:
            with suppress(Exception):
                capture(
                    reason=f"rank {self._rank} generation stalled {stall_s:.0f}s",
                    stall_seconds=stall_s,
                    rank=self._rank,
                    state_dir=self._state_dir,
                    now=now,
                )

        threading.Thread(
            target=run, name="omlx-cluster-stall-capture", daemon=True
        ).start()

    def watch_once(self, now: float) -> str | None:
        """One watchdog evaluation against ``now``; returns the action taken.

        Separated from the thread loop so tests can drive the clock. Returns
        None while the rank is idle or healthy, ``"capture"`` when the
        forensics hook fired, and ``"fatal"`` after the fatal path ran.
        """

        with self._lock:
            active = self._active_requests > 0
            started = self._episode_started_at
            last = self._last_activity_at
            tokens = self._tokens_seen
            budget = self._prefill_token_budget
            capture_fired = self._capture_fired
        if not active or started is None or last is None:
            return None
        stall = now - last
        elapsed = now - started
        if tokens > 0:
            timeout = self.decode_fatal_s
            kind = "decode"
        else:
            timeout = self.effective_prefill_timeout(budget)
            kind = "prefill"
        if (
            self.max_generation_s > 0
            and elapsed > self.max_generation_s
        ):
            self._fatal(
                f"rank {self._rank} generation exceeded the max duration "
                f"{elapsed:.0f}s > {self.max_generation_s:.0f}s; exiting to "
                f"release Metal memory and unblock the endpoint"
            )
            return "fatal"
        fired_capture = False
        if self.capture_s > 0 and not capture_fired and stall > self.capture_s:
            with self._lock:
                self._capture_fired = True
            self._fire_capture(stall_s=stall, now=now)
            fired_capture = True
        if timeout > 0 and stall > timeout:
            self._fatal(
                f"rank {self._rank} generation stalled {stall:.0f}s "
                f"({kind} window {timeout:.0f}s, budget {budget} tokens, "
                f"eval tokens {tokens}); partner rank likely dead or the "
                f"Metal queue is wedged; exiting to release wired memory"
            )
            return "fatal"
        return "capture" if fired_capture else None

    # -- thread lifecycle ----------------------------------------------------

    def start(self, *, poll_interval: float = _WATCH_POLL_S) -> None:
        if self._thread is not None:
            return
        self._stop.clear()

        def run() -> None:
            while not self._stop.wait(poll_interval):
                # The watchdog is the rank's last line of defence; a bad
                # evaluation must not silently disarm it.
                with suppress(Exception):
                    self.watch_once(self._monotonic())

        self._thread = threading.Thread(
            target=run, name="omlx-cluster-stall-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)
