# SPDX-License-Identifier: Apache-2.0
"""Idle Metal and distributed data-plane keepwarm primitives.

The controller in this module is deliberately framework-neutral.  Single-node
engines execute its actions on their private serialized MLX executor, while
cluster ranks execute them from MLX-LM's synchronized generation thread.  No
keepwarm path owns a second Metal or collective worker.  The local state
machine is adapted from ThunderMLX's Apache-2.0 keepwarm implementation and
adds oMLX-specific admission, cache-clear, shutdown, and live-settings gates.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable


def _enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _int(
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    value = max(minimum, value)
    return min(value, maximum) if maximum is not None else value


@dataclass(frozen=True)
class KeepwarmConfig:
    """One rank-identical keepwarm policy, disabled unless explicitly enabled."""

    enabled: bool = False
    interval_seconds: float = 10.0
    idle_after_seconds: float = 2.0
    matrix_size: int = 1
    repeats: int = 1
    request_start_enabled: bool = True
    request_start_idle_seconds: float = 2.0
    request_start_matrix_size: int = 128
    post_response_enabled: bool = True
    post_response_delay_seconds: float = 5.0
    post_response_matrix_size: int = 128
    large_cache_tokens: int = 8192
    large_cache_interval_seconds: float = 60.0
    slow_threshold_seconds: float = 1.0
    slow_backoff_seconds: float = 60.0
    dataplane_ping: bool = True

    @classmethod
    def from_env(cls) -> "KeepwarmConfig":
        return cls(
            enabled=_enabled("OMLX_KEEPWARM", False),
            interval_seconds=_float(
                "OMLX_KEEPWARM_INTERVAL_SECONDS", 10.0, minimum=0.25
            ),
            idle_after_seconds=_float("OMLX_KEEPWARM_IDLE_AFTER_SECONDS", 2.0),
            matrix_size=_int("OMLX_KEEPWARM_MATRIX_SIZE", 1, minimum=1, maximum=1024),
            repeats=_int("OMLX_KEEPWARM_REPEATS", 1, minimum=1, maximum=16),
            request_start_enabled=_enabled("OMLX_KEEPWARM_REQUEST_START", True),
            request_start_idle_seconds=_float(
                "OMLX_KEEPWARM_REQUEST_START_IDLE_SECONDS", 2.0
            ),
            request_start_matrix_size=_int(
                "OMLX_KEEPWARM_REQUEST_START_MATRIX_SIZE",
                128,
                minimum=1,
                maximum=1024,
            ),
            post_response_enabled=_enabled("OMLX_KEEPWARM_POST_RESPONSE", True),
            post_response_delay_seconds=_float(
                "OMLX_KEEPWARM_POST_RESPONSE_DELAY_SECONDS", 5.0
            ),
            post_response_matrix_size=_int(
                "OMLX_KEEPWARM_POST_RESPONSE_MATRIX_SIZE",
                128,
                minimum=1,
                maximum=1024,
            ),
            large_cache_tokens=_int("OMLX_KEEPWARM_LARGE_CACHE_TOKENS", 8192),
            large_cache_interval_seconds=_float(
                "OMLX_KEEPWARM_LARGE_CACHE_INTERVAL_SECONDS",
                60.0,
                minimum=0.25,
            ),
            slow_threshold_seconds=_float("OMLX_KEEPWARM_SLOW_THRESHOLD_SECONDS", 1.0),
            slow_backoff_seconds=_float("OMLX_KEEPWARM_SLOW_BACKOFF_SECONDS", 60.0),
            dataplane_ping=_enabled("OMLX_CLUSTER_KEEPWARM_DATAPLANE_PING", True),
        )


@dataclass(frozen=True)
class KeepwarmAction:
    kind: str
    matrix_size: int
    repeats: int
    idle_seconds: float
    cache_tokens: int = 0


class KeepwarmController:
    """Thread-safe request/idle state machine with bounded telemetry."""

    def __init__(
        self,
        config: KeepwarmConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or KeepwarmConfig.from_env()
        self._clock = clock
        self._lock = threading.Lock()
        now = float(clock())
        self._last_activity_at = now
        self._last_touch_at = float("-inf")
        self._slow_until = float("-inf")
        self._request_active = False
        self._cache_armed = False
        self._cache_tokens = 0
        self._clear_inhibited = False
        self._post_response_pending = False
        self._closed = False
        self._count = 0
        self._failures = 0
        self._skips = 0
        self._slow_count = 0
        self._last_event: dict[str, Any] | None = None

    def configure(self, enabled: bool) -> None:
        """Apply the master switch live without replacing request state."""

        with self._lock:
            self.config = replace(self.config, enabled=bool(enabled))
            if not enabled:
                self._post_response_pending = False

    def should_execute(self, action: KeepwarmAction) -> bool:
        """Final gate for an action selected before a live state change."""

        with self._lock:
            if self._closed or not self.config.enabled:
                return False
            if action.kind == "request_start":
                return self._request_active
            return self._cache_armed and not self._request_active

    def request_start_action(self) -> KeepwarmAction | None:
        now = float(self._clock())
        with self._lock:
            idle = max(0.0, now - self._last_activity_at)
            self._request_active = True
            self._post_response_pending = False
            self._last_activity_at = now
            if (
                self._closed
                or not self.config.enabled
                or not self._cache_armed
                or not self.config.request_start_enabled
                or idle < self.config.request_start_idle_seconds
                or now < self._slow_until
            ):
                return None
            return KeepwarmAction(
                kind="request_start",
                matrix_size=self.config.request_start_matrix_size,
                repeats=self.config.repeats,
                idle_seconds=idle,
                cache_tokens=self._cache_tokens,
            )

    def observe_request_state(
        self,
        active: bool,
        *,
        cache_tokens: int | None = None,
    ) -> None:
        """Record real request transitions and arm only after useful activity."""

        now = float(self._clock())
        with self._lock:
            if self._closed:
                return
            if cache_tokens is not None:
                self._cache_tokens = max(0, int(cache_tokens))
                if self._cache_tokens > 0 and not self._clear_inhibited:
                    self._cache_armed = True
            if active:
                self._request_active = True
                self._post_response_pending = False
                return
            if self._request_active:
                self._request_active = False
                # A completed real request is enough to arm Metal warming even
                # when prefix caching is disabled. Cache accounting, when
                # available, only selects the safer long-context cadence.
                self._clear_inhibited = False
                self._cache_armed = True
                self._post_response_pending = True
                self._last_activity_at = now

    def cancel_unstarted_request(self) -> None:
        """Roll back an exclusive admission that failed before scheduler entry."""

        with self._lock:
            self._request_active = False
            self._post_response_pending = False

    def disarm_cache(self) -> None:
        """Stop latent warming after an explicit hot-cache clear."""

        with self._lock:
            self._cache_armed = False
            self._cache_tokens = 0
            self._clear_inhibited = True
            self._post_response_pending = False

    def shutdown(self) -> None:
        """Make every future action a no-op before engine teardown."""

        with self._lock:
            self._closed = True
            self._request_active = False
            self._cache_armed = False
            self._cache_tokens = 0
            self._clear_inhibited = True
            self._post_response_pending = False

    def idle_action(self, *, cache_tokens: int | None = None) -> KeepwarmAction | None:
        now = float(self._clock())
        with self._lock:
            if cache_tokens is not None:
                self._cache_tokens = max(0, int(cache_tokens))
                if self._cache_tokens > 0 and not self._clear_inhibited:
                    self._cache_armed = True
            if (
                self._closed
                or not self.config.enabled
                or not self._cache_armed
                or self._request_active
                or now < self._slow_until
            ):
                return None
            idle = max(0.0, now - self._last_activity_at)
            if (
                self._post_response_pending
                and self.config.post_response_enabled
                and idle >= self.config.post_response_delay_seconds
            ):
                self._post_response_pending = False
                return KeepwarmAction(
                    kind="post_response",
                    matrix_size=self.config.post_response_matrix_size,
                    repeats=self.config.repeats,
                    idle_seconds=idle,
                    cache_tokens=self._cache_tokens,
                )
            interval = self.config.interval_seconds
            if (
                self.config.large_cache_tokens > 0
                and self._cache_tokens >= self.config.large_cache_tokens
            ):
                interval = self.config.large_cache_interval_seconds
            if (
                idle < self.config.idle_after_seconds
                or now - self._last_touch_at < interval
            ):
                return None
            return KeepwarmAction(
                kind="idle",
                matrix_size=self.config.matrix_size,
                repeats=self.config.repeats,
                idle_seconds=idle,
                cache_tokens=self._cache_tokens,
            )

    def record(
        self,
        action: KeepwarmAction,
        *,
        elapsed_seconds: float,
        ok: bool,
        dataplane_ping: bool = False,
        error: str | None = None,
    ) -> None:
        now = float(self._clock())
        elapsed_seconds = max(0.0, float(elapsed_seconds))
        event: dict[str, Any] = {
            "ok": bool(ok),
            "action": action.kind,
            "at_monotonic": now,
            "elapsed_ms": elapsed_seconds * 1000.0,
            "idle_seconds": max(0.0, action.idle_seconds),
            "matrix_size": action.matrix_size,
            "repeats": action.repeats,
            "cache_tokens": action.cache_tokens,
            "dataplane_ping": bool(dataplane_ping),
        }
        if error:
            event["error"] = str(error)[:500]
        with self._lock:
            self._last_touch_at = now
            self._count += 1
            if not ok:
                self._failures += 1
                self._slow_until = now + self.config.slow_backoff_seconds
                event["failure_backoff_seconds"] = self.config.slow_backoff_seconds
            if elapsed_seconds >= self.config.slow_threshold_seconds:
                self._slow_count += 1
                self._slow_until = now + self.config.slow_backoff_seconds
                event["slow_backoff_seconds"] = self.config.slow_backoff_seconds
            self._last_event = event

    def skip(self, reason: str) -> None:
        with self._lock:
            self._skips += 1
            self._last_event = {
                "ok": True,
                "action": "skip",
                "reason": str(reason)[:200],
                "at_monotonic": float(self._clock()),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "policy": asdict(self.config),
                "request_active": self._request_active,
                "cache_armed": self._cache_armed,
                "cache_tokens": self._cache_tokens,
                "clear_inhibited": self._clear_inhibited,
                "closed": self._closed,
                "count": self._count,
                "failures": self._failures,
                "skips": self._skips,
                "slow_count": self._slow_count,
                "last_event": dict(self._last_event) if self._last_event else None,
            }


def metal_warmup_touch(
    mx: Any,
    action: KeepwarmAction,
    *,
    stream: Any | None = None,
) -> float:
    """Submit one bounded fp16 matmul and wait for its Metal completion."""

    started = time.monotonic()

    def run() -> None:
        for _ in range(action.repeats):
            lhs = mx.ones((action.matrix_size, action.matrix_size), dtype=mx.float16)
            rhs = mx.ones((action.matrix_size, action.matrix_size), dtype=mx.float16)
            value = mx.sum(mx.matmul(lhs, rhs))
            mx.eval(value)

    if stream is None:
        run()
    else:
        with mx.stream(stream):
            run()
    return max(0.0, time.monotonic() - started)


def distributed_dataplane_ping(
    mx: Any,
    group: Any,
    *,
    rank: int,
    world_size: int,
) -> None:
    """Exercise every rank's real MLX distributed point-to-point data path.

    Rank zero visits workers in order.  A worker receives first and then sends
    its acknowledgement, while rank zero sends first and then receives.  This
    complementary order cannot deadlock and generalizes ThunderMLX's TP2 ping
    without assuming exactly two Macs.
    """

    if world_size <= 1:
        return
    template = mx.array([rank], dtype=mx.uint32)
    if rank == 0:
        for target in range(1, world_size):
            sent = mx.distributed.send(template, target, group=group)
            mx.eval(sent)
            received = mx.distributed.recv_like(template, target, group=group)
            mx.eval(received)
            value = int(received.item())
            if value != target:
                raise RuntimeError(
                    f"keepwarm data-plane ping expected rank {target}, got {value}"
                )
        return
    received = mx.distributed.recv_like(template, 0, group=group)
    mx.eval(received)
    if int(received.item()) != 0:
        raise RuntimeError("keepwarm data-plane ping received an invalid coordinator")
    sent = mx.distributed.send(template, 0, group=group)
    mx.eval(sent)
