# SPDX-License-Identifier: Apache-2.0
"""Opt-in idle maintenance: keepwarm touches and the graduated TTL ladder.

Ported from ThunderMLX's keepwarm / idle / TTL semantics (the agent-26 port
audit). Three mechanisms live here, all off by default so an unconfigured
server behaves exactly as before:

**Keepwarm** (``OMLX_KEEPWARM_ENABLED=1``): while a loaded engine is idle, a
tiny bounded Metal matmul runs every ``OMLX_KEEPWARM_INTERVAL_SECONDS`` so GPU
clocks and the prefill/decode path stay ramped between turns (ThunderMLX
``_touch_prompt_cache_keepwarm`` / ``_metal_warmup_touch``). Models with a
large declared context (``OMLX_KEEPWARM_LARGE_CACHE_TOKENS``, prod: 8k) are
throttled to the slower ``OMLX_KEEPWARM_LARGE_INTERVAL_SECONDS`` cadence
(prod: 30s vs 10s), and a touch that ran slow backs off for
``OMLX_KEEPWARM_SLOW_BACKOFF_SECONDS``.

**Graduated TTL**: ``OMLX_CACHE_TTL_SECONDS`` drops an idle engine's cache
tier (prefix cache + MLX buffer pool) while keeping weights resident, and
``OMLX_IDLE_RELEASE_SECONDS`` runs the deep release (caches + Metal heap)
after a longer idle horizon. Both sit *below* the existing engine TTL, which
remains the final rung that unloads the whole engine. ThunderMLX
``_janitor_tick`` is the reference: cache drop -> deep release -> unload.

**Discipline** (ThunderMLX A3/B4): background GPU ops are bounded (the touch
matrix is clamped), idle-gated by the caller, logged loudly when slow
(``OMLX_BACKGROUND_OP_SLOW_SECONDS``), and never stamp the engine's
``last_access`` idle clock — a background touch that refreshed activity would
keep the TTL rungs from ever firing.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_KEEPWARM_ENABLED_ENV = "OMLX_KEEPWARM_ENABLED"
_KEEPWARM_INTERVAL_ENV = "OMLX_KEEPWARM_INTERVAL_SECONDS"
_KEEPWARM_IDLE_AFTER_ENV = "OMLX_KEEPWARM_IDLE_AFTER_SECONDS"
_KEEPWARM_MATRIX_SIZE_ENV = "OMLX_KEEPWARM_MATRIX_SIZE"
_KEEPWARM_LARGE_TOKENS_ENV = "OMLX_KEEPWARM_LARGE_CACHE_TOKENS"
_KEEPWARM_LARGE_INTERVAL_ENV = "OMLX_KEEPWARM_LARGE_INTERVAL_SECONDS"
_KEEPWARM_SLOW_BACKOFF_ENV = "OMLX_KEEPWARM_SLOW_BACKOFF_SECONDS"
_BACKGROUND_OP_SLOW_ENV = "OMLX_BACKGROUND_OP_SLOW_SECONDS"
_CACHE_TTL_ENV = "OMLX_CACHE_TTL_SECONDS"
_IDLE_RELEASE_ENV = "OMLX_IDLE_RELEASE_SECONDS"
_EVICTION_GRACE_ENV = "OMLX_EVICTION_GRACE_SECONDS"

# Production cadence from the ThunderMLX prod snapshot (env.local
# 20260720-2302): 10s tick, 10s idle-after, size-64 matmul, >=8k-token
# contexts on a 30s cadence. Prod ran the slow backoff disabled (0); the
# default here keeps ThunderMLX's documented 60s code default — the whole
# feature is opt-in, and a first-time operator gets the safer behavior.
_DEFAULT_KEEPWARM_INTERVAL_SECONDS = 10.0
_DEFAULT_KEEPWARM_IDLE_AFTER_SECONDS = 10.0
_DEFAULT_KEEPWARM_MATRIX_SIZE = 64
_DEFAULT_KEEPWARM_LARGE_CACHE_TOKENS = 8192
_DEFAULT_KEEPWARM_LARGE_INTERVAL_SECONDS = 30.0
_DEFAULT_KEEPWARM_SLOW_BACKOFF_SECONDS = 60.0
_DEFAULT_BACKGROUND_OP_SLOW_SECONDS = 5.0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using %s", name, raw, default)
        return default


@dataclass(frozen=True)
class IdleMaintenanceSettings:
    """One snapshot of the idle-maintenance env configuration.

    Parsed fresh from the environment on every ``from_env()`` call so the
    killswitches take effect without a restart and tests can monkeypatch.
    Every mechanism defaults to off/preserve-current-behavior.
    """

    keepwarm_enabled: bool = False
    keepwarm_interval_seconds: float = _DEFAULT_KEEPWARM_INTERVAL_SECONDS
    keepwarm_idle_after_seconds: float = _DEFAULT_KEEPWARM_IDLE_AFTER_SECONDS
    keepwarm_matrix_size: int = _DEFAULT_KEEPWARM_MATRIX_SIZE
    keepwarm_large_cache_tokens: int = _DEFAULT_KEEPWARM_LARGE_CACHE_TOKENS
    keepwarm_large_interval_seconds: float = _DEFAULT_KEEPWARM_LARGE_INTERVAL_SECONDS
    keepwarm_slow_backoff_seconds: float = _DEFAULT_KEEPWARM_SLOW_BACKOFF_SECONDS
    background_op_slow_seconds: float = _DEFAULT_BACKGROUND_OP_SLOW_SECONDS
    cache_ttl_seconds: float = 0.0
    idle_release_seconds: float = 0.0

    @classmethod
    def from_env(cls) -> IdleMaintenanceSettings:
        return cls(
            keepwarm_enabled=_env_bool(_KEEPWARM_ENABLED_ENV, False),
            keepwarm_interval_seconds=max(
                0.25, _env_float(_KEEPWARM_INTERVAL_ENV, _DEFAULT_KEEPWARM_INTERVAL_SECONDS)
            ),
            keepwarm_idle_after_seconds=max(
                0.0,
                _env_float(_KEEPWARM_IDLE_AFTER_ENV, _DEFAULT_KEEPWARM_IDLE_AFTER_SECONDS),
            ),
            keepwarm_matrix_size=max(
                1, _env_int(_KEEPWARM_MATRIX_SIZE_ENV, _DEFAULT_KEEPWARM_MATRIX_SIZE)
            ),
            keepwarm_large_cache_tokens=max(
                0,
                _env_int(_KEEPWARM_LARGE_TOKENS_ENV, _DEFAULT_KEEPWARM_LARGE_CACHE_TOKENS),
            ),
            keepwarm_large_interval_seconds=max(
                0.0,
                _env_float(
                    _KEEPWARM_LARGE_INTERVAL_ENV,
                    _DEFAULT_KEEPWARM_LARGE_INTERVAL_SECONDS,
                ),
            ),
            keepwarm_slow_backoff_seconds=max(
                0.0,
                _env_float(
                    _KEEPWARM_SLOW_BACKOFF_ENV, _DEFAULT_KEEPWARM_SLOW_BACKOFF_SECONDS
                ),
            ),
            background_op_slow_seconds=max(
                0.1,
                _env_float(_BACKGROUND_OP_SLOW_ENV, _DEFAULT_BACKGROUND_OP_SLOW_SECONDS),
            ),
            cache_ttl_seconds=max(0.0, _env_float(_CACHE_TTL_ENV, 0.0)),
            idle_release_seconds=max(0.0, _env_float(_IDLE_RELEASE_ENV, 0.0)),
        )

    @property
    def any_enabled(self) -> bool:
        """Fast path for callers: with all defaults, maintenance is a no-op."""

        return bool(
            self.keepwarm_enabled
            or self.cache_ttl_seconds > 0
            or self.idle_release_seconds > 0
        )


def eviction_grace_seconds() -> float:
    """Recency grace before evicting an idle model (ThunderMLX C4).

    0 (the default) preserves today's pure-LRU victim choice. When >0, the
    engine pool prefers victims whose last access is older than the grace
    window; a model inside the window is only evicted when every idle model
    is inside it (grace is a preference, never a veto, so loads cannot fail
    because every resident model was used recently).
    """

    return max(0.0, _env_float(_EVICTION_GRACE_ENV, 0.0))


def metal_keepwarm_touch(
    mx_module: Any,
    *,
    size: int = _DEFAULT_KEEPWARM_MATRIX_SIZE,
    repeats: int = 1,
    reason: str = "keepwarm",
) -> dict[str, Any]:
    """Run one bounded GPU touch that keeps the Metal path warm.

    Port of ThunderMLX ``_metal_warmup_touch``: an independent matmul on the
    default device — deliberately NOT the KV/prompt tensors, which are tied
    to the generation thread's MLX stream and raise "no Stream(...) in
    current thread" when touched from another thread. Never raises; the
    returned event mirrors ThunderMLX's ``_metal_warmup_last_event`` shape.
    """

    started = time.time()
    size = max(16, min(1024, int(size or _DEFAULT_KEEPWARM_MATRIX_SIZE)))
    repeats = max(1, min(16, int(repeats or 1)))
    try:
        with mx_module.stream(mx_module.default_device()):
            acc = None
            for i in range(repeats):
                a = mx_module.ones((size, size), dtype=mx_module.float16) * (i + 1)
                b = mx_module.ones((size, size), dtype=mx_module.float16)
                value = mx_module.sum(a @ b)
                acc = value if acc is None else acc + value
            mx_module.eval(acc)
        event: dict[str, Any] = {
            "ok": True,
            "action": "metal_keepwarm",
            "reason": str(reason or "keepwarm")[:128],
            "at": round(time.time(), 3),
            "matrix_size": size,
            "repeats": repeats,
            "elapsed_ms": round((time.time() - started) * 1000, 3),
        }
    except Exception as e:
        event = {
            "ok": False,
            "action": "metal_keepwarm_error",
            "reason": str(reason or "keepwarm")[:128],
            "at": round(time.time(), 3),
            "matrix_size": size,
            "repeats": repeats,
            "elapsed_ms": round((time.time() - started) * 1000, 3),
            "error": str(e),
        }
        logger.debug("metal keepwarm touch failed: %s", e)
    return event


class KeepwarmTracker:
    """Per-engine keepwarm bookkeeping (ThunderMLX ``_prompt_cache_holder``).

    Tracks the throttle state the tick gates on: when the last touch ran,
    how long it took (the slow-op backoff), and how many have run. Never
    carries request activity — the idle clock it is checked against lives on
    the engine-pool entry and is only stamped by real requests (B4).
    """

    def __init__(self) -> None:
        self.touches = 0
        self.last_touch_at = 0.0
        self.last_event: dict[str, Any] | None = None

    def should_touch(
        self,
        now: float,
        *,
        idle_seconds: float,
        large_context: bool,
        settings: IdleMaintenanceSettings,
    ) -> bool:
        """Gate one keepwarm tick (ThunderMLX ``_touch_prompt_cache_keepwarm``).

        Skips when the engine has not been idle long enough, when the base
        interval has not elapsed, when a large-context model is inside its
        slower cadence window, or when the previous touch was slow enough to
        trigger the backoff window.
        """

        if idle_seconds < settings.keepwarm_idle_after_seconds:
            return False
        if (
            self.last_touch_at > 0
            and now - self.last_touch_at < settings.keepwarm_interval_seconds
        ):
            return False
        if (
            large_context
            and settings.keepwarm_large_cache_tokens > 0
            and settings.keepwarm_large_interval_seconds > 0
            and self.last_touch_at > 0
            and now - self.last_touch_at < settings.keepwarm_large_interval_seconds
        ):
            return False
        last_elapsed_ms = float((self.last_event or {}).get("elapsed_ms") or 0.0)
        return not (
            settings.keepwarm_slow_backoff_seconds > 0
            and self.last_touch_at > 0
            and last_elapsed_ms >= 1000.0
            and now - self.last_touch_at < settings.keepwarm_slow_backoff_seconds
        )

    def note_touch(self, event: dict[str, Any]) -> None:
        """Record a completed touch; failed touches count but keep ``ok``."""

        self.touches += 1
        self.last_touch_at = float(event.get("at") or time.time())
        self.last_event = dict(event)


def graduated_idle_action(
    now: float,
    last_access: float,
    *,
    cache_ttl_seconds: float,
    idle_release_seconds: float,
    last_cache_drop_at: float,
    last_deep_release_at: float,
) -> str | None:
    """Pick the graduated TTL rung an idle engine is due for, if any.

    The ladder below the full engine unload (ThunderMLX ``_janitor_tick``):
    ``"drop_caches"`` at the cache TTL horizon (drop the cache tier, keep
    weights and TTFT warm) and ``"deep_release"`` at the longer idle-release
    horizon (additionally drain the MLX/Metal heaps; the model stays
    servable). The deep release subsumes the shallow drop, and each rung
    fires at most once per idle period — the ``last_*_at < last_access``
    comparisons are ThunderMLX's "already released since the last real
    activity" guard, and background work never moves ``last_access``.
    """

    if last_access <= 0:
        return None
    idle = now - last_access
    if (
        idle_release_seconds > 0
        and idle >= idle_release_seconds
        and last_deep_release_at < last_access
    ):
        return "deep_release"
    if (
        cache_ttl_seconds > 0
        and idle >= cache_ttl_seconds
        and last_cache_drop_at < last_access
    ):
        return "drop_caches"
    return None
