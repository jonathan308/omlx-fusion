# SPDX-License-Identifier: Apache-2.0
"""Model-switch arbiter ("gateway") for single-process multi-model serving.

This module ports the arbitration semantics of ThunderMLX's
``model_gateway.py`` (port 8010) onto oMLX's architecture. ThunderMLX ran a
separate HTTP proxy because its backends were *separate processes* (the M3
cluster on :8080, oMLX on :8000) that could not be memory-resident at the
same time; the gateway routed by model id and made the mutually exclusive
backends safe to switch. oMLX serves every model from one process through
:class:`omlx.engine_pool.EnginePool`, so the equivalent arbitration point is
not a proxy hop but the *resident-model switch*: a request for a model that
is not loaded can evict the model another session is using. That decision
point is what this module arbitrates.

Ported mechanisms (audit items A1-A6):

- A1 cross-backend arbiter: requests route by model id (unchanged); the
  arbiter makes mutually exclusive residency explicit — a load that would
  evict a resident model goes through :meth:`ModelSwitchArbiter.before_model_load`.
- A2 grace windows + busy-refusal: a destructive switch is refused while the
  incumbent has in-flight work, and deferred (waited out, bounded) while the
  incumbent served traffic within the grace window. The wait-out refinement
  and the 30s default are the production-tuned values from ThunderMLX
  (``M3_GATEWAY_STOP_M3_GRACE_S=30``, prod env.local.snapshot:355). Its 900s
  counter-window (``M3_GATEWAY_OMLX_GRACE_S``) gated a *heavy* cross-process
  cluster boot; in-process switches are cheap, so both directions share the
  one ``OMLX_GATEWAY_SWITCH_GRACE_S`` knob — operators who want the stickier
  production policy can set it to 900.
- A3 sticky default model: requests with an empty ``model`` field follow the
  last explicitly routed model while it is still loaded, instead of swapping
  to the configured default out from under the session.
- A4 validate-before-destructive-switch lives in the pool itself:
  ``EnginePool.get_engine`` validates the target entry, its on-disk path, and
  any cached load failure *before* the eviction loop, so an unknown or broken
  id can never evict the incumbent (verified by tests).
- A5 passive-probe no-wakeup: health/status/model-list endpoints never load
  a model (verified by tests); the one probe-class endpoint that did —
  ``/v1/messages/count_tokens``, which agent CLIs poll — is answered with an
  estimate when the model is not loaded (opt-in via the arbiter).
- A6 wakeup-attribution logging: every request-driven load records what
  triggered it (path, client, user-agent) in the log and in a 64-event
  forensic ring surfaced at ``GET /v1/gateway/status``.

Deliberately not ported (no oMLX analogue): A8 proxy timeout split — oMLX
has no proxy hop, clients talk to the serving process directly; A9 SSE
hygiene — already native (``_with_sse_keepalive``, ``X-Accel-Buffering: no``
and ``Cache-Control: no-cache`` on every streaming response).

The arbiter never gates the distributed cluster path: entries sourced from
cluster staging or carrying a registry deployment bypass it entirely.

Everything is opt-in: with ``OMLX_GATEWAY_ENABLED`` unset the arbiter is
never constructed and request handling is byte-for-byte today's behavior.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from .exceptions import ModelBusyError

logger = logging.getLogger(__name__)

# Production constants ported from ThunderMLX model_gateway.py.
DEFAULT_SWITCH_GRACE_S = 30.0  # M3_GATEWAY_STOP_M3_GRACE_S (prod env.local:355)
DEFAULT_DEFER_POLL_S = 2.0  # wait-out loop cadence (model_gateway.py:2906)
DEFAULT_DEFER_BUFFER_S = 5.0  # wait-out deadline buffer (model_gateway.py:2905)
EVENT_RING_SIZE = 64  # forensic ring size (model_gateway.py:507)
DEFERRAL_LOG_THROTTLE_S = 60.0  # deferral event throttle (model_gateway.py:2938)

# Engine types that can serve an LLM request (sticky default candidates).
_LLM_ENGINE_TYPES = frozenset({"batched", "simple", "vlm"})


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class GatewayConfig:
    """Configuration for the model-switch arbiter.

    Attributes:
        enabled: Master switch (``OMLX_GATEWAY_ENABLED``, default off).
        sticky_empty_model: Route empty-``model`` requests to the last
            explicitly routed loaded model (``OMLX_GATEWAY_STICKY_EMPTY_MODEL``).
        switch_grace_seconds: Grace window protecting a recently served
            resident model from a destructive switch
            (``OMLX_GATEWAY_SWITCH_GRACE_S``, default 30 — the production
            fast-switch value).
        probe_no_wake: Answer token-count probes for unloaded models with an
            estimate instead of loading (``OMLX_GATEWAY_PROBE_NO_WAKE``).
        defer_poll_seconds: Wait-out loop poll cadence.
        defer_buffer_seconds: Extra wait-out budget past one grace window.
        deferral_log_throttle_seconds: Minimum interval between deferral
            events so bursty clients cannot flush the forensic ring.
    """

    enabled: bool = False
    sticky_empty_model: bool = True
    switch_grace_seconds: float = DEFAULT_SWITCH_GRACE_S
    probe_no_wake: bool = True
    defer_poll_seconds: float = DEFAULT_DEFER_POLL_S
    defer_buffer_seconds: float = DEFAULT_DEFER_BUFFER_S
    deferral_log_throttle_seconds: float = DEFERRAL_LOG_THROTTLE_S

    @classmethod
    def from_env(cls) -> GatewayConfig:
        """Build the config from ``OMLX_GATEWAY_*`` environment variables."""
        grace_raw = os.environ.get("OMLX_GATEWAY_SWITCH_GRACE_S", "").strip()
        try:
            grace = float(grace_raw) if grace_raw else DEFAULT_SWITCH_GRACE_S
        except ValueError:
            logger.warning(
                "Invalid OMLX_GATEWAY_SWITCH_GRACE_S=%r; using %.0fs",
                grace_raw,
                DEFAULT_SWITCH_GRACE_S,
            )
            grace = DEFAULT_SWITCH_GRACE_S
        return cls(
            enabled=_env_bool("OMLX_GATEWAY_ENABLED", False),
            sticky_empty_model=_env_bool("OMLX_GATEWAY_STICKY_EMPTY_MODEL", True),
            switch_grace_seconds=max(0.0, grace),
            probe_no_wake=_env_bool("OMLX_GATEWAY_PROBE_NO_WAKE", True),
        )


class ModelSwitchDeferredError(Exception):
    """A destructive model switch waited out its grace window without clearing.

    Raised when the resident model keeps serving traffic past the bounded
    wait-out (one grace window plus buffer). The caller should retry after
    ``retry_after`` seconds or unload the incumbent explicitly first.
    """

    def __init__(
        self,
        model_id: str,
        blockers: list[str],
        retry_after: float,
        grace_seconds: float,
    ):
        self.model_id = model_id
        self.blockers = list(blockers)
        self.retry_after = float(retry_after)
        super().__init__(
            f"Switch to model '{model_id}' deferred: "
            f"{', '.join(self.blockers)} served traffic within the "
            f"{grace_seconds:.0f}s grace window and kept receiving work. "
            f"Retry after ~{max(1, int(retry_after))}s, or unload the active "
            "model explicitly first."
        )


# Per-request attribution context (A6). Set by GatewayAttributionMiddleware so
# the arbiter can record *who* triggered an expensive load without threading
# the FastAPI request through every engine getter.
request_attribution: ContextVar[dict[str, str] | None] = ContextVar(
    "omlx_gateway_request_attribution",
    default=None,
)


class GatewayAttributionMiddleware:
    """Pure ASGI middleware capturing request attribution for wake logging.

    Follows the DebugRequestLoggingMiddleware pattern: raw ASGI instead of
    BaseHTTPMiddleware, which wraps StreamingResponse in an intermediate pipe
    layer and corrupts HTTP keep-alive connections.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        client_host, client_port = scope.get("client") or ("?", "?")
        user_agent = "?"
        for key, value in scope.get("headers", []):
            if key.decode("latin-1").lower() == "user-agent":
                user_agent = value.decode("latin-1")[:80]
                break
        token = request_attribution.set(
            {
                "path": str(scope.get("path", "?")),
                "client": f"{client_host}:{client_port}",
                "user_agent": user_agent,
            }
        )
        try:
            await self.app(scope, receive, send)
        finally:
            request_attribution.reset(token)


def estimate_input_tokens(payload_text: str) -> int:
    """Estimate prompt tokens as UTF-8 bytes ÷ 4 (no model required).

    Same heuristic as the ThunderMLX gateway's count_tokens, which never
    booted a backend to answer a metering probe (model_gateway.py:3221-3233).
    """
    if not payload_text:
        return 0
    return max(1, len(payload_text.encode("utf-8")) // 4)


class ModelSwitchArbiter:
    """Arbitrates request-driven model switches against resident sessions.

    The arbiter is consulted on the server request path only when the
    requested model is not loaded. Already-loaded models return before any
    arbiter state is touched, so the hot path is unaffected.
    """

    def __init__(self, config: GatewayConfig | None = None):
        self.config = config or GatewayConfig()
        self._events: deque[dict[str, Any]] = deque(maxlen=EVENT_RING_SIZE)
        self._last_routed_model: str | None = None
        self._last_deferral_event_ts = 0.0
        self._switch_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # A6: forensic event ring + wakeup attribution
    # ------------------------------------------------------------------

    def record_event(self, action: str, **fields: Any) -> None:
        """Append an event to the forensic ring (newest first)."""
        self._events.appendleft({"time": time.time(), "action": action, **fields})

    def _record_deferral_throttled(self, action: str, **fields: Any) -> None:
        """Record a deferral event at most once per throttle window.

        Bursty clients defer several times a minute; unthrottled repeats
        would flush real switch events out of the ring (model_gateway.py:2936).
        """
        now = time.time()
        if now - self._last_deferral_event_ts < self.config.deferral_log_throttle_seconds:
            return
        self._last_deferral_event_ts = now
        self.record_event(action, **fields)

    def _log_wake(self, model_id: str, *, destructive: bool) -> None:
        """Log who triggered a load and record it in the ring.

        Logged before the switch completes so deferred attempts are
        attributed too (model_gateway.py:510-528).
        """
        attribution = request_attribution.get() or {}
        path = attribution.get("path", "?")
        client = attribution.get("client", "?")
        user_agent = attribution.get("user_agent", "?")
        logger.info(
            "[model-wakeup-attribution] model=%s path=%s client=%s ua=%s "
            "destructive=%s",
            model_id,
            path,
            client,
            user_agent,
            destructive,
        )
        self.record_event(
            "model_wake",
            model=model_id,
            destructive=destructive,
            path=path,
            client=client,
            user_agent=user_agent,
        )

    # ------------------------------------------------------------------
    # A3: sticky default model
    # ------------------------------------------------------------------

    def note_explicit_model(self, model_id: str) -> None:
        """Remember the last explicitly requested model for sticky defaulting.

        Only explicit client-supplied ids may update stickiness; a defaulted
        side-call must never redefine the session's model
        (model_gateway.py:4202-4205).
        """
        model_id = str(model_id or "").strip()
        if model_id:
            self._last_routed_model = model_id

    def sticky_default_model(self, pool) -> str | None:
        """Model id to use for a request with an empty model field.

        Returns the last explicitly routed model while it is still loaded as
        an LLM-capable engine, so a model-less side-call (title/summary)
        follows the session instead of swapping to the configured default
        (model_gateway.py:430-446). None means "no opinion" and the caller
        falls back to the configured default model.
        """
        if not self.config.sticky_empty_model:
            return None
        last = self._last_routed_model
        if not last:
            return None
        entry = pool.get_entry(last)
        if entry is None or entry.engine is None or entry.is_helper:
            return None
        if entry.engine_type not in _LLM_ENGINE_TYPES:
            return None
        self.record_event("sticky_default_applied", model=last)
        return last

    # ------------------------------------------------------------------
    # A5: passive-probe no-wakeup
    # ------------------------------------------------------------------

    def count_tokens_no_wake(
        self,
        pool,
        model_id: str | None,
        payload_text: str,
    ) -> int | None:
        """Answer a token-count probe without loading an unloaded model.

        Returns the bytes÷4 estimate when the resolved model is not loaded,
        or None to let the caller take the normal (loading) path. Agent CLIs
        poll /v1/messages/count_tokens between turns; loading a multi-GB
        model just to count tokens evicts the model the user is actually
        on (model_gateway.py:4229-4247).
        """
        if not self.config.probe_no_wake or not model_id:
            return None
        entry = pool.get_entry(model_id)
        if entry is None or entry.engine is not None:
            # Unknown ids flow to the normal path for a proper 404; loaded
            # models get the exact tokenizer count.
            return None
        self.record_event("passive_probe_no_wakeup", model=model_id)
        logger.info(
            "[model-wakeup-attribution] probe no-wake: model=%s answered "
            "with estimate (not loaded)",
            model_id,
        )
        return estimate_input_tokens(payload_text)

    # ------------------------------------------------------------------
    # A1/A2: the switch gate
    # ------------------------------------------------------------------

    def _switch_blockers(
        self,
        snapshot: dict[str, dict[str, Any]],
        *,
        exclude: str,
        grace: float,
        now: float,
    ) -> tuple[list[str], list[tuple[str, float]]]:
        """Split resident models into busy blockers and in-grace blockers.

        Mirrors _m3_busy_reason (model_gateway.py:2876-2888): in-flight work
        refuses outright; recent traffic within the grace window defers.
        Pinned models are not blockers — the pool never evicts them, so a
        load that only fits by evicting pinned models fails admission with
        the pool's own error.
        """
        busy: list[str] = []
        in_grace: list[tuple[str, float]] = []
        for mid, info in snapshot.items():
            if mid == exclude or info["pinned"] or info["is_loading"]:
                continue
            if info["busy"]:
                busy.append(mid)
                continue
            remaining = grace - (now - float(info["last_access"]))
            if remaining > 0:
                in_grace.append((mid, remaining))
        return busy, in_grace

    async def before_model_load(
        self,
        pool,
        model_id: str,
        *,
        force_lm: bool = False,
    ) -> None:
        """Arbitrate a request-driven load before it can evict residents.

        No-op when the model is already loaded (hot path), when the load
        fits alongside the residents, or when the target belongs to the
        distributed cluster path. Otherwise a destructive switch:

        - refuses with ModelBusyError while a resident has in-flight work
          (the pool would refuse the eviction anyway; the arbiter makes the
          refusal semantic and records it);
        - waits out the grace window while a resident served traffic
          recently, bounded to one window plus buffer (the production
          wait-out refinement, model_gateway.py:2898-2915);
        - raises ModelSwitchDeferredError when sustained traffic outlasts
          the bounded wait.

        Every path that lets a load proceed records wake attribution first.
        """
        entry = pool.get_entry(model_id)
        if entry is None or entry.engine is not None or entry.is_loading:
            return
        if pool.entry_is_distributed(entry):
            # Cluster requests bypass model swapping entirely.
            return
        if not pool.eviction_pressure_for_load(model_id, force_lm=force_lm):
            self._log_wake(model_id, destructive=False)
            return

        grace = self.config.switch_grace_seconds
        async with self._switch_lock:
            # Re-validate under the switch lock: a concurrent request may
            # have loaded the model or refreshed traffic while we waited.
            entry = pool.get_entry(model_id)
            if entry is None or entry.engine is not None:
                return
            if pool.entry_is_distributed(entry):
                return
            if not pool.eviction_pressure_for_load(model_id, force_lm=force_lm):
                self._log_wake(model_id, destructive=False)
                return

            self._log_wake(model_id, destructive=True)
            deadline = time.time() + grace + self.config.defer_buffer_seconds
            waited = False
            while True:
                snapshot = pool.residency_snapshot()
                busy, in_grace = self._switch_blockers(
                    snapshot,
                    exclude=model_id,
                    grace=grace,
                    now=time.time(),
                )
                if busy:
                    self.record_event(
                        "model_switch_refused_busy",
                        model=model_id,
                        blockers=busy,
                    )
                    raise ModelBusyError(
                        busy[0],
                        f"be evicted for '{model_id}'",
                    )
                if not in_grace:
                    break
                now = time.time()
                if now >= deadline:
                    retry_after = min(remaining for _, remaining in in_grace)
                    self._record_deferral_throttled(
                        "model_switch_grace_timeout",
                        model=model_id,
                        blockers=[mid for mid, _ in in_grace],
                    )
                    raise ModelSwitchDeferredError(
                        model_id,
                        blockers=[mid for mid, _ in in_grace],
                        retry_after=retry_after,
                        grace_seconds=grace,
                    )
                self._record_deferral_throttled(
                    "model_switch_grace_wait",
                    model=model_id,
                    blockers=[mid for mid, _ in in_grace],
                )
                waited = True
                await asyncio.sleep(self.config.defer_poll_seconds)

            self.record_event(
                "model_switch",
                model=model_id,
                waited=waited,
                evicts=list(pool.residency_snapshot()),
            )

    # ------------------------------------------------------------------
    # Status surface
    # ------------------------------------------------------------------

    def get_status(self, pool=None) -> dict[str, Any]:
        """Arbiter state plus the forensic event ring for /v1/gateway/status."""
        status: dict[str, Any] = {
            "enabled": True,
            "config": {
                "sticky_empty_model": self.config.sticky_empty_model,
                "switch_grace_seconds": self.config.switch_grace_seconds,
                "probe_no_wake": self.config.probe_no_wake,
            },
            "last_routed_model": self._last_routed_model,
            "last_event": self._events[0] if self._events else None,
            "events": list(self._events),
        }
        if pool is not None:
            status["residency"] = pool.residency_snapshot()
        return status
