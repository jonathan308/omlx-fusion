# SPDX-License-Identifier: Apache-2.0
"""Capability-gated optimizations for the pinned MLX-LM pipeline worker."""

from __future__ import annotations

import gc
import importlib
import inspect
import logging
import math
import os
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ..idle_maintenance import metal_keepwarm_touch
from .performance import ExecutionSettings

logger = logging.getLogger(__name__)

# Floor for the prefill-loop buffer-pool clear, mirroring the single-node
# ``Scheduler._periodic_clear_threshold_bytes`` formula (memory_limit/3 with
# a 2 GiB floor). Kept as a separate copy rather than imported: the cluster
# worker must not pull in the single-node scheduler module, and both sides
# point at each other so the formula cannot silently drift apart.
_DEFAULT_PREFILL_CLEAR_THRESHOLD_BYTES = 2 * 1024**3


def prefill_clear_threshold_bytes(memory_limit_bytes: int) -> int:
    """Pool-bytes threshold above which the prefill loop may clear the pool.

    Same rule as the scheduler's decode-time gate: a clear that fires when
    the MLX buffer pool holds little produces IOGPUFamily refcount bursts for
    no benefit, while a clear that never fires lets a long prefill pin the
    pool. Keep the two formulas in sync with
    ``omlx/scheduler.py::_periodic_clear_threshold_bytes``.
    """

    memory_limit_bytes = int(memory_limit_bytes)
    if memory_limit_bytes > 0:
        return max(memory_limit_bytes // 3, _DEFAULT_PREFILL_CLEAR_THRESHOLD_BYTES)
    return _DEFAULT_PREFILL_CLEAR_THRESHOLD_BYTES


def _capability(
    *,
    enabled: bool,
    active: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "active": bool(active),
        "reason": reason,
    }


def _supports_coordinator_sampling(
    pipeline_model: Any,
    *,
    batchable: bool,
    world_size: int,
) -> tuple[bool, str]:
    if world_size < 2:
        return False, "requires more than one pipeline rank"
    if not batchable:
        return False, "model is not compatible with MLX-LM continuous batching"
    call = type(pipeline_model).__dict__.get("__call__")
    if not callable(call):
        return False, "pipeline model has no callable forward path"
    try:
        source = inspect.getsource(call)
    except (OSError, TypeError):
        return False, "pipeline forward source is unavailable for validation"
    required = (
        "pipeline_rank",
        "pipeline_size",
        "distributed.all_gather",
        "distributed.send",
    )
    if any(token not in source for token in required):
        return False, "pipeline forward does not match the validated output contract"
    if source.count("distributed.all_gather") != 1:
        return False, "pipeline forward has an ambiguous collective output path"
    if source.count("distributed.send") != 1:
        return False, "pipeline forward has an ambiguous send path"
    return True, "validated final hidden-state gather replaced by token all-sum"


def _supports_native_async_step(generation_batch: Any) -> bool:
    step = getattr(generation_batch, "_step", None)
    try:
        source = inspect.getsource(step)
    except (OSError, TypeError):
        return False
    return "async_eval" in source and "_next_tokens" in source


def _supports_rank_zero_logits(model: Any) -> tuple[bool, int, str]:
    """Validate that worker ranks may advance the model without an LM head."""

    if not getattr(model, "_omlx_supports_rank_zero_logits", False):
        return False, 0, "model adapter has no rank-zero logits contract"
    call = type(model).__dict__.get("__call__")
    if not callable(call):
        return False, 0, "model adapter has no direct callable forward path"
    try:
        signature = inspect.signature(call)
    except (TypeError, ValueError):
        return False, 0, "model adapter forward signature is unavailable"
    if "skip_logits" not in signature.parameters:
        return False, 0, "model adapter does not explicitly accept skip_logits"
    try:
        vocab_size = int(model._omlx_output_vocab_size)
    except (AttributeError, TypeError, ValueError):
        return False, 0, "model adapter does not declare its output vocabulary"
    if vocab_size < 1:
        return False, 0, "model adapter declared an invalid output vocabulary"
    return (
        True,
        vocab_size,
        "worker ranks skip the vocabulary projection and log-softmax",
    )


def _supports_pipeline_prompt(prompt_batch: Any) -> tuple[bool, str]:
    """Validate the exact MLX-LM prompt loop this module replaces.

    This deliberately checks behavior-bearing source tokens instead of merely
    checking that a method with the right name exists. A future MLX-LM release
    can change cache preparation/finalization or prompt ownership without
    silently running a stale monkeypatch.
    """

    prompt = getattr(prompt_batch, "prompt", None)
    base_prompt = getattr(prompt_batch, "_omlx_base_prompt", prompt)
    known_wrapper = getattr(prompt_batch, "_omlx_prompt_wrapper", prompt)
    if prompt not in {base_prompt, known_wrapper}:
        return False, "another prompt-processing patch owns this model's prefill"
    try:
        source = inspect.getsource(base_prompt)
    except (OSError, TypeError):
        return False, "pinned prompt-processing source is unavailable"
    required = (
        "_right_pad_prompts",
        "self.prefill_step_size",
        "self.model(",
        "c.prepare(",
        "c.finalize()",
        "mx.eval([c.state for c in self.prompt_cache])",
    )
    if any(token not in source for token in required):
        return False, "MLX-LM prompt loop does not match the validated contract"
    return True, "validated staggered chunk scheduler and queued inter-stage sends"


_LOCKSTEP_CANCEL_ENV = "OMLX_CLUSTER_LOCKSTEP_CANCEL"


def _lockstep_cancel_enabled() -> bool:
    """Kill switch for the whole lockstep cancel/sentinel surface.

    ``mlx.launch`` runs one argv under one environment on every host, so an
    env toggle cannot desync the group: either every rank installs the pins or
    none does.
    """

    return os.environ.get(_LOCKSTEP_CANCEL_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


class LockstepPrefillCancelError(Exception):
    """Raised inside the pinned prompt loop when the chunk collective fires.

    Every rank reads the same int32 all-sum result at the same chunk boundary,
    so every rank raises at the same point in the schedule. That is the
    step-boundary rule from the batch-cancel design: on a pipeline group,
    generation may only end at a boundary all ranks agreed on. The pinned
    ``BatchGenerator.next`` converts this into the stock removal path; nothing
    half-prefilled is allowed to reach decode or the prompt cache.
    """


class LockstepClusterShutdownError(Exception):
    """Raised on every rank when the shutdown sentinel crosses the share channel.

    The worker's generation-loop interposition turns this into the graceful
    rank exit from the stability wave (release Metal, then self-SIGTERM), so a
    coordinated stop releases wired memory instead of orphaning it.
    """


_LOCKSTEP_SHUTDOWN_SENTINEL = {"omlx_cluster_shutdown": True}


def _is_shutdown_sentinel(value: Any) -> bool:
    # The share channel otherwise carries None (idle), request tuples, and uid
    # lists — never a dict — so this marker cannot collide with a request.
    return isinstance(value, dict) and value.get("omlx_cluster_shutdown") is True


# Payload key for an idle-maintenance op (keepwarm touch, all-rank cache
# drop) riding the same idle request-share channel as the shutdown sentinel.
# Distinct key so neither sentinel parser can mistake the other payload.
_IDLE_OP_PAYLOAD_KEY = "omlx_cluster_idle_op"


def _idle_op_slow_seconds() -> float:
    """Loud-log threshold for a fanned-out idle op (ThunderMLX's 5s alarm).

    Read per call from the environment; the worker process env is uniform
    across ranks (one argv under mlx.launch), so this cannot drift
    rank-to-rank.
    """

    raw = os.environ.get("OMLX_BACKGROUND_OP_SLOW_SECONDS", "5.0").strip()
    try:
        return max(0.1, float(raw or "5.0"))
    except ValueError:
        return 5.0


def _is_idle_op_payload(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get(_IDLE_OP_PAYLOAD_KEY), dict)


def _drop_rank_prompt_cache(
    instance: Any,
    mx_module: Any,
    *,
    clear_memory: bool,
    reason: str,
) -> dict[str, Any]:
    """Drop this rank's whole resident prompt cache; optionally drain Metal.

    The wholesale drop is the rank-convergent action: every rank's cache ends
    empty at the same share-channel rendezvous, so the next request starts
    from identical (miss) state everywhere — a selective or rank-local drop
    would seed the per-rank cache divergence the admission ladder's
    collective-free invariant documents. Entry enumeration reuses the
    ladder's drift-tolerant walk; an mlx-lm internals drift raises on every
    rank alike and is contained by the caller's guard. ``clear_memory`` is
    the deep release: gc -> clear_cache -> metal.clear_cache, in the
    documented order, so just-collected buffers are actually unwired.
    """

    from .prefill_guard import _idle_prompt_cache_evictables

    prompt_cache = getattr(instance, "prompt_cache", None)
    dropped = 0
    for item in _idle_prompt_cache_evictables(prompt_cache):
        item["drop"]()
        dropped += 1
    if clear_memory:
        gc.collect()
        mx_module.clear_cache()
        metal = getattr(mx_module, "metal", None)
        metal_clear = getattr(metal, "clear_cache", None)
        if callable(metal_clear):
            metal_clear()
    logger.info(
        "cluster idle op drop_caches (%s): dropped %d prompt-cache entries, "
        "clear_memory=%s",
        reason,
        dropped,
        clear_memory,
    )
    return {"ok": True, "dropped_entries": dropped, "clear_memory": bool(clear_memory)}


def _execute_idle_op(instance: Any, op: dict[str, Any], *, mx_module: Any) -> dict[str, Any]:
    """Run one fanned-out idle op on this rank's generation thread.

    Called from the pinned share-object path at the idle rendezvous, so no
    request work is in flight on any rank: the op is collective-free local
    work (a bounded matmul, a rank-local cache drop) executed at the one
    point where it cannot interleave with model collectives. Never raises —
    a rank that fails an op logs and continues, keeping the share-channel
    protocol intact. A slow op logs loudly (the A3 slow-op alarm: background
    GPU work is meant to be sub-second).
    """

    started = time.time()
    name = str(op.get("op") or "")
    reason = str(op.get("reason") or name)[:128]
    try:
        if name == "keepwarm":
            event = metal_keepwarm_touch(
                mx_module,
                size=int(op.get("matrix_size") or 64),
                repeats=1,
                reason=reason,
            )
        elif name == "drop_caches":
            event = _drop_rank_prompt_cache(
                instance,
                mx_module,
                clear_memory=bool(op.get("clear_memory")),
                reason=reason,
            )
        else:
            return {"ok": False, "error": f"unknown idle op {name!r}"[:200]}
    except Exception as exc:
        logger.warning("cluster idle op %s failed on this rank: %s", name, exc)
        return {"ok": False, "error": str(exc)[:200]}
    elapsed = time.time() - started
    slow_seconds = _idle_op_slow_seconds()
    if elapsed >= slow_seconds:
        logger.warning(
            "cluster idle op %s took %.2fs on this rank (>= %.1fs slow-op "
            "threshold) — background GPU work is meant to be sub-second",
            name,
            elapsed,
            slow_seconds,
        )
    return event


class LockstepCancelController:
    """One worker process's view of cancel/shutdown requests.

    Only rank zero's localhost admin handler ever arms this latch; every other
    rank learns the decision through the collective that carries it — the
    token all-sum in decode, one int32 all-sum per prefill chunk, or the idle
    request-share channel for shutdown. Per-rank latch state is therefore
    allowed to diverge: lockstep is a property of the collectives, never of
    this object.

    The latch is epoch-based so a cancel consumed by one phase cannot leak
    into the next request: each consumer records the epoch it reacted to, and
    work that starts after the arm observes an already-consumed epoch.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._broadcast = threading.Event()
        self._armed_epoch = 0
        self._decode_fired_epoch = 0
        self._prefill_fired_epoch = 0
        self._shutdown_pending = False
        self._removal_uids: list[int] = []
        self._idle_op_pending: dict[str, Any] | None = None
        self._idle_op_broadcast = threading.Event()
        self.stop_token_ids: tuple[int, ...] = ()
        self.decode_cancel_active = False
        self.prefill_cancel_active = False
        self.share_channel_active = False

    def reset(
        self,
        *,
        stop_token_ids: Sequence[int] = (),
        decode_cancel_active: bool = False,
        prefill_cancel_active: bool = False,
        share_channel_active: bool = False,
    ) -> None:
        """Re-arm the latch for a fresh serving session (or a test)."""

        with self._lock:
            self._armed_epoch = 0
            self._decode_fired_epoch = 0
            self._prefill_fired_epoch = 0
            self._shutdown_pending = False
            self._removal_uids = []
            self._idle_op_pending = None
            self._idle_op_broadcast.clear()
            self.stop_token_ids = tuple(int(token) for token in stop_token_ids)
            self.decode_cancel_active = bool(decode_cancel_active)
            self.prefill_cancel_active = bool(prefill_cancel_active)
            self.share_channel_active = bool(share_channel_active)
            self._broadcast.clear()

    def deactivate(self) -> None:
        self.reset()

    def arm_cancel(self) -> int:
        """Begin cancelling everything currently in flight. Rank zero only."""

        with self._lock:
            self._armed_epoch += 1
            return self._armed_epoch

    def request_shutdown(self) -> None:
        """Arm the cancel drain and the shutdown sentinel together."""

        with self._lock:
            self._armed_epoch += 1
            self._shutdown_pending = True

    @property
    def shutdown_pending(self) -> bool:
        with self._lock:
            return self._shutdown_pending

    def decode_swap_token(self) -> int | None:
        """The stop id to swap into this step's token all-sum, once per arm.

        Called from the pinned generation step on rank zero only. Consuming
        the epoch here — and in ``consume_decode_latch`` when a prefill cancel
        drains the cluster first — is what keeps a cancel aimed at one moment
        from killing a later request's first decode step.
        """

        with self._lock:
            if not self.decode_cancel_active or not self.stop_token_ids:
                return None
            if self._armed_epoch <= self._decode_fired_epoch:
                return None
            self._decode_fired_epoch = self._armed_epoch
            return self.stop_token_ids[0]

    def prefill_cancel_contribution(self, rank: int) -> int:
        """This rank's int32 all-sum contribution for one prefill chunk."""

        with self._lock:
            if rank != 0 or not self.prefill_cancel_active:
                return 0
            return 1 if self._armed_epoch > self._prefill_fired_epoch else 0

    def note_prefill_cancel_fired(self, uids: Sequence[int]) -> None:
        """Record the uids a fired cancel removed, for the uid-share cleanup."""

        with self._lock:
            self._prefill_fired_epoch = max(
                self._prefill_fired_epoch,
                self._armed_epoch,
            )
            self._removal_uids.extend(int(uid) for uid in uids)

    def consume_decode_latch(self) -> None:
        with self._lock:
            self._decode_fired_epoch = max(
                self._decode_fired_epoch,
                self._armed_epoch,
            )

    def take_pending_removals(self) -> list[int]:
        with self._lock:
            uids = self._removal_uids
            self._removal_uids = []
            return uids

    def mark_sentinel_broadcast(self) -> None:
        self._broadcast.set()

    def wait_sentinel_broadcast(self, timeout_s: float) -> bool:
        return self._broadcast.wait(timeout=max(0.0, timeout_s))

    def request_idle_op(self, op: dict[str, Any]) -> None:
        """Queue one idle-maintenance op for the next idle share. Rank zero only.

        Ops coalesce under a fixed priority — shutdown > drop_caches >
        keepwarm: a queued keepwarm is replaced by any newer op, while a
        queued cache drop is only replaced by another cache drop. The drop is
        the rarer, more deliberate action and must not be lost to a keepwarm
        tick that happens to land while it waits for an idle rendezvous.
        """

        with self._lock:
            pending = self._idle_op_pending
            if (
                pending is not None
                and pending.get("op") == "drop_caches"
                and op.get("op") != "drop_caches"
            ):
                return
            self._idle_op_pending = dict(op)
            self._idle_op_broadcast.clear()

    def take_pending_idle_op(self) -> dict[str, Any] | None:
        """Consume the queued idle op (the pinned share path, rank zero)."""

        with self._lock:
            op = self._idle_op_pending
            self._idle_op_pending = None
            return op

    def note_idle_op_broadcast(self) -> None:
        self._idle_op_broadcast.set()

    def wait_idle_op_broadcast(self, timeout_s: float) -> bool:
        return self._idle_op_broadcast.wait(timeout=max(0.0, timeout_s))


_LOCKSTEP_CONTROLLER = LockstepCancelController()


def get_lockstep_controller() -> LockstepCancelController:
    """The process-wide latch shared by the admin handler and the pinned loops."""

    return _LOCKSTEP_CONTROLLER


def _lockstep_cancel_reason(
    *,
    lockstep_enabled: bool,
    decode_cancel_active: bool,
    prefill_cancel_active: bool,
    decode_cancel_reason: str,
    batch_removal_reason: str,
    share_channel_reason: str,
    sampling_active: bool,
    sampling_reason: str,
) -> str:
    """Explain precisely which cancel half is armed and which fell back."""

    if not lockstep_enabled:
        return f"{_LOCKSTEP_CANCEL_ENV} disables the lockstep cancel pins"
    if decode_cancel_active and prefill_cancel_active:
        return (
            "decode swaps a stop id in ahead of the token all-sum; prefill "
            "reads one async int32 all-sum per chunk boundary"
        )
    if decode_cancel_active:
        return (
            "decode swaps a stop id in ahead of the token all-sum; prefill "
            f"keeps the stock path ({batch_removal_reason}; "
            f"{share_channel_reason})"
        )
    if prefill_cancel_active:
        return (
            "prefill reads one async int32 all-sum per chunk boundary; "
            f"decode keeps the stock path ({decode_cancel_reason})"
        )
    if not sampling_active:
        return f"requires the validated rank-zero sampling path ({sampling_reason})"
    return f"{decode_cancel_reason}; {batch_removal_reason}"


def _supports_lockstep_decode_cancel(
    mx_module: Any,
    stop_token_ids: Sequence[int],
) -> tuple[bool, str]:
    if not callable(getattr(mx_module, "depends", None)):
        return False, "pinned MLX has no mx.depends dependency fence"
    if not callable(getattr(mx_module, "full", None)):
        return False, "pinned MLX has no mx.full constant constructor"
    if not stop_token_ids:
        return False, "worker tokenizer declares no stop token id to swap in"
    return True, "rank zero swaps the sampled ids for a stop id ahead of the token all-sum"


def _supports_batch_cancel_removal(batch_generator_cls: Any) -> tuple[bool, str]:
    """Validate the BatchGenerator contract the cancel cleanup rides on."""

    if batch_generator_cls is None:
        return False, "MLX-LM has no continuous-batching generator"
    next_method = getattr(batch_generator_cls, "next", None)
    remove_method = getattr(batch_generator_cls, "remove", None)
    if not callable(next_method) or not callable(remove_method):
        return False, "batch generator lacks the next/remove removal contract"
    try:
        next_source = inspect.getsource(next_method)
        remove_source = inspect.getsource(remove_method)
    except (OSError, TypeError):
        return False, "batch generator source is unavailable for validation"
    if "mx.stream" not in next_source or "_next(" not in next_source:
        return False, "batch generator next() does not match the validated contract"
    required = ("_unprocessed_sequences", "_prompt_batch", "_generation_batch")
    if any(token not in remove_source for token in required):
        return False, "batch generator remove() does not match the validated contract"
    return True, "cancelled sequences leave through the stock uid-removal broadcast"


def _supports_share_channel(response_generator_cls: Any) -> tuple[bool, str]:
    """Validate the idle request-share channel the shutdown sentinel rides."""

    share = getattr(response_generator_cls, "_share_object", None)
    if not callable(share):
        return False, "MLX-LM response generator has no request share channel"
    try:
        source = inspect.getsource(share)
    except (OSError, TypeError):
        return False, "request share channel source is unavailable for validation"
    required = ("pickle.dumps", "pickle.loads", "all_sum")
    if any(token not in source for token in required):
        return False, "request share channel does not match the validated contract"
    return True, "the idle request share channel can carry the shutdown sentinel"


@dataclass(frozen=True)
class PrefillSlot:
    """One rank's work in a fill/drain pipeline timeline."""

    iteration: int
    start: int | None
    end: int | None

    @property
    def is_real(self) -> bool:
        return self.start is not None


def pipeline_prefill_schedule(
    token_count: int,
    prefill_step_size: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[PrefillSlot, ...]:
    """Return the Exo-style staggered fill/steady/drain schedule for one rank.

    Dummy slots do not issue a collective. Pipeline ``recv``/``send`` calls
    provide the actual dependency between adjacent ranks; issuing a different
    collective in a dummy slot would reorder the distributed graph and can
    deadlock. The slots remain explicit so telemetry and tests can prove every
    rank has the same total timeline and the expected offset.
    """

    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    if prefill_step_size < 1:
        raise ValueError("prefill_step_size must be positive")
    if world_size < 2:
        raise ValueError("pipeline prefill requires at least two ranks")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be inside the pipeline world")

    chunks = int(math.ceil(token_count / prefill_step_size))
    slots: list[PrefillSlot] = []
    total = chunks + world_size - 1
    # MLX-LM's pipeline flows from the highest rank to rank zero: rank r
    # receives from r+1 and sends to r-1. Exo's native pipeline numbers the
    # source stage as rank zero, so its ``rank`` leading-dummy formula must be
    # mirrored here. Using it verbatim makes rank zero block in recv while the
    # highest rank is still in a dummy slot.
    leading = world_size - 1 - rank
    for iteration in range(total):
        chunk = iteration - leading
        if 0 <= chunk < chunks:
            start = chunk * prefill_step_size
            slots.append(
                PrefillSlot(
                    iteration=iteration,
                    start=start,
                    end=min(start + prefill_step_size, token_count),
                )
            )
        else:
            slots.append(PrefillSlot(iteration=iteration, start=None, end=None))
    return tuple(slots)


@contextmanager
def install_runtime_optimizations(
    model: Any,
    group: Any,
    execution: ExecutionSettings,
    *,
    batchable: bool,
    pipeline_parallel: bool = True,
    memory_limit_bytes: int = 0,
    stop_token_ids: Sequence[int] | None = None,
) -> Iterator[dict[str, dict[str, Any]]]:
    """Install opt-in token-only output while reporting every capability.

    ``memory_limit_bytes`` is this host's admission ceiling; it sizes the
    pressure gate on the staggered prefill loop's buffer-pool clear (see
    ``prefill_clear_threshold_bytes``). 0 means "unknown" and falls back to
    the absolute floor.
    """

    import mlx.core as mx

    mlx_generate = importlib.import_module("mlx_lm.generate")
    mlx_server = importlib.import_module("mlx_lm.server")

    pipeline_model = getattr(model, "model", None)
    world_size = int(group.size())
    sampling_supported, sampling_reason = _supports_coordinator_sampling(
        pipeline_model,
        batchable=batchable,
        world_size=world_size,
    )
    if not pipeline_parallel:
        sampling_supported = False
        sampling_reason = "pure tensor parallelism keeps MLX-LM's synchronized sampler"
    generation_batch_cls = getattr(mlx_generate, "GenerationBatch", None)
    prompt_batch_cls = getattr(mlx_generate, "PromptProcessingBatch", None)
    native_async = (
        _supports_native_async_step(generation_batch_cls)
        if generation_batch_cls
        else False
    )
    prompt_supported, prompt_reason = (
        _supports_pipeline_prompt(prompt_batch_cls)
        if prompt_batch_cls
        else (False, "MLX-LM has no prompt-processing batch")
    )
    sampling_active = execution.sampling_rank_only and sampling_supported
    (
        rank_zero_logits_supported,
        output_vocab_size,
        rank_zero_logits_reason,
    ) = _supports_rank_zero_logits(model)
    rank_zero_logits_active = sampling_active and rank_zero_logits_supported
    prefill_active = (
        execution.async_overlap
        and sampling_active
        and prompt_supported
        and pipeline_parallel
        and execution.prefill_step_size > 1
    )
    batching_enabled = execution.pipeline_microbatch_size > 1
    batching_active = batching_enabled and batchable
    lockstep_enabled = _lockstep_cancel_enabled()
    stop_ids = tuple(stop_token_ids or ())
    decode_cancel_ok, decode_cancel_reason = _supports_lockstep_decode_cancel(
        mx,
        stop_ids,
    )
    share_channel_ok, share_channel_reason = _supports_share_channel(
        getattr(mlx_server, "ResponseGenerator", None)
    )
    batch_removal_ok, batch_removal_reason = _supports_batch_cancel_removal(
        getattr(mlx_generate, "BatchGenerator", None)
    )
    # The decode swap lives in the pinned generation step, the prefill cancel
    # in the pinned prompt loop, and the shutdown sentinel in the request
    # share channel. Each falls back independently: a capability that cannot
    # be installed leaves exactly today's behavior behind.
    decode_cancel_active = (
        lockstep_enabled and sampling_active and decode_cancel_ok
    )
    share_channel_active = lockstep_enabled and share_channel_ok
    prefill_cancel_active = (
        lockstep_enabled
        and prefill_active
        and batch_removal_ok
        # The cancelled uids leave the batch through the uid-removal
        # broadcast, which the share-channel pin extends — without the channel
        # the server could not drop its per-request bookkeeping in lockstep.
        and share_channel_ok
    )
    capabilities = {
        "coalesced_batching": _capability(
            enabled=batching_enabled,
            active=batching_active,
            reason=(
                "MLX-LM continuous batching coalesces up to "
                f"{execution.pipeline_microbatch_size} requests per target batch"
                if batchable
                else (
                    "this model's KV cache cannot be merged, so MLX-LM serves "
                    "requests sequentially"
                )
            ),
        ),
        "sampling_rank_only": _capability(
            enabled=execution.sampling_rank_only,
            active=sampling_active,
            reason=(
                sampling_reason
                if execution.sampling_rank_only
                else "experimental optimization is disabled"
            ),
        ),
        "rank_zero_logits": _capability(
            enabled=execution.sampling_rank_only,
            active=rank_zero_logits_active,
            reason=rank_zero_logits_reason,
        ),
        "prefill_skip_logits": _capability(
            enabled=rank_zero_logits_supported,
            active=prefill_active and rank_zero_logits_active,
            reason=(
                "staggered prefill skips the discarded vocabulary projection "
                "on every chunk"
                if rank_zero_logits_supported
                else rank_zero_logits_reason
            ),
        ),
        "async_overlap": _capability(
            enabled=execution.async_overlap,
            active=execution.async_overlap and native_async,
            reason=(
                "pinned MLX-LM GenerationBatch dispatches the next token with "
                "mx.async_eval"
                if native_async
                else "pinned generation step has no validated async dispatch"
            ),
        ),
        "cache_affinity": _capability(
            enabled=execution.cache_affinity,
            active=execution.cache_affinity,
            reason=(
                "all requests for this model stay on one persistent deployment "
                "and its rank-local prompt caches"
                if execution.cache_affinity
                else "deployment cache affinity is disabled"
            ),
        ),
        "pipeline_prefill_overlap": _capability(
            enabled=execution.async_overlap and execution.sampling_rank_only,
            active=prefill_active,
            reason=(
                prompt_reason
                if prefill_active
                else (
                    prompt_reason
                    if sampling_active
                    else (
                        "requires the validated rank-zero sampling path; this model "
                        "keeps MLX-LM's synchronized prefill"
                    )
                )
            ),
        ),
        "lockstep_cancel": _capability(
            enabled=lockstep_enabled,
            active=decode_cancel_active or prefill_cancel_active,
            reason=_lockstep_cancel_reason(
                lockstep_enabled=lockstep_enabled,
                decode_cancel_active=decode_cancel_active,
                prefill_cancel_active=prefill_cancel_active,
                decode_cancel_reason=decode_cancel_reason,
                batch_removal_reason=batch_removal_reason,
                share_channel_reason=share_channel_reason,
                sampling_active=sampling_active,
                sampling_reason=sampling_reason,
            ),
        ),
        "coordinated_shutdown": _capability(
            enabled=lockstep_enabled,
            active=share_channel_active,
            reason=(
                f"{_LOCKSTEP_CANCEL_ENV} disables the shutdown sentinel"
                if not lockstep_enabled
                else share_channel_reason
            ),
        ),
    }
    controller = get_lockstep_controller()
    controller.reset(
        stop_token_ids=stop_ids,
        decode_cancel_active=decode_cancel_active,
        prefill_cancel_active=prefill_cancel_active,
        share_channel_active=share_channel_active,
    )

    original_share_object = None
    if share_channel_active:
        original_share_object = mlx_server.ResponseGenerator._share_object

        def lockstep_share_object(instance: Any, obj: Any) -> Any:
            """Carry cancel removals and the shutdown sentinel on one channel.

            MLX-LM's request share is the one collective every rank reaches
            the moment it goes idle: rank zero shares, the workers follow the
            same object protocol. That makes it the only channel that can end
            an *idle* cluster in lockstep — the token all-sum stops flowing as
            soon as quiesce drains the batch, and a per-token collective of our
            own would run concurrent with model collectives on the same QP/CQ
            (the historical wedge). The sentinel therefore rides this channel:
            it crosses between request rounds, never mid-step, so no rank ever
            leaves a model collective unpaired.
            """

            if int(group.rank()) == 0:
                if obj is None and controller.shutdown_pending:
                    # Swap the idle "no request" share for the sentinel. The
                    # workers branch on the shared size exactly as they would
                    # for a real request, so the size/data all-sums stay
                    # matched; only the payload differs.
                    original_share_object(instance, _LOCKSTEP_SHUTDOWN_SENTINEL)
                    controller.mark_sentinel_broadcast()
                    raise LockstepClusterShutdownError(
                        "rank zero broadcast the shutdown sentinel"
                    )
                if obj is None:
                    # Same swap discipline as the shutdown sentinel, for the
                    # idle-maintenance ops (keepwarm touch, all-rank cache
                    # drop). The op only crosses on an idle share, so it can
                    # never land mid-request: a request that arrives first
                    # makes this a request share instead, and the latch stays
                    # queued for the next idle poll. Rank zero runs the op
                    # locally AFTER the broadcast, at the same rendezvous the
                    # workers run theirs — cache state can never diverge.
                    idle_op = controller.take_pending_idle_op()
                    if idle_op is not None:
                        original_share_object(instance, {_IDLE_OP_PAYLOAD_KEY: idle_op})
                        _execute_idle_op(instance, idle_op, mx_module=mx)
                        controller.note_idle_op_broadcast()
                        return None
                if isinstance(obj, list):
                    # The server shares a list only for its per-round
                    # uids_to_remove broadcast. Folding the cancelled prefill
                    # uids into it keeps every rank on the same removal set
                    # and lets the server drop its per-request bookkeeping in
                    # the same pass, at the same loop position everywhere.
                    extra = controller.take_pending_removals()
                    if extra:
                        obj = obj + [uid for uid in extra if uid not in obj]
                return original_share_object(instance, obj)
            result = original_share_object(instance, obj)
            if _is_shutdown_sentinel(result):
                raise LockstepClusterShutdownError(
                    "received the shutdown sentinel from rank zero"
                )
            if _is_idle_op_payload(result):
                # Execute rank-zero's fanned-out idle op at the same idle
                # rendezvous, then keep polling as if the share were idle.
                _execute_idle_op(instance, result[_IDLE_OP_PAYLOAD_KEY], mx_module=mx)
                return None
            return result

        mlx_server.ResponseGenerator._share_object = lockstep_share_object

    try:
        if not sampling_active:
            yield capabilities
            return

        original_all_gather = mx.distributed.all_gather
        original_send = mx.distributed.send
        original_pipeline_call = type(pipeline_model).__call__
        original_generation_step = mlx_generate.GenerationBatch._step
        original_prompt = (
            mlx_generate.PromptProcessingBatch.prompt if prefill_active else None
        )
        original_batch_next = (
            mlx_generate.BatchGenerator.next if prefill_cancel_active else None
        )
        local_state = threading.local()

        def selective_all_gather(value: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(local_state, "skip_final_gather", False):
                return value
            return original_all_gather(value, *args, **kwargs)

        def local_pipeline_output(
            instance: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            previous = getattr(local_state, "skip_final_gather", False)
            local_state.skip_final_gather = True
            try:
                return original_pipeline_call(instance, *args, **kwargs)
            finally:
                local_state.skip_final_gather = previous

        def queued_pipeline_send(value: Any, *args: Any, **kwargs: Any) -> Any:
            """Materialize a stage output and defer only the transport operation."""

            if not getattr(local_state, "queue_prefill_sends", False):
                return original_send(value, *args, **kwargs)
            # Breaking the graph here is essential. Without it, the send remains
            # entangled with the entire layer graph and the downstream recv cannot
            # make forward progress while this rank starts its next chunk.
            mx.eval(value)
            pending = getattr(local_state, "pending_prefill_sends", None)
            if pending is None:
                pending = []
                local_state.pending_prefill_sends = pending
            pending.append((value, args, kwargs))
            return value

        def flush_prefill_sends() -> None:
            pending = getattr(local_state, "pending_prefill_sends", [])
            local_state.pending_prefill_sends = []
            for value, args, kwargs in pending:
                sent = original_send(value, *args, **kwargs)
                mx.async_eval(sent)

        prefill_clear_threshold = prefill_clear_threshold_bytes(memory_limit_bytes)

        def clear_prefill_pool_if_pressured() -> None:
            """Pressure-gate the prefill loop's Metal buffer-pool clear.

            The unconditional per-chunk ``mx.clear_cache()`` this replaces (both
            upstream mlx-vlm and the first version of this override) returned the
            whole pool to the OS after EVERY chunk: ThunderMLX measured its
            pipeline rank sawtoothing 3 GiB <-> 89 GiB wired during every long
            prefill, unwiring and rewiring the working set per chunk. The clear
            survives only as the pressure safety valve: once the allocator cache
            itself exceeds the admission-derived threshold, handing it back is
            worth the rewire. Below the threshold the pool stays resident — the
            IOGPU residency set is already bounded by the wired ceiling the rank
            admitted against, and a sawtooth-free prefill keeps its buffers.
            """

            if mx.get_cache_memory() > prefill_clear_threshold:
                mx.clear_cache()

        def staggered_pipeline_prompt(instance: Any, tokens: list[list[int]]) -> None:
            """Pinned PromptProcessingBatch.prompt with pipeline fill/drain."""

            if len(instance.uids) != len(tokens):
                raise ValueError("The batch length doesn't match the number of inputs")
            if not tokens:
                return
            before_prompt = getattr(instance, "_omlx_before_prompt", None)
            if callable(before_prompt):
                before_prompt()

            for stored, incoming in zip(instance.tokens, tokens):
                stored += incoming

            lengths = [len(prompt) for prompt in tokens]
            max_length = max(lengths)
            padding = [max_length - length for length in lengths]
            max_padding = max(padding)
            if max_padding > 0:
                tokens_array = mlx_generate._right_pad_prompts(
                    tokens,
                    max_length=max_length,
                )
                for cache in instance.prompt_cache:
                    cache.prepare(lengths=lengths, right_padding=padding)
            else:
                tokens_array = mx.array(tokens)

            # ``prefill_step_size`` is already the memory-admitted chunk size used
            # by MLX-LM and the rank prefill guard. Dividing it by the world size
            # here made a two-rank 4096-token deployment execute 2048-token chunks,
            # doubling every cache-state barrier and send boundary on long prompts.
            #
            # Staggering still overlaps adjacent stages: it is the rank offset in
            # ``pipeline_prefill_schedule`` that creates fill/steady/drain, not a
            # private reduction of the guarded compute chunk.
            step = max(1, int(instance.prefill_step_size))
            rank = int(group.rank())
            schedule = pipeline_prefill_schedule(
                int(tokens_array.shape[1]),
                step,
                rank=rank,
                world_size=world_size,
            )
            pending_cancel = None
            cancelled = False
            local_state.pending_prefill_sends = []
            try:
                for slot in schedule:
                    if not slot.is_real:
                        continue
                    if pending_cancel is not None:
                        # Read the PREVIOUS chunk's cancel all-sum. It was
                        # launched one real slot earlier and its ring
                        # round-trip overlapped that chunk's compute, so this
                        # read does not stall the pipeline. Every rank reads
                        # the same collective result and breaks at the same
                        # chunk boundary — cancel latency is one chunk, not
                        # one prompt. A blocking read here instead was
                        # measured at 20-25% of prefill wall time, which is
                        # why the launch is async and the read deferred.
                        hit = int(pending_cancel.item()) > 0
                        pending_cancel = None
                        if hit:
                            cancelled = True
                            break
                    if prefill_cancel_active:
                        # Launch THIS chunk's cancel all-sum without blocking,
                        # then compute the chunk. One int32 per chunk boundary
                        # — never per token — and the launch predicate is a
                        # pure function of the schedule position, so the
                        # all-sum count and order match on every rank (a
                        # mismatch would desync the group). Only rank zero's
                        # contribution can be nonzero; the collective is what
                        # carries its cancel decision to the workers.
                        pending_cancel = mx.distributed.all_sum(
                            mx.array(
                                controller.prefill_cancel_contribution(rank),
                                dtype=mx.int32,
                            ),
                            group=group,
                        )
                        mx.async_eval(pending_cancel)
                    local_state.queue_prefill_sends = True
                    try:
                        # The prompt loop never consumes the returned logits
                        # (the first sampled token comes from the generation
                        # step that follows), so a model declaring the
                        # rank-zero logits contract skips the vocabulary
                        # projection on every chunk, not just on worker-rank
                        # decode steps.
                        if rank_zero_logits_active:
                            instance.model(
                                tokens_array[:, slot.start : slot.end],
                                cache=instance.prompt_cache,
                                skip_logits=True,
                            )
                        else:
                            instance.model(
                                tokens_array[:, slot.start : slot.end],
                                cache=instance.prompt_cache,
                            )
                    finally:
                        local_state.queue_prefill_sends = False
                    flush_prefill_sends()
                    mx.eval([cache.state for cache in instance.prompt_cache])
                    clear_prefill_pool_if_pressured()
            finally:
                local_state.queue_prefill_sends = False
                # A cancelled/failed prefill must never leak an old activation into
                # the next request.
                local_state.pending_prefill_sends = []

            if not cancelled and pending_cancel is not None:
                # Catch a cancel that landed on the final launched chunk
                # before the sequences fall through to decode, where the EOS
                # swap takes over.
                cancelled = int(pending_cancel.item()) > 0
            if cancelled:
                # The collective result is identical on every rank, so every
                # rank raises at the same chunk boundary. The pinned
                # BatchGenerator.next turns this into the stock uid-removal
                # broadcast; the half-filled caches are dropped there, never
                # finished into the prompt cache.
                raise LockstepPrefillCancelError(
                    "prefill cancelled at a chunk boundary after "
                    f"{sum(slot.is_real for slot in schedule)} scheduled chunk(s)"
                )

            if max_padding > 0:
                for cache in instance.prompt_cache:
                    cache.finalize()
                mx.eval([cache.state for cache in instance.prompt_cache])
                clear_prefill_pool_if_pressured()

        def coordinator_generation_step(instance: Any) -> Any:
            """Pinned GenerationBatch._step with one token collective per batch."""

            instance._current_tokens = instance._next_tokens
            instance._current_logprobs = instance._next_logprobs
            inputs = instance._current_tokens
            coordinator = int(group.rank()) == 0

            if coordinator or not rank_zero_logits_active:
                logits = instance.model(inputs[:, None], cache=instance.prompt_cache)
                logits = logits[:, -1, :]
            else:
                instance.model(
                    inputs[:, None],
                    cache=instance.prompt_cache,
                    skip_logits=True,
                )
                # The token all-sum must be issued after this rank's stage send.
                # MiniMax anchors that lazy send in its last KV cache entry, so
                # materializing the cache state both advances the cache and fixes
                # the distributed operation order without paying for an LM head.
                cache_states = [cache.state for cache in instance.prompt_cache]
                if not cache_states:
                    raise RuntimeError(
                        "rank-zero logits requires a cache state to anchor the "
                        "worker-stage send"
                    )
                mx.eval(cache_states)
                logits = None

            token_context = []
            if any(instance.logits_processors):
                token_context = [
                    token_buffer.update_and_fetch(inputs[index : index + 1])
                    for index, token_buffer in enumerate(instance._token_context)
                ]
                if logits is not None:
                    processed_logits = []
                    for index in range(len(instance.uids)):
                        sample_logits = logits[index : index + 1]
                        for processor in instance.logits_processors[index]:
                            sample_logits = processor(
                                token_context[index],
                                sample_logits,
                            )
                        processed_logits.append(sample_logits)
                    logits = mx.concatenate(processed_logits, axis=0)

            if logits is not None:
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            else:
                # Worker ResponseGenerator instances still index this vector while
                # draining their private response queues. Its values never leave
                # the worker, but it must retain the real vocabulary width.
                placeholder = mx.zeros((output_vocab_size,), dtype=mx.float32)
                logprobs = mx.stack([placeholder] * len(instance.uids))

            if coordinator:
                if any(instance.samplers):
                    all_samples = []
                    for index in range(len(instance.uids)):
                        sampler = instance.samplers[index] or instance.fallback_sampler
                        all_samples.append(sampler(logprobs[index : index + 1]))
                    sampled = mx.concatenate(all_samples, axis=0)
                else:
                    sampled = instance.fallback_sampler(logprobs)
            else:
                sampled = mx.zeros((len(instance.uids),), dtype=mx.uint32)

            if decode_cancel_active and coordinator:
                forced_token = controller.decode_swap_token()
                if forced_token is not None:
                    # Lockstep in-flight cancel, from the batch-cancel design:
                    # rank zero swaps the sampled ids for a stop id BEFORE the
                    # token all-sum, so every rank consumes the identical
                    # stream and the server's stop-sequence matching ends every
                    # sequence at the identical step boundary. Generation may
                    # only end at a step boundary agreed by all ranks — a
                    # mid-stream break wedged the group every time it was
                    # tried — and this swap adds no collective of its own, so
                    # nothing runs concurrent with model collectives.
                    #
                    # mx.depends is LOAD-BEARING: it keeps this step's all-sum
                    # downstream of the sampled-token graph, so issuing the
                    # collective still forces this rank's forward (and posts
                    # the peer's pipeline recv) in the same order as an
                    # uncancelled step. A bare mx.full constant would let the
                    # all-sum complete without the forward, starving the
                    # peer's pipeline send.
                    eos = mx.full(sampled.shape, forced_token, dtype=sampled.dtype)
                    sampled = mx.depends(eos, sampled)

            # Rank zero contributes the selected IDs; all other ranks contribute
            # zeros. Every rank therefore advances the same local KV state without
            # gathering a hidden-state tensor.
            sampled = mx.distributed.all_sum(sampled, group=group)
            instance._next_tokens = sampled
            instance._next_logprobs = list(logprobs)
            mx.async_eval(
                instance._next_tokens,
                instance._next_logprobs,
                token_context,
            )

            mx.eval(inputs, instance._current_logprobs)
            input_values = inputs.tolist()
            for sequence_tokens, token in zip(instance.tokens, input_values):
                sequence_tokens.append(token)
            return input_values, instance._current_logprobs

        mx.distributed.all_gather = selective_all_gather
        mx.distributed.send = queued_pipeline_send
        type(pipeline_model).__call__ = local_pipeline_output
        mlx_generate.GenerationBatch._step = coordinator_generation_step
        if prefill_active:
            mlx_generate.PromptProcessingBatch.prompt = staggered_pipeline_prompt
        if original_batch_next is not None:

            def lockstep_batch_next(instance: Any) -> Any:
                """Turn a lockstep prefill cancel into the stock removal path.

                Every rank's pinned prompt raised at the same chunk boundary,
                so every rank computes the same removal set here. The uids go
                to the controller; rank zero folds them into the server's next
                ``uids_to_remove`` broadcast, which is what drops the
                per-request bookkeeping on every rank at the same loop
                position. Decoding sequences are handled by the EOS swap in
                the pinned step; the ones that already finished inside the
                cancelled round are swept here so their server-side entries
                cannot leak.
                """

                decoding_before = tuple(instance._generation_batch.uids)
                try:
                    return original_batch_next(instance)
                except LockstepPrefillCancelError:
                    still_decoding = set(instance._generation_batch.uids)
                    uids = [
                        uid
                        for uid in decoding_before
                        if uid not in still_decoding
                    ]
                    uids.extend(instance._prompt_batch.uids)
                    uids.extend(
                        sequence[0] for sequence in instance._unprocessed_sequences
                    )
                    controller.note_prefill_cancel_fired(uids)
                    if not still_decoding:
                        # Nothing survived into decode, so the armed decode
                        # latch has no in-flight sequence left to stop.
                        # Consuming it here keeps the next request's first
                        # decode step alive.
                        controller.consume_decode_latch()
                    return [], []

            mlx_generate.BatchGenerator.next = lockstep_batch_next
        try:
            yield capabilities
        finally:
            if original_batch_next is not None:
                mlx_generate.BatchGenerator.next = original_batch_next
            if original_prompt is not None:
                mlx_generate.PromptProcessingBatch.prompt = original_prompt
            mlx_generate.GenerationBatch._step = original_generation_step
            type(pipeline_model).__call__ = original_pipeline_call
            mx.distributed.send = original_send
            mx.distributed.all_gather = original_all_gather
    finally:
        if original_share_object is not None:
            mlx_server.ResponseGenerator._share_object = original_share_object
        controller.deactivate()
