# SPDX-License-Identifier: Apache-2.0
"""Refuse a prompt that would OOM a rank, before the collective starts.

``memory_guard`` stops a rank *loading* a stage it cannot hold. That is only
half the problem: a stage that loads fine can still be killed by one long
prompt, because prefill allocates KV for every token plus a transient
attention spike. Single-node oMLX already refuses those requests up front —
``raise_if_prefill_exceeds`` is the shared front door, used by engines that
have no ``Scheduler`` — and this gives a cluster rank the same door.

Two things make the cluster case different from single-node.

**A rank holds a slice, not the model.** ``set_model_info_from_model`` reads
``num_hidden_layers`` from the config, which is the whole model. A pipeline
rank holding 22 of 92 layers would be charged 4x its real KV growth and reject
prompts it could serve. So the extracted dims are corrected: KV layers to the
stage's layer count, attention heads to this rank's shard.

**A raise must not desync the collective.** Every rank checks its own slice and
then contributes a rejection vote before the first ``model()`` call. If any
slice would exceed its local ceiling, every rank leaves the request together;
otherwise every rank enters the model collectives together.

**Eviction beats rejection.** A rejection-only guard still lets a large cold
prefill reach the allocator when the wired pool is merely *close* to the
ceiling — the 2026-07-19 ThunderMLX wedge: MLX cannot allocate, the forward
makes zero progress, the watchdog kills the process, and a process dying
inside an in-flight Metal allocation orphans its wired memory, forcing a host
reboot. So before the rejection vote, every rank runs a fail-open eviction
ladder (ported from ThunderMLX's ``m3_prefill_admission.py``): trim the Metal
freed-buffer pool, then drop idle resident prompt-cache entries. Worst case
the ladder frees nothing and the request is rejected exactly as before; in the
wedge scenario it frees the headroom in about a second and the reboot-class
failure becomes a cache miss.

COLLECTIVE-FREE INVARIANT: every ladder *action* (pool trim, cache drop) is
allocator-local — no collectives, no cross-rank messaging. The one collective
is the deficit *vote*, and it is what makes eviction safe here: unequal Macs
measure different deficits, so a unilateral drop would leave one rank
retaining a prefix another rank evicted, and the next request would start at
different token offsets and block forever in the first unmatched collective
(the reason ``performance.py`` pins ``prompt_cache_size=1``). Instead every
rank contributes its post-trim deficit to one ``all_sum``, and every rank then
drops the same largest-first covering set against the common target. The vote
gate is deliberately a pure function of the broadcast prompt length — never of
local pressure, config, or env — so every rank reaches the vote on exactly the
same requests. mlx-lm hands each request a deep copy of the trie entry, so
nothing enumerated for eviction is referenced by in-flight work; losing an
entry only ever means a future cache miss.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# mlx-lm's server prefills in 2048-token chunks; the transient attention peak
# is set by the chunk, not the prompt.
_DEFAULT_PREFILL_STEP = 2048

# --- Prefill admission ladder constants (ThunderMLX production values) -------
#
# Only guard prefills at least this large: small prompts never approach the
# ceiling and must not pay for the extra deficit vote. This constant gates a
# COLLECTIVE, so it is deliberately not env-tunable — ranks that disagree on
# the gate would hang each other. oMLX launches every rank from one launcher
# with one environment, but a plain module constant cannot drift at all.
ADMISSION_MIN_PROMPT_TOKENS = 8192


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


# Use at most this fraction of the admission ceiling as the prefill budget;
# the remainder is transient-allocation + macOS headroom. The tuning knobs
# below only shift each rank's local deficit — never whether the vote runs —
# so per-rank env drift degrades into over/under-eviction, not a hang.
ADMISSION_SAFETY_FRACTION = _env_float("OMLX_CLUSTER_ADMISSION_SAFETY_FRACTION", 0.92)

# Per-token KV cost fallback (bytes), used only when the rank's monitor cannot
# produce a slice-correct estimate. ThunderMLX measured ~87.6 KB/token on its
# rank0 (38 of 60 layers) and shipped 90 KB.
ADMISSION_KV_BYTES_PER_TOKEN = _env_int(
    "OMLX_CLUSTER_ADMISSION_KV_BYTES_PER_TOKEN", 90_000
)

# Fixed transient-activation reserve on top of KV (prefill chunk buffers,
# logits, routing). Deliberately generous: over-reserving costs a little extra
# eviction, under-reserving risks the wedge the ladder exists to prevent.
ADMISSION_ACTIVATION_RESERVE_BYTES = _env_int(
    "OMLX_CLUSTER_ADMISSION_ACTIVATION_RESERVE_BYTES", 8 * 1024**3
)

_MIB = 1024**2
_GIB = 1024**3


def admission_deficit_bytes(
    prompt_tokens: int,
    current_wired_bytes: int,
    wired_limit_bytes: int,
    *,
    kv_bytes_per_token: float | None = None,
    safety_fraction: float | None = None,
    activation_reserve_bytes: int | None = None,
) -> float:
    """Bytes that must be freed before this prefill fits, or <=0 if it fits.

    deficit = (current_wired + prompt_tokens*kv_per_token + activation_reserve)
              - wired_limit*safety_fraction

    An unknown ceiling (<=0) yields 0: a rank that cannot reason about its
    headroom does nothing here and lets the rejection vote decide (fail-open).
    """
    if kv_bytes_per_token is None:
        kv_bytes_per_token = ADMISSION_KV_BYTES_PER_TOKEN
    if safety_fraction is None:
        safety_fraction = ADMISSION_SAFETY_FRACTION
    if activation_reserve_bytes is None:
        activation_reserve_bytes = ADMISSION_ACTIVATION_RESERVE_BYTES
    if wired_limit_bytes <= 0:
        return 0
    need = prompt_tokens * kv_bytes_per_token + activation_reserve_bytes
    budget = wired_limit_bytes * safety_fraction
    return (current_wired_bytes + need) - budget


def plan_eviction(
    deficit_bytes: float,
    evictables: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Pick the smallest set of idle items whose bytes cover the deficit.

    Largest-first so the headroom is freed in the fewest evictions. Every
    caller passes the rank-AGREED target (not its local deficit), so ranks
    holding identical cache entries compute identical drop sets.

    Returns (chosen_items, still_short_bytes). still_short>0 means everything
    safely evictable was chosen and the caller proceeds anyway (fail-open) —
    the rejection vote remains the backstop.
    """
    if deficit_bytes <= 0:
        return [], 0
    chosen: list[dict[str, Any]] = []
    freed = 0
    for item in sorted(evictables, key=lambda e: e.get("bytes", 0), reverse=True):
        if freed >= deficit_bytes:
            break
        if item.get("bytes", 0) <= 0:
            continue
        chosen.append(item)
        freed += item["bytes"]
    return chosen, max(0, deficit_bytes - freed)


def run_admission(
    prompt_tokens: int,
    *,
    read_wired: Any,
    read_limit: Any,
    trim_pool: Any,
    list_idle_evictables: Any,
    agree_eviction: Any | None = None,
    charge_tokens: int | None = None,
    kv_bytes_per_token: float | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Orchestrate a per-rank admission check + least-disruptive eviction ladder.

    Pure control flow; all side effects go through the injected callbacks:
    ``read_wired``/``read_limit`` measure, ``trim_pool`` returns the Metal
    freed-buffer pool to the OS (rung 1), ``list_idle_evictables`` enumerates
    droppable cache entries as ``{label, bytes, drop}`` (rung 2), and
    ``agree_eviction`` turns this rank's post-trim deficit into the rank-agreed
    eviction target (the ladder's one collective, supplied by the caller).

    ``prompt_tokens`` gates whether the ladder runs at all and MUST be a
    rank-uniform value (the broadcast prompt length): the agreement hook is a
    collective, so it is invoked exactly once per guarded prompt on every
    rank — including ranks whose local deficit is zero. ``charge_tokens`` is
    the local, possibly-smaller count the deficit is charged for (a prefix
    cache hit is already resident and must not trigger eviction of the very
    entry that made the request cheap).

    Fail-open by contract: every callback is guarded, so this function never
    raises and never refuses; worst case it behaves like the rejection-only
    path it precedes. Returns a dict describing what happened (telemetry/tests).
    """
    log = logger if logger is not None else globals()["logger"]

    def _safe(callable_: Any, default: Any) -> Any:
        try:
            return callable_()
        except Exception as exc:  # a broken probe must never block a request
            log.debug("prefill admission probe failed: %s", exc)
            return default

    info: dict[str, Any] = {"guarded": False, "prompt_tokens": int(prompt_tokens)}
    if prompt_tokens < ADMISSION_MIN_PROMPT_TOKENS:
        return info
    info["guarded"] = True
    charged = (
        max(0, int(charge_tokens)) if charge_tokens is not None else int(prompt_tokens)
    )

    limit = _safe(read_limit, 0) or 0
    before = _safe(read_wired, 0) or 0
    deficit = admission_deficit_bytes(
        charged, before, limit, kv_bytes_per_token=kv_bytes_per_token
    )
    info.update(
        {
            "wired_before_gib": round(before / _GIB, 2),
            "limit_gib": round(limit / _GIB, 2),
            "deficit_gib": round(deficit / _GIB, 2),
        }
    )

    actions: list[dict[str, Any]] = []
    # Rung 1: trim the freed-buffer pool (cheapest; frees nothing live).
    if deficit > 0:
        freed_pool = _safe(trim_pool, 0) or 0
        actions.append({"step": "trim_pool", "freed_gib": round(freed_pool / _GIB, 2)})
        after_pool = _safe(read_wired, before) or before
        deficit = admission_deficit_bytes(
            charged, after_pool, limit, kv_bytes_per_token=kv_bytes_per_token
        )

    # Rung 2: drop IDLE resident prompt-cache entries against the agreed
    # target. The hook runs even at zero local deficit — see the docstring.
    target = 0
    if agree_eviction is not None:
        target = max(0, int(_safe(lambda: agree_eviction(max(0, deficit)), 0) or 0))
    if target > 0:
        evictables = _safe(list_idle_evictables, []) or []
        chosen, still_short = plan_eviction(target, evictables)
        for item in chosen:
            try:
                item["drop"]()
                actions.append(
                    {
                        "step": "evict_idle",
                        "label": item.get("label"),
                        "freed_gib": round(item.get("bytes", 0) / _GIB, 2),
                    }
                )
            except Exception as exc:  # never let one drop block the request
                actions.append(
                    {
                        "step": "evict_idle_error",
                        "label": item.get("label"),
                        "error": str(exc)[:120],
                    }
                )
        if still_short > 0:
            actions.append(
                {"step": "evict_idle_short", "short_gib": round(still_short / _GIB, 2)}
            )
        after_evict = _safe(read_wired, before) or before
        deficit = admission_deficit_bytes(
            charged, after_evict, limit, kv_bytes_per_token=kv_bytes_per_token
        )

    info["actions"] = actions
    info["wired_after_gib"] = round((_safe(read_wired, before) or before) / _GIB, 2)
    info["residual_deficit_gib"] = round(max(0, deficit) / _GIB, 2)
    info["fits"] = deficit <= 0
    if actions:
        if deficit <= 0:
            log.warning(
                "prefill admission: freed headroom for %d-token prefill "
                "(wired %.1f->%.1f GiB, limit %.1f); actions=%s",
                prompt_tokens,
                info["wired_before_gib"],
                info["wired_after_gib"],
                info["limit_gib"],
                [a.get("step") for a in actions],
            )
        else:
            log.warning(
                "prefill admission: STILL short %.1f GiB after eviction for a "
                "%d-token prefill (wired %.1f, limit %.1f) — proceeding "
                "fail-open; the rejection vote remains the backstop",
                info["residual_deficit_gib"],
                prompt_tokens,
                info["wired_after_gib"],
                info["limit_gib"],
            )
    return info


def _trim_metal_pool(mx_module: Any) -> int:
    """Rung 1: return MLX's freed-buffer pool to the OS; bytes actually freed.

    Allocator-local — no collectives. The full-device barrier comes first for
    the same reason as the scheduler's ``_sync_and_clear_cache``: an in-flight
    command buffer on another stream may still reference pool entries (the
    #300/#888/#1106 race class), and the allocator only releases entries whose
    refcount is zero once the GPU pipeline has drained.
    """

    import gc

    before = int(mx_module.get_cache_memory())
    mx_module.synchronize()
    gc.collect()
    mx_module.clear_cache()
    return max(0, before - int(mx_module.get_cache_memory()))


def _make_prompt_cache_drop(prompt_cache: Any, model: Any, tokens: Any) -> Any:
    """Drop exactly one resident trie entry, mirroring ``trim_to`` accounting."""

    def drop() -> None:
        entry = prompt_cache._trie.pop(model, tokens)
        if entry is None:
            return
        prompt_cache._lru.remove(model, tokens)
        prompt_cache._n_bytes -= entry.nbytes
        prompt_cache._n_bytes_by_type[entry.cache_type] -= entry.nbytes

    return drop


def _idle_prompt_cache_evictables(prompt_cache: Any) -> list[dict[str, Any]]:
    """Enumerate a rank's resident prompt-cache entries as evictable items.

    mlx-lm's ``fetch_nearest_cache`` deep-copies the trie entry it returns, so
    no entry still in the trie is referenced by in-flight prefill/decode work
    — everything enumerated here is IDLE in the ThunderMLX rung-2 sense, and
    dropping it costs a future cache miss, never live KV. The current request
    is likewise unaffected: telemetry prefetched (and still holds) its copy
    before the guard ran.

    The enumeration order is deterministic given the same insert history, so
    ranks holding identical entries produce identical ``plan_eviction`` sets
    for the same agreed target. Any internals drift (an mlx-lm upgrade
    reshaping ``LRUPromptCache``) surfaces as an exception inside
    ``run_admission``'s guarded callback and simply disables rung 2.
    """

    if prompt_cache is None:
        return []
    trie = getattr(prompt_cache, "_trie", None)
    lru = getattr(prompt_cache, "_lru", None)
    if trie is None or lru is None:
        return []
    evictables: list[dict[str, Any]] = []
    for cache_type in getattr(lru, "_ordering", ()):
        for model, tokens in list(lru._lrus.get(cache_type, ())):
            entry = trie.get(model, tokens)
            nbytes = int(getattr(entry, "nbytes", 0) or 0)
            if entry is None or nbytes <= 0:
                continue
            evictables.append(
                {
                    "label": f"prompt-cache entry ({len(tokens)} tokens, {cache_type})",
                    "bytes": nbytes,
                    "drop": _make_prompt_cache_drop(prompt_cache, model, tokens),
                }
            )
    return evictables


def rank_monitor(
    model: Any,
    *,
    layer_count: int = 0,
    tensor_parallel_size: int = 1,
) -> Any | None:
    """A ``MemoryMonitor`` calibrated to the slice this rank actually holds.

    Returns None if the model's dimensions could not be read, which makes the
    guard a no-op rather than a source of spurious rejections — the same
    best-effort contract ``set_model_info_from_model`` documents.
    """

    from omlx.memory_monitor import MemoryMonitor, set_model_info_from_model

    monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    try:
        set_model_info_from_model(monitor, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not read model dims for the prefill guard: %s", exc)
        return None

    num_layers = int(getattr(monitor, "_num_layers", 0) or 0)
    kv_heads = int(getattr(monitor, "_num_kv_heads", 0) or 0)
    head_dim = int(getattr(monitor, "_head_dim", 0) or 0)
    if not kv_heads or not head_dim:
        return None

    heads = int(getattr(monitor, "_num_attention_heads", 0) or kv_heads)
    dtype_size = float(getattr(monitor, "_dtype_size", 2) or 2)
    kv_override = getattr(monitor, "_kv_bytes_per_token_override", None)

    # Pipeline: this rank stores KV for its own layers only.
    stage_layers = int(layer_count) if layer_count else num_layers
    stage_layers = max(1, min(stage_layers, num_layers or stage_layers))

    # Tensor parallel: heads are split across ranks, so both the KV this rank
    # stores and the attention transient it computes shrink with the shard.
    tp = max(1, int(tensor_parallel_size))
    if tp > 1:
        kv_heads = max(1, kv_heads // tp)
        heads = max(1, heads // tp)
        if kv_override:
            kv_override = float(kv_override) / tp

    monitor.set_model_info(
        num_layers=num_layers or stage_layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        dtype_size=dtype_size,
        num_attention_heads=heads,
        num_kv_cache_layers=stage_layers,
        kv_bytes_per_token=kv_override,
    )
    return monitor


class RankPrefillGuard:
    """Reject a prompt this rank cannot prefill, with a reason.

    Disabled (``active`` False) when the model's dims are unreadable or the
    host reports no ceiling — matching the single-node rule that an
    unmeasurable machine is never blocked, only an over-large request is.
    """

    def __init__(
        self,
        monitor: Any | None,
        *,
        rank: int = 0,
        node_id: str = "",
        ceiling_bytes: int = 0,
        prefill_step_size: int = _DEFAULT_PREFILL_STEP,
    ) -> None:
        self._monitor = monitor
        self._rank = int(rank)
        self._node_id = node_id
        self._ceiling = max(0, int(ceiling_bytes))
        self._step = max(1, int(prefill_step_size))

    @property
    def active(self) -> bool:
        return self._monitor is not None and self._ceiling > 0

    def _kv_bytes_per_token(self) -> float | None:
        """This rank's slice-correct KV cost per token, if measurable."""

        if self._monitor is None:
            return None
        try:
            per_token = float(self._monitor.estimate_prompt_kv_bytes(1024)) / 1024.0
        except Exception:  # pragma: no cover - defensive
            return None
        return per_token if per_token > 0 else None

    def _run_admission_ladder(
        self,
        num_prompt_tokens: int,
        *,
        cached_tokens: int = 0,
        current_usage_bytes: int | None = None,
        mx_module: Any,
        world_size: int,
        rank: int,
        prompt_cache: Any | None = None,
    ) -> None:
        """Free headroom before a big prefill instead of dying inside it.

        Fail-open wrapper around ``run_admission``: any ladder error falls
        through to the rejection vote below, which stays the fail-closed
        backstop. The deficit vote inside ``agree_eviction`` is the ladder's
        only collective and runs on every rank for every guarded prompt — the
        gate above is a pure function of the broadcast prompt length, so no
        local state (pressure, config, env) can make one rank skip a vote its
        peers are waiting on.
        """

        if num_prompt_tokens < ADMISSION_MIN_PROMPT_TOKENS:
            return

        from .memory_guard import current_usage_bytes as measure_usage

        pinned = current_usage_bytes is not None
        pinned_usage = max(0, int(current_usage_bytes)) if pinned else 0

        def read_wired() -> int:
            return pinned_usage if pinned else measure_usage()

        def agree_eviction(local_deficit_bytes: float) -> int:
            if world_size <= 1:
                return max(0, int(local_deficit_bytes))
            # One-hot-per-rank deficit vector, same shape as the rejection
            # vote below; all_sum leaves each rank's MiB deficit at its own
            # index and the common target is the worst rank's shortfall.
            deficits = [0] * world_size
            deficits[rank] = -(-max(0, int(local_deficit_bytes)) // _MIB)
            agreed = mx_module.distributed.all_sum(mx_module.array(deficits)).tolist()
            return int(max(agreed)) * _MIB

        try:
            run_admission(
                num_prompt_tokens,
                charge_tokens=max(0, num_prompt_tokens - max(0, int(cached_tokens))),
                read_wired=read_wired,
                read_limit=lambda: self._ceiling,
                trim_pool=lambda: _trim_metal_pool(mx_module),
                list_idle_evictables=lambda: _idle_prompt_cache_evictables(
                    prompt_cache
                ),
                agree_eviction=agree_eviction,
                kv_bytes_per_token=self._kv_bytes_per_token(),
                logger=logger,
            )
        except Exception as exc:  # fail-open: the vote below is unaffected
            logger.warning(
                "Prefill admission ladder failed open on rank %d: %s",
                self._rank,
                exc,
            )

    def check(
        self,
        num_prompt_tokens: int,
        *,
        cached_tokens: int = 0,
        request_id: str | None = None,
        current_usage_bytes: int | None = None,
    ) -> None:
        """Raise ``PrefillMemoryExceededError`` if this prompt will not fit."""

        if not self.active:
            return

        from omlx.memory_monitor import raise_if_prefill_exceeds

        from .memory_guard import current_usage_bytes as measure_usage

        usage = (
            measure_usage()
            if current_usage_bytes is None
            else max(0, int(current_usage_bytes))
        )
        try:
            raise_if_prefill_exceeds(
                self._monitor,
                prefill_memory_guard=True,
                hard_limit_bytes=self._ceiling,
                current_usage_bytes=usage,
                prefill_step_size=self._step,
                num_prompt_tokens=int(num_prompt_tokens),
                cached_tokens=max(0, int(cached_tokens)),
                request_id=request_id,
            )
        except Exception as exc:
            from omlx.exceptions import PrefillMemoryExceededError

            if not isinstance(exc, PrefillMemoryExceededError):
                raise
            where = f"rank {self._rank}" + (
                f" ({self._node_id})" if self._node_id else ""
            )
            logger.warning("Cluster prefill rejected on %s: %s", where, exc)
            raise

    def check_collective(
        self,
        num_prompt_tokens: int,
        *,
        cached_tokens: int = 0,
        request_id: str | None = None,
        current_usage_bytes: int | None = None,
        mx_module: Any | None = None,
        prompt_cache: Any | None = None,
    ) -> None:
        """Make prefill admission one rank-agreed decision.

        The request has already been broadcast by MLX-LM when its prompt cache
        is consulted. A local raise at that point lets peer ranks continue into
        the model collective and hang. Instead, every rank measures its own
        resident slice, exchanges a one-hot rejection vote, and raises before
        model execution if any rank refused the prompt.

        For prompts large enough to threaten the wired ceiling, the fail-open
        eviction ladder runs first: trim the Metal pool, vote on the remaining
        deficit, and drop the same idle prompt-cache entries on every rank.
        Rejection is the backstop for what eviction could not cover.
        ``prompt_cache`` is the rank's resident MLX-LM prompt cache (rung 2's
        eviction source); callers without one simply get a ladder whose second
        rung finds nothing to drop.
        """

        from omlx.exceptions import PrefillMemoryExceededError

        if mx_module is None:
            import mlx.core as collective_mx
        else:
            collective_mx = mx_module

        group = collective_mx.distributed.init()
        world_size = int(group.size())
        rank = int(group.rank())

        self._run_admission_ladder(
            num_prompt_tokens,
            cached_tokens=cached_tokens,
            current_usage_bytes=current_usage_bytes,
            mx_module=collective_mx,
            world_size=world_size,
            rank=rank,
            prompt_cache=prompt_cache,
        )

        local_error: PrefillMemoryExceededError | None = None
        try:
            self.check(
                num_prompt_tokens,
                cached_tokens=cached_tokens,
                request_id=request_id,
                current_usage_bytes=current_usage_bytes,
            )
        except PrefillMemoryExceededError as exc:
            local_error = exc

        if world_size <= 1:
            if local_error is not None:
                raise local_error
            return

        votes = [0] * world_size
        if local_error is not None:
            votes[rank] = 1
        agreed_votes = collective_mx.distributed.all_sum(
            collective_mx.array(votes)
        ).tolist()
        rejecting_ranks = [
            index for index, rejected in enumerate(agreed_votes) if int(rejected)
        ]
        if not rejecting_ranks:
            return
        if local_error is not None:
            raise local_error

        rejecting = rejecting_ranks[0]
        raise PrefillMemoryExceededError(
            message=(
                f"Cluster prefill rejected by rank {rejecting}: its local model "
                "slice would exceed the host memory limit. Reduce context length "
                "or free memory on that node."
            ),
            request_id=request_id,
        )


def build_guard(
    model: Any,
    *,
    rank: int,
    node_id: str = "",
    layer_count: int = 0,
    tensor_parallel_size: int = 1,
    memory_guard_tier: str = "balanced",
    prefill_step_size: int = _DEFAULT_PREFILL_STEP,
) -> RankPrefillGuard:
    """The guard for a loaded rank, using this Mac's own admission ceiling."""

    from .memory_guard import ceiling_breakdown

    try:
        ceiling = int(ceiling_breakdown(memory_guard_tier).get("hard_limit", 0))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("No admission ceiling available for the prefill guard: %s", exc)
        ceiling = 0

    return RankPrefillGuard(
        rank_monitor(
            model,
            layer_count=layer_count,
            tensor_parallel_size=tensor_parallel_size,
        ),
        rank=rank,
        node_id=node_id,
        ceiling_bytes=ceiling,
        prefill_step_size=prefill_step_size,
    )
