# SPDX-License-Identifier: Apache-2.0
"""Conditional MTP dispatch inside ``mlx_lm.generate.GenerationBatch``.

This is the integration point that lets the existing oMLX scheduler /
paged cache / prefix cache / SSD cache stack drive MTP without touching any
of those layers. ``GenerationBatch`` is mlx-lm's per-step decoder for the
active set of sequences in continuous batching. We patch:

- ``GenerationBatch.__init__`` — leave the standard mlx-lm initialization
  untouched. Fresh singleton donor batches may still be merged into a larger
  continuous batch, so MTP must not mutate cache state in ``__init__``.

- ``GenerationBatch.next`` — when the batch holds exactly one MTP-capable
  sequence, lazily initialize MTP from the standard post-prefill state. We
  emit from the per-batch queue first; once empty, we run a 2-token verify
  forward over ``[next_main, draft]`` with ``n_confirmed=1`` and a single
  MTP-head forward at the bonus position (accept) or confirmed position
  (reject), refilling the queue from the verify outputs.

- ``GenerationBatch.extend`` / ``filter`` — drop MTP state whenever continuous
  batching reshapes ownership. MTP state belongs to one uid in one singleton
  timeline; it must not survive standard batched decoding.

The throughput math (greedy, accept rate p):
  - Cost per *cycle*: 1× backbone (2-token verify) + 1× MTP head ≈ 1.15
  - Tokens per cycle: 1 + p (accept emits draft+bonus; reject emits verify_pred only)
  - At p≈1: 0.575 cost/token → ~1.74× throughput
  - At p≈0.5: ~0.77 cost/token → ~1.30× throughput

Known limitation (compute-bound single-stream Apple Silicon):
  The cost model above assumes the 2-token verify forward is nearly free
  relative to a 1-token forward, which is the bandwidth-bound decode regime
  speculative decoding targets. On lower-end single-stream Apple Silicon
  (e.g. M1/M2 base/Pro) decode is compute-bound, so the verify forward costs
  ~2× a 1-token forward and MTP can be net-negative regardless of accept
  rate. Wins are expected on M3/M4 or higher-end parts, on MoE models with a
  smaller per-step backbone, or under continuous batching where spare
  compute exists. See #1097 / #1311 for measurements.

Greedy identity (sampler is None): the patched dispatch produces the same
tokens as the standard step. PR 990's ``test_mtp_generate_identity``
encodes this contract; the oMLX-side equivalent lives in
``tests/test_mlx_lm_mtp_patch.py``.

Stochastic acceptance (sampler is not None): we use ``min(1, p_target / p_draft)``
(Leviathan & Chen 2023). On rejection we sample from the residual
``max(p_target - p_draft, 0) / Z`` so the marginal output distribution
equals the target distribution exactly.

PagedCacheManager interaction
-----------------------------
``cache.trim(1)`` on a ``BatchKVCache`` only updates ``self._idx``; the
underlying paged blocks are untouched. ``ArraysCache.rollback_state``
holds ``(conv_snap, ssm_snap)`` snapshots produced by the patched
``GatedDeltaNet.__call__`` and is restored on reject. Because both code
paths only mutate cache *length* (not block ownership), oMLX's
``PagedCacheManager`` is oblivious to the trim — its block_table is
unaffected and prefix-cache lookups continue to work normally.

TokenBuffer interaction
-----------------------
``GenerationBatch._token_context[0]`` is a ``TokenBuffer`` accumulating
the prompt + every forward-input token. We update it in lock-step with
each forward-input position so that ``logits_processors`` see the same
token sequence the standard step would see. On reject we shrink the
buffer's ``_size`` by 1 to discard the rejected draft (mirroring PR 990's
``prev_tokens = prev_tokens[:-1]``).
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Tuple

from omlx.prefill_progress import get_prefill_tracker

from . import cache_rollback as _rollback_mod
from . import prompt_priming as _prompt_priming

logger = logging.getLogger(__name__)

_MTP_RUNTIME_LOCK = threading.Lock()
_MTP_RUNTIME_TOTALS: Dict[str, Any] = {
    "sequences": 0,
    "tokens": 0,
    "cycles": 0,
    "accepted_draft_tokens": 0,
    "physical_drafted_tokens": 0,
    "drafted_tokens": 0,
    "zero_depth_cycles": 0,
    "depth_drafted": [],
    "depth_accepted": [],
    "timing_ms": {
        "backbone": 0.0,
        "mtp_head": 0.0,
        "sampling": 0.0,
        "cache_ops": 0.0,
    },
    "last_finish_reason": "",
}


def mtp_runtime_stats_snapshot() -> Dict[str, Any] | None:
    """Return bounded process-local MTP economics for cluster telemetry."""

    with _MTP_RUNTIME_LOCK:
        sequences = int(_MTP_RUNTIME_TOTALS["sequences"])
        if sequences <= 0:
            return None
        cycles = int(_MTP_RUNTIME_TOTALS["cycles"])
        drafted = int(_MTP_RUNTIME_TOTALS["drafted_tokens"])
        physical_drafted = int(_MTP_RUNTIME_TOTALS["physical_drafted_tokens"])
        accepted = int(_MTP_RUNTIME_TOTALS["accepted_draft_tokens"])
        tokens = int(_MTP_RUNTIME_TOTALS["tokens"])
        return {
            **{
                key: value
                for key, value in _MTP_RUNTIME_TOTALS.items()
                if key not in {"depth_drafted", "depth_accepted", "timing_ms"}
            },
            "acceptance_ratio": (
                accepted / drafted if drafted else 0.0
            ),
            "physical_acceptance_ratio": (
                accepted / physical_drafted if physical_drafted else 0.0
            ),
            "tokens_per_cycle": tokens / cycles if cycles else 0.0,
            "depth_drafted": list(_MTP_RUNTIME_TOTALS["depth_drafted"]),
            "depth_accepted": list(_MTP_RUNTIME_TOTALS["depth_accepted"]),
            "timing_ms": dict(_MTP_RUNTIME_TOTALS["timing_ms"]),
        }


def _record_mtp_runtime_stats(stats: "_MtpStats", finish_reason: str) -> None:
    total_emits = (
        stats.init_emits + stats.draft_emits + stats.bonus_emits + stats.verify_emits
    )
    total_drafted = sum(stats.depth_drafted) or stats.cycles
    physical_drafted = stats.physical_drafts or total_drafted
    with _MTP_RUNTIME_LOCK:
        _MTP_RUNTIME_TOTALS["sequences"] += 1
        _MTP_RUNTIME_TOTALS["tokens"] += total_emits
        _MTP_RUNTIME_TOTALS["cycles"] += stats.cycles
        _MTP_RUNTIME_TOTALS["accepted_draft_tokens"] += stats.accepts
        _MTP_RUNTIME_TOTALS["drafted_tokens"] += total_drafted
        _MTP_RUNTIME_TOTALS["physical_drafted_tokens"] += physical_drafted
        _MTP_RUNTIME_TOTALS["zero_depth_cycles"] += stats.zero_cycles
        for key, values in (
            ("depth_drafted", stats.depth_drafted),
            ("depth_accepted", stats.depth_accepted),
        ):
            totals = _MTP_RUNTIME_TOTALS[key]
            if len(totals) < len(values):
                totals.extend([0] * (len(values) - len(totals)))
            for index, value in enumerate(values):
                totals[index] += int(value)
        timing = _MTP_RUNTIME_TOTALS["timing_ms"]
        timing["backbone"] += stats.backbone_ms
        timing["mtp_head"] += stats.mtp_head_ms
        timing["sampling"] += stats.sample_ms
        timing["cache_ops"] += stats.cache_ops_ms
        _MTP_RUNTIME_TOTALS["last_finish_reason"] = str(finish_reason)[:64]


def _set_verify_qmm_armed(flag: bool) -> None:
    """Arm the verify-shape qmm routing for the duration of an MTP forward.

    Import is deferred and failure-tolerant: the kernel module is optional
    and its absence must not affect the MTP path.
    """
    try:
        from ..qwen35_verify_qmm import set_verify_qmm_armed

        set_verify_qmm_armed(flag)
    except Exception:
        pass


def _set_dspark_target_verify(model: Any, flag: bool) -> None:
    try:
        import sys

        host = _dspark_host(model)
        if host is None:
            return
        module = sys.modules.get(type(host).__module__)
        setter = getattr(module, "set_dspark_verify_armed", None)
        if setter is not None:
            setter(flag)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply() -> bool:
    """Wrap ``GenerationBatch`` and ``BatchGenerator`` MTP hooks.

    One-shot by design: the wraps capture ``original_*`` in closures so
    re-applying would chain wraps and double-init. ``GenerationBatch`` is
    not touched by dflash so the leftover-class-patch risk that motivates
    self-healing elsewhere doesn't apply here.
    """
    try:
        from mlx_lm.generate import BatchGenerator, GenerationBatch
    except ImportError:
        logger.debug("mlx_lm.generate GenerationBatch/BatchGenerator not importable")
        return False

    if not hasattr(GenerationBatch, "_omlx_mtp_patched"):
        original_init = GenerationBatch.__init__
        original_next = GenerationBatch.next
        original_filter = GenerationBatch.filter
        original_extend = GenerationBatch.extend

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Stock GenerationBatch owns an exact one-token pipeline. Qwen4
            # MTP clears this marker before its first extra target forward and
            # restores it only after an exact reconcile/park/handoff.
            self._omlx_standard_target_exact_v1 = True
            # Do not activate MTP here. Fresh singleton batches created by
            # PromptProcessingBatch.generate() may still be merged into a larger
            # continuous batch; mutating their cache in __init__ can corrupt the
            # later standard batched path. Activation is lazy in patched_next().
            uids = getattr(self, "uids", None)
            if uids:
                reason = _ineligibility_reason(self)
                if reason:
                    logger.debug("MTP path not active: %s", reason)

        def patched_next(self, *args, **kwargs):
            realign_rows = getattr(self, "_omlx_realign_rows", None)
            if callable(realign_rows):
                realign_rows()
            _maybe_clear_multirow_marker(self)

            if _is_mtp_batch_eligible(self):
                try:
                    batch_state = _prepare_mtp_batch_state_for_next(self)
                    if batch_state is not None:
                        return _mtp_batch_next(self, batch_state)
                except _MtpStepFallback as exc:
                    logger.debug("MTP batch next() fallback to standard step: %s", exc)
                    _reconcile_mtp_batch_to_standard(self)
                    _drop_mtp_batch_state(self, "batch-step-fallback")
            elif getattr(self, "_omlx_mtp_batch_state", None) is not None:
                _reconcile_mtp_batch_to_standard(self)
                _drop_mtp_batch_state(self, "batch-ineligible")

            if _is_mtp_eligible(self):
                handed_off = False
                if _singleton_mtp_handoff_ready(self):
                    # A prefill is waiting on this batch generator; hand the
                    # singleton back to the standard step at the drained-queue
                    # boundary so the late join merges this very call (#2515).
                    handed_off = _handoff_mtp_for_late_join(
                        self, self._omlx_mtp_state
                    )
                if not handed_off:
                    try:
                        state = _prepare_mtp_state_for_next(self)
                        if state is not None:
                            return _mtp_next(self, state)
                    except _MtpStepFallback as exc:
                        logger.debug(
                            "MTP next() fallback to standard step: %s", exc
                        )
                        active = getattr(self, "_omlx_mtp_state", None)
                        if active is not None:
                            if not _reconcile_mtp_to_standard(self, active):
                                raise RuntimeError(
                                    "MTP step fallback could not reconstruct "
                                    "an exact standard target cache"
                                ) from exc
                            if active.reentry_probe:
                                try:
                                    delattr(self, "_omlx_mtp_park_state")
                                except AttributeError:
                                    pass
                        _drop_mtp_state(self, "step-fallback")
            else:
                _drop_mtp_state(self, "non-singleton-or-ineligible")
            _log_multirow_mtp_inactive_once(self)
            _mark_standard_multirow_decode(self)
            tax_probe = getattr(self, "_omlx_mtp_tax_probe", None)
            park_state = _mtp_park_state_for_batch(self)
            if tax_probe is not None or park_state is not None:
                step_t0 = time.perf_counter()
                result = _standard_multirow_next(
                    self,
                    lambda: original_next(self, *args, **kwargs),
                )
                if tax_probe is not None:
                    _record_std_tax_sample(
                        self, (time.perf_counter() - step_t0) * 1000.0
                    )
                if park_state is not None:
                    _record_parked_standard_step(self)
                return _stamp_standard_terminal_responses(
                    self,
                    _stamp_qwen4_standard_terminal_responses(self, result),
                )
            return _stamp_standard_terminal_responses(
                self,
                _stamp_qwen4_standard_terminal_responses(
                    self,
                    _standard_multirow_next(
                        self,
                        lambda: original_next(self, *args, **kwargs),
                    ),
                ),
            )

        def patched_extend(self, batch, *args, **kwargs):
            # The host (self) may have active MTP about to gain a co-runner.
            # The MTP path never maintains mlx-lm's _next_tokens, so a plain
            # drop here would leave standard batched decode resuming from a
            # stale _next_tokens against an MTP-advanced cache. Reconcile
            # before merge while ownership is still well defined.
            _reconcile_mtp_batch_to_standard(self)
            _drop_mtp_batch_state(self, "extend-reconciled")
            _drop_mtp_batch_state(batch, "donor-extended")

            host_state = getattr(self, "_omlx_mtp_state", None)
            if host_state is not None and _mtp_state_valid_for_batch(self, host_state):
                if host_state.reentry_probe:
                    park_state = _mtp_park_state_for_batch(self)
                    if park_state is not None:
                        park_state.defer_probe()
                _reconcile_mtp_to_standard(self, host_state)
                _drop_mtp_state(self, "extend-reconciled")
            result = original_extend(self, batch, *args, **kwargs)
            _drop_mtp_state(batch, "donor-extended")
            _drop_invalid_mtp_state(self, "extend")
            _drop_invalid_mtp_batch_state(self, "extend")
            # Priming only serves a singleton timeline: once this batch
            # holds >1 rows, the context can never be consumed — release
            # its head cache now instead of riding the merged decode.
            uids = getattr(self, "uids", None)
            if not uids or len(uids) != 1:
                _prompt_priming.drop_ctx(getattr(self, "model", None))
            return result

        def patched_filter(self, keep, *args, **kwargs):
            old_uids = list(getattr(self, "uids", []) or [])
            had_rowwise_state = (
                getattr(self, "_omlx_mtp_batch_state", None) is not None
            )
            compact_survivor = None
            if had_rowwise_state and len(old_uids) > 1 and len(keep) == 1:
                # Prove and detach the survivor before stock filter mutates any
                # batch-owned field. A failed Qwen4 timeline proof therefore
                # leaves the original B2 entirely intact for the request-error
                # path instead of exposing a half-filtered cache/state pair.
                compact_survivor = _prepare_compact_rowwise_mtp_survivor(
                    self,
                    int(keep[0]),
                )
            result = original_filter(self, keep, *args, **kwargs)
            _drop_invalid_mtp_state(self, "filter", log_empty=True)
            if compact_survivor is not None:
                _compact_rowwise_mtp_survivor(
                    self,
                    compact=compact_survivor,
                )
            _drop_invalid_mtp_batch_state(
                self,
                "filter",
                old_uids=old_uids,
                log_empty=True,
            )
            _mtp_park_state_for_batch(self)
            return result

        GenerationBatch.__init__ = patched_init
        GenerationBatch.next = patched_next
        GenerationBatch.filter = patched_filter
        GenerationBatch.extend = patched_extend
        GenerationBatch._omlx_mtp_patched = True

    if not hasattr(BatchGenerator, "_omlx_mtp_patched"):
        original_bg_next = BatchGenerator._next

        def patched_bg_next(self, *args, **kwargs):
            gen_batch = getattr(self, "_generation_batch", None)
            if gen_batch is not None:
                local_safe = _batch_generator_allows_mtp_activation(self)
                gen_batch._omlx_mtp_activation_safe = (
                    _synchronize_mtp_activation_safe(
                        gen_batch,
                        local_safe,
                    )
                )
            if _generation_batch_has_active_mtp(
                gen_batch
            ) and not _singleton_mtp_handoff_ready(gen_batch):
                old_completion_batch_size = getattr(
                    self,
                    "completion_batch_size",
                    None,
                )
                had_completion_batch_size = hasattr(self, "completion_batch_size")
                # Force mlx-lm's "hands full" early return after generation,
                # even if an active row-wise MTP batch shrinks during next().
                self.completion_batch_size = 0
                try:
                    return original_bg_next(self, *args, **kwargs)
                finally:
                    if had_completion_batch_size:
                        self.completion_batch_size = old_completion_batch_size
                    elif hasattr(self, "completion_batch_size"):
                        delattr(self, "completion_batch_size")
            return original_bg_next(self, *args, **kwargs)

        BatchGenerator._next = patched_bg_next
        BatchGenerator.omlx_mtp_post_emit = _batch_generator_mtp_post_emit
        BatchGenerator._omlx_mtp_patched = True
    return True


def _model_has_mtp_module(model: Any) -> bool:
    """Check whether the model actually has an MTP head attached.

    The ``mtp_forward`` method is added to the class unconditionally by
    the patch, but the per-instance ``mtp`` module is only attached when
    ``mtp_enabled`` was True at load time (see qwen35_model._patch_model
    and deepseek_v4_model._patch_model). Without the inner module the
    ``mtp_forward`` call would AttributeError, so we gate eligibility on
    the actual module's presence.
    """
    inner = getattr(model, "language_model", model)
    return hasattr(inner, "mtp") and getattr(inner, "mtp", None) is not None


def _model_mtp_decode_enabled(model: Any) -> bool:
    """Return the MTP decode decision captured on the loaded model instance.

    ``mlx_lm_mtp._MTP_ACTIVE`` is a construction-time switch. It is reset
    before each model load so patched ``__init__`` methods know whether to
    attach MTP heads, but decode-time eligibility must not read that global:
    a later non-MTP load would otherwise disable already-loaded MTP models.
    """
    candidates = [model]
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            candidates.append(inner)
    return any(
        bool(getattr(candidate, "_omlx_mtp_decode_enabled", False))
        for candidate in candidates
    )


def _model_qwen4_terminal_commit_enabled(model: Any) -> bool:
    """Return the explicit Qwen4 two-phase target-cache capability.

    Model-type inference is deliberately insufficient: a foreign qwen4_exp
    implementation may not preserve PLE/QSA rollback state across scheduler
    emission.  Only the vendored runtime that owns the transaction contract
    stamps this marker.
    """

    candidates = [model]
    for attr in ("language_model", "_language_model", "model"):
        candidate = getattr(model, attr, None)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    for wrapper in list(candidates):
        inner = getattr(wrapper, "model", None)
        if inner is not None and inner not in candidates:
            candidates.append(inner)
    return any(
        getattr(candidate, "_omlx_mtp_terminal_commit_v1", False) is True
        for candidate in candidates
    )


def _stamp_standard_terminal_responses(
    gen_batch: Any,
    responses: Any,
) -> Any:
    """Prove a standard-path finish is an exact target terminal.

    Generic MTP families (Qwen3.5, DeepSeek-V4) only publish L0 when the
    response carries ``_omlx_mtp_standard_terminal_exact``. That flag used
    to be set only after an MTP verify cycle reconciled back to the public
    ledger. A request that never entered MTP already is that ledger — the
    same exact target transaction — and must carry the same proof so L0 is
    universal instead of Qwen4-only.
    """

    if (
        getattr(gen_batch, "_omlx_mtp_state", None) is not None
        or getattr(gen_batch, "_omlx_mtp_batch_state", None) is not None
        or getattr(gen_batch, "_omlx_standard_target_exact_v1", False) is not True
    ):
        return responses
    if not _model_mtp_decode_enabled(getattr(gen_batch, "model", None)):
        return responses
    for response in responses or ():
        if (
            getattr(response, "finish_reason", None) is not None
            and isinstance(getattr(response, "prompt_cache", None), list)
            and isinstance(getattr(response, "all_tokens", None), list)
        ):
            response._omlx_mtp_standard_terminal_exact = True
    return responses


def _stamp_qwen4_standard_terminal_responses(
    gen_batch: Any,
    responses: Any,
) -> Any:
    """Prove parked/reconciled Qwen4 standard responses have no MTP queue.

    This helper is reached only after the patched MTP and row-wise branches
    have either returned or reconciled and dropped their private state.  Stock
    ``GenerationBatch.next`` then owns the usual exact cache/token transaction.
    The scheduler still validates every physical cache timeline before reuse.
    """

    if not _model_qwen4_terminal_commit_enabled(getattr(gen_batch, "model", None)):
        return responses
    if (
        getattr(gen_batch, "_omlx_mtp_state", None) is not None
        or getattr(gen_batch, "_omlx_mtp_batch_state", None) is not None
        or getattr(gen_batch, "_omlx_standard_target_exact_v1", False) is not True
    ):
        return responses
    for response in responses or ():
        if (
            getattr(response, "finish_reason", None) is not None
            and isinstance(getattr(response, "prompt_cache", None), list)
            and isinstance(getattr(response, "all_tokens", None), list)
        ):
            response._omlx_qwen4_standard_terminal_v1 = True
    return responses


def _model_mtp_tokenwise_verify_enabled(model: Any) -> bool:
    """Whether this loaded model requires decode-consistent verify rows."""

    candidates = [model]
    for attr in ("language_model", "_language_model", "model"):
        candidate = getattr(model, attr, None)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    for wrapper in list(candidates):
        inner = getattr(wrapper, "model", None)
        if inner is not None and inner not in candidates:
            candidates.append(inner)
    return any(
        bool(getattr(candidate, "_omlx_mtp_tokenwise_backbone", False))
        for candidate in candidates
    )


def _model_mtp_replay_reject_enabled(model: Any) -> bool:
    """Whether rejected verify windows rebuild via one exact decode row."""

    candidates = [model]
    for attr in ("language_model", "_language_model", "model"):
        candidate = getattr(model, attr, None)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    for wrapper in list(candidates):
        inner = getattr(wrapper, "model", None)
        if inner is not None and inner not in candidates:
            candidates.append(inner)
    return any(
        bool(getattr(candidate, "_omlx_mtp_replay_reject", False))
        for candidate in candidates
    )


def _batch_generator_allows_mtp_activation(batch_gen: Any) -> bool:
    """True when lazy MTP activation cannot race with a pending batch merge."""
    try:
        prompt_empty = len(getattr(batch_gen, "_prompt_batch", [])) == 0
        processing_empty = (
            len(getattr(batch_gen, "_currently_processing", [])) == 0
        )
        if not prompt_empty or not processing_empty:
            return False
        generation_batch = getattr(batch_gen, "_generation_batch", None)
        completion_limit = int(getattr(batch_gen, "completion_batch_size", 0))
        if completion_limit == 1 and generation_batch is not None:
            # The pinned generator returns immediately after this one decode
            # row; queued, not-yet-admitted prompts cannot merge this turn.
            return len(getattr(generation_batch, "uids", ()) or ()) == 1
        return len(getattr(batch_gen, "_unprocessed_sequences", [])) == 0
    except Exception:
        return False


def _synchronize_mtp_activation_safe(
    gen_batch: Any,
    local_safe: bool,
    *,
    mx_module: Any = None,
) -> bool:
    """Rank-agree activation and drained-queue late-join decisions.

    Request ingress can become visible to rank zero one scheduler turn before
    a peer. A rank-local ``activation_safe`` decision would then send one rank
    through the standard late-join handoff while another enters a speculative
    verify collective. Vote only where that decision can change: before a new
    singleton MTP state is created, or when an active state's emit queue has at
    most one token left. Deep queues remain pinned without an extra control
    collective.
    """

    if gen_batch is None or not _mtp_common_eligible(gen_batch):
        return bool(local_safe)
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if state is not None and len(getattr(state, "queue", ())) > 1:
        return bool(local_safe)
    if mx_module is None:
        import mlx.core as mx_module
    try:
        group = mx_module.distributed.init()
        world_size = int(group.size())
        if world_size <= 1:
            return bool(local_safe)
        vote = mx_module.array(
            [1 if local_safe else 0],
            dtype=mx_module.int32,
        )
        agreed = mx_module.distributed.all_sum(vote, group=group)
        return int(agreed.item()) == world_size
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # Fail closed: a distributed MTP rank that cannot agree must not
        # activate or cross a handoff boundary independently.
        try:
            if int(mx_module.distributed.init().size()) > 1:
                return False
        except Exception:
            pass
        return bool(local_safe)


def _generation_batch_has_active_mtp(gen_batch: Any) -> bool:
    """True while a generation batch owns Native MTP cache state.

    mlx-lm's ``BatchGenerator._next`` generates first and then may promote
    pending prompt work into the same ``GenerationBatch`` via ``extend()``. That
    merge path forces MTP reconciliation, which can re-prefill a long streamed
    context outside the scheduler's guarded prefill path. Treat active MTP as
    a temporary full generation batch so late-join requests wait, except when
    ``_singleton_mtp_handoff_ready`` says the singleton path can hand off to
    the standard step this very call (#2515).
    """
    if gen_batch is None:
        return False
    try:
        if len(gen_batch) == 0:
            return False
    except Exception:
        pass
    return (
        getattr(gen_batch, "_omlx_mtp_state", None) is not None
        or getattr(gen_batch, "_omlx_mtp_batch_state", None) is not None
    )


def _singleton_mtp_handoff_ready(gen_batch: Any) -> bool:
    """True when a pending late join should be admitted this call (#2515).

    Requires pending prefill work (the activation-safe stamp is False), no
    row-wise batch state (that opt-in path keeps the deferral), a valid
    singleton MTP state, and a drained queue: with at most one committed
    token left unstreamed the handoff to the standard step is exact and
    (near-)zero cost, so ``patched_bg_next`` skips the completion pin and
    ``patched_next`` performs the handoff in the same call. Deep queues keep
    the pin — each call drains one token, bounded by depth + 1.
    """
    if gen_batch is None:
        return False
    if getattr(gen_batch, "_omlx_mtp_activation_safe", True):
        return False
    if getattr(gen_batch, "_omlx_mtp_batch_state", None) is not None:
        return False
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if not _mtp_state_valid_for_batch(gen_batch, state):
        return False
    if state.pending_emit is not None:
        return False
    pending = state.pending_commit
    if pending is not None:
        safe_tail = bool(
            len(state.queue) == 1
            and (
                (pending.kind == "init" and pending.emitted == 1)
                or (pending.kind == "tail" and pending.emitted == 0)
            )
        )
        if not safe_tail:
            return False
    return len(state.queue) <= 1


_MTP_FORCE_STANDARD_ENV = "OMLX_MTP_FORCE_STANDARD"


def _mtp_common_eligible(gen_batch: Any) -> bool:
    # Diagnostic only: preserve the loaded MTP module and model marker while
    # routing decode through the standard target path.  This isolates
    # load-time model changes from MTP execution changes in physical A/Bs.
    if os.environ.get(_MTP_FORCE_STANDARD_ENV, "").strip() == "1":
        return False
    park_state = _mtp_park_state_for_batch(gen_batch)
    if park_state is not None:
        uids = getattr(gen_batch, "uids", None) or ()
        active = getattr(gen_batch, "_omlx_mtp_state", None)
        active_probe = bool(
            active is not None
            and getattr(active, "uid", None) == park_state.uid
            and getattr(active, "reentry_probe", False)
        )
        # Performance parking is reversible, but re-entry is deliberately a
        # singleton operation. Multi-row decode keeps using the standard path
        # until the parked row is alone and its cache is activation-safe.
        if len(uids) != 1 or (not active_probe and not park_state.probe_ready()):
            return False
    if not hasattr(gen_batch, "model"):
        return False
    if not hasattr(gen_batch.model, "mtp_forward"):
        return False
    if not _model_has_mtp_module(gen_batch.model):
        return False
    if not _model_mtp_decode_enabled(gen_batch.model):
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) == 0:
        return False
    if _has_grammar_processors(gen_batch):
        return False
    return True


_ROWWISE_BATCH_MTP_ENV = "OMLX_MTP_ROWWISE_BATCH"

_FIXED_DEPTH_ENV = "OMLX_MTP_FIXED_DEPTH"
_FORCE_DEPTH_ZERO_ENV = "OMLX_MTP_FORCE_DEPTH_ZERO"
_LOCKSTEP_DEPTH_ENV = "OMLX_MTP_DISTRIBUTED_LOCKSTEP_DEPTH"
_QWEN4_ACCEPTANCE_DEPTH_ENV = "OMLX_QWEN4_ACCEPTANCE_LOCKSTEP_DEPTH"
_QWEN4_EVIDENCE_DEPTH_ENV = "OMLX_QWEN4_EVIDENCE_DEPTH"
_QWEN4_VERIFY_PARITY_PATH_ENV = "OMLX_QWEN4_VERIFY_PARITY_PATH"
_QWEN4_VERIFY_PARITY_CYCLES_ENV = "OMLX_QWEN4_VERIFY_PARITY_CYCLES"
_QWEN4_VERIFY_PARITY_PREFILL_STEP_ENV = "OMLX_QWEN4_VERIFY_PARITY_PREFILL_STEP"
_QWEN4_SEQUENTIAL_VERIFY_ENV = "OMLX_QWEN4_SEQUENTIAL_VERIFY"
_QWEN4_SEQUENTIAL_POOLED_REQUIRED_TOKENS = 32
_QWEN4_PLE_WINDOW_PREFETCH_ENV = "OMLX_QWEN4_PLE_WINDOW_PREFETCH"


def _qwen4_sequential_verify_enabled() -> bool:
    """Opt in to the B1 greedy full-model scalar verification oracle."""

    return os.environ.get(_QWEN4_SEQUENTIAL_VERIFY_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _qwen4_ple_window_prefetch_enabled() -> bool:
    return os.environ.get(_QWEN4_PLE_WINDOW_PREFETCH_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _fixed_depth_override(max_depth: int) -> Optional[int]:
    """Diagnostic: pin the adaptive controller to one fixed draft depth.

    ``OMLX_MTP_FIXED_DEPTH=N`` (clamped to 1..max_depth) disables the
    _DepthController so every cycle drafts exactly N tokens. Used to
    measure true per-depth economics (acceptance and cycle cost) without
    the controller's hysteresis/probing masking them.  Depth zero remains
    inaccessible through that setting; the explicit
    ``OMLX_MTP_FORCE_DEPTH_ZERO=1`` diagnostic selects scalar target cycles
    while leaving MTP activation and head-history maintenance intact.
    Unset = adaptive.
    """
    if os.environ.get(_FORCE_DEPTH_ZERO_ENV, "").strip() == "1":
        return 0
    raw = os.environ.get(_FIXED_DEPTH_ENV, "").strip()
    if not raw:
        return None
    try:
        return max(1, min(max_depth, int(raw)))
    except ValueError:
        return None


def _lockstep_depth_enabled() -> bool:
    return os.environ.get(_LOCKSTEP_DEPTH_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _qwen4_evidence_depth_enabled(model: Any) -> bool:
    """Opt in to the experimental evidence policy for Qwen4 only."""

    enabled = os.environ.get(_QWEN4_EVIDENCE_DEPTH_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    return enabled and _is_qwen4_exp_model(model)


def _qwen4_acceptance_depth_enabled(model: Any) -> bool:
    """Opt in to the zero-clock acceptance policy for Qwen4 only."""

    enabled = os.environ.get(_QWEN4_ACCEPTANCE_DEPTH_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    return enabled and _is_qwen4_exp_model(model)


def _rowwise_batch_mtp_override() -> Optional[bool]:
    """Return an explicit row-wise policy override, or ``None`` when unset."""

    raw = os.environ.get(_ROWWISE_BATCH_MTP_ENV, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def _is_qwen4_exp_model(model: Any) -> bool:
    """Recognize only the native Qwen4-Exp family through common wrappers."""

    pending = [model]
    seen = set()
    while pending:
        candidate = pending.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        values = [
            getattr(candidate, "model_type", None),
            getattr(getattr(candidate, "args", None), "model_type", None),
            getattr(getattr(candidate, "config", None), "model_type", None),
            getattr(
                getattr(getattr(candidate, "config", None), "text_config", None),
                "model_type",
                None,
            ),
        ]
        if any(str(value or "").startswith("qwen4_exp") for value in values):
            return True
        module_name = str(getattr(type(candidate), "__module__", ""))
        if ".qwen4_exp" in module_name:
            return True
        for name in ("language_model", "_language_model", "model"):
            child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                pending.append(child)
    return False


def _batch_wants_api_logprobs(gen_batch: Any) -> bool:
    """Fail closed when row-wise MTP cannot prove API logprobs are unrequested."""

    explicit = getattr(gen_batch, "_omlx_wants_logprobs", None)
    if explicit is not None:
        return bool(explicit)
    try:
        from omlx.scheduler import _omlx_batch_wants_logprobs

        return bool(_omlx_batch_wants_logprobs(gen_batch))
    except Exception:
        return True


def _rowwise_batch_mtp_enabled(gen_batch: Any = None) -> bool:
    """Choose row-wise MTP using an override plus one measured Qwen4 B2 policy.

    The row-wise path runs one backbone forward per row per cycle, so its
    aggregate throughput is roughly single-stream MTP throughput regardless
    of batch size, while standard batched decode amortizes one forward over
    all rows. Measured on Qwen3.6-27B-oQ4e-mtp / M3 Ultra (pp1024/tg128):
    row-wise 53.3 / 52.5 tok/s aggregate at batch 2 / 4 versus 65.2 / 86.5
    for standard batched decode — despite 83-93% draft acceptance. It only
    pays off when tokens-per-cycle exceeds the row count.  Generic models keep
    that opt-in policy.  Qwen4-Exp is the narrow measured exception: on the M3
    Ultra, exact row-wise depth-5 B2 delivered 48.72 aggregate tok/s with only
    1.13 s finish skew, versus 25.55 tok/s and 18 s skew for standard B2.

    ``OMLX_MTP_ROWWISE_BATCH=1`` remains an explicit force and ``=0`` an
    explicit disable.  With no override, only eligible Qwen4-Exp B2 requests
    that did not ask for API logprobs select row-wise MTP.
    """

    override = _rowwise_batch_mtp_override()
    if override is not None:
        return override
    if gen_batch is None:
        return False
    uids = getattr(gen_batch, "uids", None) or ()
    return bool(
        len(uids) == 2
        and _is_qwen4_exp_model(getattr(gen_batch, "model", None))
        and not _batch_wants_api_logprobs(gen_batch)
    )


def _standard_multirow_next(gen_batch: Any, call: Any) -> Any:
    """Use normal fused DS4 attention when a multi-row batch is not using MTP.

    DS4 marks attention decode-consistent globally when its MTP head is loaded,
    because singleton verification must match one-token decode exactly. A
    standard B2/B4 step performs no speculative verification, so retaining the
    slower rowwise-exact mode is unnecessary. Toggle it off for this one step
    and restore it before singleton MTP can resume.
    """

    uids = getattr(gen_batch, "uids", None)
    model = getattr(gen_batch, "model", None)
    if uids is None or len(uids) <= 1 or not _model_mtp_decode_enabled(model):
        return call()
    inner = getattr(model, "model", None) or getattr(model, "_language_model", None)
    changed: List[Tuple[Any, bool]] = []
    for layer in getattr(inner, "layers", ()) or ():
        attention = getattr(layer, "attn", None)
        if attention is None or not hasattr(attention, "_omlx_decode_consistent"):
            continue
        previous = bool(attention._omlx_decode_consistent)
        if previous:
            attention._omlx_decode_consistent = False
            changed.append((attention, previous))
    try:
        return call()
    finally:
        for attention, previous in changed:
            attention._omlx_decode_consistent = previous


def _allows_new_mtp_activation(gen_batch: Any, state_attr: str) -> bool:
    if getattr(gen_batch, state_attr, None) is not None:
        return True
    # The multirow-decode marker guards the singleton-init invariant only
    # (see _mark_standard_multirow_decode). Row-wise batch activation seeds
    # every row from a freshly extracted per-row cache, so a prior standard
    # multi-row decode is exactly the state it expects — blocking it here
    # would permanently lock batches out of MTP, because a batch's first
    # decode step is always standard.
    if state_attr == "_omlx_mtp_state" and getattr(
        gen_batch, "_omlx_mtp_saw_standard_multirow_decode", False
    ):
        return False
    return bool(getattr(gen_batch, "_omlx_mtp_activation_safe", True))


def _mark_standard_multirow_decode(gen_batch: Any) -> None:
    """Remember that this batch has decoded with shared standard cache state.

    A row that survives a standard multi-row decode and later becomes singleton
    no longer satisfies the narrow invariant that singleton MTP initialization
    relies on. Existing row-wise MTP state may continue, but starting a fresh
    singleton MTP state after late-join/late-finish reshaping is unsafe.

    The marker is not permanent: once the batch shrinks back to one row with a
    verifiably compact cache, ``_maybe_clear_multirow_marker`` lifts it so the
    surviving request regains MTP for the rest of its generation.
    """
    try:
        if len(getattr(gen_batch, "uids", []) or []) > 1:
            gen_batch._omlx_mtp_saw_standard_multirow_decode = True
    except Exception:
        pass


def _log_multirow_mtp_inactive_once(gen_batch: Any) -> None:
    """Say once, at INFO, why an MTP-capable batch is decoding without MTP.

    #2150 showed the silent fallback is easy to misread: the benchmark's
    batched phases report plain continuous-batching numbers while every log
    line about MTP inactivity hides at DEBUG. One line per batch keeps the
    signal visible without per-step spam.
    """
    if getattr(gen_batch, "_omlx_mtp_inactive_logged", False):
        return
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) <= 1:
        return
    if not _mtp_common_eligible(gen_batch):
        return
    gen_batch._omlx_mtp_inactive_logged = True
    if not _rowwise_batch_mtp_enabled(gen_batch):
        logger.info(
            "MTP inactive for %d-row batch: standard batched decode is faster "
            "at this batch size (set %s=1 to force row-wise MTP)",
            len(uids),
            _ROWWISE_BATCH_MTP_ENV,
        )
    else:
        logger.info(
            "MTP inactive for %d-row batch: %s",
            len(uids),
            _ineligibility_reason(gen_batch) or "activation deferred",
        )


def _maybe_clear_multirow_marker(gen_batch: Any) -> None:
    """Re-enable singleton MTP once a shrunken batch is verifiably safe again.

    The multirow marker exists because a row surviving batch reshaping may sit
    in a cache whose layout singleton MTP's raw backbone calls don't expect
    (left padding). ``BatchKVCache.filter()`` shifts out the minimum shared
    left padding, so a batch filtered down to one row is compact again in the
    common case — verify that per layer instead of assuming, and only then
    lift the marker. Without this, a request that ever shared a decode step
    with another request is locked out of MTP for the rest of its generation
    even once it is running alone (#2150).
    """
    if not getattr(gen_batch, "_omlx_mtp_saw_standard_multirow_decode", False):
        return
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) != 1:
        return
    # CacheList layers (GLM 5.2 / DeepSeek v3.2 lineage) keep left padding on
    # their sub-caches, not on the container — recurse instead of skipping.
    pending = list(getattr(gen_batch, "prompt_cache", None) or [])
    while pending:
        cache = pending.pop()
        sub_caches = getattr(cache, "caches", None)
        if sub_caches is not None:
            pending.extend(sub_caches)
            continue
        left_padding = getattr(cache, "left_padding", None)
        if left_padding is None:
            continue
        try:
            if max(int(v) for v in left_padding.tolist()) > 0:
                return
        except Exception:
            return
    gen_batch._omlx_mtp_saw_standard_multirow_decode = False
    logger.info("MTP singleton recovery: multirow marker cleared (compact cache)")


def _is_mtp_eligible(gen_batch: Any) -> bool:
    """``__init__`` and ``next`` only engage MTP for single-sequence batches
    when the model exposes ``mtp_forward``, has an attached MTP head, and
    was loaded with MTP decode enabled.

    The MTP head may be attached unconditionally (e.g. by the mlx-vlm
    runtime patches, which need it for weight-load matching even when
    inference-time MTP is off) — so head presence alone is not enough
    to decide whether to run the draft/verify cycle. The per-instance
    ``_omlx_mtp_decode_enabled`` marker reflects the per-load
    ``model_settings.mtp_enabled`` choice without being affected by later
    model loads in the same process.
    """
    if not _mtp_common_eligible(gen_batch):
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) != 1:
        return False
    if not _allows_new_mtp_activation(gen_batch, "_omlx_mtp_state"):
        return False
    return True


def _is_mtp_batch_eligible(gen_batch: Any) -> bool:
    if not _mtp_common_eligible(gen_batch):
        return False
    model = getattr(gen_batch, "model", None)
    if getattr(model, "_omlx_mtp_rowwise_unsupported", False) or getattr(
        getattr(model, "_language_model", None),
        "_omlx_mtp_rowwise_unsupported",
        False,
    ):
        # Multi-block window heads (inkling) keep per-request cycle state
        # on the cache list; the row-wise extract/merge path does not
        # model that.
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) <= 1:
        return False
    if not _allows_new_mtp_activation(gen_batch, "_omlx_mtp_batch_state"):
        return False
    if getattr(
        gen_batch, "_omlx_mtp_batch_state", None
    ) is None and not _rowwise_batch_mtp_enabled(gen_batch):
        return False
    # No cache-position alignment requirement: activation seeds each row from
    # its own extract_cache(idx) view and steady-state row cycles diverge the
    # per-row offsets immediately anyway (accept counts differ per row), so
    # the merge path already handles ragged rows. Under continuous batching
    # rows join at different times, so requiring aligned offsets at
    # activation kept this path from ever engaging (#2150).
    return True


def _ineligibility_reason(gen_batch: Any) -> str:
    """Return a short human-readable reason for why the MTP path isn't active.

    Only used for debug logging — the patched_init / patched_next paths
    don't act on this string.
    """
    if not hasattr(gen_batch, "model"):
        return "GenerationBatch has no .model attribute"
    if not hasattr(gen_batch.model, "mtp_forward"):
        return (
            f"model {type(gen_batch.model).__module__}.{type(gen_batch.model).__name__} "
            "has no mtp_forward (qwen35 patch may not have applied to this class)"
        )
    if not _model_has_mtp_module(gen_batch.model):
        return "model has no attached mtp head"
    if not _model_mtp_decode_enabled(gen_batch.model):
        return (
            "model instance MTP decode flag is off "
            "(model_settings.mtp_enabled was False when this model was loaded)"
        )
    uids = getattr(gen_batch, "uids", None)
    if uids is None:
        return "GenerationBatch has no uids"
    if len(uids) != 1:
        if not _allows_new_mtp_activation(gen_batch, "_omlx_mtp_batch_state"):
            return "pending prompt work may still merge into this batch"
        if (
            _rowwise_batch_mtp_override() is None
            and len(uids) == 2
            and _is_qwen4_exp_model(getattr(gen_batch, "model", None))
            and _batch_wants_api_logprobs(gen_batch)
        ):
            return "row-wise batch MTP does not serve API logprobs requests"
        if getattr(
            gen_batch, "_omlx_mtp_batch_state", None
        ) is None and not _rowwise_batch_mtp_enabled(gen_batch):
            return (
                f"row-wise batch MTP is disabled by policy "
                f"(set {_ROWWISE_BATCH_MTP_ENV}=1 to force); "
                "generic models use standard batched decode"
            )
        return ""
    if not _allows_new_mtp_activation(gen_batch, "_omlx_mtp_state"):
        return "pending prompt work may still merge into this singleton batch"
    if _has_grammar_processors(gen_batch):
        return "grammar-constrained decoding uses GenerationBatch._step hooks"
    return ""


class _MtpStepFallback(RuntimeError):
    """Raised inside the MTP path to signal a clean fallback to the standard step."""


class _Qwen4SequentialRecoveredFallback(_MtpStepFallback):
    """The scalar oracle restored its base and may use the wide verifier."""


class _Qwen4SequentialHardFailure(RuntimeError):
    """The oracle detected an invalid contract that wide verify also shares."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class _MtpStats:
    """Acceptance / throughput counters for one MTP-active sequence.

    Logged at INFO when the sequence finishes (length / stop / filter)
    so the operator can see whether the draft+verify cycle is actually
    productive on this model + sampler combo.
    """

    cycles: int = 0  # number of verify cycles run
    accepts: int = 0  # accepted draft tokens (depth-k: sum over positions)
    physical_drafts: int = 0  # all drafts built, including post-rejection tail
    rejects: int = 0  # cycles that ended in a rejection
    init_emits: int = 0  # tokens emitted from the post-init queue (always 2)
    draft_emits: int = 0  # tokens emitted as accepted drafts
    bonus_emits: int = 0  # tokens emitted as bonus (accepted + emit_bonus)
    verify_emits: int = 0  # tokens emitted as verify-position correction (reject path)
    # Per-depth accept telemetry for chained drafting: drafted[j] counts
    # cycles where a draft existed at depth j; accepted[j] counts how many
    # of those were verified. Depth-1 legacy path fills index 0 only.
    depth_drafted: List[int] = field(default_factory=list)
    depth_accepted: List[int] = field(default_factory=list)
    # Cycles the depth controller parked at 0 (plain steps, no speculation).
    zero_cycles: int = 0
    # Component-level timings. Help diagnose where MTP overhead comes from
    # when accept rate is healthy but wall-clock throughput isn't.
    backbone_ms: float = 0.0  # cumulative time inside the 2-token verify forward
    mtp_head_ms: float = 0.0  # cumulative time inside MTP-head forwards
    sample_ms: float = 0.0  # cumulative time in sampling + acceptance check
    cache_ops_ms: float = 0.0  # cumulative time in trim / rollback restore


@dataclass
class _MtpState:
    """Per-batch MTP state stashed on the GenerationBatch instance."""

    # MTP state is valid only for this exact singleton uid. It must be dropped
    # across any standard batched step or batch reshape that breaks ownership.
    uid: Any = None

    # Pending tokens to emit in upcoming next() calls. Each entry is
    # (token_id_int, logprobs_1d, source_label). source_label is one of
    # "init", "draft", "bonus", "verify" — used to bucket stats correctly
    # when the queue is drained.
    queue: Deque[Tuple[int, Any, str]] = field(default_factory=deque)

    # Cache for the MTP head (separate from gen_batch.prompt_cache).
    mtp_cache: Optional[List[Any]] = None

    # First input token of the next verify forward. Tracked as a 1-element
    # mx.array (uint32) so it can be concatenated with `draft_tok` cheaply.
    next_main: Optional[Any] = None

    # Draft logprobs (vocab,) needed by stochastic acceptance / residual sampling.
    draft_tok: Optional[Any] = None  # (1,) uint32
    draft_lp: Optional[Any] = None  # (vocab,) float
    # Filtered (sampler-applied) draft logprobs reused by the next cycle's
    # acceptance ratio + residual sampling. Mirrors PR 990's accept_lp,
    # adapted to oMLX's callable-sampler contract via metadata-introspection.
    # None when the sampler exposes no metadata (raw-lp fallback path).
    draft_accept_lp: Optional[Any] = None  # (vocab,) float
    # Host-side int copy of draft_tok. Cached at draft creation time so the
    # verify cycle can compare draft vs verify ids without a separate
    # GPU→CPU sync (`int(draft_tok.tolist()[0])` would force a stall).
    draft_id: int = -1

    # --- depth-k chained drafting (Qwen3.5/3.6 only) ---
    # chain=True routes decode through _run_verify_cycle_chain; False keeps
    # the PR-990 depth-1 legacy cycle.
    chain: bool = False
    depth: int = 1
    # head_clone=True runs speculative head steps on a per-cycle cache clone
    # (models whose head cache can't be exactly trimmed once rotated).
    head_clone: bool = False
    # Pending draft tokens for the next verify forward: (depth,) uint32 array.
    # Host-side ids are read in the verify cycle's single sync, not here.
    drafts: Optional[Any] = None
    # Per-draft raw logprob rows (vocab,) — emitted as the accepted drafts'
    # logprobs (PR 990 contract) — and sampler-filtered rows for stochastic
    # acceptance. Lazy arrays; only evaluated if a consumer touches them.
    draft_lps: List[Any] = field(default_factory=list)
    draft_accept_lps: List[Any] = field(default_factory=list)
    # MTP-head cache offset at cycle start. Chain entries beyond this offset
    # are speculative and trimmed at commit; committed history is re-appended
    # from verify-forward hidden rows so the head sees a dense, committed-only
    # timeline.
    hist_offset: int = 0
    # Qwen4 suffix-local priming keeps the draft head on a local zero-based
    # suffix timeline while the verified target cache remains absolute. Never
    # use target_expected_offset to trim the head cache.
    target_expected_offset: Optional[int] = None
    suffix_local_priming: bool = False
    # Sampler for draft tokens (lazily resolved). For stochastic target
    # samplers this is a *sharper* distribution than the target (temp 0.6 /
    # top_p 0.95 / top_k 20) — the Leviathan/Chen acceptance ratio uses the
    # true draft distribution q, so any q keeps the output distribution
    # exact, and truncating the 1-layer head's noisy tail is what keeps
    # acceptance usable on high-entropy content (creative prose collapses to
    # ~10-20% with matched-temp drafts).
    draft_sampler: Optional[Any] = None
    # Adaptive depth controller (None = fixed depth). Chooses how many
    # drafts the next chain builds from rolling accept/latency estimates.
    controller: Optional[Any] = None

    # True while this state is a bounded re-entry probe after a performance
    # handoff. Correctness fallbacks and late-join handoffs do not set it.
    reentry_probe: bool = False

    # Accept-rate / throughput counters. Surfaced via logger.info on finish.
    stats: _MtpStats = field(default_factory=_MtpStats)

    # Qwen4 Lightning-MTP advances the target verifier by a whole draft
    # window, then returns those verified tokens to the scheduler one at a
    # time.  Until the scheduler has accepted each response (including its
    # parser and text-stop checks), the verifier update is a transaction, not
    # a cache commit.  Other model families retain their existing eager
    # behavior and leave this slot unset.
    pending_commit: Optional["_MtpPendingCommit"] = None
    # Queue position handed to the scheduler but not yet acknowledged through
    # the parser/text-stop post-emit hook: (position, token, source).
    pending_emit: Optional[Tuple[int, int, str]] = None
    # Adaptive depth may decide to park after the first response from a
    # verifier window.  Qwen4 cannot hand the target cache to standard decode
    # until every already-verified response has passed the scheduler.  Latch
    # that decision and execute it at the exact final ACK boundary.
    park_after_commit: bool = False


@dataclass(frozen=True)
class _Qwen4QSARollbackSnapshot:
    """Small structural proof for one QSA verifier update.

    The large K/V and raw-index arrays stay in their live cache.  Their
    logical offsets are sufficient because QSA rollback is suffix-only; this
    record proves that the same cache advanced by exactly ``verify_width``
    before a terminal prefix is selected.
    """

    cache: Any
    base_offset: int
    full_offset: int
    base_index_offset: int
    full_index_offset: int


@dataclass(frozen=True)
class _Qwen4SequentialRecurrentSnapshot:
    """Detached pre-cycle state for one Qwen4 GDN/PLE cache leaf."""

    cache: Any
    state: Tuple[Any, ...]
    token_count: Optional[int]
    metadata: Tuple[Tuple[str, Any], ...]


@dataclass(frozen=True)
class _Qwen4SequentialQSASnapshot:
    """Logical pre-cycle QSA suffix boundary without a context-sized clone."""

    cache: Any
    offset: int
    index_offset: int
    keys_backing: Any
    values_backing: Any
    index_keys_backing: Any
    index_positions_backing: Any
    private_index_backing: bool
    text_positions_qualified: bool
    pooled_keys: Any
    pooled_offset: int
    pooled_ratio: Any
    pooled_tag: Any
    has_pooled_state: bool
    index_capacity_managed: Any
    geometric_capacity_managed: Any


@dataclass(frozen=True)
class _Qwen4SequentialBaseSnapshot:
    """Exact live-target base retained until one scheduler transaction ends."""

    base_offset: int
    owner_uid: Any
    recurrent: Tuple[_Qwen4SequentialRecurrentSnapshot, ...]
    qsa: Tuple[_Qwen4SequentialQSASnapshot, ...]
    language_model: Any
    position_ids: Any
    rope_deltas: Any


@dataclass(frozen=True)
class _Qwen4SequentialVerifyResult:
    """Canonical scalar target rows and greedy decision for one cycle."""

    snapshot: _Qwen4SequentialBaseSnapshot
    target_input_ids: Tuple[int, ...]
    draft_ids: Tuple[int, ...]
    accepted: int
    emitted_id: int
    emitted_logprobs: Any
    combined_logprobs: Any
    hidden: Any
    processor_base_snapshot: Any
    token_buffer_base_size: Optional[int]
    processor_snapshots: Tuple[Any, ...]
    target_ids: Tuple[int, ...]


@dataclass
class _MtpPendingCommit:
    """One scheduler-visible Lightning-MTP target transaction.

    ``kind=\"init\"`` covers the separate two-token activation queue: its
    first token is already resident and its second token is the ordinary
    one-token pipeline tail.  ``kind=\"verify\"`` covers one depth-k target
    window.  A verify transaction keeps the original GDN/PLE rollback inputs
    alive and does not trim/clear the target until the scheduler reports
    whether the just-emitted queue position was terminal.

    The resident cache intentionally contains only target-backbone state.
    MTP-head state is discarded on terminal and reconstructed losslessly by
    Qwen4's verified suffix-local priming on a later cache hit.
    """

    kind: str
    target_base_offset: int
    head_base_offset: int
    verify_width: int
    accepted: int
    source_map: Tuple[str, ...]
    token_map: Tuple[int, ...]
    head_committed_offset: Optional[int] = None
    emitted: int = 0
    gdn_states: Optional[List[Any]] = None
    ple_snapshots: Tuple[Tuple[Any, Any], ...] = ()
    qsa_snapshots: Tuple[_Qwen4QSARollbackSnapshot, ...] = ()
    deferred_boundary: bool = False
    final_source: str = ""
    committed: bool = False
    sequential_base: Optional[_Qwen4SequentialBaseSnapshot] = None
    target_input_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class _MtpPostEmitResult:
    handled: bool = False
    exact_terminal: bool = False
    prompt_cache: Optional[List[Any]] = None
    all_tokens: Optional[List[int]] = None
    reason: str = "not-applicable"


@dataclass
class _MtpBatchState:
    """Experimental row-wise MTP state for a multi-sequence GenerationBatch."""

    states: Dict[Any, _MtpState] = field(default_factory=dict)


# The existing depth controller decides whether a re-entry probe wins. This
# policy state only schedules retries; it deliberately does not duplicate the
# controller's acceptance and cycle-cost model.
_MTP_REENTRY_INITIAL_COOLDOWN_TOKENS = 128
_MTP_REENTRY_MAX_COOLDOWN_TOKENS = 4096
_MTP_REENTRY_INITIAL_COOLDOWN_ENV = "OMLX_MTP_REENTRY_INITIAL_COOLDOWN_TOKENS"


def _mtp_reentry_initial_cooldown_tokens() -> int:
    """Resolve one request's initial re-entry cooldown, bounded and safe."""

    raw = os.environ.get(_MTP_REENTRY_INITIAL_COOLDOWN_ENV, "").strip()
    if not raw:
        return _MTP_REENTRY_INITIAL_COOLDOWN_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return _MTP_REENTRY_INITIAL_COOLDOWN_TOKENS
    return max(
        _MTP_REENTRY_INITIAL_COOLDOWN_TOKENS,
        min(_MTP_REENTRY_MAX_COOLDOWN_TOKENS, value),
    )


@dataclass
class _MtpParkState:
    """Per-sequence policy state for reversible performance parking.

    The MTP cache itself is dropped so the standard decoder regains its
    pipelined fast path. This host-side record survives the handoff and admits
    a later MTP probe. It is keyed by uid because GenerationBatch objects are
    reused across requests.
    """

    uid: Any
    cooldown_tokens: int = _MTP_REENTRY_INITIAL_COOLDOWN_TOKENS
    tokens_remaining: int = _MTP_REENTRY_INITIAL_COOLDOWN_TOKENS

    def observe_standard(self) -> None:
        self.tokens_remaining = max(0, self.tokens_remaining - 1)

    def probe_ready(self) -> bool:
        return self.tokens_remaining <= 0 and not _prefill_activity_recent()

    def restart_after_failed_probe(self) -> None:
        self.cooldown_tokens = min(
            _MTP_REENTRY_MAX_COOLDOWN_TOKENS,
            self.cooldown_tokens * 2,
        )
        self.tokens_remaining = self.cooldown_tokens

    def defer_probe(self) -> None:
        """Yield a probe for a batch-shape handoff without penalizing it."""
        self.tokens_remaining = 0


def _new_mtp_park_state(uid: Any) -> _MtpParkState:
    """Create request-owned park state from the current operator setting."""

    initial = _mtp_reentry_initial_cooldown_tokens()
    return _MtpParkState(
        uid=uid,
        cooldown_tokens=initial,
        tokens_remaining=initial,
    )


def _mtp_park_state_for_batch(gen_batch: Any) -> Optional[_MtpParkState]:
    state = getattr(gen_batch, "_omlx_mtp_park_state", None)
    if state is None:
        return None
    uids = getattr(gen_batch, "uids", None) or ()
    if state.uid in uids:
        return state
    try:
        delattr(gen_batch, "_omlx_mtp_park_state")
    except AttributeError:
        pass
    return None


def _record_parked_standard_step(gen_batch: Any) -> None:
    state = _mtp_park_state_for_batch(gen_batch)
    if state is not None:
        state.observe_standard()


def _maybe_finish_mtp_reentry_probe(
    gen_batch: Any,
    state: _MtpState,
    *,
    was_warmup: bool,
) -> bool:
    """Clear the cooldown once the existing controller measures a win."""
    controller = state.controller
    reentry_win_proven = getattr(controller, "reentry_win_proven", None)
    if (
        not state.reentry_probe
        or controller is None
        or was_warmup
        or controller._warmup
        or controller.exit_streak != 0
        or (
            callable(reentry_win_proven)
            and not reentry_win_proven()
        )
    ):
        return False
    try:
        delattr(gen_batch, "_omlx_mtp_park_state")
    except AttributeError:
        pass
    state.reentry_probe = False
    logger.info("MTP[%s] re-entry probe succeeded", state.uid)
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_sampler(gen_batch: Any):
    """Match ``GenerationBatch._step``'s per-sequence sampler resolution (batch=1)."""
    if gen_batch.samplers and gen_batch.samplers[0] is not None:
        return gen_batch.samplers[0]
    return gen_batch.fallback_sampler


def _is_greedy(gen_batch):
    sampler = _resolve_sampler(gen_batch)
    if sampler is not None:
        return getattr(sampler, "temp", 0.0) == 0.0
    return True


def _proc_list(gen_batch: Any) -> Optional[List[Any]]:
    if gen_batch.logits_processors and gen_batch.logits_processors[0]:
        return gen_batch.logits_processors[0]
    return None


def _has_grammar_processors(gen_batch: Any) -> bool:
    """True when MTP would bypass grammar state advanced by scheduler._step."""
    processors_by_seq = getattr(gen_batch, "logits_processors", None)
    if not processors_by_seq:
        return False
    try:
        from omlx.api.grammar import GrammarConstraintProcessor
    except Exception:
        return False
    return any(
        isinstance(proc, GrammarConstraintProcessor)
        for processors in processors_by_seq
        for proc in (processors or [])
    )


def _mtp_state_valid_for_batch(gen_batch: Any, state: Optional[_MtpState]) -> bool:
    """MTP state may only represent one uid in one current singleton slot."""
    if state is None:
        return False
    uids = getattr(gen_batch, "uids", None)
    return bool(uids is not None and len(uids) == 1 and uids[0] == state.uid)


def _drop_mtp_state(
    gen_batch: Any,
    reason: str,
    *,
    log_stats: bool = False,
) -> Optional[_MtpState]:
    """Delete attached MTP state, optionally surfacing stats for external finish.

    Deliberately does NOT drop the prompt-priming context: mlx-lm's insert
    flow routes every fresh singleton through ``extend()`` (donor merge),
    which drops MTP state defensively — but the priming context must survive
    that hop or activation always sees an unprimed cache. The context is
    released at activation (``take_primed``), on real multi-row merges
    (``patched_extend``), or with the cache itself at request end.
    """
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if state is None:
        return None
    if log_stats:
        try:
            _log_mtp_stats(
                getattr(state, "uid", "?"),
                state.stats,
                getattr(state, "_finish_reason", reason),
            )
        except Exception:
            pass
    try:
        delattr(gen_batch, "_omlx_mtp_state")
    except AttributeError:
        pass
    logger.debug("MTP state dropped: %s", reason)
    return state


def _drop_invalid_mtp_state(
    gen_batch: Any,
    reason: str,
    *,
    log_empty: bool = False,
) -> Optional[_MtpState]:
    """Drop state after a batch reshape unless ownership still matches."""
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if state is None:
        return None
    if _mtp_state_valid_for_batch(gen_batch, state):
        return state
    uids = getattr(gen_batch, "uids", None)
    return _drop_mtp_state(
        gen_batch,
        reason,
        log_stats=bool(log_empty and not uids),
    )


def _mtp_batch_state_valid_for_batch(
    gen_batch: Any, batch_state: Optional[_MtpBatchState]
) -> bool:
    if batch_state is None:
        return False
    uids = getattr(gen_batch, "uids", None)
    if not uids:
        return False
    return all(uid in batch_state.states for uid in uids)


def _drop_mtp_batch_state(
    gen_batch: Any,
    reason: str,
    *,
    log_stats: bool = False,
) -> Optional[_MtpBatchState]:
    batch_state = getattr(gen_batch, "_omlx_mtp_batch_state", None)
    if batch_state is None:
        return None
    if log_stats:
        for state in list(batch_state.states.values()):
            try:
                _log_mtp_stats(
                    getattr(state, "uid", "?"),
                    state.stats,
                    getattr(state, "_finish_reason", reason),
                )
            except Exception:
                pass
    try:
        delattr(gen_batch, "_omlx_mtp_batch_state")
    except AttributeError:
        pass
    logger.debug("MTP batch state dropped: %s", reason)
    return batch_state


def _drop_invalid_mtp_batch_state(
    gen_batch: Any,
    reason: str,
    *,
    old_uids: Optional[List[Any]] = None,
    log_empty: bool = False,
) -> Optional[_MtpBatchState]:
    batch_state = getattr(gen_batch, "_omlx_mtp_batch_state", None)
    if batch_state is None:
        return None
    uids = list(getattr(gen_batch, "uids", []) or [])
    if not uids:
        return _drop_mtp_batch_state(
            gen_batch,
            reason,
            log_stats=bool(log_empty),
        )

    keep = set(uids)
    removed = set(old_uids or []) - keep
    for uid in removed:
        state = batch_state.states.pop(uid, None)
        if state is not None and log_empty:
            try:
                _log_mtp_stats(uid, state.stats, reason)
            except Exception:
                pass
    batch_state.states = {
        uid: state for uid, state in batch_state.states.items() if uid in keep
    }
    if _mtp_batch_state_valid_for_batch(gen_batch, batch_state):
        if len(uids) == 1:
            gen_batch._omlx_mtp_state = batch_state.states[uids[0]]
            _drop_mtp_batch_state(gen_batch, "filter-to-singleton")
            return None
        return batch_state
    return _drop_mtp_batch_state(gen_batch, reason)


def _row_value(values: Optional[List[Any]], idx: int, default: Any = None) -> Any:
    if values is None:
        return default
    try:
        if len(values) == 0:
            return default
        return values[idx]
    except Exception:
        return default


def _make_row_batch(
    gen_batch: Any,
    idx: int,
    *,
    prompt_cache: Optional[List[Any]] = None,
    state: Optional[_MtpState] = None,
) -> Any:
    if prompt_cache is None:
        prompt_cache = gen_batch.extract_cache(idx)

    next_tokens = getattr(gen_batch, "_next_tokens", None)
    next_logprobs = getattr(gen_batch, "_next_logprobs", None)
    row = SimpleNamespace(
        model=gen_batch.model,
        uids=[gen_batch.uids[idx]],
        prompt_cache=prompt_cache,
        tokens=[gen_batch.tokens[idx]],
        samplers=[_row_value(getattr(gen_batch, "samplers", None), idx)],
        fallback_sampler=gen_batch.fallback_sampler,
        logits_processors=[
            _row_value(getattr(gen_batch, "logits_processors", None), idx, [])
        ],
        state_machines=[_row_value(getattr(gen_batch, "state_machines", None), idx)],
        max_tokens=[_row_value(getattr(gen_batch, "max_tokens", None), idx)],
        _next_tokens=next_tokens[idx : idx + 1] if next_tokens is not None else None,
        _next_logprobs=(
            [next_logprobs[idx]]
            if next_logprobs is not None and len(next_logprobs) > idx
            else []
        ),
        _token_context=[gen_batch._token_context[idx]],
        _num_tokens=[gen_batch._num_tokens[idx]],
        _matcher_states=[gen_batch._matcher_states[idx]],
        _omlx_rowwise_mtp=True,
    )
    if state is not None:
        row._omlx_mtp_state = state
    return row


def _qsa_singleton_alignment_error(prompt_cache: List[Any]) -> Optional[str]:
    """Return a diagnostic when a compact Qwen4 QSA row is not exact.

    Row-wise MTP deliberately runs every request against extracted singleton
    caches.  When a B2 batch shrinks to B1, the surviving state is promoted to
    the singleton MTP path as well.  Its cache must therefore be the same
    compact representation: leaving a one-row ``BatchQSAKVCache`` attached
    makes the next verify use a vector logical offset against a physical
    right-aligned extent.  At a ragged late-join boundary that produced the
    observed QSA mask/KV mismatch (four columns at depth eight).

    This check is intentionally structural.  It never trims or fabricates
    auxiliary indexer rows: the QSA raw keys and position IDs are part of the
    exact selector state, so an inconsistency must fail closed rather than be
    guessed back into alignment.
    """

    pending = list(prompt_cache)
    while pending:
        cache = pending.pop()
        pending.extend(getattr(cache, "caches", ()) or ())
        if type(cache).__name__ not in {"QSAKVCache", "QSAQuantizedKVCache"}:
            continue
        offset = getattr(cache, "offset", None)
        if not isinstance(offset, int):
            return f"{type(cache).__name__} retained non-scalar offset {offset!r}"
        index_keys = getattr(cache, "index_keys", None)
        index_positions = getattr(cache, "index_position_ids", None)
        if offset == 0 and index_keys is None and index_positions is None:
            continue
        if index_keys is None or index_positions is None:
            return f"{type(cache).__name__} is missing auxiliary indexer state"
        key_length = int(index_keys.shape[1])
        position_length = int(index_positions.shape[-1])
        if key_length != offset or position_length != offset:
            return (
                f"{type(cache).__name__} lengths disagree: kv={offset}, "
                f"index_keys={key_length}, positions={position_length}"
            )
    return None


def _prepare_compact_rowwise_mtp_survivor(
    gen_batch: Any,
    idx: int,
) -> List[Any]:
    """Detach and fully prove one prospective survivor before B2 mutation."""

    compact = gen_batch.extract_cache(idx)
    error = _qsa_singleton_alignment_error(compact)
    if error is None and _model_qwen4_terminal_commit_enabled(
        getattr(gen_batch, "model", None)
    ):
        error = _rebase_qwen4_extracted_recurrent_timeline(compact)
    if error is not None:
        raise ValueError(f"Qwen4 row-wise survivor cache is not exact: {error}")
    return compact


def _compact_rowwise_mtp_survivor(
    gen_batch: Any,
    *,
    compact: Optional[List[Any]] = None,
) -> None:
    """Move a B2->B1 row-wise survivor back to exact singleton caches.

    ``GenerationBatch.filter`` keeps the cache container batched even when a
    single row remains.  The row-wise MTP state, however, is promoted to
    ``_omlx_mtp_state`` and the next Qwen4 verify is a singleton operation.
    Extract the surviving logical row after filtering: this removes physical
    left padding, restores scalar offsets, and prevents another request's
    capacity/tail from leaking into QSA selection.  Extraction is also the
    ownership boundary used by normal finished responses, so arrays remain
    request-isolated.
    """

    uids = list(getattr(gen_batch, "uids", []) or [])
    if len(uids) != 1:
        return
    batch_state = getattr(gen_batch, "_omlx_mtp_batch_state", None)
    if batch_state is None or uids[0] not in batch_state.states:
        return
    if compact is None:
        compact = _prepare_compact_rowwise_mtp_survivor(gen_batch, 0)
    gen_batch.prompt_cache = compact


def _rebase_qwen4_extracted_recurrent_timeline(
    prompt_cache: List[Any],
) -> Optional[str]:
    """Repair only metadata lost by a proven ragged B2 -> B1 extraction.

    ``SizedArraysCache`` carries one scalar ``_token_count``. Its current
    ``merge`` therefore copies the first row's count even when independently
    advanced row-wise Qwen4 requests have different target epochs. Extracting
    the other row preserves that foreign scalar although the positionless GDN
    and PLE tensors themselves are correctly row-isolated.

    This seam is deliberately narrower than the ordinary timeline reconciler:
    it runs only after ``GenerationBatch.filter`` selected one row and QSA's KV,
    raw-index, and position histories proved a compact singleton. The uniform
    QSA offset is an independent absolute target epoch. Rebase only wrapper
    bookkeeping after proving every recurrent tensor is B1; never alter tensor
    state or accept a cache without a readable QSA timeline.
    """

    qsa_offsets: List[int] = []
    sized: List[Any] = []
    pending = list(prompt_cache)
    while pending:
        cache = pending.pop()
        pending.extend(getattr(cache, "caches", ()) or ())
        cache_name = type(cache).__name__
        if cache_name in _QWEN4_QSA_CACHE_TYPES:
            if cache_name not in {"QSAKVCache", "QSAQuantizedKVCache"}:
                return "Qwen4 extracted survivor retained a batched QSA cache"
            offsets = _qwen4_qsa_offsets(cache)
            if offsets is None or offsets[0] != offsets[1]:
                return "Qwen4 extracted survivor has invalid QSA offsets"
            qsa_offsets.append(offsets[0])
            continue
        if cache_name != "SizedArraysCache":
            continue
        token_count = getattr(cache, "_token_count", None)
        inner = vars(cache).get("_inner")
        state = getattr(inner, "state", None) if inner is not None else None
        if (
            type(token_count) is not int
            or type(inner).__name__ != "ArraysCache"
            or not isinstance(state, (list, tuple))
            or len(state) < 2
            or state[0] is None
            or state[1] is None
        ):
            return "Qwen4 extracted survivor has invalid recurrent metadata"
        for recurrent in state:
            if recurrent is None:
                continue
            if getattr(recurrent, "ndim", 0) < 1 or recurrent.shape[0] != 1:
                return "Qwen4 extracted survivor retained batched recurrent state"
        sized.append(cache)

    if not qsa_offsets:
        return "Qwen4 extracted survivor has no compact singleton QSA witness"
    expected = qsa_offsets[0]
    if any(offset != expected for offset in qsa_offsets[1:]):
        return "Qwen4 extracted survivor has mixed QSA target epochs"
    if _qwen4_target_offset(prompt_cache) != expected:
        return "Qwen4 extracted survivor target leaves disagree with QSA"

    for cache in sized:
        cache._token_count = int(expected)
    return None


def _merge_row_caches(row_caches: List[List[Any]]) -> List[Any]:
    if not row_caches:
        return []
    merged = []
    for layer_idx in range(len(row_caches[0])):
        per_row = [cache[layer_idx] for cache in row_caches]
        merge = getattr(per_row[0], "merge", None)
        if not callable(merge):
            raise _MtpStepFallback(
                f"cache {type(per_row[0]).__name__} cannot merge row caches"
            )
        merged.append(merge(per_row))
    return merged


def _replace_cache_rows(
    gen_batch: Any,
    replacements: Dict[int, List[Any]],
) -> None:
    if not replacements:
        return
    row_caches = [
        replacements.get(idx) or gen_batch.extract_cache(idx)
        for idx in range(len(gen_batch.uids))
    ]
    gen_batch.prompt_cache = _merge_row_caches(row_caches)


def _prepare_mtp_batch_state_for_next(gen_batch: Any) -> Optional[_MtpBatchState]:
    """Return a valid row-wise MTP state, lazily initializing every row."""
    batch_state = getattr(gen_batch, "_omlx_mtp_batch_state", None)
    if _mtp_batch_state_valid_for_batch(gen_batch, batch_state):
        return batch_state
    if batch_state is not None:
        _drop_mtp_batch_state(gen_batch, "stale-batch-owner")

    replacements: Dict[int, List[Any]] = {}
    token_context_updates: Dict[int, Any] = {}
    states: Dict[Any, _MtpState] = {}

    for idx, uid in enumerate(gen_batch.uids):
        row = _make_row_batch(gen_batch, idx)
        _set_singleton_mrope_delta(row)
        _post_init_mtp(row)
        state = getattr(row, "_omlx_mtp_state", None)
        if not _mtp_state_valid_for_batch(row, state):
            _drop_mtp_batch_state(gen_batch, "batch-post-init-invalid")
            return None
        states[uid] = state
        replacements[idx] = row.prompt_cache
        token_context_updates[idx] = row._token_context[0]

    _replace_cache_rows(gen_batch, replacements)
    for idx, token_context in token_context_updates.items():
        gen_batch._token_context[idx] = token_context

    batch_state = _MtpBatchState(states=states)
    gen_batch._omlx_mtp_batch_state = batch_state
    gen_batch._omlx_standard_target_exact_v1 = False
    logger.info(
        "MTP row-wise batch path activated for %d sequences",
        len(gen_batch.uids),
    )
    return batch_state


def _reconcile_mtp_batch_to_standard(gen_batch: Any) -> bool:
    batch_state = getattr(gen_batch, "_omlx_mtp_batch_state", None)
    if batch_state is None:
        return True
    if not getattr(gen_batch, "uids", None):
        return True

    import mlx.core as mx

    row_caches: Dict[int, List[Any]] = {}
    next_tokens = []
    next_logprobs = []
    token_context_updates: Dict[int, Any] = {}

    try:
        for idx, uid in enumerate(gen_batch.uids):
            state = batch_state.states.get(uid)
            if state is None:
                row_caches[idx] = gen_batch.extract_cache(idx)
                if getattr(gen_batch, "_next_tokens", None) is not None:
                    next_tokens.append(gen_batch._next_tokens[idx : idx + 1])
                if len(getattr(gen_batch, "_next_logprobs", [])) > idx:
                    next_logprobs.append(gen_batch._next_logprobs[idx])
                continue

            row = _make_row_batch(gen_batch, idx, state=state)
            if not _reconcile_mtp_to_standard(row, state):
                return False
            row_caches[idx] = row.prompt_cache
            next_tokens.append(row._next_tokens)
            next_logprobs.extend(row._next_logprobs)
            token_context_updates[idx] = row._token_context[0]

        if row_caches:
            _replace_cache_rows(gen_batch, row_caches)
        if next_tokens:
            gen_batch._next_tokens = mx.concatenate(next_tokens)
            gen_batch._next_logprobs = next_logprobs
        for idx, token_context in token_context_updates.items():
            gen_batch._token_context[idx] = token_context
        gen_batch._omlx_standard_target_exact_v1 = True
        return True
    except Exception as exc:
        logger.warning("MTP batch reconcile failed: %s", exc)
        return False


def _prepare_mtp_state_for_next(gen_batch: Any) -> Optional[_MtpState]:
    """Return a valid singleton MTP state, lazily initializing if needed."""
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if _mtp_state_valid_for_batch(gen_batch, state):
        return state
    if state is not None:
        _drop_mtp_state(gen_batch, "stale-owner")

    park_state = _mtp_park_state_for_batch(gen_batch)
    _set_singleton_mrope_delta(gen_batch)
    _post_init_mtp(gen_batch)
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if not _mtp_state_valid_for_batch(gen_batch, state):
        _drop_mtp_state(gen_batch, "post-init-invalid")
        return None

    # Eligibility already admitted this parked singleton. Mark the fresh
    # state unconditionally so a prefill arriving between the two checks
    # cannot strand the cooldown marker beside a normal MTP state.
    if park_state is not None:
        state.reentry_probe = True
        prepare_reentry_probe = getattr(
            state.controller,
            "prepare_reentry_probe",
            None,
        )
        if callable(prepare_reentry_probe):
            prepare_reentry_probe()
        logger.info(
            "MTP[%s] re-entry probe started after %d standard tokens",
            state.uid,
            park_state.cooldown_tokens,
        )

    if state.suffix_local_priming:
        logger.info(
            "MTP path activated for uid=%s (model has mtp_forward, batch=1, "
            "primed=%d local suffix tokens, target_offset=%d)",
            state.uid,
            max(0, int(state.hist_offset) - 1),
            int(state.target_expected_offset or 0),
        )
    else:
        logger.info(
            "MTP path activated for uid=%s "
            "(model has mtp_forward, batch=1, primed=%d)",
            state.uid,
            max(0, int(getattr(state, "hist_offset", 0)) - 1),
        )
    return state


def _set_singleton_mrope_delta(gen_batch: Any) -> None:
    """Mirror scheduler._step's per-uid mRoPE setup for direct MTP forwards."""
    model = getattr(gen_batch, "model", None)
    uids = getattr(gen_batch, "uids", None)
    if (
        model is not None
        and getattr(model, "_uses_mrope", False)
        and getattr(model, "_uid_rope_deltas", None)
        and uids
        and len(uids) == 1
        and hasattr(model, "set_batch_rope_deltas")
    ):
        import mlx.core as mx

        delta = model._uid_rope_deltas.get(uids[0], 0.0)
        model.set_batch_rope_deltas(mx.array([delta]))


def _rebuild_singleton_cache(model: Any) -> Optional[List[Any]]:
    """Build a fresh single-sequence batch-aware cache (left_padding=[0]).

    Reuses mlx-lm's own ``_make_cache`` so the per-layer types match exactly
    what ``extend()`` / ``_extend_cache`` expects, keeping the subsequent merge
    type-compatible. Returns None if the converter is unavailable.
    """
    import sys

    try:
        make_cache = sys.modules["mlx_lm.generate"]._make_cache
        return make_cache(model, [0], None)
    except Exception as exc:
        # Qwen4's vendored ArraysCache is intentionally not a subclass of
        # mlx-lm's class with the same name, so mlx-lm's isinstance dispatch
        # rejects it. Its own cache classes expose the exact singleton merge
        # contract used everywhere else in this row-wise path; use that narrow
        # route rather than making a failed emergency reconcile fatal.
        if _is_qwen4_exp_model(model):
            try:
                fresh = model.make_cache()
                from omlx.cache.type_handlers import SizedArraysCache

                fresh = [
                    (
                        SizedArraysCache(cache, token_count=0)
                        if type(cache).__name__ == "ArraysCache"
                        else cache
                    )
                    for cache in fresh
                ]
                rebuilt = _merge_row_caches([fresh])
                if rebuilt:
                    return rebuilt
            except Exception as vendor_exc:
                logger.warning(
                    "MTP reconcile: Qwen4 cache rebuild unavailable: %s "
                    "(mlx-lm path: %s)",
                    vendor_exc,
                    exc,
                )
                return None
        logger.warning("MTP reconcile: cache rebuild unavailable: %s", exc)
        return None


def _reconcile_mtp_to_standard(gen_batch: Any, state: _MtpState) -> bool:
    """Rewind a to-be-dropped MTP singleton into a standard-resumable state.

    The MTP path never maintains mlx-lm's ``_next_tokens`` — it streams tokens
    from ``state.queue`` and advances the shared cache speculatively, and the
    GatedDeltaNet rollback snapshot is cleared on accept, so a partial rollback
    at an arbitrary drop point is not reliable. Instead, rebuild the cache by
    re-prefilling exactly the already-streamed tokens (``gen_batch.tokens[0]``)
    into a fresh cache (which deterministically reconstructs every layer state,
    KV and SSM), then set ``_next_tokens`` to the correct next-to-emit token:

    - if ``state.queue`` is non-empty, ``queue[0]`` is the correct, not-yet-
      streamed next token — reuse it (and its logprobs). The rest of the queue
      is discarded; standard decode re-derives those positions.
    - otherwise (cycle boundary) sample from the re-prefill's last-position
      logits, exactly as a standard ``_step`` would after feeding ``tokens[-1]``.

    Leaves ``tokens[0]`` / ``_num_tokens[0]`` untouched (they already reflect
    streamed tokens), so there is no duplicated or skipped token. Returns False
    (caller falls back to a plain drop) when reconcile cannot be done safely.
    """
    import mlx.core as mx

    tokens = gen_batch.tokens[0] if getattr(gen_batch, "tokens", None) else None
    if not tokens:
        return False
    try:
        new_cache = _rebuild_singleton_cache(gen_batch.model)
        if new_cache is None:
            return False
        procs = _proc_list(gen_batch)
        _set_singleton_mrope_delta(gen_batch)
        tok_arr = _ensure_uint32(mx.array(list(tokens)))
        # Inherits the per-engine stream from the enclosing BatchGenerator context.
        logits, _, _ = _call_backbone(gen_batch.model, tok_arr[None, :], new_cache)
        last_logits = logits[:, -1, :]  # (1, vocab) — dist after tokens[-1]

        if state.queue:
            next_id, next_lp_1d, _src = state.queue[0]
            next_tok = mx.array([int(next_id)], dtype=mx.uint32)
            next_lp = next_lp_1d
        else:
            prev_buf = gen_batch._token_context[0].tokens if procs is not None else None
            ll = _apply_processors(procs, prev_buf, last_logits)
            next_lp_2d = _logprobs(ll)
            next_tok = _ensure_uint32(_resolve_sampler(gen_batch)(next_lp_2d))
            next_lp = next_lp_2d.squeeze(0)

        mx.eval(next_tok)
        # Reconciliation produces committed standard-decoding state. A long
        # re-prefill is still an armed MTP-managed backbone call, so discard
        # its speculative snapshots before exposing or merging the cache.
        _clear_rollback(new_cache)
        if _model_qwen4_terminal_commit_enabled(gen_batch.model):
            expected = len(tokens)
            if (
                _qwen4_target_offset(new_cache) != expected
                or not _qwen4_reconcile_sized_recurrent_timeline(
                    new_cache,
                    expected=expected,
                    allowed_current={expected},
                )
            ):
                return False
        gen_batch.prompt_cache = new_cache
        gen_batch._next_tokens = next_tok
        gen_batch._next_logprobs = [next_lp]
        gen_batch._omlx_standard_target_exact_v1 = True
        if procs is not None:
            from mlx_lm.models.cache import TokenBuffer

            gen_batch._token_context[0] = TokenBuffer(list(tokens))
        logger.debug(
            "MTP reconciled to standard on reshape (uid=%s tokens=%d queue=%d)",
            getattr(state, "uid", "?"),
            len(tokens),
            len(state.queue),
        )
        return True
    except Exception as exc:
        logger.warning("MTP reconcile failed, falling back to plain drop: %s", exc)
        return False


def _apply_processors(processors, prev_tokens, logits_2d):
    if not processors:
        return logits_2d
    for proc in processors:
        logits_2d = proc(prev_tokens, logits_2d)
    return logits_2d


def _snap_snapshotable(procs):
    """Checkpoint state of processors exposing ``snapshot_state`` (budget).

    Returns ``None`` when no processor supports position-keyed rewind, so
    callers can skip the restore unconditionally (the DSpark path applies
    processors to speculative draft positions; only the budget processor
    tracks position-sensitive mutable state today).
    """
    if not procs:
        return None
    snaps = [p.snapshot_state() for p in procs if hasattr(p, "snapshot_state")]
    return snaps or None


def _restore_snapshotable(procs, snaps) -> None:
    """Rewind processors previously checkpointed by :func:`_snap_snapshotable`."""
    if not procs or not snaps:
        return
    it = iter(snaps)
    for p in procs:
        if hasattr(p, "restore_state"):
            try:
                p.restore_state(next(it))
            except StopIteration:
                return


def _snapshot_qwen4_sequential_processors(procs) -> Tuple[Any, ...]:
    """Capture stateful processors with stable identity/index alignment."""

    if not procs:
        return ()
    snapshots = []
    for index, proc in enumerate(procs):
        has_snapshot = callable(getattr(proc, "snapshot_state", None))
        has_restore = callable(getattr(proc, "restore_state", None))
        if has_snapshot != has_restore:
            raise _MtpStepFallback(
                "Qwen4 sequential processor cannot round-trip state"
            )
        if has_snapshot:
            snapshots.append((index, proc, proc.snapshot_state()))
        elif getattr(proc, "__dict__", None):
            # Unknown mutable processors are not safe to replay speculatively.
            raise _MtpStepFallback(
                "Qwen4 sequential processor has untracked mutable state"
            )
    return tuple(snapshots)


def _restore_qwen4_sequential_processors(procs, snapshots) -> bool:
    """Restore only the exact processor objects captured above."""

    if not snapshots:
        return True
    if not procs:
        return False
    try:
        for index, proc, snapshot in snapshots:
            if index >= len(procs) or procs[index] is not proc:
                return False
            restore = getattr(proc, "restore_state", None)
            if not callable(restore):
                return False
            restore(snapshot)
        return True
    except Exception:
        return False


def _logprobs(logits_2d):
    import mlx.core as mx

    return logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)


def _mtp_vocab_coordinator(gen_batch: Any) -> Optional[Any]:
    """Return the runtime's validated distributed-vocabulary protocol.

    The cluster runtime installs this only while a model has matching local
    vocabulary shards for every MTP projection. Keeping the dependency as a
    duck-typed model capability leaves the regular and single-device MTP paths
    completely unchanged.
    """

    model = gen_batch.model
    for candidate in (
        model,
        getattr(model, "language_model", None),
        getattr(model, "_language_model", None),
    ):
        adapter = getattr(candidate, "_omlx_mtp_vocab_coordinator", None)
        if adapter is not None:
            return adapter
    return None


def _mtp_prepare_logits(gen_batch: Any, local_logits: Any) -> Any:
    """Reconstruct a local vocab shard on rank zero, not on every rank."""

    adapter = _mtp_vocab_coordinator(gen_batch)
    if adapter is None:
        return local_logits
    full = adapter.gather_logits(local_logits)
    if full is not None:
        return full

    # Worker response objects still index emitted-token logprobs. A broadcast
    # scalar preserves the global shape without allocating a full vocabulary
    # row or running a redundant normalization on the worker GPU.
    import mlx.core as mx

    return mx.broadcast_to(
        mx.zeros((), dtype=local_logits.dtype),
        (*local_logits.shape[:-1], int(adapter.output_size)),
    )


def _mtp_logprobs(gen_batch: Any, logits: Any) -> Any:
    adapter = _mtp_vocab_coordinator(gen_batch)
    if (
        adapter is not None
        and not adapter.is_coordinator
        and not getattr(adapter, "replicated_logits", False)
    ):
        import mlx.core as mx

        return mx.broadcast_to(mx.zeros((), dtype=mx.float32), logits.shape)
    return _logprobs(logits)


def _mtp_accept_lp(gen_batch: Any, sampler: Any, logprobs: Any) -> Any:
    adapter = _mtp_vocab_coordinator(gen_batch)
    if (
        adapter is not None
        and not adapter.is_coordinator
        and not getattr(adapter, "replicated_logits", False)
    ):
        return logprobs
    return _accept_lp_for(sampler, logprobs)


def _mtp_sample(gen_batch: Any, sampler: Any, logprobs: Any) -> Any:
    """Sample on rank zero and synchronize only the fixed-width token IDs."""

    adapter = _mtp_vocab_coordinator(gen_batch)
    if adapter is None:
        return sampler(logprobs)
    proposal = sampler(logprobs) if adapter.is_coordinator else None
    return adapter.sync_tokens(proposal, tuple(logprobs.shape[:-1]))


def _mtp_sync_packet(gen_batch: Any, packet: Any, length: int) -> list[int]:
    """Broadcast a small coordinator decision packet and materialize it once."""

    adapter = _mtp_vocab_coordinator(gen_batch)
    if adapter is None:
        return packet.tolist()
    return adapter.sync_packet(packet, length)


def _mtp_sync_depth(gen_batch: Any, depth: int) -> int:
    """Keep adaptive depth identical even when heterogeneous ranks time differently."""

    adapter = _mtp_vocab_coordinator(gen_batch)
    if adapter is None:
        return depth
    return adapter.sync_scalar(depth)


def _mtp_sync_flag(gen_batch: Any, value: bool) -> bool:
    adapter = _mtp_vocab_coordinator(gen_batch)
    if adapter is None:
        return value
    return bool(adapter.sync_scalar(int(value)))


def _accept_lp_for(sampler, lp):
    """Reproduce the sampler's filter+temperature pipeline on `lp` so the
    acceptance ratio (and residual distribution) match the distribution the
    sampler actually drew from.

    Reads sampling params off the callable as function attributes (set by
    ``omlx.utils.sampling.make_sampler``). For samplers without metadata —
    e.g. mlx-lm stock callables, fallback samplers — returns `lp` unchanged
    so behavior matches the pre-PR-990 raw-lp acceptance.
    """
    import mlx.core as mx

    from omlx.utils.sampling import apply_min_p, apply_top_k, apply_top_p

    temp = float(getattr(sampler, "temp", 0.0) or 0.0)
    if temp == 0.0:
        # Greedy / unknown sampler — raw lp is the acceptance distribution.
        return lp

    out = lp
    top_p = float(getattr(sampler, "top_p", 0.0) or 0.0)
    if 0.0 < top_p < 1.0:
        out = apply_top_p(out, top_p)
    min_p = float(getattr(sampler, "min_p", 0.0) or 0.0)
    if min_p != 0.0:
        min_keep = int(getattr(sampler, "min_tokens_to_keep", 1) or 1)
        out = apply_min_p(out, min_p, min_keep)
    top_k = int(getattr(sampler, "top_k", 0) or 0)
    if top_k > 0:
        out = apply_top_k(out, top_k)

    # Temperature scale + renormalize so the output is a proper logprob
    # distribution that can be indexed by token id for the acceptance check.
    scaled = out * (1.0 / temp)
    return scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


def _trim_token_buffer(gen_batch: Any, n: int) -> None:
    """Shrink ``_token_context[0]`` by ``n`` (mirrors PR 990 ``prev[:-n]``)."""
    if n <= 0:
        return
    procs = _proc_list(gen_batch)
    if procs is None:
        return
    buf = gen_batch._token_context[0]
    buf._size = max(0, buf._size - n)


def _restore_or_trim_caches(prompt_cache: List[Any]) -> bool:
    """Roll back one token from each layer cache after a draft rejection.

    SSM / linear-attention layers expose ``rollback_state`` populated by the
    patched ``GatedDeltaNet.__call__``; we restore that snapshot. Standard
    KV cache layers (full-attention) expose ``trim`` and ``is_trimmable``;
    we trim by 1. Layers that support neither cause the entire MTP step to
    fall back to the standard path.

    All layers are checked before anything is mutated: a partial rollback
    (early layers trimmed, a later layer refusing) leaves per-layer KV
    lengths desynchronised by one position and corrupts every subsequent
    forward (the shared attention mask is built from the first layer's
    cache, so the mismatch surfaces as a broadcast error on DeepSeek-V4
    compressed-attention layers).
    """
    for c in prompt_cache:
        if getattr(c, "rollback_state", None) is not None:
            # A draft stash marks the chain-path *pre-forward* snapshot
            # semantics (qwen35_model unsplit verify); restoring it here
            # would drop the confirmed token too. Only mtp_partial_rollback
            # knows how to replay it — refuse so the caller falls back.
            if getattr(c, "_mtp_draft_stash", None) is not None:
                return False
            continue
        if hasattr(c, "is_trimmable") and c.is_trimmable():
            continue
        return False
    for c in prompt_cache:
        rollback = getattr(c, "rollback_state", None)
        if rollback is not None:
            conv_snap, ssm_snap = rollback
            c[0] = conv_snap
            c[1] = ssm_snap
            c.rollback_state = None
            continue
        c.trim(1)
    return True


def _rollback_after_reject(
    model: Any,
    prompt_cache: List[Any],
    gdn_states: Optional[list],
    accepted: int = 0,
    block_size: int = 2,
) -> bool:
    """Roll back per-layer cache state after a rejected MTP draft token.

    Two mechanisms are supported, dispatched on the model's capability:

    1. **mlx-vlm path** — when the model exposes ``rollback_speculative_cache``
       (Qwen3.5 LanguageModel ships with it upstream) AND ``gdn_states`` is
       populated, we delegate to that method. It batches the per-layer SSM
       replay into a single ``gated_delta_update`` call and trims KV
       caches by ``block_size - (accepted + 1)``. The backbone forward was
       run with both confirmed and draft tokens; the rollback replays only
       the accepted prefix through the original pre-update state.

    2. **mlx-lm model contract** — Qwen and other depth-k-capable patches
       expose ``mtp_partial_rollback`` so linear-attention state can replay
       the confirmed prefix after a rejection. This is required even for the
       synchronous depth-one protocol because those caches carry projected
       draft inputs that generic trimming cannot safely interpret.

    3. **generic mlx-lm path** — per-layer ``cache.rollback_state`` snapshot
       written during the confirmed/draft split. We restore the snapshot for
       SSM layers and trim KV layers by 1. ``gdn_states`` is None here.

    Returns True on success. False means a cache layer in the list supports
    neither mechanism, in which case the caller falls back to the standard
    non-MTP step.
    """
    if gdn_states is not None and hasattr(model, "rollback_speculative_cache"):
        model.rollback_speculative_cache(prompt_cache, gdn_states, accepted, block_size)
        return True
    partial_rollback = getattr(model, "mtp_partial_rollback", None)
    if callable(partial_rollback) and not _model_mtp_tokenwise_verify_enabled(model):
        try:
            return bool(
                partial_rollback(
                    prompt_cache,
                    accepted,
                    max(0, int(block_size) - 1),
                )
            )
        except Exception as exc:
            logger.debug("mtp_partial_rollback failed: %s", exc)
            return False
    return _restore_or_trim_caches(prompt_cache)


def _rollback_and_replay_confirmed(
    gen_batch: Any,
    state: _MtpState,
    *,
    verify_width: int,
) -> bool:
    """Restore the pre-window state, then commit one ordinary decode row.

    Qwen's recurrent verify path projects the confirmed and draft rows as one
    skinny matrix. Replaying the confirmed row from those width-2 projections
    is close but not bit-identical to ordinary L=1 decode. On a rejection,
    restore the exact pre-window recurrent state, trim every KV row from the
    window, and feed ``next_main`` once through the regular decode shape.
    Accepted windows pay none of this cost.
    """

    import mlx.core as mx

    if state.next_main is None or verify_width <= 0:
        return False
    prompt_cache = gen_batch.prompt_cache
    for cache in prompt_cache:
        if getattr(cache, "rollback_state", None) is not None:
            continue
        if hasattr(cache, "is_trimmable") and cache.is_trimmable():
            continue
        return False

    try:
        for cache in prompt_cache:
            rollback = getattr(cache, "rollback_state", None)
            if rollback is not None:
                cache[0], cache[1] = rollback
                cache.rollback_state = None
                if getattr(cache, "_mtp_draft_stash", None) is not None:
                    cache._mtp_draft_stash = None
            else:
                cache.trim(verify_width)

        logits, _, _ = _call_backbone(
            gen_batch.model,
            state.next_main[:, None],
            prompt_cache,
        )
        mx.eval(logits, [cache.state for cache in prompt_cache])
        _clear_rollback(prompt_cache)
        return True
    except Exception as exc:
        logger.debug("MTP confirmed-row replay failed: %s", exc)
        return False


def _call_backbone(
    model: Any,
    inputs: Any,
    cache: List[Any],
    n_confirmed: int = 0,
) -> Tuple[Any, Any, Optional[list]]:
    """Run the backbone with ``return_hidden=True`` and normalise the result.

    Returns ``(logits, hidden_pre_norm, gdn_states_or_None)``:

    - mlx-lm path returns the 2-tuple ``(logits, hidden)``; ``gdn_states``
      is ``None`` and rollback uses ``cache.rollback_state``.
    - mlx-vlm path returns a ``LanguageModelOutput`` or 3-tuple
      ``(logits, hidden, gdn_states)`` so a rejected draft can be rolled
      back via ``rollback_speculative_cache``.

    ``n_confirmed`` is forwarded so the mlx-lm path can split its
    GatedDeltaNet forward into confirmed and draft chunks. mlx-vlm
    discards it (irrelevant — rollback is post-hoc, not splitwise).

    The rotating-cache undo stash (cache_rollback) is armed for the
    duration of the forward so a rejected draft can be rolled back even on
    a rotated RotatingKVCache; non-MTP forwards keep stock trim semantics.
    """
    kwargs = {"cache": cache, "return_hidden": True}
    if n_confirmed:
        kwargs["n_confirmed"] = n_confirmed
    dspark_verify = bool(n_confirmed and _dspark_host(model) is not None)
    _rollback_mod.set_undo_armed(True)
    # The affine verify qmm kernel is a Qwen-specific optimization. Keep the
    # DeepSeek target on its architecture-native quantized linear path.
    _set_verify_qmm_armed(not dspark_verify)
    _set_dspark_target_verify(model, dspark_verify)
    try:
        result = model(inputs, **kwargs)
    finally:
        if dspark_verify:
            _set_dspark_target_verify(model, False)
        _set_verify_qmm_armed(False)
        _rollback_mod.set_undo_armed(False)

    # LanguageModelOutput (mlx-vlm dataclass)
    if hasattr(result, "logits") and hasattr(result, "hidden_states"):
        hidden = result.hidden_states
        if isinstance(hidden, list):
            hidden = hidden[-1] if hidden else None
        return result.logits, hidden, getattr(result, "gdn_states", None)
    if isinstance(result, tuple):
        if len(result) == 3:
            return result
        if len(result) == 2:
            return result[0], result[1], None
    raise TypeError(f"backbone returned unexpected shape: {type(result).__name__}")


def _clear_rollback(prompt_cache: List[Any]) -> None:
    """Drop rollback snapshots after a draft is accepted."""
    pending = list(prompt_cache)
    while pending:
        c = pending.pop()
        pending.extend(getattr(c, "caches", ()))
        if hasattr(c, "rollback_state") and c.rollback_state is not None:
            c.rollback_state = None
        if getattr(c, "_mtp_draft_stash", None) is not None:
            c._mtp_draft_stash = None
        if getattr(c, "_mtp_undo", None) is not None:
            c._mtp_undo = None
        if getattr(c, "_qwen4_exp_ple_speculative_state", None) is not None:
            c._qwen4_exp_ple_speculative_state = None
        if getattr(c, "_undo", None) is not None:
            c._undo = None
            c._undo_chain = False


def _generic_mtp_terminal_cache_is_exact(gen_batch: Any) -> bool:
    """Prove a generic-MTP target cache already matches visible output.

    The first/bonus token of several Qwen3.5 MTP queue shapes is materialized
    in the target backbone before it is emitted.  When that token also ends the
    request, replaying the whole committed ledger is unnecessary and delays the
    terminal (sometimes first) stream event.  Accept the live cache only when
    every readable target leaf agrees with the public token count and no
    speculative rollback/undo payload remains.  Unknown cache shapes fail
    closed to the existing full standard reconciliation.
    """

    if _model_qwen4_terminal_commit_enabled(getattr(gen_batch, "model", None)):
        return False
    tokens = getattr(gen_batch, "tokens", None)
    prompt_cache = getattr(gen_batch, "prompt_cache", None)
    if (
        not isinstance(tokens, list)
        or len(tokens) != 1
        or not isinstance(tokens[0], list)
        or not tokens[0]
        or not isinstance(prompt_cache, list)
        or not prompt_cache
    ):
        return False
    if _prompt_priming.target_cache_offset(prompt_cache) != len(tokens[0]):
        return False
    for cache in _iter_mtp_cache_leaves(prompt_cache):
        for marker in (
            "rollback_state",
            "_mtp_draft_stash",
            "_mtp_undo",
            "_qwen4_exp_ple_speculative_state",
            "_undo",
        ):
            if getattr(cache, marker, None) is not None:
                return False
    return True


def _iter_mtp_cache_leaves(cache_list: List[Any]):
    pending = list(reversed(cache_list))
    while pending:
        cache = pending.pop()
        children = getattr(cache, "caches", None)
        if isinstance(children, (list, tuple)):
            pending.extend(reversed(children))
        else:
            yield cache


def _qwen4_reconcile_sized_recurrent_timeline(
    prompt_cache: List[Any],
    *,
    expected: int,
    allowed_current: set[int],
) -> bool:
    """Commit only proven Qwen4 SizedArraysCache timeline metadata.

    ``SizedArraysCache._token_count`` is bookkeeping around positionless
    recurrent state.  Qwen4's rollback restores GDN/PLE tensors exactly, but
    the wrapper is advanced by the full verify width and cannot infer the
    shorter accepted prefix.  Reconcile it only after the independent QSA
    epoch and GDN/PLE transaction succeeded, and only when every wrapper still
    reports either that exact full-window epoch or the already-correct target.
    Unknown wrappers, missing recurrent tensors, mixed epochs, and B>1 state
    fail closed before any metadata is changed.
    """

    if expected < 0 or not allowed_current:
        return False
    pending: List[Any] = []
    for cache in _iter_mtp_cache_leaves(prompt_cache):
        if type(cache).__name__ != "SizedArraysCache":
            continue
        token_count = getattr(cache, "_token_count", None)
        inner = vars(cache).get("_inner")
        state = getattr(inner, "state", None) if inner is not None else None
        if (
            type(token_count) is not int
            or token_count not in allowed_current
            or type(inner).__name__ != "ArraysCache"
            or not isinstance(state, (list, tuple))
            or len(state) < 2
            or state[0] is None
            or state[1] is None
        ):
            return False
        for recurrent in state[:2]:
            if getattr(recurrent, "ndim", 0) < 1 or recurrent.shape[0] != 1:
                return False
        pending.append(cache)

    # Raw/cold ArraysCache has no wrapper metadata to reconcile.
    for cache in pending:
        cache._token_count = int(expected)
    return True


def _qwen4_target_offset(prompt_cache: List[Any]) -> Optional[int]:
    """Return a uniform absolute Qwen4 target offset or fail closed."""

    offset = _prompt_priming.target_cache_offset(prompt_cache)
    if type(offset) is not int or offset < 1:
        return None
    return offset


_QWEN4_QSA_CACHE_TYPES = frozenset(
    {"QSAKVCache", "QSAQuantizedKVCache", "BatchQSAKVCache"}
)


def _qwen4_scalar_offset(value: Any) -> Optional[int]:
    """Read an int or a B1 size-one MLX offset without accepting B2 state."""

    if type(value) is int:
        return value
    if value is not None and getattr(value, "size", 0) == 1:
        try:
            return int(value.reshape(()).item())
        except Exception:
            return None
    return None


def _qwen4_qsa_offsets(cache: Any) -> Optional[Tuple[int, int]]:
    """Return exact B1 KV/index offsets for singleton or batch QSA caches."""

    if type(cache).__name__ not in _QWEN4_QSA_CACHE_TYPES:
        return None
    offset = _qwen4_scalar_offset(getattr(cache, "offset", None))
    index_offset = _qwen4_scalar_offset(
        getattr(
            cache,
            "_index_offset",
            getattr(cache, "index_offset", None),
        )
    )
    if offset is None or index_offset is None:
        return None
    left_padding = getattr(cache, "left_padding", None)
    if left_padding is not None:
        try:
            if getattr(left_padding, "size", 0) != 1:
                return None
            if int(left_padding.reshape(()).item()) != 0:
                return None
        except Exception:
            return None
    return offset, index_offset


def _qwen4_language_model(model: Any) -> Any:
    """Resolve the stateful Qwen4 LanguageModel through serving wrappers."""

    pending = [model]
    seen = set()
    while pending:
        candidate = pending.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        inner = getattr(candidate, "model", None)
        if (
            hasattr(candidate, "_position_ids")
            and hasattr(candidate, "_rope_deltas")
            and inner is not None
            and type(inner).__name__ == "Qwen4ExpModel"
        ):
            return candidate
        for name in ("language_model", "_language_model", "model"):
            child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                pending.append(child)
    return None


def _start_qwen4_ple_window_prefetch(
    language_model: Any,
    prompt_cache: List[Any],
    target_input_ids: Tuple[int, ...],
) -> Tuple[Any, Any] | None:
    """Prepare and scope one best-effort immutable PLE lookup window."""

    if (
        not _qwen4_ple_window_prefetch_enabled()
        or not 2 <= len(target_input_ids) <= 6
    ):
        return None
    prepare = getattr(language_model, "prepare_ple_window_prefetch", None)
    begin = getattr(language_model, "begin_ple_window_prefetch", None)
    if not all(
        callable(method)
        for method in (
            prepare,
            begin,
            getattr(language_model, "activate_ple_prefetch_row", None),
            getattr(language_model, "finish_ple_prefetch_row", None),
            getattr(language_model, "end_ple_window_prefetch", None),
        )
    ):
        return None
    try:
        payload = prepare(prompt_cache, target_input_ids)
        if payload is None:
            return None
        return payload, begin(payload)
    except Exception as exc:
        logger.debug("Qwen4 PLE window prefetch setup failed closed: %s", exc)
        return None


def _qwen4_detach_snapshot_value(value: Any, arrays: List[Any]) -> Any:
    """Detach MLX arrays recursively and collect them for one eager barrier."""

    import mlx.core as mx

    if isinstance(value, mx.array):
        try:
            detached = mx.copy(value)
        except AttributeError:
            detached = value + mx.zeros((), dtype=value.dtype)
        arrays.append(detached)
        return detached
    if isinstance(value, list):
        return [_qwen4_detach_snapshot_value(item, arrays) for item in value]
    if isinstance(value, tuple):
        return tuple(_qwen4_detach_snapshot_value(item, arrays) for item in value)
    if isinstance(value, dict):
        return {
            key: _qwen4_detach_snapshot_value(item, arrays)
            for key, item in value.items()
        }
    return value


def _capture_qwen4_sequential_base(
    gen_batch: Any,
    *,
    base_offset: int,
) -> _Qwen4SequentialBaseSnapshot:
    """Capture recurrent tensors plus logical QSA/model-position state.

    QSA K/V and raw-index buffers are intentionally not copied: scalar target
    calls only append a suffix, and every supported QSA cache owns an exact
    trim operation over both logical timelines.  Recurrent GDN/PLE tensors are
    positionless and cannot be trimmed, so those relatively small states are
    detached eagerly.
    """

    import mlx.core as mx

    if _qwen4_target_offset(gen_batch.prompt_cache) != base_offset:
        raise _MtpStepFallback("Qwen4 sequential base offset changed")
    uids = tuple(getattr(gen_batch, "uids", ()) or ())
    if len(uids) != 1:
        raise _MtpStepFallback("Qwen4 sequential base has no exclusive B1 owner")
    language_model = _qwen4_language_model(gen_batch.model)
    if language_model is None:
        raise _MtpStepFallback("Qwen4 sequential LanguageModel is unavailable")

    arrays: List[Any] = []
    recurrent: List[_Qwen4SequentialRecurrentSnapshot] = []
    qsa: List[_Qwen4SequentialQSASnapshot] = []
    for cache in _iter_mtp_cache_leaves(gen_batch.prompt_cache):
        class_name = type(cache).__name__
        if any(
            getattr(cache, name, None) is not None
            for name in (
                "rollback_state",
                "_qwen4_exp_ple_speculative_state",
                "_mtp_draft_stash",
                "_mtp_undo",
                "_undo",
            )
        ):
            raise _MtpStepFallback(
                "Qwen4 sequential base has stale rollback state"
            )
        if class_name in _QWEN4_QSA_CACHE_TYPES:
            if class_name == "BatchQSAKVCache":
                raise _MtpStepFallback(
                    "Qwen4 sequential oracle requires compact singleton QSA"
                )
            offsets = _qwen4_qsa_offsets(cache)
            if offsets != (base_offset, base_offset):
                raise _MtpStepFallback(
                    "Qwen4 sequential QSA base offsets are not exact"
                )
            logical_keys = getattr(cache, "index_keys", None)
            logical_positions = getattr(cache, "index_position_ids", None)
            if (
                logical_keys is None
                or logical_positions is None
                or int(logical_keys.shape[1]) != base_offset
                or int(logical_positions.shape[-1]) != base_offset
            ):
                raise _MtpStepFallback(
                    "Qwen4 sequential QSA index state is incomplete"
                )
            qualified = getattr(
                cache,
                "_omlx_text_position_ids_qualified",
                False,
            )
            if type(qualified) is not bool:
                raise _MtpStepFallback(
                    "Qwen4 sequential QSA text qualification is malformed"
                )
            has_pooled = hasattr(cache, "_pooled_index_offset")
            private_index = hasattr(cache, "_index_keys") and hasattr(
                cache,
                "_index_position_ids",
            )
            keys_backing = getattr(cache, "keys", None)
            values_backing = getattr(cache, "values", None)
            key_array = (
                keys_backing[0]
                if isinstance(keys_backing, (list, tuple)) and keys_backing
                else keys_backing
            )
            value_array = (
                values_backing[0]
                if isinstance(values_backing, (list, tuple)) and values_backing
                else values_backing
            )
            index_keys_backing = (
                getattr(cache, "_index_keys", None)
                if private_index
                else logical_keys
            )
            index_positions_backing = (
                getattr(cache, "_index_position_ids", None)
                if private_index
                else logical_positions
            )
            if (
                key_array is None
                or value_array is None
                or int(key_array.shape[2]) < base_offset
                or int(value_array.shape[2]) < base_offset
                or index_keys_backing is None
                or index_positions_backing is None
                or int(index_keys_backing.shape[1]) < base_offset
                or int(index_positions_backing.shape[-1]) < base_offset
            ):
                raise _MtpStepFallback(
                    "Qwen4 sequential QSA backing capacity is incomplete"
                )
            pooled_keys = getattr(cache, "_pooled_index_keys", None)
            pooled_offset = int(getattr(cache, "_pooled_index_offset", 0) or 0)
            pooled_ratio = getattr(cache, "_pooled_index_ratio", None)
            pooled_tag = getattr(cache, "_pooled_index_tag", None)
            if has_pooled:
                if pooled_keys is None:
                    if (
                        pooled_offset != 0
                        or pooled_ratio is not None
                        or pooled_tag is not None
                        or base_offset
                        >= _QWEN4_SEQUENTIAL_POOLED_REQUIRED_TOKENS
                    ):
                        raise _MtpStepFallback(
                            "Qwen4 sequential pooled QSA base is malformed"
                        )
                elif (
                    type(pooled_ratio) is not int
                    or pooled_ratio <= 0
                    or pooled_tag is None
                    or pooled_offset != base_offset // pooled_ratio
                    or pooled_offset > int(pooled_keys.shape[1])
                ):
                    raise _MtpStepFallback(
                        "Qwen4 sequential pooled QSA epoch is malformed"
                    )
            qsa.append(
                _Qwen4SequentialQSASnapshot(
                    cache=cache,
                    offset=base_offset,
                    index_offset=base_offset,
                    keys_backing=keys_backing,
                    values_backing=values_backing,
                    index_keys_backing=(
                        index_keys_backing
                    ),
                    index_positions_backing=(
                        index_positions_backing
                    ),
                    private_index_backing=private_index,
                    text_positions_qualified=qualified,
                    pooled_keys=pooled_keys,
                    pooled_offset=pooled_offset,
                    pooled_ratio=pooled_ratio,
                    pooled_tag=pooled_tag,
                    has_pooled_state=has_pooled,
                    index_capacity_managed=getattr(
                        cache,
                        "_index_capacity_managed",
                        None,
                    ),
                    geometric_capacity_managed=getattr(
                        cache,
                        "_geometric_capacity_managed",
                        None,
                    ),
                )
            )
            continue

        inner = vars(cache).get("_inner") if class_name == "SizedArraysCache" else cache
        if type(inner).__name__ != "ArraysCache":
            raise _MtpStepFallback(
                f"Qwen4 sequential cache leaf is unsupported: {class_name}"
            )
        state = getattr(inner, "state", None)
        if not isinstance(state, (list, tuple)) or len(state) not in (2, 4):
            raise _MtpStepFallback("Qwen4 sequential recurrent state is incomplete")
        expected_ndims = (3, 4) if len(state) == 2 else (3, 4, 3, 2)
        total_bytes = 0
        for value, expected_ndim in zip(state, expected_ndims):
            if (
                value is None
                or getattr(value, "ndim", 0) != expected_ndim
                or not hasattr(value, "nbytes")
                or int(value.nbytes) > 16 * 1024 * 1024
            ):
                raise _MtpStepFallback(
                    "Qwen4 sequential recurrent schema is not bounded"
                )
            total_bytes += int(value.nbytes)
        if total_bytes > 32 * 1024 * 1024:
            raise _MtpStepFallback(
                "Qwen4 sequential recurrent snapshot exceeds its per-leaf cap"
            )
        detached_state = tuple(
            _qwen4_detach_snapshot_value(value, arrays) for value in state
        )
        for index, value in enumerate(detached_state):
            if index < 2 and value is None:
                raise _MtpStepFallback(
                    "Qwen4 sequential recurrent state is not B1"
                )
            if value is not None and (
                getattr(value, "ndim", 0) < 1 or value.shape[0] != 1
            ):
                raise _MtpStepFallback(
                    "Qwen4 sequential recurrent/PLE state is not B1"
                )
        token_count = (
            getattr(cache, "_token_count", None)
            if class_name == "SizedArraysCache"
            else None
        )
        if token_count is not None and token_count != base_offset:
            raise _MtpStepFallback(
                "Qwen4 sequential recurrent timeline is not at the base"
            )
        metadata = tuple(
            (
                name,
                _qwen4_detach_snapshot_value(getattr(inner, name, None), arrays),
            )
            for name in (
                "_left_padding",
                "_left_padding_advance",
                "_lengths",
                "_lengths_advance",
                "_qwen3_5_left_padding_info",
                "_qwen3_5_lengths_info",
            )
        )
        recurrent.append(
            _Qwen4SequentialRecurrentSnapshot(
                cache=cache,
                state=detached_state,
                token_count=token_count,
                metadata=metadata,
            )
        )

    if not recurrent or not qsa:
        raise _MtpStepFallback(
            "Qwen4 sequential base requires recurrent and QSA cache leaves"
        )
    # Qwen4 replaces these arrays rather than mutating them. Retaining the
    # references avoids copying a context-long MRoPE position tensor per cycle.
    position_ids = getattr(language_model, "_position_ids", None)
    rope_deltas = getattr(language_model, "_rope_deltas", None)
    if arrays:
        mx.eval(*arrays)
    return _Qwen4SequentialBaseSnapshot(
        base_offset=base_offset,
        owner_uid=uids[0],
        recurrent=tuple(recurrent),
        qsa=tuple(qsa),
        language_model=language_model,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
    )


def _qwen4_sequential_prefix_is_exact(
    gen_batch: Any,
    snapshot: _Qwen4SequentialBaseSnapshot,
    *,
    accepted: int,
) -> bool:
    """Prove the live target owns base + next_main + accepted drafts."""

    expected = snapshot.base_offset + int(accepted) + 1
    uids = tuple(getattr(gen_batch, "uids", ()) or ())
    if (
        uids != (snapshot.owner_uid,)
        or _qwen4_language_model(gen_batch.model) is not snapshot.language_model
    ):
        return False
    live_cache_ids = {
        id(cache) for cache in _iter_mtp_cache_leaves(gen_batch.prompt_cache)
    }
    if any(
        id(entry.cache) not in live_cache_ids
        for entry in (*snapshot.recurrent, *snapshot.qsa)
    ):
        return False
    if _qwen4_target_offset(gen_batch.prompt_cache) != expected:
        return False
    for qsa in snapshot.qsa:
        if _qwen4_qsa_offsets(qsa.cache) != (expected, expected):
            return False
        keys = getattr(qsa.cache, "index_keys", None)
        positions = getattr(qsa.cache, "index_position_ids", None)
        if (
            keys is None
            or positions is None
            or int(keys.shape[1]) != expected
            or int(positions.shape[-1]) != expected
        ):
            return False
        if qsa.has_pooled_state:
            pooled_keys = getattr(qsa.cache, "_pooled_index_keys", None)
            pooled_offset = int(
                getattr(qsa.cache, "_pooled_index_offset", 0) or 0
            )
            pooled_ratio = getattr(qsa.cache, "_pooled_index_ratio", None)
            pooled_tag = getattr(qsa.cache, "_pooled_index_tag", None)
            if pooled_keys is None:
                if (
                    expected >= _QWEN4_SEQUENTIAL_POOLED_REQUIRED_TOKENS
                    or pooled_offset != 0
                    or pooled_ratio is not None
                    or pooled_tag is not None
                ):
                    return False
            elif (
                type(pooled_ratio) is not int
                or pooled_ratio <= 0
                or pooled_tag is None
                or pooled_offset != expected // pooled_ratio
                or pooled_offset > int(pooled_keys.shape[1])
            ):
                return False
    for recurrent in snapshot.recurrent:
        if recurrent.token_count is not None and getattr(
            recurrent.cache,
            "_token_count",
            None,
        ) != expected:
            return False
        state = getattr(recurrent.cache, "state", None)
        if not isinstance(state, (list, tuple)) or len(state) < 2:
            return False
        for index, value in enumerate(state):
            if index < 2 and value is None:
                return False
            if value is not None and (
                getattr(value, "ndim", 0) < 1 or value.shape[0] != 1
            ):
                return False
    return True


def _restore_qwen4_sequential_base(
    gen_batch: Any,
    state: _MtpState,
    snapshot: _Qwen4SequentialBaseSnapshot,
    *,
    accepted_current: int,
) -> bool:
    """Restore one live scalar-verifier prefix to its detached base."""

    current = snapshot.base_offset + int(accepted_current) + 1
    if not _qwen4_sequential_prefix_is_exact(
        gen_batch,
        snapshot,
        accepted=accepted_current,
    ):
        return False
    try:
        # Preflight every suffix before mutating any cache object.
        for qsa in snapshot.qsa:
            if _qwen4_qsa_offsets(qsa.cache) != (current, current):
                return False
        for qsa in snapshot.qsa:
            qsa.cache.keys = qsa.keys_backing
            qsa.cache.values = qsa.values_backing
            qsa.cache.offset = qsa.offset
            if qsa.private_index_backing:
                qsa.cache._index_keys = qsa.index_keys_backing
                qsa.cache._index_position_ids = qsa.index_positions_backing
                qsa.cache._index_offset = qsa.index_offset
            else:
                qsa.cache.index_keys = qsa.index_keys_backing
                qsa.cache.index_position_ids = qsa.index_positions_backing
                qsa.cache.index_offset = qsa.index_offset
            if qsa.has_pooled_state:
                qsa.cache._pooled_index_keys = qsa.pooled_keys
                qsa.cache._pooled_index_offset = qsa.pooled_offset
                qsa.cache._pooled_index_ratio = qsa.pooled_ratio
                qsa.cache._pooled_index_tag = qsa.pooled_tag
            if qsa.index_capacity_managed is not None:
                qsa.cache._index_capacity_managed = qsa.index_capacity_managed
            if qsa.geometric_capacity_managed is not None:
                qsa.cache._geometric_capacity_managed = (
                    qsa.geometric_capacity_managed
                )
            qsa.cache._omlx_text_position_ids_qualified = (
                qsa.text_positions_qualified
            )

        for recurrent in snapshot.recurrent:
            inner = (
                vars(recurrent.cache).get("_inner")
                if type(recurrent.cache).__name__ == "SizedArraysCache"
                else recurrent.cache
            )
            inner.state = list(recurrent.state)
            for name, value in recurrent.metadata:
                setattr(inner, name, value)
            if recurrent.token_count is not None:
                recurrent.cache._token_count = recurrent.token_count

        snapshot.language_model._position_ids = snapshot.position_ids
        snapshot.language_model._rope_deltas = snapshot.rope_deltas
        _clear_rollback(gen_batch.prompt_cache)
    except Exception as exc:
        logger.warning("Qwen4 sequential base restore failed: %s", exc)
        return False

    if any(
        _qwen4_qsa_offsets(qsa.cache)
        != (snapshot.base_offset, snapshot.base_offset)
        for qsa in snapshot.qsa
    ):
        return False
    if not _qwen4_reconcile_sized_recurrent_timeline(
        gen_batch.prompt_cache,
        expected=snapshot.base_offset,
        allowed_current={snapshot.base_offset},
    ):
        return False
    return _set_qwen4_target_expected_offset(
        state,
        gen_batch.prompt_cache,
        snapshot.base_offset,
    )


def _restore_qwen4_sequential_partial_forward(
    gen_batch: Any,
    state: _MtpState,
    snapshot: _Qwen4SequentialBaseSnapshot,
    *,
    max_width: int,
) -> bool:
    """Restore a base even when a failed scalar row left mixed layer epochs."""

    if max_width <= 0:
        return False
    uids = tuple(getattr(gen_batch, "uids", ()) or ())
    live_cache_ids = {
        id(cache) for cache in _iter_mtp_cache_leaves(gen_batch.prompt_cache)
    }
    if (
        uids != (snapshot.owner_uid,)
        or _qwen4_language_model(gen_batch.model) is not snapshot.language_model
        or any(
            id(entry.cache) not in live_cache_ids
            for entry in (*snapshot.recurrent, *snapshot.qsa)
        )
    ):
        return False
    upper = snapshot.base_offset + int(max_width)
    try:
        # Per-leaf preflight is deliberately independent: a model exception can
        # leave early QSA layers one row ahead of later layers, making the
        # uniform target-offset helper unreadable even though every suffix is
        # still bounded and restorable.
        for qsa in snapshot.qsa:
            offsets = _qwen4_qsa_offsets(qsa.cache)
            if offsets is None:
                return False
            kv_offset, index_offset = offsets
            if not (
                snapshot.base_offset <= kv_offset <= upper
                and snapshot.base_offset <= index_offset <= upper
            ):
                return False

        for qsa in snapshot.qsa:
            qsa.cache.keys = qsa.keys_backing
            qsa.cache.values = qsa.values_backing
            qsa.cache.offset = qsa.offset
            if qsa.private_index_backing:
                qsa.cache._index_keys = qsa.index_keys_backing
                qsa.cache._index_position_ids = qsa.index_positions_backing
                qsa.cache._index_offset = qsa.index_offset
            else:
                qsa.cache.index_keys = qsa.index_keys_backing
                qsa.cache.index_position_ids = qsa.index_positions_backing
                qsa.cache.index_offset = qsa.index_offset
            if qsa.has_pooled_state:
                qsa.cache._pooled_index_keys = qsa.pooled_keys
                qsa.cache._pooled_index_offset = qsa.pooled_offset
                qsa.cache._pooled_index_ratio = qsa.pooled_ratio
                qsa.cache._pooled_index_tag = qsa.pooled_tag
            if qsa.index_capacity_managed is not None:
                qsa.cache._index_capacity_managed = qsa.index_capacity_managed
            if qsa.geometric_capacity_managed is not None:
                qsa.cache._geometric_capacity_managed = (
                    qsa.geometric_capacity_managed
                )
            qsa.cache._omlx_text_position_ids_qualified = (
                qsa.text_positions_qualified
            )

        for recurrent in snapshot.recurrent:
            inner = (
                vars(recurrent.cache).get("_inner")
                if type(recurrent.cache).__name__ == "SizedArraysCache"
                else recurrent.cache
            )
            inner.state = list(recurrent.state)
            for name, value in recurrent.metadata:
                setattr(inner, name, value)
            if recurrent.token_count is not None:
                recurrent.cache._token_count = recurrent.token_count
        snapshot.language_model._position_ids = snapshot.position_ids
        snapshot.language_model._rope_deltas = snapshot.rope_deltas
        _clear_rollback(gen_batch.prompt_cache)
    except Exception as exc:
        logger.error("Qwen4 partial sequential restore failed: %s", exc)
        return False

    if any(
        _qwen4_qsa_offsets(qsa.cache)
        != (snapshot.base_offset, snapshot.base_offset)
        for qsa in snapshot.qsa
    ):
        return False
    if not _qwen4_reconcile_sized_recurrent_timeline(
        gen_batch.prompt_cache,
        expected=snapshot.base_offset,
        allowed_current={snapshot.base_offset},
    ):
        return False
    return _set_qwen4_target_expected_offset(
        state,
        gen_batch.prompt_cache,
        snapshot.base_offset,
    )


def _replay_qwen4_sequential_prefix(
    gen_batch: Any,
    state: _MtpState,
    snapshot: _Qwen4SequentialBaseSnapshot,
    token_ids: Tuple[int, ...],
) -> Any:
    """Commit an exact width-one target prefix from a restored base."""

    import mlx.core as mx

    if not token_ids:
        raise _MtpStepFallback("Qwen4 sequential replay prefix is empty")
    hidden_rows = []
    _set_singleton_mrope_delta(gen_batch)
    for token_id in token_ids:
        token = mx.array([[int(token_id)]], dtype=mx.uint32)
        logits, hidden, _ = _call_backbone(
            gen_batch.model,
            token,
            gen_batch.prompt_cache,
            n_confirmed=0,
        )
        if hidden is None or hidden.ndim < 3 or hidden.shape[1] != 1:
            raise _MtpStepFallback(
                "Qwen4 sequential replay did not return one raw hidden row"
            )
        mx.eval(
            logits,
            hidden,
            *[cache.state for cache in gen_batch.prompt_cache],
        )
        _clear_rollback(gen_batch.prompt_cache)
        hidden_rows.append(hidden)

    accepted = len(token_ids) - 1
    expected = snapshot.base_offset + len(token_ids)
    if not _qwen4_reconcile_sized_recurrent_timeline(
        gen_batch.prompt_cache,
        expected=expected,
        allowed_current={expected},
    ) or not _set_qwen4_target_expected_offset(
        state,
        gen_batch.prompt_cache,
        expected,
    ):
        raise _MtpStepFallback(
            "Qwen4 sequential replay target timeline is not exact"
        )
    if not _qwen4_sequential_prefix_is_exact(
        gen_batch,
        snapshot,
        accepted=accepted,
    ):
        raise _MtpStepFallback(
            "Qwen4 sequential replay cache proof failed"
        )
    return mx.concatenate(hidden_rows, axis=1)


def _select_qwen4_sequential_prefix(
    gen_batch: Any,
    state: _MtpState,
    snapshot: _Qwen4SequentialBaseSnapshot,
    *,
    current_accepted: int,
    target_input_ids: Tuple[int, ...],
    accepted: int,
) -> Any:
    """Restore the base and replay next_main plus ``accepted`` drafts."""

    if accepted < 0 or accepted >= len(target_input_ids):
        raise _MtpStepFallback("Qwen4 sequential accepted prefix is invalid")
    if not _restore_qwen4_sequential_base(
        gen_batch,
        state,
        snapshot,
        accepted_current=current_accepted,
    ):
        raise _MtpStepFallback("Qwen4 sequential base restore was not exact")
    return _replay_qwen4_sequential_prefix(
        gen_batch,
        state,
        snapshot,
        target_input_ids[: accepted + 1],
    )


def _capture_qwen4_verify_snapshots(
    prompt_cache: List[Any],
    *,
    base_offset: int,
    verify_width: int,
) -> Tuple[Tuple[Tuple[Any, Any], ...], Tuple[_Qwen4QSARollbackSnapshot, ...]]:
    """Capture the lightweight PLE/QSA proof for one untouched verifier.

    GDN state is returned directly by the backbone and is retained separately
    on ``_MtpPendingCommit``.  PLE owns its own immutable pre-window tensors.
    QSA needs no tensor copy: the verifier appends a suffix and its trim path
    owns all four logical components, so base/full offsets prove the epoch.
    """

    ple: List[Tuple[Any, Any]] = []
    qsa: List[_Qwen4QSARollbackSnapshot] = []
    expected_full = int(base_offset) + int(verify_width)
    for cache in _iter_mtp_cache_leaves(prompt_cache):
        snapshot = getattr(cache, "_qwen4_exp_ple_speculative_state", None)
        if snapshot is not None:
            if getattr(snapshot, "complete", False) is not True:
                raise _MtpStepFallback("Qwen4 PLE verify snapshot is incomplete")
            ple.append((cache, snapshot))

        if type(cache).__name__ not in _QWEN4_QSA_CACHE_TYPES:
            continue
        offsets = _qwen4_qsa_offsets(cache)
        if offsets is None:
            raise _MtpStepFallback("Qwen4 QSA verifier offsets are not scalar")
        offset, index_offset = offsets
        if offset != expected_full or index_offset != expected_full:
            raise _MtpStepFallback(
                "Qwen4 QSA verifier did not advance by the exact window "
                f"(base={base_offset}, width={verify_width}, kv={offset}, "
                f"index={index_offset})"
            )
        qsa.append(
            _Qwen4QSARollbackSnapshot(
                cache=cache,
                base_offset=int(base_offset),
                full_offset=offset,
                base_index_offset=int(base_offset),
                full_index_offset=index_offset,
            )
        )
    if not ple or not qsa:
        raise _MtpStepFallback(
            "Qwen4 terminal commit requires both PLE and QSA transaction state"
        )
    return tuple(ple), tuple(qsa)


def _validate_qwen4_qsa_epoch(
    pending: _MtpPendingCommit,
    *,
    expected_offset: int,
) -> bool:
    if not pending.qsa_snapshots:
        return False
    for snapshot in pending.qsa_snapshots:
        cache = snapshot.cache
        if (
            snapshot.base_offset != pending.target_base_offset
            or snapshot.base_index_offset != pending.target_base_offset
            or snapshot.full_offset
            != pending.target_base_offset + pending.verify_width
            or snapshot.full_index_offset != snapshot.full_offset
        ):
            return False
        offsets = _qwen4_qsa_offsets(cache)
        if offsets != (expected_offset, expected_offset):
            return False
        logical_keys = getattr(cache, "index_keys", None)
        logical_positions = getattr(cache, "index_position_ids", None)
        if (
            logical_keys is None
            or logical_positions is None
            or int(logical_keys.shape[1]) != expected_offset
            or int(logical_positions.shape[-1]) != expected_offset
        ):
            return False
    return True


def _pending_ple_epoch_is_exact(pending: _MtpPendingCommit) -> bool:
    """Prove every PLE cache still owns the captured verifier transaction."""

    if not pending.ple_snapshots:
        return False
    for cache, snapshot in pending.ple_snapshots:
        current = getattr(cache, "_qwen4_exp_ple_speculative_state", None)
        if current is not snapshot:
            return False
        if getattr(snapshot, "complete", False) is not True:
            return False
    return True


def _set_qwen4_target_expected_offset(
    state: _MtpState,
    prompt_cache: List[Any],
    expected: int,
) -> bool:
    observed = _qwen4_target_offset(prompt_cache)
    if observed != expected:
        return False
    if state.suffix_local_priming:
        state.target_expected_offset = expected
    return True


def _qwen4_rollback_full_verify_to(
    gen_batch: Any,
    state: _MtpState,
    pending: _MtpPendingCommit,
    *,
    accepted: int,
) -> bool:
    """Select one target prefix from the still-full verifier transaction."""

    drafts = pending.verify_width - 1
    if accepted < 0 or accepted > drafts:
        return False
    full_offset = pending.target_base_offset + pending.verify_width
    if not _validate_qwen4_qsa_epoch(pending, expected_offset=full_offset):
        return False
    if not _qwen4_reconcile_sized_recurrent_timeline(
        gen_batch.prompt_cache,
        expected=full_offset,
        allowed_current={full_offset},
    ):
        return False

    if accepted == drafts:
        _clear_rollback(gen_batch.prompt_cache)
    else:
        if pending.gdn_states is None:
            return False
        if not _pending_ple_epoch_is_exact(pending):
            return False
        rollback = getattr(gen_batch.model, "rollback_speculative_cache", None)
        if not callable(rollback):
            return False
        try:
            rollback(
                gen_batch.prompt_cache,
                pending.gdn_states,
                accepted,
                pending.verify_width,
            )
        except Exception as exc:
            logger.warning("Qwen4 terminal target rollback failed: %s", exc)
            return False

    expected = pending.target_base_offset + accepted + 1
    if not _validate_qwen4_qsa_epoch(pending, expected_offset=expected):
        return False
    if not _qwen4_reconcile_sized_recurrent_timeline(
        gen_batch.prompt_cache,
        expected=expected,
        allowed_current={full_offset, expected},
    ):
        return False
    if not _set_qwen4_target_expected_offset(
        state, gen_batch.prompt_cache, expected
    ):
        return False
    pending.committed = True
    return True


def _qwen4_commit_sequential_verify_to(
    gen_batch: Any,
    state: _MtpState,
    pending: _MtpPendingCommit,
    *,
    accepted: int,
) -> bool:
    """Select one prefix from a canonical live scalar-verifier transaction."""

    snapshot = pending.sequential_base
    if (
        snapshot is None
        or snapshot.base_offset != pending.target_base_offset
        or len(pending.target_input_ids) != pending.verify_width
        or pending.verify_width < 2
        or not 0 <= pending.accepted <= pending.verify_width - 1
        or accepted < 0
        or accepted > pending.accepted
        or len(pending.target_input_ids) < accepted + 1
    ):
        return False
    try:
        if accepted < pending.accepted:
            _select_qwen4_sequential_prefix(
                gen_batch,
                state,
                snapshot,
                current_accepted=pending.accepted,
                target_input_ids=pending.target_input_ids,
                accepted=accepted,
            )
        elif not _qwen4_sequential_prefix_is_exact(
            gen_batch,
            snapshot,
            accepted=accepted,
        ):
            return False
        expected = pending.target_base_offset + accepted + 1
        if not _qwen4_reconcile_sized_recurrent_timeline(
            gen_batch.prompt_cache,
            expected=expected,
            allowed_current={expected},
        ) or not _set_qwen4_target_expected_offset(
            state,
            gen_batch.prompt_cache,
            expected,
        ):
            return False
        _clear_rollback(gen_batch.prompt_cache)
        pending.committed = True
        return True
    except Exception as exc:
        logger.warning("Qwen4 sequential target commit failed: %s", exc)
        return False


def _qwen4_materialize_target_tail(
    gen_batch: Any,
    state: _MtpState,
    token_id: int,
) -> bool:
    """Commit one already-emitted pipeline-tail token to the target only."""

    import mlx.core as mx

    before = _qwen4_target_offset(gen_batch.prompt_cache)
    if before is None:
        return False
    token = mx.array([int(token_id)], dtype=mx.uint32)
    try:
        logits, _hidden, _ = _call_backbone(
            gen_batch.model,
            token[:, None],
            gen_batch.prompt_cache,
        )
        mx.eval(logits, *[cache.state for cache in gen_batch.prompt_cache])
        _clear_rollback(gen_batch.prompt_cache)
    except Exception as exc:
        logger.warning("Qwen4 terminal one-token target commit failed: %s", exc)
        return False
    expected = before + 1
    return _qwen4_reconcile_sized_recurrent_timeline(
        gen_batch.prompt_cache,
        expected=expected,
        allowed_current={expected},
    ) and _set_qwen4_target_expected_offset(
        state, gen_batch.prompt_cache, expected
    )


def _ensure_uint32(arr):
    """Ensure a 1-element mx.array is uint32 (cache update_and_fetch expects it)."""
    import mlx.core as mx

    if arr.dtype == mx.uint32:
        return arr
    return arr.astype(mx.uint32)


# ---------------------------------------------------------------------------
# Depth-k chained drafting helpers (Qwen3.5/3.6): a linear draft chain through
# the MTP head, one batched verify forward covering all drafts plus a free
# bonus row, offset-trim KV rollback with GDN prefix replay, and a
# committed-only MTP-head history rebuilt from verify hidden rows each cycle.
# ---------------------------------------------------------------------------


def _resolve_mtp_chain_depth(model: Any) -> Tuple[bool, int, bool]:
    """Read the chain capability markers stamped on the model at load.

    Returns ``(chain, depth, head_clone)``. ``head_clone`` marks models
    whose MTP-head cache cannot be exactly trimmed (e.g. DeepSeek-V4's
    RotatingKVCache head once rotated): the chain then runs its speculative
    draft steps on a per-cycle clone and keeps the persistent head cache
    committed-only, instead of trimming speculative entries afterwards.
    """
    candidates = [model]
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            candidates.append(inner)
    for candidate in candidates:
        if getattr(candidate, "_omlx_mtp_chain", False):
            depth = int(getattr(candidate, "_omlx_mtp_depth", 1) or 1)
            head_clone = bool(getattr(candidate, "_omlx_mtp_head_clone", False))
            return True, max(1, min(8, depth)), head_clone
    return False, 1, False


def _clone_mtp_head_cache(mtp_cache: List[Any]) -> List[Any]:
    """Detached per-cycle copy of the MTP-head cache for speculative steps.

    ``copy.copy`` keeps scalars; mx.array attributes are detached with
    ``v + 0`` so the clone's in-place ring writes never mutate arrays the
    persistent cache still references; list attributes are shallow-copied.
    Container caches (``CacheList``-style, exposing ``.caches``) recurse.
    """
    import copy

    import mlx.core as mx

    def clone_one(c):
        if c is None:
            return None
        subs = getattr(c, "caches", None)
        if subs is not None:
            return type(c)(*[clone_one(sub) for sub in subs])
        new = copy.copy(c)
        for attr, val in vars(c).items():
            if isinstance(val, mx.array):
                setattr(new, attr, val + 0)
            elif isinstance(val, list):
                setattr(new, attr, list(val))
        return new

    return [clone_one(c) for c in mtp_cache]


def _trunk_norm_module(model: Any):
    """Final RMSNorm of the backbone (for post_norm head inputs).

    Walks both wrapper conventions: mlx-lm's outer ``Model.language_model``
    and oMLX's ``VLMModelAdapter._language_model`` (mlx-vlm path).

    Models whose ``return_hidden`` output is already post-norm mark
    ``_omlx_mtp_head_hidden_normed`` (instance or inner language model) and
    get an identity here — applying the trunk norm again would double-norm
    the head inputs, and the ``inner.model.norm`` walk does not fit every
    backbone layout (e.g. ``backbone.norm_f``).
    """
    inner = model
    for attr in ("language_model", "_language_model"):
        candidate = getattr(model, attr, None)
        if candidate is not None:
            inner = candidate
            break
    if getattr(model, "_omlx_mtp_head_hidden_normed", False) or getattr(
        inner, "_omlx_mtp_head_hidden_normed", False
    ):
        return lambda x: x
    return inner.model.norm


# Single source of truth lives in prompt_priming: the prefill-time priming
# folds and the decode-time history folds here must use the same hidden
# variant or the primed history would be inconsistent with the chained one.
_HEAD_HIDDEN_POST_NORM = _prompt_priming.HEAD_HIDDEN_POST_NORM


def _mtp_head_trim_to(mtp_cache: List[Any], offset: int) -> None:
    """Trim speculative chain entries so the head cache ends at ``offset``."""
    for c in mtp_cache:
        current = int(getattr(c, "offset", 0))
        extra = current - offset
        if extra > 0:
            c.trim(extra)


def _adopt_primed_head_state(
    state: _MtpState,
    primed: Any,
    target_cache: Optional[List[Any]],
) -> bool:
    """Install generic or Qwen4 suffix-local priming, failing closed."""

    if isinstance(primed, _prompt_priming.SuffixLocalPrimedState):
        head_offset = _prompt_priming.mtp_cache_offset(primed.mtp_cache)
        target_offset = _prompt_priming.target_cache_offset(target_cache)
        if (
            head_offset != primed.head_hist_offset
            or target_offset != primed.target_expected_offset
        ):
            logger.debug(
                "Qwen4 suffix-local priming rejected at activation: "
                "head=%s/%s target=%s/%s",
                head_offset,
                primed.head_hist_offset,
                target_offset,
                primed.target_expected_offset,
            )
            return False
        state.mtp_cache = primed.mtp_cache
        state.hist_offset = primed.head_hist_offset
        state.target_expected_offset = primed.target_expected_offset
        state.suffix_local_priming = True
        return True
    try:
        mtp_cache, hist_offset = primed
    except (TypeError, ValueError):
        return False
    state.mtp_cache = mtp_cache
    state.hist_offset = int(hist_offset)
    return True


def _trim_committed_mtp_head(state: _MtpState) -> None:
    """Trim against the local head timeline and prove suffix isolation."""

    _mtp_head_trim_to(state.mtp_cache, state.hist_offset)
    if state.suffix_local_priming:
        observed = _prompt_priming.mtp_cache_offset(state.mtp_cache)
        if observed != state.hist_offset:
            raise _MtpStepFallback(
                "Qwen4 suffix-local MTP head escaped its local trim boundary "
                f"(expected={state.hist_offset}, observed={observed})"
            )


def _advance_suffix_local_target(
    state: _MtpState,
    target_cache: Optional[List[Any]],
    retained_tokens: int,
) -> None:
    """Advance and verify the separate absolute target-cache timeline."""

    if not state.suffix_local_priming:
        return
    if state.target_expected_offset is None or retained_tokens <= 0:
        raise _MtpStepFallback("Qwen4 suffix-local target seam is incomplete")
    expected = state.target_expected_offset + int(retained_tokens)
    observed = _prompt_priming.target_cache_offset(target_cache)
    if observed != expected:
        raise _MtpStepFallback(
            "Qwen4 suffix-local target cache left its absolute seam "
            f"(expected={expected}, observed={observed})"
        )
    state.target_expected_offset = expected


# Loop-tax measurement (feeds _DepthController.EXIT_MARGIN): right after a
# hand-off the standard decoder runs on the same model, machine, and
# context, so the ratio of the exit-time t[0] to the measured standard-step
# time IS the MTP loop's synchronous-cycle tax. Stored on the model
# instance so later sequences exit against a measured margin instead of
# the fallback prior.
_STD_TAX_SKIP = 2  # first post-hand-off steps still carry transition costs
_STD_TAX_SAMPLES = 8
_STD_TAX_EMA = 0.5
_STD_TAX_MAX = 1.5
# Cycle timings sampled while another request is prefilling (any engine on
# the shared GPU) are contention-shaped, not loop-shaped. A park probe armed
# or finalized in that window latches a poisoned t0/t_std ratio on the model
# and depresses MTP for every later sequence (#2622).
_STD_TAX_CONTENTION_WINDOW_S = 3.0
_STD_TAX_WARN = 1.3  # a stored tax at/above this is worth an INFO line
_STD_TAX_DECAY_S = 600.0  # stale latch decays toward the default margin


def _prefill_activity_recent() -> bool:
    try:
        return get_prefill_tracker().recently_active(
            _STD_TAX_CONTENTION_WINDOW_S
        )
    except Exception:
        return False


def _arm_std_tax_probe(
    gen_batch: Any, t0_ms: Optional[float], uid: Any = None
) -> None:
    if not (t0_ms and t0_ms > 0.0):
        return
    if _prefill_activity_recent():
        logger.debug(
            "MTP loop-tax probe skipped: prefill activity within %.1fs, "
            "t0=%.1fms is contention-contaminated",
            _STD_TAX_CONTENTION_WINDOW_S,
            t0_ms,
        )
        return
    gen_batch._omlx_mtp_tax_probe = {
        "t0": float(t0_ms),
        "skip": _STD_TAX_SKIP,
        "samples": [],
        "uid": uid,
    }


def _record_std_tax_sample(gen_batch: Any, duration_ms: float) -> None:
    probe = getattr(gen_batch, "_omlx_mtp_tax_probe", None)
    if probe is None:
        return
    uids = getattr(gen_batch, "uids", None)
    if probe.get("uid") is not None and list(uids or ()) != [probe["uid"]]:
        # The batch gained or swapped rows since the hand-off; multi-row
        # step timings would contaminate the singleton loop-tax ratio.
        try:
            delattr(gen_batch, "_omlx_mtp_tax_probe")
        except AttributeError:
            pass
        return
    if probe["skip"] > 0:
        probe["skip"] -= 1
        return
    probe["samples"].append(float(duration_ms))
    if len(probe["samples"]) < _STD_TAX_SAMPLES:
        return
    try:
        delattr(gen_batch, "_omlx_mtp_tax_probe")
    except AttributeError:
        pass
    samples = sorted(probe["samples"])
    t_std = samples[len(samples) // 2]
    if t_std <= 0.0:
        return
    if _prefill_activity_recent():
        logger.debug(
            "MTP loop-tax probe discarded: prefill activity during the "
            "std sampling window"
        )
        return
    tax = min(_STD_TAX_MAX, max(1.0, probe["t0"] / t_std))
    model = getattr(gen_batch, "model", None)
    if model is None:
        return
    prev = getattr(model, "_omlx_mtp_loop_tax", None)
    if prev:
        tax = (1.0 - _STD_TAX_EMA) * float(prev) + _STD_TAX_EMA * tax
    try:
        model._omlx_mtp_loop_tax = tax
        model._omlx_mtp_loop_tax_ts = time.monotonic()
    except Exception:
        return
    log = logger.info if tax >= _STD_TAX_WARN else logger.debug
    log(
        "MTP loop tax measured: %.3f (parked t0=%.1fms, std step=%.1fms)",
        tax,
        probe["t0"],
        t_std,
    )


def _effective_loop_tax(model: Any) -> Optional[float]:
    """Measured loop tax, decayed toward the default exit margin with age.

    A genuine loop tax is stable and re-latches through fresh park probes,
    so decay costs nothing in the steady state. A high value that stopped
    being reproduced — the signature of a probe that sampled around a
    contention episode despite the arm/finalize guards — must not depress
    MTP until the next restart (#2622).
    """
    tax = getattr(model, "_omlx_mtp_loop_tax", None)
    if tax is None:
        return None
    ts = getattr(model, "_omlx_mtp_loop_tax_ts", None)
    if ts is None:
        return float(tax)
    age = max(0.0, time.monotonic() - float(ts))
    prior = float(_DepthController.EXIT_MARGIN)
    return prior + (float(tax) - prior) * math.exp(-age / _STD_TAX_DECAY_S)


class _DepthController:
    """Adaptive draft-depth selection.

    Pure host-side bookkeeping — no extra GPU syncs. Tracks an EMA of
    conditional acceptance per depth position and a wall-time EMA of cycle
    cost per depth used, then picks the depth with the best expected tokens
    per unit time:

        score(d) = (1 + p1 + p1*p2 + ... ) / t_est(d)

    Everything the decision uses is measured on this machine, on this model,
    under the current load — no hand-tuned per-chip or per-model value:

    - Cost: a warmup sweep runs each depth once so every ``t[d]`` starts from
      a real cycle; the marginal cost of an extra verify row is the measured
      slope between depths (``_marginal_est``), so a fine-grained MoE on a
      bandwidth-limited chip learns its true (large) marginal within the first
      few cycles. ``MARGINAL_MS`` is only the pre-measurement fallback.
    - Drift: the cost EMA horizon is wall-clock (``TAU_MS``), not a cycle
      count, so context growth, thermal throttling and external GPU contention
      are tracked at constant real-time responsiveness however long a cycle
      is; a one-off slow cycle is damped (``SPIKE_RATIO``).
    - Staleness: only the depth currently run gets fresh measurements, and a
      fresh-vs-stale cost comparison is systematically biased — e.g. a depth
      whose t was measured during the slow post-prefill cycles looks expensive
      forever, so the controller locks into its rival (measured as a depth-2
      lock costing prose ~2-4%). Probes are therefore BIDIRECTIONAL and
      staleness-directed: on a wall-clock cadence, re-run the best rival depth
      (shallower or deeper) when its score is within ``PROBE_MARGIN`` of the
      current choice, and periodically the most-stale depth, so every t[d] has
      bounded age. On heavy models a fixed wall-clock cadence would spend a
      large share of cycles probing, so probing is duty-bounded to
      ~``PROBE_DUTY`` of cycles — a scale-free ratio, not a per-model tuning.

    Depth 0 — the escape hatch: speculation is only profitable while the
    multi-token verify forward is cheap relative to a plain decode step.
    On models with a large L=1 -> L=2 forward-cost jump (gemma4 head_dim
    256/512 leaves the single-token attention kernel; MoE expert loads
    scale with verify tokens) the whole depth menu can be worse than
    standard decoding — measured 0.67x on gemma4 26B story/16k with the
    best depth choice. Depth 0 runs the cycle as a plain 1-token step
    ([next_main] only, no drafts, no rollback) whose cost is tracked as
    ``t[0]`` through the same EMA/probe machinery, so the controller
    parks at 0 when every speculative depth loses and re-enters through
    the existing bidirectional probes. ``t[0]`` gets its first real
    measurement from the warmup sweep (which ends with one depth-0
    cycle) and stays fresh via parked cycles and the staleness explorer.
    Parking alone is not enough on fast backbones — a parked cycle still
    pays the MTP loop's synchronous host round-trip that the standard
    decoder pipelines away — so a sustained park hands the sequence back
    to the standard step entirely (``_park_mtp_to_standard``).

    Content-adaptive by construction: prose/chat settles at depth 1,
    code/predictable text climbs. Rejected alternatives (interleaved
    in-process A/B, rotated order, paired per rep, on Qwen3.6-35B-A3B +
    GLM-5.2, M3 Ultra): a fixed shallow-bias constant (won earlier separate-
    process comparisons only by masking the staleness lock; this design beats
    it on 3 of 4 model x content cells and ties the 4th), a pure realized
    tok/s bandit (exploration tax), a live per-cycle learned correction
    (decision churn), a frozen cross-generation correction (content
    oscillation), and a base x shape cost decomposition (unidentifiable while
    one depth runs for long stretches).
    """

    ALPHA = 0.08  # acceptance EMA weight (token domain, content-driven)
    TAU_MS = 400.0  # cost EMA horizon in wall-clock ms (load/thermal/context)
    PROBE_PERIOD_MS = 1000.0  # min wall-time between probes (light models)
    PROBE_PERIOD_MAX_MS = 5000.0  # staleness-exploration cadence floor
    PROBE_LEN = 4
    PROBE_DUTY = 0.15  # probes never consume more than ~this share of cycles
    PROBE_MARGIN = 1.15  # a rival within this score ratio is worth re-measuring
    SPIKE_RATIO = 2.0  # a cycle above this * the EMA is treated as an outlier
    SPIKE_DAMP = 0.25  # ...and folded in at this fraction of the normal weight
    # Fallback prior for one extra verify token's cost, used only until two
    # depths have actually been measured; after that the marginal is the
    # measured slope between depths. 7 ms matches dense backbones (6-10 ms).
    MARGINAL_MS = 7.0
    HYSTERESIS = 1.03  # switch depth only for a >3% score gain
    # Hand-off gate: ``t[0]`` is measured INSIDE the MTP loop, so it carries
    # the loop's synchronous host round-trip that the standard decoder
    # pipelines away. Speculation that cannot beat this taxed baseline by
    # EXIT_MARGIN is losing to the real standard step; after EXIT_STREAK
    # consecutive losing decisions the sequence leaves the MTP path
    # entirely (_park_mtp_to_standard). EXIT_MARGIN is only the
    # pre-measurement fallback (like MARGINAL_MS): each hand-off measures
    # the actual standard-step rate right after it and stores the real
    # loop tax on the model instance, which seeds later controllers via
    # the ``exit_margin`` constructor arg — machine/model-measured, not
    # a hardcoded ratio.
    EXIT_MARGIN = 1.15
    EXIT_STREAK = 16

    def __init__(
        self,
        max_depth: int,
        marginal_ms: Optional[float] = None,
        exit_margin: Optional[float] = None,
    ):
        if marginal_ms:
            self.MARGINAL_MS = float(marginal_ms)
        if exit_margin:
            self.EXIT_MARGIN = min(
                _STD_TAX_MAX, max(1.0, float(exit_margin))
            )
        self.max_depth = max(1, int(max_depth))
        self.cur = self.max_depth  # first cycle drafts deep; warmup sweeps down
        self.p = [0.6] * self.max_depth
        self.t: Dict[int, float] = {}
        self.t_age: Dict[int, float] = {}  # ms since each depth was measured
        self.cycles = 0
        self.probe_left = 0
        self.exit_streak = 0
        self._ms_probe = 0.0  # wall-time since any probe burst
        self._ms_explore = 0.0  # wall-time since a staleness-exploration burst
        # Measure each depth once (max..1), then the depth-0 plain step
        # three times, before the score gate takes over — t[], including
        # the baseline the exit decision compares against, is data-driven
        # within max_depth + 3 cycles. The baseline gets extra samples
        # because it may never be selected again (no refresh path), and
        # a performance handoff must not hang on a single first-run sample;
        # plain-step warmup cycles cost almost nothing.
        self._warmup: List[int] = list(range(self.max_depth, 0, -1))
        if self.max_depth > 1:
            self._warmup.extend([0, 0, 0])

    def observe(
        self,
        used: int,
        accepted: int,
        cycle_ms: float,
        time_sample: bool = True,
    ) -> None:
        self.cycles += 1
        used = max(0, min(int(used), self.max_depth))
        accepted = max(0, min(int(accepted), used))
        # Acceptance: token-domain EMA (a property of model/content, not load).
        a = self.ALPHA
        for j in range(used):
            hit = 1.0 if j < accepted else 0.0
            self.p[j] = (1.0 - a) * self.p[j] + a * hit
            if j >= accepted:
                break
        # Cost: wall-time-domain EMA with a one-off-spike guard, plus per-depth
        # ages so probes can target the estimate that is most stale.
        # ``time_sample=False`` marks cycles carrying a one-off maintenance
        # cost (a multi-block head keepalive refold) whose spike would bias
        # this depth's estimate; the acceptance update above still applies.
        cycle_ms = max(0.0, float(cycle_ms))
        if time_sample:
            self._update_time(used, cycle_ms)
        for d in list(self.t_age):
            self.t_age[d] += cycle_ms
        if time_sample:
            self.t_age[used] = 0.0
        self._ms_probe += cycle_ms
        self._ms_explore += cycle_ms

        if self._speculation_losing():
            self.exit_streak += 1
        else:
            self.exit_streak = 0

        # Warmup sweep: keep walking max..1 until every depth is measured once.
        if self._warmup:
            self._warmup.pop(0)
            if self._warmup:
                self.cur = self._warmup[0]
                return
            self.cur = self._best()
            self._ms_probe = 0.0
            return

        # Finishing a probe burst.
        if self.probe_left > 0:
            self.probe_left -= 1
            if self.probe_left == 0:
                self.cur = self._best()
                self._ms_probe = 0.0
            return

        # Re-decide every cycle (cheap); HYSTERESIS in _best prevents thrash.
        self.cur = self._best()

        # Probe scheduling: bounded-staleness re-measurement in either
        # direction, at a duty-bounded wall-clock cadence.
        if self.max_depth > 1:
            period = max(
                self.PROBE_PERIOD_MS,
                self.PROBE_LEN * cycle_ms / self.PROBE_DUTY,
            )
            if self._ms_probe >= period:
                explore_due = self._ms_explore >= max(
                    self.PROBE_PERIOD_MAX_MS, 2.0 * period
                )
                target = (
                    self._most_stale() if explore_due else self._best_rival()
                )
                if target is not None:
                    self.cur = target
                    self.probe_left = self.PROBE_LEN
                    self._ms_probe = 0.0
                    if explore_due:
                        self._ms_explore = 0.0

    def _time_alpha(self, cycle_ms: float) -> float:
        # EMA weight for a cycle of this wall-time: the memory horizon is
        # ~TAU_MS regardless of cycle duration, so responsiveness is constant
        # in real time whether a cycle is 8 ms (short) or 80 ms (128k context).
        return 1.0 - math.exp(-max(0.0, float(cycle_ms)) / self.TAU_MS)

    def _update_time(self, used: int, cycle_ms: float) -> None:
        cycle_ms = max(0.0, float(cycle_ms))
        prev = self.t.get(used)
        if prev is None:
            self.t[used] = cycle_ms
            return
        if self._warmup:
            # Repeated warmup samples (the depth-0 tail): keep the fastest.
            # First-run shape/branch warmup inflates early samples, and the
            # slow EMA below would freeze that bias into the exit decision.
            self.t[used] = min(prev, cycle_ms)
            return
        # Deliberately a per-cycle EMA, NOT an irregular-sampling EMA weighted
        # by staleness age. Age-weighting (nearly replacing a stale estimate at
        # the first probe cycle) is the textbook form, but it was measured
        # WORSE here: single-cycle noise is ~±10%, so replacing an estimate
        # from a 4-cycle probe burst injects that noise straight into the depth
        # decision every probe (~1s), and prose re-over-drafted (-1.6%). The
        # slow EMA is a variance shield; stale errors are corrected by probe
        # REPETITION instead (each ~1s rival probe moves the estimate ~10% of
        # the gap, converging within a few seconds).
        a = self._time_alpha(cycle_ms)
        if cycle_ms > self.SPIKE_RATIO * prev:
            a *= self.SPIKE_DAMP  # a one-off spike moves the estimate slowly
        self.t[used] = (1.0 - a) * prev + a * cycle_ms

    def _marginal_est(self) -> float:
        # Measured cost of one extra verify row: the slope between the cheapest
        # and priciest measured depths. Falls back to the prior until two
        # depths exist. This is what self-calibrates the controller to the real
        # (model x chip x context) marginal instead of a hardcoded value.
        if len(self.t) >= 2:
            depths = sorted(self.t)
            lo, hi = depths[0], depths[-1]
            if hi > lo:
                slope = (self.t[hi] - self.t[lo]) / (hi - lo)
                if slope > 0.0:
                    return slope
        return self.MARGINAL_MS

    def _t_est(self, d: int) -> float:
        if d in self.t:
            return self.t[d]
        if not self.t:
            return 30.0 + self.MARGINAL_MS * d
        if d == 0:
            # The plain step sits below the L=1 -> L=2 verify jump, so the
            # per-row marginal says nothing about it. Estimate it at the
            # cheapest measured cycle: conservative (true t[0] is lower),
            # which keeps an unmeasured baseline from hijacking probes —
            # yet still within PROBE_MARGIN exactly when acceptance is so
            # poor that the baseline is a genuine rival.
            return min(self.t.values())
        ref = min(self.t, key=lambda x: abs(x - d))
        return max(1e-3, self.t[ref] + self._marginal_est() * (d - ref))

    def _score(self, d: int) -> float:
        expected = 1.0
        run = 1.0
        for j in range(d):
            run *= self.p[j]
            expected += run
        return expected / max(1e-6, self._t_est(d))

    def _speculation_losing(self) -> bool:
        # True when the best speculative depth cannot beat the (taxed)
        # in-loop baseline by EXIT_MARGIN. Only meaningful once the warmup
        # sweep has measured t[0].
        if self._warmup or 0 not in self.t:
            return False
        base = self._score(0)
        if base <= 0.0:
            return False
        best = max(self._score(d) for d in range(1, self.max_depth + 1))
        return best < base * self.EXIT_MARGIN

    def should_exit(self) -> bool:
        """Sustained losing speculation: hand the sequence back to the
        standard decoder."""
        return self.exit_streak >= self.EXIT_STREAK

    def _select_candidates(self) -> List[int]:
        # Depth 0 is only selectable once its cost has actually been
        # measured (or seeded) — an extrapolated baseline must never PARK
        # the sequence, only motivate a probe.
        ds = list(range(1, self.max_depth + 1))
        if 0 in self.t:
            ds.insert(0, 0)
        return ds

    def _probe_candidates(self) -> List[int]:
        # Probing depth 0 is always safe (it IS a plain decode step), so
        # the baseline is discoverable before any measurement exists.
        # Speculative depths come first: on an unmeasured-vs-unmeasured
        # staleness tie they keep priority (warmup semantics), while a
        # never-measured baseline still outranks any finite age.
        return list(range(1, self.max_depth + 1)) + [0]

    def _best_rival(self) -> Optional[int]:
        # The highest-scoring depth other than cur, if within PROBE_MARGIN —
        # i.e. a depth whose (possibly stale) estimate could flip the choice.
        # Bidirectional on purpose: re-measuring a SHALLOWER rival is what
        # breaks the depth-2 lock (a stale-high t[1] hides depth 1's true
        # advantage and nothing else would ever refresh it).
        best = self._score(self.cur)
        if best <= 0.0:
            return self._most_stale()
        rival = None
        rival_score = 0.0
        for d in self._probe_candidates():
            if d == self.cur:
                continue
            s = self._score(d)
            if s > rival_score:
                rival, rival_score = d, s
        if rival is not None and rival_score >= best / self.PROBE_MARGIN:
            return rival
        return None

    def _most_stale(self) -> Optional[int]:
        # The depth whose cost estimate has gone longest unmeasured (never
        # measured counts as infinitely stale). Keeps every t[d] fresh enough
        # that fresh-vs-stale comparison bias stays bounded.
        cand = None
        worst = -1.0
        for d in self._probe_candidates():
            if d == self.cur:
                continue
            age = self.t_age.get(d)
            age = float("inf") if age is None else age
            if age > worst:
                cand, worst = d, age
        return cand

    def _best(self) -> int:
        # argmax of measured score with switch hysteresis; ascending scan
        # with strict '>' keeps the shallower choice on an exact tie.
        best_d = self.cur
        best_score = -1.0
        for d in self._select_candidates():
            s = self._score(d)
            if s > best_score:
                best_d, best_score = d, s
        if best_d != self.cur and best_score < self._score(self.cur) * self.HYSTERESIS:
            return self.cur
        return best_d


class _EvidenceDepthController:
    """Experimental evidence-directed draft-depth selection.

    The factory selects this policy only for Qwen4 when
    ``OMLX_QWEN4_EVIDENCE_DEPTH=1``. Every other local model keeps the legacy
    controller above, and distributed lockstep remains authoritative.

    Pure host-side bookkeeping — no extra GPU syncs. Tracks an EMA of
    conditional acceptance per depth position and a wall-time EMA of cycle
    cost per depth used, then picks the depth with the best expected tokens
    per unit time:

        score(d) = (1 + p1 + p1*p2 + ... ) / t_est(d)

    Everything the decision uses is measured on this machine, on this model,
    under the current load — no hand-tuned per-chip or per-model value:

    - Cost: a warmup sweep runs each speculative depth once so every ``t[d]``
      starts from a real cycle; the marginal cost of an extra verify row is the
      measured slope between depths (``_marginal_est``), so a fine-grained MoE
      on a bandwidth-limited chip learns its true (large) marginal within the
      first few cycles. ``MARGINAL_MS`` is only the pre-measurement fallback.
    - Drift: the cost EMA horizon is wall-clock (``TAU_MS``), not a cycle
      count, so context growth, thermal throttling and external GPU contention
      are tracked at constant real-time responsiveness however long a cycle
      is; a one-off slow cycle is damped (``SPIKE_RATIO``).
    - Staleness: only the depth currently run gets fresh measurements, and a
      fresh-vs-stale cost comparison is systematically biased — e.g. a depth
      whose t was measured during the slow post-prefill cycles looks expensive
      forever, so the controller locks into its rival (measured as a depth-2
      lock costing prose ~2-4%). Probes are therefore bidirectional and
      evidence-directed: an under-observed conditional-acceptance tail gets a
      Jeffreys-posterior upper bound for exploration, while mature rivals are
      re-run only while empirical cost intervals can overlap the current
      winner. Periodic most-stale probes remain the drift backstop. On heavy
      models a fixed wall-clock cadence would spend a large share of cycles
      probing, so probing is duty-bounded to ~``PROBE_DUTY`` of cycles.

    Depth 0 — the escape hatch: speculation is only profitable while the
    multi-token verify forward is cheap relative to a plain decode step.
    On models with a large L=1 -> L=2 forward-cost jump (gemma4 head_dim
    256/512 leaves the single-token attention kernel; MoE expert loads
    scale with verify tokens) the whole depth menu can be worse than
    standard decoding — measured 0.67x on gemma4 26B story/16k with the
    best depth choice. Depth 0 runs the cycle as a plain 1-token step
    ([next_main] only, no drafts, no rollback) whose cost is tracked as
    ``t[0]`` through the same EMA/probe machinery, so the controller
    parks at 0 when every speculative depth loses and re-enters through
    the existing bidirectional probes. ``t[0]`` is measured lazily when
    optimistic speculation cannot clear the estimated baseline, or from the
    bounded-staleness explorer. This keeps high-accept sequences from paying
    three unconditional plain cycles.
    Parking alone is not enough on fast backbones — a parked cycle still
    pays the MTP loop's synchronous host round-trip that the standard
    decoder pipelines away — so a sustained park hands the sequence back
    to the standard step entirely (``_park_mtp_to_standard``).

    Content-adaptive by construction: prose/chat settles at depth 1,
    code/predictable text climbs. Rejected alternatives (interleaved
    in-process A/B, rotated order, paired per rep, on Qwen3.6-35B-A3B +
    GLM-5.2, M3 Ultra): a fixed shallow-bias constant (won earlier separate-
    process comparisons only by masking the staleness lock; this design beats
    it on 3 of 4 model x content cells and ties the 4th), a pure realized
    tok/s bandit (exploration tax), a live per-cycle learned correction
    (decision churn), a frozen cross-generation correction (content
    oscillation), and a base x shape cost decomposition (unidentifiable while
    one depth runs for long stretches).
    """

    ALPHA = 0.08  # acceptance evidence discount (token domain, content-driven)
    # Conditional acceptance is a censored Bernoulli stream: position j is
    # observed only after positions <j were accepted.  A fixed EMA seed acts
    # like a strong prior and can therefore strand an unobserved deep tail.
    # Keep weak Jeffreys evidence instead, and use its one-sided normal
    # approximation only while a position has fewer than this many effective
    # observations. Mature exploitation uses the discounted empirical ratio;
    # the prior exists only to make an unobserved censored tail explorable.
    ACCEPT_PRIOR = 0.5
    CONFIDENCE_Z = 1.645
    ACCEPT_EXPLORE_TRIALS = 4.0
    TAU_MS = 400.0  # cost EMA horizon in wall-clock ms (load/thermal/context)
    PROBE_PERIOD_MS = 1000.0  # min wall-time between probes (light models)
    PROBE_PERIOD_MAX_MS = 5000.0  # staleness-exploration cadence floor
    PROBE_LEN = 4
    PROBE_PIPELINE_TAIL = 1  # target chain already built when a probe finishes
    PROBE_DUTY = 0.15  # probes never consume more than ~this share of cycles
    SPIKE_RATIO = 2.0  # a cycle above this * the EMA is treated as an outlier
    SPIKE_DAMP = 0.25  # ...and folded in at this fraction of the normal weight
    # Fallback prior for one extra verify token's cost, used only until two
    # depths have actually been measured; after that the marginal is the
    # measured slope between depths. 7 ms matches dense backbones (6-10 ms).
    MARGINAL_MS = 7.0
    HYSTERESIS = 1.03  # switch depth only for a >3% score gain
    COST_MIN_SAMPLES = 4
    BASELINE_MIN_SAMPLES = 3
    # Hand-off gate: ``t[0]`` is measured INSIDE the MTP loop, so it carries
    # the loop's synchronous host round-trip that the standard decoder
    # pipelines away. Speculation that cannot beat this taxed baseline by
    # EXIT_MARGIN is losing to the real standard step; after EXIT_STREAK
    # consecutive losing decisions the sequence leaves the MTP path
    # entirely (_park_mtp_to_standard). EXIT_MARGIN is only the
    # pre-measurement fallback (like MARGINAL_MS): each hand-off measures
    # the actual standard-step rate right after it and stores the real
    # loop tax on the model instance, which seeds later controllers via
    # the ``exit_margin`` constructor arg — machine/model-measured, not
    # a hardcoded ratio.
    EXIT_MARGIN = 1.15
    EXIT_STREAK = 16

    def __init__(
        self,
        max_depth: int,
        marginal_ms: Optional[float] = None,
        exit_margin: Optional[float] = None,
    ):
        if marginal_ms:
            self.MARGINAL_MS = float(marginal_ms)
        if exit_margin:
            self.EXIT_MARGIN = min(
                _STD_TAX_MAX, max(1.0, float(exit_margin))
            )
        self.max_depth = max(1, int(max_depth))
        self.cur = self.max_depth  # first cycle drafts deep; warmup sweeps down
        self.p = [0.5] * self.max_depth
        self._accept_hits = [0.0] * self.max_depth
        self._accept_trials = [0.0] * self.max_depth
        self.t: Dict[int, float] = {}
        self.t_age: Dict[int, float] = {}  # ms since each depth was measured
        self.t_samples: Dict[int, int] = {}
        # Exponentially weighted cost variance uses the same spike-damped
        # alpha as ``t``. ``_t_weight_sq`` tracks the squared normalized
        # weights, so 1/q is the effective (not lifetime) sample count.
        self._t_variance: Dict[int, float] = {}
        self._t_weight_sq: Dict[int, float] = {}
        self.cycles = 0
        self.probe_left = 0
        self._probe_target: Optional[int] = None
        self._probe_forced = False
        self.exit_streak = 0
        self._ms_probe = 0.0  # wall-time since any probe burst
        self._ms_explore = 0.0  # wall-time since a staleness-exploration burst
        self._ms_baseline = 0.0  # wall-time while depth 0 remains unmeasured
        # Measure each speculative depth once (max..1).  Depth 0 is calibrated
        # lazily only when even optimistic speculative acceptance makes the
        # plain baseline competitive; the five-second staleness explorer is a
        # backstop for an unexpectedly cheap baseline.  This avoids spending
        # three plain cycles on every high-accept sequence.
        self._warmup: List[int] = list(range(self.max_depth, 0, -1))

    def observe(
        self,
        used: int,
        accepted: int,
        cycle_ms: float,
        time_sample: bool = True,
    ) -> None:
        self.cycles += 1
        used = max(0, min(int(used), self.max_depth))
        accepted = max(0, min(int(accepted), used))
        # Acceptance: discounted conditional Bernoulli evidence. Decay every
        # position on every cycle so a tail that is no longer reached gradually
        # regains uncertainty and can be explored when content changes.
        keep = 1.0 - self.ALPHA
        self._accept_hits = [keep * value for value in self._accept_hits]
        self._accept_trials = [keep * value for value in self._accept_trials]
        self.p = [
            self._accept_posterior_mean(j) for j in range(self.max_depth)
        ]
        for j in range(used):
            hit = 1.0 if j < accepted else 0.0
            self._accept_hits[j] += hit
            self._accept_trials[j] += 1.0
            self.p[j] = self._accept_posterior_mean(j)
            if j >= accepted:
                break
        # Cost: wall-time-domain EMA with a one-off-spike guard, plus per-depth
        # ages so probes can target the estimate that is most stale.
        # ``time_sample=False`` marks cycles carrying a one-off maintenance
        # cost (a multi-block head keepalive refold) whose spike would bias
        # this depth's estimate; the acceptance update above still applies.
        cycle_ms = max(0.0, float(cycle_ms))
        if time_sample:
            self._update_time(used, cycle_ms)
        for d in list(self.t_age):
            self.t_age[d] += cycle_ms
        if time_sample:
            self.t_age[used] = 0.0
        self._ms_probe += cycle_ms
        self._ms_explore += cycle_ms
        if 0 not in self.t:
            self._ms_baseline += cycle_ms

        if self._speculation_losing():
            self.exit_streak += 1
        else:
            self.exit_streak = 0

        # Warmup/probe decisions are consumed by _chain_next_drafts after the
        # current chain has already been built. Advance their bookkeeping only
        # when the requested depth was actually observed, not on the one-cycle
        # pipeline lag.
        if self._warmup:
            expected = self._warmup[0]
            if used != expected or not time_sample:
                self.cur = expected
                return
            self._warmup.pop(0)
            if self._warmup:
                self.cur = self._warmup[0]
                return
            self.cur = self._best()
            self._ms_probe = 0.0
            target = self._best_rival()
            if target is not None and self._acceptance_underexplored(target):
                self._start_probe(target)
            return

        # Finishing a probe burst.
        if self.probe_left > 0:
            target = self._probe_target
            if (target is not None and used != target) or not time_sample:
                self.cur = target if target is not None else used
                return
            self.probe_left -= 1
            point_best = self._best_candidate()
            baseline_incomplete = bool(
                target == 0
                and self._cost_sample_count(0) < self.BASELINE_MIN_SAMPLES
            )
            separated = bool(
                target is not None
                and not baseline_incomplete
                and not self._probe_forced
                and not self._rival_needs_probe(target, point_best)
            )
            if self.probe_left == 0 or (
                not self._probe_forced and (point_best == target or separated)
            ):
                self._finish_probe()
                self._ms_probe = 0.0
            return

        # Re-decide every cycle (cheap); HYSTERESIS in _best prevents thrash.
        self.cur = self._best()

        # Probe scheduling: bounded-staleness re-measurement in either
        # direction, at a duty-bounded wall-clock cadence.
        if self.max_depth > 1:
            period = max(
                self.PROBE_PERIOD_MS,
                (self.PROBE_LEN + self.PROBE_PIPELINE_TAIL)
                * cycle_ms
                / self.PROBE_DUTY,
            )
            if self._ms_probe >= period:
                explore_due = self._ms_explore >= max(
                    self.PROBE_PERIOD_MAX_MS, 2.0 * period
                )
                target = (
                    self._most_stale() if explore_due else self._best_rival()
                )
                if target is not None:
                    self._start_probe(target, forced=explore_due)
                    if explore_due:
                        self._ms_explore = 0.0

    def _accept_posterior_mean(self, j: int) -> float:
        prior = self.ACCEPT_PRIOR
        successes = self._accept_hits[j]
        trials = self._accept_trials[j]
        # Jeffreys mass is an exploration prior, not permanent exploitation
        # evidence. Once the discounted stream is mature, use its empirical
        # ratio so a truly perfect tail converges to 1.0 instead of 0.963.
        if trials >= self.ACCEPT_EXPLORE_TRIALS:
            return successes / max(1e-12, trials)
        return (successes + prior) / (trials + 2.0 * prior)

    def _accept_upper(self, j: int) -> float:
        """Jeffreys-posterior upper bound for under-explored tails only."""

        trials = self._accept_trials[j]
        mean = self.p[j]
        if trials >= self.ACCEPT_EXPLORE_TRIALS:
            return mean
        alpha = self._accept_hits[j] + self.ACCEPT_PRIOR
        beta = max(0.0, trials - self._accept_hits[j]) + self.ACCEPT_PRIOR
        total = alpha + beta
        variance = alpha * beta / max(1e-12, total * total * (total + 1.0))
        return min(1.0, mean + self.CONFIDENCE_Z * math.sqrt(variance))

    def _acceptance_underexplored(self, depth: int) -> bool:
        return any(
            self._accept_trials[j] < self.ACCEPT_EXPLORE_TRIALS
            for j in range(depth)
        )

    def _start_probe(self, target: int, *, forced: bool = False) -> None:
        self.cur = int(target)
        self._probe_target = int(target)
        self._probe_forced = bool(forced)
        self.probe_left = self.PROBE_LEN
        self._ms_probe = 0.0

    def _finish_probe(self) -> None:
        self.probe_left = 0
        self._probe_target = None
        self._probe_forced = False
        # A forced probe target is not an incumbent. Applying switch
        # hysteresis relative to it can strand the controller at a measured
        # loser whose score happens to be within 3% of the winner.
        self.cur = self._best_candidate()

    def _time_alpha(self, cycle_ms: float) -> float:
        # EMA weight for a cycle of this wall-time: the memory horizon is
        # ~TAU_MS regardless of cycle duration, so responsiveness is constant
        # in real time whether a cycle is 8 ms (short) or 80 ms (128k context).
        return 1.0 - math.exp(-max(0.0, float(cycle_ms)) / self.TAU_MS)

    def _update_time(self, used: int, cycle_ms: float) -> None:
        cycle_ms = max(0.0, float(cycle_ms))
        count = self.t_samples.get(used, 0) + 1
        self.t_samples[used] = count
        prev = self.t.get(used)
        if prev is None:
            self.t[used] = cycle_ms
            self._t_variance[used] = 0.0
            self._t_weight_sq[used] = 1.0
            return
        # Deliberately a per-cycle EMA, NOT an irregular-sampling EMA weighted
        # by staleness age. Age-weighting (nearly replacing a stale estimate at
        # the first probe cycle) is the textbook form, but it was measured
        # WORSE here: single-cycle noise is ~±10%, so replacing an estimate
        # from a 4-cycle probe burst injects that noise straight into the depth
        # decision every probe (~1s), and prose re-over-drafted (-1.6%). The
        # slow EMA is a variance shield; stale errors are corrected by probe
        # REPETITION instead (each ~1s rival probe moves the estimate ~10% of
        # the gap, converging within a few seconds).
        a = self._time_alpha(cycle_ms)
        if cycle_ms > self.SPIKE_RATIO * prev:
            a *= self.SPIKE_DAMP  # a one-off spike moves the estimate slowly
        delta = cycle_ms - prev
        variance = (1.0 - a) * (
            self._t_variance.get(used, 0.0) + a * delta * delta
        )
        weight_sq = (
            (1.0 - a) ** 2 * self._t_weight_sq.get(used, 1.0) + a * a
        )
        estimate = (1.0 - a) * prev + a * cycle_ms
        if used == 0 and count <= self.BASELINE_MIN_SAMPLES:
            # A lazy baseline probe still rejects first-run shape inflation.
            # Keep the wider discounted variance, so this optimistic center
            # fails toward more evidence rather than a premature park.
            estimate = min(prev, cycle_ms)
        self.t[used] = estimate
        self._t_variance[used] = max(0.0, variance)
        self._t_weight_sq[used] = min(1.0, max(1e-12, weight_sq))

    def _marginal_est(self) -> float:
        # Measured cost of one extra verify row: the slope between the cheapest
        # and priciest measured depths. Falls back to the prior until two
        # depths exist. This is what self-calibrates the controller to the real
        # (model x chip x context) marginal instead of a hardcoded value.
        if len(self.t) >= 2:
            depths = sorted(self.t)
            lo, hi = depths[0], depths[-1]
            if hi > lo:
                slope = (self.t[hi] - self.t[lo]) / (hi - lo)
                if slope > 0.0:
                    return slope
        return self.MARGINAL_MS

    def _t_est(self, d: int) -> float:
        if d in self.t:
            return self.t[d]
        if not self.t:
            return 30.0 + self.MARGINAL_MS * d
        if d == 0:
            # The plain step sits below the L=1 -> L=2 verify jump, so the
            # per-row marginal says nothing about it. Estimate it at the
            # cheapest measured cycle: conservative (true t[0] is lower),
            # which keeps an unmeasured baseline from hijacking probes —
            # yet still useful as a conservative trigger when optimistic
            # speculative acceptance is poor enough to justify measuring it.
            return min(self.t.values())
        ref = min(self.t, key=lambda x: abs(x - d))
        return max(1e-3, self.t[ref] + self._marginal_est() * (d - ref))

    def _expected_tokens(self, d: int, *, optimistic: bool = False) -> float:
        expected = 1.0
        run = 1.0
        for j in range(d):
            probability = self._accept_upper(j) if optimistic else self.p[j]
            run *= probability
            expected += run
        return expected

    def _score(self, d: int, *, optimistic: bool = False) -> float:
        return self._expected_tokens(d, optimistic=optimistic) / max(
            1e-6,
            self._t_est(d),
        )

    def _cost_sample_count(self, d: int) -> int:
        # Unit tests and persisted seeds may inject t directly. Treat those as
        # mature point estimates; live measurements always populate t_samples.
        if d in self.t and d not in self.t_samples:
            return self.COST_MIN_SAMPLES
        return self.t_samples.get(d, 0)

    def _cost_bounds(self, d: int) -> tuple[float, float]:
        estimate = max(1e-6, self._t_est(d))
        count = self._cost_sample_count(d)
        if d not in self.t or count < self.COST_MIN_SAMPLES:
            return 0.0, math.inf
        variance = self._t_variance.get(d)
        weight_sq = self._t_weight_sq.get(d)
        if variance is None or weight_sq is None:
            # Directly seeded test/persisted estimates have no uncertainty
            # history. Live measurements always populate both maps.
            return estimate, estimate
        if variance <= 0.0:
            return estimate, estimate
        effective_samples = 1.0 / max(1e-12, weight_sq)
        half_width = self.CONFIDENCE_Z * math.sqrt(
            variance / max(1.0, effective_samples)
        )
        return max(1e-6, estimate - half_width), estimate + half_width

    def _score_bounds(self, d: int) -> tuple[float, float]:
        lower_time, upper_time = self._cost_bounds(d)
        lower = self._expected_tokens(d) / max(1e-6, upper_time)
        upper = self._expected_tokens(d, optimistic=True) / max(
            1e-6,
            lower_time,
        )
        return lower, upper

    def _speculation_losing(self) -> bool:
        # True when the best speculative depth cannot beat the (taxed)
        # in-loop baseline by EXIT_MARGIN. Only meaningful once the warmup
        # sweep has measured t[0].
        if (
            self._warmup
            or 0 not in self.t
            or self._cost_sample_count(0) < self.BASELINE_MIN_SAMPLES
        ):
            return False
        base = self._score(0)
        if base <= 0.0:
            return False
        best = max(self._score(d) for d in range(1, self.max_depth + 1))
        return best < base * self.EXIT_MARGIN

    def should_exit(self) -> bool:
        """Sustained losing speculation: hand the sequence back to the
        standard decoder."""
        return self.exit_streak >= self.EXIT_STREAK

    def reentry_win_proven(self) -> bool:
        """Require a mature plain-step baseline before clearing a cooldown."""

        return bool(
            not self._warmup
            and self._cost_sample_count(0) >= self.BASELINE_MIN_SAMPLES
            and not self._speculation_losing()
        )

    def prepare_reentry_probe(self) -> None:
        """Bound re-entry proof with three real depth-zero measurements."""

        if 0 not in self.t and 0 not in self._warmup:
            self._warmup.extend([0] * self.BASELINE_MIN_SAMPLES)

    def _select_candidates(self) -> List[int]:
        # Depth 0 is only selectable once its cost has actually been
        # measured (or seeded) — an extrapolated baseline must never PARK
        # the sequence, only motivate a probe.
        ds = list(range(1, self.max_depth + 1))
        if (
            0 in self.t
            and self._cost_sample_count(0) >= self.BASELINE_MIN_SAMPLES
        ):
            ds.insert(0, 0)
        return ds

    def _baseline_probe_worthwhile(self) -> bool:
        if 0 in self.t:
            return True
        if not self.t or self._warmup:
            return False
        baseline = 1.0 / max(1e-6, self._t_est(0))
        optimistic_speculation = max(
            self._score(d, optimistic=True)
            for d in range(1, self.max_depth + 1)
        )
        return optimistic_speculation < baseline * self.EXIT_MARGIN

    def _probe_candidates(self, *, include_baseline: bool = False) -> List[int]:
        candidates = list(range(1, self.max_depth + 1))
        if include_baseline or self._baseline_probe_worthwhile():
            candidates.append(0)
        return candidates

    def _baseline_stale_due(self) -> bool:
        if 0 in self.t:
            return True
        # A controller already exploiting below max depth has evidence that
        # speculative yield is limited, so the ordinary five-second explorer
        # measures the unknown plain baseline. A mature max-depth winner gets
        # one extra stale interval before paying that shape/calibration cost.
        delay = self.PROBE_PERIOD_MAX_MS
        if (
            self._best_candidate() == self.max_depth
            and not self._acceptance_underexplored(self.max_depth)
            and not self._baseline_probe_worthwhile()
        ):
            delay *= 2.0
        return self._ms_baseline >= delay

    def _rival_needs_probe(self, rival: int, incumbent: int) -> bool:
        if rival == incumbent:
            return False
        incumbent_score = self._score(incumbent)
        # Acceptance optimism is only an exploration gate. A rival whose
        # under-sampled tail still cannot approach the incumbent does not earn
        # a cost probe.
        if (
            self._cost_sample_count(rival) < self.COST_MIN_SAMPLES
            and self._score(rival, optimistic=True)
            < incumbent_score / self.HYSTERESIS
        ):
            return False
        incumbent_lower, _ = self._score_bounds(incumbent)
        _, rival_upper = self._score_bounds(rival)
        return rival_upper >= incumbent_lower / self.HYSTERESIS

    def _best_rival(self) -> Optional[int]:
        # Choose the rival with the highest acceptance-UCB score, then probe it
        # only while its empirical score interval can overlap exploitation.
        incumbent = (
            self.cur
            if self.cur in self._select_candidates()
            else self._best_candidate()
        )
        if self._score(incumbent) <= 0.0:
            return self._most_stale()
        rival = None
        rival_score = -1.0
        for d in self._probe_candidates():
            if d == incumbent:
                continue
            s = self._score(d, optimistic=True)
            if s > rival_score:
                rival, rival_score = d, s
        if rival is not None and self._rival_needs_probe(rival, incumbent):
            return rival
        return None

    def _most_stale(self) -> Optional[int]:
        # The depth whose cost estimate has gone longest unmeasured (never
        # measured counts as infinitely stale). Keeps every t[d] fresh enough
        # that fresh-vs-stale comparison bias stays bounded.
        cand = None
        worst = -1.0
        for d in self._probe_candidates(
            include_baseline=self._baseline_stale_due()
        ):
            if d == self.cur:
                continue
            age = self.t_age.get(d)
            age = float("inf") if age is None else age
            if age > worst:
                cand, worst = d, age
        return cand

    def _best_candidate(self) -> int:
        # argmax of measured score with switch hysteresis; ascending scan
        # with strict '>' keeps the shallower choice on an exact tie.
        best_d = self._select_candidates()[0]
        best_score = -1.0
        for d in self._select_candidates():
            s = self._score(d)
            if s > best_score:
                best_d, best_score = d, s
        return best_d

    def _best(self) -> int:
        best_d = self._best_candidate()
        best_score = self._score(best_d)
        if best_d != self.cur and best_score < self._score(self.cur) * self.HYSTERESIS:
            return self.cur
        return best_d


class _LockstepAcceptanceDepthController:
    """Zero-collective distributed depth adaptation from accepted prefixes.

    Every rank starts at the signed maximum and observes the same ``used`` /
    ``accepted`` pair because the coordinator already broadcasts the exact
    MTP decision packet. A rejection keeps one probe beyond the accepted
    prefix; a full accept climbs one level. No local clock, cost sample, random
    choice, or rank capability enters the decision, so ``cur`` remains
    identical without another scalar broadcast.
    """

    rank_lockstep = True

    def __init__(self, max_depth: int):
        self.max_depth = max(1, int(max_depth))
        self.cur = self.max_depth
        self.cycles = 0
        # The verify loop shares a small lifecycle interface with the local
        # adaptive controllers. Lockstep has no calibration sweep and never
        # parks for performance, but explicit empty/zero state keeps that
        # common path total rather than relying on controller-specific attrs.
        self._warmup: List[int] = []
        self.exit_streak = 0

    def observe(
        self,
        used: int,
        accepted: int,
        cycle_ms: float,
        time_sample: bool = True,
    ) -> None:
        del cycle_ms, time_sample
        self.cycles += 1
        used = max(1, min(int(used), self.max_depth))
        accepted = max(0, min(int(accepted), used))
        if accepted == used:
            self.cur = min(self.max_depth, used + 1)
        else:
            self.cur = max(1, min(self.max_depth, accepted + 1))

    def should_exit(self) -> bool:
        return False

    def reentry_win_proven(self) -> bool:
        # Lockstep never performs a cost-based handoff. If an existing parked
        # row is resumed after switching policies, it can leave cooldown
        # immediately without a fictitious timing calibration.
        return True


def _new_depth_controller(model: Any, max_depth: int) -> Any:
    if _lockstep_depth_enabled() or _qwen4_acceptance_depth_enabled(model):
        return _LockstepAcceptanceDepthController(max_depth)
    controller_type = (
        _EvidenceDepthController
        if _qwen4_evidence_depth_enabled(model)
        else _DepthController
    )
    return controller_type(
        max_depth,
        marginal_ms=getattr(model, "_omlx_mtp_marginal_ms", None),
        exit_margin=_effective_loop_tax(model),
    )


# Draft sampler for stochastic (temp > 0) decoding. A sharper distribution
# than the target: the 1-layer head's noisy tail otherwise gets sampled and
# rejected, collapsing acceptance on high-entropy content. Exactness holds
# because the Leviathan/Chen ratio uses this sampler's own distribution as q.
_DRAFT_SAMPLER_TEMP = 0.6
_DRAFT_SAMPLER_TOP_P = 0.95
_DRAFT_SAMPLER_TOP_K = 20


def _resolve_draft_sampler(gen_batch: Any, state: _MtpState):
    """Sampler used to draw MTP draft tokens.

    Greedy target → greedy drafts (preserves the greedy-identity contract).
    Stochastic target → the sharper draft sampler above. The acceptance ratio
    and residual sampling use this sampler's filtered distribution as q, so
    the emitted token distribution still equals the target sampler's exactly.
    """
    if state.draft_sampler is not None:
        return state.draft_sampler
    if _is_greedy(gen_batch):
        state.draft_sampler = _resolve_sampler(gen_batch)
        return state.draft_sampler

    from omlx.utils.sampling import make_sampler

    state.draft_sampler = make_sampler(
        temp=_DRAFT_SAMPLER_TEMP,
        top_p=_DRAFT_SAMPLER_TOP_P,
        top_k=_DRAFT_SAMPLER_TOP_K,
    )
    return state.draft_sampler


def _dspark_host(model: Any) -> Optional[Any]:
    """Return the model object that owns an active embedded DSpark head."""
    candidates = [model]
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            candidates.append(inner)
    for candidate in candidates:
        if getattr(candidate, "_omlx_dspark_decode_enabled", False):
            return candidate
    return None


def _dspark_next_drafts(
    gen_batch: Any,
    state: _MtpState,
    hidden_rows: Any,
    committed: Any,
    prev_buf: Optional[Any],
) -> None:
    """Append committed target taps and sample one DSpark block.

    The expensive three-stage decoder runs once over anchor+noise positions.
    A rank-R Markov head then samples left-to-right, preserving DSpark's
    intra-block dependency without another decoder pass.
    """
    import mlx.core as mx

    host = _dspark_host(gen_batch.model)
    if host is None:
        raise _MtpStepFallback("embedded DSpark host is unavailable")

    depth = state.controller.cur if state.controller is not None else state.depth
    if state.controller is not None and not getattr(
        state.controller, "rank_lockstep", False
    ):
        depth = _mtp_sync_depth(gen_batch, depth)
    depth = min(int(depth), int(getattr(host.args, "dspark_block_size", depth)))
    n = int(committed.shape[0])
    if depth <= 0:
        host.dspark_append_context(hidden_rows, state.mtp_cache)
        state.hist_offset += n
        state.drafts = mx.zeros((0,), dtype=mx.uint32)
        state.draft_lps = []
        state.draft_accept_lps = []
        return

    anchor = committed[-1:].reshape(1, 1)
    logits, _ = host.dspark_forward(
        hidden_rows,
        anchor,
        state.mtp_cache,
        draft_length=depth,
    )
    state.hist_offset += n

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    draft_toks: List[Any] = []
    draft_lps: List[Any] = []
    draft_accept_lps: List[Any] = []
    previous = anchor.reshape(1)

    # Drafts are speculative — processor calls shape the draft
    # distribution but must not advance the thinking budget (they would
    # count tokens that are only emitted if verified later). Checkpoint
    # before the loop and rewind after.
    snap = _snap_snapshotable(procs)

    for idx in range(depth):
        bias, _ = host.dspark_markov(previous)
        logits_2d = _mtp_prepare_logits(gen_batch, logits[:, idx, :] + bias)
        if procs is not None and prev_buf is not None:
            prefix = mx.concatenate(
                [prev_buf.astype(mx.int32), anchor.reshape(1).astype(mx.int32)]
                + [token.reshape(1).astype(mx.int32) for token in draft_toks]
            )
            logits_2d = _apply_processors(procs, prefix, logits_2d)
        lp_2d = _mtp_logprobs(gen_batch, logits_2d)
        token = _ensure_uint32(_mtp_sample(gen_batch, sampler, lp_2d))
        draft_toks.append(token)
        draft_lps.append(lp_2d.squeeze(0))
        draft_accept_lps.append(
            _mtp_accept_lp(gen_batch, sampler, lp_2d).squeeze(0)
        )
        previous = token.reshape(1)

    _restore_snapshotable(procs, snap)

    state.drafts = mx.concatenate(draft_toks)
    state.draft_lps = draft_lps
    state.draft_accept_lps = draft_accept_lps
    mx.async_eval(state.drafts)


def _chain_next_drafts(
    gen_batch: Any,
    state: _MtpState,
    hidden_rows: Any,
    committed: Any,
    prev_buf: Optional[Any],
) -> None:
    """Rebuild committed MTP-head history and draft the next chain.

    ``hidden_rows`` is the trunk hidden at the positions of the n tokens
    *preceding* each committed token — (1, n, H) pre-norm for Qwen (the
    final trunk norm is applied here), or the model's native 4D raw hidden
    for DeepSeek-V4 (passed through untouched); ``committed`` is the (n,)
    uint32 committed tokens. One batched head forward appends n committed
    history entries and yields the next cycle's first draft logits for free
    (its last entry pairs the newest committed token with the hidden of its
    predecessor — exactly the fused state that predicts the next-next token).
    The remaining drafts chain on the head's own output hidden; with
    ``state.head_clone`` those speculative steps run on a per-cycle clone so
    the persistent head cache stays committed-only.

    Populates ``state.drafts`` / ``draft_lps`` / ``draft_accept_lps`` and
    advances ``state.hist_offset`` by n. All arrays are dispatched with
    ``mx.async_eval`` and stay lazy on the host; the next verify cycle's
    single sync resolves them.
    """
    import mlx.core as mx

    model = gen_batch.model
    if _dspark_host(model) is not None:
        return _dspark_next_drafts(
            gen_batch,
            state,
            hidden_rows,
            committed,
            prev_buf,
        )
    sampler = _resolve_draft_sampler(gen_batch, state)
    procs = _proc_list(gen_batch)

    depth = state.controller.cur if state.controller is not None else state.depth
    if state.controller is not None and not getattr(
        state.controller, "rank_lockstep", False
    ):
        depth = _mtp_sync_depth(gen_batch, depth)
    if depth == 0 and not state.mtp_cache:
        # Depth-0 with a stateless head (no cache to keep warm, e.g. the
        # gemma4 assistant): skip the fold entirely — on fast backbones its
        # head forward + trunk norm is a measurable per-step tax (~15% of a
        # plain step on gemma4 26B) that would keep the parked throughput
        # below baseline. Head-history models keep folding below so their
        # cache stays consistent for re-entry.
        state.drafts = mx.zeros((0,), dtype=mx.uint32)
        state.draft_lps = []
        state.draft_accept_lps = []
        return

    # Models whose MTP head normalizes its hidden input internally
    # (inkling: per-block hidden_norm, chain_hidden_post_norm=False) mark
    # themselves and receive the raw pre-norm trunk hidden.
    head_prenorm = getattr(model, "_omlx_mtp_head_prenorm", False) or getattr(
        getattr(model, "_language_model", None), "_omlx_mtp_head_prenorm", False
    )
    if _HEAD_HIDDEN_POST_NORM and not head_prenorm and hidden_rows.ndim == 3:
        hidden_rows = _trunk_norm_module(model)(hidden_rows)

    # Multi-block heads (inkling) route fold/chain by a per-cycle pass
    # counter on the cache list; reset it before the fold. Single-block
    # heads have no hook and are unaffected.
    begin = getattr(model, "mtp_begin_cycle", None) or getattr(
        getattr(model, "_language_model", None), "mtp_begin_cycle", None
    )
    if begin is not None:
        begin(state.mtp_cache, depth)

    n = committed.shape[0]
    logits, head_hidden = model.mtp_forward(
        hidden_rows,
        committed.reshape(1, n),
        state.mtp_cache,
        return_hidden=True,
        logits_keep=1,
    )
    state.hist_offset += int(n)

    draft_toks: List[Any] = []
    draft_lps: List[Any] = []
    draft_accept_lps: List[Any] = []

    chain_prefix = committed[-1:]
    h = head_hidden[:, -1:]
    chain_cache = state.mtp_cache
    if state.head_clone and depth > 1:
        chain_cache = _clone_mtp_head_cache(state.mtp_cache)

    # Speculative draft shaping — see _dspark_next_drafts.
    snap = _snap_snapshotable(procs)

    for j in range(depth):
        logits_2d = _mtp_prepare_logits(gen_batch, logits[:, -1, :])
        if procs is not None and prev_buf is not None:
            prev = mx.concatenate(
                [prev_buf.astype(mx.int32), chain_prefix.astype(mx.int32)]
                + [t.reshape(1).astype(mx.int32) for t in draft_toks]
            )
            logits_2d = _apply_processors(procs, prev, logits_2d)
        lp_2d = _mtp_logprobs(gen_batch, logits_2d)
        tok = _ensure_uint32(_mtp_sample(gen_batch, sampler, lp_2d))
        draft_toks.append(tok)
        draft_lps.append(lp_2d.squeeze(0))
        draft_accept_lps.append(
            _mtp_accept_lp(gen_batch, sampler, lp_2d).squeeze(0)
        )
        if j + 1 == depth:
            break
        logits, head_hidden = model.mtp_forward(
            h,
            tok.reshape(1, 1),
            chain_cache,
            return_hidden=True,
        )
        h = head_hidden[:, -1:]

    _restore_snapshotable(procs, snap)

    if draft_toks:
        state.drafts = mx.concatenate(draft_toks)
        # Fire-and-forget dispatch: the GPU evaluates the chain while the
        # host finishes emit bookkeeping; the next cycle's sync finds it
        # materialized.
        mx.async_eval(state.drafts)
    else:
        # Depth 0 (controller escape hatch): no drafts — the next cycle
        # verifies [next_main] alone, i.e. a plain decode step. The fold
        # above still ran so head-history models stay warm for re-entry.
        state.drafts = mx.zeros((0,), dtype=mx.uint32)
    state.draft_lps = draft_lps
    state.draft_accept_lps = draft_accept_lps


def _materialize_distributed_hidden_sibling(
    logits: Any,
    hidden: Any,
    *,
    mx_module: Any = None,
) -> bool:
    """Evaluate logits and returned hidden in one distributed graph pass.

    Sampling materializes the logits dependency, but MLX does not necessarily
    materialize a lazy sibling output.  MTP subsequently feeds that hidden
    sibling into its local head; on TP ranks this used to replay the backbone's
    already-consumed collective graph and fence every rank together.  Bind the
    two outputs to the same evaluation only for a real distributed world. The
    single-node path keeps its existing lazy overlap and pays no extra sync.
    """

    if mx_module is None:
        import mlx.core as mx_module

    try:
        group = mx_module.distributed.init()
        distributed = int(group.size()) > 1
    except (AttributeError, RuntimeError, TypeError, ValueError):
        distributed = False
    if distributed:
        mx_module.eval(logits, hidden)
    return distributed


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _qwen4_probe_transaction_snapshot(
    gen_batch: Any,
    state: "_MtpState",
) -> dict[str, Any]:
    qsa = []
    recurrent_counts = []
    rollback = []
    pending_caches = list(getattr(gen_batch, "prompt_cache", ()) or ())
    while pending_caches:
        cache = pending_caches.pop(0)
        pending_caches[0:0] = list(getattr(cache, "caches", ()) or ())
        name = type(cache).__name__
        if name in _QWEN4_QSA_CACHE_TYPES:
            qsa.append({"type": name, "offsets": _qwen4_qsa_offsets(cache)})
        count = getattr(cache, "_token_count", None)
        if type(count) is int:
            recurrent_counts.append(count)
        markers = [
            attr
            for attr in (
                "rollback_state",
                "_qwen4_exp_ple_speculative_state",
                "_mtp_undo",
                "_mtp_draft_stash",
            )
            if getattr(cache, attr, None) is not None
        ]
        if markers:
            rollback.append({"type": name, "markers": markers})

    pending = getattr(state, "pending_commit", None)
    pending_report = None
    if pending is not None:
        pending_report = {
            "kind": pending.kind,
            "target_base_offset": pending.target_base_offset,
            "verify_width": pending.verify_width,
            "accepted": pending.accepted,
            "emitted": pending.emitted,
            "source_map": list(pending.source_map),
            "token_map": list(pending.token_map),
        }
    return {
        "target_offset": _qwen4_target_offset(gen_batch.prompt_cache),
        "streamed_tokens": len(getattr(gen_batch, "tokens", [[]])[0]),
        "hist_offset": int(getattr(state, "hist_offset", 0)),
        "queue": [
            {"token": int(token), "source": source}
            for token, _logprobs, source in getattr(state, "queue", ())
        ],
        "pending_commit": pending_report,
        "pending_emit": getattr(state, "pending_emit", None),
        "qsa": qsa,
        "recurrent_counts": recurrent_counts,
        "rollback": rollback,
    }


def _record_qwen4_probe_timeline_event(
    gen_batch: Any,
    state: "_MtpState",
    event: str,
) -> None:
    if not os.environ.get(_QWEN4_VERIFY_PARITY_PATH_ENV, "").strip():
        return
    events = getattr(state, "_omlx_verify_parity_timeline", None)
    if events is None:
        events = []
        state._omlx_verify_parity_timeline = events
    if len(events) < 12:
        events.append(
            {
                "event": event,
                "state": _qwen4_probe_transaction_snapshot(gen_batch, state),
            }
        )


def _maybe_probe_qwen4_verify_parity(
    gen_batch: Any,
    state: "_MtpState",
    inputs: Any,
) -> bool:
    """Replay an observed Qwen4 verify window against scalar target decode.

    The gate is deliberately an explicit file path rather than a boolean.  An
    unset environment performs no import, token copy, host synchronization, or
    filesystem access.  The diagnostic replays against fresh caches and never
    mutates the active target/MTP transaction.  It is expensive real-model
    work and is intended only for a coordinated offline maintenance run.

    Returns ``True`` when a probe was attempted so the caller can exclude its
    wall time from adaptive-controller economics.
    """

    output_path = os.environ.get(_QWEN4_VERIFY_PARITY_PATH_ENV, "").strip()
    if not output_path:
        return False
    if (
        not _is_qwen4_exp_model(getattr(gen_batch, "model", None))
        or getattr(gen_batch, "_omlx_rowwise_mtp", False)
        or len(getattr(gen_batch, "uids", ()) or ()) != 1
    ):
        return False

    cycle_index = int(getattr(state.stats, "cycles", 0))
    if cycle_index >= _positive_env_int(_QWEN4_VERIFY_PARITY_CYCLES_ENV, 1):
        return False

    streamed = list(getattr(gen_batch, "tokens", [[]])[0])
    try:
        verify_tokens = [int(token) for token in inputs.reshape(-1).tolist()]
        if len(streamed) < 2 or not verify_tokens:
            raise ValueError("active token timeline is too short for a verify probe")
        if streamed[-1] != verify_tokens[0]:
            raise ValueError(
                "active pipeline tail does not match the first verify input "
                f"({streamed[-1]} != {verify_tokens[0]})"
            )
        committed_prefix = streamed[:-1]
        active_offset = _qwen4_target_offset(gen_batch.prompt_cache)
        if active_offset != len(committed_prefix):
            raise ValueError(
                "active target cache does not represent the committed probe prefix "
                f"({active_offset} != {len(committed_prefix)})"
            )

        from ..mlx_vlm_qwen4_exp_compat.verify_parity import (
            prepare_qwen4_active_verify_probe,
        )

        extract = getattr(gen_batch, "extract_cache", None)
        active_prefix_cache = (
            extract(0) if callable(extract) else gen_batch.prompt_cache
        )
        probe = prepare_qwen4_active_verify_probe(
            gen_batch.model,
            committed_prefix_tokens=committed_prefix,
            verify_tokens=verify_tokens,
            prefill_step=_positive_env_int(
                _QWEN4_VERIFY_PARITY_PREFILL_STEP_ENV,
                4096,
            ),
            active_prefix_cache=active_prefix_cache,
        )
        probe.report.update(
            {
                "uid": str(gen_batch.uids[0]),
                "cycle": cycle_index,
                "active_target_offset": active_offset,
                "active_streamed_tokens": len(streamed),
                "active_pre_transaction": _qwen4_probe_transaction_snapshot(
                    gen_batch,
                    state,
                ),
                "init_transaction_timeline": list(
                    getattr(state, "_omlx_verify_parity_timeline", ())
                ),
            }
        )
        state._omlx_active_verify_parity_probe = probe
    except Exception as exc:
        logger.exception("Qwen4 verify parity probe failed: %s", exc)
        try:
            from ..mlx_vlm_qwen4_exp_compat.verify_parity import append_report

            append_report(
                output_path,
                {
                    "schema_version": 1,
                    "created_unix": time.time(),
                    "uid": str((getattr(gen_batch, "uids", ()) or ("?",))[0]),
                    "cycle": cycle_index,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        except Exception:
            logger.debug("Qwen4 verify parity error report failed", exc_info=True)
    return True


def _capture_qwen4_active_verify_parity(
    gen_batch: Any,
    state: "_MtpState",
    rows: Any,
) -> None:
    probe = getattr(state, "_omlx_active_verify_parity_probe", None)
    if probe is None:
        return
    try:
        from ..mlx_vlm_qwen4_exp_compat.verify_parity import (
            capture_qwen4_active_verify_result,
        )

        extract = getattr(gen_batch, "extract_cache", None)
        active_cache = extract(0) if callable(extract) else gen_batch.prompt_cache
        report = capture_qwen4_active_verify_result(
            probe,
            active_logits=rows,
            active_cache=active_cache,
        )
        report["active_post_forward_transaction"] = (
            _qwen4_probe_transaction_snapshot(gen_batch, state)
        )
    except Exception as exc:
        probe.report["active_capture_error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Qwen4 active verify parity capture failed: %s", exc)


def _finish_qwen4_active_verify_parity(
    gen_batch: Any,
    state: "_MtpState",
    *,
    target_ids: Optional[List[int]],
    draft_ids: List[int],
    accepted: int,
    emitted_id: int,
) -> None:
    probe = getattr(state, "_omlx_active_verify_parity_probe", None)
    if probe is None:
        return
    report = probe.report
    decision: dict[str, Any] = {
        "target_ids": target_ids,
        "draft_ids": list(draft_ids),
        "accepted": int(accepted),
        "emitted_id": int(emitted_id),
    }
    active_rows = ((report.get("active") or {}).get("rows") or [])
    active_ids = [row["active"]["top1_id"] for row in active_rows]
    if (
        target_ids is not None
        and len(active_ids) >= len(draft_ids) + 1
        and draft_ids
    ):
        expected_accepted = 0
        for target, draft in zip(active_ids[:-1], draft_ids):
            if target != draft:
                break
            expected_accepted += 1
        expected_emitted = active_ids[
            expected_accepted if expected_accepted < len(draft_ids) else len(draft_ids)
        ]
        decision.update(
            {
                "active_target_ids": active_ids,
                "expected_accepted": expected_accepted,
                "expected_emitted_id": expected_emitted,
                "alignment_valid": bool(
                    int(accepted) == expected_accepted
                    and int(emitted_id) == expected_emitted
                ),
            }
        )
    report["decision"] = decision
    report["active_post_decision_transaction"] = (
        _qwen4_probe_transaction_snapshot(gen_batch, state)
    )
    try:
        from ..mlx_vlm_qwen4_exp_compat.verify_parity import append_report

        append_report(
            os.environ[_QWEN4_VERIFY_PARITY_PATH_ENV],
            report,
        )
        logger.warning(
            "Qwen4 active verify probe cycle=%s fresh_argmax=%s "
            "active_scalar=%s pre_cache=%s post_cache=%s alignment=%s",
            report.get("cycle"),
            report.get("argmax_parity"),
            all(
                row.get("active_vs_scalar_argmax")
                for row in active_rows
            ),
            (report.get("active_pre_vs_fresh_prefix_cache") or {}).get(
                "bitwise_equal"
            ),
            ((report.get("active") or {}).get("post_cache_vs_scalar") or {}).get(
                "bitwise_equal"
            ),
            decision.get("alignment_valid"),
        )
    except Exception as exc:
        logger.exception("Qwen4 active verify parity report failed: %s", exc)
    finally:
        try:
            delattr(state, "_omlx_active_verify_parity_probe")
        except AttributeError:
            pass
        try:
            delattr(state, "_omlx_verify_parity_timeline")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Post-init: run one extra backbone forward + MTP forward; queue the two
# emitted tokens; stash a draft for the first verify cycle.
# ---------------------------------------------------------------------------


def _post_init_mtp(gen_batch: Any) -> None:
    """Bridge from standard ``__init__``'s ``_step()`` into PR 990's cycle 1.

    State on entry (after standard ``__init__``):
      - cache contains the prompt up to ``prompt[-1]`` inclusive
      - ``_next_tokens`` = ``main_tok`` (token sampled from ``prompt[-1]``'s logits)
      - ``_next_logprobs[0]`` = main_tok's distribution
      - ``tokens[0]`` = original prompt list

    We perform one more 1-token backbone forward (so the cache also includes
    ``main_tok`` and we obtain the hidden state at that position), run the
    MTP head to produce a draft for the next verify cycle, and seed
    ``state.queue`` with two confirmed tokens — ``main_tok`` and the
    standard-sample at the next position. After this, the queue handles
    the first two emit calls and the third call enters the verify cycle.

    If the batch was empty when ``__init__`` ran, ``_next_tokens`` is
    ``None`` — we leave MTP inactive and the standard path runs unchanged.
    """
    import mlx.core as mx

    if gen_batch._next_tokens is None or not gen_batch.uids:
        # Nothing was sampled in the standard _step (empty batch). The
        # next() call will be a no-op anyway; leave the patch inert.
        return

    qwen4_terminal = _model_qwen4_terminal_commit_enabled(gen_batch.model)
    if qwen4_terminal:
        gen_batch._omlx_standard_target_exact_v1 = False

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    main_tok = _ensure_uint32(gen_batch._next_tokens)  # (1,)
    main_lp = gen_batch._next_logprobs[0]  # (vocab,)

    if procs is not None:
        prev_buf = gen_batch._token_context[0].update_and_fetch(main_tok)
    else:
        prev_buf = None

    # 1-token backbone forward at main_tok with hidden state. No draft yet,
    # so no rollback is possible — discard gdn_states.
    # Inherits the per-engine stream from the enclosing BatchGenerator context.
    logits, hidden, _ = _call_backbone(
        gen_batch.model, main_tok[:, None], gen_batch.prompt_cache
    )
    _materialize_distributed_hidden_sibling(logits, hidden, mx_module=mx)
    _clear_rollback(gen_batch.prompt_cache)

    next_main_logits = _mtp_prepare_logits(
        gen_batch, logits[:, -1, :]
    )  # (1, vocab) — distribution after main_tok
    next_main_logits = _apply_processors(procs, prev_buf, next_main_logits)
    next_main_lp = _mtp_logprobs(gen_batch, next_main_logits)
    next_main_tok = _mtp_sample(gen_batch, sampler, next_main_lp)  # (1,)

    chain, depth, head_clone = _resolve_mtp_chain_depth(gen_batch.model)

    if chain:
        # Depth-k seed: the history fold pairs hidden(main_tok) with
        # next_main_tok — the first committed history entry — and its logits
        # are the first draft's distribution; the rest of the chain follows.
        mx.eval(main_tok, next_main_tok)
        state = _MtpState(uid=gen_batch.uids[0])
        if qwen4_terminal:
            # Publish ownership before fallible head/transaction setup so a
            # partial activation can be reconciled rather than resuming stock
            # decode against an already-advanced target cache.
            gen_batch._omlx_mtp_state = state
        state.chain = True
        state.depth = depth
        state.head_clone = head_clone
        fixed = _fixed_depth_override(depth)
        if fixed is not None:
            state.depth = fixed
        elif depth > 1:
            state.controller = _new_depth_controller(
                gen_batch.model, depth
            )
        primed = _prompt_priming.take_primed(
            gen_batch.model, gen_batch.prompt_cache, main_tok
        )
        if primed is None or not _adopt_primed_head_state(
            state,
            primed,
            gen_batch.prompt_cache,
        ):
            state.mtp_cache = gen_batch.model.make_mtp_cache()
        state.next_main = _ensure_uint32(next_main_tok)
        main_id = int(main_tok.tolist()[0])
        next_main_id = int(next_main_tok.tolist()[0])
        state.queue.append((main_id, main_lp, "init"))
        state.queue.append(
            (next_main_id, next_main_lp.squeeze(0), "init")
        )
        head_base_offset = int(state.hist_offset)
        _chain_next_drafts(
            gen_batch,
            state,
            hidden[:, -1:],
            state.next_main,
            prev_buf,
        )
        if (
            qwen4_terminal
            and not getattr(gen_batch, "_omlx_rowwise_mtp", False)
        ):
            target_offset = _qwen4_target_offset(gen_batch.prompt_cache)
            if target_offset is None or not _qwen4_reconcile_sized_recurrent_timeline(
                gen_batch.prompt_cache,
                expected=target_offset or 0,
                allowed_current={target_offset} if target_offset is not None else set(),
            ):
                raise _MtpStepFallback(
                    "Qwen4 activation target cache timeline is not exact"
                )
            align = int(getattr(gen_batch.model, "_omlx_mtp_commit_align", 0) or 0)
            final_count = len(gen_batch.tokens[0]) + 2
            state.pending_commit = _MtpPendingCommit(
                kind="init",
                target_base_offset=target_offset,
                head_base_offset=head_base_offset,
                verify_width=0,
                accepted=0,
                source_map=("init-resident", "init-tail"),
                token_map=(main_id, next_main_id),
                head_committed_offset=int(state.hist_offset),
                deferred_boundary=bool(
                    align > 0 and final_count % align == 0
                ),
                final_source="init-tail",
            )
            _record_qwen4_probe_timeline_event(
                gen_batch,
                state,
                "post-init",
            )
        gen_batch._omlx_mtp_state = state
        return

    # MTP head sees (hidden_at_main, next_main_tok) and proposes the draft
    # that the *next* verify cycle will check against forward([next_main, draft]).
    # The legacy depth-1 cycle rebuilds head history per cycle and never
    # consumes a primed cache; release any capture leftovers.
    _prompt_priming.drop_ctx(gen_batch.model)
    mtp_cache = gen_batch.model.make_mtp_cache()
    hidden_at_main = hidden[:, -1:, :]  # (1, 1, H)
    next_ids = next_main_tok.reshape(1, 1)
    mtp_logits = gen_batch.model.mtp_forward(hidden_at_main, next_ids, mtp_cache)
    mtp_logits_2d = _mtp_prepare_logits(gen_batch, mtp_logits[:, -1, :])
    # The seed draft is speculative — shape but do not count.
    snap = _snap_snapshotable(procs)
    if procs is not None:
        prev_with_main_and_next = mx.concatenate(
            [prev_buf, _ensure_uint32(next_main_tok)]
        )
        mtp_logits_2d = _apply_processors(procs, prev_with_main_and_next, mtp_logits_2d)
    _restore_snapshotable(procs, snap)
    draft_lp_2d = _mtp_logprobs(gen_batch, mtp_logits_2d)
    draft_tok = _mtp_sample(gen_batch, sampler, draft_lp_2d)
    # Filtered draft lp — what the sampler actually drew from. The next
    # cycle's acceptance ratio uses this so the math matches the
    # sampling distribution rather than the raw softmax.
    draft_accept_lp_2d = _mtp_accept_lp(
        gen_batch, sampler, draft_lp_2d
    )

    mx.eval(main_tok, next_main_tok, draft_tok)

    # Queue the two confirmed tokens (main_tok + next_main_tok); their
    # logprobs come from the standard / patched samplers. Cache draft_id
    # while the array is already evaluated to avoid re-syncing in cycle 1.
    state = _MtpState(uid=gen_batch.uids[0])
    state.mtp_cache = mtp_cache
    state.next_main = _ensure_uint32(next_main_tok)
    state.draft_tok = _ensure_uint32(draft_tok)
    state.draft_lp = draft_lp_2d.squeeze(0)
    state.draft_accept_lp = draft_accept_lp_2d.squeeze(0)
    state.draft_id = int(draft_tok.tolist()[0])
    state.queue.append((int(main_tok.tolist()[0]), main_lp, "init"))
    state.queue.append(
        (int(next_main_tok.tolist()[0]), next_main_lp.squeeze(0), "init")
    )

    gen_batch._omlx_mtp_state = state


# ---------------------------------------------------------------------------
# next() dispatch
# ---------------------------------------------------------------------------


def _mtp_batch_next(gen_batch: Any, batch_state: _MtpBatchState) -> Any:
    """Emit one token per row using independent MTP state per active uid.

    This is intentionally conservative: rows whose queues are empty are
    advanced through the proven singleton MTP cycle against extracted row
    caches, then the modified rows are merged back into the batched cache.
    That keeps continuous-batching ownership correct while enabling MTP in
    multi-request decode without sharing singleton state across rows.
    """
    if not getattr(gen_batch, "uids", None):
        return []

    replacements: Dict[int, List[Any]] = {}
    token_context_updates: Dict[int, Any] = {}

    for idx, uid in enumerate(list(gen_batch.uids)):
        state = batch_state.states.get(uid)
        if state is None:
            raise _MtpStepFallback(f"missing row state for uid={uid}")
        if state.queue:
            continue

        row = _make_row_batch(
            gen_batch,
            idx,
            prompt_cache=gen_batch.extract_cache(idx),
            state=state,
        )
        _set_singleton_mrope_delta(row)
        _run_verify_cycle(row, state)
        if not state.queue:
            raise _MtpStepFallback(f"row uid={uid} verify produced no tokens")
        replacements[idx] = row.prompt_cache
        token_context_updates[idx] = row._token_context[0]

    _replace_cache_rows(gen_batch, replacements)
    for idx, token_context in token_context_updates.items():
        gen_batch._token_context[idx] = token_context

    return _emit_batch_responses(gen_batch, batch_state)


def _emit_batch_responses(gen_batch: Any, batch_state: _MtpBatchState) -> List[Any]:
    Response = type(gen_batch).Response

    keep = []
    responses = []
    finished_uids = []

    for idx, uid in enumerate(list(gen_batch.uids)):
        state = batch_state.states.get(uid)
        if state is None or not state.queue:
            raise _MtpStepFallback(f"row uid={uid} has no queued token")

        token_id, logprobs_1d, source = state.queue.popleft()
        token_id = _validated_emitted_token(token_id, logprobs_1d)
        _bump_emit_stat(state, source)

        finish_reason: Optional[str] = None
        match_sequence = None

        gen_batch.tokens[idx].append(token_id)
        gen_batch._num_tokens[idx] += 1
        if gen_batch._num_tokens[idx] >= gen_batch.max_tokens[idx]:
            finish_reason = "length"

        new_state, match_sequence, current_state = gen_batch.state_machines[idx].match(
            gen_batch._matcher_states[idx],
            token_id,
        )
        gen_batch._matcher_states[idx] = new_state
        if match_sequence is not None and current_state is None:
            finish_reason = "stop"

        if finish_reason is not None:
            responses.append(
                Response(
                    uid=uid,
                    token=token_id,
                    logprobs=logprobs_1d,
                    finish_reason=finish_reason,
                    current_state=current_state,
                    match_sequence=match_sequence,
                    prompt_cache=gen_batch.extract_cache(idx),
                    all_tokens=gen_batch.tokens[idx],
                )
            )
            _log_mtp_stats(uid, state.stats, finish_reason)
            finished_uids.append(uid)
        else:
            keep.append(idx)
            responses.append(
                Response(
                    uid=uid,
                    token=token_id,
                    logprobs=logprobs_1d,
                    finish_reason=None,
                    current_state=current_state,
                    match_sequence=match_sequence,
                    prompt_cache=None,
                    all_tokens=None,
                )
            )

    for uid in finished_uids:
        batch_state.states.pop(uid, None)

    if len(keep) < len(gen_batch.uids):
        gen_batch.filter(keep)

    return responses


def _feed_next_main_to_standard(gen_batch: Any, state: _MtpState) -> bool:
    """Materialize ``state.next_main`` and sample its successor.

    At a cycle boundary with an empty queue the cache is exactly one token
    behind the streamed sequence: ``state.next_main`` (already streamed) has
    no KV yet. Feed it through the backbone, sample ``_next_tokens`` from
    the resulting logits, and leave the batch in the standard-resumable
    state. Shared by the depth-0 park and the late-join handoff. Returns
    False on failure with the batch untouched.
    """
    import mlx.core as mx

    if state.next_main is None:
        return False
    qwen4_terminal = _model_qwen4_terminal_commit_enabled(gen_batch.model)
    target_before = (
        _qwen4_target_offset(gen_batch.prompt_cache) if qwen4_terminal else None
    )
    if qwen4_terminal and target_before is None:
        return False
    try:
        procs = _proc_list(gen_batch)
        _set_singleton_mrope_delta(gen_batch)
        prev_buf = None
        if procs is not None:
            prev_buf = gen_batch._token_context[0].update_and_fetch(state.next_main)
        logits, _, _ = _call_backbone(
            gen_batch.model, state.next_main[:, None], gen_batch.prompt_cache
        )
        last = _mtp_prepare_logits(gen_batch, logits[:, -1, :])
        last = _apply_processors(procs, prev_buf, last)
        lp_2d = _mtp_logprobs(gen_batch, last)
        next_tok = _ensure_uint32(
            _mtp_sample(gen_batch, _resolve_sampler(gen_batch), lp_2d)
        )
        mx.eval(next_tok)
        gen_batch._next_tokens = next_tok
        gen_batch._next_logprobs = [lp_2d.squeeze(0)]
    except Exception as exc:
        logger.debug("MTP feed-to-standard handoff failed: %s", exc)
        return False
    _clear_rollback(gen_batch.prompt_cache)
    if qwen4_terminal:
        expected = int(target_before) + 1
        if (
            _qwen4_target_offset(gen_batch.prompt_cache) != expected
            or not _qwen4_reconcile_sized_recurrent_timeline(
                gen_batch.prompt_cache,
                expected=expected,
                allowed_current={expected},
            )
        ):
            return False
    gen_batch._omlx_standard_target_exact_v1 = True
    return True


def _park_mtp_to_standard(gen_batch: Any, state: _MtpState) -> bool:
    """Hand a parked sequence back to the standard pipelined decoder.

    At a depth-0 cycle boundary the cache is committed-only and compact, so
    unlike ``_reconcile_mtp_to_standard`` no re-prefill is needed: feed
    ``state.next_main`` (already streamed, not yet in the cache), sample its
    successor as ``_next_tokens``, and drop the MTP state. The batch is
    marked for a cooldown so the standard step's async pipelining can run
    without the MTP loop tax. A later singleton probe creates a fresh depth
    controller; repeated failed probes exponentially extend the cooldown.
    """
    if not _feed_next_main_to_standard(gen_batch, state):
        return False
    park_state = _mtp_park_state_for_batch(gen_batch)
    if state.reentry_probe and park_state is not None:
        park_state.restart_after_failed_probe()
    else:
        park_state = _new_mtp_park_state(state.uid)
        gen_batch._omlx_mtp_park_state = park_state
    if state.controller is not None:
        _arm_std_tax_probe(gen_batch, state.controller.t.get(0), state.uid)
    logger.info(
        "MTP[%s] parked for %d standard tokens before re-entry probe",
        state.uid,
        park_state.cooldown_tokens,
    )
    state._finish_reason = "parked"
    _drop_mtp_state(gen_batch, "parked-at-depth-0", log_stats=True)
    return True


def _handoff_mtp_for_late_join(gen_batch: Any, state: _MtpState) -> bool:
    """Hand a singleton MTP decode to the standard step for a late join.

    A pending prefill can only merge into this batch through mlx-lm's
    promotion path, which the active-MTP completion pin blocks (#2515). At
    the drained-queue boundary the handoff is exact and cheap: with one
    queued token left it is the only committed token whose KV is absent
    from the cache, so it becomes ``_next_tokens`` verbatim and its stored
    logprobs keep the standard step's emission byte-identical; with an
    empty queue the park-style 1-token forward re-derives the same state.
    Unlike ``_park_mtp_to_standard`` this starts no performance cooldown and
    arms no std-tax probe: the sequence is yielding to a batch merge, not
    losing to standard decode, and must regain MTP once it is a compact
    singleton again.
    """
    import mlx.core as mx

    if len(state.queue) > 1:
        return False
    if len(state.queue) == 1:
        if _model_qwen4_terminal_commit_enabled(gen_batch.model):
            current = _qwen4_target_offset(gen_batch.prompt_cache)
            if current is None or not _qwen4_reconcile_sized_recurrent_timeline(
                gen_batch.prompt_cache,
                expected=current,
                allowed_current={current},
            ):
                return False
        token_id, logprobs_1d, _src = state.queue[0]
        gen_batch._next_tokens = mx.array([int(token_id)], dtype=mx.uint32)
        gen_batch._next_logprobs = [logprobs_1d]
        _clear_rollback(gen_batch.prompt_cache)
        gen_batch._omlx_standard_target_exact_v1 = True
    elif not _feed_next_main_to_standard(gen_batch, state):
        return False
    if state.reentry_probe:
        park_state = _mtp_park_state_for_batch(gen_batch)
        if park_state is not None:
            park_state.defer_probe()
    state._finish_reason = "late-join-handoff"
    _drop_mtp_state(gen_batch, "late-join-handoff", log_stats=True)
    return True


def _mark_qwen4_pending_emit(
    state: _MtpState,
    token_id: int,
    source: str,
) -> None:
    pending = state.pending_commit
    if pending is None:
        return
    if state.pending_emit is not None:
        raise _MtpStepFallback("Qwen4 emitted a second token before scheduler ACK")
    position = pending.emitted
    if position >= len(pending.token_map):
        raise _MtpStepFallback("Qwen4 pending commit source map is exhausted")
    expected_source = pending.source_map[position]
    source_matches = expected_source == source or (
        expected_source.startswith("init-") and source == "init"
    )
    if not source_matches or pending.token_map[position] != int(token_id):
        raise _MtpStepFallback(
            "Qwen4 pending commit does not match emitted queue position"
        )
    state.pending_emit = (position, int(token_id), source)


def _qwen4_post_emit_transaction(
    gen_batch: Any,
    state: _MtpState,
    *,
    terminal: bool,
) -> _MtpPostEmitResult:
    """Resolve one Qwen4 queue position after scheduler stop decisions."""

    pending = state.pending_commit
    receipt = state.pending_emit
    if pending is None or receipt is None:
        return _MtpPostEmitResult(reason="no-pending-transaction")
    if (
        pending.head_committed_offset is not None
        and int(state.hist_offset) != pending.head_committed_offset
    ):
        raise _MtpStepFallback(
            "Qwen4 MTP-head transaction offset changed before scheduler ACK"
        )
    position, token_id, _source = receipt
    if position != pending.emitted or position >= len(pending.token_map):
        return _MtpPostEmitResult(
            handled=True,
            reason="queue-position-mismatch",
        )

    exact = True
    if terminal:
        if pending.kind == "verify-sequential":
            if position < pending.accepted:
                # The live cache currently owns the complete accepted prefix.
                # An earlier parser/text stop selects a shorter prefix by
                # restoring the detached recurrent/QSA base and replaying only
                # next_main plus the terminal accepted drafts.
                exact = _qwen4_commit_sequential_verify_to(
                    gen_batch,
                    state,
                    pending,
                    accepted=position + 1,
                )
            elif position == pending.accepted:
                exact = _qwen4_commit_sequential_verify_to(
                    gen_batch,
                    state,
                    pending,
                    accepted=pending.accepted,
                ) and _qwen4_materialize_target_tail(
                    gen_batch,
                    state,
                    token_id,
                )
            else:
                exact = False
        elif pending.kind == "verify":
            if position < pending.accepted:
                # Terminal on accepted draft j: retain confirmed next_main and
                # exactly drafts d1..d(j+1), discarding every later verified
                # row before it ever becomes resident-visible.
                exact = _qwen4_rollback_full_verify_to(
                    gen_batch,
                    state,
                    pending,
                    accepted=position + 1,
                )
            elif position == pending.accepted:
                # The correction/bonus is predicted by the verifier but is not
                # one of its input rows. Select the accepted prefix once, then
                # append this terminal token with an ordinary L=1 target call.
                exact = _qwen4_rollback_full_verify_to(
                    gen_batch,
                    state,
                    pending,
                    accepted=pending.accepted,
                ) and _qwen4_materialize_target_tail(
                    gen_batch,
                    state,
                    token_id,
                )
            else:
                exact = False
        elif pending.kind == "init":
            if position == 0:
                exact = (
                    _qwen4_target_offset(gen_batch.prompt_cache)
                    == pending.target_base_offset
                    and _qwen4_reconcile_sized_recurrent_timeline(
                        gen_batch.prompt_cache,
                        expected=pending.target_base_offset,
                        allowed_current={pending.target_base_offset},
                    )
                )
            elif position == 1:
                exact = _qwen4_materialize_target_tail(
                    gen_batch,
                    state,
                    token_id,
                )
            else:
                exact = False
        elif pending.kind == "tail" and position == 0:
            exact = _qwen4_materialize_target_tail(
                gen_batch,
                state,
                token_id,
            )
        else:
            exact = False

        state.pending_emit = None
        state.pending_commit = None
        state.queue.clear()
        state.drafts = None
        state.draft_lps.clear()
        state.draft_accept_lps.clear()

        all_tokens = list(gen_batch.tokens[0])
        if exact:
            exact = (
                _qwen4_target_offset(gen_batch.prompt_cache) == len(all_tokens)
                and _qwen4_reconcile_sized_recurrent_timeline(
                    gen_batch.prompt_cache,
                    expected=len(all_tokens),
                    allowed_current={len(all_tokens)},
                )
            )
        if not exact:
            _clear_rollback(gen_batch.prompt_cache)
            return _MtpPostEmitResult(
                handled=True,
                reason="terminal-target-reconcile-failed",
            )
        return _MtpPostEmitResult(
            handled=True,
            exact_terminal=True,
            prompt_cache=gen_batch.extract_cache(0),
            all_tokens=all_tokens,
            reason="terminal-target-exact",
        )

    # Nonterminal acknowledgement. Only the last queue position is allowed to
    # mutate target state; earlier responses simply reveal another prefix of
    # the still-private verifier transaction.
    pending.emitted += 1
    state.pending_emit = None
    if pending.emitted < len(pending.token_map):
        return _MtpPostEmitResult(handled=True, reason="queue-draining")
    if state.queue:
        return _MtpPostEmitResult(handled=True, reason="queue-not-empty")

    if pending.kind == "verify-sequential":
        exact = _qwen4_commit_sequential_verify_to(
            gen_batch,
            state,
            pending,
            accepted=pending.accepted,
        )
        state.pending_commit = None
        if exact and pending.deferred_boundary:
            exact = _materialize_qwen4_deferred_boundary(
                gen_batch,
                state,
                token_id,
            )
    elif pending.kind == "verify":
        exact = _qwen4_rollback_full_verify_to(
            gen_batch,
            state,
            pending,
            accepted=pending.accepted,
        )
        state.pending_commit = None
        if exact and pending.deferred_boundary:
            exact = _materialize_qwen4_deferred_boundary(
                gen_batch,
                state,
                token_id,
            )
    else:
        # init/tail queues end in the ordinary one-token target skew expected
        # by the next verify input. Their terminal case above is the only path
        # that materializes the tail immediately, except when that final token
        # is itself a paged-cache boundary.
        state.pending_commit = None
        if pending.deferred_boundary:
            exact = _materialize_qwen4_deferred_boundary(
                gen_batch,
                state,
                token_id,
            )
        else:
            current = _qwen4_target_offset(gen_batch.prompt_cache)
            exact = current is not None and _qwen4_reconcile_sized_recurrent_timeline(
                gen_batch.prompt_cache,
                expected=current,
                allowed_current={current},
            )
    if not exact:
        raise _MtpStepFallback("Qwen4 scheduler target commit failed")
    if state.park_after_commit:
        state.park_after_commit = False
        if not _park_mtp_to_standard(gen_batch, state):
            raise _MtpStepFallback("Qwen4 deferred adaptive park failed")
    return _MtpPostEmitResult(handled=True, reason="window-committed")


def _batch_generator_mtp_post_emit(
    batch_generator: Any,
    uid: Any,
    *,
    terminal: bool,
    finish_reason: Optional[str] = None,
) -> _MtpPostEmitResult:
    """Scheduler hook: resolve parser/text stops before filtering the row."""

    gen_batch = getattr(batch_generator, "_generation_batch", None)
    if gen_batch is None:
        return _MtpPostEmitResult(reason="no-generation-batch")
    uids = list(getattr(gen_batch, "uids", ()) or ())
    if len(uids) != 1 or not uids or uids[0] != uid:
        # Target-only ExactResident is intentionally B1. Row-wise MTP keeps
        # the existing path and remains behind the resident-cache gate.
        return _MtpPostEmitResult(reason="not-b1-owner")
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if state is None or not _model_qwen4_terminal_commit_enabled(gen_batch.model):
        return _MtpPostEmitResult(reason="no-qwen4-transaction")

    try:
        _record_qwen4_probe_timeline_event(
            gen_batch,
            state,
            f"scheduler-ack-before:{'terminal' if terminal else 'continue'}",
        )
        result = _qwen4_post_emit_transaction(
            gen_batch,
            state,
            terminal=terminal,
        )
        _record_qwen4_probe_timeline_event(
            gen_batch,
            state,
            f"scheduler-ack-after:{result.reason}",
        )
    except Exception as exc:
        logger.warning("Qwen4 scheduler post-emit commit failed closed: %s", exc)
        if not terminal:
            if not _reconcile_mtp_to_standard(gen_batch, state):
                # Continuing with a verifier-ahead target would violate the
                # lossless gate. Let the engine's request-error path stop this
                # row instead of sampling from unproved state.
                raise RuntimeError(
                    "Qwen4 target commit and exact standard reconcile both failed"
                ) from exc
            _drop_mtp_state(gen_batch, "post-emit-reconciled")
        result = _MtpPostEmitResult(
            handled=True,
            reason=f"post-emit-{type(exc).__name__}",
        )

    if terminal:
        state._finish_reason = finish_reason or "terminal"
        _log_mtp_stats(uid, state.stats, state._finish_reason)
        # Exact cache/tokens were detached above before filtering. A failed
        # proof still removes the row but returns no cache, so neither L0 nor
        # durable output-cache publication can consume speculative state.
        try:
            delattr(gen_batch, "_omlx_mtp_state")
        except AttributeError:
            pass
        gen_batch.filter([])
    return result


def _mtp_next(gen_batch: Any, state: _MtpState) -> Any:
    """Emit one token; run a verify cycle if the queue is empty."""
    if state.pending_emit is not None:
        # Direct GenerationBatch consumers do not have oMLX's parser hook. A
        # subsequent next() proves the prior response was nonterminal, so ACK
        # it here. The production scheduler normally clears this immediately.
        _qwen4_post_emit_transaction(gen_batch, state, terminal=False)
    if state.queue:
        token_id, logprobs_1d, source = state.queue.popleft()
        _bump_emit_stat(state, source)
        _mark_qwen4_pending_emit(state, token_id, source)
        return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)

    _run_verify_cycle(gen_batch, state)
    if not state.queue:
        # Verify cycle should always populate the queue with at least the
        # rejected-verify token; if it didn't, fall back to the standard
        # step rather than yield an undefined response.
        raise _MtpStepFallback("verify cycle produced no emit tokens")

    token_id, logprobs_1d, source = state.queue.popleft()
    _bump_emit_stat(state, source)
    _mark_qwen4_pending_emit(state, token_id, source)
    should_exit = False
    if state.chain and state.controller is not None:
        should_exit = state.controller.should_exit()
        if not getattr(state.controller, "rank_lockstep", False):
            should_exit = _mtp_sync_flag(gen_batch, should_exit)
    if should_exit:
        if not state.queue and state.pending_commit is None:
            # Emit this cycle's token either way; on a successful handoff the
            # next next() call runs the standard step with _next_tokens set.
            _park_mtp_to_standard(gen_batch, state)
        elif state.pending_commit is not None:
            # The verifier is still private while its queue drains. Preserve
            # the controller decision and park at the scheduler-ACK seam.
            state.park_after_commit = True
    return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)


def _log_mtp_stats(uid: Any, stats: "_MtpStats", finish_reason: str) -> None:
    """Emit a one-line summary of MTP draft/verify activity for a finished sequence.

    Format chosen to match PR 990's headline metrics, plus component timings
    that make wall-clock vs. accept-rate gaps debuggable:
      MTP[<uid>] finish=<reason> tokens=<N> cycles=<C>
        accept=<A>/<conditional> (<rate>%) physical=<A>/<built> (<rate>%)
        emits[init=<i>,draft=<d>,bonus=<b>,verify=<v>]
        timing[backbone=<X>ms mtp=<Y>ms sample=<S>ms cache=<C>ms]
    """
    total_emits = (
        stats.init_emits + stats.draft_emits + stats.bonus_emits + stats.verify_emits
    )
    total_drafted = sum(stats.depth_drafted) or stats.cycles
    if total_drafted > 0:
        rate_str = f"{stats.accepts / total_drafted * 100:.1f}%"
    else:
        rate_str = "n/a"
    physical_drafted = stats.physical_drafts or total_drafted
    physical_rate_str = (
        f"{stats.accepts / physical_drafted * 100:.1f}%"
        if physical_drafted
        else "n/a"
    )
    if stats.depth_drafted:
        depth_str = " depth[" + ",".join(
            f"d{i + 1}={a}/{d}"
            for i, (a, d) in enumerate(
                zip(stats.depth_accepted, stats.depth_drafted)
            )
        ) + "]"
    else:
        depth_str = ""
    if stats.zero_cycles:
        depth_str += f" d0={stats.zero_cycles}"
    tpc = total_emits / stats.cycles if stats.cycles else 0.0
    _record_mtp_runtime_stats(stats, finish_reason)
    logger.info(
        "MTP[%s] finish=%s tokens=%d cycles=%d tok/cycle=%.2f "
        "accept=%d/%d (%s) physical=%d/%d (%s)%s "
        "emits[init=%d,draft=%d,bonus=%d,verify=%d] "
        "timing[backbone=%.1fms mtp=%.1fms sample=%.1fms cache=%.1fms]",
        uid,
        finish_reason,
        total_emits,
        stats.cycles,
        tpc,
        stats.accepts,
        total_drafted,
        rate_str,
        stats.accepts,
        physical_drafted,
        physical_rate_str,
        depth_str,
        stats.init_emits,
        stats.draft_emits,
        stats.bonus_emits,
        stats.verify_emits,
        stats.backbone_ms,
        stats.mtp_head_ms,
        stats.sample_ms,
        stats.cache_ops_ms,
    )


def _bump_emit_stat(state: _MtpState, source: str) -> None:
    if source == "init":
        state.stats.init_emits += 1
    elif source == "draft":
        state.stats.draft_emits += 1
    elif source == "bonus":
        state.stats.bonus_emits += 1
    elif source == "verify":
        state.stats.verify_emits += 1


# ---------------------------------------------------------------------------
# Verify cycle: 2-token forward + accept/reject + MTP forward for next draft.
# ---------------------------------------------------------------------------


def _run_verify_cycle(gen_batch: Any, state: _MtpState) -> None:
    """Dispatch to the depth-k chain cycle or the PR-990 depth-1 legacy cycle."""
    # A row-wise B2 batch promotes its surviving state directly back to
    # ``_omlx_mtp_state`` when the other row finishes.  The adapter's previous
    # two-row mRoPE vector is process-local model state, however, and is not
    # owned by the compact cache extracted during that B2 -> B1 transition.
    # If it remains length two, VLMModelAdapter deliberately declines to apply
    # it to the singleton.  The next Qwen4 verify window then falls through to
    # get_rope_index() and appends local positions ``0..k`` in the middle of an
    # otherwise absolute QSA timeline.  Rebind the surviving UID immediately
    # before every target verify; the helper is a no-op for non-mRoPE and
    # non-singleton routes.
    _set_singleton_mrope_delta(gen_batch)
    if state.chain:
        return _run_verify_cycle_chain(gen_batch, state)
    return _run_verify_cycle_legacy(gen_batch, state)


def _qwen4_sequential_cycle_eligible(
    gen_batch: Any,
    *,
    k: int,
    is_greedy: bool,
    two_phase_qwen4: bool,
) -> bool:
    """Admit only the initial lossless-oracle contract."""

    sampler = _resolve_sampler(gen_batch)
    try:
        explicit_greedy = bool(
            sampler is not None
            and (
                getattr(sampler, "_omlx_greedy", False) is True
                or (
                    hasattr(sampler, "temp")
                    and float(getattr(sampler, "temp")) == 0.0
                )
            )
        )
    except (TypeError, ValueError):
        explicit_greedy = False
    compact_qsa = all(
        type(cache).__name__ != "BatchQSAKVCache"
        for cache in _iter_mtp_cache_leaves(
            getattr(gen_batch, "prompt_cache", None) or []
        )
    )
    try:
        import mlx.core as mx

        single_device = int(mx.distributed.init().size()) == 1
    except Exception:
        single_device = False
    return bool(
        _qwen4_sequential_verify_enabled()
        and two_phase_qwen4
        and is_greedy
        and explicit_greedy
        and single_device
        and compact_qsa
        and 1 <= k <= 8
        and _is_qwen4_exp_model(getattr(gen_batch, "model", None))
        and _mtp_vocab_coordinator(gen_batch) is None
    )


def _run_qwen4_sequential_target(
    gen_batch: Any,
    state: _MtpState,
    *,
    k: int,
    target_base_offset: int,
    sampler: Any,
    procs: Any,
) -> _Qwen4SequentialVerifyResult:
    """Run canonical width-one target calls until reject or full accept."""

    import mlx.core as mx

    packed = mx.concatenate([state.next_main, state.drafts]).tolist()
    target_input_ids = tuple(int(value) for value in packed)
    draft_ids = tuple(target_input_ids[1:])
    if len(target_input_ids) != k + 1 or len(draft_ids) != k:
        raise _MtpStepFallback("Qwen4 sequential target window is malformed")

    try:
        snapshot = _capture_qwen4_sequential_base(
            gen_batch,
            base_offset=target_base_offset,
        )
    except _MtpStepFallback as exc:
        raise _Qwen4SequentialRecoveredFallback(str(exc)) from exc
    row_logprobs = []
    hidden_rows = []
    processor_snapshots = []
    target_ids = []
    accepted = -1
    emitted_id = -1
    processor_base_snapshot = _snapshot_qwen4_sequential_processors(procs)
    token_buffer = gen_batch._token_context[0] if procs is not None else None
    token_buffer_base_size = (
        int(getattr(token_buffer, "_size", 0)) if token_buffer is not None else None
    )
    ple_prefetch_scope = _start_qwen4_ple_window_prefetch(
        snapshot.language_model,
        gen_batch.prompt_cache,
        target_input_ids,
    )
    try:
        _set_singleton_mrope_delta(gen_batch)
        for row, token_id in enumerate(target_input_ids):
            token = mx.array([[token_id]], dtype=mx.uint32)
            prev_buf = None
            if procs is not None:
                prev_buf = gen_batch._token_context[0].update_and_fetch(
                    token.reshape(1)
                )
            prefetch_row_active = False
            if ple_prefetch_scope is not None:
                payload, _context_token = ple_prefetch_scope
                activate = getattr(
                    snapshot.language_model,
                    "activate_ple_prefetch_row",
                    None,
                )
                if callable(activate):
                    prefetch_row_active = bool(
                        activate(payload, row, token_id, token)
                    )
            try:
                logits, hidden, _ = _call_backbone(
                    gen_batch.model,
                    token,
                    gen_batch.prompt_cache,
                    n_confirmed=0,
                )
            finally:
                if ple_prefetch_scope is not None and prefetch_row_active:
                    finish = getattr(
                        snapshot.language_model,
                        "finish_ple_prefetch_row",
                        None,
                    )
                    if callable(finish):
                        finish(payload)
            if hidden is None or hidden.ndim < 3 or hidden.shape[1] != 1:
                raise _MtpStepFallback(
                    "Qwen4 sequential target did not return one raw hidden row"
                )
            row_logits = _mtp_prepare_logits(gen_batch, logits[:, -1, :])
            row_logits = _apply_processors(procs, prev_buf, row_logits)
            lp_2d = _mtp_logprobs(gen_batch, row_logits)
            target = _ensure_uint32(
                _mtp_sample(gen_batch, sampler, lp_2d).reshape(1)
            )
            mx.eval(
                target,
                hidden,
                *[cache.state for cache in gen_batch.prompt_cache],
            )
            _clear_rollback(gen_batch.prompt_cache)
            target_id = int(target.tolist()[0])
            target_ids.append(target_id)
            row_logprobs.append(lp_2d.squeeze(0))
            hidden_rows.append(hidden)
            processor_snapshots.append(
                _snapshot_qwen4_sequential_processors(procs)
            )

            if row == k:
                accepted = k
                emitted_id = target_id
                break
            if target_id != draft_ids[row]:
                accepted = row
                emitted_id = target_id
                break

        if accepted < 0 or emitted_id < 0:
            raise _MtpStepFallback("Qwen4 sequential target made no decision")
        expected = target_base_offset + accepted + 1
        if not _qwen4_reconcile_sized_recurrent_timeline(
            gen_batch.prompt_cache,
            expected=expected,
            allowed_current={expected},
        ) or not _set_qwen4_target_expected_offset(
            state,
            gen_batch.prompt_cache,
            expected,
        ):
            raise _MtpStepFallback(
                "Qwen4 sequential target timeline is not exact"
            )
        if not _qwen4_sequential_prefix_is_exact(
            gen_batch,
            snapshot,
            accepted=accepted,
        ):
            raise _MtpStepFallback(
                "Qwen4 sequential target cache proof failed"
            )
    except Exception as exc:
        processors_restored = _restore_qwen4_sequential_processors(
            procs,
            processor_base_snapshot,
        )
        if token_buffer is not None and token_buffer_base_size is not None:
            token_buffer._size = token_buffer_base_size
        restored = _restore_qwen4_sequential_partial_forward(
            gen_batch,
            state,
            snapshot,
            max_width=k + 1,
        )
        if not processors_restored or not restored:
            raise RuntimeError(
                "Qwen4 sequential target failed with an unprovable live cache"
            ) from exc
        raise _Qwen4SequentialRecoveredFallback(
            f"Qwen4 sequential target failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if ple_prefetch_scope is not None:
            _payload, context_token = ple_prefetch_scope
            end = getattr(
                snapshot.language_model,
                "end_ple_window_prefetch",
                None,
            )
            if callable(end):
                try:
                    end(context_token)
                except Exception as exc:
                    logger.warning(
                        "Qwen4 PLE window prefetch context cleanup failed: %s",
                        exc,
                    )

    return _Qwen4SequentialVerifyResult(
        snapshot=snapshot,
        target_input_ids=target_input_ids,
        draft_ids=draft_ids,
        accepted=accepted,
        emitted_id=emitted_id,
        emitted_logprobs=row_logprobs[accepted],
        combined_logprobs=mx.stack(row_logprobs),
        hidden=mx.concatenate(hidden_rows, axis=1),
        processor_base_snapshot=processor_base_snapshot,
        token_buffer_base_size=token_buffer_base_size,
        processor_snapshots=tuple(processor_snapshots),
        target_ids=tuple(target_ids),
    )


def _abort_qwen4_sequential_cycle(
    gen_batch: Any,
    state: _MtpState,
    result: _Qwen4SequentialVerifyResult,
    *,
    k: int,
    procs: Any,
) -> None:
    """Restore the cycle base before handing a post-target failure outward."""

    processors_restored = _restore_qwen4_sequential_processors(
        procs,
        result.processor_base_snapshot,
    )
    if procs is not None and result.token_buffer_base_size is not None:
        gen_batch._token_context[0]._size = result.token_buffer_base_size
    state.queue.clear()
    state.pending_commit = None
    state.pending_emit = None
    cache_restored = _restore_qwen4_sequential_partial_forward(
        gen_batch,
        state,
        result.snapshot,
        max_width=k + 1,
    )
    if not processors_restored or not cache_restored:
        raise RuntimeError(
            "Qwen4 sequential post-target failure left an unprovable cache"
        )


def _run_qwen4_sequential_verify_cycle(
    gen_batch: Any,
    state: _MtpState,
    *,
    k: int,
    target_base_offset: int,
    head_base_offset: int,
    cycle_t0: float,
    sampler: Any,
    procs: Any,
) -> None:
    """Complete one Qwen4 cycle using the canonical scalar target oracle."""

    import time

    import mlx.core as mx

    t0 = time.perf_counter()
    result = _run_qwen4_sequential_target(
        gen_batch,
        state,
        k=k,
        target_base_offset=target_base_offset,
        sampler=sampler,
        procs=procs,
    )
    state.stats.backbone_ms += (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()

    raw_accepted = result.accepted
    m = raw_accepted
    draft_ids = list(result.draft_ids)
    emit_last_id = result.emitted_id
    emit_last_lp = result.emitted_logprobs
    token_buffer = gen_batch._token_context[0] if procs is not None else None
    try:
        clamp = getattr(gen_batch.model, "mtp_clamp_accept", None)
        if m < k and callable(clamp):
            clamped = int(clamp(gen_batch.prompt_cache, m, k))
            if not 0 <= clamped <= m:
                raise _Qwen4SequentialHardFailure(
                    "Qwen4 sequential clamp returned an invalid prefix"
                )
            if clamped < m:
                m = clamped
                emit_last_id = draft_ids[m]
                emit_last_lp = result.combined_logprobs[m]

        align = int(getattr(gen_batch.model, "_omlx_mtp_commit_align", 0) or 0)
        emitted = len(gen_batch.tokens[0])
        to_boundary = (
            ((emitted // align) + 1) * align - emitted if align > 0 else 0
        )
        if 0 < to_boundary < m:
            m = to_boundary
            if callable(clamp):
                clamped = int(clamp(gen_batch.prompt_cache, m, k))
                if not 0 <= clamped <= m:
                    raise _Qwen4SequentialHardFailure(
                        "Qwen4 sequential boundary clamp is invalid"
                    )
                if clamped < m:
                    m = clamped
            emit_last_id = draft_ids[m]
            emit_last_lp = result.combined_logprobs[m]

        hidden = result.hidden
        if m < raw_accepted:
            hidden = _select_qwen4_sequential_prefix(
                gen_batch,
                state,
                result.snapshot,
                current_accepted=raw_accepted,
                target_input_ids=result.target_input_ids,
                accepted=m,
            )
            if procs is not None:
                if not _restore_qwen4_sequential_processors(
                    procs,
                    result.processor_snapshots[m],
                ):
                    raise _MtpStepFallback(
                        "Qwen4 sequential clamped processor restore failed"
                    )
                _trim_token_buffer(gen_batch, raw_accepted - m)

        materialize_boundary_emit = (
            align > 0 and to_boundary > 0 and to_boundary == m + 1
        )
    except Exception as exc:
        processors_restored = _restore_qwen4_sequential_processors(
            procs,
            result.processor_base_snapshot,
        )
        if token_buffer is not None and result.token_buffer_base_size is not None:
            token_buffer._size = result.token_buffer_base_size
        restored = _restore_qwen4_sequential_partial_forward(
            gen_batch,
            state,
            result.snapshot,
            max_width=k + 1,
        )
        if not processors_restored or not restored:
            raise RuntimeError(
                "Qwen4 sequential prefix selection failed with an unprovable cache"
            ) from exc
        if isinstance(exc, _Qwen4SequentialHardFailure):
            raise
        raise _Qwen4SequentialRecoveredFallback(
            f"Qwen4 sequential prefix selection failed: {exc}"
        ) from exc

    if not _qwen4_sequential_prefix_is_exact(
        gen_batch,
        result.snapshot,
        accepted=m,
    ):
        _abort_qwen4_sequential_cycle(
            gen_batch,
            state,
            result,
            k=k,
            procs=procs,
        )
        raise _Qwen4SequentialRecoveredFallback(
            "Qwen4 sequential accepted-prefix proof failed before staging"
        )

    state.stats.cycles += 1
    state.stats.physical_drafts += k
    if len(state.stats.depth_drafted) < state.depth:
        pad = state.depth - len(state.stats.depth_drafted)
        state.stats.depth_drafted.extend([0] * pad)
        state.stats.depth_accepted.extend([0] * pad)
    for row in range(k):
        state.stats.depth_drafted[row] += 1
        if row < m:
            state.stats.depth_accepted[row] += 1
        else:
            break
    state.stats.accepts += m
    if m < k:
        state.stats.rejects += 1
    state.stats.sample_ms += (time.perf_counter() - t0) * 1000

    try:
        t0 = time.perf_counter()
        for row in range(m):
            state.queue.append(
                (
                    draft_ids[row],
                    result.combined_logprobs[row],
                    "draft",
                )
            )
        final_source = "bonus" if m == k else "verify"
        state.queue.append((emit_last_id, emit_last_lp, final_source))

        sources = tuple(["draft"] * m + [final_source])
        tokens = tuple(draft_ids[:m] + [emit_last_id])
        state.pending_commit = _MtpPendingCommit(
            kind="verify-sequential",
            target_base_offset=target_base_offset,
            head_base_offset=head_base_offset,
            verify_width=k + 1,
            accepted=m,
            source_map=sources,
            token_map=tokens,
            deferred_boundary=materialize_boundary_emit,
            final_source=final_source,
            sequential_base=result.snapshot,
            target_input_ids=result.target_input_ids,
        )
        state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if not state.head_clone:
            _trim_committed_mtp_head(state)
        committed = mx.array(
            draft_ids[:m] + [emit_last_id],
            dtype=mx.uint32,
        )
        next_main = committed[-1:]
        prev_buf = None
        if procs is not None:
            prev_buf = gen_batch._token_context[0].tokens
        _chain_next_drafts(
            gen_batch,
            state,
            hidden[:, : m + 1],
            committed,
            prev_buf,
        )
        state.next_main = next_main
        state.pending_commit.head_committed_offset = int(state.hist_offset)
        state.stats.mtp_head_ms += (time.perf_counter() - t0) * 1000

        if state.controller is not None:
            was_warmup = bool(state.controller._warmup)
            keepalive = bool(getattr(state.mtp_cache, "fold_keepalive", False))
            if keepalive:
                state.mtp_cache.fold_keepalive = False
            state.controller.observe(
                k,
                m,
                (time.perf_counter() - cycle_t0) * 1000,
                time_sample=not keepalive,
            )
            _maybe_finish_mtp_reentry_probe(
                gen_batch,
                state,
                was_warmup=was_warmup,
            )
    except Exception as exc:
        _abort_qwen4_sequential_cycle(
            gen_batch,
            state,
            result,
            k=k,
            procs=procs,
        )
        raise _MtpStepFallback(
            f"Qwen4 sequential post-target staging failed: {exc}"
        ) from exc


def _run_verify_cycle_chain(gen_batch: Any, state: _MtpState) -> None:
    """One depth-k verify cycle.

    Verify ``[next_main, d1..dk]`` in a single backbone forward with
    ``n_confirmed=1``. Greedy acceptance is computed in-graph, so the whole
    cycle costs exactly ONE host sync (a ~2k-int ``tolist``); the next draft
    chain is dispatched with ``mx.async_eval`` and resolves inside the next
    cycle's sync. Emits ``m + 1`` tokens per cycle (m = accepted drafts, plus
    bonus on full accept or the verify-position correction on reject).
    """
    import time

    import mlx.core as mx

    if state.next_main is None or state.drafts is None:
        raise _MtpStepFallback("chain cycle entered without next_main / drafts")

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    is_greedy = _is_greedy(gen_batch)
    # Adaptive depth: the chain may have drafted fewer than state.depth
    # tokens this cycle — the verify window follows the actual drafts.
    k = int(state.drafts.shape[0])
    cycle_t0 = time.perf_counter()
    two_phase_qwen4 = bool(
        len(getattr(gen_batch, "uids", ()) or ()) == 1
        and _model_qwen4_terminal_commit_enabled(gen_batch.model)
        and not getattr(gen_batch, "_omlx_rowwise_mtp", False)
    )
    target_base_offset: Optional[int] = None
    head_base_offset = int(state.hist_offset)
    if two_phase_qwen4:
        if state.pending_commit is not None or state.pending_emit is not None:
            raise _MtpStepFallback(
                "Qwen4 verifier started before the prior scheduler commit"
            )
        target_base_offset = _qwen4_target_offset(gen_batch.prompt_cache)
        if target_base_offset is None:
            raise _MtpStepFallback(
                "Qwen4 verifier target cache has no uniform base offset"
            )

    if _qwen4_sequential_cycle_eligible(
        gen_batch,
        k=k,
        is_greedy=is_greedy,
        two_phase_qwen4=two_phase_qwen4,
    ):
        assert target_base_offset is not None
        try:
            return _run_qwen4_sequential_verify_cycle(
                gen_batch,
                state,
                k=k,
                target_base_offset=target_base_offset,
                head_base_offset=head_base_offset,
                cycle_t0=cycle_t0,
                sampler=sampler,
                procs=procs,
            )
        except _Qwen4SequentialRecoveredFallback as exc:
            # The scalar attempt restores target/model/processor state before
            # raising this recoverable signal. Continue with the established
            # wide verifier in the same cycle; an unprovable restore raises a
            # hard RuntimeError instead and must never reuse the live cache.
            logger.debug(
                "Qwen4 sequential oracle fell back to wide verification: %s",
                exc,
            )

    inputs = mx.concatenate([state.next_main, state.drafts])  # (k+1,)
    if _maybe_probe_qwen4_verify_parity(gen_batch, state, inputs):
        # Diagnostic replay is not part of target-cycle economics.  Excluding
        # it prevents an explicitly requested trace from poisoning the
        # adaptive depth controller's cost evidence.
        cycle_t0 = time.perf_counter()

    # Token buffer per input position (mirrors PR 990 _step_backbone). Row j's
    # processor prefix is everything before that input position.
    prev_rows: List[Optional[Any]] = [None] * (k + 1)
    if procs is not None:
        buf = gen_batch._token_context[0]
        prev_rows[0] = buf.update_and_fetch(state.next_main)
        for j in range(k):
            prev_rows[j + 1] = buf.update_and_fetch(state.drafts[j : j + 1])

    # --- backbone verify forward + single host sync ---
    t0 = time.perf_counter()
    logits, hidden, gdn_states = _call_backbone(
        gen_batch.model,
        inputs[None, :],
        gen_batch.prompt_cache,
        n_confirmed=1,
    )
    _materialize_distributed_hidden_sibling(logits, hidden, mx_module=mx)
    ple_snapshots: Tuple[Tuple[Any, Any], ...] = ()
    qsa_snapshots: Tuple[_Qwen4QSARollbackSnapshot, ...] = ()
    if two_phase_qwen4 and k > 0:
        assert target_base_offset is not None
        ple_snapshots, qsa_snapshots = _capture_qwen4_verify_snapshots(
            gen_batch.prompt_cache,
            base_offset=target_base_offset,
            verify_width=k + 1,
        )
    rows = _mtp_prepare_logits(gen_batch, logits[0])  # (k+1, vocab)
    _capture_qwen4_active_verify_parity(gen_batch, state, rows)
    row_snaps: List[Optional[Any]] = [None] * (k + 1)
    if procs is not None:
        applied = []
        for j in range(k + 1):
            applied.append(
                _apply_processors(procs, prev_rows[j], rows[j : j + 1]).squeeze(0)
            )
            # Checkpoint after each row: rows 0..m correspond to the m+1
            # tokens actually emitted this cycle (m accepted drafts + the
            # bonus/verify correction). Rows m+1..k are speculative — they
            # predict rejected drafts and are re-verified next cycle.
            row_snaps[j] = _snap_snapshotable(procs)
        rows = mx.stack(applied)
    combined_lp = _mtp_logprobs(gen_batch, rows)  # (k+1, V)

    adapter = _mtp_vocab_coordinator(gen_batch)
    greedy_target_ids: Optional[List[int]] = None

    if k == 0:
        # Depth-0 cycle (controller escape hatch): the forward above was a
        # plain 1-token step at [next_main]; sample its next token and emit
        # it as the bonus. No drafts to accept, nothing to roll back —
        # per-cycle cost is the baseline step the controller tracks as t[0].
        step_tok = _ensure_uint32(
            _mtp_sample(gen_batch, sampler, combined_lp[:1]).reshape(1)
        )
        m = 0
        draft_ids: List[int] = []
        emit_last_id = int(step_tok.tolist()[0])
        emit_last_lp = combined_lp[0]
        state.stats.backbone_ms += (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        state.stats.zero_cycles += 1
    elif adapter is not None:
        # Only rank zero owns full vocabulary rows. It computes the exact
        # speculative decision and contributes a fixed packet
        # ``[accepted, final-token, draft-ids...]``; workers contribute zeros.
        # One token collective replaces full-vocabulary all-gathers to every
        # rank and also pins cache rollback to the same accepted prefix.
        if adapter.is_coordinator:
            if is_greedy:
                targets = mx.argmax(rows, axis=-1).astype(mx.int32)
                matches = (
                    targets[:k] == state.drafts.astype(mx.int32)
                ).astype(mx.int32)
                m_arr = mx.cumprod(matches).sum().reshape(1).astype(mx.int32)
                final_id = mx.take(targets, m_arr).reshape(1).astype(mx.int32)
            else:
                accept_rows = _accept_lp_for(sampler, combined_lp)
                q_rows = mx.stack(state.draft_accept_lps)
                idx = state.drafts.astype(mx.int32)[:, None]
                p_at = mx.take_along_axis(
                    accept_rows[:k], idx, axis=-1
                ).squeeze(-1)
                q_at = mx.take_along_axis(q_rows, idx, axis=-1).squeeze(-1)
                ratio = p_at - q_at
                u = mx.random.uniform(shape=(k,))
                accepted = mx.logical_or(ratio >= 0, mx.log(u) < ratio)
                m_arr = (
                    mx.cumprod(accepted.astype(mx.int32))
                    .sum()
                    .reshape(1)
                    .astype(mx.int32)
                )
                p_all = mx.exp(accept_rows[:k])
                residual = mx.maximum(p_all - mx.exp(q_rows), 0.0)
                mass = residual.sum(axis=-1, keepdims=True)
                residual = mx.where(mass > 0, residual, p_all)
                residual_samples = mx.random.categorical(mx.log(residual))
                bonus = sampler(combined_lp[k : k + 1]).reshape(1)
                candidates = mx.concatenate(
                    [residual_samples.astype(mx.int32), bonus.astype(mx.int32)]
                )
                final_id = mx.take(candidates, m_arr).reshape(1)
            packet = mx.concatenate(
                [
                    m_arr,
                    final_id.astype(mx.int32),
                    state.drafts.astype(mx.int32),
                ]
            )
        else:
            packet = None
        host = _mtp_sync_packet(gen_batch, packet, k + 2)
        m = int(host[0])
        emit_last_id = int(host[1])
        draft_ids = host[2:]
        state.stats.backbone_ms += (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        emit_last_lp = combined_lp[m if m < k else k]
    elif is_greedy:
        targets = mx.argmax(rows, axis=-1).astype(mx.int32)  # (k+1,)
        matches = (targets[:k] == state.drafts.astype(mx.int32)).astype(mx.int32)
        m_arr = mx.cumprod(matches).sum().reshape(1)
        host = mx.concatenate(
            [m_arr, targets, state.drafts.astype(mx.int32)]
        ).tolist()
        m = int(host[0])
        target_ids = host[1 : k + 2]
        greedy_target_ids = target_ids
        draft_ids = host[k + 2 :]
        state.stats.backbone_ms += (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        emit_last_id = target_ids[m] if m < k else target_ids[k]
        emit_last_lp = combined_lp[m if m < k else k]
    else:
        # Stochastic: batched Leviathan/Chen acceptance computed in-graph —
        # per-position ratios of the filtered target rows (p) against the
        # draft sampler's filtered rows (q), cumulative accept, residual
        # samples for every position, and the bonus draw, all resolved in
        # ONE host sync (mirrors the greedy path's sync structure).
        accept_rows = _accept_lp_for(sampler, combined_lp)  # (k+1, V)
        q_rows = mx.stack(state.draft_accept_lps)  # (k, V)
        idx = state.drafts.astype(mx.int32)[:, None]
        p_at = mx.take_along_axis(accept_rows[:k], idx, axis=-1).squeeze(-1)
        q_at = mx.take_along_axis(q_rows, idx, axis=-1).squeeze(-1)
        ratio = p_at - q_at  # (k,) log acceptance ratios
        u = mx.random.uniform(shape=(k,))
        acc = mx.logical_or(ratio >= 0, mx.log(u) < ratio)
        m_arr = mx.cumprod(acc.astype(mx.int32)).sum().reshape(1)
        # Residual distributions max(p - q, 0) per draft position. Only the
        # reject position's sample is used; computing all k keeps the cycle
        # single-sync and costs a few elementwise vocab ops on GPU.
        p_all = mx.exp(accept_rows[:k])
        res = mx.maximum(p_all - mx.exp(q_rows), 0.0)
        z = res.sum(axis=-1, keepdims=True)
        res_dist = mx.where(z > 0, res, p_all)
        res_samples = mx.random.categorical(mx.log(res_dist))  # (k,)
        bonus_tok = sampler(combined_lp[k : k + 1]).reshape(1)
        host = mx.concatenate(
            [
                m_arr.astype(mx.int32),
                state.drafts.astype(mx.int32),
                res_samples.astype(mx.int32),
                bonus_tok.astype(mx.int32),
            ]
        ).tolist()
        m = int(host[0])
        draft_ids = host[1 : k + 1]
        res_ids = host[k + 1 : 2 * k + 1]
        bonus_id = host[2 * k + 1]
        state.stats.backbone_ms += (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        if m < k:
            emit_last_id = res_ids[m]
            emit_last_lp = combined_lp[m]
        else:
            emit_last_id = bonus_id
            emit_last_lp = combined_lp[k]

    # Clamp the accepted count to what every cache layer can roll back
    # (optional model hook — DeepSeek-V4 PoolingCache replay windows are
    # bounded). Emitting fewer verified drafts is always correct; position
    # ``m`` was itself accepted when the clamp lowers it, so its draft
    # token is a fair emit for the correction slot.
    clamp = getattr(gen_batch.model, "mtp_clamp_accept", None)
    if m < k:
        if callable(clamp):
            clamped = int(clamp(gen_batch.prompt_cache, m, k))
            if clamped < m:
                m = clamped
                emit_last_id = draft_ids[m]
                emit_last_lp = combined_lp[m]

    # Apply boundary alignment after the rollback clamp so its final accepted
    # count remains authoritative. If alignment itself requires more rollback,
    # re-run the model clamp against the shorter accepted prefix.
    align = int(getattr(gen_batch.model, "_omlx_mtp_commit_align", 0) or 0)
    emitted = len(gen_batch.tokens[0])
    to_boundary = ((emitted // align) + 1) * align - emitted if align > 0 else 0
    if 0 < to_boundary < m:
        m = to_boundary
        if callable(clamp):
            clamped = int(clamp(gen_batch.prompt_cache, m, k))
            if clamped < m:
                m = clamped
        emit_last_id = draft_ids[m]
        emit_last_lp = combined_lp[m]

    # Rewind budget-capable processors to the last emitted position.
    # Rows 0..m produced the m+1 emitted tokens (m accepted drafts + the
    # bonus/verify correction); rows m+1..k predicted rejected drafts that
    # are re-verified next cycle, so their processor calls must be undone
    # (they would over-count the thinking budget / corrupt state). Mirrors
    # MTPProcessingSampler's position-keyed snapshot/restore on vlm_mtp.
    # Uses the FINAL m (after the model clamp and boundary alignment above).
    if m < k and row_snaps[m] is not None:
        _restore_snapshotable(procs, row_snaps[m])

    # A clamp can put the final verify token on the boundary, while a full
    # accept can put its bonus token there. Neither token is present in the
    # backbone cache yet, so materialize it before the queue reaches it.
    materialize_boundary_emit = align > 0 and to_boundary > 0 and to_boundary == m + 1

    # --- stats ---
    state.stats.cycles += 1
    state.stats.physical_drafts += k
    if len(state.stats.depth_drafted) < state.depth:
        pad = state.depth - len(state.stats.depth_drafted)
        state.stats.depth_drafted.extend([0] * pad)
        state.stats.depth_accepted.extend([0] * pad)
    for j in range(k):
        state.stats.depth_drafted[j] += 1
        if j < m:
            state.stats.depth_accepted[j] += 1
        else:
            break
    state.stats.accepts += m
    if m < k:
        state.stats.rejects += 1
    state.stats.sample_ms += (time.perf_counter() - t0) * 1000

    # --- stage queue; Qwen4 defers the target commit to scheduler post-emit ---
    t0 = time.perf_counter()
    for j in range(m):
        state.queue.append((int(draft_ids[j]), state.draft_lps[j], "draft"))
    if m == k:
        state.queue.append((int(emit_last_id), emit_last_lp, "bonus"))
    else:
        state.queue.append((int(emit_last_id), emit_last_lp, "verify"))
        if procs is not None:
            _trim_token_buffer(gen_batch, k - m)

    if two_phase_qwen4:
        assert target_base_offset is not None
        if k == 0:
            # The width-one depth-0 forward committed ``next_main`` and
            # sampled exactly one pipeline-tail token. No speculative target
            # state exists, but terminal emission of that tail still needs an
            # L=1 target-only commit.
            expected = target_base_offset + 1
            if not _qwen4_reconcile_sized_recurrent_timeline(
                gen_batch.prompt_cache,
                expected=expected,
                allowed_current={expected},
            ) or not _set_qwen4_target_expected_offset(
                state,
                gen_batch.prompt_cache,
                expected,
            ):
                raise _MtpStepFallback(
                    "Qwen4 depth-zero target commit offset mismatch"
                )
            state.pending_commit = _MtpPendingCommit(
                kind="tail",
                target_base_offset=expected,
                head_base_offset=head_base_offset,
                verify_width=0,
                accepted=0,
                source_map=("bonus",),
                token_map=(int(emit_last_id),),
                deferred_boundary=materialize_boundary_emit,
                final_source="bonus",
            )
        else:
            sources = tuple(["draft"] * m + ["bonus" if m == k else "verify"])
            tokens = tuple(
                [int(draft_ids[j]) for j in range(m)] + [int(emit_last_id)]
            )
            state.pending_commit = _MtpPendingCommit(
                kind="verify",
                target_base_offset=target_base_offset,
                head_base_offset=head_base_offset,
                verify_width=k + 1,
                accepted=m,
                source_map=sources,
                token_map=tokens,
                gdn_states=gdn_states,
                ple_snapshots=ple_snapshots,
                qsa_snapshots=qsa_snapshots,
                deferred_boundary=materialize_boundary_emit,
                final_source=sources[-1],
            )
    else:
        if m == k:
            _clear_rollback(gen_batch.prompt_cache)
        elif not _chain_rollback(
            gen_batch.model, gen_batch.prompt_cache, m, k, gdn_states
        ):
            raise _MtpStepFallback("cache layer rejects chain rollback")
        # Target verification committed exactly m accepted drafts plus the
        # confirmed input row. Keep its absolute timeline independent from
        # the suffix-local draft-head offset.
        _advance_suffix_local_target(state, gen_batch.prompt_cache, m + 1)
    state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

    # --- MTP-head history + next draft chain (async-dispatched) ---
    t0 = time.perf_counter()
    if not state.head_clone:
        _trim_committed_mtp_head(state)
    committed = mx.array(
        [int(d) for d in draft_ids[:m]] + [int(emit_last_id)], dtype=mx.uint32
    )
    next_main = committed[-1:]
    hidden_rows = hidden[:, : m + 1]
    prev_buf = None
    if procs is not None:
        prev_buf = gen_batch._token_context[0].tokens
    _chain_next_drafts(gen_batch, state, hidden_rows, committed, prev_buf)
    state.next_main = next_main
    if two_phase_qwen4 and state.pending_commit is not None:
        state.pending_commit.head_committed_offset = int(state.hist_offset)
    state.stats.mtp_head_ms += (time.perf_counter() - t0) * 1000
    if materialize_boundary_emit and not two_phase_qwen4:
        _materialize_mtp_boundary_emit(gen_batch, state)
    if state.controller is not None:
        was_warmup = bool(state.controller._warmup)
        keepalive = bool(getattr(state.mtp_cache, "fold_keepalive", False))
        if keepalive:
            state.mtp_cache.fold_keepalive = False
        state.controller.observe(
            k,
            m,
            (time.perf_counter() - cycle_t0) * 1000,
            time_sample=not keepalive,
        )
        _maybe_finish_mtp_reentry_probe(
            gen_batch,
            state,
            was_warmup=was_warmup,
        )
    _finish_qwen4_active_verify_parity(
        gen_batch,
        state,
        target_ids=greedy_target_ids,
        draft_ids=draft_ids,
        accepted=m,
        emitted_id=emit_last_id,
    )


def _materialize_mtp_boundary_emit(gen_batch: Any, state: _MtpState) -> None:
    """Commit a queued verify/bonus boundary token before it is emitted.

    MTP normally leaves the final token of a cycle one position ahead of the
    backbone cache. If that token is the next block boundary, process it with a
    confirmed one-token forward and seed the following target token. The queue
    then reaches the boundary with an exactly aligned cache snapshot and keeps
    the usual one-token pipeline skew after the following token is emitted.
    """
    import time

    import mlx.core as mx

    if not state.queue:
        raise _MtpStepFallback("boundary materialization has no queued token")

    boundary_id = int(state.queue[-1][0])
    boundary_tok = mx.array([boundary_id], dtype=mx.uint32)
    procs = _proc_list(gen_batch)
    prev_buf = None
    if procs is not None:
        prev_buf = gen_batch._token_context[0].update_and_fetch(boundary_tok)

    t0 = time.perf_counter()
    logits, hidden, _ = _call_backbone(
        gen_batch.model,
        boundary_tok[:, None],
        gen_batch.prompt_cache,
    )
    _clear_rollback(gen_batch.prompt_cache)
    _advance_suffix_local_target(state, gen_batch.prompt_cache, 1)
    next_logits = _mtp_prepare_logits(gen_batch, logits[:, -1, :])
    next_logits = _apply_processors(procs, prev_buf, next_logits)
    next_lp_2d = _mtp_logprobs(gen_batch, next_logits)
    next_tok = _ensure_uint32(
        _mtp_sample(gen_batch, _resolve_sampler(gen_batch), next_lp_2d)
    )
    mx.eval(next_tok)
    state.stats.backbone_ms += (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if not state.head_clone:
        # The first next-draft chain may have appended speculative local-head
        # rows past hist_offset. Boundary materialization commits another target
        # token before rebuilding that chain, so rewind to the local committed
        # seam first (head-clone families already keep the persistent cache clean).
        _trim_committed_mtp_head(state)
    _chain_next_drafts(
        gen_batch,
        state,
        hidden[:, -1:],
        next_tok,
        prev_buf,
    )
    state.stats.mtp_head_ms += (time.perf_counter() - t0) * 1000
    next_id = int(next_tok.tolist()[0])
    state.next_main = next_tok
    state.queue.append((next_id, next_lp_2d.squeeze(0), "bonus"))


def _materialize_qwen4_deferred_boundary(
    gen_batch: Any,
    state: _MtpState,
    boundary_id: int,
) -> bool:
    """Materialize an accepted boundary only after scheduler nonterminal ACK.

    Unlike ``_materialize_mtp_boundary_emit`` this is called after the
    boundary token has actually left the queue.  It creates a fresh one-token
    tail transaction for the sampled successor, preserving exact terminal
    handling if that successor itself stops.
    """

    import time

    import mlx.core as mx

    if state.queue:
        return False
    boundary_tok = mx.array([int(boundary_id)], dtype=mx.uint32)
    procs = _proc_list(gen_batch)
    prev_buf = None
    if procs is not None:
        prev_buf = gen_batch._token_context[0].update_and_fetch(boundary_tok)

    target_before = _qwen4_target_offset(gen_batch.prompt_cache)
    if target_before is None:
        return False
    t0 = time.perf_counter()
    try:
        logits, hidden, _ = _call_backbone(
            gen_batch.model,
            boundary_tok[:, None],
            gen_batch.prompt_cache,
        )
        _clear_rollback(gen_batch.prompt_cache)
        expected = target_before + 1
        if not _qwen4_reconcile_sized_recurrent_timeline(
            gen_batch.prompt_cache,
            expected=expected,
            allowed_current={expected},
        ) or not _set_qwen4_target_expected_offset(
            state,
            gen_batch.prompt_cache,
            expected,
        ):
            return False
        next_logits = _mtp_prepare_logits(gen_batch, logits[:, -1, :])
        next_logits = _apply_processors(procs, prev_buf, next_logits)
        next_lp_2d = _mtp_logprobs(gen_batch, next_logits)
        next_tok = _ensure_uint32(
            _mtp_sample(gen_batch, _resolve_sampler(gen_batch), next_lp_2d)
        )
        mx.eval(next_tok)
    except Exception as exc:
        logger.warning("Qwen4 deferred boundary materialization failed: %s", exc)
        return False
    state.stats.backbone_ms += (time.perf_counter() - t0) * 1000

    head_before = int(state.hist_offset)
    t0 = time.perf_counter()
    try:
        if not state.head_clone:
            _trim_committed_mtp_head(state)
        _chain_next_drafts(
            gen_batch,
            state,
            hidden[:, -1:],
            next_tok,
            prev_buf,
        )
    except Exception as exc:
        logger.warning("Qwen4 deferred boundary head rebuild failed: %s", exc)
        return False
    state.stats.mtp_head_ms += (time.perf_counter() - t0) * 1000
    next_id = int(next_tok.tolist()[0])
    state.next_main = next_tok
    state.queue.append((next_id, next_lp_2d.squeeze(0), "bonus"))
    state.pending_commit = _MtpPendingCommit(
        kind="tail",
        target_base_offset=target_before + 1,
        head_base_offset=head_before,
        verify_width=0,
        accepted=0,
        source_map=("bonus",),
        token_map=(next_id,),
        head_committed_offset=int(state.hist_offset),
        final_source="bonus",
    )
    return True


def _chain_rollback(
    model: Any,
    prompt_cache: List[Any],
    accepted: int,
    num_drafts: int,
    gdn_states: Optional[list] = None,
) -> bool:
    """Roll the backbone cache back to ``accepted`` drafts after a chain verify.

    mlx-vlm path (``gdn_states`` populated): delegate to the stock
    ``rollback_speculative_cache``, which natively supports partial accepts —
    it keeps ``accepted + 1`` positions of the ``num_drafts + 1``-token
    verify window and replays the accepted prefix through the captured GDN
    states. mlx-lm path: ``mtp_partial_rollback`` (qwen35_model patch).
    """
    if gdn_states is not None and hasattr(model, "rollback_speculative_cache"):
        try:
            model.rollback_speculative_cache(
                prompt_cache, gdn_states, accepted, num_drafts + 1
            )
            return True
        except Exception as exc:
            logger.debug("rollback_speculative_cache failed: %s", exc)
            return False
    rollback = getattr(model, "mtp_partial_rollback", None)
    if callable(rollback):
        try:
            return bool(rollback(prompt_cache, accepted, num_drafts))
        except Exception as exc:
            logger.debug("mtp_partial_rollback failed: %s", exc)
            return False
    if accepted == 0 and num_drafts == 1:
        return _restore_or_trim_caches(prompt_cache)
    return False


def _run_verify_cycle_legacy(gen_batch: Any, state: _MtpState) -> None:
    """Run one verify cycle. Populates ``state.queue`` with 1 (reject) or 2
    (accept) tokens for upcoming emit calls. Updates ``state.next_main`` and
    ``state.draft_tok`` / ``state.draft_lp`` for the cycle after that.
    """
    import time

    import mlx.core as mx

    if state.next_main is None or state.draft_tok is None:
        raise _MtpStepFallback("verify cycle entered without next_main / draft")

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    is_greedy = _is_greedy(gen_batch)

    inputs = mx.concatenate([state.next_main, state.draft_tok])  # (2,)

    # Update the token buffer per-position (mirrors PR 990 _step_backbone).
    prev_main = None
    prev_draft = None
    if procs is not None:
        prev_main = gen_batch._token_context[0].update_and_fetch(state.next_main)
        prev_draft = gen_batch._token_context[0].update_and_fetch(state.draft_tok)

    # --- backbone forward (materialized before sampling) ---
    # Dispatch the backbone on the generation stream, then force ``mx.eval``
    # on the logits before the sampler runs. MLX is lazy, so without this the
    # later ``mx.eval(verify_tok, bonus_tok)`` barrier would resolve the whole
    # graph in one stall and the heavy verify forward would leak into
    # sample_ms (this is what made the sampler look like the bottleneck in
    # #1097 / #1311 / #1330). The extra eval costs one CPU<->GPU round-trip
    # per cycle (negligible vs the forward compute) and keeps the
    # backbone_ms / sample_ms split accurate.
    t0 = time.perf_counter()
    logits, hidden, gdn_states = _call_backbone(
        gen_batch.model,
        inputs[None, :],
        gen_batch.prompt_cache,
        n_confirmed=1,
    )
    verify_logits = logits[:, 0, :]
    bonus_logits = logits[:, 1, :]
    if not _materialize_distributed_hidden_sibling(logits, hidden, mx_module=mx):
        mx.eval(logits)
    state.stats.backbone_ms += (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    verify_snap = None
    if procs is not None:
        verify_logits = _apply_processors(procs, prev_main, verify_logits)
        # Checkpoint after the verify row: it produced the one token that is
        # ALWAYS emitted (draft on accept, verify correction on reject).
        verify_snap = _snap_snapshotable(procs)
        bonus_logits = _apply_processors(procs, prev_draft, bonus_logits)
    # Batched logprobs: one logsumexp over (2, vocab) instead of two over
    # (1, vocab). Shaves one reduction per cycle on the vocab dimension.
    combined_logits = mx.concatenate(
        [verify_logits, bonus_logits], axis=0
    )  # (2, vocab)
    combined_lp = combined_logits - mx.logsumexp(
        combined_logits, axis=-1, keepdims=True
    )
    verify_lp_2d = combined_lp[0:1]
    bonus_lp_2d = combined_lp[1:2]
    verify_tok = sampler(verify_lp_2d)
    bonus_tok = sampler(bonus_lp_2d)
    mx.eval(verify_tok, bonus_tok)

    # ``draft_id`` was cached when the draft was sampled (post_init or the
    # prior _step_mtp); skip the GPU→CPU sync that ``state.draft_tok.tolist()``
    # would impose on every cycle.
    draft_id = state.draft_id
    verify_id = int(verify_tok.tolist()[0])
    bonus_id = int(bonus_tok.tolist()[0])
    # Filtered logprobs — distribution the sampler actually drew from.
    # Used for acceptance ratio + residual sampling so they match the
    # sampling distribution rather than raw softmax (PR 990 alignment).
    verify_accept_lp = _accept_lp_for(sampler, verify_lp_2d)
    draft_accept_lp = (
        state.draft_accept_lp
        if state.draft_accept_lp is not None
        else _accept_lp_for(sampler, state.draft_lp)
    )

    if is_greedy:
        accept = verify_id == draft_id
    else:
        log_accept = (
            verify_accept_lp[0, draft_id].item() - draft_accept_lp[draft_id].item()
        )
        # Draw the acceptance roll from mx.random so it follows the same
        # mx.random.seed the rest of the sampler uses (line ~962 residual
        # sampling). stdlib ``random`` was never seeded by oMLX, which made
        # stochastic acceptance irreproducible even with a fixed seed (#1330).
        accept = log_accept >= 0 or float(
            mx.random.uniform(shape=()).item()
        ) < math.exp(log_accept)
    state.stats.sample_ms += (time.perf_counter() - t0) * 1000

    hidden_at_confirmed = hidden[:, 0:1, :]
    hidden_at_draft = hidden[:, 1:2, :]

    state.stats.cycles += 1
    state.stats.physical_drafts += 1
    if accept:
        state.stats.accepts += 1
        # --- cache cleanup (timed) ---
        t0 = time.perf_counter()
        _clear_rollback(gen_batch.prompt_cache)
        state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

        # --- MTP head forward for next draft (timed inside _step_mtp) ---
        new_draft, new_draft_lp = _step_mtp(
            gen_batch,
            hidden_at_draft,
            _ensure_uint32(bonus_tok),
            prev_buf=prev_draft if procs is not None else None,
            stats=state.stats,
        )
        # Queue the two emitted tokens. Per PR 990: the accepted draft uses
        # the *MTP head's* original draft distribution as its logprobs; the
        # bonus uses the verify forward's bonus distribution.
        state.queue.append((draft_id, state.draft_lp, "draft"))
        state.queue.append((bonus_id, bonus_lp_2d.squeeze(0), "bonus"))
        state.next_main = _ensure_uint32(bonus_tok)
        state.draft_tok = new_draft
        state.draft_lp = new_draft_lp
        return

    # Reject path.
    state.stats.rejects += 1
    t0 = time.perf_counter()
    # The bonus row's processor call was speculative (its token is not
    # emitted on reject) — rewind to the verify-row checkpoint. Mirrors the
    # chain cycle's restore-on-partial-accept.
    if procs is not None and verify_snap is not None:
        _restore_snapshotable(procs, verify_snap)
    # accepted=0 means only the confirmed token (verify position) is kept;
    # block_size=2 covers both the confirmed and the rejected draft.
    if _model_mtp_replay_reject_enabled(gen_batch.model):
        rollback_ok = _rollback_and_replay_confirmed(
            gen_batch,
            state,
            verify_width=2,
        )
    else:
        rollback_ok = _rollback_after_reject(
            gen_batch.model,
            gen_batch.prompt_cache,
            gdn_states,
            accepted=0,
            block_size=2,
        )
    if not rollback_ok:
        if procs is not None:
            _trim_token_buffer(gen_batch, 1)
        raise _MtpStepFallback("cache layer rejects rollback")
    if procs is not None:
        _trim_token_buffer(gen_batch, 1)
    state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

    # Pick the verify-position emit token: residual sample for stochastic.
    # Residual is computed on the *filtered* distributions so the sample
    # comes from `max(p_target_filt - p_draft_filt, 0)` — matching what the
    # sampler would have produced if it had drawn directly from the verify
    # position. emit_lp returned to the caller stays as the raw verify lp
    # so downstream logprobs reporting is consistent with non-MTP paths.
    if is_greedy:
        emit_id = verify_id
        emit_lp = verify_lp_2d.squeeze(0)
    else:
        emit_id, _ = _residual_sample(verify_accept_lp, draft_accept_lp)
        emit_lp = verify_lp_2d.squeeze(0)

    emit_tok = mx.array([emit_id], dtype=mx.uint32)
    new_draft, new_draft_lp = _step_mtp(
        gen_batch,
        hidden_at_confirmed,
        emit_tok,
        prev_buf=prev_main if procs is not None else None,
        stats=state.stats,
    )

    state.queue.append((emit_id, emit_lp, "verify"))
    state.next_main = emit_tok
    state.draft_tok = new_draft
    state.draft_lp = new_draft_lp


# ---------------------------------------------------------------------------
# Helpers used by the verify cycle.
# ---------------------------------------------------------------------------


def _step_mtp(
    gen_batch: Any,
    hidden_at_position: Any,
    next_main_tok: Any,
    prev_buf: Optional[Any],
    stats: Optional["_MtpStats"] = None,
) -> Tuple[Any, Any]:
    """Run one MTP-head forward + sample. Returns ``(draft_tok, draft_lp)``.

    Side effect: caches the host-side int copy of the new draft on
    ``gen_batch._omlx_mtp_state.draft_id`` so the next verify cycle's
    accept check is sync-free.
    """
    import time

    import mlx.core as mx

    state = gen_batch._omlx_mtp_state
    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    t0 = time.perf_counter()
    next_ids = next_main_tok.reshape(1, 1)
    mtp_logits = gen_batch.model.mtp_forward(
        hidden_at_position, next_ids, state.mtp_cache
    )
    mtp_logits_2d = _mtp_prepare_logits(gen_batch, mtp_logits[:, -1, :])
    # The draft is speculative — shape it but do not advance the budget.
    snap = _snap_snapshotable(procs)
    if procs is not None and prev_buf is not None:
        prev_with_next = mx.concatenate([prev_buf, _ensure_uint32(next_main_tok)])
        mtp_logits_2d = _apply_processors(procs, prev_with_next, mtp_logits_2d)
    _restore_snapshotable(procs, snap)
    new_lp = _mtp_logprobs(gen_batch, mtp_logits_2d)
    new_tok = _mtp_sample(gen_batch, sampler, new_lp)
    # Filtered draft lp — what the sampler actually drew from. The next
    # verify cycle's acceptance ratio uses this so the math matches the
    # sampling distribution rather than raw softmax (PR 990 alignment).
    new_accept_lp = _mtp_accept_lp(gen_batch, sampler, new_lp)
    # ``.tolist()`` forces evaluation; replaces the explicit ``mx.eval`` and
    # piggybacks the host-side int caching on the same sync.
    draft_id_int = int(new_tok.tolist()[0])
    state.draft_id = draft_id_int
    state.draft_accept_lp = new_accept_lp.squeeze(0)
    if stats is not None:
        stats.mtp_head_ms += (time.perf_counter() - t0) * 1000
    return _ensure_uint32(new_tok), new_lp.squeeze(0)


def _residual_sample(verify_lp_2d: Any, draft_lp_1d: Any) -> Tuple[int, Any]:
    """Sample from ``max(p_target - p_draft, 0)`` (Leviathan et al. 2022).

    On degenerate input (residual all zero) falls back to the target
    distribution rather than the verify-position argmax — keeps the sample
    drawn from a proper distribution and stays in-graph (no host sync).
    Mirrors mlx-lm PR 990 commit 6594348.

    Returns ``(token_id_int, verify_lp_1d)``.
    """
    import mlx.core as mx

    p_target = mx.exp(verify_lp_2d.squeeze(0))
    p_draft = mx.exp(draft_lp_1d)
    residual = mx.maximum(p_target - p_draft, 0.0)
    # Keep z in graph; mx.where switches to the target distribution when
    # the residual mass is zero. ``categorical`` treats log(0) = -inf as
    # p=0 so no safety epsilon is needed.
    z = residual.sum(keepdims=True)
    dist = mx.where(z > 0, residual, p_target)
    sample = mx.random.categorical(mx.log(dist).reshape(1, -1))
    return int(sample.item()), verify_lp_2d.squeeze(0)


# ---------------------------------------------------------------------------
# Response builder — mirrors GenerationBatch.next()'s per-sequence epilogue.
# ---------------------------------------------------------------------------


def _emit_response(
    gen_batch: Any,
    token_id: int,
    logprobs_1d: Any,
    stats: Optional["_MtpStats"] = None,
) -> List[Any]:
    """Produce a single-element response list, applying the standard
    epilogue (token append + max_tokens / matcher checks) so external
    callers (BatchGenerator, scheduler, response stream) see the same
    contract as the unmodified next().
    """
    Response = type(gen_batch).Response
    token_id = _validated_emitted_token(token_id, logprobs_1d)

    finish_reason: Optional[str] = None
    match_sequence = None

    gen_batch.tokens[0].append(token_id)
    gen_batch._num_tokens[0] += 1
    if gen_batch._num_tokens[0] >= gen_batch.max_tokens[0]:
        finish_reason = "length"

    new_state, match_sequence, current_state = gen_batch.state_machines[0].match(
        gen_batch._matcher_states[0], token_id
    )
    gen_batch._matcher_states[0] = new_state
    if match_sequence is not None and current_state is None:
        finish_reason = "stop"

    if finish_reason is not None:
        # Qwen4's two-phase target transaction must remain attached until the
        # scheduler has also evaluated protocol-parser and text-stop rules.
        # The post-emit hook reconciles the exact terminal prefix, detaches it,
        # and filters the row. Other MTP families keep the historical eager
        # finish behavior below.
        defer_terminal = bool(
            getattr(gen_batch, "_omlx_mtp_state", None)
        ) and _model_qwen4_terminal_commit_enabled(gen_batch.model)
        if defer_terminal:
            return [
                Response(
                    uid=gen_batch.uids[0],
                    token=token_id,
                    logprobs=logprobs_1d,
                    finish_reason=finish_reason,
                    current_state=current_state,
                    match_sequence=match_sequence,
                    prompt_cache=None,
                    all_tokens=None,
                )
            ]

        # Legacy Qwen3.5-style MTP does not have Qwen4's explicit two-phase
        # target transaction. Its verifier may leave the backbone cache ahead
        # of the visible terminal token while a queued draft is still parked.
        # Never replay the full committed ledger merely to manufacture a cache
        # candidate after the response is already complete: at long context
        # that one-shot attention graph can consume hundreds of GB. The output
        # is already target-verified and final. Publish the live cache only when
        # its exact target timeline is proved; otherwise fail closed on cache
        # reuse and let oMLX's block-aligned durable prefix plus idle target-only
        # tail reconstruction create the next exact resident entry.
        standard_terminal_exact = False
        active_state = getattr(gen_batch, "_omlx_mtp_state", None)
        if active_state is not None:
            standard_terminal_exact = _generic_mtp_terminal_cache_is_exact(
                gen_batch
            )
            if not standard_terminal_exact:
                logger.info(
                    "MTP terminal cache proof missed; skipping full-history "
                    "replay and suppressing terminal candidate for uid=%s",
                    getattr(active_state, "uid", "?"),
                )

        prompt_cache = gen_batch.extract_cache(0) if standard_terminal_exact else None
        all_tokens = gen_batch.tokens[0] if standard_terminal_exact else None
        response = Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=finish_reason,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=prompt_cache,
            all_tokens=all_tokens,
        )
        if standard_terminal_exact:
            response._omlx_mtp_standard_terminal_exact = True
        if stats is not None:
            _log_mtp_stats(gen_batch.uids[0], stats, finish_reason)
        # Drop state *before* filter([]) so the patched_filter epilogue
        # doesn't double-log when the standard finish path already logged.
        if hasattr(gen_batch, "_omlx_mtp_state"):
            try:
                delattr(gen_batch, "_omlx_mtp_state")
            except AttributeError:
                pass
        gen_batch.filter([])
        return [response]

    return [
        Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=None,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=None,
            all_tokens=None,
        )
    ]


def _validated_emitted_token(token_id: Any, logprobs_1d: Any) -> int:
    """Return a host token ID only when it can safely index its distribution.

    mlx-lm's private server indexes ``response.logprobs[response.token]`` on a
    background thread.  An invalid distributed decision used to escape this
    patch and kill that thread without terminating the rank process, leaving
    the coordinator to discover the failure only through a later heartbeat
    timeout.  Validate at the MTP boundary so the actual synchronized value
    and vocabulary width are preserved in the rank traceback.
    """

    value = int(token_id)
    try:
        vocab_size = int(logprobs_1d.shape[-1])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("MTP emitted token has no indexable logprob row") from exc
    if value < 0 or value >= vocab_size or value >= 2**31:
        raise RuntimeError(
            "MTP synchronized an invalid token ID: "
            f"token={value}, vocabulary={vocab_size}"
        )
    return value
