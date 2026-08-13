# SPDX-License-Identifier: Apache-2.0
"""Two-rank loopback proof of the lockstep cancel paths.

Maintainer gate for the cancel design's step-boundary rule: two real ring
ranks decode a tiny MiniMax, rank zero arms the cancel latch mid-decode, and
every rank must surface the swapped stop id at the same step; then an armed
prefill must break at the first chunk boundary on both ranks.
"""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any

from .minimax_decode_smoke_worker import _CONFIG, _DecodeBatch, _RecordingModel
from .performance import execution_profile
from .runtime_optimizations import (
    LockstepPrefillCancelError,
    get_lockstep_controller,
    install_runtime_optimizations,
)

_STOP_TOKEN_ID = 5
_DECODE_STEPS = 4
_CANCEL_AT_STEP = 2
_PREFILL_STEP_SIZE = 4
_PREFILL_TOKENS = 10


class _CountingModel(_RecordingModel):
    """Count forwards so the cancelled prefill proves where it stopped."""

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self.forward_calls = 0

    def __call__(
        self,
        inputs: Any,
        cache: Any = None,
        skip_logits: bool = False,
    ) -> Any:
        self.forward_calls += 1
        return super().__call__(inputs, cache=cache, skip_logits=skip_logits)


class _PromptBatch:
    """The state consumed by the pinned PromptProcessingBatch.prompt."""

    def __init__(self, model: Any, cache: list[Any]) -> None:
        self.model = model
        self.uids = [1]
        self.tokens = [[]]
        self.prompt_cache = cache
        self.prefill_step_size = _PREFILL_STEP_SIZE


def main() -> int:
    try:
        import mlx.core as mx
        from mlx_lm.utils import _get_classes

        from omlx.patches.minimax_m3_mlx_lm import (
            apply_minimax_m3_mlx_lm_patch,
        )

        group = mx.distributed.init(backend="ring", strict=True)
        if group.size() != 2:
            raise RuntimeError("lockstep cancel smoke requires exactly two ranks")
        rank = int(group.rank())

        apply_minimax_m3_mlx_lm_patch()
        mlx_generate = importlib.import_module("mlx_lm.generate")
        model_cls, args_cls = _get_classes(_CONFIG)
        mx.random.seed(7)
        raw_model = model_cls(args_cls.from_dict(_CONFIG))
        raw_model.model.pipeline(group)

        model = _CountingModel(raw_model)
        settings = execution_profile("balanced", sampling_rank_only=True)
        with install_runtime_optimizations(
            model,
            group,
            settings,
            batchable=True,
            stop_token_ids=(_STOP_TOKEN_ID,),
        ) as capabilities:
            if not capabilities["lockstep_cancel"]["active"]:
                raise RuntimeError(
                    "lockstep cancel capability was not active: "
                    f"{capabilities['lockstep_cancel']}"
                )
            controller = get_lockstep_controller()

            # Decode: rank zero arms the latch after two steps. The EOS swap
            # enters the consumed stream one step later — at the same index
            # on every rank, or the lockstep claim is false.
            cache = raw_model.make_cache()
            if not cache:
                raise RuntimeError("lockstep cancel smoke built no KV cache")
            batch = _DecodeBatch(mx, model, cache)
            cancel_step = None
            for step in range(_DECODE_STEPS):
                if step == _CANCEL_AT_STEP and rank == 0:
                    controller.arm_cancel()
                mlx_generate.GenerationBatch._step(batch)
                mx.eval(batch._next_tokens)
                token = int(batch._next_tokens[0].item())
                if cancel_step is None and token == _STOP_TOKEN_ID:
                    cancel_step = step
            if cancel_step != _CANCEL_AT_STEP:
                raise RuntimeError(
                    f"rank {rank} observed the decode cancel at step "
                    f"{cancel_step}, expected {_CANCEL_AT_STEP}"
                )

            # Prefill: ten tokens at a four-token step is three chunks. Armed
            # before the call, the chunk collective must break the loop after
            # exactly one computed chunk — on both ranks.
            prefill_cache = raw_model.make_cache()
            prompt_batch = _PromptBatch(model, prefill_cache)
            forwards_before = model.forward_calls
            if rank == 0:
                controller.arm_cancel()
            cancelled = False
            try:
                mlx_generate.PromptProcessingBatch.prompt(
                    prompt_batch,
                    [list(range(_PREFILL_TOKENS))],
                )
            except LockstepPrefillCancelError:
                cancelled = True
            prefill_chunks = model.forward_calls - forwards_before
            if not cancelled:
                raise RuntimeError(f"rank {rank} prefill cancel never fired")
            if prefill_chunks != 1:
                raise RuntimeError(
                    f"rank {rank} computed {prefill_chunks} chunks after the "
                    "cancel, expected exactly 1"
                )

            record = {
                "type": "lockstep_cancel_result",
                "model_type": "minimax_m3_vl",
                "rank": rank,
                "size": int(group.size()),
                "decode_steps": _DECODE_STEPS,
                "cancel_step": cancel_step,
                "prefill_cancelled": cancelled,
                "prefill_chunks": prefill_chunks,
            }
            print(json.dumps(record, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            "oMLX lockstep cancel smoke failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
