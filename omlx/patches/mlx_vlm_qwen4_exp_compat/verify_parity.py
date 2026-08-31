# SPDX-License-Identifier: Apache-2.0
"""Offline Qwen4 target-verify versus scalar-decode parity probe.

This module is diagnostic-only.  It never runs on the serving path unless
``OMLX_QWEN4_VERIFY_PARITY_PATH`` is set explicitly by an operator.  The probe
replays an already-observed MTP verify window against two fresh target caches:

* one ordinary multi-token target-verify call; and
* the canonical sequence of one-token target calls over the same input rows.

The comparison is intentionally stronger than an allclose check.  It records
the exact top-two vocabulary IDs and margins for every row, the first decoder
layer whose mixed hidden state is not bit-identical, and every persisted cache
leaf.  A greedy token mismatch is therefore observable at the cycle where it
is introduced instead of only as a final response hash.

The helper accepts a loaded Qwen4 model so the normal oMLX loader remains the
authority for quantization, PLE residency, native kernels, and checkpoint
sanitization.  It uses fresh caches and explicit text positions and suppresses
prompt-priming capture, leaving the caller's live cache and model timeline
untouched.  Running it still performs real model work and must be coordinated
like any other physical benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class ActiveVerifyParityProbe:
    """Materialized references retained until the live verify call completes."""

    report: dict[str, Any]
    scalar_logits: Any
    fresh_verify_logits: Any
    scalar_cache_leaves: list[tuple[str, Any]]
    fresh_verify_cache_leaves: list[tuple[str, Any]]
    active_base_scalar_logits: Any | None = None
    active_base_scalar_cache_leaves: list[tuple[str, Any]] | None = None


def _resolve_language_model(model: Any) -> Any:
    """Find the vendored Qwen4 ``LanguageModel`` through common wrappers."""

    pending = [model]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        inner = getattr(candidate, "model", None)
        if (
            callable(getattr(candidate, "make_cache", None))
            and inner is not None
            and type(inner).__name__ == "Qwen4ExpModel"
        ):
            return candidate
        for name in ("language_model", "_language_model", "model"):
            child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                pending.append(child)
    raise TypeError("Qwen4 verify parity probe requires a loaded qwen4_exp model")


def _token_sha256(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<I", int(value)))
    return digest.hexdigest()


def _arrays(value: Any) -> Iterable[Any]:
    import mlx.core as mx

    if isinstance(value, mx.array):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item)


def _cache_leaves(caches: Sequence[Any]) -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []
    for layer_index, cache in enumerate(caches):
        pending = [(f"layer_{layer_index}:{type(cache).__name__}", cache)]
        while pending:
            prefix, current = pending.pop()
            children = getattr(current, "caches", None)
            if children is not None:
                for child_index, child in enumerate(children):
                    pending.append((f"{prefix}/cache_{child_index}", child))
                continue
            for state_index, array in enumerate(_arrays(getattr(current, "state", ()))):
                leaves.append((f"{prefix}/state_{state_index}", array))
    return leaves


def snapshot_cache_leaves(caches: Sequence[Any]) -> list[tuple[str, Any]]:
    """Detach and materialize cache state under type-independent leaf names."""

    import mlx.core as mx

    leaves: list[tuple[str, Any]] = []
    for layer_index, cache in enumerate(caches):
        for state_index, array in enumerate(_arrays(getattr(cache, "state", ()))):
            # Arithmetic forces independent storage; a lazy contiguous view of
            # an active capacity buffer could otherwise observe the subsequent
            # verify write rather than the pre-forward state being diagnosed.
            detached = array + mx.zeros((), dtype=array.dtype)
            leaves.append((f"layer_{layer_index}/state_{state_index}", detached))
    mx.eval(*[array for _name, array in leaves])
    return leaves


def _leaf_comparisons(
    left: Sequence[tuple[str, Any]],
    right: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    left_names = [name for name, _array in left]
    right_names = [name for name, _array in right]
    layout_equal = left_names == right_names
    leaves = []
    if layout_equal:
        for (name, left_array), (_right_name, right_array) in zip(left, right):
            leaves.append(
                {"name": name, **_array_comparison(left_array, right_array)}
            )
    return {
        "layout_equal": layout_equal,
        "bitwise_equal": bool(
            layout_equal and leaves and all(item["array_equal"] for item in leaves)
        ),
        "first_mismatch": next(
            (item["name"] for item in leaves if not item["array_equal"]),
            None,
        ),
        "leaves": leaves,
    }


def _finite_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _array_comparison(left: Any, right: Any) -> dict[str, Any]:
    import mlx.core as mx

    result: dict[str, Any] = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "array_equal": False,
        "max_abs_delta": None,
    }
    if left.shape != right.shape or left.dtype != right.dtype:
        return result
    equal = bool(mx.array_equal(left, right).item())
    result["array_equal"] = equal
    if not equal and not mx.issubdtype(left.dtype, mx.integer):
        delta = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
        result["max_abs_delta"] = _finite_float(delta.item())
    return result


def _top_two_rows(logits: Any) -> list[dict[str, Any]]:
    """Materialize top-two IDs using production argmax tie semantics."""

    import mlx.core as mx

    flat = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    top1_ids = mx.argmax(flat, axis=-1).astype(mx.int32)
    top1_values = mx.take_along_axis(flat, top1_ids[:, None], axis=-1).squeeze(-1)
    vocab = mx.arange(flat.shape[-1], dtype=mx.int32)[None]
    without_top1 = mx.where(vocab == top1_ids[:, None], -mx.inf, flat)
    top2_ids = mx.argmax(without_top1, axis=-1).astype(mx.int32)
    top2_values = mx.take_along_axis(
        flat,
        top2_ids[:, None],
        axis=-1,
    ).squeeze(-1)
    mx.eval(top1_ids, top2_ids, top1_values, top2_values)
    id_rows = zip(top1_ids.tolist(), top2_ids.tolist())
    value_rows = zip(top1_values.tolist(), top2_values.tolist())
    return [
        {
            "top1_id": int(top1_id),
            "top2_id": int(top2_id),
            "top1_logit": _finite_float(top1_value),
            "top2_logit": _finite_float(top2_value),
            "margin": _finite_float(top1_value - top2_value),
        }
        for (top1_id, top2_id), (top1_value, top2_value) in zip(
            id_rows,
            value_rows,
        )
    ]


def _text_forward(
    language_model: Any,
    tokens: Any,
    cache: Sequence[Any],
    *,
    position_start: int,
    target_verify: bool,
    capture_layers: bool,
) -> tuple[Any, list[Any]]:
    """Run the real Qwen4 target trunk with explicit contiguous text positions."""

    import mlx.core as mx

    from omlx.patches.mlx_lm_mtp import prompt_priming

    module = __import__(type(language_model).__module__, fromlist=["*"])
    layer_count = len(language_model.model.layers)
    hidden_sink: list[Any] | None = [] if capture_layers else None
    gdn_sink: list[Any] | None = [] if target_verify else None
    positions = mx.arange(
        position_start,
        position_start + int(tokens.shape[1]),
        dtype=mx.int32,
    )[None]
    with prompt_priming.suppress_capture():
        hidden = language_model.model(
            tokens,
            cache=cache,
            position_ids=positions,
            capture_layer_ids=(list(range(layer_count)) if capture_layers else None),
            hidden_sink=hidden_sink,
            gdn_sink=gdn_sink,
        )
        if language_model.args.tie_word_embeddings:
            logits = module._target_verify_embedding_as_linear(
                language_model.model.embed_tokens,
                hidden,
                target_verify,
            )
        else:
            logits = module._target_verify_linear(
                language_model.lm_head,
                hidden,
                target_verify,
            )
    return logits, list(hidden_sink or ())


def _prefill(
    language_model: Any,
    token_ids: Sequence[int],
    cache: Sequence[Any],
    *,
    step: int,
    scalar_tail: int,
) -> None:
    """Build target state without projecting every prompt row through lm_head.

    A first MTP cycle's committed prefix ends in the prompt's last token and
    ``main_tok``. GenerationBatch and post-init process both as scalar rows, so
    the default two-row scalar tail reproduces that cache geometry rather than
    folding them into the final scheduler prefill chunk.
    """

    import mlx.core as mx

    from omlx.patches.mlx_lm_mtp import prompt_priming

    module = language_model.model
    values = list(token_ids)
    scalar_tail = min(max(0, int(scalar_tail)), len(values))
    bulk_stop = len(values) - scalar_tail
    spans = [
        (start, min(start + step, bulk_stop))
        for start in range(0, bulk_stop, step)
    ]
    spans.extend((index, index + 1) for index in range(bulk_stop, len(values)))
    for start, stop in spans:
        chunk = mx.array([values[start:stop]], dtype=mx.int32)
        positions = mx.arange(
            start,
            start + int(chunk.shape[1]),
            dtype=mx.int32,
        )[None]
        with prompt_priming.suppress_capture():
            hidden = module(
                chunk,
                cache=cache,
                position_ids=positions,
            )
        # The probe needs cache state only.  Skipping the 248K-vocabulary
        # projection avoids a multi-gigabyte [prompt, vocab] diagnostic tensor.
        mx.eval(hidden, *[array for _name, array in _cache_leaves(cache)])


def _scalar_window_from_active_cache(
    language_model: Any,
    verify_tokens: Sequence[int],
    cache: Sequence[Any],
) -> tuple[Any, list[tuple[str, Any]]]:
    """Run canonical L=1 target calls from an extracted live cache clone."""

    import mlx.core as mx

    from omlx.patches.mlx_lm_mtp import prompt_priming

    position_ids = getattr(language_model, "_position_ids", None)
    rope_deltas = getattr(language_model, "_rope_deltas", None)
    rows = []
    try:
        with prompt_priming.suppress_capture():
            for token in verify_tokens:
                output = language_model(
                    mx.array([[int(token)]], dtype=mx.int32),
                    cache=cache,
                )
                rows.append(output.logits)
        logits = mx.concatenate(rows, axis=1)
        mx.eval(logits, *[array for _name, array in _cache_leaves(cache)])
    finally:
        # Diagnostic replay must not alter the position state used by the live
        # verifier that immediately follows on the same model instance.
        language_model._position_ids = position_ids
        language_model._rope_deltas = rope_deltas
    return logits, snapshot_cache_leaves(cache)


def prepare_qwen4_active_verify_probe(
    model: Any,
    *,
    committed_prefix_tokens: Sequence[int],
    verify_tokens: Sequence[int],
    prefill_step: int = 4096,
    prefill_scalar_tail: int = 2,
    active_prefix_cache: Sequence[Any] | None = None,
) -> ActiveVerifyParityProbe:
    """Compare one observed Qwen4 MTP target window with scalar replay.

    ``committed_prefix_tokens`` must be exactly the tokens represented by the
    target cache immediately before the verify call.  ``verify_tokens`` is the
    physical target input window (``[next_main, draft_1, ..., draft_k]``), not
    merely the emitted continuation.
    """

    import mlx.core as mx

    prefix = [int(token) for token in committed_prefix_tokens]
    window = [int(token) for token in verify_tokens]
    if not prefix:
        raise ValueError("Qwen4 parity probe requires a non-empty committed prefix")
    if not 2 <= len(window) <= 9:
        raise ValueError("Qwen4 parity probe requires a 2..9-token verify window")
    if prefill_step <= 0:
        raise ValueError("Qwen4 parity probe prefill step must be positive")

    language_model = _resolve_language_model(model)
    batched_cache = language_model.make_cache()
    scalar_cache = language_model.make_cache()
    _prefill(
        language_model,
        prefix,
        batched_cache,
        step=prefill_step,
        scalar_tail=prefill_scalar_tail,
    )
    _prefill(
        language_model,
        prefix,
        scalar_cache,
        step=prefill_step,
        scalar_tail=prefill_scalar_tail,
    )
    fresh_prefix_leaves = snapshot_cache_leaves(scalar_cache)
    active_prefix_leaves = (
        snapshot_cache_leaves(active_prefix_cache)
        if active_prefix_cache is not None
        else None
    )
    active_base_scalar_logits = None
    active_base_scalar_cache_leaves = None
    if active_prefix_cache is not None:
        (
            active_base_scalar_logits,
            active_base_scalar_cache_leaves,
        ) = _scalar_window_from_active_cache(
            language_model,
            window,
            active_prefix_cache,
        )

    verify_array = mx.array([window], dtype=mx.int32)
    from omlx.patches import qwen35_verify_qmm

    qwen35_verify_qmm.apply_verify_qmm_patch()
    qwen35_verify_qmm.set_verify_qmm_armed(True)
    try:
        batched_logits, batched_layers = _text_forward(
            language_model,
            verify_array,
            batched_cache,
            position_start=len(prefix),
            target_verify=True,
            capture_layers=True,
        )
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)

    scalar_logits_rows = []
    scalar_layer_rows: list[list[Any]] = [
        [] for _ in range(len(language_model.model.layers))
    ]
    for row, token in enumerate(window):
        row_logits, row_layers = _text_forward(
            language_model,
            mx.array([[token]], dtype=mx.int32),
            scalar_cache,
            position_start=len(prefix) + row,
            target_verify=False,
            capture_layers=True,
        )
        scalar_logits_rows.append(row_logits)
        for layer_index, value in enumerate(row_layers):
            scalar_layer_rows[layer_index].append(value)

    scalar_logits = mx.concatenate(scalar_logits_rows, axis=1)
    scalar_layers = [mx.concatenate(rows, axis=1) for rows in scalar_layer_rows]
    materialize = [batched_logits, scalar_logits, *batched_layers, *scalar_layers]
    materialize.extend(array for _name, array in _cache_leaves(batched_cache))
    materialize.extend(array for _name, array in _cache_leaves(scalar_cache))
    mx.eval(*materialize)
    fresh_verify_cache_leaves = snapshot_cache_leaves(batched_cache)
    scalar_cache_leaves = snapshot_cache_leaves(scalar_cache)

    batched_top = _top_two_rows(batched_logits)
    scalar_top = _top_two_rows(scalar_logits)
    rows = []
    first_token_mismatch = None
    for row, (batched, scalar) in enumerate(zip(batched_top, scalar_top)):
        matches = batched["top1_id"] == scalar["top1_id"]
        if not matches and first_token_mismatch is None:
            first_token_mismatch = row
        rows.append(
            {
                "row": row,
                "input_token_id": window[row],
                "argmax_equal": matches,
                "verify": batched,
                "scalar": scalar,
            }
        )

    layer_reports = []
    first_layer_mismatch = None
    for layer_index, (batched, scalar) in enumerate(
        zip(batched_layers, scalar_layers)
    ):
        comparison = _array_comparison(batched, scalar)
        if not comparison["array_equal"] and first_layer_mismatch is None:
            first_layer_mismatch = layer_index
        layer_reports.append({"layer": layer_index, **comparison})

    fresh_cache_comparison = _leaf_comparisons(
        fresh_verify_cache_leaves,
        scalar_cache_leaves,
    )

    report = {
        "schema_version": 1,
        "created_unix": time.time(),
        "model_type": str(getattr(language_model, "model_type", "")),
        "prefix_tokens": len(prefix),
        "prefill_step": int(prefill_step),
        "prefill_scalar_tail": int(prefill_scalar_tail),
        "prefix_token_ids_sha256": _token_sha256(prefix),
        "verify_token_ids": window,
        "verify_token_ids_sha256": _token_sha256(window),
        "verify_width": len(window),
        "rows": rows,
        "logits": _array_comparison(batched_logits, scalar_logits),
        "argmax_parity": first_token_mismatch is None,
        "first_token_mismatch_row": first_token_mismatch,
        "layers": layer_reports,
        "first_hidden_mismatch_layer": first_layer_mismatch,
        "fresh_verify_vs_scalar_cache": fresh_cache_comparison,
        # Backward-compatible headline fields for the first probe artifact.
        "cache_layout_equal": fresh_cache_comparison["layout_equal"],
        "cache_leaves": fresh_cache_comparison["leaves"],
        "cache_bitwise_equal": fresh_cache_comparison["bitwise_equal"],
    }
    if active_prefix_leaves is not None:
        report["active_pre_vs_fresh_prefix_cache"] = _leaf_comparisons(
            active_prefix_leaves,
            fresh_prefix_leaves,
        )
    if active_base_scalar_logits is not None:
        report["active_base_scalar"] = {
            "rows": _top_two_rows(active_base_scalar_logits),
            "logits_vs_fresh_scalar": _array_comparison(
                active_base_scalar_logits,
                scalar_logits,
            ),
            "post_cache_vs_fresh_scalar": _leaf_comparisons(
                active_base_scalar_cache_leaves or [],
                scalar_cache_leaves,
            ),
        }
    return ActiveVerifyParityProbe(
        report=report,
        scalar_logits=scalar_logits,
        fresh_verify_logits=batched_logits,
        scalar_cache_leaves=scalar_cache_leaves,
        fresh_verify_cache_leaves=fresh_verify_cache_leaves,
        active_base_scalar_logits=active_base_scalar_logits,
        active_base_scalar_cache_leaves=active_base_scalar_cache_leaves,
    )


def capture_qwen4_active_verify_result(
    probe: ActiveVerifyParityProbe,
    *,
    active_logits: Any,
    active_cache: Sequence[Any],
) -> dict[str, Any]:
    """Attach the real verifier's logits/cache to a prepared fresh replay."""

    import mlx.core as mx

    if active_logits.ndim == 2:
        active_logits = active_logits[None]
    mx.eval(active_logits)
    active_top = _top_two_rows(active_logits)
    scalar_top = _top_two_rows(probe.scalar_logits)
    fresh_top = _top_two_rows(probe.fresh_verify_logits)
    active_base_top = (
        _top_two_rows(probe.active_base_scalar_logits)
        if probe.active_base_scalar_logits is not None
        else None
    )
    rows = []
    for row, (active, scalar, fresh) in enumerate(zip(active_top, scalar_top, fresh_top)):
        item = {
            "row": row,
            "active": active,
            "scalar": scalar,
            "fresh_verify": fresh,
            "active_vs_scalar_argmax": active["top1_id"] == scalar["top1_id"],
            "active_vs_fresh_argmax": active["top1_id"] == fresh["top1_id"],
        }
        if active_base_top is not None:
            item["active_base_scalar"] = active_base_top[row]
            item["active_vs_active_base_scalar_argmax"] = (
                active["top1_id"] == active_base_top[row]["top1_id"]
            )
        rows.append(item)
    active_leaves = snapshot_cache_leaves(active_cache)
    probe.report["active"] = {
        "rows": rows,
        "logits_vs_scalar": _array_comparison(active_logits, probe.scalar_logits),
        "logits_vs_fresh_verify": _array_comparison(
            active_logits,
            probe.fresh_verify_logits,
        ),
        "post_cache_vs_scalar": _leaf_comparisons(
            active_leaves,
            probe.scalar_cache_leaves,
        ),
        "post_cache_vs_fresh_verify": _leaf_comparisons(
            active_leaves,
            probe.fresh_verify_cache_leaves,
        ),
    }
    if probe.active_base_scalar_logits is not None:
        probe.report["active"]["logits_vs_active_base_scalar"] = _array_comparison(
            active_logits,
            probe.active_base_scalar_logits,
        )
        probe.report["active"]["post_cache_vs_active_base_scalar"] = (
            _leaf_comparisons(
                active_leaves,
                probe.active_base_scalar_cache_leaves or [],
            )
        )
    return probe.report


def compare_qwen4_verify_window(
    model: Any,
    *,
    committed_prefix_tokens: Sequence[int],
    verify_tokens: Sequence[int],
    prefill_step: int = 4096,
    prefill_scalar_tail: int = 2,
) -> dict[str, Any]:
    """Standalone fresh M-width versus scalar comparison."""

    return prepare_qwen4_active_verify_probe(
        model,
        committed_prefix_tokens=committed_prefix_tokens,
        verify_tokens=verify_tokens,
        prefill_step=prefill_step,
        prefill_scalar_tail=prefill_scalar_tail,
    ).report


def append_report(path: str | Path, report: dict[str, Any]) -> None:
    """Append one bounded JSON record to an explicitly requested trace file."""

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, separators=(",", ":")))
        handle.write("\n")


__all__ = [
    "ActiveVerifyParityProbe",
    "append_report",
    "capture_qwen4_active_verify_result",
    "compare_qwen4_verify_window",
    "prepare_qwen4_active_verify_probe",
    "snapshot_cache_leaves",
]
