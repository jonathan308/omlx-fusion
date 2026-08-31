#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Physical Qwen4 scalar-pair microbenchmark (explicit execution only)."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx


def _language_model(model: Any):
    pending = [model]
    seen = set()
    while pending:
        candidate = pending.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if callable(getattr(candidate, "scalar_pair", None)) and callable(
            getattr(candidate, "make_cache", None)
        ):
            return candidate
        for name in ("language_model", "_language_model", "model"):
            child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                pending.append(child)
    raise RuntimeError("loaded model has no Qwen4 scalar_pair primitive")


def _load_qwen4_vlm(model_path: Path):
    """Mirror the production Qwen4 VLM+mmap loader without starting an engine."""

    from mlx_vlm.utils import load as vlm_load

    from omlx.engine import vlm as engine_vlm
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    settings = SimpleNamespace(
        mtp_enabled=False,
        mtp_num_draft_tokens=0,
        qwen4_ple_ssd_offload=True,
        trust_remote_code=False,
    )
    maybe_apply_pre_load_patches(
        str(model_path),
        model_settings=settings,
        for_vlm=True,
    )
    with (
        engine_vlm._strip_audio_config_if_orphaned(model_path),
        engine_vlm._drop_gemma4_mlx_shared_kv_extras_on_load(model_path),
        engine_vlm._force_minimax_m3_moe_sanitize_on_load(model_path),
        engine_vlm._force_qwen4_exp_sanitize_on_load(model_path),
        engine_vlm._remap_nested_visual_on_load(model_path),
        engine_vlm._transpose_qwen35_mlx_vision_patch_embed_on_load(model_path),
        engine_vlm._load_optiq_vision_sidecar_on_load(model_path),
    ):
        return vlm_load(
            str(model_path),
            lazy=True,
            trust_remote_code=False,
        )


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item)


def _prefill(model, cache, tokens, chunk_size):
    for start in range(0, int(tokens.shape[1]), chunk_size):
        chunk = tokens[:, start : start + chunk_size]
        output = model(chunk, cache=cache)
        mx.eval(output.logits, *[entry.state for entry in cache])


def _snapshot(model, cache):
    recurrent = []
    qsa = []
    arrays = []
    for entry in cache:
        if type(entry).__name__ == "ArraysCache":
            state = tuple(value * 1 for value in entry.state)
            recurrent.append((entry, state))
            arrays.extend(_arrays(state))
        elif type(entry).__name__ == "QSAKVCache":
            qsa.append(
                (
                    entry,
                    int(entry.offset),
                    int(entry._index_offset),
                    entry._pooled_index_keys,
                    int(entry._pooled_index_offset),
                    entry._pooled_index_ratio,
                    entry._pooled_index_tag,
                )
            )
        else:
            raise RuntimeError(f"unsupported benchmark cache {type(entry).__name__}")
    mx.eval(*arrays)
    return (
        recurrent,
        qsa,
        model._position_ids,
        model._rope_deltas,
    )


def _restore(model, snapshot):
    recurrent, qsa, position_ids, rope_deltas = snapshot
    for entry, state in recurrent:
        entry.state = list(state)
    for entry, offset, index_offset, pooled, pooled_offset, ratio, tag in qsa:
        suffix = int(entry.offset) - offset
        if suffix < 0 or (suffix and int(entry.trim(suffix)) != suffix):
            raise RuntimeError("QSA benchmark restore failed")
        entry._index_offset = index_offset
        entry._pooled_index_keys = pooled
        entry._pooled_index_offset = pooled_offset
        entry._pooled_index_ratio = ratio
        entry._pooled_index_tag = tag
    model._position_ids = position_ids
    model._rope_deltas = rope_deltas


def _post_state(cache):
    states = []
    metadata = []
    arrays = []
    for entry in cache:
        state = tuple(value * 1 for value in entry.state if value is not None)
        states.append(state)
        arrays.extend(_arrays(state))
        if type(entry).__name__ == "QSAKVCache":
            pooled = (
                None
                if entry._pooled_index_keys is None
                else entry._pooled_index_keys[:, : entry._pooled_index_offset] * 1
            )
            metadata.append(
                (
                    int(entry.offset),
                    int(entry._index_offset),
                    pooled,
                    int(entry._pooled_index_offset),
                    entry._pooled_index_ratio,
                    entry._pooled_index_tag,
                    entry._index_capacity_managed,
                    entry._geometric_capacity_managed,
                    getattr(entry, "_omlx_text_position_ids_qualified", False),
                )
            )
            if pooled is not None:
                arrays.append(pooled)
        else:
            metadata.append(None)
    mx.eval(*arrays)
    return states, metadata


def _post_state_equal(cache, expected):
    expected_states, expected_metadata = expected
    for entry, wanted, meta in zip(cache, expected_states, expected_metadata):
        actual = tuple(value for value in entry.state if value is not None)
        mx.eval(*actual, *wanted)
        if len(actual) != len(wanted) or any(
            not mx.array_equal(left, right).item()
            for left, right in zip(actual, wanted)
        ):
            return False
        if meta is None:
            continue
        (
            offset,
            index_offset,
            pooled,
            pooled_offset,
            pooled_ratio,
            pooled_tag,
            index_managed,
            geometric_managed,
            text_qualified,
        ) = meta
        if (
            int(entry.offset) != offset
            or int(entry._index_offset) != index_offset
            or int(entry._pooled_index_offset) != pooled_offset
            or entry._pooled_index_ratio != pooled_ratio
            or entry._pooled_index_tag is not pooled_tag
            or entry._index_capacity_managed != index_managed
            or entry._geometric_capacity_managed != geometric_managed
            or getattr(entry, "_omlx_text_position_ids_qualified", False)
            != text_qualified
        ):
            return False
        actual_pooled = (
            None
            if entry._pooled_index_keys is None
            else entry._pooled_index_keys[:, : entry._pooled_index_offset]
        )
        if (actual_pooled is None) != (pooled is None):
            return False
        if pooled is not None:
            mx.eval(actual_pooled, pooled)
            if not mx.array_equal(actual_pooled, pooled).item():
                return False
    return True


def _measure_scalar(model, cache, tokens):
    started = time.perf_counter_ns()
    outputs = []
    for row in range(2):
        output = model(
            tokens[:, row : row + 1],
            cache=cache,
            return_hidden=True,
        )
        mx.eval(output.logits, output.hidden_states[-1], *[e.state for e in cache])
        outputs.append(output)
    mx.synchronize()
    return (time.perf_counter_ns() - started) / 1e6, outputs


def _measure_pair(model, cache, tokens, qualification, *, prove=False):
    started = time.perf_counter_ns()
    output = model.scalar_pair(
        tokens,
        cache=cache,
        qualification=qualification,
        use_batched_selector=prove,
        prove_selector_parity=prove,
    )
    mx.eval(output.logits, output.hidden_states[-1], *[e.state for e in cache])
    mx.synchronize()
    return (time.perf_counter_ns() - started) / 1e6, output


def _run_context(model, context, samples, warmup, chunk_size, seed):
    cache = model.make_cache()
    mx.random.seed(seed + context)
    prefix = mx.random.randint(
        2,
        int(model.args.vocab_size) - 1,
        shape=(1, context),
        dtype=mx.int32,
    )
    pair_tokens = mx.random.randint(
        2,
        int(model.args.vocab_size) - 1,
        shape=(1, 2),
        dtype=mx.int32,
    )
    mx.eval(prefix, pair_tokens)
    _prefill(model, cache, prefix, chunk_size)
    base = _snapshot(model, cache)
    qualification = model.qualify_scalar_pair_cache(cache)

    # Prove selector and complete post-state parity outside all timings.
    _restore(model, base)
    _scalar_proof_time, scalar_proof = _measure_scalar(model, cache, pair_tokens)
    scalar_logits = mx.concatenate([value.logits for value in scalar_proof], axis=1)
    scalar_hidden = mx.concatenate(
        [value.hidden_states[-1] for value in scalar_proof], axis=1
    )
    scalar_post_state = _post_state(cache)
    _restore(model, base)
    _proof_time, proof = _measure_pair(
        model,
        cache,
        pair_tokens,
        qualification,
        prove=True,
    )
    mx.eval(proof.logits, proof.hidden_states[-1], scalar_logits, scalar_hidden)
    exact = bool(mx.array_equal(proof.logits, scalar_logits).item())
    exact = exact and bool(
        mx.array_equal(proof.hidden_states[-1], scalar_hidden).item()
    )
    exact = exact and _post_state_equal(cache, scalar_post_state)

    scalar_ms = []
    pair_ms = []
    for iteration in range(warmup + samples):
        pair_first = iteration % 2 == 1
        if pair_first:
            _restore(model, base)
            pair_time, pair = _measure_pair(
                model,
                cache,
                pair_tokens,
                qualification,
            )
            _restore(model, base)
            scalar_time, scalar = _measure_scalar(model, cache, pair_tokens)
        else:
            _restore(model, base)
            scalar_time, scalar = _measure_scalar(model, cache, pair_tokens)
            _restore(model, base)
            pair_time, pair = _measure_pair(
                model,
                cache,
                pair_tokens,
                qualification,
            )
        if iteration >= warmup:
            scalar_ms.append(scalar_time)
            pair_ms.append(pair_time)
    scalar_median = statistics.median(scalar_ms)
    pair_median = statistics.median(pair_ms)
    return {
        "context": context,
        "samples": samples,
        "scalar_two_rows_ms": scalar_median,
        "scalar_selector_pair_ms": pair_median,
        "scalar_selector_pair_over_two_scalar": pair_median / scalar_median,
        "logits_hidden_array_equal": exact,
        "full_post_state_array_equal": exact,
        "batched_selector_timed": False,
        "selector_proof_untimed": True,
        "selector_parity_proven_layers": sum(
            getattr(layer.self_attn, "_omlx_scalar_pair_selector_parity", False)
            for layer in model.model.layers
            if not layer.is_linear
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contexts", default="10000,220000")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing model/GPU work without --execute")
    if args.samples < 1 or args.warmup < 1 or args.chunk_size < 1:
        raise SystemExit("--samples, --warmup, and --chunk-size must be positive")

    loaded, _processor = _load_qwen4_vlm(args.model)
    model = _language_model(loaded)
    contexts = [int(value) for value in args.contexts.split(",")]
    results = [
        _run_context(
            model,
            context,
            args.samples,
            args.warmup,
            args.chunk_size,
            args.seed,
        )
        for context in contexts
    ]
    report = {
        "model": str(args.model),
        "threshold_220k": 0.805,
        "timed_selector_mode": "scalar-row selector; batched selector proof excluded",
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")
    if args.strict:
        qsa_layers = sum(not layer.is_linear for layer in model.model.layers)
        if not any(row["context"] == 220_000 for row in results):
            raise SystemExit("strict scalar-pair gate requires context 220000")
        failures = [
            row
            for row in results
            if not row["full_post_state_array_equal"]
            or row["selector_parity_proven_layers"] != qsa_layers
            or (
                row["context"] == 220_000
                and row["scalar_selector_pair_over_two_scalar"] > 0.805
            )
        ]
        if failures:
            raise SystemExit("strict scalar-pair benchmark gate failed")


if __name__ == "__main__":
    main()
