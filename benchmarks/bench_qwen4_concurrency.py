#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Black-box OpenAI-stream benchmark for Qwen4 concurrency and cache safety.

The harness deliberately talks only to the public ``/v1/chat/completions``
surface.  It does not import MLX, mutate server settings, load a model, or
reach into scheduler internals.  That makes a run representative of an agent
harness and lets the same report compare oMLX builds (or another compatible
server) without changing the benchmark.

Measured contracts:

* 1/2/4/6-stream per-request TTFT and decode throughput;
* aggregate decode throughput, finish skew, stalls, and B1 retention/scaling;
* prefix-cache tokens and hit ratio while requests overlap;
* response/output hashes, unique response IDs, and per-stream marker isolation;
* an optional mixed long-prefill + active-decode fairness phase; and
* an optional targeted disconnect where one stream is cancelled and the
  remaining streams must finish independently.

Examples::

    python benchmarks/bench_qwen4_concurrency.py \
      --endpoint http://127.0.0.1:8000/v1 \
      --model Qwen3.8-Flash-Next-oQ4e-mtp \
      --contexts 20k,50k,100k,150k,200k,220k \
      --streams 1,2,4,6 --max-tokens 500 --prime-cache \
      --prime-each-stream \
      --mixed --cancellation --output /tmp/qwen4-concurrency.json

    python benchmarks/bench_qwen4_concurrency.py --self-test

    python benchmarks/bench_qwen4_concurrency.py \
      --model Qwen3.8-Flash-Next-oQ4e-mtp --streams 1 \
      --contexts 20k,50k,100k,150k,200k,220k \
      --prompt-mode predictable-code --calibrate-contexts \
      --no-prime-cache --max-tokens 500

Context sizes are targets.  With ``--calibrate-contexts`` the oMLX Anthropic
token-count endpoint is used to bring the generated prompt close to the target;
otherwise the final report's server-reported ``prompt_tokens`` is authoritative.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Sequence
import uuid

import httpx


SCHEMA_VERSION = 3
DEFAULT_UNIT = (
    "def qwen4_scheduler_step(sequence, cache):\n"
    "    selected = cache.exact_top_blocks(sequence, limit=512)\n"
    "    return sequence.attend(selected, fp32_scores=True)\n\n"
)
PREDICTABLE_CODE_MODE = "predictable-code"


def _predictable_code_unit(index: int) -> str:
    """One deterministic continuation unit with a monotonically numbered name."""

    return (
        f"def qwen4_transform_{index:06d}(value: int) -> int:\n"
        "    doubled = value * 2\n"
        "    adjusted = doubled + 1\n"
        "    return adjusted\n\n"
    )


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, digits)


def _parse_scaled_int(raw: str) -> int:
    text = raw.strip().lower().replace("_", "")
    scale = 1
    if text.endswith("k"):
        scale, text = 1_000, text[:-1]
    elif text.endswith("m"):
        scale, text = 1_000_000, text[:-1]
    value = float(text) * scale
    if value <= 0 or not value.is_integer():
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {raw!r}")
    return int(value)


def _parse_int_list(raw: str) -> list[int]:
    values = [_parse_scaled_int(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _chat_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def _v1_url(endpoint: str, suffix: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        endpoint = endpoint[: -len("/chat/completions")]
    elif not endpoint.endswith("/v1"):
        endpoint += "/v1"
    return endpoint + "/" + suffix.lstrip("/")


def _headers(api_key: str | None) -> dict[str, str]:
    result = {"Accept": "text/event-stream"}
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    return result


def _event_from_sse_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("SSE data must contain a JSON object")
    return value


def _flatten_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return str(value)


def _delta_payload(event: dict[str, Any]) -> str:
    parts: list[str] = []
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        text = _flatten_content(delta.get("content"))
        reasoning = _flatten_content(
            delta.get("reasoning_content") or delta.get("reasoning")
        )
        if reasoning:
            parts.append(reasoning)
        if text:
            parts.append(text)
        if delta.get("tool_calls"):
            parts.append(
                json.dumps(
                    delta["tool_calls"], sort_keys=True, separators=(",", ":")
                )
            )
        if choice.get("text"):
            parts.append(str(choice["text"]))
    return "".join(parts)


def _finish_reason(event: dict[str, Any]) -> str | None:
    for choice in event.get("choices") or []:
        value = choice.get("finish_reason")
        if value is not None:
            return str(value)
    return None


def _cached_tokens(usage: dict[str, Any]) -> int:
    details = (
        usage.get("prompt_tokens_details")
        or usage.get("input_tokens_details")
        or {}
    )
    return int(details.get("cached_tokens") or 0)


def _server_acceptance_telemetry(
    event: dict[str, Any],
) -> list[dict[str, Any]]:
    """Copy explicit server acceptance fields without deriving a ratio."""

    field_names = (
        "acceptance_ratio",
        "mtp_acceptance_ratio",
        "speculative_acceptance_ratio",
        "acceptance_percent",
        "mtp_acceptance_percent",
        "accepted_draft_tokens",
        "drafted_tokens",
        "tokens_per_cycle",
        "mtp_tokens_per_cycle",
    )
    containers = (
        ("event", event),
        ("usage", event.get("usage")),
        ("metrics", event.get("metrics")),
        ("telemetry", event.get("telemetry")),
        ("mtp", event.get("mtp")),
        ("speculative", event.get("speculative")),
    )
    found: list[dict[str, Any]] = []
    for source, container in containers:
        if not isinstance(container, dict):
            continue
        values = {
            name: container[name]
            for name in field_names
            if isinstance(container.get(name), (int, float))
            and not isinstance(container.get(name), bool)
        }
        if values:
            found.append(
                {
                    "source": source,
                    "values": values,
                    "derived": False,
                }
            )
    return found


@dataclass(frozen=True)
class StreamSpec:
    label: str
    marker: str
    prompt: str
    max_tokens: int
    cancel_after_chunks: int = 0
    prompt_mode: str = "default"


@dataclass
class StreamResult:
    label: str
    marker: str
    status: str = "pending"
    http_status: int | None = None
    error: str | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    started_at: float = 0.0
    first_token_at: float | None = None
    finished_at: float = 0.0
    content_event_times: list[float] = field(default_factory=list)
    content_events: int = 0
    output: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    acceptance_telemetry: list[dict[str, Any]] = field(default_factory=list)

    def report(self, epoch: float) -> dict[str, Any]:
        prompt_tokens = int(self.usage.get("prompt_tokens") or 0)
        completion_tokens = int(self.usage.get("completion_tokens") or 0)
        cached = _cached_tokens(self.usage)
        first = self.first_token_at
        finish = self.finished_at
        client_ttft = None if first is None else first - self.started_at
        client_generation = None if first is None else max(0.0, finish - first)
        client_tps = (
            None
            if not client_generation or completion_tokens <= 0
            else completion_tokens / client_generation
        )
        gaps = [
            b - a
            for a, b in zip(self.content_event_times, self.content_event_times[1:])
        ]
        server_tps = self.usage.get("generation_tokens_per_second")
        output_hash = hashlib.sha256(self.output.encode("utf-8")).hexdigest()
        report = {
            "label": self.label,
            "marker": self.marker,
            "status": self.status,
            "http_status": self.http_status,
            "error": self.error,
            "response_id": self.response_id,
            "finish_reason": self.finish_reason,
            "started_s": _round(self.started_at - epoch),
            "first_token_s": None if first is None else _round(first - epoch),
            "finished_s": _round(finish - epoch),
            "client_ttft_s": _round(client_ttft),
            "server_ttft_s": _round(self.usage.get("time_to_first_token")),
            "client_generation_s": _round(client_generation),
            "server_generation_s": _round(self.usage.get("generation_duration")),
            "client_decode_tps": _round(client_tps),
            "server_decode_tps": _round(float(server_tps)) if server_tps else None,
            "server_prefill_tps": _round(self.usage.get("prompt_tokens_per_second")),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached,
            "uncached_prompt_tokens": max(0, prompt_tokens - cached),
            "cache_hit_ratio": _round(cached / prompt_tokens) if prompt_tokens else 0.0,
            "content_events": self.content_events,
            "inter_event_gap_p50_s": _round(_percentile(gaps, 0.50)),
            "inter_event_gap_p95_s": _round(_percentile(gaps, 0.95)),
            "max_inter_event_gap_s": _round(max(gaps)) if gaps else None,
            "output_bytes": len(self.output.encode("utf-8")),
            "output_sha256": output_hash,
            "marker_present": self.marker in self.output,
            "output_preview": self.output[:160],
            "usage": self.usage,
        }
        if self.acceptance_telemetry:
            report["server_acceptance_telemetry"] = self.acceptance_telemetry
        return report


def _cache_coverage(streams: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only server-observed cache coverage; infer no block size."""

    observed = [row for row in streams if row["status"] == "completed"]
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in observed)
    cached_tokens = sum(int(row["cached_tokens"]) for row in observed)
    return {
        "streams_observed": len(observed),
        "streams_with_cache_hit": sum(
            int(row["cached_tokens"]) > 0 for row in observed
        ),
        "total_prompt_tokens": prompt_tokens,
        "total_cached_tokens": cached_tokens,
        "total_uncached_prompt_tokens": max(0, prompt_tokens - cached_tokens),
        "aggregate_cache_hit_ratio": _round(cached_tokens / prompt_tokens)
        if prompt_tokens
        else 0.0,
        "per_stream": [
            {
                "label": row["label"],
                "prompt_tokens": row["prompt_tokens"],
                "cached_tokens": row["cached_tokens"],
                "uncached_prompt_tokens": row["uncached_prompt_tokens"],
                "cache_hit_ratio": row["cache_hit_ratio"],
            }
            for row in observed
        ],
        "interpretation": (
            "Coverage comes from prompt_tokens_details.cached_tokens. The public "
            "API does not expose cache block size, so this report does not claim "
            "that a partially covered final block was restored."
        ),
    }


def _max_observed_active_decode_batch(
    results: Sequence[StreamResult],
) -> int | None:
    """Return maximum overlapping first-token-to-finish intervals.

    This is a black-box concurrency observation, not a claim about the
    scheduler's internal kernel batch width. End events sort before starts at
    the same timestamp so merely adjacent streams are not counted as overlap.
    """

    events: list[tuple[float, int, int]] = []
    for result in results:
        if (
            result.status != "completed"
            or result.first_token_at is None
            or result.finished_at <= result.first_token_at
        ):
            continue
        events.append((result.first_token_at, 1, 1))
        events.append((result.finished_at, 0, -1))
    if not events:
        return None
    active = 0
    maximum = 0
    for _timestamp, _order, delta in sorted(events):
        active += delta
        maximum = max(maximum, active)
    return maximum


async def _stream_request(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    model: str,
    spec: StreamSpec,
    start_gate: asyncio.Event,
    first_token_event: asyncio.Event | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
) -> StreamResult:
    result = StreamResult(label=spec.label, marker=spec.marker)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "max_tokens": spec.max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if spec.prompt_mode == PREDICTABLE_CODE_MODE:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if seed is not None:
        payload["seed"] = seed
    await start_gate.wait()
    result.started_at = time.perf_counter()
    try:
        async with client.stream("POST", _chat_url(endpoint), json=payload) as response:
            result.http_status = response.status_code
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status_code}: {body[:500]}")
            output_parts: list[str] = []
            async for line in response.aiter_lines():
                event = _event_from_sse_line(line)
                if event is None:
                    continue
                if event.get("error"):
                    raise RuntimeError(json.dumps(event["error"], sort_keys=True))
                if event.get("id") and result.response_id is None:
                    result.response_id = str(event["id"])
                if event.get("usage"):
                    result.usage = dict(event["usage"])
                for telemetry in _server_acceptance_telemetry(event):
                    if telemetry not in result.acceptance_telemetry:
                        result.acceptance_telemetry.append(telemetry)
                finish_reason = _finish_reason(event)
                if finish_reason is not None:
                    result.finish_reason = finish_reason
                delta = _delta_payload(event)
                if not delta:
                    continue
                now = time.perf_counter()
                if result.first_token_at is None:
                    result.first_token_at = now
                    if first_token_event is not None:
                        first_token_event.set()
                result.content_event_times.append(now)
                result.content_events += 1
                output_parts.append(delta)
                if (
                    spec.cancel_after_chunks > 0
                    and result.content_events >= spec.cancel_after_chunks
                ):
                    result.status = "cancelled_by_client"
                    break
            result.output = "".join(output_parts)
            if result.status == "pending":
                result.status = "completed"
    except asyncio.CancelledError:
        result.status = "cancelled_by_task"
        raise
    except Exception as exc:  # report failures without losing sibling results
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.finished_at = time.perf_counter()
    return result


def _round_report(
    *,
    label: str,
    results: Sequence[StreamResult],
    epoch: float,
    round_started: float,
    round_finished: float,
) -> dict[str, Any]:
    streams = [result.report(epoch) for result in results]
    completed = [row for row in streams if row["status"] == "completed"]
    first_times = [
        result.first_token_at
        for result in results
        if result.status == "completed" and result.first_token_at is not None
    ]
    finish_times = [
        result.finished_at for result in results if result.status == "completed"
    ]
    total_tokens = sum(int(row["completion_tokens"]) for row in completed)
    decode_window = (
        max(finish_times) - min(first_times)
        if first_times and finish_times
        else None
    )
    server_rates = [
        float(row["server_decode_tps"])
        for row in completed
        if row["server_decode_tps"] is not None
    ]
    output_hashes = [row["output_sha256"] for row in completed]
    response_ids = [row["response_id"] for row in completed if row["response_id"]]
    acceptance_rows = [
        {
            "label": row["label"],
            "telemetry": row["server_acceptance_telemetry"],
        }
        for row in completed
        if row.get("server_acceptance_telemetry")
    ]
    report = {
        "label": label,
        "stream_count": len(results),
        "completed_streams": len(completed),
        "cancelled_streams": sum(
            row["status"].startswith("cancelled") for row in streams
        ),
        "error_streams": sum(row["status"] == "error" for row in streams),
        "wall_s": _round(round_finished - round_started),
        "decode_window_s": _round(decode_window),
        "aggregate_decode_tps": _round(total_tokens / decode_window)
        if decode_window
        else None,
        "aggregate_e2e_tps": _round(total_tokens / (round_finished - round_started))
        if round_finished > round_started
        else None,
        "sum_server_decode_tps": _round(sum(server_rates)) if server_rates else None,
        "median_server_decode_tps": _round(statistics.median(server_rates))
        if server_rates
        else None,
        "min_server_decode_tps": _round(min(server_rates)) if server_rates else None,
        "max_server_decode_tps": _round(max(server_rates)) if server_rates else None,
        "per_stream_fairness_min_over_max": _round(
            min(server_rates) / max(server_rates)
        )
        if server_rates and max(server_rates) > 0
        else None,
        "finish_skew_s": _round(max(finish_times) - min(finish_times))
        if len(finish_times) > 1
        else 0.0,
        "median_cache_hit_ratio": _round(
            statistics.median(float(row["cache_hit_ratio"]) for row in completed)
        )
        if completed
        else None,
        "min_cache_hit_ratio": _round(
            min(float(row["cache_hit_ratio"]) for row in completed)
        )
        if completed
        else None,
        "all_markers_present": bool(completed)
        and all(bool(row["marker_present"]) for row in completed),
        "output_hashes_unique": len(output_hashes) == len(set(output_hashes)),
        "response_ids_unique": len(response_ids) == len(completed)
        and len(response_ids) == len(set(response_ids)),
        "cache_coverage": _cache_coverage(streams),
        "streams": streams,
    }
    observed_batch = _max_observed_active_decode_batch(results)
    if observed_batch is not None:
        report["max_observed_active_decode_batch"] = observed_batch
        report["active_decode_batch_observation"] = (
            "Maximum overlap of client-observed decode intervals (first content "
            "event through stream finish). This proves concurrent active decode "
            "windows, not the server's private per-step kernel batch width."
        )
    if acceptance_rows:
        report["server_acceptance_telemetry"] = acceptance_rows
    return report


async def _run_round(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    model: str,
    specs: Sequence[StreamSpec],
    epoch: float,
    label: str,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    gate = asyncio.Event()
    tasks = [
        asyncio.create_task(
            _stream_request(
                client,
                endpoint=endpoint,
                model=model,
                spec=spec,
                start_gate=gate,
                temperature=temperature,
                seed=seed,
            )
        )
        for spec in specs
    ]
    await asyncio.sleep(0)
    started = time.perf_counter()
    gate.set()
    results = await asyncio.gather(*tasks)
    finished = time.perf_counter()
    return _round_report(
        label=label,
        results=results,
        epoch=epoch,
        round_started=started,
        round_finished=finished,
    )


def _make_shared_prefix(
    target_tokens: int,
    *,
    run_salt: str,
    chars_per_token: float,
    corpus: str,
    repeats: int | None = None,
    prompt_mode: str = "default",
) -> tuple[str, int]:
    if prompt_mode == PREDICTABLE_CODE_MODE:
        header = (
            f"# QWEN4 PREDICTABLE CODE BENCHMARK {run_salt}\n"
            "# Continue the monotonically numbered function series exactly.\n\n"
        )
        sample = _predictable_code_unit(0)
        if repeats is None:
            target_chars = max(
                1,
                int(target_tokens * chars_per_token) - len(header),
            )
            repeats = max(1, math.ceil(target_chars / len(sample)))
        code = "".join(_predictable_code_unit(index) for index in range(repeats))
        return header + code, repeats

    header = (
        f"QWEN4 EXACT CONCURRENCY BENCHMARK {run_salt}.\n"
        "Treat all preceding source as immutable context.\n\n"
    )
    if repeats is None:
        target_chars = max(1, int(target_tokens * chars_per_token) - len(header))
        repeats = max(1, math.ceil(target_chars / len(corpus)))
    return header + corpus * repeats, repeats


def _prompt(
    prefix: str,
    marker: str,
    *,
    prompt_mode: str = "default",
) -> str:
    if prompt_mode == PREDICTABLE_CODE_MODE:
        next_index = prefix.count("def qwen4_transform_")
        return (
            prefix
            + "# CONTINUATION TASK\n"
            + f"# First output line: # {marker}\n"
            + f"# Then begin with qwen4_transform_{next_index:06d} and continue "
            "the exact numbered function pattern above.\n"
            "# Output Python code only. No explanation, reasoning, headings, or "
            "Markdown fences. Do not stop before the token limit unless a complete "
            "function would not fit.\n"
        )
    return (
        prefix
        + "\nEND IMMUTABLE CONTEXT.\n"
        + f"Your private stream marker is {marker}. "
        + f"Begin the answer with exactly `{marker}`. Then explain in technical "
        "prose why deterministic cache isolation matters. Do not copy another "
        "stream's marker. Continue until the requested token limit.\n"
    )


async def _count_prompt_tokens(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: str,
    *,
    prompt_mode: str = "default",
) -> int:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if prompt_mode == PREDICTABLE_CODE_MODE:
        payload["thinking"] = {"type": "disabled"}
    response = await client.post(
        _v1_url(endpoint, "messages/count_tokens"),
        json=payload,
    )
    response.raise_for_status()
    return int(response.json()["input_tokens"])


async def _calibrate_prefix(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    model: str,
    target_tokens: int,
    run_salt: str,
    chars_per_token: float,
    corpus: str,
    prompt_mode: str = "default",
) -> tuple[str, dict[str, Any]]:
    prefix, repeats = _make_shared_prefix(
        target_tokens,
        run_salt=run_salt,
        chars_per_token=chars_per_token,
        corpus=corpus,
        prompt_mode=prompt_mode,
    )
    marker = "ISOLATION-CALIBRATION"
    attempts: list[dict[str, int]] = []
    try:
        for attempt_index in range(4):
            actual = await _count_prompt_tokens(
                client,
                endpoint,
                model,
                _prompt(prefix, marker, prompt_mode=prompt_mode),
                prompt_mode=prompt_mode,
            )
            attempts.append({"repeats": repeats, "tokens": actual})
            error = target_tokens - actual
            if abs(error) <= max(32, int(target_tokens * 0.0025)):
                break
            tokens_per_repeat = max(actual / repeats, 0.1)
            if attempt_index == 3:
                break
            repeats = max(1, repeats + round(error / tokens_per_repeat))
            prefix, _ = _make_shared_prefix(
                target_tokens,
                run_salt=run_salt,
                chars_per_token=chars_per_token,
                corpus=corpus,
                repeats=repeats,
                prompt_mode=prompt_mode,
            )
        return prefix, {
            "method": "omlx_count_tokens",
            "target_tokens": target_tokens,
            "final_tokens": attempts[-1]["tokens"],
            "repeats": repeats,
            "prompt_mode": prompt_mode,
            "attempts": attempts,
        }
    except Exception as exc:
        return prefix, {
            "method": "character_estimate_fallback",
            "target_tokens": target_tokens,
            "repeats": repeats,
            "prompt_mode": prompt_mode,
            "error": f"{type(exc).__name__}: {exc}",
            "attempts": attempts,
        }


async def _discover_model(
    client: httpx.AsyncClient, endpoint: str, requested: str | None
) -> str:
    if requested:
        return requested
    response = await client.get(_v1_url(endpoint, "models"))
    response.raise_for_status()
    models = response.json().get("data") or []
    if not models:
        raise RuntimeError("/v1/models returned no loaded/available model")
    return str(models[0]["id"])


def _specs(
    *,
    prefix: str,
    context: int,
    count: int,
    max_tokens: int,
    phase: str,
    cancel_first_after_chunks: int = 0,
    prompt_mode: str = "default",
) -> list[StreamSpec]:
    result = []
    for index in range(count):
        if prompt_mode == PREDICTABLE_CODE_MODE:
            marker = f"ISOLATION-{phase}-C{context}-S{index}"
        else:
            marker = (
                f"ISOLATION-{phase}-C{context}-S{index}-{uuid.uuid4().hex[:10]}"
            )
        result.append(
            StreamSpec(
                label=f"{phase}-c{context}-s{index}",
                marker=marker,
                prompt=_prompt(prefix, marker, prompt_mode=prompt_mode),
                max_tokens=max_tokens,
                cancel_after_chunks=cancel_first_after_chunks if index == 0 else 0,
                prompt_mode=prompt_mode,
            )
        )
    return result


async def _prime_cache(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    model: str,
    prefix: str,
    context: int,
    epoch: float,
    temperature: float,
    seed: int | None,
    prompt_mode: str = "default",
) -> dict[str, Any]:
    return await _run_round(
        client,
        endpoint=endpoint,
        model=model,
        specs=_specs(
            prefix=prefix,
            context=context,
            count=1,
            max_tokens=1,
            phase="cache-prime",
            prompt_mode=prompt_mode,
        ),
        epoch=epoch,
        label=f"cache-prime-{context}",
        temperature=temperature,
        seed=seed,
    )


async def _prime_each_stream(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    model: str,
    specs: Sequence[StreamSpec],
    epoch: float,
    label: str,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    """Serially prime every exact prompt/marker before its measured round.

    The one-token request changes only ``max_tokens``. Prompt text and marker
    are byte-identical to the later matrix request. Serial execution makes
    cache publication deterministic and avoids calling simultaneous cold
    requests a successful prime when none could have observed another's
    completed blocks.
    """

    rounds: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        prime_spec = StreamSpec(
            label=spec.label,
            marker=spec.marker,
            prompt=spec.prompt,
            max_tokens=1,
            prompt_mode=spec.prompt_mode,
        )
        rounds.append(
            await _run_round(
                client,
                endpoint=endpoint,
                model=model,
                specs=[prime_spec],
                epoch=epoch,
                label=f"{label}-stream-{index}",
                temperature=temperature,
                seed=seed,
            )
        )
    stream_rows = [round_["streams"][0] for round_ in rounds]
    return {
        "label": label,
        "mode": "serial_exact_prompt_one_token",
        "stream_count": len(specs),
        "completed_streams": sum(
            row["status"] == "completed" for row in stream_rows
        ),
        "all_exact_prompts_primed": bool(stream_rows)
        and all(row["status"] == "completed" for row in stream_rows),
        "cache_coverage_during_prime": _cache_coverage(stream_rows),
        "rounds": rounds,
    }


async def _run_mixed_phase(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    model: str,
    victim_prefix: str,
    prefill_prefix: str,
    victim_context: int,
    prefill_context: int,
    prefillers: int,
    max_tokens: int,
    epoch: float,
    temperature: float,
    seed: int | None,
    first_token_timeout: float,
    prompt_mode: str = "default",
) -> dict[str, Any]:
    # Prime only the common victim prefix so both the standalone baseline and
    # the contended victim begin from the same cache/MTP-history state.
    victim_prime = await _prime_cache(
        client,
        endpoint=endpoint,
        model=model,
        prefix=victim_prefix,
        context=victim_context,
        epoch=epoch,
        temperature=temperature,
        seed=seed,
        prompt_mode=prompt_mode,
    )
    baseline_spec = _specs(
        prefix=victim_prefix,
        context=victim_context,
        count=1,
        max_tokens=max_tokens,
        phase="mixed-baseline",
        prompt_mode=prompt_mode,
    )[0]
    baseline = await _run_round(
        client,
        endpoint=endpoint,
        model=model,
        specs=[baseline_spec],
        epoch=epoch,
        label="mixed-baseline",
        temperature=temperature,
        seed=seed,
    )

    first_token = asyncio.Event()
    victim_gate = asyncio.Event()
    victim_spec = _specs(
        prefix=victim_prefix,
        context=victim_context,
        count=1,
        max_tokens=max_tokens,
        phase="mixed-victim",
        prompt_mode=prompt_mode,
    )[0]
    victim_task = asyncio.create_task(
        _stream_request(
            client,
            endpoint=endpoint,
            model=model,
            spec=victim_spec,
            start_gate=victim_gate,
            first_token_event=first_token,
            temperature=temperature,
            seed=seed,
        )
    )
    mixed_started = time.perf_counter()
    victim_gate.set()
    try:
        await asyncio.wait_for(first_token.wait(), timeout=first_token_timeout)
        prefill_started = time.perf_counter()
        intruder_specs = _specs(
            prefix=prefill_prefix,
            context=prefill_context,
            count=prefillers,
            max_tokens=1,
            phase="mixed-prefill",
            prompt_mode=prompt_mode,
        )
        intruder_gate = asyncio.Event()
        intruder_tasks = [
            asyncio.create_task(
                _stream_request(
                    client,
                    endpoint=endpoint,
                    model=model,
                    spec=spec,
                    start_gate=intruder_gate,
                    temperature=temperature,
                    seed=seed,
                )
            )
            for spec in intruder_specs
        ]
        intruder_gate.set()
        victim_result, intruder_results = await asyncio.gather(
            victim_task, asyncio.gather(*intruder_tasks)
        )
        mixed_finished = time.perf_counter()
    except Exception:
        victim_task.cancel()
        await asyncio.gather(victim_task, return_exceptions=True)
        raise

    mixed = _round_report(
        label="mixed-prefill-decode",
        results=[victim_result, *intruder_results],
        epoch=epoch,
        round_started=mixed_started,
        round_finished=mixed_finished,
    )
    baseline_tps = baseline.get("median_server_decode_tps")
    victim_tps = victim_result.report(epoch).get("server_decode_tps")
    mixed["prefill_started_s"] = _round(prefill_started - epoch)
    mixed["victim_baseline_decode_tps"] = baseline_tps
    mixed["victim_mixed_decode_tps"] = victim_tps
    mixed["victim_decode_retention"] = (
        _round(float(victim_tps) / float(baseline_tps))
        if victim_tps is not None and baseline_tps
        else None
    )
    return {"victim_cache_prime": victim_prime, "baseline": baseline, "mixed": mixed}


def _annotate_scaling(rounds: list[dict[str, Any]]) -> None:
    baselines: dict[int, float] = {}
    for row in rounds:
        if row.get("stream_count") != 1:
            continue
        rate = row.get("median_server_decode_tps")
        if rate:
            baselines[int(row["context_target"])] = float(rate)
    for row in rounds:
        baseline = baselines.get(int(row["context_target"]))
        median = row.get("median_server_decode_tps")
        aggregate = row.get("aggregate_decode_tps")
        row["b1_decode_tps"] = _round(baseline)
        row["per_stream_b1_retention"] = (
            _round(float(median) / baseline) if baseline and median else None
        )
        row["aggregate_b1_scaling"] = (
            _round(float(aggregate) / baseline) if baseline and aggregate else None
        )


def _evaluate_gates(
    report: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if args.prime_each_stream:
        for prime in report.get("stream_primes", []):
            if not prime["all_exact_prompts_primed"]:
                failures.append(
                    {
                        "round": prime["label"],
                        "gate": "all_exact_prompts_primed",
                    }
                )
    for row in report["matrix"]:
        label = row["label"]
        checks = {
            "all_requests_completed": row["completed_streams"] == row["stream_count"],
            "response_ids_unique": row["response_ids_unique"],
            "output_hashes_unique": row["output_hashes_unique"],
        }
        if args.require_markers:
            checks["all_markers_present"] = row["all_markers_present"]
        if args.min_cache_hit_ratio is not None and (
            args.prime_cache or args.prime_each_stream
        ):
            value = row.get("min_cache_hit_ratio")
            checks["min_cache_hit_ratio"] = (
                value is not None and value >= args.min_cache_hit_ratio
            )
        if args.min_tps_retention is not None:
            value = row.get("per_stream_b1_retention")
            checks["min_tps_retention"] = (
                value is not None and value >= args.min_tps_retention
            )
        if args.min_aggregate_scaling is not None:
            value = row.get("aggregate_b1_scaling")
            checks["min_aggregate_scaling"] = (
                value is not None and value >= args.min_aggregate_scaling
            )
        if args.max_finish_skew is not None:
            checks["max_finish_skew"] = row["finish_skew_s"] <= args.max_finish_skew
        for gate, passed in checks.items():
            if not passed:
                failures.append({"round": label, "gate": gate})

    cancellation = report.get("cancellation")
    if cancellation:
        stream_rows = cancellation["streams"]
        statuses = [row["status"] for row in stream_rows]
        if statuses.count("cancelled_by_client") != 1:
            failures.append({"round": "cancellation", "gate": "one_target_cancelled"})
        if statuses.count("completed") != len(statuses) - 1:
            failures.append({"round": "cancellation", "gate": "survivors_completed"})
        survivors = [row for row in stream_rows if row["status"] == "completed"]
        if not survivors or not all(row["marker_present"] for row in survivors):
            failures.append({"round": "cancellation", "gate": "survivor_isolation"})
    return failures


async def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    timeout = httpx.Timeout(
        connect=args.connect_timeout,
        read=args.request_timeout,
        write=args.request_timeout,
        pool=args.request_timeout,
    )
    limits = httpx.Limits(
        max_connections=max(16, max(args.streams) + 4),
        max_keepalive_connections=max(8, max(args.streams) + 2),
    )
    api_key = args.api_key or os.environ.get("OMLX_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    epoch = time.perf_counter()
    run_salt = uuid.uuid4().hex[:12]
    corpus = (
        Path(args.corpus_file).read_text(encoding="utf-8")
        if args.corpus_file
        else DEFAULT_UNIT
    )
    if not corpus:
        raise ValueError("corpus must not be empty")
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        headers=_headers(api_key),
        http2=False,
    ) as client:
        model = await _discover_model(client, args.endpoint, args.model)
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_unix": time.time(),
            "endpoint": _chat_url(args.endpoint),
            "model": model,
            "configuration": {
                "contexts": args.contexts,
                "streams": args.streams,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "prompt_mode": args.prompt_mode,
                "chat_template_kwargs": (
                    {"enable_thinking": False}
                    if args.prompt_mode == PREDICTABLE_CODE_MODE
                    else None
                ),
                "acceptance_reporting": "explicit_server_telemetry_only",
                "prime_cache": args.prime_cache,
                "prime_each_stream": args.prime_each_stream,
                "calibrate_contexts": args.calibrate_contexts,
                "mixed": args.mixed,
                "cancellation": args.cancellation,
            },
            "context_calibration": [],
            "cache_primes": [],
            "stream_primes": [],
            "matrix": [],
        }
        prefixes: dict[int, str] = {}
        for context in args.contexts:
            context_salt = (
                f"predictable-code-c{context}"
                if args.prompt_mode == PREDICTABLE_CODE_MODE
                else f"{run_salt}-c{context}"
            )
            if args.calibrate_contexts:
                prefix, calibration = await _calibrate_prefix(
                    client,
                    endpoint=args.endpoint,
                    model=model,
                    target_tokens=context,
                    run_salt=context_salt,
                    chars_per_token=args.chars_per_token,
                    corpus=corpus,
                    prompt_mode=args.prompt_mode,
                )
            else:
                prefix, repeats = _make_shared_prefix(
                    context,
                    run_salt=context_salt,
                    chars_per_token=args.chars_per_token,
                    corpus=corpus,
                    prompt_mode=args.prompt_mode,
                )
                calibration = {
                    "method": "character_estimate",
                    "target_tokens": context,
                    "repeats": repeats,
                    "prompt_mode": args.prompt_mode,
                }
            prefixes[context] = prefix
            report["context_calibration"].append(calibration)
            if args.prime_cache:
                prime = await _prime_cache(
                    client,
                    endpoint=args.endpoint,
                    model=model,
                    prefix=prefix,
                    context=context,
                    epoch=epoch,
                    temperature=args.temperature,
                    seed=args.seed,
                    prompt_mode=args.prompt_mode,
                )
                report["cache_primes"].append(prime)
            for count in args.streams:
                matrix_specs = _specs(
                    prefix=prefix,
                    context=context,
                    count=count,
                    max_tokens=args.max_tokens,
                    phase=f"matrix-b{count}",
                    prompt_mode=args.prompt_mode,
                )
                exact_prime = None
                if args.prime_each_stream:
                    exact_prime = await _prime_each_stream(
                        client,
                        endpoint=args.endpoint,
                        model=model,
                        specs=matrix_specs,
                        epoch=epoch,
                        label=f"exact-prime-c{context}-b{count}",
                        temperature=args.temperature,
                        seed=args.seed,
                    )
                    exact_prime["context_target"] = context
                    report["stream_primes"].append(exact_prime)
                row = await _run_round(
                    client,
                    endpoint=args.endpoint,
                    model=model,
                    specs=matrix_specs,
                    epoch=epoch,
                    label=f"context-{context}-b{count}",
                    temperature=args.temperature,
                    seed=args.seed,
                )
                row["context_target"] = context
                row["exact_prompt_prime"] = {
                    "enabled": args.prime_each_stream,
                    "mode": (
                        "serial_exact_prompt_one_token"
                        if args.prime_each_stream
                        else None
                    ),
                    "all_exact_prompts_primed": (
                        exact_prime["all_exact_prompts_primed"]
                        if exact_prime is not None
                        else None
                    ),
                }
                report["matrix"].append(row)
        _annotate_scaling(report["matrix"])

        if args.mixed:
            victim_context = args.mixed_victim_context
            prefill_context = args.mixed_prefill_context
            victim_prefix = prefixes.get(victim_context) or _make_shared_prefix(
                victim_context,
                run_salt=f"{run_salt}-mixed-victim",
                chars_per_token=args.chars_per_token,
                corpus=corpus,
                prompt_mode=args.prompt_mode,
            )[0]
            prefill_prefix = prefixes.get(prefill_context) or _make_shared_prefix(
                prefill_context,
                run_salt=f"{run_salt}-mixed-prefill",
                chars_per_token=args.chars_per_token,
                corpus=corpus,
                prompt_mode=args.prompt_mode,
            )[0]
            report["mixed"] = await _run_mixed_phase(
                client,
                endpoint=args.endpoint,
                model=model,
                victim_prefix=victim_prefix,
                prefill_prefix=prefill_prefix,
                victim_context=victim_context,
                prefill_context=prefill_context,
                prefillers=args.mixed_prefillers,
                max_tokens=args.mixed_victim_tokens,
                epoch=epoch,
                temperature=args.temperature,
                seed=args.seed,
                first_token_timeout=args.request_timeout,
                prompt_mode=args.prompt_mode,
            )

        if args.cancellation:
            context = args.cancel_context or args.contexts[0]
            prefix = prefixes.get(context) or _make_shared_prefix(
                context,
                run_salt=f"{run_salt}-cancel",
                chars_per_token=args.chars_per_token,
                corpus=corpus,
                prompt_mode=args.prompt_mode,
            )[0]
            report["cancellation"] = await _run_round(
                client,
                endpoint=args.endpoint,
                model=model,
                specs=_specs(
                    prefix=prefix,
                    context=context,
                    count=args.cancel_streams,
                    max_tokens=args.max_tokens,
                    phase="cancel",
                    cancel_first_after_chunks=args.cancel_after_chunks,
                    prompt_mode=args.prompt_mode,
                ),
                epoch=epoch,
                label=f"cancellation-c{context}-b{args.cancel_streams}",
                temperature=args.temperature,
                seed=args.seed,
            )

    report["elapsed_s"] = _round(time.perf_counter() - epoch)
    report["gate_failures"] = _evaluate_gates(report, args)
    report["gates_passed"] = not report["gate_failures"]
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", help="Defaults to OMLX_API_KEY/OPENAI_API_KEY")
    parser.add_argument("--model", help="Defaults to the first /v1/models entry")
    parser.add_argument(
        "--prompt-mode",
        choices=("default", PREDICTABLE_CODE_MODE),
        default="default",
        help=(
            "default preserves the cache-isolation prose task; predictable-code "
            "uses deterministic output-only numbered Python continuation and "
            "disables thinking for high-acceptance B1 measurement"
        ),
    )
    parser.add_argument("--contexts", type=_parse_int_list, default=[20_000])
    parser.add_argument("--streams", type=_parse_int_list, default=[1, 2, 4, 6])
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--chars-per-token", type=float, default=3.4)
    parser.add_argument("--corpus-file", type=Path)
    parser.add_argument("--calibrate-contexts", action="store_true")
    parser.add_argument(
        "--prime-cache", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--prime-each-stream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Before each measured B1/B2/B4/B6 round, serially run every exact "
            "prompt/marker to one token so all unique prompts can restore cache"
        ),
    )
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--mixed-victim-context", type=_parse_scaled_int, default=2_000)
    parser.add_argument(
        "--mixed-prefill-context", type=_parse_scaled_int, default=50_000
    )
    parser.add_argument("--mixed-prefillers", type=int, default=1)
    parser.add_argument("--mixed-victim-tokens", type=int, default=1_000)
    parser.add_argument("--cancellation", action="store_true")
    parser.add_argument("--cancel-context", type=_parse_scaled_int)
    parser.add_argument("--cancel-streams", type=int, default=4)
    parser.add_argument("--cancel-after-chunks", type=int, default=8)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=1_800.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-markers", action="store_true")
    parser.add_argument("--min-cache-hit-ratio", type=float)
    parser.add_argument("--min-tps-retention", type=float)
    parser.add_argument("--min-aggregate-scaling", type=float)
    parser.add_argument("--max-finish-skew", type=float)
    parser.add_argument("--self-test", action="store_true")
    return parser


def _self_test() -> None:
    assert _parse_scaled_int("20k") == 20_000
    assert _parse_int_list("1,2,4,6") == [1, 2, 4, 6]
    assert _chat_url("http://localhost:8000/v1") == (
        "http://localhost:8000/v1/chat/completions"
    )
    assert _v1_url("http://localhost:8000/v1", "models") == (
        "http://localhost:8000/v1/models"
    )
    event = _event_from_sse_line(
        'data: {"id":"x","choices":[{"delta":{"content":"hello"}}]}'
    )
    assert event is not None and _delta_payload(event) == "hello"
    assert _event_from_sse_line("data: [DONE]") is None
    prefix, repeats = _make_shared_prefix(
        1_000, run_salt="test", chars_per_token=3.4, corpus=DEFAULT_UNIT
    )
    assert repeats > 0 and "test" in prefix
    marker = "ISOLATION-SELFTEST"
    assert marker in _prompt(prefix, marker)

    predictable_a, predictable_repeats = _make_shared_prefix(
        1_000,
        run_salt="predictable-code-c1000",
        chars_per_token=3.4,
        corpus=DEFAULT_UNIT,
        prompt_mode=PREDICTABLE_CODE_MODE,
    )
    predictable_b, repeats_b = _make_shared_prefix(
        1_000,
        run_salt="predictable-code-c1000",
        chars_per_token=3.4,
        corpus=DEFAULT_UNIT,
        prompt_mode=PREDICTABLE_CODE_MODE,
    )
    assert predictable_a == predictable_b
    assert predictable_repeats == repeats_b
    assert "def qwen4_transform_000000" in predictable_a
    assert (
        f"def qwen4_transform_{predictable_repeats - 1:06d}"
        in predictable_a
    )
    predictable_prompt = _prompt(
        predictable_a,
        "ISOLATION-PREDICTABLE",
        prompt_mode=PREDICTABLE_CODE_MODE,
    )
    assert "Output Python code only" in predictable_prompt
    assert (
        f"qwen4_transform_{predictable_repeats:06d}" in predictable_prompt
    )
    predictable_specs_a = _specs(
        prefix=predictable_a,
        context=1_000,
        count=1,
        max_tokens=500,
        phase="matrix-b1",
        prompt_mode=PREDICTABLE_CODE_MODE,
    )
    predictable_specs_b = _specs(
        prefix=predictable_a,
        context=1_000,
        count=1,
        max_tokens=500,
        phase="matrix-b1",
        prompt_mode=PREDICTABLE_CODE_MODE,
    )
    assert predictable_specs_a == predictable_specs_b
    assert predictable_specs_a[0].prompt_mode == PREDICTABLE_CODE_MODE

    raw_acceptance = _server_acceptance_telemetry(
        {"metrics": {"accepted_draft_tokens": 7, "drafted_tokens": 10}}
    )
    assert raw_acceptance == [
        {
            "source": "metrics",
            "values": {"accepted_draft_tokens": 7, "drafted_tokens": 10},
            "derived": False,
        }
    ]
    assert "acceptance_ratio" not in raw_acceptance[0]["values"]

    now = time.perf_counter()
    one = StreamResult(
        label="s0",
        marker=marker,
        status="completed",
        response_id="r0",
        finish_reason="length",
        started_at=now,
        first_token_at=now + 1.0,
        finished_at=now + 3.0,
        content_event_times=[now + 1.0, now + 2.0, now + 3.0],
        content_events=3,
        output=marker + " output",
        usage={
            "prompt_tokens": 1_000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 800},
            "generation_tokens_per_second": 50.0,
        },
    )
    row = _round_report(
        label="selftest",
        results=[one],
        epoch=now,
        round_started=now,
        round_finished=now + 3.0,
    )
    assert row["completed_streams"] == 1
    assert row["median_cache_hit_ratio"] == 0.8
    assert row["all_markers_present"] is True
    assert row["aggregate_decode_tps"] == 50.0
    assert row["response_ids_unique"] is True
    assert row["cache_coverage"]["aggregate_cache_hit_ratio"] == 0.8
    assert row["max_observed_active_decode_batch"] == 1

    marker_two = "ISOLATION-SELFTEST-TWO"
    two = StreamResult(
        label="s1",
        marker=marker_two,
        status="completed",
        response_id="r1",
        finish_reason="length",
        started_at=now,
        first_token_at=now + 1.5,
        finished_at=now + 2.5,
        content_event_times=[now + 1.5, now + 2.5],
        content_events=2,
        output=marker_two + " output",
        usage={
            "prompt_tokens": 1_000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 900},
            "generation_tokens_per_second": 50.0,
        },
    )
    overlap = _round_report(
        label="overlap-selftest",
        results=[one, two],
        epoch=now,
        round_started=now,
        round_finished=now + 3.0,
    )
    assert overlap["max_observed_active_decode_batch"] == 2
    assert overlap["cache_coverage"]["aggregate_cache_hit_ratio"] == 0.85
    assert overlap["cache_coverage"]["streams_with_cache_hit"] == 2


async def _async_self_test() -> None:
    marker = "ISOLATION-ASYNC-SELFTEST"
    seen_requests: list[dict[str, Any]] = []
    body = "\n".join(
        (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-selftest",
                    "choices": [{"delta": {"content": marker}}],
                }
            ),
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-selftest",
                    "choices": [{"delta": {"content": " output"}}],
                }
            ),
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-selftest",
                    "choices": [{"delta": {}, "finish_reason": "length"}],
                }
            ),
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-selftest",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 64},
                        "generation_tokens_per_second": 42.0,
                    },
                    "metrics": {
                        "mtp_acceptance_ratio": 0.875,
                        "accepted_draft_tokens": 7,
                        "drafted_tokens": 8,
                    },
                }
            ),
            "data: [DONE]",
            "",
        )
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        seen_requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gate = asyncio.Event()
        gate.set()
        result = await _stream_request(
            client,
            endpoint="http://selftest/v1",
            model="qwen4-selftest",
            spec=StreamSpec(
                label="selftest",
                marker=marker,
                prompt="hello",
                max_tokens=2,
            ),
            start_gate=gate,
        )
        assert result.status == "completed"
        assert result.response_id == "chatcmpl-selftest"
        assert result.output == marker + " output"
        assert result.usage["completion_tokens"] == 2
        assert result.acceptance_telemetry == [
            {
                "source": "metrics",
                "values": {
                    "mtp_acceptance_ratio": 0.875,
                    "accepted_draft_tokens": 7,
                    "drafted_tokens": 8,
                },
                "derived": False,
            }
        ]
        assert "chat_template_kwargs" not in seen_requests[0]

        cancelled = await _stream_request(
            client,
            endpoint="http://selftest/v1",
            model="qwen4-selftest",
            spec=StreamSpec(
                label="cancel-selftest",
                marker=marker,
                prompt="hello",
                max_tokens=2,
                cancel_after_chunks=1,
            ),
            start_gate=gate,
        )
        assert cancelled.status == "cancelled_by_client"
        assert cancelled.content_events == 1

        primes = await _prime_each_stream(
            client,
            endpoint="http://selftest/v1",
            model="qwen4-selftest",
            specs=[
                StreamSpec(
                    label="exact-0",
                    marker=marker,
                    prompt="exact prompt zero",
                    max_tokens=500,
                    prompt_mode=PREDICTABLE_CODE_MODE,
                ),
                StreamSpec(
                    label="exact-1",
                    marker=marker,
                    prompt="exact prompt one",
                    max_tokens=500,
                    prompt_mode=PREDICTABLE_CODE_MODE,
                ),
            ],
            epoch=time.perf_counter(),
            label="exact-prime-selftest",
            temperature=0.0,
            seed=1234,
        )
        assert primes["all_exact_prompts_primed"] is True
        assert primes["stream_count"] == 2
        assert [request["max_tokens"] for request in seen_requests[-2:]] == [1, 1]
        assert [
            request["messages"][0]["content"] for request in seen_requests[-2:]
        ] == ["exact prompt zero", "exact prompt one"]
        assert all(
            request["chat_template_kwargs"] == {"enable_thinking": False}
            for request in seen_requests[-2:]
        )


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        _self_test()
        asyncio.run(_async_self_test())
        print("bench_qwen4_concurrency: self-test passed")
        return 0
    if any(value not in {1, 2, 4, 6} for value in args.streams):
        raise SystemExit("--streams must contain only 1,2,4,6")
    if args.prompt_mode == PREDICTABLE_CODE_MODE and args.streams != [1]:
        raise SystemExit(
            "--prompt-mode predictable-code is a B1 gate; pass --streams 1"
        )
    if args.prompt_mode == PREDICTABLE_CODE_MODE and args.corpus_file:
        raise SystemExit(
            "--corpus-file cannot be combined with --prompt-mode predictable-code"
        )
    if any(value <= 0 for value in (args.max_tokens, args.cancel_after_chunks)):
        raise SystemExit("token/chunk counts must be positive")
    report = asyncio.run(_benchmark(args))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if args.strict and not report["gates_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
