#!/usr/bin/env python3
"""Small, non-invasive streaming latency probe for a live oMLX server."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def post_json(url: str, payload: dict | None = None) -> tuple[dict, float]:
    body = json.dumps(payload or {}).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    return result, time.perf_counter() - start


def stream_completion(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps({**payload, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    token_times: list[float] = []
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            choices = event.get("choices") or ()
            if choices and choices[0].get("text"):
                token_times.append(time.perf_counter())
    end = time.perf_counter()
    intervals = [b - a for a, b in zip(token_times, token_times[1:])]
    return {
        "wall_s": end - start,
        "ttft_s": token_times[0] - start if token_times else None,
        "observed_chunks": len(token_times),
        "mean_chunk_interval_s": statistics.mean(intervals) if intervals else None,
        "median_chunk_interval_s": statistics.median(intervals) if intervals else None,
        "p95_chunk_interval_s": (
            sorted(intervals)[max(0, int(len(intervals) * 0.95) - 1)]
            if intervals
            else None
        ),
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-Fusion-Q8-Compute")
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--clear-ple", choices=("none", "hot", "ssd"), default="none")
    parser.add_argument("--keep-after", action="store_true")
    args = parser.parse_args()

    cleared = None
    if args.clear_ple != "none":
        endpoint = "hot-cache" if args.clear_ple == "hot" else "ssd-cache"
        cleared, _ = post_json(f"{args.url}/admin/api/{endpoint}/clear")

    payload = {
        "model": args.model,
        "prompt": " profiletoken" * args.repetitions,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream_options": {"include_usage": True},
    }
    result = stream_completion(args.url, payload)
    after = None
    if not args.keep_after:
        after, _ = post_json(f"{args.url}/admin/api/hot-cache/clear")
    print(
        json.dumps(
            {"clear_before": cleared, "result": result, "ple_after": after}, indent=2
        )
    )


if __name__ == "__main__":
    main()
