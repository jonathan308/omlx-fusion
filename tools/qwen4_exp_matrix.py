#!/usr/bin/env python3
"""Exact-token cold/warm Qwen4-Exp benchmark matrix against a live server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def post_json(url: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def varied_prompt(tokenizer, target: int, corpus: str) -> str:
    if corpus == "synthetic":
        text = "".join(
            f"\nSection {index}: A distributed inference scheduler maps request "
            f"{index} to expert {index % 512}, preserves cache offset {index * 7}, "
            f"and validates tensor shape {index % 97}. "
            for index in range(max(128, target // 20 + 64))
        )
    else:
        corpus_path = (
            Path(__file__).resolve().parents[1]
            / "omlx"
            / "admin"
            / "bench_corpora"
            / f"{corpus}.txt"
        )
        source = corpus_path.read_text()
        text = source
        while len(tokenizer.encode(text, add_special_tokens=False)) < target:
            text += "\n\n" + source
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < target:
        raise RuntimeError(f"corpus produced {len(tokens)} tokens, need {target}")
    selected = tokens[:target]
    prompt = tokenizer.decode(
        selected,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    roundtrip = tokenizer.encode(prompt, add_special_tokens=False)
    if roundtrip != selected:
        raise RuntimeError(
            f"tokenizer round-trip changed pp{target} to {len(roundtrip)} tokens"
        )
    return prompt


def stream_completion(base: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base}/v1/completions",
        data=json.dumps({**payload, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
    return {"wall_s": time.perf_counter() - started, "usage": usage}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-Fusion-Q8-Compute")
    parser.add_argument(
        "--tokenizer-path",
        default=(
            "/Users/jonathanspangler/.lmstudio/models/Qwen/"
            "Qwen3.8-Flash-Next-Fusion-Q8-Compute"
        ),
    )
    parser.add_argument("--targets", type=int, nargs="+", default=[2048, 10000, 20000])
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument(
        "--corpus",
        choices=("synthetic", "code_python", "code_mixed", "novel_en"),
        default="synthetic",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    for target in args.targets:
        prompt = varied_prompt(tokenizer, target, args.corpus)
        cleared = {
            "ssd": post_json(f"{args.url}/admin/api/ssd-cache/clear"),
            "hot": post_json(f"{args.url}/admin/api/hot-cache/clear"),
        }
        for state in ("cold", "warm"):
            result = stream_completion(
                args.url,
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "stream_options": {"include_usage": True},
                },
            )
            print(
                json.dumps(
                    {
                        "target": target,
                        "state": state,
                        "clear": cleared if state == "cold" else None,
                        **result,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
