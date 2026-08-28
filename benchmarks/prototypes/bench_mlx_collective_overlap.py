#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-rank proof for MLX compute/collective overlap.

This is an isolated scheduler probe, not a serving benchmark.  It uses the
same activation and hyper-connection shapes as one DS4 M=1024 TP reduction:

* one BF16 ``[1, 1024, 4096]`` all-sum on MLX's CPU communication stream;
* one independent FP32 ``[1, 1024, 4, 4] @ [1, 1024, 4, 4096]`` residual
  branch on Metal; and
* the original FP32 add followed by a BF16 cast.

``serial`` materializes the reduction before the residual branch. ``lazy``
puts both branches in one graph. ``async`` explicitly submits the reduction
before evaluating the independent GPU branch.  Every mode has identical
arithmetic and collective order on both ranks.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any


def _free_port_span(count: int = 2) -> int:
    for _ in range(64):
        sockets: list[socket.socket] = []
        try:
            first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(first)
            first.bind(("127.0.0.1", 0))
            port = int(first.getsockname()[1])
            if port + count - 1 > 65535:
                continue
            for offset in range(1, count):
                item = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(item)
                item.bind(("127.0.0.1", port + offset))
            return port
        except OSError:
            continue
        finally:
            for item in sockets:
                item.close()
    raise RuntimeError("could not reserve two loopback ports")


def _barrier(mx: Any, group: Any, stream: Any) -> None:
    value = mx.distributed.all_sum(
        mx.array(1, dtype=mx.int32), group=group, stream=stream
    )
    mx.eval(value)


def _forward(
    mx: Any,
    *,
    mode: str,
    payload: Any,
    residual: Any,
    comb: Any,
    post: Any,
    group: Any,
    stream: Any,
    nonce: int,
) -> Any:
    # A fresh primitive keeps each timing cycle independent while preserving
    # the exact BF16 activation shape used by DS4 TP.
    local = payload + mx.array(nonce & 1, dtype=payload.dtype)
    reduced = mx.distributed.all_sum(local, group=group, stream=stream)

    if mode == "serial":
        mx.eval(reduced)

    if mode == "async":
        # This is the only mode that deliberately fixes submission order:
        # local Metal producer -> CPU collective -> independent Metal branch.
        mx.async_eval(reduced)

    residual_term = mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    reduced_term = post[..., None] * reduced[:, :, None, :].astype(mx.float32)
    output = (reduced_term + residual_term).astype(payload.dtype)
    mx.eval(output)
    return output


def _worker(args: argparse.Namespace) -> int:
    import mlx.core as mx

    group = mx.distributed.init()
    rank = int(group.rank())
    if int(group.size()) != 2:
        raise RuntimeError("probe requires exactly two ranks")

    # JACCL and Ring both coerce collective streams to CPU.  A new stream is
    # used here to prove isolation from unrelated default-CPU work while one
    # stream still owns the entire collective sequence.
    comm_stream = mx.new_stream(mx.cpu) if args.dedicated_stream else mx.cpu
    mx.random.seed(args.seed)
    payload = mx.random.uniform(shape=(1, args.tokens, args.hidden)).astype(
        mx.bfloat16
    )
    residual = mx.random.uniform(
        shape=(1, args.tokens, args.hc_mult, args.hidden)
    ).astype(mx.bfloat16)
    comb = mx.random.uniform(
        shape=(1, args.tokens, args.hc_mult, args.hc_mult)
    ).astype(mx.float32)
    post = mx.random.uniform(shape=(1, args.tokens, args.hc_mult)).astype(
        mx.float32
    )
    mx.eval(payload, residual, comb, post)

    hashes: dict[str, tuple[float, float]] = {}
    timings: dict[str, list[float]] = {}
    modes = ("serial", "lazy", "async")
    nonce = 0
    for mode in modes:
        print(f"rank={rank} mode={mode} start", file=sys.stderr, flush=True)
        for _ in range(args.warmup):
            _barrier(mx, group, comm_stream)
            out = _forward(
                mx,
                mode=mode,
                payload=payload,
                residual=residual,
                comb=comb,
                post=post,
                group=group,
                stream=comm_stream,
                nonce=nonce,
            )
            nonce += 2
        samples: list[float] = []
        for _ in range(args.iterations):
            _barrier(mx, group, comm_stream)
            started = time.perf_counter()
            out = _forward(
                mx,
                mode=mode,
                payload=payload,
                residual=residual,
                comb=comb,
                post=post,
                group=group,
                stream=comm_stream,
                nonce=nonce,
            )
            samples.append((time.perf_counter() - started) * 1000.0)
            nonce += 2
        # Two stable FP32 summaries catch arithmetic or rank divergence
        # without moving the full 128 MiB HC result to Python.
        summary = (
            float(mx.sum(out.astype(mx.float32)).item()),
            float(mx.max(mx.abs(out.astype(mx.float32))).item()),
        )
        hashes[mode] = summary
        timings[mode] = samples
        print(f"rank={rank} mode={mode} done", file=sys.stderr, flush=True)

    print(
        json.dumps(
            {
                "type": "ds4_collective_overlap",
                "rank": rank,
                "dedicated_stream": bool(args.dedicated_stream),
                "timings_ms": timings,
                "summaries": hashes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _driver(args: argparse.Namespace) -> int:
    port = _free_port_span()
    launcher = (
        "from mlx._distributed_utils.launch import main; "
        "raise SystemExit(main() or 0)"
    )
    script = str(Path(__file__).resolve())
    command = [
        sys.executable,
        "-c",
        launcher,
        "--backend",
        "ring",
        "--hosts",
        "127.0.0.1",
        "--repeat-hosts",
        "2",
        "--starting-port",
        str(port),
        "--",
        sys.executable,
        script,
        "--worker",
        "--tokens",
        str(args.tokens),
        "--hidden",
        str(args.hidden),
        "--hc-mult",
        str(args.hc_mult),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
    ]
    if args.dedicated_stream:
        command.append("--dedicated-stream")
    env = os.environ.copy()
    env["MLX_METAL_FAST_SYNCH"] = "1" if args.fast_synch else "0"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5.0)
        raise RuntimeError(f"probe timed out: {stderr.strip() or stdout.strip()}")
    if process.returncode != 0:
        raise RuntimeError(
            f"launcher exited {process.returncode}: {stderr.strip() or stdout.strip()}"
        )
    records = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("type") == "ds4_collective_overlap":
            records.append(item)
    if len(records) != 2:
        raise RuntimeError(f"expected two rank records, got {len(records)}: {stdout}")
    print(json.dumps({"backend": "ring", "ranks": records}, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dedicated-stream", action="store_true", default=True)
    parser.add_argument("--default-stream", dest="dedicated_stream", action="store_false")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fast-synch", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.tokens < 1 or args.hidden < 1 or args.hc_mult < 1:
        raise SystemExit("shape dimensions must be positive")
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("warmup must be non-negative and iterations positive")
    return _worker(args) if args.worker else _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
