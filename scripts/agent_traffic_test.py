#!/usr/bin/env python3
"""Agent-traffic wedge test against an oMLX endpoint — the OTHER decode regime.

Long-essay soaks do not cover agent/coding traffic: many short decodes,
tool-call turns, growing multi-turn context with cache reuse, and bursty
cadence with think-time gaps. (ThunderMLX's 2026-07-05 wedge hit a ~127-token
plain decode; this is the regime that certifies agent serving.) The script
simulates a Codex/Claude-Code-style session against the OpenAI-compatible
API.

Cycle (per ``--cycles``): short answer -> tool turn (tools advertised) ->
follow-up on grown context -> medium analysis -> rapid-fire trio of short
turns. Context accumulates within a session; sessions rotate every
``--session-turns`` to exercise cache eviction/reuse.

Contract: exit 0 when every cycle is clean, exit 75 (EX_TEMPFAIL) on the
first wedge or stream loss; one JSONL record per turn; one console line per
cycle. ``--dry-run`` prints the plan without contacting the server.

Adapted from ThunderMLX ``ops/agent_traffic_test.py``: httpx instead of
requests, oMLX API-key auth (``OMLX_API_KEY`` or ~/.omlx/settings.json), and
an explicit ``--model`` (oMLX serves many models; none is assumed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

READ_TIMEOUT_S = 240.0
EXIT_WEDGE = 75

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a shell command and return stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "the command"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

SEED_TASKS = [
    "We're debugging a Python web service that intermittently returns 502s behind nginx.",
    "We're profiling a Rust CLI that parses 10GB of JSONL too slowly.",
    "We're adding dark mode to a React dashboard with CSS variables.",
    "We're migrating a Postgres schema without downtime using triggers.",
    "We're fixing a flaky pytest suite that fails only in CI.",
    "We're optimizing an MLX training loop that underutilizes the GPU.",
]

# One cycle is the agent regime in miniature: a short opener, a tool turn, a
# follow-up on the grown context, one longer analysis, then a rapid trio.
CYCLE_STEPS = [
    ("short", " Where do we start? Two sentences max.", 200, None, 0.2),
    ("tool", " Inspect the relevant config/entrypoint first. Use a tool.", 512, TOOLS, 0.2),
    (
        "followup",
        " Tool output: '(3 matching files found, largest is 48KB, modified today)'. What next?",
        300,
        TOOLS,
        0.2,
    ),
    (
        "analysis",
        " Write a detailed step-by-step plan with commands and expected pitfalls.",
        1200,
        None,
        0.4,
    ),
    ("rapid1", " Shorter version, 3 bullets.", 120, None, 0.2),
    ("rapid2", " Which step is riskiest?", 120, None, 0.2),
    ("rapid3", " One-line summary for the commit message.", 60, None, 0.2),
]


def api_key(settings_path: Path) -> str:
    """OMLX_API_KEY first, then the key the server itself persists."""

    import os

    environment_key = os.environ.get("OMLX_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        settings = json.loads(settings_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    return str(settings.get("auth", {}).get("api_key", "")).strip()


def stream_chat(
    *,
    base_url: str,
    key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None,
    read_timeout: float = READ_TIMEOUT_S,
) -> dict[str, Any]:
    """One streaming chat turn; returns timing/chunk/text/finish evidence."""

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    started = time.monotonic()
    chunks = 0
    tool_chunks = 0
    text: list[str] = []
    finish = None
    with httpx.stream(
        "POST",
        base_url.rstrip("/") + "/v1/chat/completions",
        json=body,
        headers=headers,
        timeout=httpx.Timeout(15.0, read=read_timeout),
    ) as response:
        response.raise_for_status()
        for raw in response.iter_lines():
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    chunks += 1
                    text.append(delta["content"])
                if delta.get("tool_calls"):
                    tool_chunks += 1
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    return {
        "seconds": round(time.monotonic() - started, 1),
        "chunks": chunks,
        "text": "".join(text),
        "finish": finish,
        "tool_chunks": tool_chunks,
    }


class AgentTrafficSession:
    """Accumulating multi-turn context with periodic session rotation."""

    def __init__(self, *, session_turns: int) -> None:
        self._session_turns = max(1, session_turns)
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a terse, expert coding assistant."}
        ]
        self._turns_in_session = 0
        self.turn_no = 0

    @property
    def context_size(self) -> int:
        return len(self._messages)

    def turn(
        self,
        caller: Any,
        log: Any,
        *,
        kind: str,
        user_msg: str,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        temperature: float,
    ) -> bool:
        self.turn_no += 1
        self._messages.append({"role": "user", "content": user_msg})
        record = {
            "turn": self.turn_no,
            "kind": kind,
            "ctx_msgs": len(self._messages),
            "start": datetime.now().isoformat(),
        }
        try:
            result = caller(self._messages, max_tokens, temperature, tools)
        except httpx.ReadTimeout:
            record.update(status="wedge", error=f"no tokens for {READ_TIMEOUT_S}s")
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as exc:
            record.update(status="wedge", error=f"stream/connection lost: {exc}")
        except Exception as exc:  # noqa: BLE001 - any failure is evidence
            record.update(status="error", error=repr(exc))
        else:
            record.update(
                status="ok",
                **{k: v for k, v in result.items() if k != "text"},
            )
            reply = result["text"] or "(tool call)"
            self._messages.append({"role": "assistant", "content": reply[:2000]})
        log.write(json.dumps(record) + "\n")
        log.flush()
        self._turns_in_session += 1
        if self._turns_in_session >= self._session_turns:
            del self._messages[1:]
            self._turns_in_session = 0
        return record["status"] == "ok"


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="served model id to drive")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument(
        "--session-turns",
        type=int,
        default=12,
        help="rotate the conversation after this many turns",
    )
    parser.add_argument(
        "--think-gap",
        type=float,
        default=3.0,
        help="seconds between turns (agent think time)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSONL turn log (default: agent_traffic_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("~/.omlx/settings.json").expanduser(),
        help="settings.json to read the API key from when OMLX_API_KEY is unset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without contacting the server",
    )
    args = parser.parse_args(argv)
    if args.cycles < 1:
        parser.error("--cycles must be at least 1")
    if args.think_gap < 0:
        parser.error("--think-gap must be non-negative")
    return args


def main(argv: list[str] | None = None, *, caller: Any = None, sleep: Any = time.sleep) -> int:
    args = _arguments(argv)
    out_path = args.out or Path(
        f"agent_traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    if args.dry_run:
        print(
            f"AGENT-TRAFFIC plan: {args.cycles} cycles x {len(CYCLE_STEPS)} steps "
            f"against {args.base_url} model={args.model}; session rotates every "
            f"{args.session_turns} turns; think-gap {args.think_gap}s; log={out_path}"
        )
        for kind, msg, max_tokens, tools, temp in CYCLE_STEPS:
            print(
                f"  {kind}: max_tokens={max_tokens} temp={temp} "
                f"tools={'yes' if tools else 'no'} prompt={msg.strip()[:60]!r}"
            )
        return 0

    key = api_key(args.settings)
    if caller is None:

        def caller(messages: list[dict[str, Any]], max_tokens: float, temperature: float, tools: Any) -> dict[str, Any]:
            return stream_chat(
                base_url=args.base_url,
                key=key,
                model=args.model,
                messages=list(messages),
                max_tokens=int(max_tokens),
                temperature=temperature,
                tools=tools,
            )

    session = AgentTrafficSession(session_turns=args.session_turns)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"AGENT-TRAFFIC START {stamp} cycles={args.cycles} log={out_path}", flush=True)
    with open(out_path, "a") as log:
        for cycle in range(args.cycles):
            task = SEED_TASKS[cycle % len(SEED_TASKS)]
            for kind, msg, max_tokens, tools, temp in CYCLE_STEPS:
                ok = session.turn(
                    caller,
                    log,
                    kind=kind,
                    user_msg=task + msg,
                    max_tokens=max_tokens,
                    tools=tools,
                    temperature=temp,
                )
                if not ok:
                    print(
                        f"CYCLE {cycle + 1}/{args.cycles}: WEDGE/ERROR at "
                        f"'{kind}' turn {session.turn_no} — aborting (see {out_path})",
                        flush=True,
                    )
                    return EXIT_WEDGE
                sleep(args.think_gap)
            print(
                f"CYCLE {cycle + 1}/{args.cycles}: CLEAN "
                f"({len(CYCLE_STEPS)} turns, ctx {session.context_size} msgs)",
                flush=True,
            )
    print(
        f"AGENT-TRAFFIC COMPLETE: {args.cycles}/{args.cycles} cycles clean "
        f"({session.turn_no} turns).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
