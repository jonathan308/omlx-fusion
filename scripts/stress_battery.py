#!/usr/bin/env python3
"""Pre-handover certification battery for an oMLX server.

Adapted from ThunderMLX ``ops/stress_battery.sh`` (9-phase, sequential,
single-owner). Every phase is documented below with what it certifies and —
where the oMLX port deviates — why:

- P0 cache artifact (hard only with ``--cache-dir``, else SKIP). A long
  unique prompt must leave a new artifact in the SSD cache tier. oMLX's
  cache layout is an implementation detail, so the check is artifact-count
  growth within a settle window, not a path grep.
- P1 cache reuse (advisory). ThunderMLX restarted the whole cluster and
  verified an SSD restore beat cold prefill; an oMLX server is app/launchd
  managed and a script must never kill it, so the restart is dropped. The
  same long prompt is sent twice and warm-vs-cold TTFT is *reported*; the
  hard gate is that the repeat completes, because cache behavior is
  settings-dependent.
- P2 stop hammer (hard). oMLX's public cancel contract is the client
  disconnect — there is no public ``/v1/stop`` — so each cycle abandons a
  long stream mid-decode, then requires a probe request to answer promptly
  (the slot actually freed). The distributed lockstep-cancel admin path is
  covered by its own unit tests.
- P3 decode marathon (hard). A long generation must complete at or above
  ``--min-decode-tps`` — the wedge regime that essay soaks miss is P4's;
  this one certifies sustained decode.
- P4 agent cycles (hard). Drives ``agent_traffic_test.py``'s cycle
  in-process: short/tool/follow-up/analysis/rapid-fire turns with
  accumulating context and session rotation.
- P5 prefill + mid-prefill stop (hard). A large prompt's stream is abandoned
  during prefill and the slot must free; then a fresh large prompt must
  complete — the mid-prefill-cancel wedge check.
- P6 concurrency (hard). N small requests in flight at once must all answer
  correctly (oMLX's scheduler batches by design; ThunderMLX single-fight
  queueing is the thing being contrasted, not ported).
- P7 idle survival (hard). The server must answer /health for the whole
  idle window — an idle cluster once read as wedged and self-killed.
- P9 scorecard (advisory). /health and /api/status snapshots plus the
  battery's own measurements.

Dropped outright: P8 (rank-1 wired memory over ssh) — ThunderMLX-specific;
oMLX's per-host memory view is the cluster tab's peer telemetry, and these
scripts authenticate with the API key only. The port also never scrapes log
files (ThunderMLX anchored greps to line counts for stale-log immunity);
oMLX phases read API-visible state instead.

Exit codes: 90 when the API is not up at battery start; otherwise the number
of failed hard phases (0 = certified). ``--dry-run`` prints the phase plan
without contacting the server.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

EXIT_API_DOWN = 90

# SKIP is a verdict, not a failure: the phase did not apply to this server.
SKIP = "skip"


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


def filler_prompt(records: int) -> str:
    """A deterministic large prompt with a unique archive id (cache poison guard)."""

    body = "".join(
        f"Record {i:05d}: subsystem nominal, checksum verified, latency in budget. "
        for i in range(records)
    )
    return f"Archive {uuid.uuid4()}: {body} Summarize in one sentence."


class HttpxTransport:
    """The battery's server access: chat (stream or not), abandon, GET."""

    def __init__(self, base_url: str, key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.key = key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def get(self, path: str, *, timeout: float = 5.0) -> dict[str, Any]:
        response = httpx.get(
            self.base_url + path, headers=self._headers(), timeout=timeout
        )
        response.raise_for_status()
        return response.json()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float = 0.2,
        tools: list[dict[str, Any]] | None = None,
        read_timeout: float = 600.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        body: dict[str, Any] = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        response = httpx.post(
            self.base_url + "/v1/chat/completions",
            json=body,
            headers=self._headers(),
            timeout=httpx.Timeout(15.0, read=read_timeout),
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        return {
            "text": (choices[0].get("message") or {}).get("content", "")
            if choices
            else "",
            "seconds": round(time.monotonic() - started, 2),
            "usage": body.get("usage") or {},
        }

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float = 0.2,
        keep_chunks: int | None = None,
        read_timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Stream a chat; ``keep_chunks`` abandons the stream early (a client
        disconnect — oMLX's cancel contract) after that many content chunks."""

        started = time.monotonic()
        first_chunk_at: float | None = None
        chunks = 0
        text: list[str] = []
        abandoned = False
        with httpx.stream(
            "POST",
            self.base_url + "/v1/chat/completions",
            json={
                "model": "default",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
            headers=self._headers(),
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
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        chunks += 1
                        text.append(delta["content"])
                if keep_chunks is not None and chunks >= keep_chunks:
                    abandoned = True
                    break
            # Leaving the context manager closes the connection: the server
            # observes a client disconnect and cancels the request.
        return {
            "text": "".join(text),
            "chunks": chunks,
            "ttft": (
                round(first_chunk_at - started, 2) if first_chunk_at is not None else None
            ),
            "seconds": round(time.monotonic() - started, 2),
            "abandoned": abandoned,
        }


def count_cache_artifacts(cache_dir: Path, *, cap: int = 200_000) -> int:
    """Bounded artifact count — the check is growth, not layout."""

    count = 0
    stack = [cache_dir]
    while stack and count <= cap:
        entry = stack.pop()
        try:
            if entry.is_dir():
                stack.extend(entry.iterdir())
            else:
                count += 1
        except OSError:
            continue
    return count


class Battery:
    """Runs the phases in order against one transport; collects a scorecard."""

    def __init__(
        self,
        transport: Any,
        args: argparse.Namespace,
        *,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> None:
        self.transport = transport
        self.args = args
        self.sleep = sleep
        self.monotonic = monotonic
        self.results: list[dict[str, Any]] = []

    def _verdict(self, phase: str, ok: bool | str, detail: str) -> bool | str:
        status = SKIP if ok == SKIP else ("pass" if ok else "fail")
        self.results.append({"phase": phase, "status": status, "detail": detail})
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[status]
        print(f"[{phase}] {mark}: {detail}", flush=True)
        return ok

    def _user(self, text: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": text}]

    def p0_cache_artifact(self) -> None:
        cache_dir = self.args.cache_dir
        if cache_dir is None:
            self._verdict(
                "P0", SKIP, "no --cache-dir given; SSD-tier artifact check skipped"
            )
            return
        before = count_cache_artifacts(cache_dir)
        self.transport.chat(
            self._user(filler_prompt(1400) + " Reply SEED-1."), max_tokens=12
        )
        after = before
        deadline = self.monotonic() + self.args.cache_settle_seconds
        while self.monotonic() < deadline:
            after = count_cache_artifacts(cache_dir)
            if after > before:
                break
            self.sleep(1.0)
        self._verdict(
            "P0",
            after > before,
            f"cache artifacts {before} -> {after} within "
            f"{self.args.cache_settle_seconds}s settle window",
        )

    def p1_cache_reuse(self) -> None:
        prompt = self._user(filler_prompt(1400))
        cold = self.transport.stream(prompt, max_tokens=12)
        warm = self.transport.stream(prompt, max_tokens=12)
        detail = f"cold ttft={cold['ttft']}s warm ttft={warm['ttft']}s"
        self._verdict("P1", bool(warm["text"]), detail + " (repeat completed)")

    def p2_stop_hammer(self) -> None:
        failures = 0
        for cycle in range(self.args.hammer_cycles):
            self.transport.stream(
                self._user(
                    "Write an endless numbered story."
                    f" run {cycle} {uuid.uuid4()}"
                ),
                max_tokens=4000,
                keep_chunks=4 + (cycle % 3),
            )
            try:
                probe = self.transport.chat(
                    self._user(f"Reply with the word FREE-{cycle} only."),
                    max_tokens=8,
                    read_timeout=self.args.slot_free_seconds,
                )
                if f"FREE-{cycle}" not in probe["text"]:
                    failures += 1
            except Exception:  # noqa: BLE001 - any probe failure is a stuck slot
                failures += 1
        self._verdict(
            "P2",
            failures == 0,
            f"{self.args.hammer_cycles} disconnect-stop cycles, "
            f"{failures} slot(s) still busy after {self.args.slot_free_seconds}s",
        )

    def p3_decode_marathon(self) -> None:
        result = self.transport.stream(
            self._user(
                "Write an endless glossary of invented terms with definitions."
                f" Never stop. {uuid.uuid4()}"
            ),
            max_tokens=self.args.decode_tokens,
            read_timeout=max(600.0, self.args.decode_tokens * 2.0),
        )
        tps = (
            result["chunks"] / result["seconds"]
            if result["seconds"] > 0
            else 0.0
        )
        self._verdict(
            "P3",
            result["chunks"] >= self.args.decode_tokens * 0.9
            and tps >= self.args.min_decode_tps,
            f"{result['chunks']} chunks in {result['seconds']}s "
            f"({tps:.1f} chunks/s; floor {self.args.min_decode_tps})",
        )

    def p4_agent_cycles(self) -> None:
        # Imported here so --dry-run and unrelated phases never require it,
        # and resolved against this script's own directory so the battery is
        # runnable from any cwd (and importable from tests).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from agent_traffic_test import CYCLE_STEPS, AgentTrafficSession

        def caller(messages: Any, max_tokens: float, temperature: float, tools: Any) -> dict[str, Any]:
            result = self.transport.chat(
                list(messages),
                max_tokens=int(max_tokens),
                temperature=temperature,
                tools=tools,
            )
            return {
                "seconds": result["seconds"],
                "chunks": 1,
                "text": result["text"],
                "finish": "stop",
                "tool_chunks": 0,
            }

        import io

        log = io.StringIO()
        session = AgentTrafficSession(session_turns=self.args.session_turns)
        clean = 0
        for cycle in range(self.args.agent_cycles):
            task = f"Agent battery cycle {cycle}."
            ok = all(
                session.turn(
                    caller,
                    log,
                    kind=kind,
                    user_msg=task + msg,
                    max_tokens=max_tokens,
                    tools=tools,
                    temperature=temp,
                )
                for kind, msg, max_tokens, tools, temp in CYCLE_STEPS
            )
            if not ok:
                break
            clean += 1
            self.sleep(self.args.think_gap)
        self._verdict(
            "P4",
            clean == self.args.agent_cycles,
            f"{clean}/{self.args.agent_cycles} agent cycles clean "
            f"({session.turn_no} turns)",
        )

    def p5_prefill_stop(self) -> None:
        self.transport.stream(
            self._user(filler_prompt(self.args.prefill_records)),
            max_tokens=80,
            keep_chunks=1,
            read_timeout=300.0,
        )
        freed = True
        try:
            self.transport.chat(
                self._user("Reply with the word SLOT only."),
                max_tokens=8,
                read_timeout=self.args.slot_free_seconds,
            )
        except Exception:  # noqa: BLE001
            freed = False
        if not freed:
            self._verdict(
                "P5", False, "slot still busy after mid-prefill disconnect"
            )
            return
        follow = self.transport.chat(
            self._user(filler_prompt(self.args.prefill_records)),
            max_tokens=80,
            read_timeout=600.0,
        )
        self._verdict(
            "P5",
            bool(follow["text"]),
            "mid-prefill stop freed the slot; fresh large prefill completed",
        )

    def p6_concurrency(self) -> None:
        outcomes: list[str] = [""] * self.args.concurrent
        lock = threading.Lock()

        def ask(index: int) -> None:
            try:
                result = self.transport.chat(
                    self._user(f"Reply with the word QUEUE-{index} only."),
                    max_tokens=8,
                    read_timeout=240.0,
                )
                outcome = result["text"]
            except Exception:  # noqa: BLE001
                outcome = ""
            with lock:
                outcomes[index] = outcome

        threads = [
            threading.Thread(target=ask, args=(index,), daemon=True)
            for index in range(self.args.concurrent)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        missed = [
            index
            for index, outcome in enumerate(outcomes)
            if f"QUEUE-{index}" not in outcome
        ]
        self._verdict(
            "P6",
            not missed,
            f"{self.args.concurrent - len(missed)}/{self.args.concurrent} "
            f"concurrent requests answered correctly",
        )

    def p7_idle_survival(self) -> None:
        deadline = self.monotonic() + self.args.idle_seconds
        died_at: float | None = None
        while self.monotonic() < deadline:
            try:
                self.transport.get("/health", timeout=5.0)
            except Exception:  # noqa: BLE001 - any failure is a dead server
                died_at = self.args.idle_seconds - (deadline - self.monotonic())
                break
            self.sleep(min(6.0, max(0.0, deadline - self.monotonic())))
        self._verdict(
            "P7",
            died_at is None,
            (
                f"survived {self.args.idle_seconds}s idle"
                if died_at is None
                else f"/health failed after ~{died_at:.0f}s idle"
            ),
        )

    def p9_scorecard(self) -> None:
        lines = []
        try:
            health = self.transport.get("/health", timeout=5.0)
            pool = health.get("engine_pool") or {}
            lines.append(
                f"health={health.get('status')} loaded={pool.get('loaded_count')}"
                f"/{pool.get('model_count')}"
            )
        except Exception as exc:  # noqa: BLE001
            lines.append(f"health unavailable: {type(exc).__name__}")
        try:
            status = self.transport.get("/api/status", timeout=5.0)
            metrics = status.get("metrics") or {}
            if metrics:
                lines.append(f"metrics={json.dumps(metrics)[:200]}")
        except Exception:  # noqa: BLE001 - /api/status is best-effort garnish
            pass
        self._verdict("P9", True, "; ".join(lines) or "no server stats")

    def run(self) -> int:
        try:
            self.transport.get("/health", timeout=5.0)
        except Exception:  # noqa: BLE001
            print(
                f"ABORT: API not up at battery start ({self.transport.base_url})",
                flush=True,
            )
            return EXIT_API_DOWN
        self.p0_cache_artifact()
        self.p1_cache_reuse()
        self.p2_stop_hammer()
        self.p3_decode_marathon()
        self.p4_agent_cycles()
        self.p5_prefill_stop()
        self.p6_concurrency()
        self.p7_idle_survival()
        self.p9_scorecard()
        failures = sum(1 for r in self.results if r["status"] == "fail")
        passed = sum(1 for r in self.results if r["status"] == "pass")
        skipped = sum(1 for r in self.results if r["status"] == SKIP)
        print(
            f"=========== RESULT: {passed} pass / {failures} fail "
            f"/ {skipped} skip ===========",
            flush=True,
        )
        print(
            "STRESS-CERTIFIED — ready for handover"
            if failures == 0
            else "NOT ready — fix and re-run",
            flush=True,
        )
        return failures


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("~/.omlx/settings.json").expanduser(),
        help="settings.json to read the API key from when OMLX_API_KEY is unset",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="SSD cache directory for the P0 artifact check (else P0 skips)",
    )
    parser.add_argument("--cache-settle-seconds", type=float, default=10.0)
    parser.add_argument("--hammer-cycles", type=int, default=12)
    parser.add_argument("--slot-free-seconds", type=float, default=15.0)
    parser.add_argument("--decode-tokens", type=int, default=10_000)
    parser.add_argument("--min-decode-tps", type=float, default=20.0)
    parser.add_argument("--agent-cycles", type=int, default=10)
    parser.add_argument("--session-turns", type=int, default=12)
    parser.add_argument("--think-gap", type=float, default=3.0)
    parser.add_argument("--prefill-records", type=int, default=2000)
    parser.add_argument("--concurrent", type=int, default=3)
    parser.add_argument("--idle-seconds", type=float, default=180.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the phase plan and exit without contacting the server",
    )
    args = parser.parse_args(argv)
    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.expanduser()
    for name in ("hammer_cycles", "decode_tokens", "agent_cycles", "concurrent"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    for name in ("slot_free_seconds", "idle_seconds", "think_gap"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def main(argv: list[str] | None = None, *, transport: Any = None, sleep: Any = time.sleep) -> int:
    args = _arguments(argv)
    if args.dry_run:
        phases = [
            ("P0", "SSD cache artifact growth (hard with --cache-dir, else skip)"),
            ("P1", "cache reuse: repeat long prompt completes; warm/cold TTFT reported"),
            ("P2", f"stop hammer: {args.hammer_cycles} disconnect-stop cycles + slot probe"),
            ("P3", f"decode marathon: {args.decode_tokens} tokens at >= {args.min_decode_tps} t/s"),
            ("P4", f"agent regime: {args.agent_cycles} cycles via agent_traffic_test steps"),
            ("P5", f"{args.prefill_records}-record prefill, mid-prefill stop, fresh completion"),
            ("P6", f"concurrency: {args.concurrent} simultaneous requests all answered"),
            ("P7", f"idle survival: /health for {args.idle_seconds}s"),
            ("P9", "scorecard from /health and /api/status"),
        ]
        print(f"STRESS BATTERY plan against {args.base_url}:")
        for name, description in phases:
            print(f"  {name}: {description}")
        return 0
    if transport is None:
        transport = HttpxTransport(args.base_url, api_key(args.settings))
    return Battery(transport, args, sleep=sleep).run()


if __name__ == "__main__":
    sys.exit(main())
