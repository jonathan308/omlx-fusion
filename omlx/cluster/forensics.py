# SPDX-License-Identifier: Apache-2.0
"""Capture live-wedge evidence *before* any fatal action erases it.

Every distributed wedge autopsy that ends in a kill loses the evidence with
the process: a watchdog force-exit beats a post-mortem every time, and the
one question that matters — was the rank spinning in a Metal fence wait, in a
JACCL completion-queue poll, or in a Python-level deadlock — is unanswerable
afterwards. The stall watchdog therefore fires this capture at the
still-spinning ranks well before its fatal window (ThunderMLX's
``live_wedge_capture.sh`` discipline, minus the privileged parts).

Deliberately unprivileged: no sudo, no ``spindump``. Everything here runs as
the rank's own user — ``sample`` on the local rank processes, a ``vm_stat``
wired snapshot, and copies of the rank markers and deployment registry. A
stall that resolves on its own costs one harmless bundle; a stall that does
not leaves enough evidence to reconstruct why.

Every step is best-effort and individually bounded: a capture that fails,
hangs, or partially writes must never block or kill the watchdog that called
it. The bundle manifest records per-step errors instead.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FORENSICS_DIR_ENV = "OMLX_CLUSTER_FORENSICS_DIR"
# ``sample`` is given this many seconds per process; the runner timeout adds
# headroom so a wedged target cannot park the watchdog thread.
_SAMPLE_SECONDS = 5
_SAMPLE_TIMEOUT_S = 30.0
_VM_STAT_TIMEOUT_S = 5.0
_PGREP_TIMEOUT_S = 5.0
# Bound the bundle: a rank marker is a few KiB, and a capture that floods the
# disk is its own incident.
_MAX_RANK_PIDS = 8
_MAX_MARKERS_COPIED = 16
_MAX_MARKER_BYTES = 1024 * 1024


def default_forensics_dir(state_dir: str | Path) -> Path:
    """The bundle root: a sibling of the rank runtime-marker directory."""

    override = os.environ.get(_FORENSICS_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(state_dir).expanduser().parent / "forensics"


def _local_rank_pids(
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> list[int]:
    """This rank's pid plus any sibling local rank/launcher processes.

    ``pgrep -f`` matches the worker module path, which is how a second local
    rank or the mlx.launch parent gets sampled too. Remote ranks run their own
    watchdog and capture on their own Mac — there is deliberately no SSH here.
    """

    pids = [os.getpid()]
    try:
        completed = runner(
            ["pgrep", "-f", "omlx.cluster.inference_worker"],
            capture_output=True,
            text=True,
            timeout=_PGREP_TIMEOUT_S,
            check=False,
        )
        if getattr(completed, "returncode", 1) == 0:
            for token in (completed.stdout or "").split():
                with suppress(ValueError):
                    pid = int(token)
                    if pid > 0 and pid not in pids:
                        pids.append(pid)
    except (OSError, subprocess.SubprocessError):
        pass
    return pids[:_MAX_RANK_PIDS]


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def capture_wedge_forensics(
    *,
    reason: str,
    stall_seconds: float,
    rank: int,
    state_dir: str | Path = "~/.omlx/cluster/runtime",
    out_dir: Path | None = None,
    tag: str | None = None,
    pids: list[int] | None = None,
    registry_path: Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
    now: float | None = None,
) -> Path | None:
    """Write one forensics bundle for the in-progress stall. Never raises.

    Layout: ``<out_dir>/<tag>/`` with a ``manifest.json`` (reason, stall,
    per-step errors), one ``sample_<pid>.txt`` per local rank process, a
    ``vm_stat.txt`` wired snapshot, and copies of the small JSON state files
    (rank markers, deployment registry) that explain what the cluster believed
    at the moment it wedged.

    Returns the bundle directory, or None when even the directory could not be
    created. Explicitly *not* done here, on purpose: sudo, ``spindump``, and
    remote capture over SSH — the rank's own user can gather everything above,
    and a forensics hook that needs privileges fails exactly when the machine
    is already in trouble.
    """

    started = time.time() if now is None else now
    root = out_dir if out_dir is not None else default_forensics_dir(state_dir)
    if tag is None:
        tag = time.strftime("stall_%Y%m%d_%H%M%S", time.localtime(started))
    bundle = root / tag
    errors: list[str] = []
    try:
        bundle.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("wedge forensics: cannot create %s: %s", bundle, exc)
        return None

    if pids is None:
        pids = _local_rank_pids(runner=runner)
    sample = shutil.which("sample")
    sampled: list[int] = []
    for pid in pids:
        if sample is None:
            errors.append("sample(1) is unavailable")
            break
        target = bundle / f"sample_{pid}.txt"
        try:
            completed = runner(
                [sample, str(pid), str(_SAMPLE_SECONDS), "-file", str(target)],
                capture_output=True,
                text=True,
                timeout=_SAMPLE_TIMEOUT_S,
                check=False,
            )
            if getattr(completed, "returncode", 1) == 0 and target.is_file():
                sampled.append(pid)
            else:
                detail = (getattr(completed, "stderr", "") or "").strip()[:200]
                errors.append(f"sample pid {pid} failed: {detail or 'no output'}")
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"sample pid {pid} failed: {type(exc).__name__}: {exc}")

    wired_pages: int | None = None
    try:
        completed = runner(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=_VM_STAT_TIMEOUT_S,
            check=False,
        )
        if getattr(completed, "returncode", 1) == 0:
            _write_text(bundle / "vm_stat.txt", completed.stdout or "")
            wired = _wired_pages_from_vm_stat(completed.stdout or "")
            wired_pages = wired
        else:
            errors.append("vm_stat failed")
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"vm_stat failed: {type(exc).__name__}: {exc}")

    markers_copied: list[str] = []
    marker_root = Path(state_dir).expanduser()
    marker_dir = bundle / "markers"
    try:
        candidates = sorted(marker_root.glob("*.json"))[:_MAX_MARKERS_COPIED]
        if candidates:
            marker_dir.mkdir(exist_ok=True)
        for candidate in candidates:
            try:
                if candidate.stat().st_size > _MAX_MARKER_BYTES:
                    continue
                shutil.copyfile(candidate, marker_dir / candidate.name)
                markers_copied.append(candidate.name)
            except OSError as exc:
                errors.append(f"marker copy {candidate.name} failed: {exc}")
    except OSError as exc:
        errors.append(f"marker scan failed: {exc}")

    registry_copied = False
    if registry_path is None:
        registry_path = Path.home() / ".omlx" / "cluster" / "deployments.json"
    try:
        if registry_path.is_file() and registry_path.stat().st_size <= (
            _MAX_MARKER_BYTES
        ):
            shutil.copyfile(registry_path, bundle / "deployments.json")
            registry_copied = True
    except OSError as exc:
        errors.append(f"registry copy failed: {exc}")

    manifest = {
        "schema_version": 1,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "reason": reason,
        "stall_seconds": round(max(0.0, stall_seconds), 3),
        "rank": rank,
        "pid": os.getpid(),
        "pids": list(pids),
        "sampled_pids": sampled,
        "wired_pages": wired_pages,
        "markers_copied": markers_copied,
        "registry_copied": registry_copied,
        "errors": errors,
    }
    try:
        _write_text(
            bundle / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    except OSError as exc:
        logger.warning("wedge forensics: manifest write failed: %s", exc)
    logger.info("wedge forensics captured at %s (%s)", bundle, reason)
    return bundle


def _wired_pages_from_vm_stat(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("Pages wired down:"):
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                return int(digits)
    return None
