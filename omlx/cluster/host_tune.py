# SPDX-License-Identifier: Apache-2.0
"""Operator-invoked host tuning for dedicated inference Macs.

Two ThunderMLX host-level knobs, ported with their opt-in shape intact.
Nothing here runs at server startup, at cluster launch, or on any timer —
each helper executes only when the operator explicitly runs the matching
``omlx cluster`` command, and the privileged one never runs without a
visible confirmation or an explicit ``--yes``.

**Wired limit.** macOS refuses ``mx.set_wired_limit`` above Apple's default
(~75% of RAM) unless the kernel ``iogpu.wired_limit_mb`` ceiling is raised
first, so on a large-memory Mac this one sysctl decides whether a big model
— sharded or single-node — can wire its weights at all. Writing it needs
root, which is why oMLX otherwise only reads the knob and suggests a value
(in logs, the admin banner, and the Mac app) while never applying it.

The ceiling is not "as high as possible". Wiring a Mac to effective
unified-memory saturation is the documented path to the 90 s ``watchdogd``
check-in panic (the 2026-07-12 report: ~126 GiB wired, 14 MiB free pages,
and macOS never labelled it memory pressure — the whole system stalled,
then panicked; it was not an OOM). The default target therefore reuses the
memory enforcer's own suggestion clamp — physical RAM minus 5%, the
field-stable margin from #2184 (488 GiB wired stable on a 512 GiB box,
510 GiB crash-looped) — and anything above it needs ``--force``. The
sysctl is runtime-only and resets at reboot; nothing here touches nvram,
``defaults write``, or launchd.

**Noise reduction.** On a dedicated inference Mac, Spotlight indexing bursts
and the photo/Siri analysis daemons steal GPU/Neural-Engine time and memory
bandwidth, perturbing decode cadence and watchdog heartbeats. ThunderMLX
automated one combined tier behind ``M3_ENABLE_MACOS_NOISE_REDUCTION=1``;
oMLX splits it along the audit's safe-to-automate line. Killing the
analysis daemons is the default tier (they are launchd-managed, restart on
demand, and own no user data), the Spotlight tier needs sudo and sits behind
``--include-spotlight``, and quitting an interactive app stays behind
``--quit-safari``. On a daily-driver Mac this command is the wrong tool —
that is why it is a command and not a startup step.

Deliberately absent, per the same audit: no password-file sudo
(ThunderMLX's ``priv.sh`` read ``~/.config/thundermlx-ops/sudo_pass``; for
oMLX's desktop-user base the posture is passwordless-sudo or an interactive
prompt, never a secret on disk), and no automatic invocation from the
cluster launcher.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_SYSCTL = "/usr/sbin/sysctl"
_SUDO = "/usr/bin/sudo"
_KILLALL = "/usr/bin/killall"
_MDUTIL = "/usr/bin/mdutil"
_OSASCRIPT = "/usr/bin/osascript"
_WIRED_LIMIT_KEY = "iogpu.wired_limit_mb"
_MEMSIZE_KEY = "hw.memsize"
_MIB = 1024**2

_SYSCTL_TIMEOUT_S = 5.0
_SUDO_PROBE_TIMEOUT_S = 5.0
_STEP_TIMEOUT_S = 10.0

# The default (unprivileged) tier. Every one of these is a launchd-managed
# background consumer that restarts on demand and owns no user data, so
# killing it only defers background analysis — the same list ThunderMLX's
# M3_Start.command ran under M3_ENABLE_MACOS_NOISE_REDUCTION=1.
USER_TIER_PROCESSES: tuple[str, ...] = (
    "assistantd",
    "siriinferenced",
    "siriknowledged",
    "intelligenceplatformd",
    "intelligencecontextd",
    "knowledge-agent",
    "photoanalysisd",
    "mediaanalysisd",
    "photolibraryd",
)

# The Spotlight tier additionally needs root: mdutil flips indexing on every
# volume, and these daemons do not run as the console user.
SPOTLIGHT_PROCESSES: tuple[str, ...] = ("mds", "mds_stores", "corespotlightd")


def _parse_first_int(text: str) -> int:
    match = re.search(r"(\d+)", text or "")
    return int(match.group(1)) if match else 0


def _read_sysctl_int(key: str, *, runner: Callable[..., Any]) -> int:
    """An integer sysctl value, 0 when unreadable. Never raises."""

    try:
        completed = runner(
            [_SYSCTL, "-n", key],
            capture_output=True,
            text=True,
            timeout=_SYSCTL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if getattr(completed, "returncode", 1) != 0:
        return 0
    return _parse_first_int(getattr(completed, "stdout", ""))


def resolve_sudo_prefix(
    *,
    runner: Callable[..., Any] = subprocess.run,
    stdin_is_tty: bool | None = None,
) -> list[str] | None:
    """How to run a privileged command here, or None when sudo is unusable.

    Passwordless sudo (a NOPASSWD rule or cached credentials) runs
    noninteractively; otherwise an interactive terminal gets the normal sudo
    password prompt. Anything else — a noninteractive shell with no NOPASSWD
    rule — gets None plus the caller's printed instructions, never a
    password file and never a hidden prompt.
    """

    try:
        completed = runner(
            [_SUDO, "-n", "true"],
            capture_output=True,
            text=True,
            timeout=_SUDO_PROBE_TIMEOUT_S,
            check=False,
        )
        if getattr(completed, "returncode", 1) == 0:
            return [_SUDO, "-n"]
    except (OSError, subprocess.SubprocessError):
        pass
    if stdin_is_tty is None:
        try:
            stdin_is_tty = sys.stdin.isatty()
        except Exception:  # noqa: BLE001
            stdin_is_tty = False
    if stdin_is_tty:
        return [_SUDO]
    return None


@dataclass(frozen=True)
class WiredLimitPlan:
    """What ``apply-wired-limit`` would do, computed before any sudo call."""

    current_mb: int
    target_mb: int
    ram_bytes: int
    # The enforcer's suggestion ceiling (physical RAM − 5%) in MiB; 0 when
    # the memory size could not be read.
    safe_ceiling_mb: int
    # "apply" | "already_sufficient" | "needs_force" | "refused"
    action: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_mb": self.current_mb,
            "target_mb": self.target_mb,
            "ram_bytes": self.ram_bytes,
            "safe_ceiling_mb": self.safe_ceiling_mb,
            "action": self.action,
            "detail": self.detail,
        }


def plan_wired_limit(
    *,
    requested_mb: int | None = None,
    force: bool = False,
    runner: Callable[..., Any] = subprocess.run,
) -> WiredLimitPlan:
    """Decide the target ``iogpu.wired_limit_mb`` before touching sudo.

    With no explicit value the target is the same suggestion the enforcer,
    the logs and the admin banner already make: physical RAM minus 5%
    (#2184), so following it cannot wire the Mac into the unified-memory
    saturation whose documented end is the 90 s watchdogd check-in panic.
    An explicit value is honoured as-is below that ceiling; above it the
    plan refuses unless ``force`` — and above physical RAM it refuses
    outright, because wiring more memory than the machine has is a typo,
    not a tune. An explicit ``0`` restores Apple's default and is always
    the safe direction.
    """

    current_mb = _read_sysctl_int(_WIRED_LIMIT_KEY, runner=runner)
    ram_bytes = _read_sysctl_int(_MEMSIZE_KEY, runner=runner)

    # Resolved through the enforcer so this suggestion can never drift from
    # the one the runtime logs and the admin banner show.
    from ..process_memory_enforcer import wired_limit_suggestion_bytes

    safe_ceiling_mb = (
        wired_limit_suggestion_bytes(ram_bytes) // _MIB if ram_bytes > 0 else 0
    )

    if requested_mb is None:
        if safe_ceiling_mb <= 0:
            return WiredLimitPlan(
                current_mb,
                0,
                ram_bytes,
                safe_ceiling_mb,
                "refused",
                "could not read this Mac's memory size (sysctl hw.memsize "
                "failed); re-run with an explicit --mb value",
            )
        target_mb = safe_ceiling_mb
        if current_mb >= target_mb and current_mb > 0:
            return WiredLimitPlan(
                current_mb,
                target_mb,
                ram_bytes,
                safe_ceiling_mb,
                "already_sufficient",
                f"current limit ({current_mb} MB) is already at or above "
                f"the suggested {target_mb} MB",
            )
        return WiredLimitPlan(
            current_mb,
            target_mb,
            ram_bytes,
            safe_ceiling_mb,
            "apply",
            f"raise iogpu.wired_limit_mb from "
            f"{'unset (Apple default)' if current_mb == 0 else f'{current_mb} MB'} "
            f"to {target_mb} MB (physical RAM − 5%)",
        )

    target_mb = int(requested_mb)
    if target_mb < 0:
        return WiredLimitPlan(
            current_mb,
            target_mb,
            ram_bytes,
            safe_ceiling_mb,
            "refused",
            "wired limit must be >= 0 (0 restores Apple's default)",
        )
    if ram_bytes > 0 and target_mb * _MIB > ram_bytes:
        return WiredLimitPlan(
            current_mb,
            target_mb,
            ram_bytes,
            safe_ceiling_mb,
            "refused",
            f"{target_mb} MB exceeds this Mac's physical RAM "
            f"({ram_bytes // _MIB} MB); refusing to wire more memory than "
            "the machine has",
        )
    if (
        target_mb > 0
        and safe_ceiling_mb > 0
        and target_mb > safe_ceiling_mb
        and not force
    ):
        return WiredLimitPlan(
            current_mb,
            target_mb,
            ram_bytes,
            safe_ceiling_mb,
            "needs_force",
            f"{target_mb} MB is above the safe ceiling of "
            f"{safe_ceiling_mb} MB (physical RAM − 5%, #2184); wiring "
            "into the last 5% risks the unified-memory saturation whose "
            "documented end is a watchdogd panic. Re-run with --force "
            "if you accept that.",
        )
    if current_mb == target_mb:
        return WiredLimitPlan(
            current_mb,
            target_mb,
            ram_bytes,
            safe_ceiling_mb,
            "already_sufficient",
            f"iogpu.wired_limit_mb is already {current_mb} MB",
        )
    direction = "raise" if target_mb > current_mb else "lower"
    return WiredLimitPlan(
        current_mb,
        target_mb,
        ram_bytes,
        safe_ceiling_mb,
        "apply",
        f"{direction} iogpu.wired_limit_mb from "
        f"{'unset (Apple default)' if current_mb == 0 else f'{current_mb} MB'} "
        f"to {target_mb} MB",
    )


def wired_limit_sysctl_command(target_mb: int) -> list[str]:
    """The exact sysctl write, without the sudo prefix."""

    return [_SYSCTL, f"{_WIRED_LIMIT_KEY}={int(target_mb)}"]


@dataclass(frozen=True)
class WiredLimitResult:
    ok: bool
    action: str
    command: tuple[str, ...]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "command": list(self.command),
            "detail": self.detail,
        }


def apply_wired_limit(
    plan: WiredLimitPlan,
    *,
    sudo_prefix: list[str] | None,
    runner: Callable[..., Any] = subprocess.run,
) -> WiredLimitResult:
    """Run the plan's sysctl write. Never raises; verifies by read-back.

    An interactive sudo gets the terminal (no capture, no timeout) so the
    password prompt works; the passwordless path stays captured and bounded.
    """

    if plan.action != "apply":
        return WiredLimitResult(False, plan.action, (), plan.detail)
    if not sudo_prefix:
        return WiredLimitResult(
            False,
            "no_sudo",
            (),
            "sudo is required to write iogpu.wired_limit_mb but is not "
            "usable here. Re-run from an interactive terminal, or configure "
            "a passwordless sudo rule for /usr/sbin/sysctl.",
        )
    command = [*sudo_prefix, *wired_limit_sysctl_command(plan.target_mb)]
    interactive = sudo_prefix == [_SUDO]
    try:
        completed = runner(
            command,
            capture_output=not interactive,
            text=True,
            timeout=None if interactive else _SYSCTL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return WiredLimitResult(
            False, "apply", tuple(command), f"{type(exc).__name__}: {exc}"
        )
    if getattr(completed, "returncode", 1) != 0:
        detail = (getattr(completed, "stderr", "") or "").strip()[:200]
        return WiredLimitResult(
            False,
            "apply",
            tuple(command),
            f"sysctl write failed: {detail or 'no output'}",
        )
    observed = _read_sysctl_int(_WIRED_LIMIT_KEY, runner=runner)
    if observed == plan.target_mb:
        return WiredLimitResult(
            True,
            "apply",
            tuple(command),
            f"iogpu.wired_limit_mb is now {observed} MB (verified by "
            "read-back; resets at reboot)",
        )
    return WiredLimitResult(
        False,
        "apply",
        tuple(command),
        f"sysctl reported success but read-back shows {observed} MB, "
        f"not the requested {plan.target_mb} MB",
    )


@dataclass(frozen=True)
class NoiseStep:
    """One best-effort suppression step, in the order it should run."""

    argv: tuple[str, ...]
    needs_sudo: bool
    summary: str


def noise_reduction_steps(
    *,
    include_spotlight: bool = False,
    quit_safari: bool = False,
) -> list[NoiseStep]:
    """The steps for `omlx cluster reduce-noise`, most privileged first.

    The default tier is unprivileged and only stops daemons that relaunch
    on demand. ``--include-spotlight`` adds the sudo tier (indexing off on
    every volume plus the root-owned Spotlight daemons); ``--quit-safari``
    adds the one step that touches an interactive app.
    """

    steps: list[NoiseStep] = []
    if include_spotlight:
        steps.append(
            NoiseStep(
                (_MDUTIL, "-a", "-i", "off"),
                True,
                "disable Spotlight indexing on all volumes",
            )
        )
        steps.extend(
            NoiseStep((_KILLALL, name), True, f"stop {name}")
            for name in SPOTLIGHT_PROCESSES
        )
    steps.extend(
        NoiseStep((_KILLALL, name), False, f"stop {name}")
        for name in USER_TIER_PROCESSES
    )
    if quit_safari:
        steps.append(
            NoiseStep(
                (_OSASCRIPT, "-e", 'quit app "Safari"'),
                False,
                "quit Safari",
            )
        )
    return steps


@dataclass(frozen=True)
class NoiseStepResult:
    step: NoiseStep
    # "ok" | "not_running" | "skipped_no_sudo" | "failed"
    outcome: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.step.argv),
            "needs_sudo": self.step.needs_sudo,
            "summary": self.step.summary,
            "outcome": self.outcome,
            "detail": self.detail,
        }


def apply_noise_reduction(
    steps: list[NoiseStep],
    *,
    sudo_prefix: list[str] | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> list[NoiseStepResult]:
    """Run each step best-effort. Never raises; per-step outcomes instead.

    ``killall`` exits non-zero when nothing matched, which on an idle Mac is
    the common case, not a failure — that stderr is classified
    ``not_running`` so the summary stays honest without crying wolf.
    """

    results: list[NoiseStepResult] = []
    for step in steps:
        if step.needs_sudo and not sudo_prefix:
            results.append(
                NoiseStepResult(
                    step,
                    "skipped_no_sudo",
                    "needs sudo; re-run from an interactive terminal or "
                    "configure passwordless sudo",
                )
            )
            continue
        argv = [*(sudo_prefix or []), *step.argv] if step.needs_sudo else list(step.argv)
        interactive = step.needs_sudo and sudo_prefix == [_SUDO]
        try:
            completed = runner(
                argv,
                capture_output=not interactive,
                text=True,
                timeout=None if interactive else _STEP_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            results.append(
                NoiseStepResult(step, "failed", f"{type(exc).__name__}: {exc}")
            )
            continue
        returncode = getattr(completed, "returncode", 1)
        stderr = (getattr(completed, "stderr", "") or "").strip()
        if returncode == 0:
            results.append(NoiseStepResult(step, "ok", ""))
        elif "no matching processes" in stderr.lower():
            results.append(NoiseStepResult(step, "not_running", stderr[:200]))
        else:
            detail = stderr or (getattr(completed, "stdout", "") or "").strip()
            results.append(
                NoiseStepResult(step, "failed", detail[:200] or f"exit {returncode}")
            )
    return results
