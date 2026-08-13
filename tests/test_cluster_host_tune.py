# SPDX-License-Identifier: Apache-2.0
"""Host tuning helpers: decision logic and command construction, no sudo.

Every runner here is fake — no test may execute a real sysctl, sudo,
killall, mdutil, or osascript. What is under test is the decision logic
(plan actions, force discipline, outcome classification) and the exact
command lines the helpers would run.
"""

import io
import json
import subprocess
import sys
from types import SimpleNamespace

from omlx import cli
from omlx.cluster import host_tune
from omlx.cluster.host_tune import (
    WiredLimitPlan,
    apply_noise_reduction,
    apply_wired_limit,
    noise_reduction_steps,
    plan_wired_limit,
    resolve_sudo_prefix,
    wired_limit_sysctl_command,
)

_GIB = 1024**3
_MIB = 1024**2


def _sysctl_runner(values: dict, failures: tuple = ()):
    """A runner answering only `sysctl -n <key>` reads."""

    def runner(command, **_kwargs):
        assert command[:2] == ["/usr/sbin/sysctl", "-n"]
        key = command[2]
        if key in failures:
            raise OSError("sysctl unavailable")
        return SimpleNamespace(
            returncode=0, stdout=str(values.get(key, 0)), stderr=""
        )

    return runner


def _ram_runner(ram_bytes: int, current_mb: int = 0):
    return _sysctl_runner(
        {"iogpu.wired_limit_mb": current_mb, "hw.memsize": ram_bytes}
    )


def _fake_host(monkeypatch, ram_bytes: int, current_mb: int = 0):
    """A fake Mac whose RAM is visible to both sysctl and the enforcer.

    The safe-ceiling clamp lives in the memory enforcer and resolves
    physical RAM through omlx.settings.get_system_memory, so faking only
    the sysctl read would let the test host's real RAM leak into the plan.
    """

    import omlx.settings

    monkeypatch.setattr(
        omlx.settings, "get_system_memory", lambda: ram_bytes
    )
    return _ram_runner(ram_bytes, current_mb)


def _suggestion_mb(ram_bytes: int) -> int:
    return (ram_bytes - ram_bytes // 20) // _MIB


class TestPlanWiredLimit:
    def test_default_target_is_ram_minus_five_percent(self, monkeypatch):
        ram = 256 * _GIB
        plan = plan_wired_limit(runner=_fake_host(monkeypatch, ram))
        assert plan.action == "apply"
        assert plan.current_mb == 0
        assert plan.target_mb == _suggestion_mb(ram)
        assert plan.safe_ceiling_mb == _suggestion_mb(ram)
        assert "unset (Apple default)" in plan.detail

    def test_existing_limit_is_reported_in_detail(self, monkeypatch):
        ram = 128 * _GIB
        plan = plan_wired_limit(
            runner=_fake_host(monkeypatch, ram, current_mb=40960)
        )
        assert plan.action == "apply"
        assert "40960 MB" in plan.detail

    def test_current_above_suggestion_is_already_sufficient(self, monkeypatch):
        ram = 128 * _GIB
        current = _suggestion_mb(ram) + 1000
        plan = plan_wired_limit(
            runner=_fake_host(monkeypatch, ram, current_mb=current)
        )
        assert plan.action == "already_sufficient"
        assert str(current) in plan.detail

    def test_explicit_value_is_honoured_below_the_ceiling(self, monkeypatch):
        plan = plan_wired_limit(
            requested_mb=120 * 1024,
            runner=_fake_host(monkeypatch, 128 * _GIB),
        )
        assert plan.action == "apply"
        assert plan.target_mb == 120 * 1024

    def test_explicit_zero_restores_apple_default_without_force(
        self, monkeypatch
    ):
        plan = plan_wired_limit(
            requested_mb=0,
            runner=_fake_host(monkeypatch, 256 * _GIB, current_mb=200000),
        )
        assert plan.action == "apply"
        assert plan.target_mb == 0
        assert "lower" in plan.detail

    def test_explicit_above_safe_ceiling_needs_force(self, monkeypatch):
        ram = 128 * _GIB
        requested = _suggestion_mb(ram) + 512
        plan = plan_wired_limit(
            requested_mb=requested, runner=_fake_host(monkeypatch, ram)
        )
        assert plan.action == "needs_force"
        assert "--force" in plan.detail

    def test_explicit_above_safe_ceiling_with_force_applies(self, monkeypatch):
        ram = 128 * _GIB
        requested = _suggestion_mb(ram) + 512
        plan = plan_wired_limit(
            requested_mb=requested,
            force=True,
            runner=_fake_host(monkeypatch, ram),
        )
        assert plan.action == "apply"
        assert plan.target_mb == requested

    def test_explicit_above_physical_ram_is_refused_even_with_force(
        self, monkeypatch
    ):
        ram = 128 * _GIB
        plan = plan_wired_limit(
            requested_mb=ram // _MIB + 1,
            force=True,
            runner=_fake_host(monkeypatch, ram),
        )
        assert plan.action == "refused"
        assert "physical RAM" in plan.detail

    def test_negative_value_is_refused(self, monkeypatch):
        plan = plan_wired_limit(
            requested_mb=-1, runner=_fake_host(monkeypatch, 128 * _GIB)
        )
        assert plan.action == "refused"

    def test_explicit_equal_to_current_is_already_sufficient(self, monkeypatch):
        plan = plan_wired_limit(
            requested_mb=40960,
            runner=_fake_host(monkeypatch, 128 * _GIB, current_mb=40960),
        )
        assert plan.action == "already_sufficient"

    def test_unreadable_ram_refuses_the_automatic_suggestion(self):
        runner = _sysctl_runner(
            {"iogpu.wired_limit_mb": 0}, failures=("hw.memsize",)
        )
        plan = plan_wired_limit(runner=runner)
        assert plan.action == "refused"
        assert "--mb" in plan.detail

    def test_unreadable_ram_still_allows_an_explicit_value(self):
        runner = _sysctl_runner(
            {"iogpu.wired_limit_mb": 0}, failures=("hw.memsize",)
        )
        plan = plan_wired_limit(requested_mb=120 * 1024, runner=runner)
        assert plan.action == "apply"
        assert plan.safe_ceiling_mb == 0

    def test_unreadable_current_limit_is_treated_as_unset(self):
        runner = _sysctl_runner(
            {"hw.memsize": 128 * _GIB}, failures=("iogpu.wired_limit_mb",)
        )
        plan = plan_wired_limit(runner=runner)
        assert plan.action == "apply"
        assert plan.current_mb == 0


class TestWiredLimitCommand:
    def test_command_shape(self):
        assert wired_limit_sysctl_command(249037) == [
            "/usr/sbin/sysctl",
            "iogpu.wired_limit_mb=249037",
        ]


class TestResolveSudoPrefix:
    def test_passwordless_sudo_wins(self):
        def runner(command, **kwargs):
            assert command == ["/usr/bin/sudo", "-n", "true"]
            assert kwargs.get("check") is False
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        assert resolve_sudo_prefix(runner=runner, stdin_is_tty=True) == [
            "/usr/bin/sudo",
            "-n",
        ]

    def test_interactive_fallback_when_passwordless_fails(self):
        def runner(command, **_kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="")

        assert resolve_sudo_prefix(runner=runner, stdin_is_tty=True) == [
            "/usr/bin/sudo"
        ]

    def test_no_tty_and_no_passwordless_means_no_sudo(self):
        def runner(command, **_kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="")

        assert resolve_sudo_prefix(runner=runner, stdin_is_tty=False) is None

    def test_sudo_probe_raising_means_no_sudo_without_tty(self):
        def runner(command, **_kwargs):
            raise OSError("no sudo binary")

        assert resolve_sudo_prefix(runner=runner, stdin_is_tty=False) is None


def _apply_plan(target_mb: int = 249037) -> WiredLimitPlan:
    return WiredLimitPlan(
        current_mb=0,
        target_mb=target_mb,
        ram_bytes=256 * _GIB,
        safe_ceiling_mb=249037,
        action="apply",
        detail="raise",
    )


class TestApplyWiredLimit:
    def test_non_apply_plan_never_runs_a_command(self):
        plan = WiredLimitPlan(0, 0, 0, 0, "refused", "no")

        def runner(command, **_kwargs):
            raise AssertionError(f"unexpected command: {command}")

        result = apply_wired_limit(plan, sudo_prefix=["/usr/bin/sudo", "-n"], runner=runner)
        assert result.ok is False
        assert result.action == "refused"
        assert result.command == ()

    def test_missing_sudo_never_runs_a_command(self):
        def runner(command, **_kwargs):
            raise AssertionError(f"unexpected command: {command}")

        result = apply_wired_limit(_apply_plan(), sudo_prefix=None, runner=runner)
        assert result.ok is False
        assert result.action == "no_sudo"
        assert "passwordless" in result.detail

    def test_success_is_verified_by_read_back(self):
        commands: list[list[str]] = []

        def runner(command, **kwargs):
            commands.append((list(command), kwargs))
            if command[1] == "-n" and command[2] == "true":
                raise AssertionError("sudo probe is resolve_sudo_prefix's job")
            if command[:2] == ["/usr/sbin/sysctl", "-n"]:
                return SimpleNamespace(returncode=0, stdout="249037", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        result = apply_wired_limit(
            _apply_plan(), sudo_prefix=["/usr/bin/sudo", "-n"], runner=runner
        )
        assert result.ok is True
        assert "verified" in result.detail
        written = commands[0][0]
        assert written == [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/sysctl",
            "iogpu.wired_limit_mb=249037",
        ]
        # The passwordless path stays captured and time-bounded.
        assert commands[0][1]["capture_output"] is True
        assert commands[0][1]["timeout"] is not None

    def test_interactive_sudo_gets_the_terminal(self):
        seen: dict = {}

        def runner(command, **kwargs):
            if command[:2] == ["/usr/sbin/sysctl", "-n"]:
                return SimpleNamespace(returncode=0, stdout="249037", stderr="")
            seen.update(kwargs)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        result = apply_wired_limit(
            _apply_plan(), sudo_prefix=["/usr/bin/sudo"], runner=runner
        )
        assert result.ok is True
        assert seen["capture_output"] is False
        assert seen["timeout"] is None

    def test_sysctl_failure_is_reported_not_raised(self):
        def runner(command, **_kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="operation not permitted")

        result = apply_wired_limit(
            _apply_plan(), sudo_prefix=["/usr/bin/sudo", "-n"], runner=runner
        )
        assert result.ok is False
        assert "operation not permitted" in result.detail

    def test_read_back_mismatch_fails(self):
        def runner(command, **_kwargs):
            if command[:2] == ["/usr/sbin/sysctl", "-n"]:
                return SimpleNamespace(returncode=0, stdout="0", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        result = apply_wired_limit(
            _apply_plan(), sudo_prefix=["/usr/bin/sudo", "-n"], runner=runner
        )
        assert result.ok is False
        assert "read-back" in result.detail

    def test_runner_raising_is_a_result_not_an_exception(self):
        def runner(command, **_kwargs):
            raise subprocess.TimeoutExpired(command, 5)

        result = apply_wired_limit(
            _apply_plan(), sudo_prefix=["/usr/bin/sudo", "-n"], runner=runner
        )
        assert result.ok is False
        assert "TimeoutExpired" in result.detail


class TestNoiseReductionSteps:
    def test_default_tier_is_unprivileged_killalls_only(self):
        steps = noise_reduction_steps()
        assert len(steps) == len(host_tune.USER_TIER_PROCESSES)
        assert all(not step.needs_sudo for step in steps)
        assert all(step.argv[0] == "/usr/bin/killall" for step in steps)
        assert {step.argv[1] for step in steps} == set(host_tune.USER_TIER_PROCESSES)

    def test_spotlight_tier_comes_first_and_needs_sudo(self):
        steps = noise_reduction_steps(include_spotlight=True)
        assert steps[0].argv == ("/usr/bin/mdutil", "-a", "-i", "off")
        assert steps[0].needs_sudo is True
        sudo_killalls = steps[1 : 1 + len(host_tune.SPOTLIGHT_PROCESSES)]
        assert all(step.needs_sudo for step in sudo_killalls)
        assert {step.argv[1] for step in sudo_killalls} == set(
            host_tune.SPOTLIGHT_PROCESSES
        )
        user_tier = steps[1 + len(host_tune.SPOTLIGHT_PROCESSES) :]
        assert all(not step.needs_sudo for step in user_tier)
        assert len(user_tier) == len(host_tune.USER_TIER_PROCESSES)

    def test_quit_safari_is_an_explicit_osascript_step(self):
        steps = noise_reduction_steps(quit_safari=True)
        assert steps[-1].argv == ("/usr/bin/osascript", "-e", 'quit app "Safari"')
        assert steps[-1].needs_sudo is False
        assert len(steps) == len(host_tune.USER_TIER_PROCESSES) + 1


class TestApplyNoiseReduction:
    def test_successful_steps_run_without_sudo(self):
        commands: list[list[str]] = []

        def runner(command, **kwargs):
            commands.append(list(command))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        steps = noise_reduction_steps()
        results = apply_noise_reduction(steps, runner=runner)
        assert all(r.outcome == "ok" for r in results)
        assert commands == [list(step.argv) for step in steps]

    def test_no_matching_processes_is_not_a_failure(self):
        def runner(command, **_kwargs):
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr='No matching processes belonging to you were found',
            )

        results = apply_noise_reduction(noise_reduction_steps(), runner=runner)
        assert all(r.outcome == "not_running" for r in results)

    def test_other_killall_failures_are_failures(self):
        def runner(command, **_kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="weird error")

        results = apply_noise_reduction(noise_reduction_steps(), runner=runner)
        assert all(r.outcome == "failed" for r in results)
        assert all("weird error" in r.detail for r in results)

    def test_sudo_steps_are_skipped_without_a_prefix(self):
        def runner(command, **_kwargs):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        steps = noise_reduction_steps(include_spotlight=True)
        results = apply_noise_reduction(steps, sudo_prefix=None, runner=runner)
        sudo_results = [r for r in results if r.step.needs_sudo]
        assert all(r.outcome == "skipped_no_sudo" for r in sudo_results)
        assert len(sudo_results) == 1 + len(host_tune.SPOTLIGHT_PROCESSES)
        user_results = [r for r in results if not r.step.needs_sudo]
        assert all(r.outcome == "ok" for r in user_results)

    def test_sudo_steps_get_the_prefix_when_available(self):
        commands: list[list[str]] = []

        def runner(command, **kwargs):
            commands.append(list(command))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        steps = noise_reduction_steps(include_spotlight=True)
        results = apply_noise_reduction(
            steps, sudo_prefix=["/usr/bin/sudo", "-n"], runner=runner
        )
        assert all(r.outcome == "ok" for r in results)
        assert commands[0] == ["/usr/bin/sudo", "-n", "/usr/bin/mdutil", "-a", "-i", "off"]
        user_commands = commands[1 + len(host_tune.SPOTLIGHT_PROCESSES) :]
        assert all(not c[0].endswith("sudo") for c in user_commands)

    def test_runner_raising_is_classified_not_propagated(self):
        def runner(command, **_kwargs):
            raise OSError("missing binary")

        results = apply_noise_reduction(noise_reduction_steps(), runner=runner)
        assert all(r.outcome == "failed" for r in results)
        assert all("OSError" in r.detail for r in results)


def _cluster_args(**overrides):
    base = {
        "cluster_action": "apply-wired-limit",
        "mb": None,
        "force": False,
        "yes": True,
        "dry_run": False,
        "json": True,
        "include_spotlight": False,
        "quit_safari": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestApplyWiredLimitCli:
    def test_dry_run_prints_plan_and_runs_nothing(self, monkeypatch, capsys):
        monkeypatch.setattr(
            host_tune, "plan_wired_limit", lambda **kw: _apply_plan()
        )

        def forbidden(**_kwargs):
            raise AssertionError("dry-run must not resolve sudo or apply")

        monkeypatch.setattr(host_tune, "resolve_sudo_prefix", forbidden)
        monkeypatch.setattr(host_tune, "apply_wired_limit", forbidden)

        rc = cli.cluster_command(_cluster_args(dry_run=True))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["plan"]["action"] == "apply"
        assert payload["command"] == [
            "sudo",
            "/usr/sbin/sysctl",
            "iogpu.wired_limit_mb=249037",
        ]

    def test_yes_applies_through_the_helpers(self, monkeypatch, capsys):
        monkeypatch.setattr(
            host_tune, "plan_wired_limit", lambda **kw: _apply_plan()
        )
        monkeypatch.setattr(
            host_tune, "resolve_sudo_prefix", lambda **kw: ["/usr/bin/sudo", "-n"]
        )
        seen: dict = {}

        def fake_apply(plan, *, sudo_prefix, runner=subprocess.run):
            seen["plan"] = plan
            seen["sudo_prefix"] = sudo_prefix
            return host_tune.WiredLimitResult(
                True, "apply", tuple(sudo_prefix), "applied"
            )

        monkeypatch.setattr(host_tune, "apply_wired_limit", fake_apply)
        rc = cli.cluster_command(_cluster_args())
        assert rc == 0
        assert seen["sudo_prefix"] == ["/usr/bin/sudo", "-n"]
        assert json.loads(capsys.readouterr().out)["result"]["ok"] is True

    def test_non_interactive_shell_needs_yes(self, monkeypatch, capsys):
        monkeypatch.setattr(
            host_tune, "plan_wired_limit", lambda **kw: _apply_plan()
        )
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))

        def forbidden(**_kwargs):
            raise AssertionError("must not apply without confirmation")

        monkeypatch.setattr(host_tune, "apply_wired_limit", forbidden)
        rc = cli.cluster_command(_cluster_args(yes=False))
        assert rc == 2
        assert "--yes" in capsys.readouterr().err

    def test_interactive_decline_aborts(self, monkeypatch, capsys):
        monkeypatch.setattr(
            host_tune, "plan_wired_limit", lambda **kw: _apply_plan()
        )

        class _Tty(io.StringIO):
            def isatty(self):
                return True

        monkeypatch.setattr(sys, "stdin", _Tty(""))
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")

        def forbidden(**_kwargs):
            raise AssertionError("declined apply must not run")

        monkeypatch.setattr(host_tune, "apply_wired_limit", forbidden)
        rc = cli.cluster_command(_cluster_args(yes=False, json=False))
        assert rc == 1
        assert "Aborted" in capsys.readouterr().out

    def test_interactive_accept_applies(self, monkeypatch, capsys):
        monkeypatch.setattr(
            host_tune, "plan_wired_limit", lambda **kw: _apply_plan()
        )

        class _Tty(io.StringIO):
            def isatty(self):
                return True

        monkeypatch.setattr(sys, "stdin", _Tty(""))
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        monkeypatch.setattr(
            host_tune, "resolve_sudo_prefix", lambda **kw: ["/usr/bin/sudo"]
        )
        monkeypatch.setattr(
            host_tune,
            "apply_wired_limit",
            lambda plan, *, sudo_prefix, runner=None: host_tune.WiredLimitResult(
                True, "apply", tuple(sudo_prefix), "applied"
            ),
        )
        rc = cli.cluster_command(_cluster_args(yes=False, json=False))
        assert rc == 0
        assert "applied" in capsys.readouterr().out

    def test_refused_plan_exits_2_without_sudo(self, monkeypatch):
        monkeypatch.setattr(
            host_tune,
            "plan_wired_limit",
            lambda **kw: WiredLimitPlan(0, 0, 0, 0, "needs_force", "too high"),
        )
        monkeypatch.setattr(
            host_tune,
            "resolve_sudo_prefix",
            lambda **kw: (_ for _ in ()).throw(AssertionError("no sudo")),
        )
        assert cli.cluster_command(_cluster_args()) == 2

    def test_already_sufficient_exits_0(self, monkeypatch, capsys):
        monkeypatch.setattr(
            host_tune,
            "plan_wired_limit",
            lambda **kw: WiredLimitPlan(
                249037, 249037, 0, 249037, "already_sufficient", "already there"
            ),
        )
        rc = cli.cluster_command(_cluster_args())
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["plan"]["action"] == (
            "already_sufficient"
        )

    def test_unusable_sudo_exits_1_with_instructions(self, monkeypatch, capsys):
        monkeypatch.setattr(
            host_tune, "plan_wired_limit", lambda **kw: _apply_plan()
        )
        monkeypatch.setattr(host_tune, "resolve_sudo_prefix", lambda **kw: None)
        rc = cli.cluster_command(_cluster_args())
        assert rc == 1
        assert "passwordless" in capsys.readouterr().err

    def test_failed_apply_exits_1(self, monkeypatch, capsys):
        monkeypatch.setattr(
            host_tune, "plan_wired_limit", lambda **kw: _apply_plan()
        )
        monkeypatch.setattr(
            host_tune, "resolve_sudo_prefix", lambda **kw: ["/usr/bin/sudo", "-n"]
        )
        monkeypatch.setattr(
            host_tune,
            "apply_wired_limit",
            lambda plan, *, sudo_prefix, runner=None: host_tune.WiredLimitResult(
                False, "apply", (), "sysctl write failed"
            ),
        )
        rc = cli.cluster_command(_cluster_args())
        assert rc == 1
        assert json.loads(capsys.readouterr().out)["result"]["ok"] is False


class TestReduceNoiseCli:
    @staticmethod
    def _results_with(steps, outcomes):
        assert len(outcomes) == len(steps)
        return [
            host_tune.NoiseStepResult(step, outcome, "")
            for step, outcome in zip(steps, outcomes, strict=True)
        ]

    def _ok_results(self, steps):
        return self._results_with(steps, ["ok"] * len(steps))

    def test_dry_run_lists_steps_and_runs_nothing(self, monkeypatch, capsys):
        monkeypatch.setattr(
            host_tune,
            "apply_noise_reduction",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no run")),
        )
        rc = cli.cluster_command(
            _cluster_args(cluster_action="reduce-noise", dry_run=True)
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["dry_run"] is True
        assert len(payload["steps"]) == len(host_tune.USER_TIER_PROCESSES)
        assert all(not step["needs_sudo"] for step in payload["steps"])

    def test_sudo_resolved_only_for_the_spotlight_tier(self, monkeypatch, capsys):
        calls: list[bool] = []
        monkeypatch.setattr(
            host_tune,
            "resolve_sudo_prefix",
            lambda **kw: calls.append(True) or ["/usr/bin/sudo", "-n"],
        )
        monkeypatch.setattr(
            host_tune,
            "apply_noise_reduction",
            lambda steps, *, sudo_prefix=None, runner=None: self._ok_results(steps),
        )
        capsys.readouterr()
        rc = cli.cluster_command(_cluster_args(cluster_action="reduce-noise"))
        assert rc == 0
        assert calls == []

        rc = cli.cluster_command(
            _cluster_args(cluster_action="reduce-noise", include_spotlight=True)
        )
        assert rc == 0
        assert calls == [True]

    def test_ok_and_not_running_is_a_clean_run(self, monkeypatch, capsys):
        def fake_apply(steps, *, sudo_prefix=None, runner=None):
            return self._results_with(
                steps, ["ok", "not_running"] + ["ok"] * (len(steps) - 2)
            )

        monkeypatch.setattr(host_tune, "apply_noise_reduction", fake_apply)
        rc = cli.cluster_command(
            _cluster_args(cluster_action="reduce-noise", json=False)
        )
        assert rc == 0
        assert "not running" in capsys.readouterr().out

    def test_failures_exit_1(self, monkeypatch, capsys):
        def fake_apply(steps, *, sudo_prefix=None, runner=None):
            return self._results_with(
                steps, ["failed"] + ["ok"] * (len(steps) - 1)
            )

        monkeypatch.setattr(host_tune, "apply_noise_reduction", fake_apply)
        rc = cli.cluster_command(_cluster_args(cluster_action="reduce-noise"))
        assert rc == 1

    def test_skipped_sudo_tier_exits_1_with_instructions(
        self, monkeypatch, capsys
    ):
        def fake_apply(steps, *, sudo_prefix=None, runner=None):
            return [
                host_tune.NoiseStepResult(
                    step,
                    "skipped_no_sudo" if step.needs_sudo else "ok",
                    "",
                )
                for step in steps
            ]

        monkeypatch.setattr(host_tune, "resolve_sudo_prefix", lambda **kw: None)
        monkeypatch.setattr(host_tune, "apply_noise_reduction", fake_apply)
        rc = cli.cluster_command(
            _cluster_args(cluster_action="reduce-noise", include_spotlight=True)
        )
        assert rc == 1
        assert "sudo" in capsys.readouterr().err


class TestCliExposure:
    """The new commands are real subcommands, safe to invoke read-only."""

    def test_cluster_help_lists_host_tuning_commands(self):
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "cluster", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "apply-wired-limit" in result.stdout
        assert "reduce-noise" in result.stdout

    def test_apply_wired_limit_help_marks_sudo_and_opt_in(self):
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "cluster", "apply-wired-limit", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "sudo" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--force" in result.stdout

    def test_reduce_noise_help_lists_tiers(self):
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "cluster", "reduce-noise", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--include-spotlight" in result.stdout
        assert "--quit-safari" in result.stdout

    def test_apply_wired_limit_dry_run_needs_no_sudo(self):
        # An explicit small value below any real Mac's safe ceiling keeps
        # this deterministic: the plan is always apply/already_sufficient,
        # never needs_force, and --dry-run never resolves sudo.
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "omlx.cli",
                "cluster",
                "apply-wired-limit",
                "--mb",
                "1024",
                "--dry-run",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["plan"]["action"] in {"apply", "already_sufficient"}
        assert payload["command"] == [
            "sudo",
            "/usr/sbin/sysctl",
            "iogpu.wired_limit_mb=1024",
        ]


def test_user_tier_matches_thundermlx_list():
    # The audit's safe-to-automate tier, verbatim from M3_Start.command.
    assert set(host_tune.USER_TIER_PROCESSES) == {
        "assistantd",
        "siriinferenced",
        "siriknowledged",
        "intelligenceplatformd",
        "intelligencecontextd",
        "knowledge-agent",
        "photoanalysisd",
        "mediaanalysisd",
        "photolibraryd",
    }


def test_spotlight_tier_matches_thundermlx_list():
    assert set(host_tune.SPOTLIGHT_PROCESSES) == {
        "mds",
        "mds_stores",
        "corespotlightd",
    }
