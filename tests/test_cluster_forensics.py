# SPDX-License-Identifier: Apache-2.0
"""Live-wedge forensics capture: bundle contents and the no-sudo contract."""

import json
import os
from types import SimpleNamespace

import pytest

from omlx.cluster import forensics
from omlx.cluster.forensics import capture_wedge_forensics

_VM_STAT = (
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
    "Pages wired down:                            1024.\n"
)


@pytest.fixture(autouse=True)
def _sample_available(monkeypatch):
    """The capture must not depend on the test host having sample(1)."""

    monkeypatch.setattr(
        forensics.shutil, "which", lambda name: f"/usr/bin/{name}"
    )


def _runner_ok(command, **_kwargs):
    if command[0].endswith("sample"):
        target = command[command.index("-file") + 1]
        with open(target, "w") as stream:
            stream.write("fake sample output")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    if command[0] == "vm_stat":
        return SimpleNamespace(returncode=0, stdout=_VM_STAT, stderr="")
    if command[0] == "pgrep":
        return SimpleNamespace(returncode=1, stdout="", stderr="")
    raise AssertionError(f"unexpected command: {command}")


def test_bundle_contains_samples_vm_stat_markers_and_registry(tmp_path):
    state_dir = tmp_path / "runtime"
    state_dir.mkdir()
    (state_dir / "dep-1-rank-0.json").write_text(json.dumps({"phase": "ready"}))
    (state_dir / "dep-1-rank-1.json").write_text(json.dumps({"phase": "ready"}))
    registry = tmp_path / "cluster" / "deployments.json"
    registry.parent.mkdir()
    registry.write_text(json.dumps({"schema_version": 1, "deployments": []}))
    out_dir = tmp_path / "forensics"

    bundle = capture_wedge_forensics(
        reason="rank 0 generation stalled 61s",
        stall_seconds=61.0,
        rank=0,
        state_dir=state_dir,
        out_dir=out_dir,
        tag="stall_test",
        pids=[os.getpid()],
        registry_path=registry,
        runner=_runner_ok,
        now=1_700_000_000.0,
    )

    assert bundle == out_dir / "stall_test"
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["rank"] == 0
    assert manifest["stall_seconds"] == 61.0
    assert manifest["sampled_pids"] == [os.getpid()]
    assert manifest["wired_pages"] == 1024
    assert manifest["errors"] == []
    assert (bundle / f"sample_{os.getpid()}.txt").is_file()
    assert (bundle / "vm_stat.txt").read_text() == _VM_STAT
    assert sorted(manifest["markers_copied"]) == [
        "dep-1-rank-0.json",
        "dep-1-rank-1.json",
    ]
    assert (bundle / "markers" / "dep-1-rank-0.json").is_file()
    assert manifest["registry_copied"] is True
    assert (bundle / "deployments.json").is_file()


def test_capture_never_uses_sudo_or_spindump(tmp_path):
    commands: list[list[str]] = []

    def recording_runner(command, **kwargs):
        commands.append(list(command))
        return _runner_ok(command, **kwargs)

    capture_wedge_forensics(
        reason="test",
        stall_seconds=75.0,
        rank=1,
        state_dir=tmp_path / "runtime",
        out_dir=tmp_path / "out",
        pids=[1234],
        registry_path=tmp_path / "missing-registry.json",
        runner=recording_runner,
    )

    flat = {part for command in commands for part in command}
    assert "sudo" not in flat
    assert not any("spindump" in part for part in flat)
    assert commands[0][0].endswith("sample")
    assert commands[0][1:3] == ["1234", "5"]


def test_failed_steps_are_recorded_not_raised(tmp_path):
    def broken_runner(command, **_kwargs):
        if command[0] == "vm_stat":
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")
        raise OSError("no such tool")

    bundle = capture_wedge_forensics(
        reason="test",
        stall_seconds=61.0,
        rank=0,
        state_dir=tmp_path / "runtime",
        out_dir=tmp_path / "out",
        pids=[4321],
        registry_path=tmp_path / "missing.json",
        runner=broken_runner,
    )

    assert bundle is not None
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["sampled_pids"] == []
    assert manifest["wired_pages"] is None
    assert manifest["errors"], "failed steps must be recorded in the manifest"


def test_capture_returns_none_when_the_directory_cannot_be_created(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")

    bundle = capture_wedge_forensics(
        reason="test",
        stall_seconds=61.0,
        rank=0,
        state_dir=tmp_path / "runtime",
        out_dir=blocker / "impossible",
        pids=[],
        runner=_runner_ok,
    )

    assert bundle is None


def test_default_bundle_root_is_the_cluster_forensics_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("OMLX_CLUSTER_FORENSICS_DIR", raising=False)
    assert (
        forensics.default_forensics_dir("~/.omlx/cluster/runtime")
        .as_posix()
        .endswith("/.omlx/cluster/forensics")
    )
    monkeypatch.setenv("OMLX_CLUSTER_FORENSICS_DIR", str(tmp_path / "custom"))
    assert forensics.default_forensics_dir("ignored") == tmp_path / "custom"


def test_pgrep_discovers_sibling_rank_processes(monkeypatch):
    def runner(command, **kwargs):
        if command[0] == "pgrep":
            return SimpleNamespace(
                returncode=0, stdout=f"{os.getpid()}\n54321\n", stderr=""
            )
        return _runner_ok(command, **kwargs)

    pids = forensics._local_rank_pids(runner=runner)

    assert pids[0] == os.getpid()
    assert 54321 in pids
    assert len(pids) == len(set(pids))
