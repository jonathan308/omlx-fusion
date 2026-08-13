# SPDX-License-Identifier: Apache-2.0
"""Gold restore points: snapshot, verify, restore — never caches or models."""

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script(name):
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gold_restore = _load_script("gold_restore")


@pytest.fixture()
def base(tmp_path):
    (tmp_path / "cluster").mkdir()
    (tmp_path / "settings.json").write_text(
        json.dumps({"auth": {"api_key": "k"}, "server": {"port": 8000}})
    )
    (tmp_path / "model_settings.json").write_text(json.dumps({"models": {}}))
    (tmp_path / "cluster" / "deployments.json").write_text(
        json.dumps({"schema_version": 1, "deployments": []})
    )
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "prompt-kv").write_text("precious")
    return tmp_path


def test_create_snapshot_list_restore_round_trip(base):
    gold_restore.create_restore_point(base, "gold-v1")

    points = gold_restore.list_restore_points(base)
    assert len(points) == 1
    assert points[0]["name"] == "gold-v1"
    assert sorted(points[0]["files"]) == [
        "cluster/deployments.json",
        "model_settings.json",
        "settings.json",
    ]

    (base / "settings.json").write_text(json.dumps({"broken": True}))
    result = gold_restore.restore_gold(base, "gold-v1")
    assert result["restored"] == [
        "cluster/deployments.json",
        "model_settings.json",
        "settings.json",
    ]
    assert json.loads((base / "settings.json").read_text())["auth"]["api_key"] == "k"
    # The prompt cache survives, exactly as ThunderMLX preserved its caches.
    assert (base / "cache" / "prompt-kv").read_text() == "precious"


def test_create_refuses_to_overwrite_without_force(base):
    gold_restore.create_restore_point(base, "gold-v1")
    with pytest.raises(FileExistsError, match="immutable"):
        gold_restore.create_restore_point(base, "gold-v1")
    gold_restore.create_restore_point(base, "gold-v1", force=True)


def test_create_fails_when_there_is_nothing_to_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing to snapshot"):
        gold_restore.create_restore_point(tmp_path, "gold-v1")


def test_names_cannot_escape_the_restore_points_dir(base):
    for name in ("../etc", "a/b", "", "name with spaces"):
        with pytest.raises(ValueError, match="letters, digits"):
            gold_restore.create_restore_point(base, name)


def test_restore_refuses_a_tampered_point(base):
    point = gold_restore.create_restore_point(base, "gold-v1")
    (point / "settings.json").write_text(json.dumps({"evil": True}))

    with pytest.raises(ValueError, match="does not match its manifest digest"):
        gold_restore.restore_gold(base, "gold-v1")
    # Nothing was written.
    assert json.loads((base / "settings.json").read_text())["auth"]["api_key"] == "k"


def test_restore_refuses_a_partial_point(base):
    point = gold_restore.create_restore_point(base, "gold-v1")
    (point / "model_settings.json").unlink()

    with pytest.raises(FileNotFoundError, match="partial restore"):
        gold_restore.restore_gold(base, "gold-v1")


def test_restore_of_a_missing_point_is_a_stated_error(base):
    with pytest.raises(FileNotFoundError, match="no restore point"):
        gold_restore.restore_gold(base, "nope")


def test_dry_run_verifies_without_writing(base):
    gold_restore.create_restore_point(base, "gold-v1")
    (base / "settings.json").write_text(json.dumps({"broken": True}))

    result = gold_restore.restore_gold(base, "gold-v1", dry_run=True)

    assert result["dry_run"] is True
    assert result["restored"] == []
    assert len(result["planned"]) == 3
    assert json.loads((base / "settings.json").read_text()) == {"broken": True}


def test_a_point_with_only_some_config_files_restores_just_those(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"a": 1}))
    gold_restore.create_restore_point(tmp_path, "minimal")

    (tmp_path / "settings.json").write_text(json.dumps({"a": 2}))
    result = gold_restore.restore_gold(tmp_path, "minimal")

    assert result["restored"] == ["settings.json"]
    assert json.loads((tmp_path / "settings.json").read_text()) == {"a": 1}


def test_list_marks_a_broken_point_instead_of_dying(base):
    gold_restore.create_restore_point(base, "good")
    broken = base / "restore-points" / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("not json")

    points = gold_restore.list_restore_points(base)
    by_name = {point["name"]: point for point in points}
    assert "error" in by_name["broken"]
    assert by_name["good"]["files"]


def test_main_create_list_restore(base, capsys):
    argv = ["--base-path", str(base)]
    assert gold_restore.main([*argv, "create", "--name", "gold-v1"]) == 0
    assert gold_restore.main([*argv, "list"]) == 0
    assert "gold-v1" in capsys.readouterr().out

    (base / "settings.json").write_text(json.dumps({"broken": True}))
    assert gold_restore.main([*argv, "restore", "--name", "gold-v1", "--dry-run"]) == 0
    assert json.loads((base / "settings.json").read_text()) == {"broken": True}

    assert gold_restore.main([*argv, "restore", "--name", "gold-v1", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "Restored 'gold-v1'" in out
    assert "Restart oMLX" in out
    assert json.loads((base / "settings.json").read_text())["auth"]["api_key"] == "k"


def test_main_restore_needs_yes_off_terminal(base, monkeypatch, capsys):
    gold_restore.create_restore_point(base, "gold-v1")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    rc = gold_restore.main(
        ["--base-path", str(base), "restore", "--name", "gold-v1"]
    )

    assert rc == 1
    assert "--yes" in capsys.readouterr().err


def test_main_reports_errors_on_stderr(base, capsys):
    rc = gold_restore.main(
        ["--base-path", str(base), "restore", "--name", "missing", "--yes"]
    )
    assert rc == 1
    assert "no restore point" in capsys.readouterr().err
