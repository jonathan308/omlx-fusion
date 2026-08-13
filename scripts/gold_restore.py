#!/usr/bin/env python3
"""Gold restore points for oMLX — the certified-known-good rollback.

Adapted from ThunderMLX ``M3_Restore_Gold.command``, which reverted a
git-checkout app to a certified tag plus its ``.env.local`` and re-synced the
second Mac's code. oMLX is not a git checkout on the machines it serves, so
the pieces map differently:

- code revert  -> the installed ``omlx`` package version recorded in the
  manifest. Restoring prints the exact ``pip install "omlx==<version>"``
  command when the running install differs; it never runs pip itself,
  because oMLX may be a source checkout or app bundle where a pip downgrade
  would be wrong.
- .env.local   -> ``settings.json`` + ``model_settings.json`` + the cluster
  deployment registry (``cluster/deployments.json``), snapshotted byte-for-byte
  with SHA-256 digests so a partial copy cannot be restored silently.
- rank-1 sync  -> nothing to port: every oMLX node runs its own install;
  version parity is each node restoring the same point.

What is deliberately preserved, exactly as ThunderMLX kept prompt caches and
lifetime stats: ``cache/`` (SSD prompt cache), ``models/``, logs, and the
supervision restart budget are never touched — only configuration comes from
the restore point.

Usage:

    python scripts/gold_restore.py create --name gold-2026-08-13
    python scripts/gold_restore.py list
    python scripts/gold_restore.py restore --name gold-2026-08-13 [--dry-run]

A restore rewrites config files atomically; restart oMLX afterwards to apply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

# Configuration state covered by a restore point, relative to the base path.
# Runtime state (runtime markers, restart budgets, logs) is excluded on
# purpose: it describes a moment, not a configuration.
SNAPSHOT_FILES = (
    "settings.json",
    "model_settings.json",
    "cluster/deployments.json",
)

_RESTORE_POINTS_DIR = "restore-points"


def _default_base_path() -> Path:
    """OMLX_BASE_PATH first, then ~/.omlx.

    The macOS app can relocate the data root (its bootstrap file is app
    state, not oMLX state); operators with a moved root pass ``--base-path``.
    """

    env_value = os.environ.get("OMLX_BASE_PATH", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path("~/.omlx").expanduser().resolve()


def _omlx_version() -> str:
    try:
        return package_version("omlx")
    except PackageNotFoundError:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _points_root(base_path: Path) -> Path:
    return base_path / _RESTORE_POINTS_DIR


def _validate_name(name: str) -> str:
    """Restore point names are directory names; keep them boring."""

    cleaned = name.strip()
    if (
        not cleaned
        or len(cleaned) > 64
        or cleaned in {".", ".."}
        or any(char not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_" for char in cleaned)
    ):
        raise ValueError(
            "restore point names may contain only letters, digits, '-' and '_'"
        )
    return cleaned


def create_restore_point(base_path: Path, name: str, *, force: bool = False) -> Path:
    """Snapshot the current configuration as a named restore point."""

    name = _validate_name(name)
    target = _points_root(base_path) / name
    if target.exists() and not force:
        raise FileExistsError(
            f"restore point {name!r} already exists; gold points are immutable "
            f"(use --force only when re-certifying under the same name)"
        )
    files: dict[str, dict[str, Any]] = {}
    staged: list[tuple[Path, Path]] = []
    for relative in SNAPSHOT_FILES:
        source = base_path / relative
        if not source.is_file():
            continue
        staged.append((source, target / relative))
        files[relative] = {
            "sha256": _sha256(source),
            "bytes": source.stat().st_size,
        }
    if not staged:
        raise FileNotFoundError(
            f"no oMLX configuration found under {base_path} — nothing to snapshot"
        )
    target.mkdir(parents=True, exist_ok=True)
    try:
        for source, destination in staged:
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = source.read_bytes()
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
        manifest = {
            "schema_version": 1,
            "name": name,
            "created_at": datetime.now(UTC).isoformat(),
            "hostname": socket.gethostname(),
            "omlx_version": _omlx_version(),
            "files": files,
        }
        manifest_path = target / "manifest.json"
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    except OSError:
        # A half-written point must never look certified.
        for _source, destination in staged:
            destination.unlink(missing_ok=True)
        (target / "manifest.json").unlink(missing_ok=True)
        raise
    return target


def _load_manifest(point: Path) -> dict[str, Any]:
    manifest_path = point / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"restore point {point.name!r} has no readable manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"restore point {point.name!r} has an unsupported manifest")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError(f"restore point {point.name!r} carries no files")
    return manifest


def list_restore_points(base_path: Path) -> list[dict[str, Any]]:
    """Every restore point with its manifest summary, oldest first."""

    root = _points_root(base_path)
    points: list[dict[str, Any]] = []
    if not root.is_dir():
        return points
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if not entry.is_dir():
            continue
        try:
            manifest = _load_manifest(entry)
        except ValueError as exc:
            points.append({"name": entry.name, "error": str(exc)})
            continue
        points.append(
            {
                "name": entry.name,
                "created_at": manifest.get("created_at"),
                "hostname": manifest.get("hostname"),
                "omlx_version": manifest.get("omlx_version"),
                "files": sorted(manifest["files"]),
            }
        )
    return points


def restore_gold(
    base_path: Path,
    name: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Restore one point's configuration; never touches caches or models.

    Returns the plan/result: restored files, the recorded oMLX version, and
    whether the running install matches it. ``dry_run`` verifies the point
    and reports without writing.
    """

    name = _validate_name(name)
    point = _points_root(base_path) / name
    if not point.is_dir():
        raise FileNotFoundError(f"no restore point named {name!r} under {point.parent}")
    manifest = _load_manifest(point)
    planned: list[str] = []
    restored: list[str] = []
    for relative, recorded in sorted(manifest["files"].items()):
        source = point / relative
        if not source.is_file():
            raise FileNotFoundError(
                f"restore point {name!r} is missing {relative} — refusing a partial restore"
            )
        if _sha256(source) != recorded.get("sha256"):
            raise ValueError(
                f"restore point {name!r} failed verification: {relative} does not "
                f"match its manifest digest"
            )
        planned.append(relative)
        if dry_run:
            continue
        destination = base_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(source.read_bytes())
        # Config files hold the API key; keep the server's own permissions.
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        restored.append(relative)
    recorded_version = str(manifest.get("omlx_version") or "unknown")
    current_version = _omlx_version()
    return {
        "name": name,
        "dry_run": dry_run,
        "planned": planned,
        "restored": restored,
        "recorded_omlx_version": recorded_version,
        "current_omlx_version": current_version,
        "version_matches": recorded_version in (current_version, "unknown"),
        "created_at": manifest.get("created_at"),
    }


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="oMLX data root (default: OMLX_BASE_PATH or ~/.omlx)",
    )
    command = parser.add_subparsers(dest="command", required=True)
    create = command.add_parser("create", help="snapshot the current configuration")
    create.add_argument("--name", required=True)
    create.add_argument("--force", action="store_true", help="overwrite an existing point")
    listing = command.add_parser("list", help="show restore points")
    restore = command.add_parser("restore", help="restore a point's configuration")
    restore.add_argument("--name", required=True)
    restore.add_argument("--dry-run", action="store_true", help="verify and report only")
    restore.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    del listing  # no options
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    base_path = (
        args.base_path.expanduser().resolve()
        if args.base_path is not None
        else _default_base_path()
    )
    try:
        if args.command == "create":
            point = create_restore_point(base_path, args.name, force=args.force)
            print(f"Created restore point {args.name!r} at {point}")
            print(f"Certified oMLX version: {_omlx_version()}")
            return 0
        if args.command == "list":
            points = list_restore_points(base_path)
            if not points:
                print(f"No restore points under {_points_root(base_path)}")
                return 0
            for point in points:
                if "error" in point:
                    print(f"  {point['name']}: INVALID ({point['error']})")
                    continue
                print(
                    f"  {point['name']}: {point['created_at']} "
                    f"on {point['hostname']} · oMLX {point['omlx_version']} · "
                    f"{len(point['files'])} files"
                )
            return 0
        # restore
        result = restore_gold(base_path, args.name, dry_run=args.dry_run)
    except (OSError, ValueError) as exc:
        print(f"gold-restore: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"Restore point {args.name!r} verified; would restore:")
        for relative in result["planned"]:
            print(f"  {relative}")
    else:
        if not args.yes:
            if not sys.stdin.isatty():
                print(
                    "gold-restore: refusing to restore without --yes when not "
                    "on a terminal (or pass --dry-run)",
                    file=sys.stderr,
                )
                return 1
            answer = input(
                f"Restore {args.name!r} over the configuration in {base_path}? "
                f"Caches, models and stats are kept. [y/N] "
            )
            if answer.strip().lower() != "y":
                print("Cancelled.")
                return 0
            try:
                result = restore_gold(base_path, args.name, dry_run=False)
            except (OSError, ValueError) as exc:
                print(f"gold-restore: {exc}", file=sys.stderr)
                return 1
        print(f"Restored {args.name!r}:")
        for relative in result["restored"]:
            print(f"  {relative}")
    if not result["version_matches"]:
        print(
            f"NOTE: this point was certified on oMLX "
            f"{result['recorded_omlx_version']}, but the installed package is "
            f"{result['current_omlx_version']}. Reinstall to match:\n"
            f'  pip install "omlx=={result["recorded_omlx_version"]}"'
        )
    if not args.dry_run and result["restored"]:
        print("Restart oMLX to apply the restored configuration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
