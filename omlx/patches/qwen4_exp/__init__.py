# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3.8 Flash Next registration for Fusion's pinned mlx-lm.

Qwen3.8 Flash Next identifies its text backbone as ``qwen4_exp`` and is not
shape-compatible with qwen3_5 or qwen3_next.  mlx-lm resolves models by
importing ``mlx_lm.models.<model_type>``; this package registers the bounded
Fusion implementation under that exact name when the pinned dependency does
not provide it.

The implementation is intentionally fail-closed at the QSA boundary.  Merely
being able to deserialize the checkpoint must never be mistaken for correct
generation with dense attention or a DeepSeek index cache.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_MODULE_NAME = "mlx_lm.models.qwen4_exp"
_APPLIED = False
_MODEL_DIR: Path | None = None


def _register_overlay() -> None:
    file_path = Path(__file__).parent / "model.py"
    if not file_path.exists():
        raise ImportError(f"Qwen4-Exp overlay is incomplete: missing {file_path}")

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {_MODULE_NAME} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.qwen4_exp = module
    except BaseException:
        if sys.modules.get(_MODULE_NAME) is module:
            sys.modules.pop(_MODULE_NAME, None)
        raise

    logger.info("Registered native Qwen4-Exp overlay from %s", file_path)


def set_model_dir(model_dir: str | Path | None) -> None:
    """Bind the checkpoint directory used by the SSD-backed PLE pool."""
    global _MODEL_DIR
    if model_dir is None:
        _MODEL_DIR = None
        return
    compute_dir = Path(model_dir).expanduser().resolve()
    ple_dir = compute_dir
    config_path = compute_dir / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid Qwen4-Exp config at {config_path}") from error
        artifact = config.get("qwen4_exp_artifact") or {}
        ple_artifact = artifact.get("ple_artifact")
        if ple_artifact is not None:
            if not isinstance(ple_artifact, str) or not ple_artifact:
                raise ValueError("qwen4_exp_artifact.ple_artifact must be a path")
            relative = Path(ple_artifact)
            if relative.is_absolute():
                raise ValueError("Qwen4-Exp PLE artifact path must be relative")
            ple_dir = (compute_dir / relative).resolve()
            artifact_root = compute_dir.parent.resolve()
            if ple_dir != artifact_root and artifact_root not in ple_dir.parents:
                raise ValueError("Qwen4-Exp PLE artifact escapes its artifact root")
    _MODEL_DIR = ple_dir


def get_model_dir() -> Path | None:
    return _MODEL_DIR


def apply_qwen4_exp_patch(model_dir: str | Path | None = None) -> bool:
    """Register the native Qwen4-Exp implementation, idempotently.

    A future mlx-lm version may ship a native implementation.  Such a module
    wins only when it explicitly advertises the same strict QSA contract;
    older or partial modules are replaced so Fusion never silently executes a
    semantically different attention path.
    """
    global _APPLIED
    if model_dir is not None:
        set_model_dir(model_dir)
    if _APPLIED:
        return False

    existing = sys.modules.get(_MODULE_NAME)
    if existing is None:
        try:
            existing = importlib.import_module(_MODULE_NAME)
        except ModuleNotFoundError as error:
            if error.name == "mlx_lm":
                logger.debug("mlx_lm not importable - qwen4_exp patch skipped")
                return False
            if error.name != _MODULE_NAME:
                raise

    if existing is None or not getattr(existing, "QWEN4_EXP_STRICT_QSA", False):
        _register_overlay()
        applied = True
    else:
        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.qwen4_exp = existing
        applied = False

    _APPLIED = True
    return applied


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "apply_qwen4_exp_patch",
    "get_model_dir",
    "is_applied",
    "set_model_dir",
]
