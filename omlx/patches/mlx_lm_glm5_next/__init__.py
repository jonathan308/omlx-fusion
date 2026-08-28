# SPDX-License-Identifier: Apache-2.0
"""Register oMLX's native GLM-5.3 text backbone with mlx-lm.

Single-node GLM-5.3 is served by the vendored mlx-vlm implementation. Cluster
ranks are mlx-lm workers, so they need the same implementation presented under
``mlx_lm.models.glm5_next``. This adapter is text-only by design; multimodal
cluster serving needs a separate vision-feature handoff contract.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_QUALNAME = "mlx_lm.models.glm5_next"


def _register_module() -> None:
    if _QUALNAME in sys.modules:
        return
    file_path = Path(__file__).parent / "glm5_next_model.py"
    spec = importlib.util.spec_from_file_location(_QUALNAME, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {_QUALNAME} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[_QUALNAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_QUALNAME, None)
        raise
    logger.info("Registered %s from %s", _QUALNAME, file_path.name)


def apply_mlx_lm_glm5_next_patch() -> bool:
    """Make ``glm5_next`` loadable by mlx-lm. Safe to call repeatedly."""

    try:
        _register_module()
    except Exception as exc:
        logger.warning("Could not register %s: %s", _QUALNAME, exc)
        return False
    return True


__all__ = ["apply_mlx_lm_glm5_next_patch"]
