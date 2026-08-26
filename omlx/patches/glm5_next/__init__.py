"""Strict GLM5-Next contract, conversion, and mlx-lm registration.

GLM-5.3-Flash is never registered as ``glm_moe_dsa``. The bounded overlay is
published only at ``mlx_lm.models.glm5_next`` and advertises separately whether
its exact runtime prerequisites are complete.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

from .contract import (
    OFFICIAL_REVISION,
    Glm5NextContractError,
    Glm5NextSourceContract,
    validate_config,
    validate_source_contract,
    validate_weight_index,
)
from .convert import (
    Glm5NextConversionPlan,
    Glm5NextUnsupportedMathError,
    build_conversion_plan,
    convert_glm53_flash,
)

logger = logging.getLogger(__name__)

_MODULE_NAME = "mlx_lm.models.glm5_next"
_VLM_MODULE_NAME = "mlx_vlm.models.glm5_next"
_APPLIED = False
_VLM_APPLIED = False
# The native tower/processor can be ready before oMLX has a complete mlx-vlm
# engine adapter. Discovery must not route media traffic until both halves are
# affirmative or it would silently fall back to text and drop the media.
GLM5_NEXT_VLM_ADAPTER_READY = True


def _register_overlay() -> None:
    file_path = Path(__file__).parent / "model.py"
    if not file_path.is_file():
        raise ImportError(f"GLM5-Next overlay is incomplete: missing {file_path}")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {_MODULE_NAME}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        if not getattr(module, "GLM5_NEXT_STRICT_GRAPH", False):
            raise ImportError("GLM5-Next overlay does not advertise its strict graph")
        models_package = importlib.import_module("mlx_lm.models")
        models_package.glm5_next = module
    except BaseException:
        if sys.modules.get(_MODULE_NAME) is module:
            sys.modules.pop(_MODULE_NAME, None)
        raise


def apply_glm5_next_patch() -> bool:
    """Register only the strict ``glm5_next`` architecture, idempotently."""

    global _APPLIED
    if _APPLIED:
        return False
    existing = sys.modules.get(_MODULE_NAME)
    if existing is None:
        try:
            existing = importlib.import_module(_MODULE_NAME)
        except ModuleNotFoundError as error:
            if error.name == "mlx_lm":
                logger.debug("mlx_lm not importable - glm5_next patch skipped")
                return False
            if error.name != _MODULE_NAME:
                raise
    if existing is None or not getattr(existing, "GLM5_NEXT_STRICT_GRAPH", False):
        _register_overlay()
        applied = True
    else:
        models_package = importlib.import_module("mlx_lm.models")
        models_package.glm5_next = existing
        applied = False
    _APPLIED = True
    from .cache_handlers import register_glm5_next_cache_handlers

    register_glm5_next_cache_handlers()
    return applied


def _register_vlm_overlay() -> None:
    file_path = Path(__file__).parent / "vlm.py"
    if not file_path.is_file():
        raise ImportError(f"GLM5-Next VLM overlay is incomplete: missing {file_path}")
    spec = importlib.util.spec_from_file_location(
        _VLM_MODULE_NAME,
        str(file_path),
        submodule_search_locations=[],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {_VLM_MODULE_NAME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_VLM_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        if not getattr(module, "GLM5_NEXT_NATIVE_VLM", False):
            raise ImportError("GLM5-Next overlay does not advertise native VLM support")
        module.install_mlx_vlm_submodules(module)
        module.install_mlx_format_sanitize_patch()
        module.install_prompt_format_patch()
        models_package = importlib.import_module("mlx_vlm.models")
        models_package.glm5_next = module
    except BaseException:
        if sys.modules.get(_VLM_MODULE_NAME) is module:
            sys.modules.pop(_VLM_MODULE_NAME, None)
        for name in tuple(sys.modules):
            if name.startswith(_VLM_MODULE_NAME + "."):
                sys.modules.pop(name, None)
        raise


def apply_glm5_next_vlm_patch() -> bool:
    """Register the exact native GLM5-Next mlx-vlm package, idempotently."""

    global _VLM_APPLIED
    if _VLM_APPLIED:
        return False
    existing = sys.modules.get(_VLM_MODULE_NAME)
    if existing is None:
        try:
            existing = importlib.import_module(_VLM_MODULE_NAME)
        except ModuleNotFoundError as error:
            if error.name == "mlx_vlm":
                logger.debug("mlx_vlm not importable - glm5_next VLM patch skipped")
                return False
            if error.name != _VLM_MODULE_NAME:
                raise
    if existing is None or not getattr(existing, "GLM5_NEXT_NATIVE_VLM", False):
        _register_vlm_overlay()
        applied = True
    else:
        existing.install_mlx_vlm_submodules(existing)
        existing.install_mlx_format_sanitize_patch()
        existing.install_prompt_format_patch()
        models_package = importlib.import_module("mlx_vlm.models")
        models_package.glm5_next = existing
        applied = False
    from .cache_handlers import register_glm5_next_cache_handlers
    from .processor import install_glm5_next_processor_namespace

    install_glm5_next_processor_namespace()
    register_glm5_next_cache_handlers()
    _VLM_APPLIED = True
    return applied


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "OFFICIAL_REVISION",
    "Glm5NextContractError",
    "Glm5NextConversionPlan",
    "Glm5NextSourceContract",
    "Glm5NextUnsupportedMathError",
    "GLM5_NEXT_VLM_ADAPTER_READY",
    "apply_glm5_next_patch",
    "apply_glm5_next_vlm_patch",
    "build_conversion_plan",
    "convert_glm53_flash",
    "is_applied",
    "validate_config",
    "validate_source_contract",
    "validate_weight_index",
]
