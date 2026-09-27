# SPDX-License-Identifier: Apache-2.0
"""Per-phase Metal command-buffer budget (needs mlx with set_command_buffer_limits).

mlx commits a command buffer once its encoded inputs exceed max_mb_per_buffer
(50 MB on Ultra chips) and counts every input's full size, including multi-GB
expert weight tensors of which a decode step reads a few experts. Decode steps
of MoE models therefore commit after nearly every weight-reading op (GLM-5.3:
~170 command buffers per token). Forwards of at most 8 tokens (decode and
speculative verify) now encode with a large MB budget, so buffers are bounded
by the op count only; prefill keeps the default, whose purpose (bounding the
temporaries a command buffer keeps alive) matters there. Scheduling only: the
math is unchanged. Without the mlx setter this is a no-op.
OMLX_DECODE_CBUF_BUDGET=0 disables it.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_DECODE_MAX_ROWS = 8
_DECODE_MB = 1_000_000
_default_mb: int | None = None
_wrapped: set[type] = set()


def _available() -> bool:
    if os.environ.get("OMLX_DECODE_CBUF_BUDGET", "1").strip().lower() in {"0", "false", "off"}:
        return False
    return hasattr(mx.metal, "set_command_buffer_limits")


def _wrap(cls: type) -> None:
    if cls in _wrapped or not callable(getattr(cls, "__call__", None)):
        return
    original = cls.__call__

    def __call__(self, inputs, *args, **kwargs):
        rows = inputs.shape[1] if getattr(inputs, "ndim", 0) >= 2 else 1
        mx.metal.set_command_buffer_limits(
            0, _DECODE_MB if rows <= _DECODE_MAX_ROWS else _default_mb
        )
        return original(self, inputs, *args, **kwargs)

    cls.__call__ = __call__
    _wrapped.add(cls)


def apply(model: Any) -> bool:
    """Wrap the model's (and its language model's) forward. Idempotent."""
    global _default_mb
    if model is None or not _available():
        return False
    if _default_mb is None:
        _default_mb = mx.metal.set_command_buffer_limits(0, 0)[1]
    for obj in (model, getattr(model, "language_model", None)):
        if obj is not None:
            _wrap(type(obj))
    logger.info("decode command-buffer budget enabled (prefill keeps %d MB)", _default_mb)
    return True
