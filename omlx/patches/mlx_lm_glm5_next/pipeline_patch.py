# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel execution contract for GLM-5.3's text backbone."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from omlx.cluster.pipeline_compat import _cache_dependency

logger = logging.getLogger(__name__)

_ORIGINAL_MODEL_CALL: Any | None = None
_ORIGINAL_MAKE_CACHE: Any | None = None


def _send_anchor(cache_entry: Any, value: mx.array) -> None:
    """Anchor a lazy stage send in either recurrent or composite cache state."""

    if cache_entry is None:
        return
    # GLM sparse layers use CacheList(KVCache, PoolingCache); dependency belongs
    # on the actual KV tensor, not on the CacheList's KVCache object.
    try:
        first = cache_entry[0]
    except (IndexError, KeyError, TypeError):
        first = None
    if first is not None and hasattr(first, "keys"):
        cache_entry = first
    _cache_dependency(cache_entry, value, mx)


def _stage_make_cache(self: Any) -> list[Any]:
    """Build caches in local stage order from the layers this rank owns."""

    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, PoolingCache

    layers = self.model.pipeline_layers
    if not layers or any(layer is None for layer in layers):
        raise RuntimeError("GLM-5.3 pipeline stage has no complete local layer slice")
    caches: list[Any] = []
    for layer in layers:
        if layer.is_linear:
            caches.append(ArraysCache(size=2))
        else:
            caches.append(
                CacheList(
                    KVCache(),
                    PoolingCache(layer.self_attn.indexer.index_kpool),
                )
            )
    return caches


def _pipelined_call(
    self: Any,
    inputs: mx.array,
    cache: list[Any] | None = None,
    inputs_embeds: mx.array | None = None,
) -> mx.array:
    """Run only this rank's contiguous GLM layers and exchange MHC state."""

    from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

    base_hidden = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
    layers = self.pipeline_layers
    if not layers or any(layer is None for layer in layers):
        raise RuntimeError("GLM-5.3 pipeline stage contains an unbuilt layer")
    if cache is None:
        cache = [None] * len(layers)
    if len(cache) != len(layers):
        raise ValueError("GLM-5.3 pipeline cache/layer count mismatch")

    linear_index = next(
        (index for index, layer in enumerate(layers) if layer.is_linear), None
    )
    sparse_index = next(
        (index for index, layer in enumerate(layers) if not layer.is_linear), None
    )
    ssm_mask = (
        create_ssm_mask(base_hidden, cache[linear_index])
        if linear_index is not None
        else None
    )
    if sparse_index is not None:
        sparse_cache = cache[sparse_index]
        try:
            attention_cache = sparse_cache[0]
        except (IndexError, KeyError, TypeError):
            attention_cache = sparse_cache
        fa_mask = create_attention_mask(
            base_hidden,
            attention_cache,
            return_array=True,
        )
    else:
        fa_mask = None

    hidden = mx.contiguous(
        mx.broadcast_to(
            base_hidden[:, :, None, :],
            (
                base_hidden.shape[0],
                base_hidden.shape[1],
                self.hc_mult,
                base_hidden.shape[2],
            ),
        )
    )
    pipeline_rank = int(getattr(self, "pipeline_rank", 0))
    pipeline_size = int(getattr(self, "pipeline_size", 1))
    if pipeline_rank < pipeline_size - 1:
        hidden = mx.distributed.recv_like(hidden, pipeline_rank + 1)

    for layer, layer_cache in zip(layers, cache, strict=True):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=layer_cache)

    if pipeline_rank != 0:
        hidden = mx.distributed.send(hidden, (pipeline_rank - 1) % pipeline_size)
        if cache:
            _send_anchor(cache[-1], hidden)

    # Keep the pinned mlx-lm collective contract: every rank returns the final
    # rank-zero hidden state. Runtime coordinator-sampling may later replace
    # this one gather with a token collective after validating this source.
    if pipeline_size > 1:
        hidden = mx.distributed.all_gather(hidden)[: hidden.shape[0]]

    return self.norm(hidden.mean(axis=2))


def apply_glm5_pipeline_patch() -> bool:
    """Install pipeline and stage-cache methods on the vendored GLM classes."""

    global _ORIGINAL_MAKE_CACHE, _ORIGINAL_MODEL_CALL

    try:
        from mlx_lm.models.pipeline import PipelineMixin
        from mlx_vlm.models.glm5_next.language import Glm5NextModel, LanguageModel
    except (ImportError, AttributeError) as exc:
        logger.warning("Cannot pipeline GLM-5.3: %s", exc)
        return False
    if getattr(Glm5NextModel, "_omlx_pipelined", False):
        return True

    _ORIGINAL_MODEL_CALL = Glm5NextModel.__call__
    _ORIGINAL_MAKE_CACHE = LanguageModel.make_cache

    def pipeline(self: Any, group: Any) -> None:
        # Resolve dynamically so Cluster v2's temporary unequal-plan override
        # on PipelineMixin.pipeline is honored.
        PipelineMixin.pipeline(self, group)
        logger.info(
            "GLM-5.3 rank %s holds layers %s-%s (%s layers)",
            self.pipeline_rank,
            self.start_idx,
            self.end_idx,
            len(self.pipeline_layers),
        )

    pipeline._omlx_honors_pipeline_assignment = True
    Glm5NextModel.pipeline = pipeline
    Glm5NextModel.pipeline_layers = PipelineMixin.pipeline_layers
    Glm5NextModel.__call__ = _pipelined_call
    Glm5NextModel.pipeline_rank = 0
    Glm5NextModel.pipeline_size = 1
    Glm5NextModel.start_idx = 0
    Glm5NextModel.end_idx = None
    Glm5NextModel._omlx_pipelined = True
    LanguageModel.make_cache = _stage_make_cache
    logger.info("GLM-5.3 pipelining installed")
    return True


__all__ = ["apply_glm5_pipeline_patch"]
