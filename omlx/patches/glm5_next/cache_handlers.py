# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache handlers for the native GLM5-Next cache ABI."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from omlx.cache.type_handlers import (
    CacheStateAxisInfo,
    CacheType,
    CacheTypeHandler,
)
from omlx.cache.type_registry import CacheTypeRegistry


class Glm5NextKDACacheHandler(CacheTypeHandler):
    """Preserve all four recurrent KDA states at block boundaries."""

    @property
    def cache_type(self) -> CacheType:
        return CacheType.GLM5_NEXT_KDA_CACHE

    @property
    def supports_block_slicing(self) -> bool:
        return False

    def get_state_axis_info(self) -> tuple[CacheStateAxisInfo, ...]:
        return tuple(
            CacheStateAxisInfo(name, sequence_axis=None, sliceable=False)
            for name in ("q_conv", "k_conv", "v_conv", "recurrent")
        )

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        return {
            "states": tuple(cache_obj.state),
            "token_count": int(cache_obj.size()),
            "cache_type": self.cache_type.value,
            "is_full_state": True,
        }

    def get_seq_len(self, state: dict[str, Any]) -> int:
        return int(state.get("token_count", 0))

    def slice_state(
        self, state: dict[str, Any], start_idx: int, end_idx: int
    ) -> dict[str, Any] | None:
        del start_idx, end_idx
        return state

    def concatenate_states(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        return states[-1] if states else {}

    def deserialize_state(
        self, elements: tuple[Any, ...], meta_state: Any | None = None
    ) -> Any:
        if len(elements) != 4:
            return None
        from omlx.patches.glm5_next.model import Glm5NextKDACache

        cache = Glm5NextKDACache()
        cache.state = elements
        if meta_state:
            cache.meta_state = tuple(meta_state)
        return cache

    def reconstruct_cache(
        self,
        state: dict[str, Any],
        meta_state: tuple | None = None,
        **_kwargs: Any,
    ) -> Any:
        return self.deserialize_state(tuple(state.get("states", ())), meta_state)

    def _get_state_keys(self) -> tuple[str, ...]:
        return ("q_conv", "k_conv", "v_conv", "recurrent")

    def _get_meta_state_keys(self) -> tuple[str, ...]:
        return ("offset",)


class Glm5NextDsaCacheHandler(CacheTypeHandler):
    """Slice the latent and packed indexer streams along token axis one."""

    @property
    def cache_type(self) -> CacheType:
        return CacheType.GLM5_NEXT_DSA_CACHE

    @property
    def supports_block_slicing(self) -> bool:
        return True

    def get_state_axis_info(self) -> tuple[CacheStateAxisInfo, ...]:
        return (
            CacheStateAxisInfo("kv_latent", sequence_axis=1, sliceable=True),
            CacheStateAxisInfo("indexer_states", sequence_axis=1, sliceable=True),
        )

    def extract_state(self, cache_obj: Any) -> dict[str, Any]:
        kv_latent, indexer_states = cache_obj.state
        return {
            "states": (kv_latent, indexer_states),
            "kv_latent": kv_latent,
            "indexer_states": indexer_states,
            "cache_type": self.cache_type.value,
        }

    def get_seq_len(self, state: dict[str, Any]) -> int:
        latent = state.get("kv_latent")
        return int(latent.shape[1]) if latent is not None else 0

    def slice_state(
        self, state: dict[str, Any], start_idx: int, end_idx: int
    ) -> dict[str, Any] | None:
        latent = state.get("kv_latent")
        packed = state.get("indexer_states")
        if latent is None or packed is None:
            return None
        return {
            "states": (
                latent[:, start_idx:end_idx],
                packed[:, start_idx:end_idx],
            ),
            "kv_latent": latent[:, start_idx:end_idx],
            "indexer_states": packed[:, start_idx:end_idx],
            "cache_type": self.cache_type.value,
        }

    def concatenate_states(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        if not states:
            return {}

        def elements(state: dict[str, Any]) -> tuple[Any, Any]:
            if "kv_latent" in state and "indexer_states" in state:
                return state["kv_latent"], state["indexer_states"]
            if "states" in state:
                values = tuple(state["states"])
                if len(values) == 2:
                    return values
            # Legacy two-tensor block payloads are collected by the generic
            # prefix-cache loop under keys/values.  Their tensors are still
            # exact DSA latent/indexer slices; preserve them under the native
            # cache class instead of rejecting a mixed old/new block chain.
            return state["keys"], state["values"]

        pairs = [elements(state) for state in states]
        latents = [pair[0] for pair in pairs]
        packed = [pair[1] for pair in pairs]
        latent = mx.concatenate(latents, axis=1)
        indexer = mx.concatenate(packed, axis=1)
        return {
            "states": (latent, indexer),
            "kv_latent": latent,
            "indexer_states": indexer,
            "cache_type": self.cache_type.value,
        }

    def deserialize_state(
        self, elements: tuple[Any, ...], meta_state: Any | None = None
    ) -> Any:
        if len(elements) != 2 or any(element is None for element in elements):
            return None
        from omlx.patches.glm5_next.dsa import Glm5NextDsaCache

        return Glm5NextDsaCache.from_state(elements, meta_state or ())

    def reconstruct_cache(
        self,
        state: dict[str, Any],
        meta_state: tuple | None = None,
        **_kwargs: Any,
    ) -> Any:
        elements = state.get("states")
        if elements is None:
            elements = (state.get("kv_latent"), state.get("indexer_states"))
        return self.deserialize_state(tuple(elements), meta_state)

    def _get_state_keys(self) -> tuple[str, ...]:
        return ("kv_latent", "indexer_states")

    def _get_meta_state_keys(self) -> tuple[str, ...]:
        return ("offset", "left_padding", "kv_width", "index_width")


def register_glm5_next_cache_handlers() -> None:
    """Install exact handlers idempotently for storage and reconstruction."""

    CacheTypeRegistry.register_class_name(
        "Glm5NextKDACache",
        CacheType.GLM5_NEXT_KDA_CACHE,
        Glm5NextKDACacheHandler(),
    )
    CacheTypeRegistry.register_class_name(
        "Glm5NextDsaCache",
        CacheType.GLM5_NEXT_DSA_CACHE,
        Glm5NextDsaCacheHandler(),
    )


__all__ = [
    "Glm5NextDsaCacheHandler",
    "Glm5NextKDACacheHandler",
    "register_glm5_next_cache_handlers",
]
