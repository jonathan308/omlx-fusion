# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mlx.core as mx
import numpy as np

from omlx.cache.type_handlers import CacheType
from omlx.cache.type_registry import CacheTypeRegistry
from omlx.patches.glm5_next.cache_handlers import (
    Glm5NextDsaCacheHandler,
    Glm5NextKDACacheHandler,
    register_glm5_next_cache_handlers,
)
from omlx.patches.glm5_next.dsa import Glm5NextDsaCache, Glm5NextDsaConfig
from omlx.patches.glm5_next.model import Glm5NextKDACache


def _tiny_dsa_config() -> Glm5NextDsaConfig:
    return Glm5NextDsaConfig(
        hidden_size=8,
        num_attention_heads=2,
        q_lora_rank=4,
        kv_lora_rank=3,
        qk_nope_head_dim=2,
        v_head_dim=2,
        index_n_heads=2,
        index_head_dim=3,
        index_topk=4,
        index_kpool=2,
    )


def test_registry_resolves_exact_glm_cache_handlers():
    register_glm5_next_cache_handlers()
    assert CacheTypeRegistry.detect_cache_type(Glm5NextKDACache()) == (
        CacheType.GLM5_NEXT_KDA_CACHE
    )
    assert CacheTypeRegistry.detect_cache_type(Glm5NextDsaCache()) == (
        CacheType.GLM5_NEXT_DSA_CACHE
    )
    assert isinstance(
        CacheTypeRegistry.get_handler_by_class_name("Glm5NextKDACache"),
        Glm5NextKDACacheHandler,
    )
    assert isinstance(
        CacheTypeRegistry.get_handler_by_class_name("Glm5NextDsaCache"),
        Glm5NextDsaCacheHandler,
    )


def test_kda_handler_preserves_four_state_recurrent_abi():
    cache = Glm5NextKDACache()
    cache.state = (
        mx.arange(24).reshape(1, 3, 8),
        mx.arange(24, 48).reshape(1, 3, 8),
        mx.arange(48, 72).reshape(1, 3, 8),
        mx.arange(32).reshape(1, 2, 4, 4),
    )
    cache.offset = 17
    handler = Glm5NextKDACacheHandler()
    elements = handler.serialize_state(cache)
    restored = handler.deserialize_state(elements, cache.meta_state)
    assert isinstance(restored, Glm5NextKDACache)
    assert restored.offset == 17
    assert len(restored.state) == 4
    for expected, actual in zip(cache.state, restored.state):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    # Legacy malformed two-state payloads are rejected rather than restored as
    # a plain KVCache and allowed to fail inside the model.
    assert handler.deserialize_state(elements[:2], cache.meta_state) is None


def test_dsa_handler_slices_axis_one_and_round_trips_cache_metadata():
    config = _tiny_dsa_config()
    cache = Glm5NextDsaCache(config)
    latent = mx.arange(1 * 9 * 3).reshape(1, 9, 3).astype(mx.float32)
    packed = mx.arange(1 * 9 * 7).reshape(1, 9, 7).astype(mx.float32)
    cache.append(latent, packed)
    handler = Glm5NextDsaCacheHandler()
    state = handler.extract_state(cache)
    first = handler.slice_state(state, 0, 4)
    second = handler.slice_state(state, 4, 9)
    combined = handler.concatenate_states([first, second])
    restored = handler.reconstruct_cache(combined, cache.meta_state)
    assert isinstance(restored, Glm5NextDsaCache)
    assert restored.size() == 9
    assert restored.kv_latent.shape == (1, 9, 3)
    assert restored.indexer_states.shape == (1, 9, 7)
    np.testing.assert_array_equal(np.asarray(restored.kv_latent), np.asarray(latent))
    np.testing.assert_array_equal(
        np.asarray(restored.indexer_states), np.asarray(packed)
    )


def test_dsa_handler_accepts_legacy_two_tensor_block_payloads():
    handler = Glm5NextDsaCacheHandler()
    latent_a = mx.ones((1, 2, 3))
    latent_b = mx.ones((1, 1, 3)) * 2
    index_a = mx.ones((1, 2, 7)) * 3
    index_b = mx.ones((1, 1, 7)) * 4

    for legacy in (
        [
            {"keys": latent_a, "values": index_a},
            {"keys": latent_b, "values": index_b},
        ],
        [
            {"states": (latent_a, index_a)},
            {"states": (latent_b, index_b)},
        ],
    ):
        combined = handler.concatenate_states(legacy)
        np.testing.assert_array_equal(
            np.asarray(combined["kv_latent"]),
            np.asarray(mx.concatenate([latent_a, latent_b], axis=1)),
        )
        np.testing.assert_array_equal(
            np.asarray(combined["indexer_states"]),
            np.asarray(mx.concatenate([index_a, index_b], axis=1)),
        )


def test_prefix_block_extraction_uses_ntuple_for_axis_one_dsa_and_four_state_kda():
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    class TinyModel:
        layers = [object(), object()]

    prefix = BlockAwarePrefixCache(
        model=TinyModel(),
        paged_cache_manager=PagedCacheManager(
            block_size=4,
            max_blocks=8,
            initial_blocks=8,
            model_name="glm-handler-test",
        ),
    )
    kda = Glm5NextKDACache()
    kda.state = (
        mx.ones((1, 3, 8)),
        mx.ones((1, 3, 8)) * 2,
        mx.ones((1, 3, 8)) * 3,
        mx.ones((1, 2, 4, 4)) * 4,
    )
    kda.offset = 8
    dsa = Glm5NextDsaCache(_tiny_dsa_config())
    dsa.append(mx.ones((1, 8, 3)), mx.ones((1, 8, 7)))
    cache_data = [
        {
            "state": tuple(kda.state),
            "cache_type": "Glm5NextKDACache",
            "class_name": "Glm5NextKDACache",
            "meta_state": kda.meta_state,
        },
        {
            "state": tuple(dsa.state),
            "cache_type": "Glm5NextDsaCache",
            "class_name": "Glm5NextDsaCache",
            "meta_state": dsa.meta_state,
        },
    ]
    slices = prefix._extract_block_tensor_slice(
        cache_data,
        0,
        4,
        is_last_block=True,
        snapshot_cache_data=cache_data,
    )
    assert slices[0][0:2] == ("__nstate__", "Glm5NextKDACache")
    assert len(slices[0][2]) == 4
    assert slices[1][0:2] == ("__nstate__", "Glm5NextDsaCache")
    assert len(slices[1][2]) == 2
    assert slices[1][2][0].shape == (1, 4, 3)
    assert slices[1][2][1].shape == (1, 4, 7)
