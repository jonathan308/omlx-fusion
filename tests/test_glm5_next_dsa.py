from __future__ import annotations

import inspect

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.patches.glm5_next.contract import OFFICIAL_REVISION
from omlx.patches.glm5_next.dsa import (
    ALL_DSA_LAYERS,
    GLM5_NEXT_DSA_MODULE_READY,
    MAIN_DSA_LAYERS,
    MTP_DSA_LAYER,
    OFFICIAL_MODEL_REVISION,
    UPSTREAM_TRANSFORMERS_COMMIT,
    UPSTREAM_TRANSFORMERS_PATH,
    UPSTREAM_TRANSFORMERS_SHA256,
    Glm5NextDsa,
    Glm5NextDsaCache,
    Glm5NextDsaConfig,
    Glm5NextDsaContractError,
    dsa_weight_shapes,
    sparse_mla_attention,
    validate_dsa_layer_index,
    validate_dsa_weights,
)


def _tiny_config() -> Glm5NextDsaConfig:
    return Glm5NextDsaConfig(
        hidden_size=6,
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


def _weights(config: Glm5NextDsaConfig, seed: int = 7):
    mx.random.seed(seed)
    result = {}
    for name, shape in dsa_weight_shapes(config).items():
        result[name] = mx.random.normal(shape) * 0.2
    result["q_a_layernorm.weight"] = mx.ones(result["q_a_layernorm.weight"].shape)
    result["kv_a_layernorm.weight"] = mx.ones(result["kv_a_layernorm.weight"].shape)
    result["indexer.k_norm.weight"] = mx.ones(result["indexer.k_norm.weight"].shape)
    result["indexer.k_norm.bias"] = mx.zeros(result["indexer.k_norm.bias"].shape)
    return result


def _dense_selected_reference(q, latent, indices, kv_b, config):
    """Independent NumPy expansion of the selected-token attention equation."""

    q = np.asarray(q)
    latent = np.asarray(latent)
    indices = np.asarray(indices)
    kv_b = np.asarray(kv_b).reshape(
        config.num_attention_heads,
        config.qk_nope_head_dim + config.v_head_dim,
        config.kv_lora_rank,
    )
    key_w = kv_b[:, : config.qk_nope_head_dim]
    value_w = kv_b[:, config.qk_nope_head_dim :]
    all_keys = np.einsum("bkc,hdc->bhkd", latent, key_w)
    all_values = np.einsum("bkc,hdc->bhkd", latent, value_w)
    batch, heads, q_length, _ = q.shape
    result = np.zeros((batch, heads, q_length, config.v_head_dim), np.float32)
    for b in range(batch):
        for h in range(heads):
            for row in range(q_length):
                selected = [int(i) for i in indices[b, row] if 0 <= i < latent.shape[1]]
                # The production indexer never emits duplicates, while the mask
                # contract treats a repeated index as one selected key.
                selected = list(dict.fromkeys(selected))
                if not selected:
                    continue
                logits = (all_keys[b, h, selected] @ q[b, h, row]) * (
                    config.qk_nope_head_dim**-0.5
                )
                logits = logits - logits.max()
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum()
                result[b, h, row] = probabilities @ all_values[b, h, selected]
    return result.transpose(0, 2, 1, 3).reshape(
        batch, q_length, config.num_attention_heads * config.v_head_dim
    )


def test_sparse_mla_matches_expanded_reference_without_dense_scores():
    config = _tiny_config()
    weights = _weights(config)
    mx.random.seed(11)
    query = mx.random.normal(
        (
            2,
            config.num_attention_heads,
            3,
            config.qk_nope_head_dim,
        )
    )
    latent = mx.random.normal((2, 7, config.kv_lora_rank))
    indices = mx.array(
        [
            [[0, 2, 4, -1, -1], [1, 3, 5, -1, -1], [0, 4, 6, -1, -1]],
            [[1, 2, -1, -1, -1], [0, 3, 4, -1, -1], [-1, -1, -1, -1, -1]],
        ],
        dtype=mx.int32,
    )
    actual = sparse_mla_attention(
        query, latent, indices, weights["kv_b_proj.weight"], config
    )
    expected = _dense_selected_reference(
        query, latent, indices, weights["kv_b_proj.weight"], config
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)

    source = inspect.getsource(sparse_mla_attention)
    assert "_batch_take_rows" in source
    assert "dense attention" in source
    assert "mx.zeros" not in source
    assert '"bhlk"' not in source


def test_indexer_selection_is_causal_padding_safe_and_tail_exact():
    config = _tiny_config()
    layer = Glm5NextDsa(config, _weights(config), layer_idx=3)
    mx.random.seed(17)
    hidden = mx.random.normal((2, 6, config.hidden_size))
    mask = mx.array(
        [
            [False, True, True, True, True, True],
            [True, True, True, True, False, False],
        ],
        dtype=mx.bool_,
    )
    _, selected = layer(hidden, mask, return_topk=True)
    selected = np.asarray(selected)

    # Invalid query rows never select anything.
    assert np.all(selected[0, 0] == -1)
    assert np.all(selected[1, 4:] == -1)
    # Left padding can never be selected and every key is causal for its row.
    for batch in range(2):
        for row in range(6):
            valid = selected[batch, row][selected[batch, row] >= 0]
            assert np.all(valid <= row)
            assert all(mask[batch, int(index)].item() for index in valid)
    assert 0 not in selected[0]

    # Pool size two: at row 2, [1,2] is the first complete left-pad-aligned pool.
    assert set(selected[0, 2][selected[0, 2] >= 0]) == {1, 2}
    # At row 3 the newly incomplete pool is appended as an exact raw-token tail.
    assert set(selected[0, 3][selected[0, 3] >= 0]) == {1, 2, 3}


def test_prefill_and_token_decode_match_with_batch_cache():
    config = _tiny_config()
    layer = Glm5NextDsa(config, _weights(config, seed=23), layer_idx=7)
    mx.random.seed(29)
    hidden = mx.random.normal((2, 7, config.hidden_size))
    mask = mx.array(
        [
            [False, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        dtype=mx.bool_,
    )

    prefill, prefill_topk = layer(hidden, mask, return_topk=True)
    cache = Glm5NextDsaCache()
    decoded = []
    decoded_topk = []
    for position in range(hidden.shape[1]):
        out, indices = layer(
            hidden[:, position : position + 1],
            mask[:, position : position + 1],
            cache,
            return_topk=True,
        )
        decoded.append(out)
        decoded_topk.append(indices)
    decoded = mx.concatenate(decoded, axis=1)
    decoded_topk = mx.concatenate(decoded_topk, axis=1)
    mx.eval(prefill, prefill_topk, decoded, decoded_topk)

    np.testing.assert_allclose(
        np.asarray(decoded), np.asarray(prefill), rtol=3e-5, atol=3e-5
    )
    # The upstream ABI pads selected complete pools before/after the tail at a
    # position that depends on how many pool columns currently exist.  Sparse
    # attention consumes these as a set, so compare the selected sets exactly.
    prefill_topk_np = np.asarray(prefill_topk)
    decoded_topk_np = np.asarray(decoded_topk)
    for batch in range(hidden.shape[0]):
        for row in range(hidden.shape[1]):
            assert set(prefill_topk_np[batch, row]) - {-1} == (
                set(decoded_topk_np[batch, row]) - {-1}
            )
    assert cache.offset == hidden.shape[1]
    assert cache.batch_size == 2
    assert cache.kv_latent.shape == (2, 7, config.kv_lora_rank)
    assert cache.indexer_states.shape == (
        2,
        7,
        2 * config.index_head_dim + 1,
    )

    with pytest.raises(Glm5NextDsaContractError, match="cache batch changed"):
        layer(hidden[:1, :1], mask[:1, :1], cache)


def test_cache_reorder_preserves_both_state_streams():
    config = _tiny_config()
    cache = Glm5NextDsaCache()
    latent = (
        mx.arange(2 * 3 * config.kv_lora_rank)
        .reshape(2, 3, config.kv_lora_rank)
        .astype(mx.float32)
    )
    packed = (
        mx.arange(2 * 3 * (2 * config.index_head_dim + 1))
        .reshape(2, 3, 2 * config.index_head_dim + 1)
        .astype(mx.float32)
    )
    cache.append(latent, packed)
    cache.reorder(mx.array([1, 1, 0], dtype=mx.int32))
    mx.eval(cache.kv_latent, cache.indexer_states)
    np.testing.assert_array_equal(np.asarray(cache.kv_latent[0]), np.asarray(latent[1]))
    np.testing.assert_array_equal(
        np.asarray(cache.indexer_states[2]), np.asarray(packed[0])
    )
    assert cache.batch_size == 3
    assert cache.offset == 3


def test_exact_dsa_layer_positions_include_only_main_schedule_and_mtp():
    assert MAIN_DSA_LAYERS == (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)
    assert MTP_DSA_LAYER == 45
    assert ALL_DSA_LAYERS == MAIN_DSA_LAYERS + (45,)
    for layer_idx in ALL_DSA_LAYERS:
        validate_dsa_layer_index(layer_idx)
    for layer_idx in set(range(46)) - set(ALL_DSA_LAYERS):
        with pytest.raises(Glm5NextDsaContractError, match="is not DSA"):
            validate_dsa_layer_index(layer_idx)


def test_config_and_weight_contracts_fail_closed():
    config = _tiny_config()
    weights = _weights(config)
    validate_dsa_weights(weights, config)

    with pytest.raises(Glm5NextDsaContractError, match="qk_rope_head_dim"):
        Glm5NextDsaConfig(
            hidden_size=6,
            num_attention_heads=2,
            q_lora_rank=4,
            kv_lora_rank=3,
            qk_nope_head_dim=2,
            v_head_dim=2,
            index_n_heads=2,
            index_head_dim=3,
            index_topk=4,
            index_kpool=2,
            qk_rope_head_dim=2,
        )
    missing = dict(weights)
    del missing["indexer.k_norm.bias"]
    with pytest.raises(Glm5NextDsaContractError, match="missing="):
        validate_dsa_weights(missing, config)
    extra = dict(weights)
    extra["kv_a_proj_with_mqa.weight_scale_inv"] = mx.ones((1,))
    with pytest.raises(Glm5NextDsaContractError, match="extra="):
        validate_dsa_weights(extra, config)
    wrong = dict(weights)
    wrong["o_proj.weight"] = mx.zeros((1, 1))
    with pytest.raises(Glm5NextDsaContractError, match="o_proj.weight shape changed"):
        validate_dsa_weights(wrong, config)

    official = Glm5NextDsaConfig.official()
    assert dsa_weight_shapes(official) == {
        "q_a_proj.weight": (1536, 4096),
        "q_a_layernorm.weight": (1536,),
        "q_b_proj.weight": (16384, 1536),
        "kv_a_proj_with_mqa.weight": (512, 4096),
        "kv_a_layernorm.weight": (512,),
        "kv_b_proj.weight": (32768, 512),
        "o_proj.weight": (4096, 16384),
        "indexer.wq_b.weight": (4096, 1536),
        "indexer.wk.weight": (128, 4096),
        "indexer.k_norm.weight": (128,),
        "indexer.k_norm.bias": (128,),
        "indexer.weights_proj.weight": (32, 4096),
        "indexer.index_kpool_compress_ape": (4, 128),
        "indexer.index_kpool_compress_gate": (128, 4096),
    }


def test_source_audit_is_pinned_to_the_official_model_and_transformers_merge():
    assert OFFICIAL_MODEL_REVISION == OFFICIAL_REVISION
    assert OFFICIAL_MODEL_REVISION == "84c6a6aa9497188e15a635ba793b0f95a79b1033"
    assert UPSTREAM_TRANSFORMERS_COMMIT == "eb4d9e2a64a013bec12289288b85d0b1210ba0aa"
    assert UPSTREAM_TRANSFORMERS_PATH.endswith("glm5_next/modeling_glm5_next.py")
    assert UPSTREAM_TRANSFORMERS_SHA256 == (
        "2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b"
    )


def test_module_parameter_tree_is_lazy_loadable_with_official_sanitized_names():
    config = _tiny_config()
    module = Glm5NextDsa(config, layer_idx=3)
    parameters = dict(tree_flatten(module.parameters()))
    assert GLM5_NEXT_DSA_MODULE_READY is True
    assert set(parameters) == set(dsa_weight_shapes(config))
    assert {name: tuple(value.shape) for name, value in parameters.items()} == (
        dsa_weight_shapes(config)
    )
    assert not hasattr(module, "weights")

    # Strict MLX loading binds the complete official suffix tree without a
    # wrapper mapping remaining in the runtime module.
    loaded = Glm5NextDsa(config, _weights(config, seed=101), layer_idx=3)
    loaded_parameters = dict(tree_flatten(loaded.parameters()))
    assert set(loaded_parameters) == set(parameters)
    np.testing.assert_allclose(
        np.asarray(loaded_parameters["indexer.index_kpool_compress_gate"]),
        np.asarray(_weights(config, seed=101)["indexer.index_kpool_compress_gate"]),
    )


def _cache_payload(config, batch, length, base=0):
    latent = (
        mx.arange(batch * length * config.kv_lora_rank)
        .reshape(batch, length, config.kv_lora_rank)
        .astype(mx.float32)
        + base
    )
    packed = mx.zeros((batch, length, 2 * config.index_head_dim + 1), dtype=mx.float32)
    packed[..., 0] = latent[..., 0]
    packed[..., -1] = 1
    return latent, packed


def test_cache_prepare_finalize_state_and_mask_follow_mlx_lm_batch_abi():
    config = _tiny_config()
    cache = Glm5NextDsaCache(config)
    cache.prepare(lengths=[3, 1], right_padding=[0, 2])
    latent, packed = _cache_payload(config, 2, 3)
    assert cache.current_valid_mask(3).tolist() == [
        [True, True, True],
        [True, False, False],
    ]
    cache.append(latent, packed)
    cache.finalize()
    mx.eval(cache.state)

    assert cache.size() == 3
    assert cache.offsets.tolist() == [3, 1]
    assert cache.left_padding.tolist() == [0, 2]
    assert cache.indexer_states[1, :, -1].tolist() == [0.0, 0.0, 1.0]
    mask = cache.make_mask(1)
    assert mask.shape[-1] == 4

    restored = Glm5NextDsaCache.from_state(cache.state, cache.meta_state)
    assert restored.size() == cache.size()
    assert restored.left_padding.tolist() == [0, 2]
    np.testing.assert_array_equal(
        np.asarray(restored.kv_latent), np.asarray(cache.kv_latent)
    )


def test_cache_merge_extract_extend_filter_and_trim_are_batch_compatible():
    config = _tiny_config()
    short = Glm5NextDsaCache(config)
    long = Glm5NextDsaCache(config)
    short.append(*_cache_payload(config, 1, 2, base=10))
    long.append(*_cache_payload(config, 1, 4, base=100))

    merged = Glm5NextDsaCache.merge([short, long])
    mx.eval(merged.state)
    assert merged.batch_size == 2
    assert merged.size() == 4
    assert merged.left_padding.tolist() == [2, 0]
    assert merged.indexer_states[0, :2, -1].tolist() == [0.0, 0.0]
    np.testing.assert_array_equal(
        np.asarray(merged.extract(0).kv_latent), np.asarray(short.kv_latent)
    )
    np.testing.assert_array_equal(
        np.asarray(merged.extract(1).kv_latent), np.asarray(long.kv_latent)
    )

    # Every member can receive a common decode token, then be reordered and
    # reduced by the same API used by mlx-lm continuous batching.
    merged.append(*_cache_payload(config, 2, 1, base=1000))
    assert merged.size() == 5
    merged.filter(mx.array([1, 0, 1], dtype=mx.int32))
    assert merged.batch_size == 3
    assert merged.left_padding.tolist() == [0, 2, 0]
    assert merged.trim(1) == 1
    assert merged.size() == 4

    first = merged.extract(0)
    second = merged.extract(1)
    first.extend(second)
    assert first.batch_size == 2
    assert first.size() == 4
