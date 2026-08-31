# SPDX-License-Identifier: Apache-2.0
"""Exact disconnected Qwen4 W=2 layer-major scalar wavefront tests."""

from __future__ import annotations

import mlx.core as mx
import pytest
from test_qwen4_suffix_local_priming import (
    _assert_target_cache_equal,
    _model,
    _tiny_config,
)

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _prefill(model, tokens):
    cache = model.make_cache()
    model._position_ids = None
    model._rope_deltas = None
    output = model(tokens, cache=cache)
    mx.eval(output.logits, *[entry.state for entry in cache])
    return cache


def _pair_model(tied):
    if not tied:
        return _model()
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    config = _tiny_config()
    config.text_config.tie_word_embeddings = True
    model = LanguageModel(config.text_config, config)
    mx.eval(model.parameters())
    return model


def _pooled_state(cache):
    return [
        (
            entry._pooled_index_keys,
            entry._pooled_index_offset,
            entry._pooled_index_ratio,
            entry._pooled_index_tag,
        )
        for entry in cache
        if type(entry).__name__.startswith("QSA")
    ]


def _cache_snapshot(cache):
    arrays = []
    metadata = []
    for entry in cache:
        values = tuple(value * 1 for value in entry.state if value is not None)
        arrays.append(values)
        if type(entry).__name__.startswith("QSA"):
            metadata.append(
                (
                    entry.offset,
                    entry._index_offset,
                    entry._pooled_index_offset,
                    entry._pooled_index_ratio,
                    entry._pooled_index_tag,
                    getattr(entry, "_omlx_text_position_ids_qualified", False),
                )
            )
        else:
            metadata.append(None)
    mx.eval(*[value for values in arrays for value in values])
    return arrays, metadata


def _assert_snapshot_unchanged(cache, snapshot):
    arrays, metadata = snapshot
    for entry, expected_values, expected_metadata in zip(cache, arrays, metadata):
        actual_values = tuple(value for value in entry.state if value is not None)
        mx.eval(*actual_values, *expected_values)
        assert len(actual_values) == len(expected_values)
        assert all(
            mx.array_equal(actual, expected).item()
            for actual, expected in zip(actual_values, expected_values)
        )
        if expected_metadata is not None:
            assert (
                entry.offset,
                entry._index_offset,
                entry._pooled_index_offset,
                entry._pooled_index_ratio,
                entry._pooled_index_tag,
                getattr(entry, "_omlx_text_position_ids_qualified", False),
            ) == expected_metadata


@pytest.mark.parametrize("prefix_length", [9, 10, 11, 12])
@pytest.mark.parametrize("use_batched_selector", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_scalar_pair_is_array_equal_to_two_scalar_calls(
    prefix_length,
    use_batched_selector,
    tied,
):
    mx.random.seed(9100 + prefix_length)
    model = _pair_model(tied)
    prefix = (mx.arange(prefix_length, dtype=mx.int32) % 50 + 2)[None]
    tokens = mx.array([[51, 52]], dtype=mx.int32)
    scalar_cache = _prefill(model, prefix)
    pair_cache = _prefill(model, prefix)

    scalar_logits = []
    scalar_hidden = []
    for row in range(2):
        output = model(
            tokens[:, row : row + 1],
            cache=scalar_cache,
            return_hidden=True,
        )
        scalar_logits.append(output.logits)
        scalar_hidden.append(output.hidden_states[-1])
        mx.eval(output.logits, *[entry.state for entry in scalar_cache])

    pair = model.scalar_pair(
        tokens,
        cache=pair_cache,
        prove_selector_parity=True,
        use_batched_selector=use_batched_selector,
    )
    mx.eval(pair.logits, pair.hidden_states[-1], *[entry.state for entry in pair_cache])
    expected_logits = mx.concatenate(scalar_logits, axis=1)
    expected_hidden = mx.concatenate(scalar_hidden, axis=1)
    assert mx.array_equal(pair.logits, expected_logits).item()
    assert mx.array_equal(pair.hidden_states[-1], expected_hidden).item()
    pair_lp = pair.logits - mx.logsumexp(pair.logits, axis=-1, keepdims=True)
    scalar_lp = expected_logits - mx.logsumexp(
        expected_logits,
        axis=-1,
        keepdims=True,
    )
    mx.eval(pair_lp, scalar_lp)
    assert mx.array_equal(pair_lp, scalar_lp).item()
    _assert_target_cache_equal(pair_cache, scalar_cache)

    pair_pooled = _pooled_state(pair_cache)
    scalar_pooled = _pooled_state(scalar_cache)
    assert len(pair_pooled) == len(scalar_pooled) > 0
    for left, right in zip(pair_pooled, scalar_pooled):
        mx.eval(left[0], right[0])
        assert left[1:3] == right[1:3]
        assert left[3] is right[3]
        assert mx.array_equal(left[0], right[0]).item()


@pytest.mark.parametrize("tail_residue", [0, 1, 2, 3])
def test_factored_m2_selector_and_scalar_decode_are_bit_exact(
    tail_residue,
    monkeypatch,
):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import qsa_fast

    contiguous_shapes = []
    original_contiguous = qsa_fast.mx.contiguous

    def tracked_contiguous(value, *args, **kwargs):
        contiguous_shapes.append(tuple(value.shape))
        return original_contiguous(value, *args, **kwargs)

    monkeypatch.setattr(qsa_fast.mx, "contiguous", tracked_contiguous)

    mx.random.seed(9200 + tail_residue)
    query_start = 2400 + tail_residue
    key_tokens = query_start + 2
    max_blocks = key_tokens // 4
    index_queries = mx.random.normal((1, 2, 4, 128)).astype(mx.bfloat16)
    pooled = mx.random.normal((1, max_blocks, 128)).astype(mx.bfloat16)
    pair_selected = qsa_fast.qsa_decode_selected_blocks(
        index_queries,
        pooled,
        indexer_head_dim=128,
        compress_ratio=4,
        token_budget=2048,
        query_start=query_start,
    )
    scalar_selected = mx.concatenate(
        [
            qsa_fast.qsa_decode_selected_blocks(
                index_queries[:, row : row + 1],
                pooled,
                indexer_head_dim=128,
                compress_ratio=4,
                token_budget=2048,
                query_start=query_start + row,
            )
            for row in range(2)
        ],
        axis=1,
    )
    mx.eval(pair_selected, scalar_selected)
    assert mx.array_equal(pair_selected, scalar_selected).item()

    queries = mx.random.normal((1, 24, 2, 256)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 2, key_tokens, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, key_tokens, 256)).astype(mx.bfloat16)
    pair_output = qsa_fast.qsa_decode_from_selected_blocks(
        queries,
        keys,
        values,
        pair_selected,
        head_dim=256,
        compress_ratio=4,
        query_start=query_start,
    )
    scalar_output = mx.concatenate(
        [
            qsa_fast.qsa_decode_from_selected_blocks(
                queries[:, :, row : row + 1],
                keys,
                values,
                scalar_selected[:, row : row + 1],
                head_dim=256,
                compress_ratio=4,
                query_start=query_start + row,
            )
            for row in range(2)
        ],
        axis=1,
    )
    mx.eval(pair_output, scalar_output)
    assert mx.array_equal(pair_output, scalar_output).item()
    assert contiguous_shapes.count((1, 24, 1, 256)) == 4


def test_scalar_pair_rejects_noncanonical_shapes_and_caches():
    model = _model()
    cache = _prefill(
        model,
        mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=mx.int32),
    )
    with pytest.raises(ValueError, match=r"\[1,2\]"):
        model.scalar_pair(mx.array([[6]], dtype=mx.int32), cache=cache)

    qsa = next(entry for entry in cache if type(entry).__name__.startswith("QSA"))
    qsa.offset = mx.array([qsa.offset], dtype=mx.int32)
    with pytest.raises(ValueError, match="scalar cache offset"):
        model.scalar_pair(mx.array([[6, 7]], dtype=mx.int32), cache=cache)


def test_scalar_pair_preflight_rejects_short_context_without_cache_mutation():
    model = _model()
    cache = _prefill(model, mx.array([[2, 3, 4, 5]], dtype=mx.int32))
    snapshot = _cache_snapshot(cache)
    with pytest.raises(ValueError, match="row zero is not sparse"):
        model.scalar_pair(mx.array([[6, 7]], dtype=mx.int32), cache=cache)
    _assert_snapshot_unchanged(cache, snapshot)


def test_scalar_pair_preflight_rejects_misaligned_qsa_without_cache_mutation():
    model = _model()
    cache = _prefill(
        model,
        mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 12]], dtype=mx.int32),
    )
    qsa = next(entry for entry in cache if type(entry).__name__.startswith("QSA"))
    qsa._index_offset -= 1
    snapshot = _cache_snapshot(cache)
    with pytest.raises(ValueError, match="qualification is not text"):
        model.scalar_pair(mx.array([[13, 14]], dtype=mx.int32), cache=cache)
    _assert_snapshot_unchanged(cache, snapshot)


def test_scalar_pair_preflight_rejects_world_two_without_cache_mutation(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.language as language

    model = _model()
    cache = _prefill(
        model,
        mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 12]], dtype=mx.int32),
    )
    snapshot = _cache_snapshot(cache)
    monkeypatch.setattr(language, "_scalar_pair_world_size", lambda: 2)
    with pytest.raises(ValueError, match="world size one"):
        model.scalar_pair(mx.array([[13, 14]], dtype=mx.int32), cache=cache)
    _assert_snapshot_unchanged(cache, snapshot)


def test_scalar_pair_preflight_rejects_multimodal_positions_without_mutation():
    model = _model()
    cache = _prefill(
        model,
        mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 12]], dtype=mx.int32),
    )
    qsa = next(entry for entry in cache if type(entry).__name__.startswith("QSA"))
    qsa._index_position_ids = mx.broadcast_to(
        qsa._index_position_ids[None],
        (3, *qsa._index_position_ids.shape),
    ) + mx.zeros((), dtype=qsa._index_position_ids.dtype)
    qsa._index_position_ids = qsa._index_position_ids.at[1, :, :1].add(1)
    qsa._omlx_text_position_ids_qualified = False
    snapshot = _cache_snapshot(cache)
    with pytest.raises(ValueError, match="not canonical text"):
        model.scalar_pair(mx.array([[13, 14]], dtype=mx.int32), cache=cache)
    _assert_snapshot_unchanged(cache, snapshot)


def test_scalar_pair_rejects_stale_qualification_without_cache_mutation():
    model = _model()
    cache = _prefill(
        model,
        mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 12]], dtype=mx.int32),
    )
    qualification = model.qualify_scalar_pair_cache(cache)
    qsa = next(entry for entry in cache if type(entry).__name__.startswith("QSA"))
    qsa._index_position_ids = qsa._index_position_ids + 0
    mx.eval(qsa._index_position_ids)
    snapshot = _cache_snapshot(cache)
    with pytest.raises(ValueError, match="qualification is stale"):
        model.scalar_pair(
            mx.array([[13, 14]], dtype=mx.int32),
            cache=cache,
            qualification=qualification,
        )
    _assert_snapshot_unchanged(cache, snapshot)


def test_scalar_pair_rejects_wrong_owner_qualification_without_cache_mutation():
    model = _model()
    other = _model()
    prefix = mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 12]], dtype=mx.int32)
    cache = _prefill(model, prefix)
    other_cache = _prefill(other, prefix)
    qualification = other.qualify_scalar_pair_cache(other_cache)
    snapshot = _cache_snapshot(cache)
    with pytest.raises(ValueError, match="wrong owner"):
        model.scalar_pair(
            mx.array([[13, 14]], dtype=mx.int32),
            cache=cache,
            qualification=qualification,
        )
    _assert_snapshot_unchanged(cache, snapshot)


def test_scalar_pair_accepts_equal_plane_3d_text_history_exactly():
    model = _model()
    prefix = mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 12]], dtype=mx.int32)
    tokens = mx.array([[13, 14]], dtype=mx.int32)
    scalar_cache = _prefill(model, prefix)
    pair_cache = _prefill(model, prefix)
    for cache in (scalar_cache, pair_cache):
        for entry in cache:
            if type(entry).__name__.startswith("QSA"):
                entry._index_position_ids = mx.broadcast_to(
                    entry._index_position_ids[None],
                    (3, *entry._index_position_ids.shape),
                ) + mx.zeros((), dtype=entry._index_position_ids.dtype)
                mx.eval(entry._index_position_ids)
    scalar = []
    for row in range(2):
        output = model(tokens[:, row : row + 1], cache=scalar_cache)
        scalar.append(output.logits)
        mx.eval(output.logits)
    qualification = model.qualify_scalar_pair_cache(pair_cache)
    pair = model.scalar_pair(
        tokens,
        cache=pair_cache,
        qualification=qualification,
        prove_selector_parity=True,
        use_batched_selector=True,
    )
    expected = mx.concatenate(scalar, axis=1)
    mx.eval(pair.logits, expected)
    assert mx.array_equal(pair.logits, expected).item()
    _assert_target_cache_equal(pair_cache, scalar_cache)


def test_scalar_pair_preflight_rejects_b2_kv_without_cache_mutation():
    model = _model()
    cache = _prefill(
        model,
        mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 12]], dtype=mx.int32),
    )
    qsa = next(entry for entry in cache if type(entry).__name__.startswith("QSA"))
    qsa.keys = mx.concatenate([qsa.keys, qsa.keys], axis=0)
    qsa.values = mx.concatenate([qsa.values, qsa.values], axis=0)
    mx.eval(qsa.keys, qsa.values)
    snapshot = _cache_snapshot(cache)
    with pytest.raises(ValueError, match="QSA cache is misaligned"):
        model.scalar_pair(mx.array([[13, 14]], dtype=mx.int32), cache=cache)
    _assert_snapshot_unchanged(cache, snapshot)


def test_scalar_pair_preflight_rejects_malformed_pooled_bank_without_mutation():
    model = _model()
    cache = _prefill(
        model,
        mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 12]], dtype=mx.int32),
    )
    qsa = next(entry for entry in cache if type(entry).__name__.startswith("QSA"))
    qsa._pooled_index_keys = qsa._pooled_index_keys[..., :-1]
    mx.eval(qsa._pooled_index_keys)
    snapshot = _cache_snapshot(cache)
    with pytest.raises(ValueError, match="pooled QSA cache is incomplete"):
        model.scalar_pair(mx.array([[13, 14]], dtype=mx.int32), cache=cache)
    _assert_snapshot_unchanged(cache, snapshot)


def test_scalar_pair_keeps_every_nonselector_stage_at_m1(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.language as language

    model = _model()
    cache = _prefill(
        model,
        mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 12]], dtype=mx.int32),
    )
    seen = []

    original_linears = language._target_verify_linears
    original_linear = language._target_verify_linear
    original_moe = language._qwen4_moe_forward
    original_gdn = language.Qwen4ExpGatedDeltaNet.__call__
    original_ple = language.Qwen4ExpPLELayer.__call__
    original_hc = language.Qwen4ExpGatedResidual.__call__
    embedding_type = type(model.model.embed_tokens)
    original_head = embedding_type.as_linear

    def linears(modules, values, target_verify):
        seen.append(("linears", int(values.shape[1])))
        return original_linears(modules, values, target_verify)

    def linear(module, values, target_verify):
        seen.append(("linear", int(values.shape[1])))
        return original_linear(module, values, target_verify)

    def moe(module, values, target_verify):
        seen.append(("moe", int(values.shape[1])))
        return original_moe(module, values, target_verify)

    def gdn(self, values, *args, **kwargs):
        seen.append(("gdn", int(values.shape[1])))
        return original_gdn(self, values, *args, **kwargs)

    def ple(self, hidden, input_ids, *args, **kwargs):
        seen.append(("ple-hidden", int(hidden.shape[1])))
        seen.append(("ple-ids", int(input_ids.shape[1])))
        return original_ple(self, hidden, input_ids, *args, **kwargs)

    def hc(self, values, *args, **kwargs):
        seen.append(("hc", int(values.shape[1])))
        return original_hc(self, values, *args, **kwargs)

    def head(self, values, *args, **kwargs):
        seen.append(("lm-head", int(values.shape[1])))
        return original_head(self, values, *args, **kwargs)

    monkeypatch.setattr(language, "_target_verify_linears", linears)
    monkeypatch.setattr(language, "_target_verify_linear", linear)
    monkeypatch.setattr(language, "_qwen4_moe_forward", moe)
    monkeypatch.setattr(language.Qwen4ExpGatedDeltaNet, "__call__", gdn)
    monkeypatch.setattr(language.Qwen4ExpPLELayer, "__call__", ple)
    monkeypatch.setattr(language.Qwen4ExpGatedResidual, "__call__", hc)
    monkeypatch.setattr(embedding_type, "as_linear", head)
    output = model.scalar_pair(
        mx.array([[10, 11]], dtype=mx.int32),
        cache=cache,
        prove_selector_parity=True,
        use_batched_selector=True,
    )
    mx.eval(output.logits)
    assert seen
    assert all(width == 1 for _stage, width in seen)


def test_batched_selector_is_not_used_when_actual_pair_parity_fails(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.language as language

    model = _model()
    prefix = mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 12]], dtype=mx.int32)
    tokens = mx.array([[10, 11]], dtype=mx.int32)
    scalar_cache = _prefill(model, prefix)
    pair_cache = _prefill(model, prefix)
    scalar = []
    for row in range(2):
        output = model(tokens[:, row : row + 1], cache=scalar_cache)
        scalar.append(output.logits)
        mx.eval(output.logits)

    original = language.qsa_decode_selected_blocks

    def corrupt_only_batched(index_queries, *args, **kwargs):
        selected = original(index_queries, *args, **kwargs)
        if index_queries.shape[1] == 2:
            selected = selected.at[:, :1, :1].add(1)
        return selected

    monkeypatch.setattr(language, "qsa_decode_selected_blocks", corrupt_only_batched)
    pair = model.scalar_pair(
        tokens,
        cache=pair_cache,
        prove_selector_parity=True,
        use_batched_selector=True,
    )
    expected = mx.concatenate(scalar, axis=1)
    mx.eval(pair.logits, expected)
    assert mx.array_equal(pair.logits, expected).item()
    for layer in model.model.layers:
        if not layer.is_linear:
            assert layer.self_attn._omlx_scalar_pair_selector_parity is False


def test_batched_selector_never_reuses_stale_parity_from_an_earlier_pair(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.language as language

    model = _model()
    prefix = mx.array([[2, 3, 4, 5, 6, 7, 8, 9, 12]], dtype=mx.int32)
    first_cache = _prefill(model, prefix)
    first = model.scalar_pair(
        mx.array([[10, 11]], dtype=mx.int32),
        cache=first_cache,
        prove_selector_parity=True,
        use_batched_selector=True,
    )
    mx.eval(first.logits)
    assert all(
        layer.is_linear or layer.self_attn._omlx_scalar_pair_selector_parity
        for layer in model.model.layers
    )

    tokens = mx.array([[13, 14]], dtype=mx.int32)
    scalar_cache = _prefill(model, prefix)
    pair_cache = _prefill(model, prefix)
    scalar = []
    for row in range(2):
        output = model(tokens[:, row : row + 1], cache=scalar_cache)
        scalar.append(output.logits)
        mx.eval(output.logits)
    original = language.qsa_decode_selected_blocks

    def corrupt_only_batched(index_queries, *args, **kwargs):
        selected = original(index_queries, *args, **kwargs)
        if index_queries.shape[1] == 2:
            selected = selected.at[:, :1, :1].add(1)
        return selected

    monkeypatch.setattr(language, "qsa_decode_selected_blocks", corrupt_only_batched)
    pair = model.scalar_pair(
        tokens,
        cache=pair_cache,
        prove_selector_parity=False,
        use_batched_selector=True,
    )
    expected = mx.concatenate(scalar, axis=1)
    mx.eval(pair.logits, expected)
    assert mx.array_equal(pair.logits, expected).item()
