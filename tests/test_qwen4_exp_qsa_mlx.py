from __future__ import annotations

import inspect
import math

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.qwen4_exp.qsa import (
    Qwen4ExpQSAContract,
    Qwen4ExpQSAExecutor,
    Qwen4ExpQSAInputError,
)
from omlx.patches.qwen4_exp.qsa_mlx import (
    Qwen4ExpMLXQSABackend,
    Qwen4ExpQSAGeometry,
    Qwen4ExpQSAKVCache,
    apply_partial_rope,
    micro_block_sparse_qsa,
    prepare_qsa_request,
    select_qsa_token_indices,
)

SMALL = Qwen4ExpQSAGeometry(
    num_query_heads=4,
    num_key_value_heads=2,
    head_dim=4,
    rotary_dim=2,
    indexer_query_heads=2,
    indexer_head_dim=2,
    compress_ratio=4,
    token_budget=8,
)


def _identity(value):
    return value


def _positions(batch: int, length: int, rotary_dim: int = 2):
    angle = mx.arange(length, dtype=mx.float32)[None, :, None] * 0.17
    angle = mx.broadcast_to(angle, (batch, length, rotary_dim))
    return mx.cos(angle), mx.sin(angle)


def _small_request(*, query_tokens=3, key_tokens=3, cache=None, mask=None):
    mx.random.seed(71 + query_tokens + key_tokens)
    cos, sin = _positions(1, key_tokens)
    if mask is None:
        query_start = key_tokens - query_tokens
        qpos = query_start + mx.arange(query_tokens)
        kpos = mx.arange(key_tokens)
        mask = (kpos[None, :] <= qpos[:, None])[None, None]
    return prepare_qsa_request(
        queries=mx.random.normal((1, 4, query_tokens, 4)),
        keys=mx.random.normal((1, 2, query_tokens, 4))
        if cache is not None
        else mx.random.normal((1, 2, key_tokens, 4)),
        values=mx.random.normal((1, 2, query_tokens, 4))
        if cache is not None
        else mx.random.normal((1, 2, key_tokens, 4)),
        index_queries=mx.random.normal((1, query_tokens, 2, 2)),
        index_keys=mx.random.normal((1, query_tokens, 2))
        if cache is not None
        else mx.random.normal((1, key_tokens, 2)),
        position_cos=cos[:, -query_tokens:] if cache is not None else cos,
        position_sin=sin[:, -query_tokens:] if cache is not None else sin,
        attention_mask=mask,
        cache=cache,
    )


def _numpy_partial_rope(states, cos, sin, rotary_dim):
    rope = states[..., :rotary_dim]
    half = rotary_dim // 2
    rotated_half = np.concatenate((-rope[..., half:], rope[..., :half]), axis=-1)
    return np.concatenate(
        (rope * cos + rotated_half * sin, states[..., rotary_dim:]), axis=-1
    )


def test_selection_scores_complete_four_token_blocks_and_keeps_tail():
    # Three complete blocks score 3, 1, 2. Budget=2 blocks, so block 1 is
    # omitted and the final three incomplete tokens remain visible.
    raw_keys = mx.array(
        [[3.0, 0.0]] * 4 + [[1.0, 0.0]] * 4 + [[2.0, 0.0]] * 4 + [[-20.0, 0.0]] * 3
    )
    cos = mx.ones((15, 2))
    sin = mx.zeros((15, 2))
    selected, complete, selected_blocks = select_qsa_token_indices(
        mx.array([[1.0, 0.0], [1.0, 0.0]]),
        raw_keys,
        cos,
        sin,
        mx.ones((15,), dtype=mx.bool_),
        geometry=SMALL,
        index_key_norm=_identity,
    )
    mx.eval(selected)

    assert complete == 3
    assert selected_blocks == 2
    assert set(np.asarray(selected).tolist()) == set(range(4)) | set(range(8, 15))
    assert set(np.asarray(selected[-3:]).tolist()) == {12, 13, 14}


def test_visible_holes_are_compacted_before_forming_micro_blocks():
    keys = mx.arange(24, dtype=mx.float32).reshape(12, 2)
    visible = mx.array(
        [True, False, True, True, False, True, True, False, True, True, False, True]
    )
    seen = []

    def record_norm(pooled):
        seen.append(np.asarray(pooled))
        return pooled

    selected, complete, _ = select_qsa_token_indices(
        mx.ones((2, 2)),
        keys,
        mx.ones((12, 2)),
        mx.zeros((12, 2)),
        visible,
        geometry=SMALL,
        index_key_norm=record_norm,
    )
    mx.eval(selected)

    visible_indices = np.flatnonzero(np.asarray(visible))
    expected_groups = visible_indices[:8].reshape(2, 4)
    expected_means = np.asarray(keys)[expected_groups].mean(axis=1)
    assert complete == 2
    np.testing.assert_allclose(seen[0], expected_means)
    assert np.asarray(selected).tolist() == visible_indices.tolist()


def test_sparse_output_matches_dense_reference_when_every_visible_block_fits():
    request = _small_request(query_tokens=3, key_tokens=3)
    got = micro_block_sparse_qsa(request, geometry=SMALL, index_key_norm=_identity)
    mx.eval(got)

    q = np.asarray(request.queries)
    k = np.asarray(request.keys)
    v = np.asarray(request.values)
    cos = np.asarray(request.position_cos)
    sin = np.asarray(request.position_sin)
    mask = np.asarray(request.attention_mask)
    expected_rows = []
    for row in range(3):
        qr = _numpy_partial_rope(q[0, :, row], cos[0, row], sin[0, row], 2)
        visible = np.flatnonzero(mask[0, 0, row])
        kr = np.swapaxes(k[0][:, visible, :], 0, 1)
        kr = _numpy_partial_rope(
            kr, cos[0, visible, None, :], sin[0, visible, None, :], 2
        )
        kr = np.swapaxes(kr, 0, 1)
        vr = v[0][:, visible, :]
        grouped_q = qr.reshape(2, 2, 4)
        scores = grouped_q @ np.swapaxes(kr, -1, -2) / math.sqrt(4)
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        expected_rows.append((probs @ vr).reshape(4, 4))
    expected = np.asarray(expected_rows)[None]

    # MLX may route even float32 toy matmuls through reduced-precision Apple
    # accelerator paths (the guarded M5 build does); production Qwen states
    # are BF16. Keep this strict enough to catch selection/RoPE mistakes while
    # allowing the expected hardware-dependent accumulation rounding.
    np.testing.assert_allclose(np.asarray(got), expected, rtol=8e-3, atol=2e-3)


def test_long_row_observer_proves_main_attention_never_gathers_full_context():
    key_tokens = 43
    request = _small_request(query_tokens=1, key_tokens=key_tokens)
    traces = []
    got = micro_block_sparse_qsa(
        request,
        geometry=SMALL,
        index_key_norm=_identity,
        row_observer=traces.append,
    )
    mx.eval(got)

    assert got.shape == (1, 1, 4, 4)
    assert len(traces) == 1
    trace = traces[0]
    assert trace.full_key_tokens == 43
    assert trace.complete_blocks == 10
    assert trace.selected_blocks == 2
    assert trace.selected_tokens == 11  # 8 selected block tokens + tail 3
    assert trace.selected_tokens < trace.full_key_tokens


def test_stateless_single_query_defaults_to_the_final_full_context_position():
    cos, sin = _positions(1, 9)
    request = prepare_qsa_request(
        queries=mx.ones((1, 4, 1, 4)),
        keys=mx.ones((1, 2, 9, 4)),
        values=mx.ones((1, 2, 9, 4)),
        index_queries=mx.ones((1, 1, 2, 2)),
        index_keys=mx.ones((1, 9, 2)),
        position_cos=cos,
        position_sin=sin,
    )

    assert request.attention_mask.shape == (1, 1, 1, 9)
    assert bool(mx.all(request.attention_mask).item())


def test_bool_and_additive_masks_have_identical_sparse_causal_semantics():
    boolean = _small_request(query_tokens=3, key_tokens=3)
    additive = mx.where(
        boolean.attention_mask,
        mx.array(0.0, dtype=mx.float32),
        mx.array(-float("inf"), dtype=mx.float32),
    )
    additive_request = type(boolean)(
        **{
            **boolean.__dict__,
            "attention_mask": additive,
        }
    )
    a = micro_block_sparse_qsa(boolean, geometry=SMALL, index_key_norm=_identity)
    b = micro_block_sparse_qsa(
        additive_request, geometry=SMALL, index_key_norm=_identity
    )
    mx.eval(a, b)

    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


def test_cache_seam_appends_prefill_then_single_token_decode():
    cache = Qwen4ExpQSAKVCache(step=4)
    mx.random.seed(90)
    prefill_cos, prefill_sin = _positions(1, 4)
    prefill = prepare_qsa_request(
        queries=mx.random.normal((1, 4, 4, 4)),
        keys=mx.random.normal((1, 2, 4, 4)),
        values=mx.random.normal((1, 2, 4, 4)),
        index_queries=mx.random.normal((1, 4, 2, 2)),
        index_keys=mx.random.normal((1, 4, 2)),
        position_cos=prefill_cos,
        position_sin=prefill_sin,
        cache=cache,
    )
    prefill_out = micro_block_sparse_qsa(
        prefill, geometry=SMALL, index_key_norm=_identity
    )

    decode_cos, decode_sin = _positions(1, 5)
    decode = prepare_qsa_request(
        queries=mx.random.normal((1, 4, 1, 4)),
        keys=mx.random.normal((1, 2, 1, 4)),
        values=mx.random.normal((1, 2, 1, 4)),
        index_queries=mx.random.normal((1, 1, 2, 2)),
        index_keys=mx.random.normal((1, 1, 2)),
        position_cos=decode_cos[:, -1:],
        position_sin=decode_sin[:, -1:],
        cache=cache,
    )
    decode_out = micro_block_sparse_qsa(
        decode, geometry=SMALL, index_key_norm=_identity
    )
    mx.eval(prefill_out, decode_out)

    assert prefill_out.shape == (1, 4, 4, 4)
    assert decode_out.shape == (1, 1, 4, 4)
    assert cache.offset == 5
    assert decode.keys.shape == (1, 2, 5, 4)
    assert decode.index_keys.shape == (1, 5, 2)
    assert decode.attention_mask.shape == (1, 1, 1, 5)
    assert bool(mx.all(decode.attention_mask).item())


def test_short_decode_fast_path_matches_row_reference():
    request = _small_request(query_tokens=1, key_tokens=9)
    fast = micro_block_sparse_qsa(request, geometry=SMALL, index_key_norm=_identity)
    traced = []
    row_reference = micro_block_sparse_qsa(
        request,
        geometry=SMALL,
        index_key_norm=_identity,
        row_observer=traced.append,
    )
    mx.eval(fast, row_reference)

    np.testing.assert_allclose(
        np.asarray(fast), np.asarray(row_reference), rtol=2e-5, atol=2e-5
    )
    assert len(traced) == 1


@pytest.mark.parametrize("additive", [False, True])
def test_vectorized_prefill_matches_row_reference_with_padding_holes_and_tail(additive):
    batch, query_tokens, key_tokens = 2, 19, 43
    mx.random.seed(314)
    cos, sin = _positions(batch, key_tokens)
    query_start = key_tokens - query_tokens
    qpos = query_start + np.arange(query_tokens)
    kpos = np.arange(key_tokens)
    causal = kpos[None, :] <= qpos[:, None]
    masks = np.broadcast_to(causal[None], (batch, query_tokens, key_tokens)).copy()
    # Non-prefix visibility proves stable compaction semantics. Keep every
    # query's own position visible so attention never receives an empty row.
    masks[0, :, 1::5] = False
    masks[1, :, 2::7] = False
    for b in range(batch):
        masks[b, np.arange(query_tokens), qpos] = True
    mask = mx.array(masks[:, None])
    if additive:
        mask = mx.where(mask, mx.array(0.0), mx.array(-float("inf")))
    request = prepare_qsa_request(
        queries=mx.random.normal((batch, 4, query_tokens, 4)),
        keys=mx.random.normal((batch, 2, key_tokens, 4)),
        values=mx.random.normal((batch, 2, key_tokens, 4)),
        index_queries=mx.random.normal((batch, query_tokens, 2, 2)),
        index_keys=mx.random.normal((batch, key_tokens, 2)),
        position_cos=cos,
        position_sin=sin,
        attention_mask=mask,
    )
    vectorized = micro_block_sparse_qsa(
        request, geometry=SMALL, index_key_norm=_identity
    )
    traces = []
    reference = micro_block_sparse_qsa(
        request,
        geometry=SMALL,
        index_key_norm=_identity,
        row_observer=traces.append,
    )
    mx.eval(vectorized, reference)

    np.testing.assert_allclose(
        np.asarray(vectorized), np.asarray(reference), rtol=2e-5, atol=2e-5
    )
    assert len(traces) == batch * query_tokens
    assert max(trace.selected_tokens for trace in traces) <= (
        SMALL.token_budget + SMALL.compress_ratio - 1
    )


@pytest.mark.parametrize("additive", [False, True])
def test_contiguous_causal_prefill_matches_generic_sparse_path(additive):
    query_tokens, key_tokens = 19, 43
    mx.random.seed(812)
    cos, sin = _positions(1, key_tokens)
    query_start = key_tokens - query_tokens
    qpos = query_start + mx.arange(query_tokens)
    kpos = mx.arange(key_tokens)
    mask = (kpos[None, :] <= qpos[:, None])[None, None]
    if additive:
        mask = mx.where(mask, mx.array(0.0), mx.array(-float("inf")))
    request = prepare_qsa_request(
        queries=mx.random.normal((1, 4, query_tokens, 4)),
        keys=mx.random.normal((1, 2, key_tokens, 4)),
        values=mx.random.normal((1, 2, key_tokens, 4)),
        index_queries=mx.random.normal((1, query_tokens, 2, 2)),
        index_keys=mx.random.normal((1, key_tokens, 2)),
        position_cos=cos,
        position_sin=sin,
        attention_mask=mask,
        contiguous_causal=True,
    )
    generic = type(request)(**{**request.__dict__, "contiguous_causal": False})
    fast = micro_block_sparse_qsa(request, geometry=SMALL, index_key_norm=_identity)
    expected = micro_block_sparse_qsa(generic, geometry=SMALL, index_key_norm=_identity)
    mx.eval(fast, expected)

    np.testing.assert_allclose(
        np.asarray(fast), np.asarray(expected), rtol=2e-5, atol=2e-5
    )


def test_contiguous_causal_query_chunk_adapts_to_context() -> None:
    import omlx.patches.qwen4_exp.qsa_mlx as module

    assert module._contiguous_causal_query_chunk(4096) == 32
    assert module._contiguous_causal_query_chunk(4097) == 64
    assert module._contiguous_causal_query_chunk(16384) == 64
    assert module._contiguous_causal_query_chunk(16385) == 128


def test_production_prefill_never_enters_rowwise_visible_item_path(monkeypatch):
    import omlx.patches.qwen4_exp.qsa_mlx as module

    def forbidden(_mask):
        raise AssertionError("row-wise visible compaction was called")

    monkeypatch.setattr(module, "_visible_indices", forbidden)
    request = _small_request(query_tokens=19, key_tokens=43)
    result = micro_block_sparse_qsa(request, geometry=SMALL, index_key_norm=_identity)
    mx.eval(result)
    assert result.shape == (1, 19, 4, 4)


def _official_text_config():
    return {
        "model_type": "qwen4_exp_text",
        "hidden_size": 2560,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {"partial_rotary_factor": 0.25},
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "num_hidden_layers": 48,
        "full_attention_interval": 4,
        "attention_bias": False,
        "output_gate_type": "sigmoid",
        "layer_types": [
            "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
            for layer in range(48)
        ],
    }


def test_backend_matches_protocol_and_dispatches_exact_production_contract():
    backend = Qwen4ExpMLXQSABackend(index_key_norm=_identity)
    contract = Qwen4ExpQSAContract.from_config(_official_text_config())
    assert backend.supports(contract)

    # One-token production geometry is small in memory while exercising the
    # actual 24Q/2KV/256D and 4-index-head adapter path.
    request = prepare_qsa_request(
        queries=mx.ones((1, 24, 1, 256)),
        keys=mx.ones((1, 2, 1, 256)),
        values=mx.ones((1, 2, 1, 256)),
        index_queries=mx.ones((1, 1, 4, 128)),
        index_keys=mx.ones((1, 1, 128)),
        position_cos=mx.ones((1, 1, 64)),
        position_sin=mx.zeros((1, 1, 64)),
    )
    result = Qwen4ExpQSAExecutor(_official_text_config(), backend=backend)(request)
    mx.eval(result)

    assert result.shape == (1, 1, 24, 256)
    np.testing.assert_allclose(np.asarray(result), 1.0, rtol=0, atol=0)


def test_backend_source_contains_no_dense_or_deepseek_fallback():
    import omlx.patches.qwen4_exp.qsa_mlx as module

    source = inspect.getsource(module)
    assert "scaled_dot_product_attention" not in source
    assert "index_cache" not in source
    assert "deepseek" not in source.lower()
    assert "mlx.core.fast" not in source


def test_partial_rope_rejects_mismatched_shapes():
    with pytest.raises(Qwen4ExpQSAInputError, match="dimensions"):
        apply_partial_rope(
            mx.ones((2, 4)),
            mx.ones((4,)),
            mx.ones((4,)),
            rotary_dim=2,
        )
