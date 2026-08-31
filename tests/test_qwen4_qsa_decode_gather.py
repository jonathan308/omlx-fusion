# SPDX-License-Identifier: Apache-2.0
"""Batch-one Qwen4 QSA decode gather regression tests."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


def _tiny_text_config():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import TextConfig

    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )


def test_qwen4_decode_gathers_budget_and_tail_and_matches_official(monkeypatch):
    config = _tiny_text_config()
    import mlx_vlm.models.qwen4_exp.language as language
    import mlx_vlm.models.qwen4_exp.qsa_fast as qsa_fast

    attention = language.Qwen4ExpAttention(config)
    mx.eval(attention.parameters())
    fast_cache = language.QSAKVCache()
    reference_cache = language.QSAKVCache()

    mx.random.seed(19)
    prefix = mx.random.normal((1, 10, config.hidden_size))
    decode = mx.random.normal((1, 1, config.hidden_size))
    fast_prefix = attention(prefix, mask="causal", cache=fast_cache)
    reference_prefix = attention(prefix, mask="causal", cache=reference_cache)
    mx.eval(fast_prefix, reference_prefix)

    gathered_lengths = []
    original_sdpa = qsa_fast._decode_qsa_sdpa

    def tracked_sdpa(queries, keys, values, scale):
        gathered_lengths.append(int(keys.shape[2]))
        return original_sdpa(queries, keys, values, scale)

    monkeypatch.setattr(qsa_fast, "_decode_qsa_sdpa", tracked_sdpa)
    actual = attention(decode, cache=fast_cache)

    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_decode_eligible",
        lambda *args, **kwargs: False,
    )
    expected = attention(decode, cache=reference_cache)
    mx.eval(actual, expected)

    # key_len=11, budget=8, incomplete causal tail=1.
    assert gathered_lengths == [9]
    assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()
    assert mx.array_equal(
        mx.argmax(actual, axis=-1),
        mx.argmax(expected, axis=-1),
    ).item()
    assert fast_cache.offset == reference_cache.offset == 11
    for fast_value, reference_value in zip(
        fast_cache.state,
        reference_cache.state,
    ):
        assert mx.array_equal(fast_value, reference_value).item()


@pytest.mark.parametrize("verify_width", [4, 6, 8, 9])
def test_qwen4_verify_window_uses_direct_qsa_and_matches_official(
    monkeypatch,
    verify_width,
):
    config = _tiny_text_config()
    import mlx_vlm.models.qwen4_exp.language as language

    attention = language.Qwen4ExpAttention(config)
    monkeypatch.setattr(language, "_QSA_DIRECT_VERIFY_MIN_TOKENS", 0)
    mx.eval(attention.parameters())
    fast_cache = language.QSAKVCache()
    reference_cache = language.QSAKVCache()

    mx.random.seed(29)
    prefix = mx.random.normal((1, 10, config.hidden_size))
    verify = mx.random.normal((1, verify_width, config.hidden_size))
    mx.eval(
        attention(prefix, mask="causal", cache=fast_cache),
        attention(prefix, mask="causal", cache=reference_cache),
    )

    calls = []
    original = language.contiguous_causal_gathered_qsa

    def tracked(*args, **kwargs):
        calls.append(
            (
                args[0].shape[2],
                args[1].shape[2],
                kwargs["mtp_m6_target_verify"],
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(language, "contiguous_causal_gathered_qsa", tracked)
    actual = attention(
        verify,
        mask="causal",
        cache=fast_cache,
        target_verify=True,
    )

    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_verify_eligible",
        lambda *args, **kwargs: False,
    )
    expected = attention(
        verify,
        mask="causal",
        cache=reference_cache,
        target_verify=True,
    )
    mx.eval(actual, expected)

    assert calls == [(verify_width, 10 + verify_width, verify_width == 6)]
    assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()
    assert mx.array_equal(
        mx.argmax(actual, axis=-1),
        mx.argmax(expected, axis=-1),
    ).item()
    assert fast_cache.offset == reference_cache.offset == 10 + verify_width
    for fast_value, reference_value in zip(
        fast_cache.state,
        reference_cache.state,
    ):
        assert mx.array_equal(fast_value, reference_value).item()


@pytest.mark.parametrize(
    ("verify_width", "accepted"),
    [
        (verify_width, accepted)
        for verify_width in (8, 9)
        for accepted in range(verify_width)
    ],
)
def test_qwen4_mixed_qsa_gdn_verify_rollback_matches_canonical_prefix(
    monkeypatch,
    verify_width,
    accepted,
):
    """Every depth-7/8 accepted prefix restores canonical mixed-layer state.

    QSA storage is representation-identical.  GDN's FP32 recurrent replay can
    differ from the shorter canonical graph by a final rounding ULP, so it is
    bounded tightly and followed by an end-to-end greedy-token parity check.
    """

    config = _tiny_text_config()
    import mlx_vlm.models.qwen4_exp.language as language

    root_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
    )
    model = language.LanguageModel(config, root_config)
    monkeypatch.setattr(language, "_QSA_DIRECT_VERIFY_MIN_TOKENS", 0)
    mx.eval(model.parameters())

    verify_cache = model.make_cache()
    singleton_cache = model.make_cache()
    prefix = mx.arange(2, 12, dtype=mx.int32)[None]
    verify_tokens = mx.arange(
        12,
        12 + verify_width,
        dtype=mx.int32,
    )[None]
    mx.eval(
        model(prefix, cache=verify_cache).logits,
        model(prefix, cache=singleton_cache).logits,
    )

    verified = model(
        verify_tokens,
        cache=verify_cache,
        return_hidden=True,
    )
    mx.eval(verified.logits, *verified.gdn_states)
    full_qsa_state = tuple(value * 1 for value in verify_cache[1].state)
    mx.eval(*full_qsa_state)
    gdn_state = verified.gdn_states[0]
    conv_input = gdn_state[9]
    kernel_size = int(gdn_state[10])
    intermediate_states = gdn_state[11]
    expected_conv = conv_input[:, accepted + 1 : accepted + kernel_size]
    expected_recurrent = intermediate_states[:, accepted]
    mx.eval(expected_conv, expected_recurrent)
    model.rollback_speculative_cache(
        verify_cache,
        verified.gdn_states,
        accepted=accepted,
        block_size=verify_width,
    )

    retained = accepted + 1
    retained_total = prefix.shape[1] + retained
    expected_qsa_state = (
        full_qsa_state[0][:, :, :retained_total],
        full_qsa_state[1][:, :, :retained_total],
        full_qsa_state[2][:, :retained_total],
        full_qsa_state[3][..., :retained_total],
    )
    assert mx.array_equal(verify_cache[0][0], expected_conv).item()
    assert mx.array_equal(verify_cache[0][1], expected_recurrent).item()
    for actual, expected in zip(verify_cache[1].state, expected_qsa_state):
        mx.eval(actual, expected)
        assert mx.array_equal(actual, expected).item()

    mx.eval(
        model(
            verify_tokens[:, :retained],
            cache=singleton_cache,
        ).logits
    )

    for actual_cache, expected_cache in zip(verify_cache, singleton_cache):
        actual_state = actual_cache.state
        expected_state = expected_cache.state
        assert len(actual_state) == len(expected_state)
        for actual, expected in zip(actual_state, expected_state):
            if actual is None or expected is None:
                assert actual is expected
                continue
            mx.eval(actual, expected)
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            if actual.dtype == mx.int32:
                assert mx.array_equal(actual, expected).item()
            else:
                assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()

    probe = mx.array([[31]], dtype=mx.int32)
    actual_next = model(probe, cache=verify_cache).logits
    expected_next = model(probe, cache=singleton_cache).logits
    mx.eval(actual_next, expected_next)
    assert mx.allclose(actual_next, expected_next, rtol=2e-5, atol=2e-5).item()
    assert mx.array_equal(
        mx.argmax(actual_next[:, -1], axis=-1),
        mx.argmax(expected_next[:, -1], axis=-1),
    ).item()


def test_qwen4_verify_window_eligibility_fails_closed(monkeypatch):
    config = _tiny_text_config()
    import mlx_vlm.models.qwen4_exp.language as language

    attention = language.Qwen4ExpAttention(config)
    monkeypatch.setattr(language, "_QSA_DIRECT_VERIFY_MIN_TOKENS", 0)
    cache = language.QSAKVCache()
    prefix = mx.random.normal((1, 10, config.hidden_size))
    mx.eval(attention(prefix, mask="causal", cache=cache))
    window = mx.random.normal((1, 6, config.hidden_size))

    assert attention._gathered_text_verify_eligible(
        window, None, cache, None, None, True
    )
    assert not attention._gathered_text_verify_eligible(
        window, None, cache, None, None, False
    )
    assert not attention._gathered_text_verify_eligible(
        window, None, cache, mx.zeros((3, 1, 6), dtype=mx.int32), None, True
    )
    assert not attention._gathered_text_verify_eligible(
        mx.broadcast_to(window, (2, 6, config.hidden_size)),
        None,
        cache,
        None,
        None,
        True,
    )

    incomplete = language.QSAKVCache()
    incomplete.offset = cache.offset
    assert not attention._gathered_text_verify_eligible(
        window, None, incomplete, None, None, True
    )


def test_qwen4_language_wrapper_routes_2d_text_positions_to_gather(monkeypatch):
    config = _tiny_text_config()
    import mlx_vlm.models.qwen4_exp.language as language

    root_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
    )
    model = language.LanguageModel(config, root_config)
    mx.eval(model.parameters())
    fast_cache = model.make_cache()
    reference_cache = model.make_cache()
    calls = []

    original_prefill = language.Qwen4ExpAttention._gathered_text_prefill
    original_decode = language.Qwen4ExpAttention._gathered_text_decode
    original_prefill_eligible = (
        language.Qwen4ExpAttention._gathered_text_prefill_eligible
    )
    original_decode_eligible = language.Qwen4ExpAttention._gathered_text_decode_eligible

    def tracked_prefill(self, x, cache, position_ids=None):
        calls.append(("prefill", position_ids.ndim, position_ids.shape))
        return original_prefill(self, x, cache, position_ids)

    def tracked_decode(self, x, cache, position_ids=None):
        calls.append(("decode", position_ids.ndim, position_ids.shape))
        return original_decode(self, x, cache, position_ids)

    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_prefill",
        tracked_prefill,
    )
    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_decode",
        tracked_decode,
    )

    prefix = mx.arange(2, 12, dtype=mx.int32)[None]
    fast_prefix = model(prefix, cache=fast_cache)

    # Replay the same wrapper-owned text sequence through the official path.
    model._position_ids = None
    model._rope_deltas = None
    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_prefill_eligible",
        lambda *args, **kwargs: False,
    )
    reference_prefix = model(prefix, cache=reference_cache)
    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_prefill_eligible",
        original_prefill_eligible,
    )

    decode_token = mx.array([[12]], dtype=mx.int32)
    actual = model(decode_token, cache=fast_cache)
    monkeypatch.setattr(
        language.Qwen4ExpAttention,
        "_gathered_text_decode_eligible",
        lambda *args, **kwargs: False,
    )
    expected = model(decode_token, cache=reference_cache)
    mx.eval(fast_prefix.logits, reference_prefix.logits, actual.logits, expected.logits)

    assert calls == [
        ("prefill", 2, (1, 10)),
        ("decode", 2, (1, 1)),
    ]
    assert mx.allclose(
        fast_prefix.logits,
        reference_prefix.logits,
        rtol=2e-5,
        atol=2e-5,
    ).item()
    assert mx.allclose(actual.logits, expected.logits, rtol=2e-5, atol=2e-5).item()
    assert mx.array_equal(
        mx.argmax(actual.logits[:, -1], axis=-1),
        mx.argmax(expected.logits[:, -1], axis=-1),
    ).item()


def test_qwen4_decode_keeps_official_path_until_complete_block_crossover(
    monkeypatch,
):
    config = _tiny_text_config()
    import mlx_vlm.models.qwen4_exp.language as language

    attention = language.Qwen4ExpAttention(config)
    cache = language.QSAKVCache()
    prefix = mx.random.normal((1, 8, config.hidden_size))
    attention(prefix, mask="causal", cache=cache)

    def must_not_gather(*args, **kwargs):
        raise AssertionError("decode at the QSA block budget must stay official")

    monkeypatch.setattr(language, "contiguous_causal_gathered_qsa_decode", must_not_gather)
    output = attention(mx.random.normal((1, 1, config.hidden_size)), cache=cache)
    mx.eval(output)

    # Nine visible rows still contain only four complete two-token blocks.
    assert cache.offset == 9
    assert output.shape == (1, 1, config.hidden_size)


def test_qwen4_decode_gather_eligibility_fails_closed_for_general_paths():
    config = _tiny_text_config()
    import mlx_vlm.models.qwen4_exp.language as language

    attention = language.Qwen4ExpAttention(config)
    cache = language.QSAKVCache()
    prefix = mx.random.normal((1, 10, config.hidden_size))
    mx.eval(attention(prefix, mask="causal", cache=cache))
    token = mx.random.normal((1, 1, config.hidden_size))

    assert attention._gathered_text_decode_eligible(
        token, None, cache, None, None, False
    )
    assert not attention._gathered_text_decode_eligible(
        token, "left_padded_decode", cache, None, None, False
    )
    assert attention._gathered_text_decode_eligible(
        token, None, cache, mx.array([[10]], dtype=mx.int32), None, False
    )
    assert not attention._gathered_text_decode_eligible(
        token,
        None,
        cache,
        mx.array([[[10]], [[10]], [[10]]], dtype=mx.int32),
        None,
        False,
    )
    assert not attention._gathered_text_decode_eligible(
        token, None, cache, None, None, True
    )
    assert not attention._gathered_text_decode_eligible(
        mx.broadcast_to(token, (2, 1, config.hidden_size)),
        None,
        cache,
        None,
        None,
        False,
    )

    incomplete = language.QSAKVCache()
    incomplete.offset = cache.offset
    assert not attention._gathered_text_decode_eligible(
        token, None, incomplete, None, None, False
    )


@pytest.mark.parametrize("key_tokens", [4097, 32769])
def test_qwen4_decode_gather_stays_budget_bounded_at_long_cache(
    monkeypatch,
    key_tokens,
):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.qsa_fast as qsa_fast

    mx.random.seed(23)
    queries = mx.random.normal((1, 4, 1, 8)).astype(mx.float32)
    keys = mx.random.normal((1, 2, key_tokens, 8)).astype(mx.float32)
    values = mx.random.normal((1, 2, key_tokens, 8)).astype(mx.float32)
    index_queries = mx.random.normal((1, 1, 2, 8)).astype(mx.float32)
    pooled = mx.random.normal((1, key_tokens // 2, 8)).astype(mx.float32)

    gathered_lengths = []
    original_sdpa = qsa_fast._decode_qsa_sdpa

    def tracked_sdpa(q, k, v, scale):
        gathered_lengths.append(int(k.shape[2]))
        return original_sdpa(q, k, v, scale)

    monkeypatch.setattr(qsa_fast, "_decode_qsa_sdpa", tracked_sdpa)
    output = qsa_fast.contiguous_causal_gathered_qsa_decode(
        queries,
        keys,
        values,
        index_queries,
        pooled,
        num_query_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        indexer_head_dim=8,
        compress_ratio=2,
        token_budget=8,
    )
    mx.eval(output)

    assert gathered_lengths == [9]
    assert output.shape == (1, 1, 4, 8)


def test_qwen4_decode_sdpa_fails_closed_when_native_shape_is_rejected(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.qsa_fast as qsa_fast
    from omlx.custom_kernels.decode_fast import fast

    q = mx.random.normal((1, 4, 1, 8))
    k = mx.random.normal((1, 2, 9, 8))
    v = mx.random.normal((1, 2, 9, 8))
    monkeypatch.setattr(fast, "NATIVE_AVAILABLE", True)
    monkeypatch.setattr(
        fast,
        "_ext",
        SimpleNamespace(sdpa_decode_supported=lambda *args: False),
    )

    def must_not_run(*args, **kwargs):
        raise AssertionError("rejected decode_fast shape must use MLX SDPA")

    monkeypatch.setattr(fast, "sdpa_decode", must_not_run)
    actual = qsa_fast._decode_qsa_sdpa(q, k, v, 8**-0.5)
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=8**-0.5)
    mx.eval(actual, expected)

    assert mx.array_equal(actual, expected).item()


def test_qwen4_decode_sdpa_uses_native_only_after_capability_accepts(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.qsa_fast as qsa_fast
    from omlx.custom_kernels.decode_fast import fast

    q = mx.random.normal((1, 24, 1, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, 2051, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, 2051, 256)).astype(mx.bfloat16)
    calls = []

    def supported(queries, keys, values):
        calls.append((queries.shape, keys.shape, values.shape, "probe"))
        return True

    def native(queries, keys, values, scale, causal=False):
        calls.append((scale, causal, "native"))
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
        )

    monkeypatch.setattr(fast, "NATIVE_AVAILABLE", True)
    monkeypatch.setattr(
        fast,
        "_ext",
        SimpleNamespace(sdpa_decode_supported=supported),
    )
    monkeypatch.setattr(fast, "sdpa_decode", native)
    actual = qsa_fast._decode_qsa_sdpa(q, k, v, 256**-0.5)
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=256**-0.5)
    mx.eval(actual, expected)

    assert calls == [
        (q.shape, k.shape, v.shape, "probe"),
        (256**-0.5, False, "native"),
    ]
    assert mx.array_equal(actual, expected).item()
