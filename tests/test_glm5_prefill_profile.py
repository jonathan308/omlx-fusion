# SPDX-License-Identifier: Apache-2.0
"""Strict gating and one-eval behavior for the GLM-5.3 prefill profiler."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_glm5_next_compat as compat


@pytest.fixture(autouse=True)
def _apply_glm5_next_compat():
    compat.apply_mlx_vlm_glm5_next_compat_patch()


def _tiny_text_config():
    from mlx_vlm.models.glm5_next import TextConfig

    return TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        qk_nope_head_dim=8,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        index_topk=4,
        index_head_dim=8,
        index_n_heads=2,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["sparse", "sparse"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        n_group=1,
        topk_group=1,
    )


def _parse_profile(caplog):
    message = next(
        record.message
        for record in caplog.records
        if record.message.startswith("[glm5-prefill-profile]")
    )
    return json.loads(message.split("] ", 1)[1])


def test_prefill_profile_gate_is_env_and_trace_scoped(monkeypatch):
    import mlx_vlm.models.glm5_next.language as language

    model = SimpleNamespace(fa_idx=0)
    cache = [SimpleNamespace(offset=0)]
    inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    monkeypatch.setattr(language, "_PREFILL_PROFILE_CALLS", 0)
    monkeypatch.delenv("OMLX_GLM5_PREFILL_PROFILE", raising=False)

    assert language._new_prefill_profile_sample(
        model,
        inputs,
        None,
        None,
        cache,
        skip_lm_head=True,
        benchmark_trace=True,
        source="language",
    ) is None

    monkeypatch.setenv("OMLX_GLM5_PREFILL_PROFILE", "1")
    assert language._new_prefill_profile_sample(
        model,
        inputs,
        None,
        None,
        cache,
        skip_lm_head=True,
        benchmark_trace=False,
        source="language",
    ) is None
    sample = language._new_prefill_profile_sample(
        model,
        inputs,
        None,
        None,
        cache,
        skip_lm_head=True,
        benchmark_trace=True,
        source="language",
    )
    assert sample is not None
    assert (sample.chunk_tokens, sample.context_tokens) == (4, 0)

    # The pinned DFlash request lacks benchmark_trace; the isolated process env
    # is its explicit second gate.
    dflash_sample = language._new_prefill_profile_sample(
        model,
        inputs,
        None,
        None,
        cache,
        skip_lm_head=False,
        benchmark_trace=False,
        source="dflash",
    )
    assert dflash_sample is not None


@pytest.mark.parametrize(
    ("inputs", "inputs_embeds", "mask", "cache"),
    [
        (mx.array([[1]], dtype=mx.int32), None, None, [SimpleNamespace(offset=0)]),
        (mx.ones((2, 4), dtype=mx.int32), None, None, [SimpleNamespace(offset=0)]),
        (
            mx.ones((1, 4), dtype=mx.int32),
            mx.zeros((1, 4, 8)),
            None,
            [SimpleNamespace(offset=0)],
        ),
        (
            mx.ones((1, 4), dtype=mx.int32),
            None,
            mx.ones((1, 4), dtype=mx.bool_),
            [SimpleNamespace(offset=0)],
        ),
        (mx.ones((1, 4), dtype=mx.int32), None, None, None),
    ],
)
def test_prefill_profile_fails_closed_for_non_cold_text_shapes(
    monkeypatch,
    inputs,
    inputs_embeds,
    mask,
    cache,
):
    import mlx_vlm.models.glm5_next.language as language

    monkeypatch.setenv("OMLX_GLM5_PREFILL_PROFILE", "1")
    assert language._new_prefill_profile_sample(
        SimpleNamespace(fa_idx=0),
        inputs,
        inputs_embeds,
        mask,
        cache,
        skip_lm_head=False,
        benchmark_trace=True,
        source="language",
    ) is None


def test_language_prefill_profile_reports_phases_with_one_eval(
    monkeypatch,
    caplog,
):
    import mlx_vlm.models.glm5_next.language as language

    config = _tiny_text_config()
    model = language.LanguageModel(config)
    reference_cache = model.make_cache()
    profiled_cache = model.make_cache()
    inputs = mx.array([[2, 3, 4, 5, 6, 7]], dtype=mx.int32)

    monkeypatch.delenv("OMLX_GLM5_PREFILL_PROFILE", raising=False)
    reference = model(inputs, cache=reference_cache, num_logits_to_keep=1)

    eval_calls = []
    original_eval_once = language._profile_eval_once

    def tracked_eval(value):
        eval_calls.append(value)
        return original_eval_once(value)

    monkeypatch.setattr(language, "_profile_eval_once", tracked_eval)
    monkeypatch.setenv("OMLX_GLM5_PREFILL_PROFILE", "1")
    monkeypatch.setenv("OMLX_GLM5_PREFILL_PROFILE_WARMUP", "0")
    monkeypatch.setenv("OMLX_GLM5_PREFILL_PROFILE_INTERVAL", "1")
    caplog.set_level("INFO", logger=language.__name__)
    actual = model(
        inputs,
        cache=profiled_cache,
        num_logits_to_keep=1,
        _omlx_benchmark_trace=True,
    )
    mx.eval(reference.logits)

    assert len(eval_calls) == 1
    assert mx.array_equal(actual.logits, reference.logits).item()
    report = _parse_profile(caplog)
    assert report["source"] == "language"
    assert report["mode"] == "lazy_submit_one_final_eval"
    assert report["chunk_tokens"] == 6
    assert report["context_before"] == 0
    assert report["lm_head_skipped"] is False
    assert report["phase_calls"]["kda_attention"] == 1
    assert report["phase_calls"]["dsa_attention"] == 1
    assert report["phase_calls"]["dsa_indexer"] == 1
    assert report["phase_calls"]["mhc"] == 8
    assert report["phase_calls"]["moe_router"] == 2
    assert report["phase_calls"]["routed_moe"] == 2
    assert report["phase_calls"]["shared_moe"] == 2
    assert "logits" in report["phase_submit_ms"]
    assert set(report["layer_submit_ms"]) == {"0", "1"}
    assert report["route_delta"]["indexer_fallback"] == 1


def test_default_language_path_never_forces_profiler_eval(monkeypatch):
    import mlx_vlm.models.glm5_next.language as language

    model = language.LanguageModel(_tiny_text_config())
    cache = model.make_cache()
    monkeypatch.delenv("OMLX_GLM5_PREFILL_PROFILE", raising=False)

    def must_not_eval(*_args, **_kwargs):
        raise AssertionError("disabled profiler inserted an evaluation boundary")

    monkeypatch.setattr(language, "_profile_eval_once", must_not_eval)
    output = model(
        mx.array([[2, 3, 4, 5]], dtype=mx.int32),
        cache=cache,
        _omlx_benchmark_trace=True,
    )
    mx.eval(output.logits)


def test_dflash_target_capture_uses_same_profile_and_default_is_noop(
    monkeypatch,
    caplog,
):
    import mlx_vlm.models.glm5_next.language as language

    from omlx.patches import dflash_glm5

    Glm5NextTargetOps = dflash_glm5.Glm5NextTargetOps

    wrapper = language.LanguageModel(_tiny_text_config())
    target = SimpleNamespace(language_model=wrapper)
    ops = Glm5NextTargetOps()
    inputs = mx.array([[2, 3, 4, 5, 6, 7]], dtype=mx.int32)

    monkeypatch.delenv("OMLX_GLM5_PREFILL_PROFILE", raising=False)
    disabled_cache = wrapper.make_cache()

    def must_not_sample(*_args, **_kwargs):
        raise AssertionError("disabled DFlash path consulted the profiler")

    original_new_sample = language._new_prefill_profile_sample
    monkeypatch.setattr(language, "_new_prefill_profile_sample", must_not_sample)
    logits, captured = ops.forward_with_hidden_capture(
        target,
        input_ids=inputs,
        cache=disabled_cache,
        capture_layer_ids={0, 1, 2},
        logits_last_only=True,
    )
    mx.eval(logits, captured)

    # Env alone must not profile verify/capture calls outside the runtime's
    # cold-prefill generator scope.
    monkeypatch.setenv("OMLX_GLM5_PREFILL_PROFILE", "1")
    verify_cache = wrapper.make_cache()
    ops.forward_with_hidden_capture(
        target,
        input_ids=inputs,
        cache=verify_cache,
        capture_layer_ids={0, 1, 2},
        logits_last_only=True,
    )

    monkeypatch.setattr(language, "_new_prefill_profile_sample", original_new_sample)
    caplog.clear()
    caplog.set_level("INFO", logger=language.__name__)
    profiled_cache = wrapper.make_cache()
    dflash_glm5._GLM5_PREFILL_PROFILE_SCOPE.active = True
    try:
        logits, captured = ops.forward_with_hidden_capture(
            target,
            input_ids=inputs,
            cache=profiled_cache,
            capture_layer_ids={0, 1, 2},
            logits_last_only=True,
        )
    finally:
        del dflash_glm5._GLM5_PREFILL_PROFILE_SCOPE.active

    report = _parse_profile(caplog)
    assert report["source"] == "dflash"
    assert report["phase_calls"]["capture_mhc"] == 3
    assert report["phase_calls"]["logits"] == 1
    assert logits.shape == (1, 1, 128)
    assert set(captured) == {-1, 0, 1, 2}
