# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiMo V2.5 mlx-lm monkey-patch (PR 1219 port)."""

import importlib
import json
import sys
import types

import mlx.core as mx
import pytest


def _minimal_config(**overrides):
    config = {
        "model_type": "mimo_v2",
        "architectures": ["MiMoV2ForCausalLM"],
        "vocab_size": 1000,
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "v_head_dim": 24,
        "rope_theta": 1000.0,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 32,
        "swa_v_head_dim": 24,
        "swa_rope_theta": 1000.0,
        "sliding_window_size": 32,
        "add_full_attention_sink_bias": False,
        "add_swa_attention_sink_bias": True,
        "hybrid_layer_pattern": [0, 1, 1, 0],
        "moe_layer_freq": [0, 1, 1, 1],
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "topk_method": "noaux_tc",
        "partial_rotary_factor": 0.5,
        "attention_bias": False,
        "layernorm_epsilon": 1e-5,
        "max_position_embeddings": 1000,
        "attention_value_scale": 0.707,
    }
    config.update(overrides)
    return config


def _load_patch_module():
    from omlx.patches.mimo_v2 import apply_mimo_v2_patch

    apply_mimo_v2_patch()
    return importlib.import_module("mlx_lm.models.mimo_v2")


def test_apply_registers_mimo_v2_module():
    module = _load_patch_module()

    assert module.__package__ == "mlx_lm.models"
    assert sys.modules["mlx_lm.models.mimo_v2"] is module
    assert sys.modules["mlx_lm.models.mimo_v2_flash"] is module

    import mlx_lm.models as models_pkg

    assert models_pkg.mimo_v2 is module
    assert models_pkg.mimo_v2_flash is module


def test_apply_is_idempotent():
    from omlx.patches.mimo_v2 import apply_mimo_v2_patch, is_applied

    first = apply_mimo_v2_patch()
    second = apply_mimo_v2_patch()

    assert is_applied() is True
    assert second is False
    assert first in (True, False)


def test_apply_replaces_upstream_module(monkeypatch):
    import mlx_lm.models as models_pkg

    import omlx.patches.mimo_v2 as patch

    upstream = types.ModuleType("mlx_lm.models.mimo_v2")
    upstream.__file__ = "/tmp/upstream/mlx_lm/models/mimo_v2.py"
    monkeypatch.setitem(sys.modules, "mlx_lm.models.mimo_v2", upstream)
    monkeypatch.setattr(models_pkg, "mimo_v2", upstream, raising=False)
    monkeypatch.setattr(patch, "_APPLIED", False)

    assert patch.apply_mimo_v2_patch() is True
    registered = sys.modules["mlx_lm.models.mimo_v2"]
    assert registered is not upstream
    assert registered.__file__.endswith("omlx/patches/mimo_v2/mimo_v2_model.py")
    assert models_pkg.mimo_v2 is registered
    assert sys.modules["mlx_lm.models.mimo_v2_flash"] is registered
    assert models_pkg.mimo_v2_flash is registered


@pytest.mark.parametrize("model_type", ["mimo_v2", "mimo_v2_flash"])
def test_get_classes_resolves_mimo_v2(model_type):
    _load_patch_module()

    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(_minimal_config(model_type=model_type))

    assert model_cls.__name__ == "Model"
    assert args_cls.__name__ == "ModelArgs"


def test_router_preserves_fp32_score_difference():
    module = _load_patch_module()
    args = module.ModelArgs.from_dict(_minimal_config(hidden_size=2))
    gate = module.MoEGate(args)
    gate.weight = mx.array([[1.0, 0.0], [1.0, 1.0]], dtype=mx.bfloat16)
    gate.e_score_correction_bias = mx.zeros((2,))
    hidden = mx.array([[[1.0, 1.0 / 256]]], dtype=mx.bfloat16)

    experts, _ = gate(hidden)

    assert experts.item() == 1


def test_mixed_cache_forward_and_continuous_batching():
    mimo_v2 = _load_patch_module()
    from mlx_lm.generate import BatchGenerator

    model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(_minimal_config()))
    cache = model.make_cache()

    assert [type(layer).__name__ for layer in cache] == [
        "KVCache",
        "RotatingKVCache",
        "RotatingKVCache",
        "KVCache",
    ]

    prefill = model(mx.array([[1, 2, 3], [4, 5, 6]]), cache=cache)
    decode = model(mx.array([[7], [8]]), cache=cache)
    mx.eval(prefill, decode)

    assert prefill.shape == (2, 3, 1000)
    assert decode.shape == (2, 1, 1000)

    generator = BatchGenerator(
        model,
        max_tokens=2,
        prefill_batch_size=2,
        completion_batch_size=2,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    )
    uids = generator.insert([[1, 2, 3], [4, 5, 6]], max_tokens=[2, 2])
    finished = []
    for _ in range(8):
        _, generation_responses = generator.next()
        finished.extend(
            response
            for response in generation_responses
            if response.finish_reason is not None
        )
        if len(finished) == 2:
            break

    assert uids == [0, 1]
    assert {response.uid for response in finished} == {0, 1}
    assert all(response.finish_reason == "length" for response in finished)


def test_window_layers_pad_the_projection_input_not_the_queries(monkeypatch):
    """Padding the q_proj input for the blocked window path is bit-exact."""
    mimo_v2 = _load_patch_module()
    from omlx.utils import fast_attention

    mx.random.seed(5)
    config = _minimal_config(sliding_window_size=128, hybrid_layer_pattern=[1, 1, 0, 1])
    model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    first_chunk = mx.random.randint(0, 1000, (1, 300))  # 84 padding rows
    second_chunk = mx.random.randint(0, 1000, (1, 257))  # 127, after a prefix
    real_pad = mimo_v2.window_query_padding
    asked = []

    def run(pad_fn):
        monkeypatch.setattr(mimo_v2, "window_query_padding", pad_fn)
        cache = model.make_cache()
        out = [model(first_chunk, cache=cache), model(second_chunk, cache=cache)]
        mx.eval(out)
        return out

    padded = run(lambda n: asked.append(n) or real_pad(n))
    assert set(asked) == {300, 257}
    unpadded = run(lambda n: 0)  # the blocked path pads the queries itself
    for a, b in zip(padded, unpadded):
        assert mx.array_equal(a, b).item()

    monkeypatch.setattr(fast_attention, "_ENABLED", False)  # masked full SDPA
    reference = run(real_pad)
    for a, b in zip(padded, reference):
        assert mx.allclose(
            a.astype(mx.float32), b.astype(mx.float32), atol=5e-2, rtol=5e-2
        ).item()


def test_sanitize_handles_fused_fp8_and_text_only_weights():
    mimo_v2 = _load_patch_module()
    config = _minimal_config(
        num_hidden_layers=2,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 1],
        num_nextn_predict_layers=1,
    )
    from omlx.patches.mlx_lm_mtp import set_mtp_active

    set_mtp_active(True)
    try:
        model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    finally:
        set_mtp_active(False)

    weights = {
        "model.layers.0.self_attn.qkv_proj.weight": mx.to_fp8(mx.ones((240, 128))),
        "model.layers.0.self_attn.qkv_proj.weight_scale_inv": mx.ones((2, 1)),
        "model.layers.0.self_attn.o_proj.weight": mx.to_fp8(mx.ones((128, 96))),
        "model.layers.0.self_attn.o_proj.weight_scale_inv": mx.ones((1, 1)),
        "visual.ignored": mx.ones((1,)),
        "audio_encoder.ignored": mx.ones((1,)),
        "speech_embeddings.ignored": mx.ones((1,)),
        "model.mtp.ignored": mx.ones((1,)),
        "model.mtp.layers.0.self_attn.qkv_proj.weight": mx.to_fp8(mx.ones((240, 128))),
        "model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv": mx.ones((2, 1)),
    }
    for projection, shape in (
        ("gate_proj", (64, 128)),
        ("up_proj", (64, 128)),
        ("down_proj", (128, 64)),
    ):
        for expert in range(2):
            weights[f"model.layers.1.mlp.experts.{expert}.{projection}.weight"] = (
                mx.ones(shape)
            )

    sanitized = model.sanitize(weights)

    assert sanitized["model.layers.0.self_attn.q_proj.weight"].shape == (128, 128)
    assert sanitized["model.layers.0.self_attn.k_proj.weight"].shape == (64, 128)
    assert sanitized["model.layers.0.self_attn.v_proj.weight"].shape == (48, 128)
    assert sanitized["model.layers.0.self_attn.o_proj.weight"].shape == (128, 96)
    assert sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"].shape == (
        2,
        64,
        128,
    )
    assert "model.mtp.ignored" in sanitized
    assert sanitized["model.mtp.layers.0.self_attn.q_proj.weight"].shape == (
        128,
        128,
    )
    assert sanitized["model.mtp.layers.0.self_attn.k_proj.weight"].shape == (64, 128)
    assert sanitized["model.mtp.layers.0.self_attn.v_proj.weight"].shape == (48, 128)
    assert not any(
        key.startswith(("visual.", "audio_encoder.", "speech_embeddings."))
        for key in sanitized
    )

    inactive = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    assert "model.mtp.ignored" not in inactive.sanitize(
        {"model.mtp.ignored": mx.ones((1,))}
    )


def test_sanitize_loads_and_splits_quantized_mtp_sidecar(monkeypatch):
    mimo_v2 = _load_patch_module()
    from omlx.patches.mlx_lm_mtp import set_mtp_active

    sidecar_path = "/models/mimo/mtp/model_mtp.safetensors"
    config = _minimal_config(
        num_nextn_predict_layers=1,
        omlx_mtp_sidecar=sidecar_path,
    )
    set_mtp_active(True)
    try:
        model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    finally:
        set_mtp_active(False)

    prefix = "model.mtp.layers.0.self_attn.qkv_proj"
    sidecar = {
        f"{prefix}.weight": mx.ones((240, 16), dtype=mx.uint32),
        f"{prefix}.scales": mx.ones((240, 2)),
        f"{prefix}.biases": mx.ones((240, 2)),
    }
    loaded = []

    def fake_load(path):
        loaded.append(path)
        return sidecar

    monkeypatch.setattr(mimo_v2.mx, "load", fake_load)
    sanitized = model.sanitize({})

    assert loaded == [sidecar_path]
    for suffix, width in (("weight", 16), ("scales", 2), ("biases", 2)):
        assert sanitized[f"model.mtp.layers.0.self_attn.q_proj.{suffix}"].shape == (
            128,
            width,
        )
        assert sanitized[f"model.mtp.layers.0.self_attn.k_proj.{suffix}"].shape == (
            64,
            width,
        )
        assert sanitized[f"model.mtp.layers.0.self_attn.v_proj.{suffix}"].shape == (
            48,
            width,
        )


def test_lightning_mtp_heads_forward_and_adapter_contract():
    mimo_v2 = _load_patch_module()
    from omlx.patches.mimo_v2.omnimodal import MiMoLanguageAdapter
    from omlx.patches.mlx_lm_mtp import (
        set_mtp_active,
        set_mtp_depth,
    )

    set_mtp_active(True)
    set_mtp_depth(3)
    try:
        args = mimo_v2.ModelArgs.from_dict(_minimal_config(num_nextn_predict_layers=3))
        model = mimo_v2.Model(args)
    finally:
        set_mtp_active(False)
        set_mtp_depth(1)

    assert len(model.mtp.layers) == 3
    assert model._omlx_mtp_decode_enabled is True
    assert model._omlx_mtp_depth == 3

    cache = model.make_cache()
    logits, hidden = model(mx.array([[1, 2]]), cache=cache, return_hidden=True)
    assert logits.shape == (1, 2, 1000)
    assert hidden.shape == (1, 2, 128)

    mtp_cache = model.make_mtp_cache()
    assert len(mtp_cache) == 3
    model.mtp_begin_cycle(mtp_cache, 3)
    first_logits, first_hidden = model.mtp_forward(
        hidden,
        mx.array([[2, 3]]),
        mtp_cache,
        return_hidden=True,
        logits_keep=1,
    )
    assert first_logits.shape == (1, 1, 1000)
    assert first_hidden.shape == (1, 2, 128)
    assert mtp_cache.layer_idx == 1

    second_logits = model.mtp_forward(first_hidden[:, -1:], mx.array([[4]]), mtp_cache)
    mx.eval(logits, first_logits, second_logits)
    assert second_logits.shape == (1, 1, 1000)
    assert mtp_cache.layer_idx == 2

    changed_hidden = mx.concatenate([hidden[:, :1], hidden[:, 1:] + 5], axis=1)
    original_logits = model.mtp_forward(
        hidden, mx.array([[2, 3]]), model.make_mtp_cache()
    )
    changed_logits = model.mtp_forward(
        changed_hidden, mx.array([[2, 3]]), model.make_mtp_cache()
    )
    assert mx.allclose(original_logits[:, :1], changed_logits[:, :1]).item()

    adapter = MiMoLanguageAdapter(model)
    assert adapter._omlx_mtp_decode_enabled is True
    assert adapter._omlx_mtp_chain is True
    adapter.mtp_begin_cycle(mtp_cache, 3)
    assert mtp_cache.layer_idx == 0
    adapter_logits, adapter_hidden = adapter(
        mx.array([[5]]), cache=model.make_cache(), return_hidden=True
    )
    mx.eval(adapter_logits, adapter_hidden)
    assert adapter_logits.shape == (1, 1, 1000)
    assert adapter_hidden.shape == (1, 1, 128)


@pytest.mark.parametrize("model_type", ["mimo_v2", "mimo_v2_flash"])
def test_pre_load_dispatch_calls_mimo_patch(tmp_path, monkeypatch, model_type):
    calls = []
    monkeypatch.setattr(
        "omlx.patches.mimo_v2.apply_mimo_v2_patch",
        lambda: calls.append(True) or True,
    )
    (tmp_path / "config.json").write_text(
        json.dumps(_minimal_config(model_type=model_type))
    )

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))

    assert calls == [True]


def test_mtp_sidecar_counts_as_checkpoint_weights(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    from omlx.utils.model_loading import _checkpoint_has_mtp_weights

    sidecar = tmp_path / "mtp" / "model_mtp.safetensors"
    sidecar.parent.mkdir()
    save_file({"model.mtp.layers.0.weight": np.ones((1,), dtype=np.float32)}, sidecar)

    assert _checkpoint_has_mtp_weights(tmp_path) is True


def test_load_text_model_injects_mtp_sidecar(tmp_path, monkeypatch):
    import omlx.utils.model_loading as ml

    sidecar = tmp_path / "mtp" / "model_mtp.safetensors"
    sidecar.parent.mkdir()
    sidecar.touch()
    captured = {}
    monkeypatch.setattr(ml, "maybe_apply_pre_load_patches", lambda *_a, **_k: None)

    def fake_load(model_name, **kwargs):
        captured["model_name"] = model_name
        captured.update(kwargs)
        return object(), object()

    monkeypatch.setattr(ml, "lm_load_compat", fake_load)
    ml.load_text_model(str(tmp_path))

    assert captured["model_name"] == str(tmp_path)
    assert captured["model_config"] == {"omlx_mtp_sidecar": str(sidecar)}


def test_multimodal_mimo_is_explicitly_routed_to_text_engine(tmp_path, caplog):
    from omlx.model_discovery import detect_model_type

    config = _minimal_config(
        vision_config={"hidden_size": 32},
        audio_config={"hidden_size": 16},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))

    with caplog.at_level("WARNING"):
        assert detect_model_type(tmp_path) == "llm"

    assert "no supported vision sidecar" in caplog.text


def test_oq_uses_mlx_lm_sanitizer_for_multimodal_mimo(monkeypatch):
    import mlx_vlm.utils as vlm_utils

    from omlx.oq import _build_model_sanitizer

    monkeypatch.setattr(
        vlm_utils,
        "get_model_and_args",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("mlx-vlm lookup must be skipped")
        ),
    )
    config = _minimal_config(
        num_hidden_layers=2,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 1],
        vision_config={"hidden_size": 32},
        audio_config={"hidden_size": 16},
    )

    sanitize = _build_model_sanitizer(config, text_only=False)

    assert sanitize is not None
    assert sanitize({"visual.ignored": mx.ones((1,))}) == {}


def _neutralize_sensitivity_deps(monkeypatch):
    """Stub _measure_sensitivity's non-routing dependencies.

    Leaves the ``is_vlm``-driven loader selection intact so a test can assert
    which load path a config takes, without loading a real model or running
    calibration.
    """
    import omlx.oq as oq
    import omlx.utils.model_loading as ml

    monkeypatch.setattr(ml, "_checkpoint_has_mtp_weights", lambda *_a, **_k: False)
    monkeypatch.setattr(ml, "_has_mtp_heads", lambda *_a, **_k: False)
    monkeypatch.setattr(ml, "maybe_apply_pre_load_patches", lambda *_a, **_k: None)
    monkeypatch.setattr(
        oq,
        "_measure_sensitivity_from_model",
        lambda *_a, **_k: {"model.layers.0": 1.0},
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "qwen2_vl", "vision_config": {"hidden_size": 32}}, True),
        ({"model_type": "mimo_v2", "vision_config": {"hidden_size": 32}}, False),
        ({"model_type": "mimo-v2", "vision_config": {"hidden_size": 32}}, False),
        ({"model_type": "llama"}, False),
        ({"model_type": "mimo_v2"}, False),
    ],
    ids=[
        "genuine_vlm_is_vlm",
        "text_only_mimo_with_vision_is_not_vlm",
        "dashed_model_type_normalizes",
        "plain_llm_is_not_vlm",
        "mimo_text_only_quant_is_not_vlm",
    ],
)
def test_is_vlm_load_predicate(config, expected):
    from omlx.oq import _is_vlm_load

    assert _is_vlm_load(config) is expected


def test_measure_sensitivity_routes_multimodal_mimo_to_mlx_lm(monkeypatch):
    # Exception path: a text-only-served mimo base ships a vision_config but must
    # load via mlx-lm, not fall through to the mlx-vlm drafter lookup.
    # _measure_sensitivity wraps the load in try/except -> {}, so record the
    # loader calls rather than raising (a raise would be swallowed).
    import mlx_vlm.utils as vlm_utils

    import omlx.utils.model_loading as ml
    from omlx.oq import _measure_sensitivity

    _neutralize_sensitivity_deps(monkeypatch)
    vlm_calls, lm_calls = [], []
    monkeypatch.setattr(
        vlm_utils, "load_model", lambda *_a, **_k: vlm_calls.append(True) or object()
    )
    monkeypatch.setattr(
        ml,
        "lm_load_compat",
        lambda *_a, **_k: lm_calls.append(True) or (object(), object()),
    )

    config = _minimal_config(
        vision_config={"hidden_size": 32},
        audio_config={"hidden_size": 16},
    )
    result = _measure_sensitivity("/unused/path", config, oq_level=4)

    assert vlm_calls == []
    assert lm_calls == [True]
    assert result == {"model.layers.0": 1.0}


def test_measure_sensitivity_routes_genuine_vlm_to_mlx_vlm(monkeypatch):
    # Happy path: a real VLM (vision_config + non-text-only model_type) still
    # loads through mlx-vlm.
    import mlx_lm.tokenizer_utils as tok_utils
    import mlx_vlm.utils as vlm_utils

    import omlx.utils.model_loading as ml
    from omlx.oq import _measure_sensitivity

    _neutralize_sensitivity_deps(monkeypatch)
    vlm_calls, lm_calls = [], []
    monkeypatch.setattr(
        vlm_utils, "load_model", lambda *_a, **_k: vlm_calls.append(True) or object()
    )
    monkeypatch.setattr(tok_utils, "load", lambda *_a, **_k: object())
    monkeypatch.setattr(
        ml,
        "lm_load_compat",
        lambda *_a, **_k: lm_calls.append(True) or (object(), object()),
    )

    config = {"model_type": "qwen2_vl", "vision_config": {"hidden_size": 32}}
    result = _measure_sensitivity("/unused/path", config, oq_level=4)

    assert vlm_calls == [True]
    assert lm_calls == []
    assert result == {"model.layers.0": 1.0}


@pytest.mark.parametrize("model_type", ["mimo_v2", "mimo_v2_flash"])
def test_official_mxfp4_checkpoint_loads_without_requantizing(tmp_path, model_type):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    module = _load_patch_module()
    config = _minimal_config(model_type=model_type)
    model = module.Model(module.ModelArgs.from_dict(config))
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        mode="mxfp4",
        class_predicate=lambda path, layer: ".switch_mlp." in path
        and hasattr(layer, "to_quantized"),
    )
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        if ".switch_mlp." not in name:
            weights[name] = value
            continue
        prefix, projection = name.split(".switch_mlp.")
        projection, suffix = projection.rsplit(".", 1)
        for expert, tensor in enumerate(value):
            key = f"{prefix}.experts.{expert}.{projection}.weight"
            if suffix == "scales":
                weights[key + "_scale"] = tensor
            else:
                weights[key] = tensor.view(mx.uint8)
    mx.eval(weights)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    config["quantization_config"] = {"quant_method": "fp8", "store_dtype": "mxfp4"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    maybe_apply_pre_load_patches(str(tmp_path))

    loaded, loaded_config = load_model(tmp_path)
    ids = mx.array([[1, 2, 3]])
    expected, actual = model(ids), loaded(ids)
    mx.eval(expected, actual)

    assert mx.array_equal(expected, actual).item()
    assert loaded_config["quantization"]["mode"] == "mxfp4"
    original = model.layers[1].mlp.switch_mlp.gate_proj
    restored = loaded.layers[1].mlp.switch_mlp.gate_proj
    assert mx.array_equal(original.weight, restored.weight).item()
    assert mx.array_equal(original.scales, restored.scales).item()

    from omlx.oq import quantize_oq_streaming

    output = tmp_path / "oq"
    quantize_oq_streaming(
        str(tmp_path),
        str(output),
        4,
        sensitivity_map_override={i: 1.0 for i in range(4)},
    )
    converted, _ = load_model(output)
    expert = converted.layers[1].mlp.switch_mlp.gate_proj
    assert mx.array_equal(original.weight, expert.weight).item()
    assert mx.array_equal(original.scales, expert.scales).item()
    assert mx.isfinite(converted(ids)).all().item()


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_mtp_partial_rollback_after_rotation(accepted):
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active
    from omlx.patches.mlx_lm_mtp.batch_generator import _call_backbone

    module = _load_patch_module()
    apply_mlx_lm_mtp_patch()
    set_mtp_active(True)
    try:
        model = module.Model(
            module.ModelArgs.from_dict(
                _minimal_config(num_nextn_predict_layers=3, sliding_window_size=8)
            )
        )
    finally:
        set_mtp_active(False)
    prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    verify = mx.array([[10, 11, 12, 13]])
    speculative, reference = model.make_cache(), model.make_cache()
    model(prompt, cache=speculative)
    model(prompt, cache=reference)
    _call_backbone(model, verify, speculative, n_confirmed=1)
    assert model.mtp_partial_rollback(speculative, accepted, 3)
    model(verify[:, : accepted + 1], cache=reference)
    continuation = mx.array([[14]])
    actual, expected = model(continuation, cache=speculative), model(
        continuation, cache=reference
    )
    mx.eval(actual, expected)
    assert [c.offset for c in speculative] == [c.offset for c in reference]
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()


def test_mtp_draft_clone_preserves_head_index_and_isolates_cache():
    from omlx.patches.mlx_lm_mtp import set_mtp_active, set_mtp_depth
    from omlx.patches.mlx_lm_mtp.batch_generator import _clone_mtp_head_cache

    module = _load_patch_module()
    set_mtp_active(True)
    set_mtp_depth(3)
    try:
        model = module.Model(
            module.ModelArgs.from_dict(_minimal_config(num_nextn_predict_layers=3))
        )
    finally:
        set_mtp_active(False)
        set_mtp_depth(1)
    cache = model.make_mtp_cache()
    hidden = mx.ones((1, 1, 128))
    ids = mx.array([[1]])
    model.mtp_begin_cycle(cache, 3)
    model.mtp_forward(hidden, ids, cache)
    clone = _clone_mtp_head_cache(cache)
    model.mtp_forward(hidden, ids, clone)
    assert clone.layer_idx == 2
    assert cache.layer_idx == 1
    assert clone[1].offset == 1
    assert cache[1].offset == 0


def _mtp_model(depth=3, **overrides):
    from omlx.patches.mlx_lm_mtp import set_mtp_active, set_mtp_depth

    module = _load_patch_module()
    config = _minimal_config(num_nextn_predict_layers=3, **overrides)
    set_mtp_active(True)
    set_mtp_depth(depth)
    try:
        return module.Model(module.ModelArgs.from_dict(config))
    finally:
        set_mtp_active(False)
        set_mtp_depth(1)


def _parallel_reference(model, hidden, tokens, drafts=()):
    """Each head over a fresh cache: layer k pairs trunk row q with token
    q+k+1 (``tokens[:, q]`` is token q+1; ``drafts`` continue the stream)."""
    embed = model.model.embed_tokens
    stream = mx.concatenate([tokens] + [mx.array([[int(d)]]) for d in drafts], axis=1)
    n = int(hidden.shape[1])
    outputs = []
    for k, (layer, cache) in enumerate(zip(model.mtp.layers, model.make_mtp_cache())):
        m = min(n, n + len(drafts) - k)
        outputs.append(layer(hidden[:, :m], embed(stream[:, k : k + m]), cache))
    return outputs


def _fold_chunks(model, cache, hidden, tokens, chunks, *, begin):
    outs, start = [], 0
    for size in chunks:
        if begin:
            model.mtp_begin_cycle(cache, 3)
        _, out = model.mtp_forward(
            hidden[:, start : start + size],
            tokens[:, start : start + size],
            cache,
            return_hidden=True,
            logits_keep=1,
        )
        outs.append(out)
        start += size
    assert start == hidden.shape[1]
    return mx.concatenate(outs, axis=1)


def _draft_two(model, cache, last_out, d1, d2):
    from omlx.patches.mlx_lm_mtp.batch_generator import _clone_mtp_head_cache

    clone = _clone_mtp_head_cache(cache)
    # The generator passes the previous head output; MiMo's heads ignore it.
    _, out1 = model.mtp_forward(last_out[:, -1:], mx.array([[d1]]), clone, return_hidden=True)
    _, out2 = model.mtp_forward(out1[:, -1:], mx.array([[d2]]), clone, return_hidden=True)
    return clone, out1, out2


def _close(a, b):
    return mx.allclose(a, b, atol=1e-4, rtol=1e-4).item()


@pytest.mark.parametrize(
    "window,chunks", [(32, [3, 1, 4, 2, 2]), (8, [8, 1, 1, 5, 2, 3])]
)
def test_mtp_heads_fold_and_draft_from_the_trunk_hidden(window, chunks):
    model = _mtp_model(sliding_window_size=window)
    n, d1, d2 = sum(chunks), 7, 9
    mx.random.seed(0)
    hidden = mx.random.normal((1, n, 128))
    tokens = mx.random.randint(0, 1000, (1, n))
    ref = _parallel_reference(model, hidden, tokens, drafts=(d1, d2))

    cache = model.make_mtp_cache()
    out0 = _fold_chunks(model, cache, hidden, tokens, chunks, begin=True)
    clone, out1, out2 = _draft_two(model, cache, out0, d1, d2)
    mx.eval(out0, out1, out2, *ref)

    assert _close(out0, ref[0])
    assert out1.shape[1] == 1 and _close(out1, ref[1][:, -1:])
    assert out2.shape[1] == 2 and _close(out2, ref[2][:, -2:])
    # Committed history only in the persistent cache; drafts live on the clone.
    assert [c.offset for c in cache] == [n, n - 1, n - 2]
    assert [c.offset for c in clone] == [n, n, n]
    assert cache.layer_idx == 1 and clone.layer_idx == 3
    assert cache.draft_tokens == () and len(clone.draft_tokens) == 2


def test_mtp_priming_folds_feed_every_head_without_a_cycle():
    # Prompt priming folds the prompt tail and the activation seam without
    # mtp_begin_cycle; every head's history must advance all the same.
    model = _mtp_model()
    n = 6
    mx.random.seed(1)
    hidden = mx.random.normal((1, n + 1, 128))
    tokens = mx.random.randint(0, 1000, (1, n + 1))
    ref = _parallel_reference(model, hidden, tokens, drafts=(11, 12))

    cache = model.make_mtp_cache()
    _fold_chunks(model, cache, hidden[:, :n], tokens[:, :n], [5, 1], begin=False)
    assert cache.in_cycle is False
    assert [c.offset for c in cache] == [n, n - 1, n - 2]
    out0 = _fold_chunks(model, cache, hidden[:, n:], tokens[:, n:], [1], begin=True)
    _, out1, out2 = _draft_two(model, cache, out0, 11, 12)
    mx.eval(out0, out1, out2, *ref)

    assert _close(out0, ref[0][:, -1:])
    assert _close(out1, ref[1][:, -1:])
    assert _close(out2, ref[2][:, -2:])


def test_mtp_fold_skips_heads_past_the_loaded_depth():
    model = _mtp_model(depth=1)
    cache = model.make_mtp_cache()
    model.mtp_begin_cycle(cache, 1)
    model.mtp_forward(mx.ones((1, 4, 128)), mx.array([[1, 2, 3, 4]]), cache)
    assert [c.offset for c in cache] == [4, 0, 0]
    assert cache.trunk_rows is None
    with pytest.raises(ValueError):
        model.mtp_forward(mx.ones((1, 1, 128)), mx.array([[5]]), cache)


def test_mtp_predictor_module_runs_the_heads_in_parallel():
    model = _mtp_model()
    mx.random.seed(2)
    hidden = mx.random.normal((1, 5, 128))
    tokens = mx.random.randint(0, 1000, (1, 5))
    outs = model.mtp(hidden, tokens, model.model.embed_tokens, model.make_mtp_cache())
    ref = _parallel_reference(model, hidden, tokens)
    mx.eval(*outs, *ref)
    assert [o.shape[1] for o in outs] == [5, 4, 3]
    assert all(_close(o, r) for o, r in zip(outs, ref))


@pytest.mark.parametrize("layout", ["root", "nested"])
def test_oq_preserves_mtp_shards_and_calibrates_all_heads(tmp_path, layout):
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model

    from omlx.oq import (
        OQImatrixCollector,
        _collect_mtp_head_imatrix,
        quantize_oq_streaming,
    )
    from omlx.patches.mlx_lm_mtp import set_mtp_active

    module = _load_patch_module()
    config = _minimal_config(
        num_nextn_predict_layers=3,
        v_head_dim=32,
        swa_v_head_dim=32,
        vocab_size=1024,
    )
    set_mtp_active(True)
    try:
        model = module.Model(module.ModelArgs.from_dict(config))
    finally:
        set_mtp_active(False)
    weights = dict(tree_flatten(model.parameters()))
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    trunk = {k: v for k, v in weights.items() if not k.startswith("model.mtp.")}
    head = {k: v for k, v in weights.items() if k.startswith("model.mtp.")}
    mx.save_safetensors(str(source / "model.safetensors"), trunk)
    sidecar = source / (
        "model_mtp.safetensors" if layout == "root" else "mtp/model_mtp.safetensors"
    )
    sidecar.parent.mkdir(exist_ok=True)
    mx.save_safetensors(str(sidecar), head)
    output = tmp_path / "output"
    quantize_oq_streaming(
        str(source),
        str(output),
        4,
        group_size=32,
        preserve_mtp=True,
        sensitivity_map_override={i: 1.0 for i in range(4)},
    )
    set_mtp_active(True)
    try:
        loaded, output_config = load_model(output)
    finally:
        set_mtp_active(False)
    assert output_config["num_nextn_predict_layers"] == 3
    assert len(loaded.mtp.layers) == 3
    ids = mx.array([[1, 2, 3, 4]])
    loaded.model.norm.weight = mx.linspace(0.5, 1.5, config["hidden_size"])
    _, hidden = loaded(ids, return_hidden=True)
    head = loaded.mtp.layers[0]
    expected_input = mx.concatenate(
        [head.enorm(loaded.model.embed_tokens(ids[:, 1:])), head.hnorm(hidden[:, :-1])],
        axis=-1,
    )
    expected_energy = mx.sum(expected_input.astype(mx.float32) ** 2, axis=(0, 1))
    collector = OQImatrixCollector()
    collector.install(loaded)
    try:
        assert _collect_mtp_head_imatrix(loaded, ids, hidden)
        for index in range(3):
            assert f"model.mtp.layers.{index}.eh_proj" in collector.entries
        actual_energy = collector.entries["model.mtp.layers.0.eh_proj"].in_sum2
        assert mx.allclose(mx.array(actual_energy), expected_energy, atol=1e-5).item()
    finally:
        collector.restore(loaded)


def _fp8_fused_qkv_sidecar(prefix, *, tp, n_h=4, n_kv=2, hd=32, vhd=24, cols=128):
    """A pre-sharded FP8 fused qkv whose rows say which shard/part they are (values chosen to be exact in e4m3)."""
    from omlx.patches.mimo_v2.fused_qkv_layout import (
        FUSED_QKV_BLOCK_SIZE,
        fused_qkv_part_rows,
        fused_qkv_shard_rows,
    )

    q_pr, k_pr, v_pr = fused_qkv_part_rows(n_h, n_kv, hd, vhd, tp)
    actual_pr, padded_pr = fused_qkv_shard_rows(n_h, n_kv, hd, vhd, tp)
    # On disk the shards are stored back to back without padding; only the
    # block scales are laid out on the per-shard padded grid.
    rows = []
    for shard in range(tp):
        rows.extend([1.0 + shard] * q_pr + [10.0 + shard] * k_pr + [32.0 + 4 * shard] * v_pr)
    assert len(rows) == tp * actual_pr
    weight = mx.to_fp8(mx.array(rows, dtype=mx.float32)[:, None] * mx.ones((1, cols)))
    scale = mx.ones((tp * padded_pr // FUSED_QKV_BLOCK_SIZE, cols // FUSED_QKV_BLOCK_SIZE))
    return {f"{prefix}.weight": weight, f"{prefix}.weight_scale_inv": scale}, (q_pr, k_pr, v_pr)


def _sanitized_qkv_rows(sidecar, config_extra, monkeypatch):
    mimo_v2 = _load_patch_module()
    from omlx.patches.mlx_lm_mtp import set_mtp_active

    config = _minimal_config(
        num_nextn_predict_layers=1,
        omlx_mtp_sidecar="/models/mimo/mtp/model_mtp.safetensors",
        **config_extra,
    )
    set_mtp_active(True)
    try:
        model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    finally:
        set_mtp_active(False)
    monkeypatch.setattr(mimo_v2.mx, "load", lambda path: sidecar)
    out = model.sanitize({})
    prefix = "model.mtp.layers.0.self_attn"
    return {
        name: out[f"{prefix}.{name}_proj.weight"][:, 0].astype(mx.float32).tolist()
        for name in ("q", "k", "v")
    }


def test_fp8_sidecar_qkv_assumes_the_official_tp4_layout_when_main_is_split(monkeypatch):
    prefix = "model.mtp.layers.0.self_attn.qkv_proj"
    sidecar, (q_pr, k_pr, v_pr) = _fp8_fused_qkv_sidecar(prefix, tp=4)
    rows = _sanitized_qkv_rows(sidecar, {}, monkeypatch)
    # Each projection is the concatenation of its per-shard parts, in order.
    assert rows["q"] == [1.0 + s for s in range(4) for _ in range(q_pr)]
    assert rows["k"] == [10.0 + s for s in range(4) for _ in range(k_pr)]
    assert rows["v"] == [32.0 + 4 * s for s in range(4) for _ in range(v_pr)]


def test_fp8_sidecar_qkv_tp_can_be_pinned(monkeypatch):
    prefix = "model.mtp.layers.0.self_attn.qkv_proj"
    sidecar, (q_pr, k_pr, v_pr) = _fp8_fused_qkv_sidecar(prefix, tp=2)
    rows = _sanitized_qkv_rows(sidecar, {"omlx_mtp_sidecar_tp": 2}, monkeypatch)
    assert rows["q"] == [1.0 + s for s in range(2) for _ in range(q_pr)]
    assert rows["k"] == [10.0 + s for s in range(2) for _ in range(k_pr)]
    assert rows["v"] == [32.0 + 4 * s for s in range(2) for _ in range(v_pr)]


def test_prompt_priming_folds_the_prompt_into_the_mimo_heads():
    mimo_v2 = _load_patch_module()
    from omlx.patches.mlx_lm_mtp import prompt_priming, set_mtp_active

    set_mtp_active(True)
    try:
        model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(_minimal_config(num_nextn_predict_layers=1)))
    finally:
        set_mtp_active(False)
    cache = model.make_cache()
    prompt_priming.drop_ctx(model)
    model(mx.array([[1, 2, 3, 4, 5]]), cache=cache)
    # Five prompt tokens give four (hidden, next-token) pairs for the heads.
    assert prompt_priming.prime_ctx_stats(model) == 4
    # A plain decode step extends the context; the activation forward
    # (return_hidden=True) is left for take_primed to fold.
    model(mx.array([[6]]), cache=cache)
    assert prompt_priming.prime_ctx_stats(model) == 5
    model(mx.array([[7]]), cache=cache, return_hidden=True)
    assert prompt_priming.prime_ctx_stats(model) == 5
    prompt_priming.drop_ctx(model)

    # Longer prompts prime only the head window's tail (window 32 -> 16 here).
    cache = model.make_cache()
    model(mx.array([list(range(1, 41))]), cache=cache)
    assert prompt_priming.prime_ctx_stats(model) == 16
    prompt_priming.drop_ctx(model)


def test_tail_priming_skips_chunks_a_later_chunk_replaces(monkeypatch):
    """Only the last prefill chunk's tail reaches MiMo's heads; earlier chunks
    skip the capture (no hidden state requested), and the primed head cache is
    identical to capturing every chunk."""
    mimo_v2 = _load_patch_module()
    from omlx.patches.mlx_lm_mtp import prompt_priming, set_mtp_active

    set_mtp_active(True)
    try:
        model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(_minimal_config(num_nextn_predict_layers=1)))
    finally:
        set_mtp_active(False)
    prompt = list(range(1, 81))
    chunks = [prompt[:40], prompt[40:79]]  # the last prompt token is decoded

    def prime(skip):
        if not skip:
            monkeypatch.setattr(
                prompt_priming, "claim_superseded_tail_chunk", lambda *a: False
            )
        requested = []
        inner_call = type(model.model).__call__

        def spy(self, inputs, cache=None, input_embeddings=None, return_hidden=False, **kw):
            requested.append(return_hidden)
            return inner_call(self, inputs, cache, input_embeddings, return_hidden=return_hidden, **kw)

        monkeypatch.setattr(type(model.model), "__call__", spy)
        prompt_priming.drop_ctx(model)
        setattr(
            model,
            prompt_priming._PLAN_ATTR,
            prompt_priming._PrimePlan(
                request_id="r",
                prompt_tokens=tuple(prompt),
                # Fusion's plan records the scheduler's cached prefix (a fresh
                # prompt here, as the scheduler passes it).
                cached_tokens=0,
                block_size=0,
                prefix_cache=None,
            ),
        )
        cache = model.make_cache()
        for chunk in chunks:
            model(mx.array([chunk]), cache=cache)
        ctx = prompt_priming._find_ctx(model)
        state = [
            a
            for c in ctx.mtp_cache
            for sub in (getattr(c, "caches", None) or (c,))
            for a in (getattr(sub, "keys", None), getattr(sub, "values", None))
            if a is not None
        ]
        mx.eval(state, ctx.pending_hidden)
        stats = prompt_priming.prime_ctx_stats(model)
        prompt_priming.drop_ctx(model)
        monkeypatch.undo()
        return requested, stats, state, ctx.pending_hidden

    req_skip, stats_skip, state_skip, pending_skip = prime(skip=True)
    req_all, stats_all, state_all, pending_all = prime(skip=False)
    assert req_skip == [False, True]
    assert req_all == [True, True]
    assert stats_skip == stats_all == 16
    assert len(state_skip) == len(state_all) > 0
    for a, b in zip(state_skip, state_all):
        assert a.shape == b.shape and mx.array_equal(a, b).item()
    assert mx.array_equal(pending_skip, pending_all).item()


# Fusion starts no generic head timeline on a restored prompt suffix; the
# tests below cover the tail-only exception that lets MiMo's last prefill
# chunk start after claim_superseded_tail_chunk skipped the earlier ones.


def _mimo_tail_model():
    mimo_v2 = _load_patch_module()
    from omlx.patches.mlx_lm_mtp import set_mtp_active

    set_mtp_active(True)
    try:
        model = mimo_v2.Model(
            mimo_v2.ModelArgs.from_dict(_minimal_config(num_nextn_predict_layers=1))
        )
    finally:
        set_mtp_active(False)
    mx.eval(model.parameters())
    return model


def _mimo_head_state(ctx):
    state = [
        a
        for c in ctx.mtp_cache
        for sub in (getattr(c, "caches", None) or (c,))
        for a in (getattr(sub, "keys", None), getattr(sub, "values", None))
        if a is not None
    ]
    mx.eval(state, ctx.pending_hidden)
    return state


def _set_mimo_plan(prompt_priming, model, prompt, cached_tokens):
    prompt_priming.drop_ctx(model)
    setattr(
        model,
        prompt_priming._PLAN_ATTR,
        prompt_priming._PrimePlan(
            request_id="r",
            prompt_tokens=tuple(prompt),
            cached_tokens=cached_tokens,
            block_size=0,
            prefix_cache=None,
        ),
    )


def _assert_same_head(a, b):
    (folded_a, state_a, pending_a), (folded_b, state_b, pending_b) = a, b
    assert folded_a == folded_b
    assert len(state_a) == len(state_b) > 0
    for x, y in zip(state_a, state_b):
        assert x.shape == y.shape and mx.array_equal(x, y).item()
    assert mx.array_equal(pending_a, pending_b).item()


def test_fusion_short_last_chunk_after_a_superseded_chunk_primes_like_every_chunk(
    monkeypatch,
):
    """A last chunk no longer than the head tail starts the tail-only context
    itself (a fresh context would not take the restart) and keeps exactly
    the pairs the every-chunk capture's restart keeps."""
    from omlx.patches.mlx_lm_mtp import prompt_priming

    model = _mimo_tail_model()
    tail = int(model._omlx_mtp_prime_tail)
    prompt = [(3 * i + 2) % 60 + 1 for i in range(40 + tail // 2 + 1)]
    chunks = [prompt[:40], prompt[40:-1]]  # the last prompt token is decoded

    def prime(skip):
        if not skip:
            monkeypatch.setattr(
                prompt_priming, "claim_superseded_tail_chunk", lambda *a: False
            )
        _set_mimo_plan(prompt_priming, model, prompt, 0)
        cache = model.make_cache()
        for chunk in chunks:
            model(mx.array([chunk]), cache=cache)
        ctx = prompt_priming._find_ctx(model)
        assert ctx is not None and ctx.tail_only
        out = (ctx.folded, _mimo_head_state(ctx), ctx.pending_hidden)
        prompt_priming.drop_ctx(model)
        monkeypatch.undo()
        return out

    skipped = prime(skip=True)
    assert skipped[0] == len(chunks[-1]) - 1
    _assert_same_head(skipped, prime(skip=False))


def test_fusion_restored_prefix_keeps_the_fail_closed_start():
    """A prefix restored from the prefix cache (the plan records it, no head
    context exists) still starts no head timeline; a cold prompt's own
    superseded prefix does."""
    from omlx.patches.mlx_lm_mtp import prompt_priming

    model = _mimo_tail_model()
    tail = int(model._omlx_mtp_prime_tail)
    prompt = [(3 * i + 2) % 60 + 1 for i in range(81)]
    for cached_tokens, primed in ((40, False), (0, True)):
        cache = model.make_cache()
        # The backbone cache holds the first 40 tokens; no head context does.
        model.model(mx.array([prompt[:40]]), cache)
        _set_mimo_plan(prompt_priming, model, prompt, cached_tokens)
        model(mx.array([prompt[40:80]]), cache=cache)
        ctx = prompt_priming._find_ctx(model)
        if primed:
            assert ctx is not None and ctx.tail_only and ctx.folded == tail
        else:
            assert ctx is None
        prompt_priming.drop_ctx(model)


def test_fusion_batched_prefill_scope_primes_a_superseded_row(monkeypatch):
    """In the batched prefill scope a row prefilled from offset 0 whose first
    chunk was superseded primes like the every-chunk capture."""
    from omlx.patches.mlx_lm_mtp import prompt_priming

    model = _mimo_tail_model()
    prompt = [(5 * i + 3) % 60 + 1 for i in range(81)]
    uid = 7

    def prime(skip):
        if not skip:
            monkeypatch.setattr(
                prompt_priming, "claim_superseded_tail_chunk", lambda *a: False
            )
        prompt_priming.drop_ctx(model)
        prompt_priming.release_uids(model, [uid])
        cache = model.make_cache()
        with prompt_priming.prefill_scope(model, [uid], [prompt[:80]], cache):
            for chunk in (prompt[:40], prompt[40:80]):
                model(mx.array([chunk]), cache=cache)
        _, state = prompt_priming._owned(model)
        ctx, _ = state.uids[uid]
        assert ctx is not None and ctx.tail_only
        out = (ctx.folded, _mimo_head_state(ctx), ctx.pending_hidden)
        prompt_priming.release_uids(model, [uid])
        monkeypatch.undo()
        return out

    _assert_same_head(prime(skip=True), prime(skip=False))


def test_tail_priming_computes_the_last_layer_for_the_tail_rows_only(monkeypatch):
    """With a tail-only head, a long prefill chunk computes the last (full
    attention) layer's outputs for the tail rows only; the backbone cache and
    the primed head state match the full-width forward."""
    mimo_v2 = _load_patch_module()
    from omlx.patches.mlx_lm_mtp import prompt_priming, set_mtp_active

    config = _minimal_config(num_nextn_predict_layers=1)
    set_mtp_active(True)
    try:
        model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    finally:
        set_mtp_active(False)
    if model.model.layers[-1].is_sliding_window:
        import pytest

        pytest.skip("minimal config ends with a sliding-window layer")
    mx.eval(model.parameters())
    prompt = [(5 * i + 1) % 64 + 1 for i in range(41)]

    def prime(tail_rows):
        monkeypatch.setattr(prompt_priming, "tail_hidden_rows", lambda *a: tail_rows)
        seen = []
        inner_call = type(model.model).__call__

        def spy(self, inputs, cache=None, input_embeddings=None, return_hidden=False, hidden_tail=None):
            seen.append(hidden_tail)
            return inner_call(self, inputs, cache, input_embeddings, return_hidden=return_hidden, hidden_tail=hidden_tail)

        monkeypatch.setattr(type(model.model), "__call__", spy)
        prompt_priming.drop_ctx(model)
        cache = model.make_cache()
        model(mx.array([prompt[:40]]), cache=cache)
        ctx = prompt_priming._find_ctx(model)
        head = [
            a
            for c in ctx.mtp_cache
            for sub in (getattr(c, "caches", None) or (c,))
            for a in (getattr(sub, "keys", None), getattr(sub, "values", None))
            if a is not None
        ]
        trunk = [a for c in cache for a in (c.keys, c.values) if a is not None]
        mx.eval(head, trunk, ctx.pending_hidden)
        stats = prompt_priming.prime_ctx_stats(model)
        prompt_priming.drop_ctx(model)
        monkeypatch.undo()
        return seen, stats, head, trunk, ctx.pending_hidden

    seen_t, stats_t, head_t, trunk_t, pend_t = prime(model._omlx_mtp_prime_tail + 1)
    seen_f, stats_f, head_f, trunk_f, pend_f = prime(None)
    assert seen_t == [model._omlx_mtp_prime_tail + 1] and seen_f == [None]
    assert stats_t == stats_f == model._omlx_mtp_prime_tail
    for a, b in zip(trunk_t, trunk_f):
        assert mx.array_equal(a, b).item()
    assert len(head_t) == len(head_f) > 0
    for a, b in zip(head_t + [pend_t], head_f + [pend_f]):
        assert a.shape == b.shape
        assert mx.allclose(a, b, atol=1e-2, rtol=1e-2).item()
