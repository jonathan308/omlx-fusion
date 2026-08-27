from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches.qwen4_exp import nvfp4


def _config():
    return {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": 48,
            "num_experts": 512,
        },
        "qwen4_exp_artifact": {"layout": "qwen4-exp-modelopt-nvfp4-v1"},
        "quantization": {
            "bits": 4,
            "group_size": 16,
            "mode": "nvfp4",
            "scope": "qwen4_exp_routed_experts",
            "modelopt_global_scale": True,
        },
    }


def test_nvfp4_config_is_strict():
    assert nvfp4.is_supported_config(_config())
    for path, wrong in (
        (("text_config", "num_experts"), 256),
        (("quantization", "group_size"), 32),
        (("quantization", "mode"), "mxfp4"),
        (("qwen4_exp_artifact", "layout"), "unknown"),
    ):
        config = _config()
        config[path[0]][path[1]] = wrong
        assert not nvfp4.is_supported_config(config)


def test_scaled_switch_matches_native_nvfp4_gather():
    module = nvfp4.make_scaled_switch(32, 4, 2)
    x = mx.random.normal((1, 1, 1, 32)).astype(mx.bfloat16)
    indices = mx.array([[[1]]], dtype=mx.int32)
    module.global_scale = mx.array([0.5, 1.75], dtype=mx.float32)
    expected = (
        mx.gather_qmm(
            x,
            module.weight,
            module.scales,
            rhs_indices=indices,
            transpose=True,
            group_size=16,
            bits=4,
            mode="nvfp4",
            sorted_indices=False,
        )
        * module.global_scale[indices].astype(x.dtype)[..., None, None]
    )
    actual = module(x, indices)
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected, actual).item())


def test_modelopt_experts_repack_without_scale_folding(monkeypatch):
    monkeypatch.setattr(nvfp4, "_LAYERS", 1)
    monkeypatch.setattr(nvfp4, "_EXPERTS", 2)
    weights = {"model.language_model.embed_tokens.weight": mx.ones((2, 2))}
    shapes = {
        "gate_proj": ((640, 1280), (640, 160)),
        "up_proj": ((640, 1280), (640, 160)),
        "down_proj": ((2560, 320), (2560, 40)),
    }
    for expert in range(2):
        for projection, (weight_shape, scale_shape) in shapes.items():
            prefix = f"model.language_model.layers.0.mlp.experts.{expert}.{projection}"
            weights[f"{prefix}.weight"] = mx.full(
                weight_shape, expert + 1, dtype=mx.uint8
            )
            weights[f"{prefix}.weight_scale"] = mx.full(
                scale_shape, expert + 3, dtype=mx.uint8
            )
            weights[f"{prefix}.weight_scale_2"] = mx.array(
                0.25 + expert, dtype=mx.float32
            )
            weights[f"{prefix}.input_scale"] = mx.array(1.0, dtype=mx.float32)

    output = nvfp4.transform_weights_exact(weights)
    assert "model.language_model.embed_tokens.weight" in output
    for projection, (weight_shape, scale_shape) in shapes.items():
        prefix = f"model.language_model.layers.0.mlp.switch_mlp.{projection}"
        assert output[f"{prefix}.weight"].shape == (
            2,
            weight_shape[0],
            weight_shape[1] // 4,
        )
        assert output[f"{prefix}.scales"].shape == (2, *scale_shape)
        assert output[f"{prefix}.global_scale"].shape == (2,)
        mx.eval(output[f"{prefix}.global_scale"])
        assert output[f"{prefix}.global_scale"].tolist() == [0.25, 1.25]
    assert not any("input_scale" in key for key in output)


def test_fp8_ple_table_is_rejected():
    with pytest.raises(nvfp4.Qwen4ExpNVFP4Error, match="BF16 SSD"):
        nvfp4.transform_weights_exact(
            {
                "model.language_model.layers.1.ple.ple_embedding."
                "ngram_embedding.shard_0.weight": mx.zeros((1,), dtype=mx.uint8)
            }
        )
