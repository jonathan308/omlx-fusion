# SPDX-License-Identifier: Apache-2.0
"""Deferred MLP residual writes in Qwen4ExpModel prefill: bit-identical outputs."""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def _model():
    """Three decoder layers (the middle, linear one carries PLE) whose hyper-connections
    take the fused prefill kernels: hc_count 4, 64-aligned widths, 4-bit
    group-64 projections."""
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    config = _tiny_config()
    config.text_config = dataclasses.replace(
        config.text_config,
        hidden_size=512,
        hc_count=4,
        hc_lowrank=64,
        ple_embed_dim=512,
        num_hidden_layers=3,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        ple_layer_ids=[2],
    )
    mx.random.seed(5)
    model = LanguageModel(config.text_config, config)
    nn.quantize(
        model,
        group_size=64,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear)
        and "hyper_connection" in path,
    )
    # Checkpoint-like quantization statistics (see test_qwen4_hc_fused).
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.scales = (
                mx.abs(mx.random.normal(module.scales.shape)) * 0.01 + 0.002
            ).astype(mx.bfloat16)
            module.biases = (mx.random.normal(module.biases.shape) * 0.005).astype(
                mx.bfloat16
            )
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model, config


def _logits(model, inputs):
    out = model(inputs)
    out = getattr(out, "logits", out)
    mx.eval(out)
    return out


@needs_metal
@pytest.mark.parametrize("rows", [40, 129])
def test_deferred_write_is_bit_identical_and_fused(monkeypatch, rows):
    from mlx_vlm.models.qwen4_exp import hc_fused, language

    model, config = _model()
    layers = model.model.layers
    assert "ple" in layers[1] and "ple" not in layers[2]
    assert hc_fused.prefill_compatible(
        layers[0].attn_hyper_connection,
        mx.zeros((1, rows, 4 * 512), dtype=mx.bfloat16),
    )
    inputs = mx.random.randint(0, config.text_config.vocab_size, (1, rows))

    monkeypatch.setattr(language, "_DEFERRED_HC_WRITE", False)
    eager = _logits(model, inputs)

    writes = []
    real = hc_fused.prefill_forward

    def spy(module, hyper_input, write=None):
        writes.append(
            (
                module is layers[2].attn_hyper_connection,
                write is not None,
            )
        )
        return real(module, hyper_input, write)

    monkeypatch.setattr(hc_fused, "prefill_forward", spy)
    monkeypatch.setattr(language, "_DEFERRED_HC_WRITE", True)
    deferred = _logits(model, inputs)

    # Layer 1 carries PLE, so layer 0's write is applied eagerly there; layer
    # 1's write is fused into layer 2's stream norm; layer 2 writes eagerly.
    assert (True, True) in writes
    assert eager.dtype == deferred.dtype
    view = mx.uint32 if eager.dtype == mx.float32 else mx.uint16
    assert mx.array_equal(eager.view(view), deferred.view(view)).item()


@needs_metal
def test_decode_rows_and_kill_switch_never_defer(monkeypatch):
    from mlx_vlm.models.qwen4_exp import language

    model, config = _model()
    deferred = []
    real_call = language.Qwen4ExpDecoderLayer.__call__

    def spy(self, *args, **kwargs):
        deferred.append(bool(kwargs.get("defer_write")))
        return real_call(self, *args, **kwargs)

    monkeypatch.setattr(language.Qwen4ExpDecoderLayer, "__call__", spy)
    # 16 rows take the fused decode kernels, which have no pending-write input.
    _logits(model, mx.random.randint(0, config.text_config.vocab_size, (1, 16)))
    assert deferred and not any(deferred)

    deferred.clear()
    monkeypatch.setattr(language, "_DEFERRED_HC_WRITE", False)
    _logits(model, mx.random.randint(0, config.text_config.vocab_size, (1, 40)))
    assert deferred and not any(deferred)

    deferred.clear()
    monkeypatch.setattr(language, "_DEFERRED_HC_WRITE", True)
    _logits(model, mx.random.randint(0, config.text_config.vocab_size, (1, 40)))
    # Every layer but the last defers.
    assert deferred == [True, True, False]
