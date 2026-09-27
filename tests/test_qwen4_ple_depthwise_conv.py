# SPDX-License-Identifier: Apache-2.0
"""Dilated depthwise conv kernel for the Qwen4 PLE short conv: bit-equal to mx.conv1d."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    monkeypatch.setitem(language._DEPTHWISE_CONV_STATE, "enabled", True)
    monkeypatch.setitem(language._DEPTHWISE_CONV_STATE, "validated", False)
    return language


def _conv(channels, taps=4, dilation=3, dtype=mx.bfloat16, bias=False):
    conv = nn.Conv1d(
        channels, channels, kernel_size=taps, dilation=dilation, groups=channels, bias=bias
    )
    conv.weight = (mx.random.normal(conv.weight.shape) * 0.3).astype(dtype)
    if bias:
        conv.bias = mx.random.normal((channels,)).astype(dtype)
    return conv


@needs_metal
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("dilation,taps", [(3, 4), (1, 4), (2, 3)])
def test_kernel_is_bit_equal_to_conv1d(_fresh_state, dtype, dilation, taps):
    language = _fresh_state
    mx.random.seed(dilation * 10 + taps)
    conv = _conv(320, taps=taps, dilation=dilation, dtype=dtype)
    for batch, rows in ((1, 1), (1, 7), (2, 33), (1, 517)):
        x = mx.random.normal((batch, rows + (taps - 1) * dilation, 320)).astype(dtype)
        got = language._depthwise_conv1d(conv, x)
        ref = conv(x)
        mx.eval(got, ref)
        assert got.shape == ref.shape and got.dtype == ref.dtype
        view = {mx.float32: mx.uint32}.get(dtype, mx.uint16)
        assert mx.array_equal(got.view(view), ref.view(view)).item()
    assert language._DEPTHWISE_CONV_STATE["validated"]


@needs_metal
def test_one_pipeline_serves_every_length(_fresh_state, monkeypatch):
    language = _fresh_state
    names = []
    real = mx.fast.metal_kernel

    def spy(*args, **kwargs):
        names.append(kwargs.get("name"))
        return real(*args, **kwargs)

    monkeypatch.setattr(language.mx.fast, "metal_kernel", spy)
    monkeypatch.setitem(language._DEPTHWISE_CONV_STATE, "kernel", None)
    conv = _conv(64)
    for rows in (5, 9, 100):
        mx.eval(language._depthwise_conv1d(conv, mx.ones((1, rows + 9, 64), mx.bfloat16)))
    assert names == ["omlx_qwen4_depthwise_conv1d"]


def test_unsupported_layouts_use_conv1d(_fresh_state, monkeypatch):
    language = _fresh_state
    calls = []
    monkeypatch.setattr(
        language.mx.fast,
        "metal_kernel",
        lambda *a, **k: calls.append(k) or (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setitem(language._DEPTHWISE_CONV_STATE, "kernel", None)
    x = mx.ones((1, 12, 64), mx.bfloat16)
    grouped = nn.Conv1d(64, 64, kernel_size=4, dilation=3, groups=32, bias=False)
    grouped.weight = grouped.weight.astype(mx.bfloat16)
    with_bias = _conv(64, bias=True)
    padded = _conv(64)
    padded.padding = 1
    mixed_dtype = _conv(64, dtype=mx.float32)
    for conv in (grouped, with_bias, padded, mixed_dtype):
        mx.eval(language._depthwise_conv1d(conv, x))
    monkeypatch.setitem(language._DEPTHWISE_CONV_STATE, "enabled", False)
    mx.eval(language._depthwise_conv1d(_conv(64), x))
    assert calls == []


@needs_metal
def test_validation_mismatch_disables_kernel(_fresh_state, monkeypatch):
    language = _fresh_state

    class Wrong:
        def __call__(self, inputs, **kwargs):
            return [mx.zeros(kwargs["output_shapes"][0], kwargs["output_dtypes"][0])]

    monkeypatch.setitem(language._DEPTHWISE_CONV_STATE, "kernel", Wrong())
    conv = _conv(64)
    x = mx.random.normal((1, 20, 64)).astype(mx.bfloat16)
    out = language._depthwise_conv1d(conv, x)
    assert mx.array_equal(out, conv(x)).item()
    assert language._DEPTHWISE_CONV_STATE["enabled"] is False
    assert language._DEPTHWISE_CONV_STATE["validated"] is False
