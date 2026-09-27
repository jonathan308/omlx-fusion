# SPDX-License-Identifier: Apache-2.0
"""Tensor-unit (NAX) hyper-connection prefill: bitwise parity with the MLX matmul path."""

from __future__ import annotations

from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

HC, HIDDEN, LOWRANK = 4, 2560, 320
WIDTH = HC * HIDDEN


def _nax() -> bool:
    if not mx.metal.is_available():
        return False
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


needs_nax = pytest.mark.skipif(not _nax(), reason="requires Metal tensor units (M5)")


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def _module(bits_down: int, bits_up: int, bits_inject: int | None, seed: int):
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

    mx.random.seed(seed)
    module = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
    nn.Module.__init__(module)
    module.hc_count, module.hidden_size, module.hc_lowrank = HC, HIDDEN, LOWRANK
    module.hc_norm = Qwen4ExpRMSNorm(WIDTH, group_size=HIDDEN, eps=1e-6)
    module.hc_norm.weight = (mx.random.normal((WIDTH,)) * 0.05).astype(mx.bfloat16)
    module.input_mix_weight_down = nn.QuantizedLinear(
        WIDTH, LOWRANK, bias=False, group_size=64, bits=bits_down
    )
    module.input_mix_weight_up = nn.QuantizedLinear(
        LOWRANK, WIDTH, bias=False, group_size=64, bits=bits_up
    )
    if bits_inject is not None:
        module.block_inject_weight = nn.QuantizedLinear(
            WIDTH, HC, bias=False, group_size=64, bits=bits_inject
        )
    for name in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        projection = getattr(module, name, None)
        if projection is not None:
            # Checkpoint-like statistics (see test_qwen4_hc_fused).
            projection.scales = (
                mx.abs(mx.random.normal(projection.scales.shape)) * 0.01 + 0.002
            ).astype(mx.bfloat16)
            projection.biases = (
                mx.random.normal(projection.biases.shape) * 0.005
            ).astype(mx.bfloat16)
    mx.eval(module.parameters())
    return module


def _inputs(rows: int, write: bool):
    hyper = (mx.random.normal((1, rows, WIDTH)) * 2).astype(mx.bfloat16)
    pending = None
    if write:
        branch = mx.random.normal((1, rows, HIDDEN)).astype(mx.bfloat16)
        gate = (2 * mx.sigmoid(mx.random.normal((1, rows, HC)))).astype(mx.bfloat16)
        pending = (branch, gate)
    mx.eval(hyper, pending)
    return hyper, pending


def _bits(a: mx.array) -> mx.array:
    return a.view(mx.uint16) if a.dtype == mx.bfloat16 else a


def _mlx_path(monkeypatch, module, hyper, write):
    from mlx_vlm.models.qwen4_exp import hc_fused

    with monkeypatch.context() as patch:
        patch.setattr(hc_fused, "_NAX_DISABLED", True)
        out = hc_fused.prefill_forward(module, hyper, write)
        mx.eval(out)
    return out


def _spies(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_prefill_nax

    spies = {}
    for name in ("norm_inject", "down_silu", "up_tail"):
        spies[name] = Mock(wraps=getattr(hc_prefill_nax, name))
        monkeypatch.setattr(hc_prefill_nax, name, spies[name])
    return spies


@needs_nax
@pytest.mark.parametrize("bits", [(4, 4, 4), (5, 5, 5), (6, 6, 6), (8, 8, 8), (4, 6, 5)])
@pytest.mark.parametrize("rows", [65, 3265, 4133])
@pytest.mark.parametrize("write", [False, True])
def test_prefill_is_bit_identical_to_mlx_path(monkeypatch, bits, rows, write):
    """Mixed input, written stream and inject weights match the six-dispatch path
    bit for bit; the down projection runs on the NAX kernel from 3265 rows on."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(*bits, seed=rows + 7 * bits[0])
    hyper, write_args = _inputs(rows, write)
    reference = _mlx_path(monkeypatch, module, hyper, write_args)
    spies = _spies(monkeypatch)
    out = hc_fused.prefill_forward(module, hyper, write_args)
    mx.eval(out)
    assert spies["up_tail"].call_count == 1
    assert spies["down_silu"].call_count == (1 if rows >= 3265 else 0)
    mixed, passthrough, injection = out
    assert mixed.shape == (1, rows, HIDDEN) and injection.shape == (1, rows, HC)
    if not write:
        assert passthrough is hyper
    for observed, expected in zip(out, reference):
        assert observed.dtype == expected.dtype == mx.bfloat16
        assert mx.array_equal(_bits(observed), _bits(expected)).item()


@needs_nax
def test_batched_prefill_is_bit_identical(monkeypatch):
    """[batch, seq] rows flatten like MLX's non-batched quantized matmul."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4, 5, 6, seed=9)
    hyper = (mx.random.normal((2, 2048, WIDTH)) * 2).astype(mx.bfloat16)
    branch = mx.random.normal((2, 2048, HIDDEN)).astype(mx.bfloat16)
    gate = (2 * mx.sigmoid(mx.random.normal((2, 2048, HC)))).astype(mx.bfloat16)
    reference = _mlx_path(monkeypatch, module, hyper, (branch, gate))
    spies = _spies(monkeypatch)
    out = hc_fused.prefill_forward(module, hyper, (branch, gate))
    mx.eval(out)
    assert spies["down_silu"].call_count == 1
    assert out[0].shape == (2, 2048, HIDDEN) and out[2].shape == (2, 2048, HC)
    for observed, expected in zip(out, reference):
        assert mx.array_equal(_bits(observed), _bits(expected)).item()


def test_plain_qmm_nax_mirrors_mlx_dispatch():
    from mlx_vlm.models.qwen4_exp.hc_prefill_nax import plain_qmm_nax

    # Up projection (N = 10240): single 64-row tiles use MLX's split K.
    assert not plain_qmm_nax(64, WIDTH)
    assert plain_qmm_nax(65, WIDTH)
    # Down projection (N = 320): release wheels split K below 801 rows and
    # M5 source builds split on the tensor-unit path below 3265 rows.
    assert not plain_qmm_nax(800, LOWRANK)
    assert not plain_qmm_nax(3264, LOWRANK)
    assert plain_qmm_nax(3265, LOWRANK)
    assert plain_qmm_nax(8191, LOWRANK)


@needs_nax
def test_few_rows_and_missing_inject_keep_mlx_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    spies = _spies(monkeypatch)
    module = _module(4, 4, 4, seed=1)
    hyper, _ = _inputs(64, False)
    mx.eval(hc_fused.prefill_forward(module, hyper))
    no_inject = _module(4, 4, None, seed=2)
    hyper, _ = _inputs(128, False)
    mx.eval(hc_fused.prefill_forward(no_inject, hyper))
    assert not spies["norm_inject"].called


@needs_nax
def test_kill_switch(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    spies = _spies(monkeypatch)
    monkeypatch.setattr(hc_fused, "_NAX_DISABLED", True)
    module = _module(4, 4, 4, seed=4)
    hyper, _ = _inputs(128, False)
    mx.eval(hc_fused.prefill_forward(module, hyper))
    assert not spies["norm_inject"].called


@needs_nax
def test_kernel_failure_falls_back_to_mlx_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused, hc_prefill_nax

    module = _module(4, 4, 4, seed=5)
    hyper, write = _inputs(128, True)
    reference = _mlx_path(monkeypatch, module, hyper, write)
    monkeypatch.setattr(hc_fused, "_NAX_BROKEN", False)
    monkeypatch.setattr(
        hc_prefill_nax, "up_tail", Mock(side_effect=RuntimeError("compile"))
    )
    out = hc_fused.prefill_forward(module, hyper, write)
    mx.eval(out)
    for observed, expected in zip(out, reference):
        assert mx.array_equal(_bits(observed), _bits(expected)).item()
    # Later calls do not retry the broken kernels.
    assert hc_fused._NAX_BROKEN
    assert not hc_fused._nax_prefill_ok(module, 128, WIDTH, module.block_inject_weight)


@needs_nax
def test_first_call_probe_keeps_mlx_path_on_mismatch(monkeypatch):
    """A specialization whose first result differs from the MLX path in any
    bit (e.g. an MLX with different quantized-matmul arithmetic) is replaced
    by the MLX result and the NAX kernels are not used again."""
    from mlx_vlm.models.qwen4_exp import hc_fused, hc_prefill_nax

    module = _module(4, 4, 4, seed=6)
    hyper, write = _inputs(128, True)
    reference = _mlx_path(monkeypatch, module, hyper, write)
    real = hc_prefill_nax.up_tail

    def off_by_one_ulp(*args, **kwargs):
        mixed, inj = real(*args, **kwargs)
        return (mixed.view(mx.uint16) ^ 1).view(mx.bfloat16), inj

    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_NAX_BROKEN", False)
    monkeypatch.setattr(hc_prefill_nax, "up_tail", off_by_one_ulp)
    out = hc_fused.prefill_forward(module, hyper, write)
    mx.eval(out)
    for observed, expected in zip(out, reference):
        assert mx.array_equal(_bits(observed), _bits(expected)).item()
    assert hc_fused._NAX_BROKEN


@needs_nax
def test_first_call_probe_accepts_bitwise_results(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(5, 6, 8, seed=7)
    hyper, write = _inputs(3300, True)
    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_NAX_BROKEN", False)
    mx.eval(hc_fused.prefill_forward(module, hyper, write))
    assert not hc_fused._NAX_BROKEN
    assert any(sig[0] == "prefill_nax" for sig in hc_fused._VALIDATED)


@needs_nax
def test_model_prefill_logits_are_bit_identical(monkeypatch):
    """A small quantized model's prefill logits with the NAX kernels in every
    hyper-connection (and deferred writes) match the MLX path bit for bit."""
    from mlx_vlm.models.qwen4_exp import hc_fused
    from tests.test_qwen4_hc_deferred_write import _logits, _model

    model, config = _model()
    inputs = mx.random.randint(0, config.text_config.vocab_size, (1, 520))
    with monkeypatch.context() as patch:
        patch.setattr(hc_fused, "_NAX_DISABLED", True)
        reference = _logits(model, inputs)
    spies = _spies(monkeypatch)
    observed = _logits(model, inputs)
    # 3 layers x 2 hyper-connections; the final mixer has no inject weights.
    assert spies["up_tail"].call_count == 6
    view = mx.uint32 if observed.dtype == mx.float32 else mx.uint16
    assert mx.array_equal(observed.view(view), reference.view(view)).item()
