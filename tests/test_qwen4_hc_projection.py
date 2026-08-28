from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


def _production_module(bits: int):
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedResidual,
        Qwen4ExpRMSNorm,
    )

    module = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
    nn.Module.__init__(module)
    module.hc_count = 4
    module.hidden_size = 2560
    module.hc_lowrank = 320
    module.hc_norm = Qwen4ExpRMSNorm(
        10240,
        group_size=2560,
        eps=1e-6,
    )
    module.hc_norm.weight = (
        mx.random.normal((10240,)) * 0.02
    ).astype(mx.bfloat16)
    module.input_mix_weight_down = nn.QuantizedLinear(
        10240,
        320,
        bias=False,
        group_size=64,
        bits=bits,
        mode="affine",
    )
    module.block_inject_weight = nn.QuantizedLinear(
        10240,
        4,
        bias=False,
        group_size=64,
        bits=bits,
        mode="affine",
    )
    module.input_mix_weight_up = nn.QuantizedLinear(
        320,
        10240,
        bias=False,
        group_size=64,
        bits=bits,
        mode="affine",
    )
    for projection in (
        module.input_mix_weight_down,
        module.block_inject_weight,
        module.input_mix_weight_up,
    ):
        projection.scales = projection.scales.astype(mx.bfloat16)
        projection.biases = projection.biases.astype(mx.bfloat16)
    return module


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_exact_hybrid_raw_and_full_outputs_are_bit_exact(bits):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.hc_projection import hybrid_projection
    from mlx_vlm.models.qwen4_exp.language import (
        compile_hyper_connections,
        fuse_hyper_connection_projections,
        hyper_connection_fused_copy_nbytes,
    )

    mx.random.seed(20260900 + bits)
    module = _production_module(bits)
    down = module.input_mix_weight_down
    injection = module.block_inject_weight
    down_id = id(down)
    injection_id = id(injection)

    for seed in range(4):
        mx.random.seed(20261000 + 100 * bits + seed)
        stream = mx.random.normal((1, 1, 10240)).astype(mx.bfloat16)
        normed = module.hc_norm(stream)
        expected_down = down(normed)
        expected_injection = injection(normed)
        combined = hybrid_projection(normed, down, injection)
        assert combined is not None
        mx.eval(expected_down, expected_injection, combined)
        assert mx.array_equal(expected_down, combined[..., :320]).item()
        assert mx.array_equal(
            expected_injection,
            combined[..., 320:324],
        ).item()

    stream = mx.random.normal((1, 1, 10240)).astype(mx.bfloat16)
    canonical = module._forward(stream)
    mx.eval(*canonical)
    assert fuse_hyper_connection_projections(module) == 1
    assert fuse_hyper_connection_projections(module) == 0
    assert id(module.input_mix_weight_down) == down_id
    assert id(module.block_inject_weight) == injection_id
    assert not hasattr(module, "input_inject_weight")
    assert hasattr(module, "_omlx_exact_verify_fused_projection")
    fused = module._omlx_exact_verify_fused_projection
    assert fused.weight.shape[0] == 324
    assert hyper_connection_fused_copy_nbytes(module) == sum(
        value.nbytes for value in (fused.weight, fused.scales, fused.biases)
    )
    assert compile_hyper_connections(module) == 1
    actual = module(stream)
    mx.eval(*actual)
    for expected, value in zip(canonical, actual):
        assert mx.array_equal(expected, value).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_exact_hybrid_fallbacks_never_enter_native(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import hc_projection
    from mlx_vlm.models.qwen4_exp import language

    mx.random.seed(20261100)
    module = _production_module(5)
    assert language.fuse_hyper_connection_projections(module) == 1
    assert language.compile_hyper_connections(module) == 1
    native = hc_projection.hybrid_projection
    spy = MagicMock(wraps=native)
    monkeypatch.setattr(hc_projection, "hybrid_projection", spy)

    cases = [
        (mx.random.normal((2, 1, 10240)).astype(mx.bfloat16), False),
        (mx.random.normal((1, 2, 10240)).astype(mx.bfloat16), False),
        (mx.random.normal((1, 1, 10240)).astype(mx.float16), False),
        (mx.random.normal((1, 1, 10240)).astype(mx.bfloat16), True),
    ]
    for stream, target_verify in cases:
        output = module(stream, target_verify=target_verify)
        mx.eval(*output)

    monkeypatch.setattr(
        language,
        "_MTP_RUNTIME",
        language.Qwen4ExpMTPRuntime(enabled=True),
    )
    mtp_output = module(mx.zeros((1, 1, 10240), dtype=mx.bfloat16))
    mx.eval(*mtp_output)
    assert spy.call_count == 1


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_exact_verify_fused_width_allowlist(bits):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    mx.random.seed(20261300 + bits)
    module = _production_module(bits)
    streams = {
        width: mx.random.normal((1, width, 10240)).astype(mx.bfloat16)
        for width in (*range(2, 7), 7, 64)
    }
    canonical = {
        width: module._forward(stream, target_verify=True)
        for width, stream in streams.items()
    }
    mx.eval(*[value for output in canonical.values() for value in output])
    assert language.fuse_hyper_connection_projections(module) == 1

    for width in range(2, 7):
        actual = module._forward(streams[width], target_verify=True)
        mx.eval(*actual)
        assert all(
            mx.array_equal(expected, value).item()
            for expected, value in zip(canonical[width], actual)
        )

    fused = module._omlx_exact_verify_fused_projection

    class Bomb:
        def __call__(self, _):
            raise AssertionError("unqualified width entered fused HC copy")

    module._omlx_exact_verify_fused_projection = Bomb()
    for width in (7, 64):
        actual = module._forward(streams[width], target_verify=True)
        mx.eval(*actual)
        assert all(
            mx.array_equal(expected, value).item()
            for expected, value in zip(canonical[width], actual)
        )
    module._omlx_exact_verify_fused_projection = fused


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_mtp_trunk_and_head_hc_parity(bits, monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    modules = []
    expected = []
    for index, target_verify in enumerate((False, True)):
        mx.random.seed(20261400 + 10 * bits + index)
        module = _production_module(bits)
        rows = {
            width: mx.random.normal((1, width, 10240)).astype(mx.bfloat16)
            for width in range(1, 7)
        }
        outputs = {
            width: module._forward(value, target_verify=target_verify)
            for width, value in rows.items()
        }
        mx.eval(*[value for output in outputs.values() for value in output])
        modules.append((module, rows, target_verify))
        expected.append(outputs)

    monkeypatch.setattr(
        language,
        "_MTP_RUNTIME",
        language.Qwen4ExpMTPRuntime(enabled=True),
    )
    for module, _, _ in modules:
        assert language.fuse_hyper_connection_projections(module) == 1

    for (module, rows, target_verify), expected_outputs in zip(modules, expected):
        for width, value in rows.items():
            actual = module._forward(value, target_verify=target_verify)
            mx.eval(*actual)
            assert all(
                mx.array_equal(expected_value, actual_value).item()
                for expected_value, actual_value in zip(
                    expected_outputs[width],
                    actual,
                )
            )


def test_qwen4_exact_hybrid_preparation_fails_closed_for_other_geometry():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.hc_projection import compatible_projections
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedResidual,
        fuse_hyper_connection_projections,
    )

    small = Qwen4ExpGatedResidual(
        SimpleNamespace(
            hc_count=2,
            hidden_size=32,
            hc_lowrank=32,
            rms_norm_eps=1e-6,
        )
    )
    assert fuse_hyper_connection_projections(small) == 0
    assert hasattr(small, "input_mix_weight_down")
    assert hasattr(small, "block_inject_weight")

    mx.random.seed(20261200)
    unsupported = _production_module(3)
    assert not compatible_projections(
        unsupported.input_mix_weight_down,
        unsupported.block_inject_weight,
    )
    assert fuse_hyper_connection_projections(unsupported) == 0
    assert not hasattr(unsupported, "_omlx_exact_hybrid_projection")

    missing = _production_module(5)
    del missing.input_mix_weight_down.scales
    assert not compatible_projections(
        missing.input_mix_weight_down,
        missing.block_inject_weight,
    )
    assert fuse_hyper_connection_projections(missing) == 0

    none_metadata = _production_module(5)
    none_metadata.block_inject_weight.biases = None
    assert not compatible_projections(
        none_metadata.input_mix_weight_down,
        none_metadata.block_inject_weight,
    )
    assert fuse_hyper_connection_projections(none_metadata) == 0
