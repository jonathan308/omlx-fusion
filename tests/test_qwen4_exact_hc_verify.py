# SPDX-License-Identifier: Apache-2.0
"""Host-only gates for Qwen4 scalar-row HC target verification."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import hc_projection, language  # noqa: E402


@dataclass(frozen=True)
class _Tensor:
    shape: tuple[int, ...]
    dtype: object
    rows: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, index):
        assert isinstance(index, tuple) and len(index) == 2
        batch, row_slice = index
        assert batch == slice(None)
        assert isinstance(row_slice, slice)
        selected = self.rows[row_slice]
        return _Tensor((1, len(selected), self.shape[-1]), self.dtype, selected)


def _install_fake_mx(monkeypatch, dtype):
    def concatenate(values, axis):
        assert axis == 1
        values = list(values)
        assert values
        return _Tensor(
            (1, sum(value.shape[1] for value in values), values[0].shape[-1]),
            values[0].dtype,
            tuple(row for value in values for row in value.rows),
        )

    monkeypatch.setattr(
        language,
        "mx",
        SimpleNamespace(bfloat16=dtype, concatenate=concatenate),
    )


def _module(enabled=True):
    return SimpleNamespace(
        _omlx_exact_hybrid_projection=enabled,
        input_mix_weight_down=object(),
        block_inject_weight=object(),
    )


@pytest.mark.parametrize("width", [2, 6, 9])
def test_exact_hc_verify_replays_scalar_rows_in_original_order(
    monkeypatch,
    width,
):
    dtype = object()
    _install_fake_mx(monkeypatch, dtype)
    monkeypatch.setenv("OMLX_QWEN4_EXACT_HC_VERIFY", "1")
    monkeypatch.setattr(language, "_EXACT_HC_VERIFY_LOGGED", False)
    calls = []

    def hybrid(value, down, injection):
        assert down is module.input_mix_weight_down
        assert injection is module.block_inject_weight
        assert value.shape == (1, 1, 10240)
        calls.append(value.rows[0])
        return _Tensor((1, 1, 324), dtype, value.rows)

    module = _module()
    monkeypatch.setattr(hc_projection, "hybrid_projection", hybrid)
    normed = _Tensor((1, width, 10240), dtype, tuple(range(width)))

    output = language._exact_hc_verify_projection(module, normed, True)

    assert output == _Tensor((1, width, 324), dtype, tuple(range(width)))
    assert calls == list(range(width))
    assert language._EXACT_HC_VERIFY_LOGGED is True


@pytest.mark.parametrize(
    ("env", "target_verify", "width", "dtype_matches", "module_enabled"),
    [
        (None, True, 6, True, True),
        ("0", True, 6, True, True),
        ("1", False, 6, True, True),
        ("1", True, 1, True, True),
        ("1", True, 10, True, True),
        ("1", True, 6, False, True),
        ("1", True, 6, True, False),
    ],
)
def test_exact_hc_verify_gate_fails_closed_without_calling_hybrid(
    monkeypatch,
    env,
    target_verify,
    width,
    dtype_matches,
    module_enabled,
):
    dtype = object()
    _install_fake_mx(monkeypatch, dtype)
    if env is None:
        monkeypatch.delenv("OMLX_QWEN4_EXACT_HC_VERIFY", raising=False)
    else:
        monkeypatch.setenv("OMLX_QWEN4_EXACT_HC_VERIFY", env)
    monkeypatch.setattr(
        hc_projection,
        "hybrid_projection",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("ineligible exact-HC route called the hybrid kernel")
        ),
    )
    normed = _Tensor(
        (1, width, 10240),
        dtype if dtype_matches else object(),
        tuple(range(width)),
    )

    assert (
        language._exact_hc_verify_projection(
            _module(enabled=module_enabled),
            normed,
            target_verify,
        )
        is None
    )


def test_exact_hc_verify_partial_capability_failure_preserves_fallback(
    monkeypatch,
):
    dtype = object()
    _install_fake_mx(monkeypatch, dtype)
    monkeypatch.setenv("OMLX_QWEN4_EXACT_HC_VERIFY", "1")
    monkeypatch.setattr(language, "_EXACT_HC_VERIFY_LOGGED", False)
    calls = []

    def hybrid(value, _down, _injection):
        row = value.rows[0]
        calls.append(row)
        if row == 2:
            return None
        return _Tensor((1, 1, 324), dtype, value.rows)

    monkeypatch.setattr(hc_projection, "hybrid_projection", hybrid)
    normed = _Tensor((1, 6, 10240), dtype, tuple(range(6)))

    assert language._exact_hc_verify_projection(_module(), normed, True) is None
    assert calls == [0, 1, 2]
    assert language._EXACT_HC_VERIFY_LOGGED is False


def test_forward_uses_exact_rows_before_existing_fused_verify_fallback():
    source = inspect.getsource(language.Qwen4ExpGatedResidual._forward)
    exact = source.index("_exact_hc_verify_projection")
    fallback = source.index("verified_fused is None")
    fused_call = source.index("verified_fused = fused_projection(normed)")
    assert exact < fallback < fused_call


def test_forward_bypasses_fused_bank_when_exact_rows_succeed(monkeypatch):
    width = 6
    hyper_input = np.zeros((1, width, 10240), dtype=np.float32)
    exact = np.zeros((1, width, 324), dtype=np.float32)
    exact[..., 320:] = 1.0
    calls = []

    monkeypatch.setattr(
        language,
        "_exact_hc_verify_projection",
        lambda module, normed, target_verify: (
            calls.append((module, normed.shape, target_verify)) or exact
        ),
    )
    monkeypatch.setattr(
        language,
        "mx",
        SimpleNamespace(
            sigmoid=lambda value: 1.0 / (1.0 + np.exp(-value)),
            mean=np.mean,
        ),
    )
    monkeypatch.setattr(language, "nn", SimpleNamespace(silu=lambda value: value))

    def forbidden_fused(_value):
        raise AssertionError("exact scalar rows must bypass the fused verify bank")

    module = SimpleNamespace(
        hc_norm=lambda value: value,
        hc_lowrank=320,
        hc_count=4,
        hidden_size=2560,
        input_mix_weight_up=lambda value: np.zeros(
            (*value.shape[:-1], 10240), dtype=value.dtype
        ),
        _omlx_exact_verify_fused_projection=forbidden_fused,
    )

    result = language.Qwen4ExpGatedResidual._forward(
        module,
        hyper_input,
        target_verify=True,
    )

    assert calls == [(module, (1, width, 10240), True)]
    assert isinstance(result, tuple) and len(result) == 3
    assert result[0].shape == (1, width, 2560)
