# SPDX-License-Identifier: Apache-2.0
"""Host-only gates for Qwen4 tokenwise MoE target verification."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import language  # noqa: E402


def _install_host_mx(monkeypatch):
    monkeypatch.setattr(
        language,
        "mx",
        SimpleNamespace(concatenate=lambda values, axis: np.concatenate(values, axis)),
    )


@pytest.mark.parametrize("width", [2, 6, 9])
def test_tokenwise_moe_replays_scalar_rows_in_original_order(monkeypatch, width):
    _install_host_mx(monkeypatch)
    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_MOE_VERIFY", "1")
    monkeypatch.setattr(language, "_TOKENWISE_MOE_VERIFY_LOGGED", False)
    mixed = np.arange(width * 4, dtype=np.float32).reshape(1, width, 4)
    calls = []

    def moe(value, *, target_verify):
        calls.append((value.copy(), target_verify))
        return value + 100.0

    output = language._tokenwise_moe_verify(moe, mixed, True)

    assert np.array_equal(output, mixed + 100.0)
    assert [call[0].shape for call in calls] == [(1, 1, 4)] * width
    assert [int(call[0][0, 0, 0] // 4) for call in calls] == list(range(width))
    assert all(call[1] is False for call in calls)
    assert language._TOKENWISE_MOE_VERIFY_LOGGED is True


@pytest.mark.parametrize(
    ("env", "target_verify", "shape"),
    [
        (None, True, (1, 6, 4)),
        ("0", True, (1, 6, 4)),
        ("1", False, (1, 6, 4)),
        ("1", True, (1, 1, 4)),
        ("1", True, (1, 10, 4)),
        ("1", True, (2, 6, 4)),
        ("1", True, (6, 4)),
    ],
)
def test_tokenwise_moe_gate_is_explicit_and_fail_closed(
    monkeypatch,
    env,
    target_verify,
    shape,
):
    _install_host_mx(monkeypatch)
    if env is None:
        monkeypatch.delenv("OMLX_QWEN4_TOKENWISE_MOE_VERIFY", raising=False)
    else:
        monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_MOE_VERIFY", env)
    mixed = np.zeros(shape, dtype=np.float32)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ineligible tokenwise-MoE route called the module")

    assert language._tokenwise_moe_verify(forbidden, mixed, target_verify) is None


def test_tokenwise_moe_row_failure_returns_to_wide_verify_path(monkeypatch):
    _install_host_mx(monkeypatch)
    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_MOE_VERIFY", "1")
    monkeypatch.setattr(language, "_TOKENWISE_MOE_VERIFY_LOGGED", False)
    mixed = np.arange(24, dtype=np.float32).reshape(1, 6, 4)
    calls = []

    def moe(value, *, target_verify):
        row = int(value[0, 0, 0] // 4) if value.shape[1] == 1 else None
        calls.append((value.shape, row, target_verify))
        if row == 2:
            raise RuntimeError("controlled scalar-row failure")
        return value + (200.0 if target_verify else 100.0)

    output = language._qwen4_moe_forward(moe, mixed, True)

    assert np.array_equal(output, mixed + 200.0)
    assert calls == [
        ((1, 1, 4), 0, False),
        ((1, 1, 4), 1, False),
        ((1, 1, 4), 2, False),
        ((1, 6, 4), None, True),
    ]
    assert language._TOKENWISE_MOE_VERIFY_LOGGED is False


def test_default_moe_forward_calls_existing_wide_path_once(monkeypatch):
    _install_host_mx(monkeypatch)
    monkeypatch.delenv("OMLX_QWEN4_TOKENWISE_MOE_VERIFY", raising=False)
    mixed = np.zeros((1, 6, 4), dtype=np.float32)
    calls = []

    def moe(value, *, target_verify):
        calls.append((value.shape, target_verify))
        return value

    assert language._qwen4_moe_forward(moe, mixed, True) is mixed
    assert calls == [((1, 6, 4), True)]


def test_normal_and_profiled_layers_share_the_tokenwise_moe_router():
    normal = inspect.getsource(language.Qwen4ExpDecoderLayer.__call__)
    profiled = inspect.getsource(language.Qwen4ExpDecoderLayer._profiled_call)
    assert "_qwen4_moe_forward(self.mlp, mixed, target_verify)" in normal
    assert "_qwen4_moe_forward(self.mlp, mixed, target_verify)" in profiled
