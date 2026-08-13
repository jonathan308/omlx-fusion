# SPDX-License-Identifier: Apache-2.0
"""The known-answer gate compares GPU and CPU streams and fails closed.

Tests drive it with a numpy-backed fake ``mlx.core``: exact ops pass, and a
"corrupt kernel" mode that makes GPU-stream results garbage must fail every
check — the token-salad-at-normal-t/s failure this gate exists to catch.
"""

import numpy as np

from omlx.custom_kernels import known_answer


class _Array:
    def __init__(self, data):
        self._data = np.asarray(data)

    def astype(self, dtype):
        return _Array(self._data.astype(dtype))

    def item(self):
        return self._data.item()

    def __sub__(self, other):
        return _Array(self._data - _unwrap(other))

    def __add__(self, other):
        return _Array(self._data + _unwrap(other))

    def __mul__(self, other):
        return _Array(self._data * _unwrap(other))

    def __eq__(self, other):
        return _Array(self._data == _unwrap(other))


def _unwrap(value):
    return value._data if isinstance(value, _Array) else value


class _FakeRandom:
    def __init__(self):
        self._state = np.random.RandomState(0)

    def seed(self, value):
        self._state = np.random.RandomState(value)

    def normal(self, shape):
        return _Array(self._state.standard_normal(shape))


class _FakeMx:
    """Just enough mlx.core for the gate; ``stream != cpu`` is the GPU path."""

    float32 = np.float32
    float16 = np.float16
    gpu = "gpu"
    cpu = "cpu"
    __version__ = "0.0.0-fake"

    def __init__(self, *, corrupt=False):
        self.random = _FakeRandom()
        self._corrupt = corrupt

    def _skew(self, value, stream):
        # A corrupt Metal kernel returns O(1) relative garbage, not noise.
        if self._corrupt and stream != self.cpu:
            return _Array(value._data * 1000.0)
        return value

    def matmul(self, a, b, stream=None):
        return self._skew(_Array(a._data @ b._data), stream)

    def quantize(self, w, bits=4):
        rows = w._data.shape[0]
        return w, _Array(np.ones(rows)), _Array(np.zeros(rows))

    def quantized_matmul(self, x, wq, scales, biases, bits=4, stream=None):
        return self._skew(_Array(x._data @ wq._data.T), stream)

    def softmax(self, x, axis=-1, stream=None):
        shifted = x._data - x._data.max(axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return _Array(exp / exp.sum(axis=axis, keepdims=True))

    def argmax(self, x, axis=-1, stream=None):
        indices = np.argmax(x._data, axis=axis)
        if self._corrupt and stream != self.cpu:
            indices = (indices + 1) % x._data.shape[axis]
        return _Array(indices)

    def mean(self, x, axis=-1, keepdims=False, stream=None):
        result = np.mean(x._data, axis=axis, keepdims=keepdims)
        return self._skew(_Array(result), stream)

    def var(self, x, axis=-1, keepdims=False, stream=None):
        return _Array(np.var(x._data, axis=axis, keepdims=keepdims))

    def rsqrt(self, x, stream=None):
        return _Array(1.0 / np.sqrt(x._data))

    def abs(self, x):
        return _Array(np.abs(x._data))

    def max(self, x):
        return _Array(np.max(x._data))

    def sum(self, x):
        return _Array(np.sum(x._data))

    def eval(self, *args):
        return None


def test_a_healthy_build_passes_every_check():
    result = known_answer.run_checks(_FakeMx())

    assert result["ok"] is True
    assert result["failures"] == []
    assert result["mlx_version"] == "0.0.0-fake"
    assert len(result["report"]) == 4
    assert any("qmm_4bit" in line for line in result["report"])
    assert any("softmax/argmax: 4/4 rows agree" in line for line in result["report"])


def test_a_corrupt_build_fails_every_check():
    result = known_answer.run_checks(_FakeMx(corrupt=True))

    assert result["ok"] is False
    assert sorted(result["failures"]) == [
        "matmul_fp32 gpu/cpu divergence",
        "norm_chain fp16 gpu/cpu divergence",
        "quantized_matmul_4bit gpu/cpu divergence",
        "softmax/argmax gpu/cpu divergence",
    ]


def test_main_exit_codes_follow_the_verdict(monkeypatch, capsys):
    monkeypatch.setattr(
        known_answer,
        "run_checks",
        lambda: {
            "ok": True,
            "mlx_version": "0.0.0-fake",
            "report": ["    matmul_fp32: ok"],
            "failures": [],
        },
    )
    assert known_answer.main() == 0
    assert "PASS" in capsys.readouterr().out

    monkeypatch.setattr(
        known_answer,
        "run_checks",
        lambda: {
            "ok": False,
            "mlx_version": "0.0.0-fake",
            "report": [],
            "failures": ["quantized_matmul_4bit gpu/cpu divergence"],
        },
    )
    assert known_answer.main() == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "quantized_matmul_4bit" in out
