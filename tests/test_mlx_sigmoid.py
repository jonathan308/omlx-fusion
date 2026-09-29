# SPDX-License-Identifier: Apache-2.0
"""The compiled-sigmoid exp that fused kernels mirror follows the MLX build."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.utils import mlx_sigmoid
from omlx.utils.mlx_sigmoid import compiled_sigmoid, precise_compiled_sigmoid


@pytest.mark.parametrize(
    "version,release",
    [
        ("0.32.2", (0, 32, 2)),
        ("0.32.3", (0, 32, 3)),
        ("0.32.3.dev20260929+b067c185", (0, 32, 3)),
        ("0.32.2.dev20260926+f693ea9b", (0, 32, 2)),
        ("0.33.0rc1", (0, 33, 0)),
    ],
)
def test_mlx_release_parses_dev_and_local_versions(version, release):
    assert mlx_sigmoid._mlx_release(version) == release


def test_precise_from_mlx_0323(monkeypatch):
    monkeypatch.delenv("OMLX_MLX_PRECISE_SIGMOID", raising=False)
    assert not precise_compiled_sigmoid("0.32.2")
    assert not precise_compiled_sigmoid("0.32.2.dev20260926+f693ea9b")
    assert precise_compiled_sigmoid("0.32.3")
    assert precise_compiled_sigmoid("0.32.3.dev20260929+b067c185")
    assert precise_compiled_sigmoid("0.33.0")


def test_env_overrides_the_version(monkeypatch):
    monkeypatch.setenv("OMLX_MLX_PRECISE_SIGMOID", "0")
    assert not precise_compiled_sigmoid("0.32.3")
    monkeypatch.setenv("OMLX_MLX_PRECISE_SIGMOID", "1")
    assert precise_compiled_sigmoid("0.32.2")


def test_rewrite_touches_only_the_fast_sigmoid_call(monkeypatch):
    src = (
        "a = 1 / (1 + metal::exp(metal::abs(x)));\n"
        "b = 1 / (1 + metal::precise::exp(metal::abs(x)));\n"
        "c = metal::exp(x);\n"
    )
    monkeypatch.setenv("OMLX_MLX_PRECISE_SIGMOID", "0")
    assert compiled_sigmoid(src) == src
    monkeypatch.setenv("OMLX_MLX_PRECISE_SIGMOID", "1")
    assert compiled_sigmoid(src) == (
        "a = 1 / (1 + metal::precise::exp(metal::abs(x)));\n"
        "b = 1 / (1 + metal::precise::exp(metal::abs(x)));\n"
        "c = metal::exp(x);\n"
    )


_SILU_SOURCE = """
    uint i = thread_position_in_grid.x;
    T x = inp[i];
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    T s = (x < 0) ? y : 1 - y;
    out[i] = x * s;
"""


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_rewritten_sigmoid_matches_compiled_silu_bitwise(monkeypatch, dtype):
    monkeypatch.delenv("OMLX_MLX_PRECISE_SIGMOID", raising=False)
    x = mx.concatenate(
        [
            mx.linspace(-24.0, 24.0, 1 << 15),
            mx.random.normal((1 << 15,), key=mx.random.key(11)) * 4.0,
        ]
    ).astype(dtype)
    kernel = mx.fast.metal_kernel(
        name="omlx_test_compiled_silu",
        input_names=["inp"],
        output_names=["out"],
        source=compiled_sigmoid(_SILU_SOURCE),
    )
    (out,) = kernel(
        inputs=[x],
        template=[("T", dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[dtype],
    )
    ref = mx.compile(nn.silu)(x)
    view = {2: mx.uint16, 4: mx.uint32}[x.dtype.size]
    assert mx.array_equal(out.view(view), ref.view(view)).item()
