# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3's compiled decode FFN must not pin a layer's weights after unload.

MLX 0.32.2's ``mx.compile`` leaks each multi-output primitive that is an
intermediate of a traced graph (ml-explore/mlx#4453), together with the
arrays it references. The fused decode kernels are multi-output primitives
fed by the layer's weights, so a plain ``mx.compile(layer._ffn_block)`` kept
those weights alive after the model was dropped. ``compile_ffn_block`` traces
the weights as inputs instead.
"""

from __future__ import annotations

import gc
import threading

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_vlm_glm5_next_compat as compat

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="custom Metal kernels need a Metal GPU"
)

_WORDS = 1 << 21  # 8 MB of float32 per weight: a pinned weight is unmistakable


@pytest.fixture(autouse=True)
def _apply_glm5_next_compat():
    compat.apply_mlx_vlm_glm5_next_compat_patch()


def _language():
    from mlx_vlm.models.glm5_next import language

    return language


_KERNELS = {}


def _kernel(n_in: int, n_out: int):
    """out_j[i] = sum_k in_k[i] + j (a stand-in for the fused decode kernels)."""
    key = (n_in, n_out)
    if key not in _KERNELS:
        total = " + ".join(f"in{k}[i]" for k in range(n_in))
        _KERNELS[key] = mx.fast.metal_kernel(
            name=f"compile_ffn_probe_{n_in}_{n_out}",
            input_names=[f"in{k}" for k in range(n_in)],
            output_names=[f"out{j}" for j in range(n_out)],
            source="uint i = thread_position_in_grid.x;\n"
            + "".join(f"out{j}[i] = {total} + {j};\n" for j in range(n_out)),
        )
    return _KERNELS[key]


def _run(inputs, n_out):
    return _kernel(len(inputs), n_out)(
        inputs=inputs,
        grid=(8, 1, 1),
        threadgroup=(8, 1, 1),
        output_shapes=[(8,)] * n_out,
        output_dtypes=[mx.float32] * n_out,
    )


class _Weights(nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.weight = mx.arange(_WORDS, dtype=mx.float32) * 1e-6 + offset


class _Layer(nn.Module):
    """The FFN half of a decoder layer, with the leaking topology: multi-output
    kernels fed by the weights whose outputs feed further kernels."""

    def __init__(self):
        super().__init__()
        self.ffn_hc = _Weights(1.0)
        self.post_attention_layernorm = _Weights(2.0)
        self.mlp = _Weights(3.0)
        self.mlp.experts = [_Weights(4.0)]

    def _ffn_block(self, x):
        a, b = _run([x, self.ffn_hc.weight], 2)  # like the router logits kernel
        act, routes, scores = _run(
            [a, b, self.mlp.weight, self.mlp.experts[0].weight], 3
        )  # like the route-selecting gate/up kernel
        y = _run([act, routes, scores, self.post_attention_layernorm.weight], 1)[0]
        return y * 2


def _leaked_bytes(compile_fn) -> int:
    """Build, run and drop a layer on a worker thread (like an engine thread,
    whose final reclaim clears its compile cache); bytes left behind."""
    result = {}

    def work():
        gc.collect()
        mx.clear_cache()
        base = mx.get_active_memory()
        layer = _Layer()
        mx.eval(layer.parameters())
        x = mx.ones((8,), dtype=mx.float32)
        mx.eval(compile_fn(layer)(x))
        del layer, x
        gc.collect()
        mx.synchronize()
        mx.clear_streams()
        gc.collect()
        mx.clear_cache()
        result["leak"] = mx.get_active_memory() - base

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    return result["leak"]


def test_compile_ffn_block_releases_the_layer_weights():
    language = _language()
    leak = _leaked_bytes(lambda layer: language.compile_ffn_block(layer, layer._ffn_block))
    # A pinned weight would leave 8 MB+; only tiny trace constants may remain.
    assert leak < (64 << 10), f"{leak} bytes still active after the layer was dropped"


def test_compile_ffn_block_matches_eager_and_plain_compile():
    language = _language()
    layer = _Layer()
    x = mx.arange(8, dtype=mx.float32)
    eager = layer._ffn_block(x)
    plain = mx.compile(layer._ffn_block)(x)
    fixed = language.compile_ffn_block(layer, layer._ffn_block)
    first, second = fixed(x), fixed(x + 1)
    mx.eval(eager, plain, first, second)
    assert mx.array_equal(first, eager).item()
    assert mx.array_equal(first, plain).item()
    assert mx.array_equal(second, layer._ffn_block(x + 1)).item()


def test_compile_ffn_block_keeps_module_arrays_when_the_trace_raises():
    language = _language()
    layer = _Layer()
    before = [layer.ffn_hc.weight, layer.mlp.weight, layer.mlp.experts[0].weight]

    def broken(x):
        raise RuntimeError("trace failed")

    with pytest.raises(RuntimeError, match="trace failed"):
        language.compile_ffn_block(layer, broken)(mx.ones((8,)))
    after = [layer.ffn_hc.weight, layer.mlp.weight, layer.mlp.experts[0].weight]
    assert all(a is b for a, b in zip(before, after))
    mx.eval(layer._ffn_block(mx.ones((8,))))
