# SPDX-License-Identifier: Apache-2.0
"""LayerPipeline keeps a bounded number of layers in flight and preserves results."""

import mlx.core as mx

from omlx.utils.layer_pipeline import LayerPipeline


def _layers(n, width=256):
    mx.random.seed(0)
    return [mx.random.normal((width, width)) * 0.05 for _ in range(n)]


def test_pipeline_matches_eager_evaluation():
    ws = _layers(6)
    x0 = mx.random.normal((4, 256))
    eager = x0
    for w in ws:
        eager = mx.tanh(eager @ w)
        mx.eval(eager)
    pipe = LayerPipeline(depth=1)
    h = x0
    for w in ws:
        h = mx.tanh(h @ w)
        pipe.push(h)
    pipe.drain()
    assert mx.array_equal(h, eager).item()


def test_pipeline_bounds_in_flight_work(monkeypatch):
    evaluated = []
    waited = []
    monkeypatch.setattr(mx, "async_eval", lambda *a: evaluated.append(a))
    monkeypatch.setattr(mx, "eval", lambda *a: waited.append(a))
    pipe = LayerPipeline(depth=1)
    arrays = [object() for _ in range(4)]
    for a in arrays:
        pipe.push(a)
    # each push queues its layer; the layer before the previous one is waited on
    assert [e[0] for e in evaluated] == arrays
    assert [w[0] for w in waited] == arrays[:3]
    pipe.drain()
    assert [w[0] for w in waited] == arrays


def test_pipeline_depth_zero_is_synchronous(monkeypatch):
    waited = []
    monkeypatch.setattr(mx, "async_eval", lambda *a: None)
    monkeypatch.setattr(mx, "eval", lambda *a: waited.append(a))
    pipe = LayerPipeline(depth=0)
    pipe.push("a")
    pipe.push("b")
    assert [w[0] for w in waited] == ["a", "b"]
