# SPDX-License-Identifier: Apache-2.0
"""Bounded asynchronous evaluation for layer-by-layer prefill.

Several model loops evaluate the hidden state after every decoder layer during
prefill to bound the size of the lazy graph (and with it peak memory). A
blocking ``mx.eval`` per layer serializes the host and the GPU: the GPU idles
while Python builds and encodes the next layer, which on large GPUs (M5 Ultra)
is a large share of prefill wall time. Pairing it with ``mx.clear_cache()``
additionally returns every cached buffer to the OS, so the next layer's
allocations are fresh pages that the GPU has to fault in.

``LayerPipeline`` keeps the same memory bound with overlap: each layer is
queued with ``mx.async_eval`` and the host only waits for the layer queued
``depth`` steps earlier. With the default ``depth=1`` at most two layers are
in flight, and the buffer cache is left intact so consecutive layers reuse
each other's (same-shaped) buffers.
"""

from __future__ import annotations

from collections import deque

import mlx.core as mx


class LayerPipeline:
    """Queue per-layer results asynchronously with a bounded in-flight depth."""

    __slots__ = ("_depth", "_inflight")

    def __init__(self, depth: int = 1):
        self._depth = max(0, int(depth))
        self._inflight: deque = deque()

    def push(self, *arrays) -> None:
        """Queue ``arrays`` for evaluation; block on the oldest beyond depth."""
        mx.async_eval(*arrays)
        self._inflight.append(arrays)
        while len(self._inflight) > self._depth:
            mx.eval(*self._inflight.popleft())

    def drain(self) -> None:
        """Wait for every queued layer."""
        while self._inflight:
            mx.eval(*self._inflight.popleft())
