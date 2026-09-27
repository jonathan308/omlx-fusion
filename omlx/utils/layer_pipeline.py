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
in flight; an optional ``on_evaluated`` hook (e.g. ``mx.clear_cache``) runs
after each completed layer, like the per-layer release of a blocking loop.
"""

from __future__ import annotations

from collections import deque

import mlx.core as mx


class LayerPipeline:
    """Queue per-layer results asynchronously with a bounded in-flight depth.

    ``on_evaluated`` runs after each layer is waited for (for example
    ``mx.clear_cache`` to keep the per-layer allocator release of a blocking
    eval + clear loop; buffers still used by in-flight layers are not in the
    cache, so clearing is safe).
    """

    __slots__ = ("_depth", "_inflight", "_on_evaluated")

    def __init__(self, depth: int = 1, on_evaluated=None):
        self._depth = max(0, int(depth))
        self._inflight: deque = deque()
        self._on_evaluated = on_evaluated

    def _wait_oldest(self) -> None:
        mx.eval(*self._inflight.popleft())
        if self._on_evaluated is not None:
            self._on_evaluated()

    def push(self, *arrays) -> None:
        """Queue ``arrays`` for evaluation; block on the oldest beyond depth."""
        mx.async_eval(*arrays)
        self._inflight.append(arrays)
        while len(self._inflight) > self._depth:
            self._wait_oldest()

    def drain(self) -> None:
        """Wait for every queued layer."""
        while self._inflight:
            self._wait_oldest()
