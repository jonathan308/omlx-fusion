# SPDX-License-Identifier: Apache-2.0
"""Head-trimmed exact-resident entries: bounded RAM without losing reuse.

The paged tier already serves every durable prompt boundary.  Retaining a
second full copy of those tokens in the exact-resident tier is the
unbounded growth term (full terminal per entry).  These tests prove the
tail-trim path: stage keeps only post-boundary tail KV plus fixed
recurrent state, restore composes the paged head with that tail, and every
unknown layout fails closed to today's full-retain behavior.
"""

from collections import defaultdict

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from omlx.cache.exact_resident import ExactResidentPrefixCache
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


def _scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._exact_resident_cache = ExactResidentPrefixCache(2)
    scheduler._phase_total_ms = defaultdict(float)
    scheduler._phase_count = defaultdict(int)
    scheduler.model = object()
    scheduler._vlm_mtp_drafter = None
    return scheduler


def _kv_layer(n_tokens, dim=8):
    layer = KVCache()
    layer.keys = mx.zeros((1, 2, n_tokens, dim))
    layer.values = mx.ones((1, 2, n_tokens, dim))
    layer.offset = n_tokens
    mx.eval(layer.keys, layer.values)
    return layer


def _gdn_layer():
    layer = ArraysCache(size=2)
    layer[0] = mx.zeros((1, 3, 16))
    layer[1] = mx.zeros((1, 4, 8, 8), mx.float32)
    mx.eval(layer[0], layer[1])
    return layer


def _request(tokens):
    request = Request("request", list(tokens), SamplingParams())
    request.prompt_token_ids = list(tokens)
    request.num_prompt_tokens = len(tokens)
    return request


def test_trim_keeps_tail_kv_and_fixed_state():
    total, durable = 5000, 4096
    cache = [_kv_layer(total), _gdn_layer(), _kv_layer(total)]
    trimmed = Scheduler._trim_exact_resident_head(cache, total, durable)
    assert trimmed is not None
    assert len(trimmed) == 3
    for leaf in trimmed[:1] + trimmed[2:]:
        assert leaf._omlx_kv_tail_span == (durable, total)
        assert tuple(leaf.keys.shape) == (1, 2, total - durable, 8)
        assert tuple(leaf.values.shape) == (1, 2, total - durable, 8)
        assert leaf.offset == total
    # Fixed GDN layer kept by reference (no copy, fixed size).
    assert trimmed[1] is cache[1]
    # Original live cache untouched.
    assert tuple(cache[0].keys.shape) == (1, 2, total, 8)
    assert not hasattr(cache[0], "_omlx_kv_tail_span")
    assert Scheduler._is_tail_trimmed_cache(trimmed)
    assert not Scheduler._is_tail_trimmed_cache(cache)
    assert _scheduler()._validate_trimmed_tail(trimmed, total, durable)


def test_trim_fail_closed_cases():
    cache = [_kv_layer(100), _gdn_layer()]
    assert Scheduler._trim_exact_resident_head(cache, 100, 0) is None
    assert Scheduler._trim_exact_resident_head(cache, 100, 100) is None
    assert Scheduler._trim_exact_resident_head(cache, 100, 200) is None
    assert Scheduler._trim_exact_resident_head([], 100, 64) is None
    # Offset mismatch (padded backing beyond the logical terminal is fine,
    # but a short offset is corrupt).
    short = _kv_layer(100)
    short.offset = 90
    assert Scheduler._trim_exact_resident_head([short], 100, 64) is None
    # Composite cache objects are not trimmed.
    composite = type("CacheList", (), {})()
    composite.caches = [_kv_layer(100)]
    assert Scheduler._trim_exact_resident_head([composite], 100, 64) is None


def test_trim_rejects_token_length_auxiliaries():
    # QSA-style index arrays at token length must abort the trim: dropping
    # the head while keeping full-length auxiliaries would corrupt reuse.
    layer = _kv_layer(100)
    layer.index_keys = mx.zeros((1, 100, 4))
    assert Scheduler._trim_exact_resident_head([layer], 100, 64) is None
    fixed = _gdn_layer()
    fixed.stray_positions = mx.zeros((100,), dtype=mx.int32)
    assert Scheduler._trim_exact_resident_head([fixed], 100, 64) is None


def test_validate_trimmed_tail():
    scheduler = _scheduler()
    total, durable = 5000, 4096
    cache = [_kv_layer(total), _gdn_layer()]
    trimmed = Scheduler._trim_exact_resident_head(cache, total, durable)
    assert trimmed is not None
    assert scheduler._validate_trimmed_tail(trimmed, total, durable)
    assert not scheduler._validate_trimmed_tail(trimmed, total, 2048)
    assert not scheduler._validate_trimmed_tail(cache, total, durable)
    assert not scheduler._validate_trimmed_tail([], total, durable)


def test_tail_copies_do_not_alias_live_buffers():
    total, durable = 1000, 512
    live = _kv_layer(total)
    trimmed = Scheduler._trim_exact_resident_head([live], total, durable)
    assert trimmed is not None
    # Mutating the retained tail must not touch the live parent buffer.
    trimmed[0].values = trimmed[0].values * 2 + 1
    mx.eval(trimmed[0].values, live.values)
    assert bool(mx.all(live.values[:, :, durable:total] == 1).item())
    assert bool(mx.all(trimmed[0].values == 3).item())


class _FakeTable:
    def __init__(self, num_tokens, n_blocks=2):
        self.num_tokens = num_tokens
        self.block_ids = list(range(n_blocks))


class _FakePaged:
    def __init__(self, head_layers):
        self.head_layers = head_layers
        self.fetched_prompts = []

    def fetch_cache(self, request_id, prompt_tokens, **kwargs):
        self.fetched_prompts.append(list(prompt_tokens))
        return _FakeTable(len(prompt_tokens), n_blocks=2), []

    def reconstruct_cache(self, block_table, promote_to_hot_cache=True):
        return self.head_layers


def _head_layer(n_tokens, dim=8):
    layer = KVCache()
    layer.keys = mx.zeros((1, 2, n_tokens, dim))
    layer.values = mx.zeros((1, 2, n_tokens, dim))
    layer.offset = n_tokens
    mx.eval(layer.keys, layer.values)
    return layer


def test_restore_composes_paged_head_with_exact_tail():
    scheduler = _scheduler()
    total, durable = 5000, 4096
    live = [_kv_layer(total), _gdn_layer()]
    trimmed = Scheduler._trim_exact_resident_head(live, total, durable)
    assert trimmed is not None
    prompt = list(range(total + 37))
    stored_tokens = list(range(total))
    assert scheduler._exact_resident_cache.put(
        stored_tokens,
        trimmed,
        cache_nbytes=123,
        durable_tokens=durable,
    )
    scheduler.block_aware_cache = _FakePaged(
        [_head_layer(durable), _gdn_layer()]
    )
    hit = scheduler._exact_resident_cache.acquire_prefix(prompt)
    assert hit is not None
    request = _request(prompt)
    scheduler.paged_cache_manager = object()
    assert scheduler._restore_trimmed_exact_hit(request, hit, prompt)
    assert request.cached_tokens == total
    assert request.remaining_tokens == prompt[total:]
    assert request.block_table is None
    assert request.shared_prefix_blocks == 0
    assert getattr(request, "_exact_resident_hit", False) is True
    # Head came from paged (zeros), tail from the retained ones.
    assert tuple(request.prompt_cache[0].keys.shape) == (1, 2, total, 8)
    assert bool(
        mx.all(request.prompt_cache[0].values[:, :, durable:] == 1).item()
    )
    assert bool(
        mx.all(request.prompt_cache[0].values[:, :, :durable] == 0).item()
    )
    # GDN fixed state is the retained live object (same arrays).
    assert request.prompt_cache[1] is live[1]
    # Paged fetch saw exactly the durable head prompt.
    assert scheduler.block_aware_cache.fetched_prompts == [prompt[:durable]]


def test_restore_fails_closed_on_head_mismatch():
    scheduler = _scheduler()
    total, durable = 5000, 4096
    live = [_kv_layer(total)]
    trimmed = Scheduler._trim_exact_resident_head(live, total, durable)
    assert scheduler._exact_resident_cache.put(
        list(range(total)), trimmed, cache_nbytes=1, durable_tokens=durable
    )
    prompt = list(range(total + 5))

    class _ShortPaged(_FakePaged):
        def fetch_cache(self, request_id, prompt_tokens, **kwargs):
            return _FakeTable(len(prompt_tokens) - 100, n_blocks=1), []

    scheduler.block_aware_cache = _ShortPaged([_head_layer(durable)])
    scheduler.paged_cache_manager = object()
    hit = scheduler._exact_resident_cache.acquire_prefix(prompt)
    assert hit is not None
    request = _request(prompt)
    assert scheduler._restore_trimmed_exact_hit(request, hit, prompt) is False
    assert getattr(request, "_exact_resident_hit", False) is False


def test_publish_trims_when_allowed_and_keeps_full_on_kill_switch(monkeypatch):
    import os

    total, durable = 5000, 4096
    monkeypatch.setenv("OMLX_EXACT_TAIL_TRIM", "1")
    scheduler = _scheduler()
    request = _request(list(range(100)))
    live = [_kv_layer(total), _gdn_layer()]
    request._exact_resident_candidate = (
        list(range(total)),
        live,
        Scheduler._resident_cache_nbytes(live),
    )
    request._exact_resident_durable_fallback_tokens = durable
    assert scheduler._publish_exact_resident_cache(request) is True
    hit = scheduler._exact_resident_cache.acquire_prefix(
        list(range(total + 10))
    )
    assert hit is not None
    assert Scheduler._is_tail_trimmed_cache(hit.cache)

    monkeypatch.setenv("OMLX_EXACT_TAIL_TRIM", "0")
    scheduler2 = _scheduler()
    request2 = _request(list(range(100)))
    live2 = [_kv_layer(total), _gdn_layer()]
    request2._exact_resident_candidate = (
        list(range(total)),
        live2,
        Scheduler._resident_cache_nbytes(live2),
    )
    request2._exact_resident_durable_fallback_tokens = durable
    assert scheduler2._publish_exact_resident_cache(request2) is True
    hit2 = scheduler2._exact_resident_cache.acquire_prefix(
        list(range(total + 10))
    )
    assert hit2 is not None
    assert not Scheduler._is_tail_trimmed_cache(hit2)


def test_publish_falls_back_to_full_for_qsa_layouts():
    from test_exact_resident_cache import QSAKVCache

    scheduler = _scheduler()
    total, durable = 64, 32
    request = _request(list(range(10)))
    live = [QSAKVCache(offset=total)]
    request._exact_resident_candidate = (
        list(range(total)),
        live,
        Scheduler._resident_cache_nbytes(live),
    )
    request._exact_resident_durable_fallback_tokens = durable
    assert scheduler._publish_exact_resident_cache(request) is True
    hit = scheduler._exact_resident_cache.acquire_prefix(
        list(range(total + 5))
    )
    assert hit is not None
    assert not Scheduler._is_tail_trimmed_cache(hit.cache)


def test_stable_boundary_trims_to_tail_only():
    scheduler = _scheduler()
    total, durable = 4096 + 512, 4096
    live = [_kv_layer(total), _kv_layer(total)]
    trimmed = Scheduler._trim_exact_resident_head(
        live, total, durable, max_tail_tokens=2048
    )
    assert trimmed is not None
    # KV-only stable boundary: every layer marked, nothing fixed retained.
    assert all(
        getattr(leaf, "_omlx_kv_tail_span", None) == (durable, total)
        for leaf in trimmed
    )
    assert scheduler._validate_trimmed_tail(trimmed, total, durable)


def test_tail_bound_rejects_multi_block_tails():
    total, durable = 8192, 2048
    live = [_kv_layer(total)]
    assert (
        Scheduler._trim_exact_resident_head(
            live, total, durable, max_tail_tokens=2048
        )
        is None
    )
    assert (
        Scheduler._trim_exact_resident_head(live, total, durable) is not None
    )


def test_stable_trimmed_hit_falls_back_and_returns_entry():
    scheduler = _scheduler()
    scheduler._gdn_split_active = lambda: True
    total, durable = 5000, 4096
    live = [_kv_layer(total)]
    trimmed = Scheduler._trim_exact_resident_head(live, total, durable)
    assert trimmed is not None
    prompt = list(range(total + 9))
    assert scheduler._exact_resident_cache.put(
        list(range(total)), trimmed, cache_nbytes=7, durable_tokens=durable
    )
    hit = scheduler._exact_resident_cache.acquire_prefix(prompt)
    assert hit is not None
    request = _request(prompt)
    scheduler.paged_cache_manager = object()
    # No retained recurrent state: must NOT compose (would corrupt
    # generation on GDN-split models).
    assert scheduler._restore_trimmed_exact_hit(request, hit, prompt) is False
    assert getattr(request, "_exact_resident_hit", False) is False
    # Entry was returned to the tier instead of consumed.
    hit2 = scheduler._exact_resident_cache.acquire_prefix(prompt)
    assert hit2 is not None
    assert Scheduler._is_tail_trimmed_cache(hit2.cache)


def test_terminal_trimmed_hit_composes_with_gdn_active():
    scheduler = _scheduler()
    scheduler._gdn_split_active = lambda: True
    total, durable = 5000, 4096
    live = [_kv_layer(total), _gdn_layer()]
    trimmed = Scheduler._trim_exact_resident_head(live, total, durable)
    prompt = list(range(total + 11))
    assert scheduler._exact_resident_cache.put(
        list(range(total)), trimmed, cache_nbytes=9, durable_tokens=durable
    )
    scheduler.block_aware_cache = _FakePaged(
        [_head_layer(durable), _gdn_layer()]
    )
    scheduler.paged_cache_manager = object()
    hit = scheduler._exact_resident_cache.acquire_prefix(prompt)
    request = _request(prompt)
    assert scheduler._restore_trimmed_exact_hit(request, hit, prompt) is True
    assert request.cached_tokens == total
    assert request.remaining_tokens == prompt[total:]
