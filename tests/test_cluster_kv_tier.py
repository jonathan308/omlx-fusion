# SPDX-License-Identifier: Apache-2.0
"""Durable rank-local KV tier: roundtrip, validation, vote, and budget tests.

Everything here is mocked or uses tiny real MLX arrays — no model is ever
loaded. The collective discipline tests use a scripted ``all_sum``; the
budget acceptance test uses real artifact files in a temporary directory.
"""

from __future__ import annotations

import ast
import inspect
import threading

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache

import omlx.cluster.kv_tier as kv_tier
from omlx.cluster.kv_tier import (
    RankKVTierConfig,
    RankLocalKVTier,
    build_rank_kv_tier,
    cache_fingerprint,
)


def _fingerprint(classes=("KVCache", "KVCache"), max_kv_size=None) -> str:
    return cache_fingerprint(
        model_path="org/model",
        cache_classes=list(classes),
        start_layer=0,
        end_layer=len(classes),
        world_size=2,
        tensor_parallel_size=1,
        max_kv_size=max_kv_size,
    )


def _config(tmp_path, **overrides) -> RankKVTierConfig:
    values = {
        "local_enabled": True,
        "directory": tmp_path / "tier",
        "max_bytes": 1 << 40,
        "min_tokens": 8,
        "append_reserve_tokens": 0,
    }
    values.update(overrides)
    return RankKVTierConfig(**values)


def _tier(tmp_path, *, rank=0, world_size=1, classes=("KVCache", "KVCache"), **config):
    subdir = config.pop("subdir", "tier")
    max_kv_size = config.pop("max_kv_size", None)
    resolved = _config(tmp_path, directory=tmp_path / subdir, **config)
    return RankLocalKVTier(
        resolved,
        rank=rank,
        world_size=world_size,
        model_fingerprint=_fingerprint(classes),
        cache_factory=lambda: [_fresh(cls) for cls in classes],
        max_kv_size=max_kv_size,
    )


class MiniMaxM3KVCache:
    """Mirror of the vendored cluster model's sparse-index cache contract.

    Nested ``(kv_state, index_state)`` with a None-able index leaf, an offset
    property that reseats both offsets, and a string ``meta_state`` — the
    shape the production two-Mac MiniMax deployment hands the tier.
    """

    def __init__(self):
        self.kv_cache = KVCache()
        self.index_keys = None
        self.index_offset = 0

    @property
    def offset(self):
        return self.kv_cache.offset

    @offset.setter
    def offset(self, value):
        self.kv_cache.offset = int(value)
        self.index_offset = int(value)

    @property
    def state(self):
        kv_state = None if self.kv_cache.empty() else self.kv_cache.state
        index_state = (
            None
            if self.index_keys is None
            else self.index_keys[..., : self.index_offset, :]
        )
        return kv_state, index_state

    @state.setter
    def state(self, value):
        kv_state, index_state = value
        self.kv_cache = KVCache()
        if kv_state is not None:
            self.kv_cache.state = kv_state
        self.index_keys = index_state
        self.index_offset = 0 if index_state is None else index_state.shape[2]

    @property
    def meta_state(self):
        return str(self.index_offset)

    @meta_state.setter
    def meta_state(self, value):
        self.index_offset = int(value) if value else 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self.index_offset = max(0, self.index_offset - trimmed)
        return trimmed

    def empty(self):
        return self.kv_cache.empty()

    @property
    def nbytes(self):
        index_bytes = 0 if self.index_keys is None else self.index_keys.nbytes
        return self.kv_cache.nbytes + index_bytes


def _fresh(class_name):
    if class_name == "KVCache":
        return KVCache()
    if class_name == "QuantizedKVCache":
        return QuantizedKVCache(group_size=32, bits=4)
    if class_name == "RotatingKVCache":
        return RotatingKVCache(max_size=64, keep=4)
    if class_name == "MiniMaxM3KVCache":
        return MiniMaxM3KVCache()
    raise AssertionError(f"unknown test cache class {class_name}")


def _minimax_stack(tokens, *, with_index=True):
    """A sparse-index stack whose kv and index lengths equal len(tokens)."""

    (cache,) = _conversation(tokens, layers=1)
    hybrid = MiniMaxM3KVCache()
    hybrid.kv_cache = cache
    if with_index:
        hybrid.index_keys = mx.full(
            (1, 1, len(tokens), 2), 0.5, dtype=mx.float16
        )
        hybrid.index_offset = len(tokens)
    return [hybrid, *_conversation(tokens, layers=1)]


def _conversation(tokens, layers=2, fill=0.0):
    """A trimmable two-layer stack whose offsets equal len(tokens)."""

    caches = []
    for _ in range(layers):
        cache = KVCache()
        cache.update_and_fetch(
            mx.full((1, 2, len(tokens), 4), fill, dtype=mx.float16),
            mx.full((1, 2, len(tokens), 4), fill, dtype=mx.float16),
        )
        caches.append(cache)
    return caches


def _materialized(tokens, layers=2, fill=0.0):
    """A conversation whose state is evaluated, as the serving loop leaves it.

    Async saves hand arrays to the tier's saver thread; production cache
    state is materialized by the generation loop long before insert, and the
    worker's saver adopts the generation stream for anything still lazy.
    """

    caches = _conversation(tokens, layers=layers, fill=fill)
    for cache in caches:
        mx.eval(cache.state)
    return caches


class _FakeArray:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeCollectives:
    """A scripted all_sum returning the cluster-wide summed offer vector."""

    def __init__(self, summed):
        self.summed = summed
        self.calls = 0

    def all_sum(self, values):
        self.calls += 1
        return _FakeArray(self.summed)


class _FakeMX:
    def __init__(self, summed):
        self.distributed = _FakeCollectives(summed)

    def array(self, values):
        return list(values)


# --- Rank-local configuration -----------------------------------------------


def test_config_defaults_and_rank_aware_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv(kv_tier.KV_TIER_ENV, raising=False)
    monkeypatch.delenv(kv_tier.KV_TIER_BYTES_ENV, raising=False)
    config = RankKVTierConfig.from_env(rank=1)
    assert config.local_enabled is False  # killswitch default preserves today
    assert config.max_bytes == kv_tier._DEFAULT_TIER_BYTES
    assert config.min_tokens == kv_tier._DEFAULT_MIN_TOKENS
    assert config.append_reserve_tokens == (
        kv_tier._DEFAULT_APPEND_RESERVE_TOKENS
    )
    assert config.directory.name == "rank-1"

    monkeypatch.setenv(kv_tier.KV_TIER_ENV, "1")
    monkeypatch.setenv(kv_tier.KV_TIER_BYTES_ENV, "1000")
    monkeypatch.setenv(f"{kv_tier.KV_TIER_BYTES_ENV}_RANK1", "2000")
    monkeypatch.setenv(f"{kv_tier.KV_TIER_APPEND_RESERVE_ENV}_RANK0", "8192")
    rank_one = RankKVTierConfig.from_env(rank=1)
    rank_zero = RankKVTierConfig.from_env(rank=0)
    assert rank_one.local_enabled is True
    # Each rank's disk budget is its own: the per-rank key wins on rank 1 and
    # is invisible to rank 0 — budgets are never shared across ranks.
    assert rank_one.max_bytes == 2000
    assert rank_zero.max_bytes == 1000
    assert rank_zero.append_reserve_tokens == 8192
    assert rank_one.append_reserve_tokens == (
        kv_tier._DEFAULT_APPEND_RESERVE_TOKENS
    )


def test_config_tolerates_garbage_rank_local_values(monkeypatch, tmp_path):
    monkeypatch.setenv(kv_tier.KV_TIER_BYTES_ENV, "not-a-number")
    monkeypatch.setenv(kv_tier.KV_TIER_MIN_TOKENS_ENV, "-5")
    config = RankKVTierConfig.from_env(rank=0)
    assert config.max_bytes == kv_tier._DEFAULT_TIER_BYTES
    assert config.min_tokens == 1


# --- Save validation ---------------------------------------------------------


def test_save_refuses_untrimmable_or_offset_mismatched_stacks(tmp_path):
    tier = _tier(tmp_path)
    tokens = list(range(16))

    class Windowed:
        offset = 16
        nbytes = 1

        def is_trimmable(self):
            return False

    assert not tier.save(tokens, [Windowed(), *_conversation(tokens, 1)])

    short = KVCache()  # offset 0 while the key claims 16 tokens
    assert not tier.save(tokens, [short, *_conversation(tokens, 1)])
    assert tier.artifact_count == 0
    tier.close()


def test_save_refuses_prompts_below_the_minimum_length(tmp_path):
    tier = _tier(tmp_path, min_tokens=32)
    tokens = list(range(16))
    assert not tier.save(tokens, _conversation(tokens))
    assert tier.artifact_count == 0
    tier.close()


# --- Roundtrip + the bounded append reserve ----------------------------------


def test_save_restore_roundtrip_with_bounded_append_reserve(tmp_path):
    tier = _tier(tmp_path, append_reserve_tokens=16)
    tokens = list(range(40))
    caches = _conversation(tokens, fill=1.5)
    assert tier.save(tokens, caches)

    result = tier.restore_prompt_cache(tokens + [100, 101], mx_module=mx)
    assert result is not None
    restored, rest = result
    assert rest == [100, 101]
    for original, cache in zip(caches, restored):
        assert cache.offset == 40
        # The acceptance gate: the restored capacity is the logical length
        # plus the bounded reserve — never the request's full output ceiling.
        assert cache.keys.shape[2] == 40 + 16
        assert bool(mx.allclose(cache.keys[..., :40, :], original.state[0]))

    # The first append lands in the reserve without reallocating the backing.
    for cache in restored:
        keys, _ = cache.update_and_fetch(
            mx.ones((1, 2, 4, 4), dtype=mx.float16),
            mx.ones((1, 2, 4, 4), dtype=mx.float16),
        )
        assert cache.offset == 44
        assert keys.shape[2] == 44
        assert float(mx.abs(keys[..., 40:44, :] - 1).max()) == 0.0
    tier.close()


def test_restore_reserve_is_capped_by_max_kv_size(tmp_path):
    tier = _tier(tmp_path, append_reserve_tokens=16, max_kv_size=44)
    tokens = list(range(40))
    assert tier.save(tokens, _conversation(tokens))
    result = tier.restore_prompt_cache(tokens, mx_module=mx)
    assert result is not None
    restored, rest = result
    assert rest == []
    assert restored[0].keys.shape[2] == 44  # capped, not 40 + 16
    tier.close()


def test_quantized_cache_roundtrip_restores_meta_state_and_pads(tmp_path):
    tier = _tier(
        tmp_path,
        classes=("QuantizedKVCache",),
        append_reserve_tokens=8,
    )
    tokens = list(range(24))
    cache = QuantizedKVCache(group_size=32, bits=4)
    cache.update_and_fetch(
        mx.random.normal((1, 2, 24, 64)).astype(mx.float16),
        mx.random.normal((1, 2, 24, 64)).astype(mx.float16),
    )
    assert tier.save(tokens, [cache])
    result = tier.restore_prompt_cache(tokens + [7], mx_module=mx)
    assert result is not None
    restored, rest = result
    assert rest == [7]
    (cache,) = restored
    assert cache.offset == 24
    assert cache.group_size == 32 and cache.bits == 4
    assert cache.keys[0].shape[2] == 32  # 24 logical + 8 reserve, packed axis
    cache.update_and_fetch(
        mx.random.normal((1, 2, 4, 64)).astype(mx.float16),
        mx.random.normal((1, 2, 4, 64)).astype(mx.float16),
    )
    assert cache.offset == 28
    tier.close()


def test_rotating_cache_roundtrip_caps_the_reserve_at_its_window(tmp_path):
    tier = _tier(
        tmp_path,
        classes=("RotatingKVCache",),
        append_reserve_tokens=100,
    )
    tokens = list(range(24))
    cache = RotatingKVCache(max_size=64, keep=4)
    cache.update_and_fetch(
        mx.random.normal((1, 2, 24, 8)).astype(mx.float16),
        mx.random.normal((1, 2, 24, 8)).astype(mx.float16),
    )
    assert cache.is_trimmable()
    assert tier.save(tokens, [cache])
    result = tier.restore_prompt_cache(tokens + [9], mx_module=mx)
    assert result is not None
    (restored,) = result[0]
    assert restored.keys.shape[2] == 64  # min(24 + 100, max_size)
    assert restored.offset == 24
    tier.close()


def test_a_rotated_window_is_not_a_prefix_and_is_never_saved(tmp_path):
    tier = _tier(tmp_path, classes=("RotatingKVCache",))
    cache = RotatingKVCache(max_size=16, keep=4)
    for _ in range(5):
        cache.update_and_fetch(
            mx.random.normal((1, 1, 8, 4)).astype(mx.float16),
            mx.random.normal((1, 1, 8, 4)).astype(mx.float16),
        )
    assert cache.offset == 40 and not cache.is_trimmable()
    assert not tier.save(list(range(40)), [cache])
    assert tier.artifact_count == 0
    tier.close()


def test_minimax_sparse_index_stack_roundtrips_with_index_and_reserve(tmp_path):
    tier = _tier(
        tmp_path,
        classes=("MiniMaxM3KVCache", "KVCache"),
        append_reserve_tokens=8,
    )
    tokens = list(range(24))
    assert tier.save(tokens, _minimax_stack(tokens))

    result = tier.restore_prompt_cache(tokens + [7], mx_module=mx)
    assert result is not None
    (hybrid, plain), rest = result[0], result[1]
    assert rest == [7]
    assert hybrid.offset == 24 and hybrid.index_offset == 24
    # Both the KV leaves and the sparse-index leaf got the bounded reserve.
    assert hybrid.kv_cache.keys.shape[2] == 32
    assert hybrid.index_keys.shape[2] == 32
    assert float(mx.abs(hybrid.index_keys[..., :24, :] - 0.5).max()) == 0.0
    assert plain.offset == 24 and plain.keys.shape[2] == 32
    tier.close()


def test_minimax_stack_roundtrips_when_the_index_leaf_is_absent(tmp_path):
    tier = _tier(tmp_path, classes=("MiniMaxM3KVCache", "KVCache"))
    tokens = list(range(24))
    assert tier.save(tokens, _minimax_stack(tokens, with_index=False))

    result = tier.restore_prompt_cache(tokens, mx_module=mx)
    assert result is not None
    (hybrid, plain) = result[0]
    assert hybrid.offset == 24
    assert hybrid.index_keys is None and hybrid.index_offset == 0
    assert plain.offset == 24
    tier.close()


# --- Validation before restore -----------------------------------------------


def test_restore_misses_on_fingerprint_token_and_class_mismatches(tmp_path):
    tokens = list(range(24))
    caches = _conversation(tokens)
    tier = _tier(tmp_path)
    assert tier.save(tokens, caches)

    # A different model identity never even matches.
    other = RankLocalKVTier(
        _config(tmp_path),
        rank=0,
        world_size=1,
        model_fingerprint=_fingerprint(classes=("KVCache",)),
        cache_factory=lambda: [KVCache()],
    )
    assert other.match(tokens) is None
    assert other.restore_prompt_cache(tokens, mx_module=mx) is None

    # Same length, different content: the prefix hash gates the offer.
    assert tier.match([999] * len(tokens)) is None

    # A factory whose stack disagrees with the artifact fails validation even
    # with a matching fingerprint.
    mismatched = RankLocalKVTier(
        _config(tmp_path),
        rank=0,
        world_size=1,
        model_fingerprint=tier._fingerprint,
        cache_factory=lambda: [KVCache()],  # one layer where two were saved
    )
    record = tier.match(tokens)
    assert mismatched._load_artifact(record) is None
    tier.close()


def test_restore_misses_safely_on_schema_tampering_and_lost_files(tmp_path):
    tokens = list(range(24))
    tier = _tier(tmp_path)
    assert tier.save(tokens, _conversation(tokens))
    record = tier.match(tokens)
    path = tmp_path / "tier" / record["file"]

    # Foreign schema version: load the real arrays, rewrite the metadata.
    arrays, metadata = mx.load(str(path), return_metadata=True)
    metadata["schema_version"] = "0"
    mx.save_safetensors(str(path), arrays, metadata)
    assert tier._load_artifact(record) is None

    # Torn/corrupt file.
    path.write_bytes(b"not safetensors")
    assert tier._load_artifact(record) is None

    # Vanished file.
    path.unlink()
    assert tier._load_artifact(record) is None
    tier.close()


# --- The min-prefix vote ------------------------------------------------------


def test_vote_is_the_cluster_minimum_and_skipped_without_peers(tmp_path):
    tier = _tier(tmp_path, world_size=2, rank=0)
    fake = _FakeMX([32, 16])
    assert tier.agree_restore_tokens(32, mx_module=fake) == 16
    assert fake.distributed.calls == 1
    # A peer offering nothing forces cold — that is the skew-heal path.
    assert tier.agree_restore_tokens(32, mx_module=_FakeMX([32, 0])) == 0
    # No peers: no collective at all.
    solo = _tier(tmp_path, world_size=1, subdir="solo")
    assert solo.agree_restore_tokens(32, mx_module=_FakeMX([])) == 32
    tier.close()
    solo.close()


def test_restore_trims_the_longer_artifact_to_the_agreed_prefix(tmp_path):
    long_tokens = list(range(32))
    short_tokens = list(range(16))
    rank_zero = _tier(
        tmp_path, world_size=2, rank=0, subdir="r0", append_reserve_tokens=4
    )
    rank_one = _tier(
        tmp_path, world_size=2, rank=1, subdir="r1", append_reserve_tokens=4
    )
    assert rank_zero.save(long_tokens, _conversation(long_tokens))
    assert rank_one.save(short_tokens, _conversation(short_tokens))

    prompt = long_tokens + [99, 98]
    # Both ranks read the summed offers [32, 16]; agreed = 16.
    result_zero = rank_zero.restore_prompt_cache(prompt, mx_module=_FakeMX([32, 16]))
    result_one = rank_one.restore_prompt_cache(prompt, mx_module=_FakeMX([32, 16]))
    assert result_zero is not None and result_one is not None
    for result in (result_zero, result_one):
        caches, rest = result
        assert rest == prompt[16:]
        assert caches[0].offset == 16
        assert caches[0].keys.shape[2] == 20  # agreed + reserve on both ranks
    rank_zero.close()
    rank_one.close()


def test_restore_vote_still_runs_once_when_the_local_probe_fails(tmp_path):
    tier = _tier(tmp_path, world_size=2, rank=0)
    tokens = list(range(16))
    assert tier.save(tokens, _conversation(tokens))
    record = tier.match(tokens)
    (tmp_path / "tier" / record["file"]).unlink()  # the disk read will fail

    fake = _FakeMX([0, 0])
    assert tier.restore_prompt_cache(tokens, mx_module=fake) is None
    # The vote is the contract with the peers: it must fire even when this
    # rank has nothing to offer, or the group hangs waiting for it.
    assert fake.distributed.calls == 1
    tier.close()


def test_a_locally_disabled_tier_offers_nothing_but_still_votes(tmp_path):
    tier = RankLocalKVTier(
        _config(tmp_path, local_enabled=False),
        rank=1,
        world_size=2,
        model_fingerprint=_fingerprint(),
        cache_factory=lambda: [KVCache(), KVCache()],
    )
    tokens = list(range(16))
    assert not tier.save(tokens, _conversation(tokens))
    fake = _FakeMX([16, 0])
    assert tier.restore_prompt_cache(tokens, mx_module=fake) is None
    assert fake.distributed.calls == 1


# --- Bounded-capacity acceptance ----------------------------------------------


def test_the_tier_never_exceeds_its_configured_budget(tmp_path):
    """Acceptance: interleaved saves and restores keep disk use <= budget."""

    probe = _tier(tmp_path, subdir="probe")
    lengths = [16, 24, 32, 40]
    for length in lengths:
        tokens = list(range(length))
        assert probe.save(tokens, _conversation(tokens))
    sizes = {
        length: (tmp_path / "probe" / record["file"]).stat().st_size
        for length in lengths
        for record in [probe.match(list(range(length)))]
    }
    probe.close()

    # Exactly the two smallest artifacts fit.
    budget = sizes[16] + sizes[24]
    tier = _tier(tmp_path, subdir="budget", max_bytes=budget)
    for length in lengths:
        tokens = list(range(length))
        assert tier.save(tokens, _conversation(tokens)) == (sizes[length] <= budget)
        assert tier.total_bytes <= budget
        # Restores between saves must not change the invariant either.
        tier.restore_prompt_cache(tokens, mx_module=mx)
        assert tier.total_bytes <= budget
    # The 32- and 40-token artifacts each exceed budget - sizes[24], so only
    # the most recent survivors remain.
    assert tier.artifact_count >= 1
    tier.close()


def test_a_single_artifact_larger_than_the_budget_is_not_kept(tmp_path):
    tier = _tier(tmp_path, max_bytes=1)
    tokens = list(range(16))
    assert not tier.save(tokens, _conversation(tokens))
    assert tier.artifact_count == 0
    assert tier.total_bytes == 0
    tier.close()


def test_eviction_is_lru_by_last_access_not_insertion_order(tmp_path):
    probe = _tier(tmp_path, subdir="probe")
    saved = {}
    for length in (16, 24, 32):
        tokens = list(range(length))
        assert probe.save(tokens, _conversation(tokens))
        saved[length] = (
            tmp_path / "probe" / probe.match(tokens)["file"]
        ).stat().st_size
    probe.close()

    # Holds exactly {16, 32}: adding the 32-token artifact to {16, 24} must
    # evict exactly one, and which one is the LRU question under test.
    budget = saved[16] + saved[32]
    tier = _tier(tmp_path, subdir="budget", max_bytes=budget)
    tokens16, tokens24 = list(range(16)), list(range(24))
    assert tier.save(tokens16, _conversation(tokens16))
    assert tier.save(tokens24, _conversation(tokens24))
    assert tier.artifact_count == 2

    # Touch the 16-token artifact so the 24-token one becomes the oldest.
    assert tier.restore_prompt_cache(tokens16, mx_module=mx) is not None
    tokens32 = list(range(32))
    assert tier.save(tokens32, _conversation(tokens32))
    surviving = {record["token_count"] for record in tier._artifacts.values()}
    # The 24-token artifact was saved after the 16-token one, and insertion
    # order would have evicted 16; last-access order evicts 24 instead.
    assert surviving == {16, 32}
    assert tier.total_bytes <= budget
    tier.close()


def test_eviction_is_collective_free_by_construction():
    """The budget path must never touch mlx or the distributed runtime."""

    import textwrap

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(RankLocalKVTier._enforce_budget))
    )
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
    assert "mx" not in identifiers
    assert "distributed" not in identifiers

    # And the module itself must not import mlx/mlx-lm at import time: the
    # worker imports it before installing its torch stub and importing mlx.
    tree = ast.parse(inspect.getsource(kv_tier))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all("mlx" not in alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert "mlx" not in (node.module or "")


# --- Persistence, autosave, and construction ----------------------------------


def test_artifacts_survive_a_restart_and_orphans_are_swept(tmp_path):
    tokens = list(range(24))
    tier = _tier(tmp_path)
    assert tier.save(tokens, _conversation(tokens))
    stray = tmp_path / "tier" / "stray.safetensors"
    stray.write_bytes(b"interrupted write")
    tier.close()

    reopened = _tier(tmp_path)
    assert reopened.artifact_count == 1
    assert reopened.total_bytes > 0
    assert not stray.exists()
    # Simulated repeated prompt after a cluster restart: full restore.
    result = reopened.restore_prompt_cache(tokens + [1, 2], mx_module=mx)
    assert result is not None
    assert result[1] == [1, 2]
    reopened.close()


def test_save_async_completes_and_drops_the_oldest_pending_save(
    tmp_path, monkeypatch
):
    tier = _tier(tmp_path)
    started = threading.Event()
    release = threading.Event()
    saved = []
    original = tier._save_guarded

    def blocking(tokens, caches):
        started.set()
        release.wait(5)
        saved.append(list(tokens))
        return original(tokens, caches)

    monkeypatch.setattr(tier, "_save_guarded", blocking)
    first, second, third = (list(range(n)) for n in (16, 24, 32))
    tier.save_async(first, _materialized(first))
    assert started.wait(5)
    tier.save_async(second, _materialized(second))
    tier.save_async(third, _materialized(third))
    release.set()
    assert tier.flush(5)
    # One save in flight, one pending slot: the middle one was superseded.
    assert saved == [first, third]
    assert tier.artifact_count == 2
    tier.close()


def test_build_rank_kv_tier_constructs_only_for_enabled_plans(tmp_path, monkeypatch):
    monkeypatch.setenv(kv_tier.KV_TIER_ENV, "1")
    monkeypatch.setenv(kv_tier.KV_TIER_DIR_ENV, str(tmp_path))

    class _Model:
        layers = (object(), object())

    tier = build_rank_kv_tier(
        plan_enabled=True,
        rank=0,
        world_size=2,
        model_path="org/model",
        model=_Model(),
        start_layer=0,
        end_layer=2,
        tensor_parallel_size=1,
        max_kv_size=None,
    )
    assert isinstance(tier, RankLocalKVTier)
    assert tier.stats()["enabled"] is True
    tier.close()

    assert (
        build_rank_kv_tier(
            plan_enabled=False,
            rank=0,
            world_size=2,
            model_path="org/model",
            model=_Model(),
            start_layer=0,
            end_layer=2,
            tensor_parallel_size=1,
            max_kv_size=None,
        )
        is None
    )

    # A model whose caches cannot be probed yields a disabled-but-voting tier:
    # the object exists so the restore vote still runs on this rank.
    degraded = build_rank_kv_tier(
        plan_enabled=True,
        rank=1,
        world_size=2,
        model_path="org/model",
        model=object(),
        start_layer=0,
        end_layer=2,
        tensor_parallel_size=1,
        max_kv_size=None,
    )
    assert degraded is not None
    assert degraded.stats()["enabled"] is False
    fake = _FakeMX([0, 0])
    assert degraded.restore_prompt_cache(list(range(16)), mx_module=fake) is None
    assert fake.distributed.calls == 1


# --- Telemetry wiring: the reuse ladder in the serving path -------------------


class _Marker:
    def __init__(self):
        self.updates = []

    def update(self, phase, **extra):
        self.updates.append((phase, extra))


class _TrieEntry:
    nbytes = 1

    def is_trimmable(self):
        return True


class _StubTier:
    def __init__(self, restored=None):
        self.restored = restored
        self.restore_calls = []
        self.saved = []

    def restore_prompt_cache(self, tokens, *, mx_module):
        self.restore_calls.append(list(tokens))
        return self.restored

    def save_async(self, tokens, caches):
        self.saved.append((list(tokens), list(caches)))


def test_prefetch_consults_the_tier_only_on_a_full_ram_miss():
    import mlx_lm.server as mlx_server

    from omlx.cluster.telemetry import install_server_telemetry

    tier = _StubTier(restored=(["ssd-cache"], [3, 4]))
    with install_server_telemetry(_Marker(), kv_tier=tier):
        cache = mlx_server.LRUPromptCache(max_size=4)
        # RAM miss -> the durable rung answers and is replayed to mlx-lm.
        result = cache.prefetch_nearest_cache("model", [1, 2, 3, 4])
        assert result == (["ssd-cache"], [3, 4])
        assert tier.restore_calls == [[1, 2, 3, 4]]
        assert cache.fetch_nearest_cache("model", [1, 2, 3, 4]) == (
            ["ssd-cache"],
            [3, 4],
        )
        assert tier.restore_calls == [[1, 2, 3, 4]]  # replay, not a second vote

        # A resident RAM hit wins the ladder; SSD is never consulted.
        cache.insert_cache("model", [1, 2], [_TrieEntry()])
        tier.restore_calls.clear()
        result = cache.prefetch_nearest_cache("model", [1, 2, 3, 4])
        assert tier.restore_calls == []
        assert result[1] == [3, 4]  # two resident tokens reused
        tier.saved.clear()


def test_insert_cache_autosaves_finished_conversations():
    import mlx_lm.server as mlx_server

    from omlx.cluster.telemetry import install_server_telemetry

    tier = _StubTier()
    with install_server_telemetry(_Marker(), kv_tier=tier):
        cache = mlx_server.LRUPromptCache(max_size=4)
        cache.insert_cache("model", [1, 2, 3], [_TrieEntry()])
        assert len(tier.saved) == 1
        tokens, caches = tier.saved[0]
        assert tokens == [1, 2, 3]
        assert len(caches) == 1
