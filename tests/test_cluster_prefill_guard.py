# SPDX-License-Identifier: Apache-2.0
"""A rank must refuse a prompt it cannot prefill — without hanging its peers."""

from __future__ import annotations

import pytest

from omlx.cluster.prefill_guard import (
    _DEFAULT_PREFILL_STEP,
    ADMISSION_ACTIVATION_RESERVE_BYTES,
    ADMISSION_KV_BYTES_PER_TOKEN,
    ADMISSION_MIN_PROMPT_TOKENS,
    ADMISSION_SAFETY_FRACTION,
    RankPrefillGuard,
    admission_deficit_bytes,
    build_guard,
    plan_eviction,
    rank_monitor,
    run_admission,
)
from omlx.exceptions import PrefillMemoryExceededError

GiB = 1024**3
MiB = 1024**2


class _Config:
    """The dims mlx-lm models expose, minimal and real (Qwen3-32B shaped)."""

    num_hidden_layers = 64
    num_key_value_heads = 8
    num_attention_heads = 64
    head_dim = 128
    hidden_size = 5120


class _Model:
    args = _Config()


def _guard(*, layer_count=0, tp=1, ceiling=8 * GiB, rank=0) -> RankPrefillGuard:
    return RankPrefillGuard(
        rank_monitor(_Model(), layer_count=layer_count, tensor_parallel_size=tp),
        rank=rank,
        node_id="studio",
        ceiling_bytes=ceiling,
    )


def test_a_prompt_that_would_not_fit_is_refused():
    guard = _guard(ceiling=4 * GiB)
    with pytest.raises(PrefillMemoryExceededError) as excinfo:
        guard.check(200_000, current_usage_bytes=3 * GiB)
    assert "Prefill would require" in str(excinfo.value)


def test_a_prompt_that_fits_is_allowed():
    _guard(ceiling=64 * GiB).check(2048, current_usage_bytes=1 * GiB)


def test_a_pipeline_rank_is_only_charged_for_the_layers_it_holds():
    """The whole point: 16 of 64 layers must not be charged 64 layers of KV."""

    whole = rank_monitor(_Model())
    stage = rank_monitor(_Model(), layer_count=16)
    assert stage.estimate_prompt_kv_bytes(8192) == pytest.approx(
        whole.estimate_prompt_kv_bytes(8192) / 4, rel=0.01
    )


def test_a_stage_accepts_a_prompt_the_whole_model_would_refuse():
    """Not just smaller arithmetic — a prompt that is served instead of 400ed.

    The threshold is derived from the two estimates rather than guessed, so
    the test states the property and cannot drift with the SDPA model.
    """

    tokens, usage = 120_000, 4 * GiB
    whole = rank_monitor(_Model())
    stage = rank_monitor(_Model(), layer_count=16)
    stage_peak = stage.estimate_prefill_peak_bytes(tokens, 2048)
    whole_peak = whole.estimate_prefill_peak_bytes(tokens, 2048)
    assert stage_peak < whole_peak

    # A ceiling between the two: the uncorrected guard rejects, the corrected
    # one serves.
    ceiling = int(usage + (stage_peak + whole_peak) / 2)
    with pytest.raises(PrefillMemoryExceededError):
        _guard(ceiling=ceiling).check(tokens, current_usage_bytes=usage)
    _guard(layer_count=16, ceiling=ceiling).check(tokens, current_usage_bytes=usage)


def test_a_tensor_parallel_rank_is_charged_for_its_head_shard():
    whole = rank_monitor(_Model())
    half = rank_monitor(_Model(), tensor_parallel_size=2)
    assert half.estimate_prompt_kv_bytes(8192) == pytest.approx(
        whole.estimate_prompt_kv_bytes(8192) / 2, rel=0.01
    )


def test_cached_tokens_are_not_charged_twice():
    """Prefix-cache hits are already resident; charging them over-rejects."""

    guard = _guard(ceiling=6 * GiB)
    with pytest.raises(PrefillMemoryExceededError):
        guard.check(150_000, current_usage_bytes=4 * GiB)
    guard.check(150_000, cached_tokens=149_000, current_usage_bytes=4 * GiB)


# --- The desync rule: all ranks vote and leave the request together. ---------


def test_follower_ranks_guard_their_own_slice():
    follower = _guard(ceiling=1 * GiB, rank=1)
    assert follower.active
    with pytest.raises(PrefillMemoryExceededError):
        follower.check(500_000, current_usage_bytes=1 * GiB)


class _CollectiveValue:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class _CollectiveMX:
    def __init__(self, *, rank, votes):
        self._rank = rank
        self._votes = votes
        self.distributed = self

    def init(self):
        return self

    def rank(self):
        return self._rank

    def size(self):
        return len(self._votes)

    def array(self, value):
        return value

    def all_sum(self, _value):
        return _CollectiveValue(self._votes)


def test_peer_rejection_makes_an_accepting_rank_leave_before_model_execution():
    guard = _guard(ceiling=64 * GiB, rank=0)
    mx = _CollectiveMX(rank=0, votes=[0, 1])

    with pytest.raises(PrefillMemoryExceededError, match="rejected by rank 1"):
        guard.check_collective(
            2048,
            current_usage_bytes=1 * GiB,
            mx_module=mx,
        )


def test_collective_admission_allows_every_rank_to_continue():
    guard = _guard(ceiling=64 * GiB, rank=1)
    mx = _CollectiveMX(rank=1, votes=[0, 0])

    guard.check_collective(
        2048,
        current_usage_bytes=1 * GiB,
        mx_module=mx,
    )


def test_an_unreadable_model_disables_the_guard():
    guard = RankPrefillGuard(rank_monitor(object()), rank=0, ceiling_bytes=8 * GiB)
    assert not guard.active
    guard.check(500_000)


def test_no_ceiling_disables_the_guard():
    assert not _guard(ceiling=0).active


def test_build_guard_uses_this_macs_ceiling(monkeypatch):
    monkeypatch.setattr(
        "omlx.cluster.memory_guard.ceiling_breakdown",
        lambda tier: {"hard_limit": 12 * GiB},
    )
    guard = build_guard(_Model(), rank=0, node_id="mbp", layer_count=32)
    assert guard.active
    assert guard._ceiling == 12 * GiB


def test_build_guard_survives_a_host_with_no_enforcer(monkeypatch):
    def _boom(_tier):
        raise RuntimeError("no enforcer here")

    monkeypatch.setattr("omlx.cluster.memory_guard.ceiling_breakdown", _boom)
    assert not build_guard(_Model(), rank=0).active


# --- The fail-open admission ladder (ported pure logic). ---------------------


def test_admission_constants_match_the_production_values():
    """The 2026-07-19 wedge fix shipped these exact numbers."""
    min_tokens = ADMISSION_MIN_PROMPT_TOKENS
    fraction = ADMISSION_SAFETY_FRACTION
    kv_per_token = ADMISSION_KV_BYTES_PER_TOKEN
    reserve = ADMISSION_ACTIVATION_RESERVE_BYTES
    assert min_tokens == 8192
    assert fraction == 0.92
    assert kv_per_token == 90_000
    assert reserve == 8 * GiB


def test_admission_deficit_is_the_wired_headroom_shortfall():
    deficit = admission_deficit_bytes(
        8192,
        current_wired_bytes=100 * GiB,
        wired_limit_bytes=128 * GiB,
        kv_bytes_per_token=100_000,
        safety_fraction=0.92,
        activation_reserve_bytes=8 * GiB,
    )
    expected = (100 * GiB + 8192 * 100_000 + 8 * GiB) - 128 * GiB * 0.92
    assert deficit == pytest.approx(expected)


def test_admission_deficit_is_nonpositive_when_the_prompt_fits():
    assert admission_deficit_bytes(8192, 1 * GiB, 256 * GiB, kv_bytes_per_token=1) <= 0


def test_an_unknown_ceiling_disables_the_deficit():
    """Fail-open: a rank that cannot reason about headroom does nothing."""
    assert admission_deficit_bytes(1_000_000, 10**15, 0) == 0
    assert admission_deficit_bytes(1_000_000, 10**15, -5) == 0


def test_plan_eviction_picks_the_smallest_covering_set_largest_first():
    evictables = [
        {"label": "small", "bytes": 1 * GiB},
        {"label": "big", "bytes": 4 * GiB},
        {"label": "mid", "bytes": 2 * GiB},
    ]
    chosen, still_short = plan_eviction(5 * GiB, evictables)
    assert [item["label"] for item in chosen] == ["big", "mid"]
    assert still_short == 0


def test_plan_eviction_reports_the_uncoverable_remainder():
    chosen, still_short = plan_eviction(10 * GiB, [{"label": "a", "bytes": 3 * GiB}])
    assert [item["label"] for item in chosen] == ["a"]
    assert still_short == 7 * GiB


def test_plan_eviction_ignores_zero_byte_items_and_nonpositive_deficits():
    assert plan_eviction(0, [{"label": "a", "bytes": 1}]) == ([], 0)
    chosen, _ = plan_eviction(1, [{"label": "empty", "bytes": 0}])
    assert chosen == []


class _LadderRig:
    """Scripted callbacks for run_admission: ordering + fail-open invariants."""

    def __init__(self, *, wired, limit, trim_frees=0, evictables=(), agree=0):
        self._wired = list(wired)
        self._limit = limit
        self._trim_frees = trim_frees
        self._evictables = list(evictables)
        self._agree = agree
        self.trims = 0
        self.agree_calls = []
        self.dropped = []

    def read_wired(self):
        # Each call consumes the next scripted watermark; the last one sticks.
        return self._wired.pop(0) if len(self._wired) > 1 else self._wired[0]

    def read_limit(self):
        return self._limit

    def trim_pool(self):
        self.trims += 1
        return self._trim_frees

    def list_idle_evictables(self):
        evictables = []
        for label, nbytes in self._evictables:
            evictables.append(
                {
                    "label": label,
                    "bytes": nbytes,
                    "drop": lambda label=label: self.dropped.append(label),
                }
            )
        return evictables

    def agree_eviction(self, deficit):
        self.agree_calls.append(deficit)
        return self._agree


def test_the_ladder_ignores_small_prompts_entirely():
    rig = _LadderRig(wired=[1 * GiB], limit=8 * GiB)
    info = run_admission(
        ADMISSION_MIN_PROMPT_TOKENS - 1,
        read_wired=rig.read_wired,
        read_limit=rig.read_limit,
        trim_pool=rig.trim_pool,
        list_idle_evictables=rig.list_idle_evictables,
        agree_eviction=rig.agree_eviction,
    )
    assert info == {"guarded": False, "prompt_tokens": ADMISSION_MIN_PROMPT_TOKENS - 1}
    assert rig.trims == 0
    assert rig.agree_calls == []


def test_a_prompt_that_fits_still_votes_but_does_nothing_else():
    """The deficit vote is a collective: it must run even with no deficit."""
    rig = _LadderRig(wired=[1 * GiB] * 4, limit=256 * GiB)
    info = run_admission(
        ADMISSION_MIN_PROMPT_TOKENS,
        read_wired=rig.read_wired,
        read_limit=rig.read_limit,
        trim_pool=rig.trim_pool,
        list_idle_evictables=rig.list_idle_evictables,
        agree_eviction=rig.agree_eviction,
    )
    assert info["guarded"] and info["fits"]
    assert rig.trims == 0
    assert len(rig.agree_calls) == 1
    assert rig.agree_calls[0] == 0
    assert rig.dropped == []


def test_rung_one_trims_the_pool_before_anything_is_evicted():
    """A pool trim that covers the deficit makes rung 2 a no-op."""
    # deficit = 6 + 8192*0.1MB + 8 - 16*0.92 GiB; trim frees 4 GiB of pool.
    rig = _LadderRig(
        wired=[6 * GiB, 2 * GiB, 2 * GiB],
        limit=16 * GiB,
        trim_frees=4 * GiB,
        evictables=[("entry", 4 * GiB)],
        agree=0,
    )
    info = run_admission(
        8192,
        read_wired=rig.read_wired,
        read_limit=rig.read_limit,
        trim_pool=rig.trim_pool,
        list_idle_evictables=rig.list_idle_evictables,
        agree_eviction=rig.agree_eviction,
        kv_bytes_per_token=100_000,
    )
    assert rig.trims == 1
    assert [a["step"] for a in info["actions"]] == ["trim_pool"]
    assert rig.dropped == []


def test_rung_two_drops_the_agreed_covering_set_largest_first():
    rig = _LadderRig(
        wired=[115 * GiB] * 4,
        limit=128 * GiB,
        trim_frees=0,
        evictables=[("small", 1 * GiB), ("big", 4 * GiB), ("mid", 2 * GiB)],
        agree=5 * GiB,
    )
    info = run_admission(
        8192,
        read_wired=rig.read_wired,
        read_limit=rig.read_limit,
        trim_pool=rig.trim_pool,
        list_idle_evictables=rig.list_idle_evictables,
        agree_eviction=rig.agree_eviction,
        kv_bytes_per_token=100_000,
    )
    # The agreed target — not the local deficit — sizes the drop, and the
    # largest-first order keeps the smaller entries warm.
    assert rig.dropped == ["big", "mid"]
    steps = [a["step"] for a in info["actions"]]
    assert steps == ["trim_pool", "evict_idle", "evict_idle"]


def test_a_zero_agreed_target_suppresses_eviction_even_under_pressure():
    rig = _LadderRig(
        wired=[115 * GiB] * 4,
        limit=128 * GiB,
        evictables=[("big", 4 * GiB)],
        agree=0,
    )
    info = run_admission(
        8192,
        read_wired=rig.read_wired,
        read_limit=rig.read_limit,
        trim_pool=rig.trim_pool,
        list_idle_evictables=rig.list_idle_evictables,
        agree_eviction=rig.agree_eviction,
        kv_bytes_per_token=100_000,
    )
    assert rig.dropped == []
    assert info["fits"] is False
    assert info["residual_deficit_gib"] > 0


def test_the_ladder_is_fail_open_at_every_rung():
    class _ExplodingRig(_LadderRig):
        def trim_pool(self):
            raise RuntimeError("pool probe died")

        def list_idle_evictables(self):
            raise RuntimeError("cache internals drifted")

    rig = _ExplodingRig(wired=[100 * GiB] * 4, limit=128 * GiB, agree=5 * GiB)
    info = run_admission(  # must not raise
        8192,
        read_wired=rig.read_wired,
        read_limit=rig.read_limit,
        trim_pool=rig.trim_pool,
        list_idle_evictables=rig.list_idle_evictables,
        agree_eviction=rig.agree_eviction,
        kv_bytes_per_token=100_000,
    )
    assert info["guarded"]
    assert rig.agree_calls, "the vote must still run after rung-1 failures"


def test_one_bad_drop_never_blocks_the_remaining_drops():
    rig = _LadderRig(
        wired=[100 * GiB] * 4,
        limit=128 * GiB,
        agree=6 * GiB,
    )

    def _boom():
        raise RuntimeError("trie pop failed")

    rig._evictables = []
    evictables = [
        {"label": "bad", "bytes": 4 * GiB, "drop": _boom},
        {
            "label": "good",
            "bytes": 3 * GiB,
            "drop": lambda: rig.dropped.append("good"),
        },
    ]
    rig.list_idle_evictables = lambda: evictables
    info = run_admission(
        8192,
        read_wired=rig.read_wired,
        read_limit=rig.read_limit,
        trim_pool=rig.trim_pool,
        list_idle_evictables=rig.list_idle_evictables,
        agree_eviction=rig.agree_eviction,
        kv_bytes_per_token=100_000,
    )
    assert rig.dropped == ["good"]
    steps = [a["step"] for a in info["actions"]]
    assert "evict_idle_error" in steps and "evict_idle" in steps


def test_a_raising_agreement_hook_falls_through_without_eviction():
    rig = _LadderRig(wired=[100 * GiB] * 4, limit=128 * GiB)

    def _boom(_deficit):
        raise RuntimeError("collective broke")

    info = run_admission(  # must not raise — the rejection vote is next
        8192,
        read_wired=rig.read_wired,
        read_limit=rig.read_limit,
        trim_pool=rig.trim_pool,
        list_idle_evictables=rig.list_idle_evictables,
        agree_eviction=_boom,
        kv_bytes_per_token=100_000,
    )
    assert info["guarded"]
    assert rig.dropped == []


def test_cached_tokens_are_not_charged_against_headroom():
    """A prefix hit is already resident; charging it evicts the entry that
    made the request cheap."""
    rig = _LadderRig(wired=[1 * GiB] * 4, limit=16 * GiB)
    info = run_admission(
        100_000,
        charge_tokens=1_000,
        read_wired=rig.read_wired,
        read_limit=rig.read_limit,
        trim_pool=rig.trim_pool,
        list_idle_evictables=rig.list_idle_evictables,
        agree_eviction=rig.agree_eviction,
        kv_bytes_per_token=100_000,
    )
    assert info["deficit_gib"] <= 0
    assert rig.trims == 0


# --- The wired ladder: agreed eviction, fail-open backstop. ------------------


class _FakeKV:
    def __init__(self, nbytes):
        self.nbytes = nbytes

    def is_trimmable(self):
        return False


def _resident_cache(entries):
    """A real mlx-lm LRUPromptCache holding fake-KV entries of given bytes."""

    from mlx_lm.models.cache import LRUPromptCache

    cache = LRUPromptCache(max_size=10)
    for index, nbytes in enumerate(entries):
        cache.insert_cache("model", [1000 + index] * (index + 3), [_FakeKV(nbytes)])
    return cache


class _LadderMX(_CollectiveMX):
    """Adds pool-trim instrumentation and programmed multi-vote responses."""

    def __init__(self, *, rank, votes, cache_memory=0):
        super().__init__(rank=rank, votes=votes[0] if votes else [])
        self._programmed = list(votes)
        self.cache_memory = cache_memory
        self.syncs = 0
        self.clears = 0
        self.sent = []

    def all_sum(self, value):
        self.sent.append(list(value))
        return _CollectiveValue(self._programmed.pop(0))

    def get_cache_memory(self):
        return self.cache_memory

    def synchronize(self):
        self.syncs += 1

    def clear_cache(self):
        self.clears += 1


def test_small_prompts_never_run_the_ladder_or_its_vote():
    guard = _guard(ceiling=64 * GiB)
    mx = _LadderMX(rank=0, votes=[[0, 0]])
    guard.check_collective(
        2048,
        current_usage_bytes=1 * GiB,
        mx_module=mx,
        prompt_cache=_resident_cache([4 * GiB]),
    )
    assert len(mx.sent) == 1, "only the rejection vote may run"
    assert mx.clears == 0


def test_a_guarded_prompt_votes_the_deficit_before_the_rejection():
    guard = _guard(ceiling=64 * GiB)
    cache = _resident_cache([1 * GiB])
    mx = _LadderMX(rank=0, votes=[[0, 0], [0, 0]])
    guard.check_collective(
        9000,
        current_usage_bytes=1 * GiB,
        mx_module=mx,
        prompt_cache=cache,
    )
    assert len(mx.sent) == 2, "deficit vote first, rejection vote second"
    assert mx.sent[0] == [0, 0]  # no deficit on this rank
    assert len(cache) == 1, "no rank was short, so nothing is dropped"


def test_every_rank_drops_the_same_entries_when_one_rank_is_short():
    """The desync rule: eviction is agreed, never unilateral.

    Rank 1 is 3 GiB short; rank 0 has headroom to spare. Both must drop the
    largest-first covering set against the AGREED target so their resident
    caches stay identical and the next prefill starts at the same offset.
    """

    tokens = 8192  # 2 GiB of KV on this monitor + 8 GiB reserve
    kv_per_token = rank_monitor(_Model()).estimate_prompt_kv_bytes(1024) / 1024
    deficit_b = admission_deficit_bytes(
        tokens, 2 * GiB, 12 * GiB, kv_bytes_per_token=kv_per_token
    )
    assert deficit_b > 0
    deficit_b_mib = -(-int(deficit_b) // MiB)

    caches = [
        _resident_cache([1 * GiB, 4 * GiB]),
        _resident_cache([1 * GiB, 4 * GiB]),
    ]
    agreed_deficits = [0, deficit_b_mib]
    mx_a = _LadderMX(rank=0, votes=[agreed_deficits, [0, 0]])
    mx_b = _LadderMX(rank=1, votes=[agreed_deficits, [0, 0]])

    _guard(rank=0, ceiling=64 * GiB).check_collective(
        tokens,
        current_usage_bytes=1 * GiB,
        mx_module=mx_a,
        prompt_cache=caches[0],
    )
    _guard(rank=1, ceiling=12 * GiB).check_collective(
        tokens,
        current_usage_bytes=2 * GiB,
        mx_module=mx_b,
        prompt_cache=caches[1],
    )

    assert mx_b.sent[0][1] == deficit_b_mib, "rank 1 reports its own deficit"
    for cache in caches:
        # 4 GiB alone covers the ~1 GiB agreed target; the small entry stays.
        assert len(cache) == 1
        assert cache.nbytes == 1 * GiB


def test_the_ladder_turns_a_rejection_into_a_cache_miss(monkeypatch):
    """The wedge scenario: without eviction the guard refuses the prompt."""

    tokens = 8192
    monitor = rank_monitor(_Model())
    peak = monitor.estimate_prefill_peak_bytes(tokens, _DEFAULT_PREFILL_STEP)
    cache = _resident_cache([4 * GiB])
    base_usage = 2 * GiB

    monkeypatch.setattr(
        "omlx.cluster.memory_guard.current_usage_bytes",
        lambda: base_usage + cache.nbytes,
    )

    # A ceiling between the two states: refuse with the entry resident,
    # serve once it is dropped. Derived from the monitor's own estimate so
    # the test states the property and cannot drift with the SDPA model.
    ceiling = int(base_usage + peak + 2 * GiB)
    guard = RankPrefillGuard(monitor, rank=0, node_id="solo", ceiling_bytes=ceiling)

    with pytest.raises(PrefillMemoryExceededError):
        # No ladder below the admission gate: 8191 tokens is one short.
        guard.check_collective(
            tokens - 1,
            current_usage_bytes=base_usage + cache.nbytes,
            mx_module=_LadderMX(rank=0, votes=[]),
        )

    mx = _LadderMX(rank=0, votes=[])
    guard.check_collective(tokens, mx_module=mx, prompt_cache=cache)

    assert len(cache) == 0, "the resident entry paid for the headroom"
    assert mx.clears == 1 and mx.syncs == 1, "rung 1 synced before clearing"


def test_a_broken_ladder_falls_through_to_the_rejection_vote(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("ladder exploded")

    monkeypatch.setattr("omlx.cluster.prefill_guard.run_admission", _boom)
    guard = _guard(ceiling=4 * GiB)
    mx = _LadderMX(rank=0, votes=[[1, 0]])
    with pytest.raises(PrefillMemoryExceededError):
        guard.check_collective(
            9000,
            current_usage_bytes=3 * GiB,
            mx_module=mx,
            prompt_cache=_resident_cache([4 * GiB]),
        )
    assert len(mx.sent) == 1, "the backstop vote is exactly the old behavior"


def test_a_rank_with_no_ceiling_still_joins_the_deficit_vote():
    """An unmeasurable rank votes 0 but must NOT skip the collective."""
    guard = RankPrefillGuard(None, rank=1, ceiling_bytes=0)
    assert not guard.active
    mx = _LadderMX(rank=1, votes=[[2048, 0], [0, 0]])
    guard.check_collective(
        9000,
        current_usage_bytes=1 * GiB,
        mx_module=mx,
        prompt_cache=_resident_cache([3 * GiB]),
    )
    assert mx.sent[0] == [0, 0]
    assert len(mx.sent) == 2
