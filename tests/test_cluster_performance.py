# SPDX-License-Identifier: Apache-2.0
"""Performance-aware planner, launch probe, and runtime capability tests."""

import importlib
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.launch import run_cluster_performance_probe
from omlx.cluster.performance import (
    ExecutionSettings,
    NodePerformanceProfile,
    execution_profile,
    performance_profiles_from_records,
    tune_execution_settings,
)
from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PipelineAssignment,
    plan_unequal_pipeline,
)
from omlx.cluster.runtime_optimizations import (
    LockstepClusterShutdownError,
    LockstepPrefillCancelError,
    get_lockstep_controller,
    install_runtime_optimizations,
    pipeline_prefill_schedule,
    prefill_clear_threshold_bytes,
)

mlx_generate = importlib.import_module("mlx_lm.generate")
mlx_server = importlib.import_module("mlx_lm.server")


def _profile(node_id: str, rank: int, rate: float) -> NodePerformanceProfile:
    return NodePerformanceProfile(
        node_id=node_id,
        rank=rank,
        decode_weight_bytes_per_second=rate,
        prefill_weight_bytes_per_second=rate,
        collective_latency_seconds=0.001,
        collective_bandwidth_bytes_per_second=10_000,
        backend="ring",
        measured_at="2026-07-26T12:00:00+00:00",
        samples=5,
    )


def test_performance_planner_prefers_faster_node_without_exceeding_memory():
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=0,
        layer_weight_bytes=(10,) * 8,
        activation_bytes_per_token=2,
    )
    plan = plan_unequal_pipeline(
        model,
        [
            NodeBudget(
                "slow",
                100,
                rank=0,
                performance=_profile("slow", 0, 10),
            ),
            NodeBudget(
                "fast",
                100,
                rank=1,
                performance=_profile("fast", 1, 40),
            ),
        ],
    )

    slow, fast = plan.assignments
    assert plan.optimization == "performance"
    assert fast.layer_count > slow.layer_count
    assert all(item.headroom_bytes >= 0 for item in plan.assignments)
    assert all(item.predicted_stage_seconds is not None for item in plan.assignments)
    assert plan.to_dict()["strategy"].startswith("performance_aware")


def test_partial_measurements_fall_back_to_original_memory_objective():
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=0,
        layer_weight_bytes=(10,) * 8,
    )
    plan = plan_unequal_pipeline(
        model,
        [
            NodeBudget(
                "first",
                100,
                rank=0,
                performance=_profile("first", 0, 10),
            ),
            NodeBudget("second", 100, rank=1),
        ],
    )

    assert plan.optimization == "memory"
    assert [item.layer_count for item in plan.assignments] == [4, 4]
    assert all(item.predicted_stage_seconds is None for item in plan.assignments)


def test_execution_tuner_reduces_concurrency_and_synchronizes_prompt_cache():
    settings = execution_profile("throughput")
    assignments = [
        SimpleNamespace(headroom_bytes=3 * 1024**3),
        SimpleNamespace(headroom_bytes=20 * 1024**3),
    ]

    tuned = tune_execution_settings(settings, assignments, backend="jaccl")

    assert tuned.decode_concurrency == 2
    assert tuned.prompt_concurrency == 1
    assert tuned.prefill_step_size == 512
    assert tuned.pipeline_microbatch_size == 1
    assert tuned.prompt_cache_size == 1
    assert tuned.prompt_cache_bytes is None
    assert tuned.ring_connections_per_ip == 1
    assert "critical headroom" in tuned.tuning_reason
    assert "synchronized single-prefix cache" in tuned.tuning_reason


def test_prompt_cache_is_synchronized_even_when_auto_tuning_is_disabled():
    settings = replace(
        execution_profile("throughput", auto_tune=False),
        prompt_cache_size=16,
        prompt_cache_bytes=8 * 1024**3,
    )

    tuned = tune_execution_settings(
        settings,
        [
            SimpleNamespace(headroom_bytes=3 * 1024**3),
            SimpleNamespace(headroom_bytes=20 * 1024**3),
        ],
        backend="jaccl",
    )

    assert tuned.decode_concurrency == settings.decode_concurrency
    assert tuned.prompt_cache_size == 1
    assert tuned.prompt_cache_bytes is None
    assert "synchronized single-prefix cache" in tuned.tuning_reason


# --- Plan-agreed multi-slot cache + the durable rank-local KV tier -----------


def test_prompt_cache_size_override_lifts_the_pin_through_the_plan(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_PROMPT_CACHE_SIZE", "6")
    assignments = [
        SimpleNamespace(headroom_bytes=40 * 1024**3),
        SimpleNamespace(headroom_bytes=60 * 1024**3),
    ]

    tuned = tune_execution_settings(
        execution_profile("balanced"), assignments, backend="jaccl"
    )

    assert tuned.prompt_cache_size == 6
    assert tuned.prompt_cache_bytes is None  # byte eviction stays banned
    assert "synchronized 6-slot prefix cache" in tuned.tuning_reason


def test_prompt_cache_size_override_is_bounded_by_the_weakest_stage(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_PROMPT_CACHE_SIZE", "6")
    assignments = [
        SimpleNamespace(headroom_bytes=3 * 1024**3),
        SimpleNamespace(headroom_bytes=60 * 1024**3),
    ]

    tuned = tune_execution_settings(
        execution_profile("balanced"), assignments, backend="jaccl"
    )

    # Critical headroom caps the agreed count at 2: every resident prefix is
    # wired memory on every Mac, and the smallest stage sets the bound.
    assert tuned.prompt_cache_size == 2
    assert "critical headroom" in tuned.tuning_reason

    untuned = tune_execution_settings(
        execution_profile("balanced", auto_tune=False), assignments, backend="jaccl"
    )
    assert untuned.prompt_cache_size == 6


@pytest.mark.parametrize("bad", ["0", "-2", "65", "a lot"])
def test_prompt_cache_size_override_garbage_refuses_the_plan(monkeypatch, bad):
    monkeypatch.setenv("OMLX_CLUSTER_PROMPT_CACHE_SIZE", bad)
    with pytest.raises(ValueError, match="OMLX_CLUSTER_PROMPT_CACHE_SIZE"):
        tune_execution_settings(
            execution_profile("balanced"),
            [SimpleNamespace(headroom_bytes=40 * 1024**3)],
            backend="jaccl",
        )


def test_kv_tier_is_a_plan_field_and_rejects_byte_budget_eviction(monkeypatch):
    monkeypatch.delenv("OMLX_CLUSTER_KV_TIER", raising=False)
    tuned = tune_execution_settings(
        execution_profile("balanced"),
        [SimpleNamespace(headroom_bytes=40 * 1024**3)],
        backend="jaccl",
    )
    assert tuned.kv_tier is False  # default preserves today's behavior

    monkeypatch.setenv("OMLX_CLUSTER_KV_TIER", "1")
    tuned = tune_execution_settings(
        execution_profile("balanced"),
        [SimpleNamespace(headroom_bytes=40 * 1024**3)],
        backend="jaccl",
    )
    assert tuned.kv_tier is True
    assert "rank-local KV tier" in tuned.tuning_reason
    assert ExecutionSettings.from_dict(tuned.to_dict()).kv_tier is True

    # The plan-refusal discipline: byte-budget eviction diverges pipeline
    # ranks, so a plan combining it with the tier is invalid on its face.
    with pytest.raises(ValueError, match="prompt_cache_bytes diverges"):
        replace(tuned, prompt_cache_bytes=8 * 1024**3)


# --- SSD boundary-snapshot cache: plan-level opt-in, default off -------------


def test_prompt_cache_ssd_defaults_off_and_opts_in_through_the_plan(monkeypatch):
    monkeypatch.delenv("OMLX_CLUSTER_PROMPT_CACHE_SSD", raising=False)
    assignments = [SimpleNamespace(headroom_bytes=40 * 1024**3)]

    # The dataclass default is off too: the synchronous per-boundary extract
    # and safetensors write run on the serving thread mid-prefill.
    assert execution_profile("balanced").prompt_cache_ssd is False

    tuned = tune_execution_settings(
        execution_profile("balanced"), assignments, backend="jaccl"
    )
    assert tuned.prompt_cache_ssd is False
    assert "SSD boundary-snapshot" not in tuned.tuning_reason

    # The untuned branch resolves the same env, so a plan cannot drift rank
    # to rank — the boundary restore vote is a collective.
    untuned = tune_execution_settings(
        execution_profile("balanced", auto_tune=False),
        assignments,
        backend="jaccl",
    )
    assert untuned.prompt_cache_ssd is False

    monkeypatch.setenv("OMLX_CLUSTER_PROMPT_CACHE_SSD", "1")
    enabled = tune_execution_settings(
        execution_profile("balanced"), assignments, backend="jaccl"
    )
    assert enabled.prompt_cache_ssd is True
    assert "SSD boundary-snapshot cache" in enabled.tuning_reason
    assert ExecutionSettings.from_dict(enabled.to_dict()).prompt_cache_ssd is True


def test_prompt_cache_ssd_explicit_env_off_beats_the_settings_field(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_PROMPT_CACHE_SSD", "off")
    requested = replace(execution_profile("balanced"), prompt_cache_ssd=True)

    tuned = tune_execution_settings(
        requested,
        [SimpleNamespace(headroom_bytes=40 * 1024**3)],
        backend="jaccl",
    )

    assert tuned.prompt_cache_ssd is False


def test_prompt_cache_ssd_signed_plan_payload_stays_authoritative():
    # A plan dict is deserialized verbatim: an older signed plan that opted in
    # still opts in, a missing key falls back to the (off) profile default.
    assert (
        ExecutionSettings.from_dict(
            {"profile": "balanced", "prompt_cache_ssd": True}
        ).prompt_cache_ssd
        is True
    )
    assert ExecutionSettings.from_dict({"profile": "balanced"}).prompt_cache_ssd is (
        False
    )


def test_performance_profiles_reject_nonfinite_measurements():
    payload = _profile("node", 0, 10).to_dict()
    payload["decode_weight_bytes_per_second"] = float("nan")

    with pytest.raises(ValueError, match="finite positive"):
        NodePerformanceProfile.from_dict(payload)


def _deployment() -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id="probe",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("peer", "peer.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 2, 4, 20, 0, 0, 100),
            PipelineAssignment("peer", 1, 0, 2, 20, 0, 0, 100),
        ),
        plan_hash="a" * 64,
        execution=replace(
            execution_profile("balanced"),
            ring_connections_per_ip=3,
        ),
    )


def test_cluster_performance_probe_uses_ring_connections_and_validates_ranks():
    def runner(argv, *, timeout, env):
        assert timeout == 12.0
        assert argv[argv.index("--connections-per-ip") + 1] == "3"
        assert "omlx.cluster.performance_worker" in argv
        assert env["SSH_ASKPASS_REQUIRE"] == "never"
        records = [
            {
                "type": "performance_result",
                "rank": rank,
                "size": 2,
                "decode_weight_bytes_per_second": 100 + rank,
                "prefill_weight_bytes_per_second": 200 + rank,
                "collective_latency_seconds": 0.001,
                "collective_bandwidth_bytes_per_second": 10_000,
                "samples": 5,
                "measured_at": "2026-07-26T12:00:00+00:00",
            }
            for rank in (0, 1)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="\n".join(json.dumps(record) for record in records),
            stderr="",
        )

    report = run_cluster_performance_probe(
        _deployment(),
        timeout=12.0,
        python_executable="/opt/omlx/bin/python",
        runner=runner,
    )

    assert report["ok"] is True
    assert report["connections_per_ip"] == 3
    profiles = performance_profiles_from_records(
        [
            {"type": "noise"},
            *[
                {"type": "performance_result"} | profile
                for profile in report["profiles"]
            ],
        ],
        node_ids=("local", "peer"),
        backend="ring",
    )
    assert [profile.rank for profile in profiles] == [0, 1]


def test_cluster_performance_probe_never_passes_ring_connections_to_jaccl():
    deployment = replace(
        _deployment(),
        backend="jaccl",
        hosts=(
            ClusterHost(
                "local",
                "127.0.0.1",
                ("10.0.0.1",),
                (None, "rdma_en5"),
            ),
            ClusterHost(
                "peer",
                "peer.local",
                ("10.0.0.2",),
                ("rdma_en5", None),
            ),
        ),
    )

    def runner(argv, *, timeout, env):
        assert "--connections-per-ip" not in argv
        records = [
            {
                "type": "performance_result",
                "rank": rank,
                "size": 2,
                "decode_weight_bytes_per_second": 100 + rank,
                "prefill_weight_bytes_per_second": 200 + rank,
                "collective_latency_seconds": 0.001,
                "collective_bandwidth_bytes_per_second": 10_000,
                "samples": 5,
                "measured_at": "2026-07-26T12:00:00+00:00",
            }
            for rank in (0, 1)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="\n".join(json.dumps(record) for record in records),
            stderr="",
        )

    report = run_cluster_performance_probe(deployment, runner=runner)

    assert report["ok"] is True
    assert report["backend"] == "jaccl"
    assert report["connections_per_ip"] == 1


class _ValidatedPipeline:
    pipeline_rank = 0
    pipeline_size = 2

    def __init__(self):
        self.seen = []

    def __call__(self, value, cache=None):
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        self.seen.append(value.tolist())
        if pipeline_rank != 0:
            value = mx.distributed.send(
                value,
                (pipeline_rank - 1) % pipeline_size,
            )
        if pipeline_size > 1:
            value = mx.distributed.all_gather(value)
        return value


class _Group:
    @staticmethod
    def rank():
        return 0

    @staticmethod
    def size():
        return 2


class _WorkerGroup:
    @staticmethod
    def rank():
        return 1

    @staticmethod
    def size():
        return 2


def test_sampling_rank_optimization_is_capability_gated_and_restored():
    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
    )
    model = SimpleNamespace(model=_ValidatedPipeline())
    original_gather = mx.distributed.all_gather
    original_send = mx.distributed.send
    original_call = _ValidatedPipeline.__call__
    original_step = mlx_generate.GenerationBatch._step
    original_prompt = mlx_generate.PromptProcessingBatch.prompt
    original_batch_next = mlx_generate.BatchGenerator.next
    original_share_object = mlx_server.ResponseGenerator._share_object

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is True
        assert capabilities["rank_zero_logits"]["active"] is False
        assert capabilities["pipeline_prefill_overlap"]["active"] is True, (
            capabilities["pipeline_prefill_overlap"]["reason"]
        )
        # No stop ids were handed over, so the decode EOS swap stays off —
        # but the prefill cancel/removal and the shutdown sentinel validated.
        assert capabilities["lockstep_cancel"]["active"] is True
        assert capabilities["coordinated_shutdown"]["active"] is True
        assert mx.distributed.all_gather is not original_gather
        assert mx.distributed.send is not original_send
        assert _ValidatedPipeline.__call__ is not original_call
        assert mlx_generate.GenerationBatch._step is not original_step
        assert mlx_generate.PromptProcessingBatch.prompt is not original_prompt
        assert mlx_generate.BatchGenerator.next is not original_batch_next
        assert mlx_server.ResponseGenerator._share_object is not original_share_object

    assert mx.distributed.all_gather is original_gather
    assert mx.distributed.send is original_send
    assert _ValidatedPipeline.__call__ is original_call
    assert mlx_generate.GenerationBatch._step is original_step
    assert mlx_generate.PromptProcessingBatch.prompt is original_prompt
    assert mlx_generate.BatchGenerator.next is original_batch_next
    assert mlx_server.ResponseGenerator._share_object is original_share_object


def test_worker_rank_skips_vocab_projection_when_adapter_declares_contract(
    monkeypatch,
):
    class Cache:
        state = mx.array([0])

    class RankLocalLogitsModel:
        _omlx_supports_rank_zero_logits = True
        _omlx_output_vocab_size = 32

        def __init__(self):
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1
            self.calls = []

        def __call__(self, value, cache=None, skip_logits=False):
            self.calls.append(skip_logits)
            value = self.model(value, cache=cache)
            if skip_logits:
                return None
            return mx.zeros((*value.shape, self._omlx_output_vocab_size))

    class Batch:
        def __init__(self, model):
            self.model = model
            self.uids = [1]
            self.prompt_cache = [Cache()]
            self.tokens = [[]]
            self.samplers = [None]
            self.fallback_sampler = lambda value: mx.argmax(value, axis=-1)
            self.logits_processors = [[]]
            self.state_machines = []
            self.max_tokens = [2]
            self._current_tokens = None
            self._current_logprobs = []
            self._next_tokens = mx.array([3], dtype=mx.uint32)
            self._next_logprobs = []
            self._token_context = []
            self._num_tokens = [0]
            self._matcher_states = []

    model = RankLocalLogitsModel()
    batch = Batch(model)
    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
    )
    monkeypatch.setattr(
        mx.distributed,
        "all_sum",
        lambda value, group=None: value,
    )
    monkeypatch.setattr(mx.distributed, "send", lambda value, *_a, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **_k: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    with install_runtime_optimizations(
        model,
        _WorkerGroup(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["rank_zero_logits"]["active"] is True
        mlx_generate.GenerationBatch._step(batch)

    assert model.calls == [True]
    assert len(batch._next_logprobs) == 1
    assert batch._next_logprobs[0].shape == (32,)


def test_pipeline_prefill_schedule_has_equal_fill_and_drain_timeline():
    schedules = [
        pipeline_prefill_schedule(10, 4, rank=rank, world_size=3)
        for rank in range(3)
    ]

    assert {len(schedule) for schedule in schedules} == {5}
    # MLX-LM runs the first stage on the highest rank and the final stage on
    # rank zero, so the Exo fill/drain offset is mirrored.
    assert [(slot.start, slot.end) for slot in schedules[0]] == [
        (None, None),
        (None, None),
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    assert [(slot.start, slot.end) for slot in schedules[2]] == [
        (0, 4),
        (4, 8),
        (8, 10),
        (None, None),
        (None, None),
    ]
    assert all(sum(slot.is_real for slot in schedule) == 3 for schedule in schedules)


def test_staggered_prompt_queues_and_flushes_every_real_chunk(monkeypatch):
    sends = []
    gathers = []
    async_values = []
    original_prompt = mlx_generate.PromptProcessingBatch.prompt

    monkeypatch.setattr(
        mx.distributed,
        "send",
        lambda value, destination, **kwargs: sends.append(destination) or value,
    )
    monkeypatch.setattr(
        mx.distributed,
        "all_gather",
        lambda value, **kwargs: gathers.append(value) or value,
    )
    # The lockstep prefill cancel rides one int32 all-sum per chunk boundary;
    # nothing is armed here, so every contribution reads back as zero.
    monkeypatch.setattr(
        mx.distributed,
        "all_sum",
        lambda value, group=None: value,
    )
    monkeypatch.setattr(mx, "async_eval", lambda *values: async_values.extend(values))

    class Cache:
        state = mx.array([0])

    class Batch:
        uids = ["request"]
        tokens = [[]]
        prompt_cache = [Cache()]
        prefill_step_size = 8

        def __init__(self):
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1

    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
        async_overlap=True,
        prefill_step_size=8,
    )
    model = SimpleNamespace(model=_ValidatedPipeline())
    batch = Batch()

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["pipeline_prefill_overlap"]["active"] is True, (
            capabilities["pipeline_prefill_overlap"]["reason"]
        )
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(9))])

    # The scheduler honours the same eight-token step the memory guard approved,
    # so 9 tokens make two real chunks. Each chunk reaches send; the final
    # hidden-state gather is skipped. Each chunk also launches one scalar
    # cancel all-sum (0-dim), deferred-read, alongside its queued send flush.
    assert sends == [0, 0]
    assert len([value for value in async_values if value.ndim > 0]) == 2
    assert len([value for value in async_values if value.ndim == 0]) == 2
    assert gathers == []
    assert batch.tokens == [list(range(9))]
    assert mlx_generate.PromptProcessingBatch.prompt is original_prompt


def test_staggered_prompt_matches_stock_chunking_padding_and_cache_lifecycle(
    monkeypatch,
):
    """The faster scheduler must preserve MLX-LM's prompt/cache contract."""

    original_prompt = mlx_generate.PromptProcessingBatch.prompt
    monkeypatch.setattr(mx.distributed, "send", lambda value, *_a, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_sum", lambda value, group=None: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    class Cache:
        def __init__(self):
            self.state = mx.array([0])
            self.events = []

        def prepare(self, *, lengths, right_padding):
            self.events.append(("prepare", tuple(lengths), tuple(right_padding)))

        def finalize(self):
            self.events.append(("finalize",))

    class Batch:
        uids = ["first", "second"]
        prefill_step_size = 8

        def __init__(self):
            self.tokens = [[], []]
            self.prompt_cache = [Cache()]
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1

    prompts = [list(range(9)), list(range(20, 25))]
    stock = Batch()
    original_prompt(stock, [list(prompt) for prompt in prompts])

    patched = Batch()
    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
        async_overlap=True,
        prefill_step_size=8,
    )
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        settings,
        batchable=True,
    ):
        mlx_generate.PromptProcessingBatch.prompt(
            patched,
            [list(prompt) for prompt in prompts],
        )

    assert patched.model.seen == stock.model.seen
    assert [len(chunk[0]) for chunk in patched.model.seen] == [8, 1]
    assert patched.tokens == stock.tokens == prompts
    assert patched.prompt_cache[0].events == stock.prompt_cache[0].events


def test_staggered_prompt_skips_discarded_logits_when_adapter_allows(monkeypatch):
    """The prompt loop never samples from its return value, so a model
    declaring the rank-zero logits contract skips the vocabulary projection
    on every prefill chunk."""

    monkeypatch.setattr(mx.distributed, "send", lambda value, *_a, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_sum", lambda value, group=None: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    class Cache:
        state = mx.array([0])

    class SkipLogitsModel:
        _omlx_supports_rank_zero_logits = True
        _omlx_output_vocab_size = 32

        def __init__(self):
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1
            self.calls = []

        def __call__(self, value, cache=None, skip_logits=False):
            self.calls.append(skip_logits)
            return self.model(value, cache=cache)

    class Batch:
        uids = ["request"]
        tokens = [[]]
        prompt_cache = [Cache()]
        prefill_step_size = 8

        def __init__(self, model):
            self.model = model

    model = SkipLogitsModel()
    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
        async_overlap=True,
        prefill_step_size=8,
    )

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["prefill_skip_logits"]["active"] is True
        mlx_generate.PromptProcessingBatch.prompt(
            Batch(model),
            [list(range(9))],
        )

    # Nine tokens at an eight-token step: two chunks, both skip the head.
    assert model.calls == [True, True]


def test_staggered_prompt_keeps_the_head_without_adapter_contract(monkeypatch):
    """No rank-zero logits contract, no skip_logits kwarg: an adapter whose
    forward rejects the keyword must run exactly as before."""

    monkeypatch.setattr(mx.distributed, "send", lambda value, *_a, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_sum", lambda value, group=None: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    class Cache:
        state = mx.array([0])

    class Batch:
        uids = ["request"]
        tokens = [[]]
        prompt_cache = [Cache()]
        prefill_step_size = 8

        def __init__(self):
            # _ValidatedPipeline.__call__ takes no skip_logits kwarg, so a
            # wrongful pass would raise TypeError here.
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1

    batch = Batch()
    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
        async_overlap=True,
        prefill_step_size=8,
    )

    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["prefill_skip_logits"]["active"] is False
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(9))])

    assert [len(chunk[0]) for chunk in batch.model.seen] == [8, 1]


def _run_staggered_prompt(monkeypatch, cache_memory_bytes, memory_limit_bytes=0):
    """Run the patched staggered prompt with a scripted allocator pool size."""

    clears = []
    monkeypatch.setattr(mx, "get_cache_memory", lambda: cache_memory_bytes)
    monkeypatch.setattr(mx, "clear_cache", lambda: clears.append(1))
    monkeypatch.setattr(mx.distributed, "send", lambda value, *_a, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **_k: value)
    # The lockstep prefill-cancel all_sum needs a real Group; stub it like the
    # other prefill tests (contribution is 0, so no cancel fires).
    monkeypatch.setattr(mx.distributed, "all_sum", lambda value, group=None: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    class Cache:
        state = mx.array([0])

        def prepare(self, *, lengths, right_padding):
            pass

        def finalize(self):
            pass

    class Batch:
        uids = ["request"]
        tokens = [[]]
        prompt_cache = [Cache()]
        prefill_step_size = 8

        def __init__(self):
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1

    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
        async_overlap=True,
        prefill_step_size=8,
    )
    batch = Batch()
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        settings,
        batchable=True,
        memory_limit_bytes=memory_limit_bytes,
    ):
        mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(9))])
    return clears


def test_prefill_clear_threshold_mirrors_the_scheduler_formula():
    """memory_limit/3 with a 2 GiB floor — keep scheduler.py in sync."""
    assert prefill_clear_threshold_bytes(0) == 2 * 1024**3
    assert prefill_clear_threshold_bytes(-5) == 2 * 1024**3
    assert prefill_clear_threshold_bytes(3 * 1024**3) == 2 * 1024**3
    assert prefill_clear_threshold_bytes(120 * 1024**3) == 40 * 1024**3


def test_staggered_prompt_never_clears_a_small_pool_per_chunk(monkeypatch):
    """The sawtooth fix: a 9-token, 2-chunk prefill with a 1 GiB pool must
    not return the pool to the OS after every chunk."""
    clears = _run_staggered_prompt(monkeypatch, cache_memory_bytes=1024**3)
    assert clears == []


def test_staggered_prompt_keeps_the_pressure_safety_clear(monkeypatch):
    """Above the threshold the clear still fires — once per real chunk."""
    clears = _run_staggered_prompt(monkeypatch, cache_memory_bytes=3 * 1024**3)
    assert clears == [1, 1]


def test_staggered_prompt_threshold_tracks_the_memory_limit(monkeypatch):
    """A 3 GiB pool is pressure on a 6 GiB host (limit/3) but not on a 90 GiB one."""
    assert (
        _run_staggered_prompt(monkeypatch, 3 * 1024**3, memory_limit_bytes=90 * 1024**3)
        == []
    )
    assert _run_staggered_prompt(
        monkeypatch, 3 * 1024**3, memory_limit_bytes=6 * 1024**3
    ) == [1, 1]


def test_staggered_prompt_gates_the_padded_finalize_clear(monkeypatch):
    """The finalize path after right-padded batches is gated the same way."""
    clears = []
    monkeypatch.setattr(mx, "get_cache_memory", lambda: 3 * 1024**3)
    monkeypatch.setattr(mx, "clear_cache", lambda: clears.append(1))
    monkeypatch.setattr(mx.distributed, "send", lambda value, *_a, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **_k: value)
    # The lockstep prefill-cancel all_sum needs a real Group; stub it like the
    # other prefill tests (contribution is 0, so no cancel fires).
    monkeypatch.setattr(mx.distributed, "all_sum", lambda value, group=None: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    class Cache:
        state = mx.array([0])

        def prepare(self, *, lengths, right_padding):
            pass

        def finalize(self):
            pass

    class Batch:
        uids = ["first", "second"]
        prefill_step_size = 8

        def __init__(self):
            self.tokens = [[], []]
            self.prompt_cache = [Cache()]
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1

    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
        async_overlap=True,
        prefill_step_size=8,
    )
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        settings,
        batchable=True,
    ):
        mlx_generate.PromptProcessingBatch.prompt(
            Batch(),
            [list(range(9)), list(range(20, 25))],
        )
    assert clears == [1, 1, 1], "two chunks plus one gated finalize clear"


def test_sampling_rank_optimization_keeps_normal_path_for_unvalidated_model():
    settings = replace(
        execution_profile("interactive"),
        sampling_rank_only=True,
    )
    model = SimpleNamespace(model=SimpleNamespace())
    original_gather = mx.distributed.all_gather

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=True,
    ) as capabilities:
        assert capabilities["sampling_rank_only"]["active"] is False
        assert capabilities["pipeline_prefill_overlap"]["active"] is False
        assert mx.distributed.all_gather is original_gather


def test_non_batchable_model_never_reports_continuous_batching_active():
    settings = execution_profile("balanced")
    model = SimpleNamespace(model=SimpleNamespace())

    with install_runtime_optimizations(
        model,
        _Group(),
        settings,
        batchable=False,
    ) as capabilities:
        batching = capabilities["coalesced_batching"]
        assert batching["enabled"] is True
        assert batching["active"] is False
        assert "sequentially" in batching["reason"]


# ---------------------------------------------------------------------------
# Lockstep in-flight cancel + coordinated shutdown sentinel
#
# The step-boundary rule from the batch-cancel design: on a pipeline group,
# generation may only end at a boundary every rank agreed on. Decode cancels
# by rank zero swapping a stop id in ahead of the token all-sum; prefill
# reads one async int32 all-sum per chunk boundary; the shutdown sentinel
# rides the idle request-share channel. Each path falls back to stock
# behavior when its capability check fails.
# ---------------------------------------------------------------------------


class _CancelCache:
    state = mx.array([0])


class _LogitsAdapter:
    """Batch-facing model wrapper returning fixed logits for the pinned step."""

    def __init__(self, vocab=64):
        self.model = _ValidatedPipeline()
        self._vocab = vocab

    def __call__(self, value, cache=None):
        return mx.zeros((1, 1, self._vocab))


class _CoordinatorBatch:
    """The state consumed by the pinned GenerationBatch._step."""

    def __init__(self, model):
        self.model = model
        self.uids = [1]
        self.prompt_cache = [_CancelCache()]
        self.tokens = [[]]
        self.samplers = [None]
        self.fallback_sampler = lambda value: mx.argmax(value, axis=-1)
        self.logits_processors = [[]]
        self._current_tokens = None
        self._current_logprobs = []
        self._next_tokens = mx.array([3], dtype=mx.uint32)
        self._next_logprobs = []
        self._token_context = []


def test_lockstep_cancel_swaps_a_stop_id_before_the_token_all_sum(monkeypatch):
    calls = []
    original_depends = mx.depends

    def recording_depends(*args, **kwargs):
        calls.append("depends")
        return original_depends(*args, **kwargs)

    def fake_all_sum(value, group=None):
        calls.append("all_sum")
        mx.eval(value)
        calls.append(int(value.reshape(-1)[0].item()))
        return value

    monkeypatch.setattr(mx, "depends", recording_depends)
    monkeypatch.setattr(mx.distributed, "all_sum", fake_all_sum)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    settings = replace(execution_profile("balanced"), sampling_rank_only=True)
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        settings,
        batchable=True,
        stop_token_ids=(99,),
    ) as capabilities:
        assert capabilities["lockstep_cancel"]["active"] is True
        controller = get_lockstep_controller()
        controller.arm_cancel()
        batch = _CoordinatorBatch(_LogitsAdapter())
        mlx_generate.GenerationBatch._step(batch)

        # The mx.depends fence ties the swap to the sampled-token graph and
        # must land BEFORE the cross-rank token all-sum consumes it — that
        # order is what keeps the peer's pipeline send fed.
        assert calls.index("depends") < calls.index("all_sum")
        assert calls[-1] == 99
        mx.eval(batch._next_tokens)
        assert int(batch._next_tokens[0].item()) == 99
        # One swap per arm: the latch is consumed, the next request lives.
        assert controller.decode_swap_token() is None


def test_lockstep_cancel_breaks_both_ranks_at_the_identical_step(monkeypatch):
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)
    settings = replace(execution_profile("balanced"), sampling_rank_only=True)
    steps = 5
    arm_at = 2

    rank_zero_wire = []

    def record_all_sum(value, group=None):
        mx.eval(value)
        rank_zero_wire.append(value)
        return value

    monkeypatch.setattr(mx.distributed, "all_sum", record_all_sum)
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        settings,
        batchable=True,
        stop_token_ids=(99,),
    ):
        controller = get_lockstep_controller()
        batch = _CoordinatorBatch(_LogitsAdapter())
        rank_zero_stream = []
        for step in range(steps):
            if step == arm_at:
                controller.arm_cancel()
            tokens, _ = mlx_generate.GenerationBatch._step(batch)
            rank_zero_stream.extend(tokens)

    # The worker never sees the latch: its token all-sum replays rank zero's
    # exact contributions, which is what the real ring delivers to it.
    replay = iter(rank_zero_wire)
    monkeypatch.setattr(
        mx.distributed,
        "all_sum",
        lambda value, group=None: next(replay),
    )
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _WorkerGroup(),
        settings,
        batchable=True,
        stop_token_ids=(99,),
    ):
        batch = _CoordinatorBatch(_LogitsAdapter())
        worker_stream = []
        for _ in range(steps):
            tokens, _ = mlx_generate.GenerationBatch._step(batch)
            worker_stream.extend(tokens)

    assert rank_zero_stream == worker_stream
    # EOS enters the consumed stream one step after the armed swap, and both
    # ranks meet it at the same index — the identical step boundary.
    assert 99 in rank_zero_stream
    assert rank_zero_stream.index(99) == arm_at + 1


def test_prefill_cancel_breaks_at_a_chunk_boundary(monkeypatch):
    sends = []
    async_values = []
    monkeypatch.setattr(
        mx.distributed,
        "send",
        lambda value, destination, **_k: sends.append(destination) or value,
    )
    monkeypatch.setattr(mx.distributed, "all_gather", lambda value, **_k: value)
    monkeypatch.setattr(mx.distributed, "all_sum", lambda value, group=None: value)
    monkeypatch.setattr(mx, "async_eval", lambda *values: async_values.extend(values))

    class Cache:
        state = mx.array([0])

    class Batch:
        uids = ["request"]
        tokens = [[]]
        prompt_cache = [Cache()]
        prefill_step_size = 8

        def __init__(self):
            self.model = _ValidatedPipeline()
            self.model.pipeline_rank = 1

    settings = replace(
        execution_profile("balanced"),
        sampling_rank_only=True,
        async_overlap=True,
        prefill_step_size=8,
    )
    batch = Batch()
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        settings,
        batchable=True,
        stop_token_ids=(99,),
    ) as capabilities:
        assert capabilities["lockstep_cancel"]["active"] is True
        get_lockstep_controller().arm_cancel()
        with pytest.raises(LockstepPrefillCancelError):
            mlx_generate.PromptProcessingBatch.prompt(batch, [list(range(9))])

    # Nine tokens at an eight-token step is two chunks; the collective fired
    # at the first boundary, so only the first chunk ever computed.
    assert [len(chunk[0]) for chunk in batch.model.seen] == [8]
    assert sends == [0]
    # Exactly one async cancel launch: the break precedes the second chunk's.
    assert len([value for value in async_values if value.ndim == 0]) == 1


def test_cancelled_prefill_uids_ride_the_next_removal_broadcast(monkeypatch):
    def raising_next(self):
        # Keeps the validated source contract: mx.stream ... self._next(
        raise LockstepPrefillCancelError("fired")

    monkeypatch.setattr(mlx_generate.BatchGenerator, "next", raising_next)

    class _PromptBatchView:
        uids = [11]

    class _GenerationBatchView:
        uids = []

    class _Batch:
        _prompt_batch = _PromptBatchView()
        _generation_batch = _GenerationBatchView()
        _unprocessed_sequences = [(13, None, None, None, None, None, None, None)]

    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        replace(execution_profile("balanced"), sampling_rank_only=True),
        batchable=True,
        stop_token_ids=(99,),
    ):
        controller = get_lockstep_controller()
        controller.arm_cancel()
        assert mlx_generate.BatchGenerator.next(_Batch()) == ([], [])
        # The cancelled prefill and queued uids are handed to rank zero's next
        # uid-removal share. The empty generation batch consumed the decode
        # latch, so no EOS swap can leak into the next request's first step.
        assert controller.take_pending_removals() == [11, 13]
        assert controller.decode_swap_token() is None


def test_share_channel_injects_removals_and_broadcasts_the_sentinel(monkeypatch):
    shared = []

    def fake_share(self, obj):
        # Keeps the validated source contract: pickle.dumps pickle.loads all_sum
        shared.append(obj)
        return obj

    monkeypatch.setattr(mlx_server.ResponseGenerator, "_share_object", fake_share)
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        replace(execution_profile("balanced"), sampling_rank_only=True),
        batchable=True,
        stop_token_ids=(99,),
    ) as capabilities:
        assert capabilities["coordinated_shutdown"]["active"] is True
        controller = get_lockstep_controller()
        pinned = mlx_server.ResponseGenerator._share_object

        controller.note_prefill_cancel_fired([7, 9])
        assert pinned(object(), [1]) == [1, 7, 9]
        assert shared[-1] == [1, 7, 9]

        # The idle "no request" share becomes the shutdown sentinel.
        controller.request_shutdown()
        with pytest.raises(LockstepClusterShutdownError):
            pinned(object(), None)
        assert shared[-1] == {"omlx_cluster_shutdown": True}
        assert controller.wait_sentinel_broadcast(0.1) is True


def test_share_channel_raises_on_the_sentinel_for_worker_ranks(monkeypatch):
    def fake_share(self, obj):
        # Keeps the validated source contract: pickle.dumps pickle.loads all_sum
        return obj

    monkeypatch.setattr(mlx_server.ResponseGenerator, "_share_object", fake_share)
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _WorkerGroup(),
        replace(execution_profile("balanced"), sampling_rank_only=True),
        batchable=True,
        stop_token_ids=(99,),
    ):
        pinned = mlx_server.ResponseGenerator._share_object
        # An ordinary idle share passes through untouched.
        assert pinned(object(), None) is None
        # The worker learns about the shutdown only from rank zero's payload.
        with pytest.raises(LockstepClusterShutdownError):
            pinned(object(), {"omlx_cluster_shutdown": True})


def test_lockstep_cancel_falls_back_without_stop_token_ids(monkeypatch):
    calls = []
    original_depends = mx.depends
    monkeypatch.setattr(
        mx,
        "depends",
        lambda *a, **k: calls.append("depends") or original_depends(*a, **k),
    )
    monkeypatch.setattr(mx.distributed, "all_sum", lambda value, group=None: value)
    monkeypatch.setattr(mx, "async_eval", lambda *_values: None)

    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        replace(execution_profile("balanced"), sampling_rank_only=True),
        batchable=True,
    ):
        controller = get_lockstep_controller()
        assert controller.decode_cancel_active is False
        # The prefill half still validated; only the EOS swap is off.
        assert controller.prefill_cancel_active is True
        controller.arm_cancel()
        batch = _CoordinatorBatch(_LogitsAdapter())
        mlx_generate.GenerationBatch._step(batch)
        assert "depends" not in calls
        assert int(batch._next_tokens[0].item()) != 99


def test_lockstep_cancel_kill_switch_leaves_everything_stock(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_LOCKSTEP_CANCEL", "0")
    original_next = mlx_generate.BatchGenerator.next
    original_share = mlx_server.ResponseGenerator._share_object
    original_step = mlx_generate.GenerationBatch._step
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        replace(execution_profile("balanced"), sampling_rank_only=True),
        batchable=True,
        stop_token_ids=(99,),
    ) as capabilities:
        assert capabilities["lockstep_cancel"]["active"] is False
        assert capabilities["coordinated_shutdown"]["active"] is False
        assert mlx_generate.BatchGenerator.next is original_next
        assert mlx_server.ResponseGenerator._share_object is original_share
        # The kill switch is scoped: the sampling pins are unaffected.
        assert mlx_generate.GenerationBatch._step is not original_step


def test_coordinated_shutdown_falls_back_when_the_share_channel_drifts(monkeypatch):
    def bare_share(self, obj):
        return obj

    monkeypatch.setattr(mlx_server.ResponseGenerator, "_share_object", bare_share)
    with install_runtime_optimizations(
        SimpleNamespace(model=_ValidatedPipeline()),
        _Group(),
        replace(execution_profile("balanced"), sampling_rank_only=True),
        batchable=True,
        stop_token_ids=(99,),
    ) as capabilities:
        assert capabilities["coordinated_shutdown"]["active"] is False
        assert "share channel" in capabilities["coordinated_shutdown"]["reason"]
        # The pin was never installed.
        assert mlx_server.ResponseGenerator._share_object is bare_share
        controller = get_lockstep_controller()
        assert controller.share_channel_active is False
        # The decode EOS swap still validated; the prefill cancel refuses to
        # install without the uid-removal cleanup channel.
        assert controller.decode_cancel_active is True
        assert controller.prefill_cancel_active is False
