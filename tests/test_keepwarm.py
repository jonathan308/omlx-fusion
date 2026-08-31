# SPDX-License-Identifier: Apache-2.0
"""Safety, lifecycle, and RDMA gates for latent Metal keepwarm."""

from __future__ import annotations

import threading
from contextlib import contextmanager, suppress
from types import SimpleNamespace

from omlx.engine_core import EngineCore
from omlx.keepwarm import (
    CompiledMetalKeepwarmTouch,
    KeepwarmAction,
    KeepwarmConfig,
    KeepwarmController,
    distributed_dataplane_ping,
)


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def config(**overrides) -> KeepwarmConfig:
    values = {
        "enabled": True,
        "interval_seconds": 10.0,
        "idle_after_seconds": 2.0,
        "matrix_size": 1,
        "repeats": 1,
        "request_start_enabled": True,
        "request_start_idle_seconds": 2.0,
        "request_start_matrix_size": 128,
        "post_response_enabled": True,
        "post_response_delay_seconds": 5.0,
        "post_response_matrix_size": 128,
        "large_cache_tokens": 8192,
        "large_cache_interval_seconds": 60.0,
        "slow_threshold_seconds": 1.0,
        "slow_backoff_seconds": 60.0,
        "dataplane_ping": True,
    }
    values.update(overrides)
    return KeepwarmConfig(**values)


def test_keepwarm_is_default_off(monkeypatch):
    monkeypatch.delenv("OMLX_KEEPWARM", raising=False)
    assert KeepwarmConfig.from_env().enabled is False


def test_idle_touch_does_not_arm_before_a_real_request_or_cache():
    clock = Clock()
    controller = KeepwarmController(config(), clock=clock)
    clock.value = 300.0
    assert controller.idle_action(cache_tokens=0) is None
    assert controller.snapshot()["cache_armed"] is False


def test_completed_request_arms_request_start_and_post_response_actions():
    clock = Clock()
    controller = KeepwarmController(config(), clock=clock)
    controller.observe_request_state(True)
    controller.observe_request_state(False, cache_tokens=4096)

    clock.value = 3.0
    request_start = controller.request_start_action()
    assert request_start is not None
    assert request_start.kind == "request_start"
    assert request_start.matrix_size == 128

    clock.value = 4.0
    controller.observe_request_state(False, cache_tokens=4096)
    clock.value = 9.0
    post_response = controller.idle_action(cache_tokens=4096)
    assert post_response is not None
    assert post_response.kind == "post_response"


def test_large_cache_stretches_periodic_interval():
    clock = Clock()
    controller = KeepwarmController(
        config(post_response_enabled=False),
        clock=clock,
    )
    controller.observe_request_state(True)
    controller.observe_request_state(False, cache_tokens=10_000)
    clock.value = 2.0
    first = controller.idle_action(cache_tokens=10_000)
    assert first is not None and first.kind == "idle"
    controller.record(first, elapsed_seconds=0.001, ok=True)

    clock.value = 20.0
    assert controller.idle_action(cache_tokens=10_000) is None
    clock.value = 62.0
    assert controller.idle_action(cache_tokens=10_000) is not None


def test_slow_touch_enters_backoff_and_bounds_failure():
    clock = Clock()
    controller = KeepwarmController(
        config(post_response_enabled=False),
        clock=clock,
    )
    controller.observe_request_state(True)
    controller.observe_request_state(False)
    clock.value = 2.0
    action = controller.idle_action()
    assert action is not None
    controller.record(action, elapsed_seconds=1.5, ok=False, error="x" * 1000)
    clock.value = 30.0
    assert controller.idle_action() is None
    snapshot = controller.snapshot()
    assert snapshot["failures"] == 1
    assert snapshot["slow_count"] == 1
    assert len(snapshot["last_event"]["error"]) == 500


def test_quick_failed_touch_also_enters_backoff():
    clock = Clock()
    controller = KeepwarmController(
        config(post_response_enabled=False),
        clock=clock,
    )
    controller.observe_request_state(True)
    controller.observe_request_state(False)
    clock.value = 2.0
    action = controller.idle_action()
    assert action is not None
    controller.record(action, elapsed_seconds=0.001, ok=False, error="failed")
    clock.value = 30.0
    assert controller.idle_action() is None
    clock.value = 62.0
    assert controller.idle_action() is not None


def test_live_toggle_preserves_cache_arming_and_disable_stops_actions():
    clock = Clock()
    controller = KeepwarmController(config(enabled=False), clock=clock)
    controller.observe_request_state(True)
    controller.observe_request_state(False, cache_tokens=2048)
    clock.value = 3.0
    assert controller.idle_action() is None

    controller.configure(True)
    queued = controller.idle_action()
    assert queued is not None
    assert controller.should_execute(queued) is True
    controller.configure(False)
    assert controller.should_execute(queued) is False
    clock.value = 100.0
    assert controller.idle_action() is None


def test_clear_and_shutdown_disarm_without_retaining_cache_state():
    clock = Clock()
    controller = KeepwarmController(config(), clock=clock)
    controller.observe_request_state(True)
    controller.observe_request_state(False, cache_tokens=32_000)
    controller.disarm_cache()
    clock.value = 100.0
    # Stale cache accounting must not undo an explicit clear.
    assert controller.idle_action(cache_tokens=32_000) is None
    assert controller.snapshot()["clear_inhibited"] is True

    controller.observe_request_state(True)
    controller.observe_request_state(False, cache_tokens=32_000)
    clock.value = 102.0
    assert controller.idle_action(cache_tokens=32_000) is not None
    assert controller.snapshot()["clear_inhibited"] is False

    controller.shutdown()
    controller.configure(True)
    assert controller.request_start_action() is None
    assert controller.snapshot()["closed"] is True


class Scheduler:
    def __init__(
        self,
        *,
        busy: bool = False,
        cache_tokens: int = 0,
        exact_resident_tokens: int = 0,
    ) -> None:
        self.busy = busy
        self.added = []
        self._exact_resident_tokens = exact_resident_tokens
        self.block_aware_cache = SimpleNamespace(
            paged_cache=SimpleNamespace(
                stats=SimpleNamespace(total_tokens_cached=cache_tokens)
            )
        )

    def _exact_resident_stats(self):
        return {"max_token_count": self._exact_resident_tokens}

    def has_requests(self) -> bool:
        return self.busy

    def add_request(self, request) -> None:
        self.added.append(request)


def core_with_scheduler(scheduler: Scheduler) -> EngineCore:
    core = EngineCore.__new__(EngineCore)
    core.scheduler = scheduler
    core._keepwarm = KeepwarmController(config())
    core.config = SimpleNamespace(keepwarm_config=core._keepwarm.config)
    core._pending_admissions_lock = threading.Lock()
    core._pending_admissions = 0
    core._wake_engine_loop = lambda: None
    return core


def test_resident_cache_tokens_include_exact_resident_l0():
    core = core_with_scheduler(
        Scheduler(cache_tokens=4096, exact_resident_tokens=220_000)
    )

    assert core._resident_cache_tokens() == 220_000


def test_concurrent_admission_skips_keepwarm_and_still_adds_request():
    scheduler = Scheduler()
    core = core_with_scheduler(scheduler)
    core._pending_admissions = 2
    core._run_keepwarm_action = lambda _action: (_ for _ in ()).throw(
        AssertionError("keepwarm must skip concurrent admission")
    )

    request = object()
    core._admit_request(request)
    assert scheduler.added == [request]
    snapshot = core._keepwarm.snapshot()
    assert snapshot["skips"] == 1
    assert snapshot["request_active"] is True


def test_failed_exclusive_admission_does_not_arm_or_leave_request_active():
    scheduler = Scheduler()
    core = core_with_scheduler(scheduler)
    core._pending_admissions = 1

    def reject(_request):
        raise ValueError("rejected")

    scheduler.add_request = reject
    try:
        core._admit_request(object())
    except ValueError:
        pass
    else:
        raise AssertionError("admission must propagate scheduler failure")

    snapshot = core._keepwarm.snapshot()
    assert snapshot["request_active"] is False
    assert snapshot["cache_armed"] is False


def test_all_rejected_concurrent_admissions_never_arm_keepwarm():
    scheduler = Scheduler()
    core = core_with_scheduler(scheduler)
    core._pending_admissions = 2

    def reject(_request):
        raise ValueError("rejected")

    scheduler.add_request = reject
    for _ in range(2):
        with suppress(ValueError):
            core._admit_request(object())

    snapshot = core._keepwarm.snapshot()
    assert snapshot["request_active"] is False
    assert snapshot["cache_armed"] is False


def test_second_admission_arriving_before_request_start_touch_wins():
    scheduler = Scheduler()
    core = core_with_scheduler(scheduler)
    core._keepwarm = KeepwarmController(
        config(request_start_idle_seconds=0.0),
    )
    core.config.keepwarm_config = core._keepwarm.config
    core._keepwarm.observe_request_state(True)
    core._keepwarm.observe_request_state(False, cache_tokens=4096)
    core._pending_admissions = 1
    original = core._keepwarm.request_start_action

    def race_second_admission():
        action = original()
        core._pending_admissions = 2
        return action

    core._keepwarm.request_start_action = race_second_admission
    core._compiled_metal_keepwarm = SimpleNamespace(
        touch=lambda _action: (_ for _ in ()).throw(
            AssertionError("touch must skip after a second admission arrives")
        )
    )

    request = object()
    core._admit_request(request)
    assert scheduler.added == [request]
    assert core._keepwarm.snapshot()["skips"] == 1


def test_idle_lane_rechecks_pending_admission_before_touching():
    scheduler = Scheduler(cache_tokens=16_384)
    core = core_with_scheduler(scheduler)
    core._keepwarm.observe_request_state(True)
    core._keepwarm.observe_request_state(False, cache_tokens=16_384)
    core._pending_admissions = 1
    core._run_keepwarm_action = lambda _action: (_ for _ in ()).throw(
        AssertionError("keepwarm must skip queued admission")
    )

    core._idle_keepwarm_if_due()
    assert core._keepwarm.snapshot()["request_active"] is True


def test_loaded_engine_live_reconfigure_updates_config_and_controller():
    core = core_with_scheduler(Scheduler())
    core.configure_keepwarm(False)
    assert core.config.keepwarm_config.enabled is False
    assert core._keepwarm.snapshot()["enabled"] is False
    core.configure_keepwarm(True)
    assert core.config.keepwarm_config.enabled is True
    assert core._keepwarm.snapshot()["enabled"] is True


class Value:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Distributed:
    def __init__(self, receives):
        self.receives = list(receives)
        self.calls = []

    def send(self, value, target, *, group):
        self.calls.append(("send", target, group))
        return value

    def recv_like(self, value, source, *, group):
        self.calls.append(("recv", source, group))
        return Value(self.receives.pop(0))


class FakeMX:
    uint32 = "uint32"

    def __init__(self, receives):
        self.distributed = Distributed(receives)
        self.evaluated = []

    @staticmethod
    def array(values, dtype):
        del dtype
        return Value(values[0])

    def eval(self, value):
        self.evaluated.append(value)


def test_rank_zero_dataplane_ping_visits_every_worker_in_complementary_order():
    mx = FakeMX([1, 2])
    group = SimpleNamespace(name="jaccl")
    distributed_dataplane_ping(mx, group, rank=0, world_size=3)
    assert mx.distributed.calls == [
        ("send", 1, group),
        ("recv", 1, group),
        ("send", 2, group),
        ("recv", 2, group),
    ]


def test_worker_dataplane_ping_receives_before_acknowledging():
    mx = FakeMX([0])
    group = SimpleNamespace(name="jaccl")
    distributed_dataplane_ping(mx, group, rank=2, world_size=3)
    assert mx.distributed.calls == [
        ("recv", 0, group),
        ("send", 0, group),
    ]


def test_action_shape_is_bounded_by_configuration_parser(monkeypatch):
    monkeypatch.setenv("OMLX_KEEPWARM", "1")
    monkeypatch.setenv("OMLX_KEEPWARM_MATRIX_SIZE", "99999")
    monkeypatch.setenv("OMLX_KEEPWARM_REPEATS", "999")
    parsed = KeepwarmConfig.from_env()
    assert parsed.enabled is True
    assert parsed.matrix_size == 1024
    assert parsed.repeats == 16


def test_action_is_an_immutable_transport_value():
    action = KeepwarmAction("idle", 1, 1, 2.0)
    assert action.kind == "idle"


class SyntheticArray:
    def __init__(self, value: float) -> None:
        self.value = value
        self.shape = (1,)
        self.dtype = "float32"
        self.nbytes = 4
        self.item_calls = 0

    def item(self):
        self.item_calls += 1
        return self.value


class SyntheticFast:
    def __init__(self, owner: SyntheticMX) -> None:
        self.owner = owner

    def metal_kernel(self, **kwargs):
        self.owner.kernel_creations += 1
        self.owner.kernel_options.append(kwargs)
        if self.owner.creation_fails:
            raise RuntimeError("JIT creation failed")

        def kernel(**call_kwargs):
            self.owner.kernel_runs += 1
            self.owner.kernel_calls.append(call_kwargs)
            if self.owner.invocation_fails:
                raise RuntimeError("JIT invocation failed")
            return [SyntheticArray(self.owner.output_value)]

        return kernel


class SyntheticMX:
    float32 = "float32"

    def __init__(
        self,
        *,
        creation_fails: bool = False,
        invocation_fails: bool = False,
        output_value: float = 1.0 + 1e-7,
    ) -> None:
        self.creation_fails = creation_fails
        self.invocation_fails = invocation_fails
        self.output_value = output_value
        self.fast = SyntheticFast(self)
        self.array_allocations = 0
        self.kernel_creations = 0
        self.kernel_runs = 0
        self.kernel_options: list[dict] = []
        self.kernel_calls: list[dict] = []
        self.evaluated: list[SyntheticArray] = []
        self.stream_entries: list[object] = []

    def array(self, values, *, dtype):
        assert values == [1.0]
        assert dtype == self.float32
        self.array_allocations += 1
        return SyntheticArray(values[0])

    def eval(self, value):
        self.evaluated.append(value)

    @contextmanager
    def stream(self, stream):
        self.stream_entries.append(stream)
        yield


def test_metal_pulse_jits_once_during_idle_then_request_start_reuses_it():
    fake_mx = SyntheticMX()
    touch = CompiledMetalKeepwarmTouch(fake_mx, stream="engine-stream")

    touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
    touch.touch(KeepwarmAction("request_start", 128, 1, 2.0))
    touch.touch(KeepwarmAction("post_response", 128, 2, 2.0))

    assert fake_mx.array_allocations == 1
    assert fake_mx.kernel_creations == 1
    assert fake_mx.kernel_runs == 4
    assert fake_mx.stream_entries == ["engine-stream"] * 3
    options = fake_mx.kernel_options[0]
    assert options["name"] == "omlx_keepwarm_pulse"
    assert options["compile_options"] == {"math_mode": "safe"}
    assert options["atomic_outputs"] is False
    assert options["ensure_row_contiguous"] is True
    for call in fake_mx.kernel_calls:
        assert call["grid"] == (1, 1, 1)
        assert call["threadgroup"] == (1, 1, 1)
    # Only the first, JIT-preparation output crosses to a host scalar.
    outputs = fake_mx.evaluated[1:]
    assert [output.item_calls for output in outputs] == [1, 0, 0, 0]


def test_request_start_miss_skips_without_allocation_or_jit():
    fake_mx = SyntheticMX()
    touch = CompiledMetalKeepwarmTouch(fake_mx)

    result = touch.touch(KeepwarmAction("request_start", 128, 1, 2.0))

    assert result is None
    assert fake_mx.array_allocations == 0
    assert fake_mx.kernel_creations == 0
    assert fake_mx.kernel_runs == 0
    assert fake_mx.evaluated == []


def test_missing_metal_kernel_and_jit_failure_never_fall_back_to_matmul():
    unavailable_mx = SyntheticMX()
    unavailable_mx.fast = SimpleNamespace()
    unavailable = CompiledMetalKeepwarmTouch(unavailable_mx)
    failing_mx = SyntheticMX(invocation_fails=True)
    failing = CompiledMetalKeepwarmTouch(failing_mx)

    for touch in (unavailable, failing):
        try:
            touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Metal pulse failure must propagate to backoff")

    assert unavailable_mx.array_allocations == 0
    assert failing_mx.array_allocations == 1
    assert failing._input is None
    assert failing._kernel is None
    assert failing._prepared is False


def test_metal_pulse_rejects_an_inexact_first_output_and_drops_partial_jit():
    fake_mx = SyntheticMX(output_value=7.0)
    touch = CompiledMetalKeepwarmTouch(fake_mx)

    try:
        touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
    except RuntimeError as exc:
        assert "unexpected result" in str(exc)
    else:
        raise AssertionError("inexact Metal pulse output must fail closed")

    assert touch._input is None
    assert touch._kernel is None
    assert touch._prepared is False


def test_metal_pulse_enforces_existing_action_bounds_before_jit():
    fake_mx = SyntheticMX()
    touch = CompiledMetalKeepwarmTouch(fake_mx)

    for action in (
        KeepwarmAction("idle", 0, 1, 2.0),
        KeepwarmAction("idle", 1025, 1, 2.0),
        KeepwarmAction("idle", 1, 0, 2.0),
        KeepwarmAction("idle", 1, 17, 2.0),
        KeepwarmAction("unknown", 1, 1, 2.0),
    ):
        try:
            touch.touch(action)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-bounds Metal pulse must fail closed")

    assert fake_mx.array_allocations == 0
    assert fake_mx.kernel_creations == 0


def test_metal_pulse_retains_exactly_one_four_byte_input_and_close_drops_it():
    fake_mx = SyntheticMX()
    touch = CompiledMetalKeepwarmTouch(fake_mx, stream="engine-stream")
    touch.touch(KeepwarmAction("idle", 1, 1, 2.0))

    assert touch._input is not None
    assert touch._input.nbytes == 4
    assert touch._kernel is not None
    assert not hasattr(touch, "_output")

    touch.close()

    assert touch._input is None
    assert touch._kernel is None
    assert touch._mx is None
    assert touch._stream is None
    try:
        touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed Metal pulse must not allocate again")


def test_engine_compiled_touch_failure_is_nonfatal_to_inference():
    class FailingTouch:
        @staticmethod
        def touch(_action):
            raise RuntimeError("synthetic failure")

    core = core_with_scheduler(Scheduler())
    core._compiled_metal_keepwarm = FailingTouch()
    core._keepwarm.observe_request_state(True)
    core._keepwarm.observe_request_state(False, cache_tokens=4096)
    action = KeepwarmAction("idle", 1, 1, 2.0, cache_tokens=4096)

    assert core._run_keepwarm_action(action) is False
    assert core._keepwarm.snapshot()["failures"] == 1


def test_engine_request_start_pulse_miss_is_a_skip_not_a_failure():
    core = core_with_scheduler(Scheduler())
    core._compiled_metal_keepwarm = SimpleNamespace(touch=lambda _action: None)
    core._keepwarm.observe_request_state(True)
    action = KeepwarmAction("request_start", 128, 1, 2.0, cache_tokens=4096)

    assert core._run_keepwarm_action(action) is False
    snapshot = core._keepwarm.snapshot()
    assert snapshot["skips"] == 1
    assert snapshot["failures"] == 0
