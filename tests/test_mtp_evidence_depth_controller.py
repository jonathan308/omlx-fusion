# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the adaptive MTP draft-depth controller.

The controller scores each depth as expected committed tokens over measured
cycle cost. Discounted Jeffreys evidence keeps censored acceptance tails
explorable, empirical cost intervals retire separated rival probes, and a
staleness backstop catches later cost drift without eager depth-0 calibration.
These tests exercise the host-side logic in isolation (no MLX / GPU).
"""

import math
import random

from types import SimpleNamespace

from omlx.patches.mlx_lm_mtp.batch_generator import (
    _DepthController,
    _EvidenceDepthController,
    _LockstepAcceptanceDepthController,
    _MtpState,
    _maybe_finish_mtp_reentry_probe,
    _new_depth_controller,
)


def _simulate(controller, cycles, p_by_depth, ms_by_depth, seed=0):
    """Drive the controller like the real loop: draft ``cur``, observe outcome."""
    rng = random.Random(seed)
    for _ in range(cycles):
        depth = controller.cur
        accepted = 0
        for j in range(depth):
            if rng.random() < p_by_depth[j]:
                accepted += 1
            else:
                break
        controller.observe(depth, accepted, ms_by_depth[depth])
    return controller


def _seed_acceptance(controller, probabilities, trials=8.0):
    """Install mature discounted evidence for deterministic policy tests."""

    controller._accept_trials = [float(trials)] * controller.max_depth
    controller._accept_hits = [
        max(0.0, min(float(trials), float(p) * trials))
        for p in probabilities
    ]
    controller.p = list(map(float, probabilities))


def _simulate_with_decision_lag(
    controller,
    cycles,
    p_by_depth,
    ms_by_depth,
    seed=0,
):
    """Model the real loop: the next chain is built before observe changes cur."""

    rng = random.Random(seed)
    built_depth = controller.cur
    picks = []
    for _ in range(cycles):
        depth = built_depth
        picks.append(depth)
        accepted = 0
        for j in range(depth):
            if rng.random() < p_by_depth[j]:
                accepted += 1
            else:
                break
        built_depth = controller.cur
        controller.observe(depth, accepted, ms_by_depth[depth])
    return picks


def test_observe_signature_is_three_positional():
    # Guards the call site batch_generator.py: controller.observe(k, m, ms).
    c = _EvidenceDepthController(2)
    c.observe(2, 1, 12.5)
    assert c.cycles == 1


def test_warmup_measures_every_depth_once():
    c = _EvidenceDepthController(3)
    assert c.cur == 3  # speculative sweep walks 3 -> 2 -> 1
    c.observe(3, 3, 30.0)
    assert c.cur == 2
    c.observe(2, 2, 20.0)
    assert c.cur == 1
    c.observe(1, 1, 10.0)
    assert c.t == {1: 10.0, 2: 20.0, 3: 30.0}
    assert 0 not in c.t
    assert c._warmup == []


def test_depth_zero_is_lazy_with_stale_backstop():
    c = _EvidenceDepthController(3)
    _simulate_with_zero(
        c,
        80,
        p_by_depth=[0.99, 0.99, 0.99],
        ms_by_depth={0: 10.0, 1: 18.0, 2: 20.0, 3: 22.0},
    )
    assert 0 not in c.t  # under ten seconds and a mature max-depth winner

    # A shallow winner gets the ordinary five-second unknown-baseline probe.
    shallow = _EvidenceDepthController(3)
    picks = _simulate_with_zero(
        shallow,
        520,
        p_by_depth=[0.55, 0.5, 0.45],
        ms_by_depth={0: 11.5, 1: 26.0, 2: 27.5, 3: 29.0},
    )
    assert 0 in shallow.t
    assert picks[-80:].count(0) > 60


def test_marginal_est_uses_measured_slope_not_prior():
    c = _EvidenceDepthController(3, marginal_ms=7.0)
    assert c._marginal_est() == 7.0  # fallback prior before two depths measured
    c.t = {1: 10.0, 2: 40.0, 3: 70.0}
    assert math.isclose(c._marginal_est(), 30.0, rel_tol=1e-9)
    c.t = {1: 10.0, 3: 70.0}
    assert math.isclose(c._t_est(2), 10.0 + 30.0 * 1, rel_tol=1e-9)


def test_time_alpha_horizon_is_wall_clock():
    c = _EvidenceDepthController(2)
    assert math.isclose(c._time_alpha(c.TAU_MS), 1.0 - math.exp(-1.0), rel_tol=1e-9)
    assert c._time_alpha(80.0) > c._time_alpha(8.0)
    assert c._time_alpha(0.0) == 0.0


def test_spike_guard_damps_one_off_outlier():
    c = _EvidenceDepthController(2)
    c.t[2] = 20.0
    c._update_time(2, 200.0)  # a 10x spike must not drag the estimate near 200
    assert c.t[2] < 60.0


def test_expensive_extra_verify_settles_at_depth_1():
    # MoE on a bandwidth-limited chip (M4 Max analog): depth-2 nearly doubles
    # the cycle cost at low d2 acceptance, so even starting deep it drops to 1.
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.p = [0.8, 0.25]
    c.t = {1: 10.0, 2: 19.0}
    c.cur = 2
    assert c._score(1) > c._score(2)
    assert c._best() == 1


def test_cheap_extra_verify_keeps_depth_2():
    # High-bandwidth chip (M3 Ultra analog) with a genuine depth-2 win: cheap
    # extra verify and high d2 acceptance -> the measured score keeps depth 2
    # (no shallow bias suppressing a real deep win — the GLM case).
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.p = [0.85, 0.7]
    c.t = {1: 10.0, 2: 10.5}
    c.cur = 1
    assert c._best() == 2


def test_exact_tie_does_not_move_deeper():
    # On an exact score tie, hysteresis + the strict '>' shallow-to-deep scan
    # keep the current (shallow) depth: no churn, no drift deeper.
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.p = [0.5, 0.0]
    c.t = {1: 10.0, 2: 10.0}
    c.cur = 1
    assert math.isclose(c._score(1), c._score(2), rel_tol=1e-12)
    assert c._best() == 1


def test_best_rival_is_bidirectional():
    # Sitting DEEP with a shallower rival whose evidence interval overlaps:
    # the probe must target the shallower depth — this is what breaks the
    # depth-2 lock (stale-high t[1] can only be corrected by re-running it).
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.cur = 2
    c.p = [0.8, 0.5]
    c.t = {1: 11.0, 2: 12.0}  # t[1] stale-high; scores land within the margin
    assert c._score(2) >= c._score(1)  # cur currently looks better...
    assert c._best_rival() == 1  # ...but depth 1 is worth re-measuring

    # And a clearly-worse rival is not probed (no probe tax).
    c2 = _EvidenceDepthController(2)
    c2._warmup = []
    c2.cur = 1
    c2.p = [0.8, 0.1]
    c2.t = {1: 10.0, 2: 19.0}
    assert c2._best_rival() is None


def test_most_stale_prefers_unmeasured_then_oldest():
    c = _EvidenceDepthController(3)
    c._warmup = []
    c.cur = 1
    c.t_age = {1: 0.0, 2: 500.0}  # depth 3 never measured -> infinitely stale
    assert c._most_stale() == 3
    # Baseline measured (the realistic post-seed state): oldest depth wins.
    c.t_age = {0: 100.0, 1: 0.0, 2: 900.0, 3: 200.0}
    assert c._most_stale() == 2
    # Once the lazy-baseline stale clock expires, an unmeasured baseline
    # outranks any finite speculative age.
    c.t_age = {1: 0.0, 2: 900.0, 3: 200.0}
    c._ms_baseline = 2.0 * c.PROBE_PERIOD_MAX_MS + 1.0
    assert c._most_stale() == 0


def test_stale_lock_is_broken_by_repeated_probes():
    # Reproduce the measured failure: warmup right after prefill measures t[1]
    # inflated (11ms vs true 10ms), the controller settles at depth 2, and
    # without bidirectional probes t[1] would never refresh (the depth-2 lock).
    # With rival probes re-running depth 1 every ~1s, the slow EMA converges
    # over a few bursts and the lock breaks.
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.cur = 2
    c.p = [0.8, 0.3]
    c.t = {0: 8.0, 1: 11.0, 2: 12.0}  # baseline measured, not competitive here  # stale-high t[1] hides depth 1's advantage
    c.t_age = {0: 0.0, 1: 0.0, 2: 0.0}
    assert c._best() == 2  # locked on the stale estimate
    # Drive real cycles: depth 2 truly costs 12ms, depth 1 truly costs 10ms.
    _simulate(c, 1500, p_by_depth=[0.8, 0.3], ms_by_depth={0: 8.0, 1: 10.0, 2: 12.0})
    assert c.t[1] < 10.5  # repeated probes converged t[1] toward the truth
    assert c._best() == 1  # lock broken


def test_probe_duty_bound_scales_period_on_heavy_models():
    # On a 100ms-cycle model, a 1s cadence would spend ~40% of cycles probing
    # (4-cycle burst every 10 cycles). The duty bound stretches the period so
    # probes stay under ~PROBE_DUTY of cycles.
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.cur = 1
    _seed_acceptance(c, [0.8, 0.25])
    c.t = {1: 100.0, 2: 110.0}
    c.t_age = {1: 0.0, 2: 0.0}
    c._ms_probe = c.PROBE_PERIOD_MS + 1.0  # past the light-model cadence...
    c.observe(1, 1, 100.0)
    assert c.probe_left == 0  # ...but under the duty-bounded period: no probe
    assert c.cur == 1
    # Past the duty-bounded period the rival probe fires.
    c._ms_probe = (
        (c.PROBE_LEN + c.PROBE_PIPELINE_TAIL) * 100.0 / c.PROBE_DUTY + 1.0
    )
    c.observe(1, 1, 100.0)
    assert c.probe_left == c.PROBE_LEN
    assert c.cur == 2


def test_uncertain_rival_gets_probed_after_wall_clock_period():
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.probe_left = 0
    _seed_acceptance(c, [0.85, 0.5])
    c.t = {1: 10.0, 2: 12.5}
    c.t_age = {1: 0.0, 2: 0.0}
    c.cur = 1
    c._ms_probe = c.PROBE_PERIOD_MS - 100.0
    c.observe(1, 1, 10.0)  # under the period -> no probe yet
    assert c.probe_left == 0
    assert c.cur == 1
    c._ms_probe = c.PROBE_PERIOD_MS - 5.0
    c.observe(1, 1, 10.0)  # crosses the period while rival is close -> probe
    assert c.probe_left == c.PROBE_LEN
    assert c.cur == 2


def test_exploration_probe_targets_most_stale_depth():
    # When the exploration clock lapses, the probe goes to the most-stale
    # depth even if it is not a close rival (bounded staleness for all depths).
    c = _EvidenceDepthController(3)
    c._warmup = []
    c.probe_left = 0
    c.cur = 1
    c.p = [0.9, 0.1, 0.1]  # depths 2/3 score far below depth 1
    c.t = {0: 7.0, 1: 10.0, 2: 30.0, 3: 50.0}  # baseline measured, not stale
    c.t_age = {0: 50.0, 1: 0.0, 2: 100.0, 3: 9000.0}
    assert c._best_rival() is None  # no close rival
    c._ms_probe = c.PROBE_PERIOD_MS + 1.0
    c._ms_explore = c.PROBE_PERIOD_MAX_MS + 1.0
    c.observe(1, 1, 10.0)
    assert c.probe_left == c.PROBE_LEN
    assert c.cur == 3  # the never/least-recently measured depth


def test_probe_burst_completes_and_resets_cadence():
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.p = [0.85, 0.55]
    c.t = {1: 10.0, 2: 11.5}
    c.cur = 2
    c.probe_left = c.PROBE_LEN
    for _ in range(c.PROBE_LEN):
        c.observe(2, 1, 11.5)
    assert c.probe_left == 0
    assert c._ms_probe == 0.0


def test_depth_five_high_accept_converges_without_baseline_tax():
    costs = {0: 30.0, 1: 55.0, 2: 60.0, 3: 65.0, 4: 70.0, 5: 80.0}
    for probability in (1.0, 0.99):
        c = _EvidenceDepthController(5)
        picks = _simulate_with_decision_lag(
            c,
            95,
            p_by_depth=[probability] * 5,
            ms_by_depth=costs,
            seed=0,
        )
        assert picks.count(5) >= 75
        assert picks.count(0) == 0
        assert c._best_candidate() == 5


def test_censored_tail_gets_immediate_acceptance_ucb_probe_and_recovers():
    c = _EvidenceDepthController(5)
    c._warmup = []
    c.cur = 4
    c.t = {1: 55.0, 2: 60.0, 3: 65.0, 4: 70.0, 5: 80.0}
    c.t_age = {depth: 0.0 for depth in c.t}
    c.t_samples = {depth: 4 for depth in c.t}
    _seed_acceptance(c, [0.99, 0.99, 0.99, 0.99, 0.5])
    c._accept_hits[4] = 0.0
    c._accept_trials[4] = 0.0
    c._ms_probe = (
        (c.PROBE_LEN + c.PROBE_PIPELINE_TAIL) * 70.0 / c.PROBE_DUTY + 1.0
    )

    c.observe(4, 4, 70.0)
    assert c._probe_target == 5
    assert c.cur == 5

    for _ in range(6):
        c.observe(5, 5, 80.0)
    assert c._accept_trials[4] >= c.ACCEPT_EXPLORE_TRIALS
    assert c._best_candidate() == 5
    assert c.cur == 5


def test_high_low_high_content_adapts_and_recovers_max_depth():
    c = _EvidenceDepthController(5)
    costs = {0: 30.0, 1: 55.0, 2: 60.0, 3: 65.0, 4: 70.0, 5: 80.0}
    high = _simulate_with_decision_lag(
        c, 120, [0.99] * 5, costs, seed=1
    )
    low = _simulate_with_decision_lag(
        c, 350, [0.5, 0.4, 0.3, 0.2, 0.1], costs, seed=2
    )
    recovered = _simulate_with_decision_lag(
        c, 500, [0.99] * 5, costs, seed=3
    )

    assert high[-50:].count(5) > 40
    assert low[-50:].count(0) > 40
    assert recovered[-50:].count(5) > 35
    assert c._best_candidate() == 5


def test_stale_cost_drift_reopens_probe_and_flips_winner():
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.cur = 1
    _seed_acceptance(c, [0.9, 0.9])
    c.t = {0: 15.0, 1: 12.0, 2: 20.0}
    c.t_age = {0: 0.0, 1: 0.0, 2: 0.0}
    c.t_samples = {0: 4, 1: 4, 2: 4}
    c._t_stat_mean = dict(c.t)
    c._t_stat_m2 = {0: 0.0, 1: 0.0, 2: 0.0}

    _simulate(
        c,
        900,
        p_by_depth=[1.0, 1.0],
        ms_by_depth={0: 15.0, 1: 12.0, 2: 10.0},
    )
    assert c.t_samples[2] > 4
    assert c._best_candidate() == 2
    assert c.cur == 2


def test_warmup_and_probe_wait_for_lagged_requested_depth():
    c = _EvidenceDepthController(3)
    picks = _simulate_with_decision_lag(
        c,
        6,
        p_by_depth=[1.0, 1.0, 1.0],
        ms_by_depth={0: 5.0, 1: 10.0, 2: 20.0, 3: 30.0},
    )
    assert picks == [3, 3, 2, 2, 1, 1]
    assert c._warmup == []
    assert c.t_samples == {1: 2, 2: 2, 3: 2}

    c._start_probe(2)
    c.observe(1, 1, 10.0)  # already-built old chain
    assert c._probe_target == 2
    assert c.probe_left == c.PROBE_LEN
    c.observe(2, 2, 20.0)
    assert c.probe_left < c.PROBE_LEN


def test_expensive_extra_verify_settles_at_depth_1_end_to_end():
    # Baseline (depth 0) is measured by the warmup tail but stays clearly
    # non-competitive at 80% acceptance, so the run settles at depth 1.
    c = _EvidenceDepthController(2)
    _simulate(c, 200, p_by_depth=[0.8, 0.25], ms_by_depth={0: 8.0, 1: 10.0, 2: 19.0})
    assert c._best() == 1


def test_max_depth_one_is_inert():
    c = _EvidenceDepthController(1)
    _simulate(c, 40, p_by_depth=[0.9], ms_by_depth={1: 10.0})
    assert c.cur == 1
    assert c._best() == 1


# ---------------------------------------------------------------------------
# Depth 0 — the no-speculation escape hatch.
# ---------------------------------------------------------------------------


def _simulate_with_zero(controller, cycles, p_by_depth, ms_by_depth, seed=0):
    """Like _simulate, but honors depth-0 selections (no drafts, base cost)."""
    rng = random.Random(seed)
    picks = []
    for _ in range(cycles):
        depth = controller.cur
        picks.append(depth)
        accepted = 0
        for j in range(depth):
            if rng.random() < p_by_depth[j]:
                accepted += 1
            else:
                break
        controller.observe(depth, accepted, ms_by_depth[depth])
    return picks


def test_zero_not_selectable_without_measurement():
    # Extrapolated baselines must never park the sequence — only a measured
    # (or seeded) t[0] makes depth 0 selectable.
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.p = [0.1, 0.1]
    c.t = {1: 20.0, 2: 22.0}
    c.cur = 1
    assert 0 not in c._select_candidates()
    assert c._best() >= 1


def test_unmeasured_zero_is_most_stale_probe_target():
    # Discovery path without a post-init seed: the staleness explorer sees
    # the unmeasured baseline as infinitely stale and probes it (a probe of
    # depth 0 is just a plain decode step, so it is always safe).
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.t = {1: 10.0, 2: 12.0}
    c.t_age = {1: 0.0, 2: 50.0}
    c.cur = 1
    c._ms_baseline = c.PROBE_PERIOD_MAX_MS + 1.0
    assert c._most_stale() == 0


def test_observe_zero_updates_base_cost_only():
    c = _EvidenceDepthController(2)
    c._warmup = []
    c.p = [0.5, 0.5]
    c.t = {1: 20.0, 2: 22.0}
    c.cur = 1
    p_before = list(c.p)
    c.observe(0, 0, 10.0)
    assert c.t[0] == 10.0
    assert c.t[1] == 20.0 and c.t[2] == 22.0
    assert c.p == p_before  # no acceptance evidence from a plain step


def test_parks_at_zero_when_every_depth_loses():
    # gemma4 26B story/16k analog: baseline 11.5 ms/token, the L=1->2 verify
    # step makes even depth 1 cost ~26 ms at ~55% acceptance. Every
    # speculative depth scores below the plain step, so the controller must
    # park at 0 for the bulk of the run (probe bursts excepted).
    c = _EvidenceDepthController(3)
    picks = _simulate_with_zero(
        c,
        520,
        p_by_depth=[0.55, 0.5, 0.45],
        ms_by_depth={0: 11.5, 1: 26.0, 2: 27.5, 3: 29.0},
    )
    # Lazy baseline discovery occurs after the five-second backstop; once it
    # does, the tail parks just as the eager-calibration controller did.
    parked = sum(1 for d in picks[-80:] if d == 0)
    assert parked / 80 > 0.7
    assert c._best() == 0


def test_reenters_speculation_when_content_turns_predictable():
    # Park first (story analog), then flip the content to code-like accept
    # rates: rival probes re-measure the speculative depths, acceptance
    # evidence refreshes, and the controller must leave depth 0.
    c = _EvidenceDepthController(3)
    _simulate_with_zero(
        c,
        200,
        p_by_depth=[0.55, 0.5, 0.45],
        ms_by_depth={0: 11.5, 1: 26.0, 2: 27.5, 3: 29.0},
        seed=1,
    )
    picks = _simulate_with_zero(
        c,
        600,
        p_by_depth=[0.95, 0.92, 0.9],
        ms_by_depth={0: 11.5, 1: 13.0, 2: 14.0, 3: 15.0},
        seed=2,
    )
    tail = picks[-100:]
    speculative = sum(1 for d in tail if d >= 1)
    assert speculative / len(tail) > 0.7
    assert c._best() >= 1


def test_high_accept_workload_never_parks():
    # code/4k analog: speculation clearly wins; the escape hatch must not
    # tax it (depth 0 may appear only inside rare probe bursts).
    c = _EvidenceDepthController(3)
    picks = _simulate_with_zero(
        c,
        300,
        p_by_depth=[0.9, 0.85, 0.8],
        ms_by_depth={0: 10.0, 1: 12.0, 2: 13.0, 3: 14.5},
    )
    zero_share = sum(1 for d in picks[20:] if d == 0) / len(picks[20:])
    assert zero_share < 0.2
    assert c._best() >= 1



def test_losing_speculation_builds_exit_streak():
    # story/4k analog: best speculative score sits between 1.0x and
    # EXIT_MARGIN of the taxed baseline — locally "fine", globally losing
    # to the pipelined standard step. The streak must build toward exit.
    c = _EvidenceDepthController(3)
    c._warmup = []
    c.t = {0: 12.0, 1: 20.0, 2: 21.5, 3: 23.0}
    c.t_age = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    _seed_acceptance(c, [0.6, 0.5, 0.4])
    picks = _simulate_with_zero(
        c,
        60,
        p_by_depth=[0.6, 0.5, 0.4],
        ms_by_depth={0: 12.0, 1: 20.0, 2: 21.5, 3: 23.0},
    )
    assert c.should_exit()
    assert c.exit_streak >= c.EXIT_STREAK
    del picks


def test_winning_speculation_never_exits():
    # code analogs: clear speculative wins keep the exit streak at zero.
    c = _EvidenceDepthController(3)
    _simulate_with_zero(
        c,
        120,
        p_by_depth=[0.9, 0.85, 0.8],
        ms_by_depth={0: 10.0, 1: 12.0, 2: 13.0, 3: 14.5},
    )
    assert not c.should_exit()
    assert c.exit_streak == 0


def test_exit_margin_arg_overrides_prior_with_clamp():
    # A measured loop tax seeds later controllers; the fallback prior only
    # applies until the first hand-off measured the real ratio.
    from omlx.patches.mlx_lm_mtp.batch_generator import _STD_TAX_MAX

    c = _EvidenceDepthController(3, exit_margin=1.06)
    assert math.isclose(c.EXIT_MARGIN, 1.06, rel_tol=1e-9)
    assert math.isclose(_EvidenceDepthController(3).EXIT_MARGIN, 1.15, rel_tol=1e-9)
    assert _EvidenceDepthController(3, exit_margin=9.0).EXIT_MARGIN == _STD_TAX_MAX
    assert _EvidenceDepthController(3, exit_margin=0.5).EXIT_MARGIN == 1.0


def test_std_tax_probe_measures_and_smooths():
    from types import SimpleNamespace

    from omlx.patches.mlx_lm_mtp.batch_generator import (
        _STD_TAX_SAMPLES,
        _STD_TAX_SKIP,
        _arm_std_tax_probe,
        _record_std_tax_sample,
    )

    model = SimpleNamespace()
    gb = SimpleNamespace(model=model)
    _arm_std_tax_probe(gb, 12.0)
    # Transition steps are skipped, then the median of the samples is used.
    for _ in range(_STD_TAX_SKIP):
        _record_std_tax_sample(gb, 99.0)
    for _ in range(_STD_TAX_SAMPLES):
        _record_std_tax_sample(gb, 10.0)
    assert math.isclose(model._omlx_mtp_loop_tax, 1.2, rel_tol=1e-9)
    assert not hasattr(gb, "_omlx_mtp_tax_probe")

    # A second hand-off EMA-blends toward the new measurement.
    _arm_std_tax_probe(gb, 11.0)
    for _ in range(_STD_TAX_SKIP):
        _record_std_tax_sample(gb, 99.0)
    for _ in range(_STD_TAX_SAMPLES):
        _record_std_tax_sample(gb, 11.0)
    assert 1.0 < model._omlx_mtp_loop_tax < 1.2


# ---------------------------------------------------------------------------
# Qwen4 gate and review-blocker regressions.
# ---------------------------------------------------------------------------


def _model(model_type):
    return SimpleNamespace(model_type=model_type)


def test_evidence_policy_is_off_by_default_for_qwen4(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN4_EVIDENCE_DEPTH", raising=False)
    controller = _new_depth_controller(_model("qwen4_exp"), 5)
    assert type(controller) is _DepthController


def test_evidence_policy_requires_both_env_and_qwen4(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_EVIDENCE_DEPTH", "1")
    qwen = _new_depth_controller(_model("qwen4_exp"), 5)
    other = _new_depth_controller(_model("qwen3_5_moe"), 5)
    assert type(qwen) is _EvidenceDepthController
    assert type(other) is _DepthController


def test_lockstep_override_remains_authoritative(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_EVIDENCE_DEPTH", "1")
    monkeypatch.setenv("OMLX_MTP_DISTRIBUTED_LOCKSTEP_DEPTH", "1")
    controller = _new_depth_controller(_model("qwen4_exp"), 5)
    assert type(controller) is _LockstepAcceptanceDepthController


def test_mature_perfect_acceptance_exploitation_converges_to_one():
    controller = _EvidenceDepthController(5)
    controller._warmup = []
    for _ in range(80):
        controller.observe(5, 5, 80.0)
    assert controller.p == [1.0] * 5
    assert math.isclose(controller._expected_tokens(5), 6.0, rel_tol=1e-12)


def test_mature_empirical_acceptance_has_no_permanent_prior_bias():
    controller = _EvidenceDepthController(1)
    controller._accept_hits[0] = 9.9
    controller._accept_trials[0] = 10.0
    assert math.isclose(controller._accept_posterior_mean(0), 0.99, rel_tol=1e-12)


def test_maintenance_sample_does_not_consume_warmup_or_probe_slot():
    controller = _EvidenceDepthController(3)
    before = list(controller._warmup)
    controller.observe(3, 3, 30.0, time_sample=False)
    assert controller._warmup == before
    assert 3 not in controller.t

    controller._warmup = []
    controller._start_probe(2)
    remaining = controller.probe_left
    controller.observe(2, 2, 20.0, time_sample=False)
    assert controller.probe_left == remaining
    assert 2 not in controller.t
    assert controller.cur == 2


def test_discounted_cost_interval_tracks_drift_without_lifetime_lock():
    controller = _EvidenceDepthController(2)
    controller._warmup = []
    for _ in range(40):
        controller._update_time(2, 10.0)
    stable = controller.t[2]
    for _ in range(8):
        controller._update_time(2, 16.0)
    lower, upper = controller._cost_bounds(2)
    assert controller.t[2] > stable
    assert lower < controller.t[2] < upper
    assert 1.0 / controller._t_weight_sq[2] < controller.t_samples[2]


def test_probe_duty_period_accounts_for_already_built_tail():
    controller = _EvidenceDepthController(2)
    controller._warmup = []
    controller.cur = 1
    _seed_acceptance(controller, [0.8, 0.25])
    controller.t = {1: 100.0, 2: 110.0}
    controller.t_age = {1: 0.0, 2: 0.0}
    threshold = (
        (controller.PROBE_LEN + controller.PROBE_PIPELINE_TAIL)
        * 100.0
        / controller.PROBE_DUTY
    )
    controller._ms_probe = threshold - 101.0
    controller.observe(1, 1, 100.0)
    assert controller.probe_left == 0
    controller._ms_probe = threshold - 99.0
    controller.observe(1, 1, 100.0)
    assert controller.probe_left == controller.PROBE_LEN


def test_reentry_cannot_succeed_without_mature_plain_baseline():
    controller = _EvidenceDepthController(3)
    controller._warmup = []
    controller.p = [1.0, 1.0, 1.0]
    controller.t = {1: 20.0, 2: 21.0, 3: 22.0}
    controller.t_samples = {1: 4, 2: 4, 3: 4}
    state = _MtpState(uid="row", controller=controller, reentry_probe=True)
    batch = SimpleNamespace(_omlx_mtp_park_state=object())
    assert not _maybe_finish_mtp_reentry_probe(
        batch,
        state,
        was_warmup=False,
    )
    assert state.reentry_probe is True

    controller.t[0] = 18.0
    controller.t_samples[0] = controller.BASELINE_MIN_SAMPLES
    assert controller.reentry_win_proven()
    assert _maybe_finish_mtp_reentry_probe(
        batch,
        state,
        was_warmup=False,
    )
    assert state.reentry_probe is False


def test_reentry_preparation_schedules_bounded_plain_measurements():
    controller = _EvidenceDepthController(3)
    controller.prepare_reentry_probe()
    assert controller._warmup[-controller.BASELINE_MIN_SAMPLES :] == [
        0
    ] * controller.BASELINE_MIN_SAMPLES
