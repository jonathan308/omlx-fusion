"""Contract tests for the GLM-5.3 DFlash2 target adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("dflash_mlx")


def _fake_target(*, hidden_size=4096, vocab_size=154880, layers=45):
    inner = SimpleNamespace(
        layers=[SimpleNamespace() for _ in range(layers)],
        embed_tokens=object(),
        fa_idx=3,
        ssm_idx=0,
    )
    language_model = SimpleNamespace(
        args=SimpleNamespace(
            model_type="glm5_next_text",
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            mhc=True,
            hc_mult=4,
        ),
        model=inner,
    )
    return SimpleNamespace(model_type="glm5_next", language_model=language_model)


def _fake_draft(*, hidden_size=4096, vocab_size=154880, layers=45):
    args = SimpleNamespace(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_target_layers=layers,
    )
    return SimpleNamespace(
        args=args,
        is_dflash2=True,
        target_layer_ids=[5, 14, 24, 33, 42],
    )


def _draft_meta(architecture="DFlash2DraftModel"):
    return {"config": {"architectures": [architecture]}}


def test_glm5_target_gate_is_explicit(tmp_path):
    from omlx.engine.dflash import is_dflash_compatible

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "glm5_next"}), encoding="utf-8"
    )
    assert is_dflash_compatible(tmp_path) == (True, "")


def test_dflash2_pair_validation_accepts_published_geometry():
    from omlx.patches.dflash_glm5 import validate_glm5_dflash_pair

    validate_glm5_dflash_pair(_fake_target(), _fake_draft(), _draft_meta())


@pytest.mark.parametrize(
    ("draft", "match"),
    [
        (_fake_draft(hidden_size=2048), "hidden_size mismatch"),
        (_fake_draft(vocab_size=32000), "vocab_size mismatch"),
        (_fake_draft(layers=44), "num_target_layers mismatch"),
    ],
)
def test_dflash2_pair_validation_rejects_geometry_mismatch(draft, match):
    from omlx.patches.dflash_glm5 import validate_glm5_dflash_pair

    with pytest.raises(ValueError, match=match):
        validate_glm5_dflash_pair(_fake_target(), draft, _draft_meta())


def test_dflash2_pair_validation_rejects_non_dflash2():
    from omlx.patches.dflash_glm5 import validate_glm5_dflash_pair

    with pytest.raises(ValueError, match="DFlash2DraftModel"):
        validate_glm5_dflash_pair(
            _fake_target(), _fake_draft(), _draft_meta("DFlashDraftModel")
        )


def test_mhc_capture_contract_matches_official_serving_contract():
    from omlx.patches.dflash_glm5 import _contract_mhc_hidden

    hidden = mx.arange(48, dtype=mx.float32).reshape(1, 2, 4, 6)
    actual = _contract_mhc_hidden(hidden)
    expected = hidden.mean(axis=2)
    mx.eval(actual, expected)
    assert actual.shape == (1, 2, 6)
    assert mx.array_equal(actual, expected).item()


def test_hidden_extraction_maps_target_layer_k_to_capture_k_plus_one():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    captured = {
        6: mx.full((1, 2, 3), 6),
        15: mx.full((1, 2, 3), 15),
        25: mx.full((1, 2, 3), 25),
        34: mx.full((1, 2, 3), 34),
        43: mx.full((1, 2, 3), 43),
    }
    feature = Glm5NextTargetOps().extract_context_feature(captured, [5, 14, 24, 33, 42])
    mx.eval(feature)
    assert feature.shape == (1, 2, 15)
    assert feature[0, 0].tolist() == [
        6,
        6,
        6,
        15,
        15,
        15,
        25,
        25,
        25,
        34,
        34,
        34,
        43,
        43,
        43,
    ]


def test_capabilities_fail_closed_for_unproven_paths():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    caps = Glm5NextTargetOps().capabilities_for(_fake_target())
    assert caps.supports_dflash is True
    assert caps.supports_recurrent_rollback is True
    assert caps.supports_kv_trim is True
    assert caps.supports_prefix_snapshot is False
    assert caps.supports_verify_linear is False
    assert caps.supports_tree_verify is False


def test_composite_dsa_cache_rollback_uses_kv_offset_and_checks_trim():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    class Composite:
        def __init__(self):
            self.kv = SimpleNamespace(offset=18)
            self.trimmed = 0

        def __getitem__(self, index):
            assert index == 0
            return self.kv

        def trim(self, count):
            self.trimmed += count
            self.kv.offset -= count
            return count

    cache = Composite()
    elapsed = Glm5NextTargetOps().restore_after_acceptance(
        [cache], target_len=15, acceptance_length=1, drafted_tokens=7
    )
    assert elapsed > 0
    assert cache.trimmed == 3
    assert cache.kv.offset == 15


def test_composite_dsa_cache_rollback_fails_closed_on_partial_trim():
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    class Composite:
        def __getitem__(self, index):
            return SimpleNamespace(offset=18)

        def trim(self, count):
            return count - 1

    with pytest.raises(RuntimeError, match="rollback failed"):
        Glm5NextTargetOps().restore_after_acceptance(
            [Composite()], target_len=15, acceptance_length=1, drafted_tokens=7
        )


def test_actual_cache_list_rollback_crosses_pooling_boundary_exactly():
    from mlx_lm.models.cache import CacheList, KVCache

    from omlx.patches.deepseek_v4 import apply_pooling_cache_support
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    apply_pooling_cache_support()
    from mlx_lm.models.cache import PoolingCache

    def append(cache, tokens, *, offset):
        kv = mx.arange(tokens * 3, dtype=mx.float32).reshape(1, tokens, 3)
        gate = mx.ones((1, tokens, 1), dtype=mx.float32)
        ready_kv, _ready_gate, _ = cache[1].accumulate_windows(kv, gate, offset)
        pooled = ready_kv[:, ::4] if ready_kv.shape[1] else ready_kv
        cache[1].update_and_fetch(pooled)
        keys = kv[:, None]
        values = mx.zeros((1, 1, tokens, 0), dtype=mx.float32)
        cache[0].update_and_fetch(keys, values)

    actual = CacheList(KVCache(), PoolingCache(4))
    append(actual, 3, offset=0)
    append(actual, 4, offset=3)
    Glm5NextTargetOps().restore_after_acceptance(
        [actual], target_len=4, acceptance_length=0, drafted_tokens=3
    )

    reference = CacheList(KVCache(), PoolingCache(4))
    append(reference, 3, offset=0)
    append(reference, 1, offset=3)
    mx.eval(
        *[v for v in actual[0].state if v is not None],
        *[v for v in reference[0].state if v is not None],
        *[v for v in actual[1].state if v is not None],
        *[v for v in reference[1].state if v is not None],
    )
    assert actual[0].offset == reference[0].offset == 4
    assert actual[1].remainder == reference[1].remainder == 0
    for lhs, rhs in zip(actual[0].state, reference[0].state, strict=True):
        assert mx.array_equal(lhs, rhs).item()
    for lhs, rhs in zip(actual[1].state, reference[1].state, strict=True):
        if lhs is None or rhs is None:
            assert lhs is rhs
        else:
            assert mx.array_equal(lhs, rhs).item()


def test_backend_install_is_idempotent_and_registers_before_resolution():
    from dflash_mlx.engine import target_ops

    from omlx.patches.dflash_glm5 import (
        _BACKEND_PATH,
        install_dflash_glm5_backend,
        restore_glm5_dflash_class_patches,
    )

    try:
        install_dflash_glm5_backend()
        install_dflash_glm5_backend()
        assert target_ops.TARGET_BACKENDS.count(_BACKEND_PATH) == 1
    finally:
        restore_glm5_dflash_class_patches()


def test_verify_block_scopes_pooling_undo_gate_even_on_error(monkeypatch):
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps
    from omlx.patches.mlx_lm_mtp import cache_rollback

    ops = Glm5NextTargetOps()

    def fail_while_armed(*args, **kwargs):
        assert cache_rollback._is_undo_armed() is True
        raise RuntimeError("verify failed")

    monkeypatch.setattr(ops, "forward_with_hidden_capture", fail_while_armed)
    with pytest.raises(RuntimeError, match="verify failed"):
        ops.verify_block(
            target_model=object(),
            verify_ids=mx.zeros((1, 2), dtype=mx.int32),
            target_cache=[],
        )
    assert cache_rollback._is_undo_armed() is False


def test_shared_lifecycle_restore_removes_glm_hook_without_qwen_backups():
    from dflash_mlx.engine.spec_epoch import SpeculativeSession

    from omlx.patches.dflash_glm5 import (
        _install_glm5_prefill_chunking_bridge,
        _install_glm5_recurrent_hook,
    )
    from omlx.patches.dflash_lifecycle import restore_dflash_class_patches

    class Attention:
        def __call__(self, inputs, mask=None, cache=None):
            return inputs

    original = Attention.__call__
    original_prefill = SpeculativeSession._run_prefill_events
    _install_glm5_recurrent_hook(Attention())
    _install_glm5_prefill_chunking_bridge()
    assert Attention.__call__ is not original
    assert SpeculativeSession._run_prefill_events is not original_prefill
    restore_dflash_class_patches()
    assert Attention.__call__ is original
    assert SpeculativeSession._run_prefill_events is original_prefill


def _run_fake_prefill(*, backend_name: str, prompt_len: int = 11):
    from dflash_mlx.engine.spec_epoch import (
        SpeculativeSession,
        _RequestState,
        _SessionRequest,
        _YieldPauseTracker,
    )

    calls = []

    class TargetOps:
        def __init__(self):
            self.backend_name = backend_name

        def forward_with_hidden_capture(
            self,
            target_model,
            *,
            input_ids,
            cache,
            capture_layer_ids,
            logits_last_only,
        ):
            del target_model, cache, capture_layer_ids, logits_last_only
            width = int(input_ids.shape[1])
            calls.append(width)
            return (
                mx.zeros((1, 1, 16), dtype=mx.float32),
                {1: mx.zeros((1, width, 2), dtype=mx.float32)},
            )

        def extract_context_feature(self, captured, target_layer_ids):
            del target_layer_ids
            return captured[1]

    class Draft:
        @staticmethod
        def project_target_hidden(features):
            return features

    class SnapshotTrap:
        active = True

        def should_publish_frontier(self, *_args, **_kwargs):
            raise AssertionError("GLM prefill must not consult snapshot publication")

        def publish(self, *_args, **_kwargs):
            raise AssertionError("GLM prefill must not publish a snapshot")

    session = SpeculativeSession(
        target_model=object(),
        draft_model=Draft(),
        target_ops=TargetOps(),
        target_cache=[],
        draft_cache=[],
        draft_backend=object(),
        runtime_config=SimpleNamespace(prefill_step_size=4),
        quantize_kv_cache=False,
        snap_prefix_len=0,
        supports_prefix_snapshot=False,
        allow_full_context_draft_layers=False,
        draft_sink_size=0,
        draft_window_size=0,
        target_layer_id_list=[0],
        capture_layer_ids={1},
        profile_cycles=False,
        memory_waterfall=False,
        clear_cache_boundaries=False,
        target_fa_window=0,
        copyspec_index=object(),
        copyspec_mode="off",
    )
    request = _SessionRequest.from_tokens(
        prompt_tokens=list(range(prompt_len)),
        max_new_tokens=0,
        block_tokens=1,
        stop_token_ids=None,
        suppress_token_ids=None,
        prefix_snapshot=object(),
        snapshot_service=SnapshotTrap(),
        stable_prefix_len=5,
        prefix_cache_active=True,
        publish_generation_snapshot=True,
    )
    generator = session._run_prefill_events(
        request=request,
        state=_RequestState(),
        yield_pause=_YieldPauseTracker(enabled=False),
    )
    while True:
        try:
            next(generator)
        except StopIteration as stopped:
            return calls, stopped.value, session


def test_glm_prefill_chunks_without_enabling_snapshots():
    from omlx.patches.dflash_glm5 import (
        _install_glm5_prefill_chunking_bridge,
        restore_glm5_dflash_class_patches,
    )

    _install_glm5_prefill_chunking_bridge()
    try:
        calls, result, session = _run_fake_prefill(backend_name="glm5_next")
        assert calls == [4, 4, 2, 1]
        assert result.supports_prefix_snapshot is False
        assert session.supports_prefix_snapshot is False
    finally:
        restore_glm5_dflash_class_patches()


def test_glm_prefill_bridge_never_enables_snapshot_hydration(monkeypatch):
    from dflash_mlx.engine import spec_epoch
    from dflash_mlx.engine.spec_epoch import SpeculativeSession
    from dflash_mlx.runtime.context import build_offline_runtime_context

    from omlx.patches.dflash_glm5 import (
        _install_glm5_prefill_chunking_bridge,
        restore_glm5_dflash_class_patches,
    )

    class TargetOps:
        @staticmethod
        def make_cache(*_args, **_kwargs):
            return []

    class DraftBackend:
        @staticmethod
        def make_cache(**_kwargs):
            return []

    draft = SimpleNamespace(
        args=SimpleNamespace(sliding_window=0, layer_types=()),
        target_layer_ids=[0],
    )
    snapshot = SimpleNamespace(prefix_len=1, token_ids=(7,))
    monkeypatch.setattr(
        spec_epoch,
        "hydrate_target_cache",
        lambda *_args, **_kwargs: pytest.fail("GLM snapshots must never hydrate"),
    )

    _install_glm5_prefill_chunking_bridge()
    try:
        session = SpeculativeSession.open(
            target_model=object(),
            draft_model=draft,
            draft_backend=DraftBackend(),
            target_ops=TargetOps(),
            supports_prefix_snapshot=False,
            allow_full_context_draft_layers=False,
            prompt_tokens=[7, 8],
            max_new_tokens=1,
            prefix_snapshot=snapshot,
            quantize_kv_cache=False,
            target_fa_window=0,
            runtime_context=build_offline_runtime_context(),
        )
        assert session.snap_prefix_len == 0
        assert session.supports_prefix_snapshot is False
    finally:
        restore_glm5_dflash_class_patches()


def test_glm_prefill_bridge_leaves_non_glm_behavior_unchanged():
    from omlx.patches.dflash_glm5 import (
        _install_glm5_prefill_chunking_bridge,
        restore_glm5_dflash_class_patches,
    )

    _install_glm5_prefill_chunking_bridge()
    try:
        calls, result, session = _run_fake_prefill(backend_name="qwen_gdn")
        assert calls == [11]
        assert result.supports_prefix_snapshot is False
        assert session.supports_prefix_snapshot is False
    finally:
        restore_glm5_dflash_class_patches()


def test_fully_accepted_cycle_clears_pooling_undo_without_changing_state():
    from mlx_lm.models.cache import CacheList, KVCache

    from omlx.patches.deepseek_v4 import apply_pooling_cache_support
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps
    from omlx.patches.mlx_lm_mtp import cache_rollback

    apply_pooling_cache_support()
    from mlx_lm.models.cache import PoolingCache

    cache = CacheList(KVCache(), PoolingCache(4))
    keys = mx.ones((1, 1, 1, 3), dtype=mx.float32)
    values = mx.zeros((1, 1, 1, 0), dtype=mx.float32)
    cache[0].update_and_fetch(keys, values)
    cache_rollback.set_undo_armed(True)
    try:
        kv = mx.ones((1, 1, 3), dtype=mx.float32)
        gate = mx.ones((1, 1, 1), dtype=mx.float32)
        cache[1].accumulate_windows(kv, gate, 0)
    finally:
        cache_rollback.set_undo_armed(False)
    state_before = cache.state
    assert cache[1]._undo is not None
    assert cache[1]._undo_chain is True

    Glm5NextTargetOps().restore_after_acceptance(
        [cache], target_len=1, acceptance_length=0, drafted_tokens=0
    )
    assert cache[1]._undo is None
    assert cache[1]._undo_chain is False
    assert cache[0].offset == 1
    for before, after in zip(state_before[0], cache[0].state, strict=True):
        assert mx.array_equal(before, after).item()


def test_glm_target_loader_prefers_omlx_custom_vlm_loader(tmp_path, monkeypatch):
    import mlx_vlm.utils as vlm_utils

    from omlx.patches import dflash_glm5
    from omlx.patches.dflash_glm5 import (
        _load_glm5_target_bundle,
        install_dflash_glm5_backend,
    )
    from omlx.utils import model_loading

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "glm5_next"}), encoding="utf-8"
    )
    target = _fake_target()
    processor = SimpleNamespace(tokenizer=object())
    seen = []

    def custom_loader(model_ref, *, is_vlm):
        seen.append(("load", model_ref, is_vlm))
        return target, processor

    monkeypatch.setattr(model_loading, "maybe_load_custom_quantization", custom_loader)
    monkeypatch.setattr(
        model_loading,
        "materialize_lazy_state",
        lambda model: seen.append(("materialize", model)),
    )
    monkeypatch.setattr(
        dflash_glm5.Glm5NextTargetOps,
        "install_speculative_hooks",
        lambda self, model: seen.append(("hooks", model)),
    )
    monkeypatch.setattr(
        vlm_utils,
        "load",
        lambda *args, **kwargs: pytest.fail("plain mlx-vlm loader must not run"),
    )
    install_dflash_glm5_backend()
    bundle = _load_glm5_target_bundle(tmp_path)
    assert bundle.model is target
    assert bundle.tokenizer is processor.tokenizer
    assert seen == [
        ("load", str(tmp_path), True),
        ("materialize", target),
        ("hooks", target),
    ]


def test_glm_target_loader_fails_before_hooks_if_materialization_fails(
    tmp_path, monkeypatch
):
    from omlx.patches import dflash_glm5
    from omlx.patches.dflash_glm5 import (
        _load_glm5_target_bundle,
        install_dflash_glm5_backend,
    )
    from omlx.utils import model_loading

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "glm5_next"}), encoding="utf-8"
    )
    target = _fake_target()
    processor = SimpleNamespace(tokenizer=object())
    hooks = []

    def fail_materialize(_model):
        raise RuntimeError("materialize failed")

    monkeypatch.setattr(
        model_loading,
        "maybe_load_custom_quantization",
        lambda *_args, **_kwargs: (target, processor),
    )
    monkeypatch.setattr(
        model_loading,
        "materialize_lazy_state",
        fail_materialize,
    )
    monkeypatch.setattr(
        dflash_glm5.Glm5NextTargetOps,
        "install_speculative_hooks",
        lambda self, model: hooks.append(model),
    )

    install_dflash_glm5_backend()
    with pytest.raises(RuntimeError, match="materialize failed"):
        _load_glm5_target_bundle(tmp_path)
    assert hooks == []


def test_recurrent_verify_hook_matches_target_and_replays_accepted_prefix():
    """The vector-gate tape path must preserve target output and KDA state."""
    from mlx_lm.models.cache import ArraysCache

    from omlx.patches.mlx_vlm_glm5_next_compat import (
        apply_mlx_vlm_glm5_next_compat_patch,
    )

    apply_mlx_vlm_glm5_next_compat_patch()
    from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache
    from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention

    from omlx.patches.dflash_glm5 import (
        Glm5NextTargetOps,
        _install_glm5_recurrent_hook,
        restore_glm5_dflash_class_patches,
    )

    config = SimpleNamespace(
        hidden_size=64,
        linear_num_heads=2,
        linear_head_dim=32,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        linear_lower_bound=-5.0,
    )
    mx.random.seed(7)
    attention = Glm5NextLinearAttention(config)
    prefix = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    verify = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)

    baseline_cache = ArraysCache(size=2)
    attention(prefix, cache=baseline_cache)
    expected = attention(verify, cache=baseline_cache)
    mx.eval(expected, *[v for v in baseline_cache.state if v is not None])

    rollback_cache = RecurrentRollbackCache(size=2, conv_kernel_size=4)
    attention(prefix, cache=rollback_cache)
    mx.eval(*[v for v in rollback_cache.state if v is not None])
    rollback_cache.arm_rollback(prefix_len=3)
    _install_glm5_recurrent_hook(attention)
    try:
        full_cache = RecurrentRollbackCache(size=2, conv_kernel_size=4)
        attention(prefix, cache=full_cache)
        full_cache.arm_rollback(prefix_len=3)
        attention(verify, cache=full_cache)
        full_state = [v for v in full_cache.state]
        Glm5NextTargetOps().restore_after_acceptance(
            [full_cache],
            target_len=7,
            acceptance_length=3,
            drafted_tokens=3,
        )
        mx.eval(
            *[v for v in full_cache.state if v is not None],
            *[v for v in full_state if v is not None],
        )
        for retained, expected_retained in zip(
            full_cache.state, full_state, strict=True
        ):
            assert mx.array_equal(retained, expected_retained).item()
        assert not hasattr(full_cache, "_omlx_glm5_verify")

        actual = attention(verify, cache=rollback_cache)
        mx.eval(actual, *[v for v in rollback_cache.state if v is not None])
        assert mx.array_equal(actual, expected).item()
        assert rollback_cache._omlx_glm5_verify is not None

        # acceptance_length=1 commits the target-owned first token plus one
        # accepted draft token. Compare rollback with a serial two-token run.
        Glm5NextTargetOps()._rollback_glm_recurrent(rollback_cache, accepted_steps=2)
        reference_cache = ArraysCache(size=2)
        attention(prefix, cache=reference_cache)
        attention(verify[:, :2], cache=reference_cache)
        mx.eval(
            *[v for v in rollback_cache.state if v is not None],
            *[v for v in reference_cache.state if v is not None],
        )
        for replayed, reference in zip(
            rollback_cache.state, reference_cache.state, strict=True
        ):
            assert mx.array_equal(replayed, reference).item()

        # A second rejection from the replayed state catches cumulative drift
        # that a one-cycle snapshot test would miss.
        verify_two = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)
        baseline_two = ArraysCache(size=2)
        attention(prefix, cache=baseline_two)
        attention(verify[:, :2], cache=baseline_two)
        expected_two = attention(verify_two, cache=baseline_two)
        rollback_cache.arm_rollback(prefix_len=5)
        actual_two = attention(verify_two, cache=rollback_cache)
        mx.eval(actual_two, expected_two)
        assert mx.array_equal(actual_two, expected_two).item()
        Glm5NextTargetOps()._rollback_glm_recurrent(rollback_cache, accepted_steps=1)

        reference_two = ArraysCache(size=2)
        attention(prefix, cache=reference_two)
        attention(verify[:, :2], cache=reference_two)
        attention(verify_two[:, :1], cache=reference_two)
        mx.eval(
            *[v for v in rollback_cache.state if v is not None],
            *[v for v in reference_two.state if v is not None],
        )
        for replayed, reference in zip(
            rollback_cache.state, reference_two.state, strict=True
        ):
            assert mx.array_equal(replayed, reference).item()
    finally:
        restore_glm5_dflash_class_patches()
