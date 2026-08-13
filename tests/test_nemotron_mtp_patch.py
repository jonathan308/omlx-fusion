from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nh = pytest.importorskip("mlx_lm.models.nemotron_h")

from omlx.patches.mlx_lm_mtp import (  # noqa: E402
    nemotron_h_chain,
    nemotron_h_model,
    set_mtp_active,
)

TINY_CONFIG = {
    "model_type": "nemotron_h",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "max_position_embeddings": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "attention_bias": False,
    "mamba_num_heads": 4,
    "mamba_head_dim": 16,
    "mamba_proj_bias": False,
    "ssm_state_size": 32,
    "conv_kernel": 4,
    "n_groups": 2,
    "mlp_bias": False,
    "layer_norm_epsilon": 1e-5,
    "use_bias": False,
    "use_conv_bias": True,
    "hybrid_override_pattern": ["M", "*"],
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "moe_shared_expert_intermediate_size": 32,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "num_nextn_predict_layers": 1,
}


@pytest.fixture(autouse=True)
def _apply_patches():
    assert nemotron_h_model.apply()
    assert nemotron_h_chain.apply()
    yield
    set_mtp_active(False)


class TestLoaderGate:
    def test_nemotron_h_is_mtp_compatible(self):
        # The stock loader must route nemotron_h through the MTP patch;
        # without this gate the whole feature is inert on a stock server.
        from omlx.utils.model_loading import _is_mtp_compatible

        assert _is_mtp_compatible({"num_nextn_predict_layers": 1}, "nemotron_h")
        assert not _is_mtp_compatible({}, "nemotron_h")


class TestApply:
    def test_idempotent(self):
        mixer_call = nh.NemotronHMamba2Mixer.__call__
        assert nemotron_h_model.apply()
        assert nemotron_h_chain.apply()
        assert nh.NemotronHMamba2Mixer.__call__ is mixer_call

    def test_markers(self):
        assert getattr(nh.NemotronHMamba2Mixer.__call__, "_omlx_nh_chain", False)
        assert getattr(nh.Model.mtp_forward, "_omlx_nh_chain", False)
        assert callable(getattr(nh.Model, "mtp_partial_rollback", None))


class TestModelStamps:
    def test_mtp_attached_and_flagged_when_active(self):
        set_mtp_active(True)
        model = nh.Model(nh.ModelArgs.from_dict(TINY_CONFIG))
        assert hasattr(model, "mtp")
        assert model._omlx_mtp_decode_enabled
        assert model._omlx_mtp_chain
        assert model._omlx_mtp_head_hidden_normed

    def test_no_mtp_when_inactive(self):
        set_mtp_active(False)
        model = nh.Model(nh.ModelArgs.from_dict(TINY_CONFIG))
        assert not hasattr(model, "mtp")
        assert not model._omlx_mtp_decode_enabled

    def test_mtp_forward_return_hidden(self):
        set_mtp_active(True)
        model = nh.Model(nh.ModelArgs.from_dict(TINY_CONFIG))
        hidden = mx.zeros((1, 1, TINY_CONFIG["hidden_size"]))
        ids = mx.zeros((1, 1), dtype=mx.uint32)
        cache = model.make_mtp_cache()
        logits, head_hidden = model.mtp_forward(
            hidden, ids, cache, return_hidden=True, logits_keep=1
        )
        mx.eval(logits, head_hidden)
        assert logits.shape == (1, 1, TINY_CONFIG["vocab_size"])
        assert head_hidden.shape == hidden.shape


class TestVerifyCapture:
    def _mixer_and_cache(self):
        from mlx_lm.models.cache import ArraysCache

        args = nh.ModelArgs.from_dict(TINY_CONFIG)
        mixer = nh.NemotronHMamba2Mixer(args)
        mx.eval(mixer.parameters())
        return mixer, ArraysCache(size=2)

    def test_per_position_restore_matches_prefix_recompute(self):
        mx.random.seed(0)
        mixer, cache = self._mixer_and_cache()
        prefix = mx.random.normal((1, 3, TINY_CONFIG["hidden_size"]))
        window = mx.random.normal((1, 4, TINY_CONFIG["hidden_size"]))
        mx.eval(prefix, window)

        # Establish state, then run a verify window with capture armed.
        mixer(prefix, None, cache)
        mixer(window, None, cache, n_confirmed=1)
        assert cache._mtp_pos_states is not None
        assert len(cache._mtp_pos_states) == 4

        # Restore to keep=2 (confirmed + 1 accepted draft).
        conv_m, ssm_m = cache._mtp_pos_states[1]

        # Reference: fresh cache, prefix + the kept 2 window tokens.
        mixer2, cache2 = self._mixer_and_cache()
        mixer2.update(mixer.parameters())
        mx.eval(mixer2.parameters())
        mixer2(prefix, None, cache2)
        mixer2(window[:, :2], None, cache2)
        mx.eval(conv_m, ssm_m, cache2[0], cache2[1])

        assert mx.allclose(conv_m, cache2[0], atol=1e-5).item()
        assert mx.allclose(ssm_m, cache2[1], atol=1e-4, rtol=1e-3).item()


# ---------------------------------------------------------------------------
# Pipeline-parallel awareness (omlx/cluster worker)
# ---------------------------------------------------------------------------

PIPE_CONFIG = {
    **TINY_CONFIG,
    "num_hidden_layers": 4,
    "hybrid_override_pattern": ["M", "*", "M", "*"],
}


class _Group:
    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _stub_collectives(monkeypatch, calls=None):
    """Let a single process run one rank's forward without a peer.

    Only the transport is faked; every line of the forward under test runs.
    """

    def record(name):
        def stub(x, *_a, **_k):
            if calls is not None:
                calls.append(name)
            return x

        return stub

    monkeypatch.setattr(mx.distributed, "send", record("send"))
    monkeypatch.setattr(mx.distributed, "recv_like", record("recv"))
    monkeypatch.setattr(mx.distributed, "all_gather", record("all_gather"))


def _apply_stage_assignment(backbone, rank, size, start, end):
    """Mimic planner.apply_pipeline_assignment + the compat hook's stage-local
    fa/ssm recompute, without entering the hook's context."""
    backbone.pipeline_rank = rank
    backbone.pipeline_size = size
    backbone.start_idx = start
    backbone.end_idx = end
    backbone.layers = backbone.layers[:end]
    backbone.layers[:start] = [None] * start
    # Stage-local cache indices over the owned M/* blocks (the cache list
    # holds entries for M and * blocks only, in layer order).
    fa_idx = ssm_idx = None
    cache_index = 0
    for layer in backbone.layers[start:end]:
        if layer.block_type == "*" and fa_idx is None:
            fa_idx = cache_index
        if layer.block_type == "M" and ssm_idx is None:
            ssm_idx = cache_index
        if layer.block_type in ("M", "*"):
            cache_index += 1
    backbone.fa_idx = fa_idx
    backbone.ssm_idx = ssm_idx


def _stage_cache():
    from mlx_lm.models.cache import ArraysCache, KVCache

    # Stage [2, 4) of ["M", "*", "M", "*"] owns [M, *] -> two cache entries.
    return [ArraysCache(size=2), KVCache()]


class TestPipelineContract:
    """The MTP patch's backbone ``__call__`` must carry the same collectives
    as the cluster worker's pipeline hook (recv -> stage-local layers -> send
    -> all_gather) AND the MTP kwargs — the worker loads the model INSIDE the
    hook's context, so the patch's self-healing re-apply can install this body
    over the hook's; without the collectives that ordering silently broke
    pipeline serving, and before this fix the hook's body in turn dropped
    ``n_confirmed`` (one always clobbered the other's contract)."""

    def test_installed_backbone_body_carries_pipeline_primitives(self):
        import inspect

        src = inspect.getsource(nh.NemotronHModel.__call__)
        assert "recv_like" in src
        assert "all_gather" in src
        assert "start_idx" in src

    def test_stage_forward_runs_collectives_and_stage_layers_only(
        self, monkeypatch
    ):
        model = nh.Model(nh.ModelArgs.from_dict(PIPE_CONFIG))
        mx.eval(model.parameters())
        backbone = model.backbone
        # Rank 0 of 2 owns the LAST layers: the old body TypeError'd on the
        # first None sentinel and never issued a collective.
        _apply_stage_assignment(backbone, rank=0, size=2, start=2, end=4)

        calls = []
        _stub_collectives(monkeypatch, calls)
        cache = _stage_cache()
        hidden = backbone(mx.array([[1, 2, 3]], dtype=mx.uint32), cache=cache)
        mx.eval(hidden)

        # Rank 0 receives from rank 1 and joins the broadcast; only rank != 0
        # sends (mlx-lm's protocol places the first stage on the last rank).
        assert calls == ["recv", "all_gather"]
        assert hidden.shape == (1, 3, PIPE_CONFIG["hidden_size"])

    def test_stage_forward_still_threads_n_confirmed(self, monkeypatch):
        """MTP kwargs must survive on the pipeline-carrying body: a verify
        window snapshots the Mamba state on the stage-local cache."""
        model = nh.Model(nh.ModelArgs.from_dict(PIPE_CONFIG))
        mx.eval(model.parameters())
        _apply_stage_assignment(model.backbone, rank=0, size=2, start=2, end=4)
        _stub_collectives(monkeypatch)

        cache = _stage_cache()
        model(mx.array([[4, 5, 6, 7]], dtype=mx.uint32), cache=cache)
        model(mx.array([[8, 9, 10]], dtype=mx.uint32), cache=cache, n_confirmed=1)
        assert cache[0].rollback_state is not None

    def test_single_node_attempts_no_collectives(self, monkeypatch):
        """No pipeline attributes: rank 0 / size 1 / the full layer list —
        byte-identical to the pre-pipeline body, and no peer is ever called."""
        model = nh.Model(nh.ModelArgs.from_dict(TINY_CONFIG))
        mx.eval(model.parameters())

        calls = []
        _stub_collectives(monkeypatch, calls)
        cache = model.make_cache()
        logits = model(mx.array([[1, 2, 3, 4]], dtype=mx.uint32), cache=cache)
        mx.eval(logits)

        assert calls == [], f"single node attempted {calls}"
        assert logits.shape == (1, 4, TINY_CONFIG["vocab_size"])
        assert bool(mx.all(mx.isfinite(logits)))

    def test_single_node_output_matches_pre_pipeline_body(self):
        """Parity: the pipeline-carrying backbone must be byte-identical
        single-node to the body it replaced (commit e2c05d80)."""
        model = nh.Model(nh.ModelArgs.from_dict(TINY_CONFIG))
        mx.eval(model.parameters())
        tokens = mx.array([[3, 1, 4, 1, 5]], dtype=mx.uint32)

        def pre_pipeline_call(self, inputs, cache=None, n_confirmed=0):
            # Verbatim pre-fix body, reinstalled as the reference below.
            h = self.embeddings(inputs)
            if cache is None:
                cache = [None] * len(self.layers)
            attn_mask = nh.create_attention_mask(h, cache[self.fa_idx])
            ssm_mask = nh.create_ssm_mask(h, cache[self.ssm_idx])
            cc = 0
            for layer in self.layers:
                if layer.block_type in ("M", "*"):
                    c = cache[cc]
                    cc += 1
                else:
                    c = None
                mask = attn_mask if layer.block_type == "*" else ssm_mask
                if layer.block_type == "M":
                    h = layer(h, mask=mask, cache=c, n_confirmed=n_confirmed)
                else:
                    h = layer(h, mask=mask, cache=c)
            return self.norm_f(h)

        cls = nh.NemotronHModel
        patched_call = cls.__call__
        cls.__call__ = pre_pipeline_call
        try:
            reference = model.backbone(tokens, cache=model.make_cache())
        finally:
            cls.__call__ = patched_call
        current = model.backbone(tokens, cache=model.make_cache())
        mx.eval(reference, current)
        assert mx.array_equal(reference, current)


class TestPipelineComposition:
    """The worker's ``_install_nemotron_h_pipeline`` hook and this patch
    replace the same ``NemotronHModel.__call__`` and install in either order
    (the worker loads the model INSIDE the hook's context). Whichever wins,
    the active body must satisfy BOTH contracts."""

    @staticmethod
    def _assignments():
        from omlx.cluster.planner import PipelineAssignment

        def make(rank, start, end):
            return PipelineAssignment(
                node_id=f"n{rank}",
                rank=rank,
                start_layer=start,
                end_layer=end,
                layer_weight_bytes=0,
                fixed_weight_bytes=0,
                reserve_bytes=0,
                capacity_bytes=0,
            )

        # mlx-lm convention: rank 0 holds the LAST layers.
        return [make(0, 2, 4), make(1, 0, 2)]

    def test_hook_call_accepts_mtp_kwargs_and_patch_reapply_keeps_collectives(
        self, monkeypatch
    ):
        from omlx.cluster.pipeline_compat import _install_nemotron_h_pipeline

        set_mtp_active(True)
        model = nh.Model(nh.ModelArgs.from_dict(PIPE_CONFIG))
        mx.eval(model.parameters())
        calls = []
        _stub_collectives(monkeypatch, calls)

        with _install_nemotron_h_pipeline(self._assignments()):
            # Order 1: the hook installed over the MTP patch — its
            # pipeline_call now owns the backbone forward.
            assert not getattr(nh.NemotronHModel.__call__, "_omlx_nh_mtp", False)
            model.model.pipeline(_Group(0, 2))
            backbone = model.backbone
            assert (backbone.start_idx, backbone.end_idx) == (2, 4)
            assert (backbone.ssm_idx, backbone.fa_idx) == (0, 1)

            cache = model.make_cache()  # stage-local M/* entries only
            assert len(cache) == 2
            model(mx.array([[4, 5, 6, 7]], dtype=mx.uint32), cache=cache)
            # n_confirmed through the hook's body must reach the Mamba mixer.
            model(mx.array([[8, 9, 10]], dtype=mx.uint32), cache=cache, n_confirmed=1)
            assert cache[0].rollback_state is not None
            assert calls == ["recv", "all_gather"] * 2

            # Order 2: the patch's self-healing re-apply installs its own body
            # over the hook's — it must keep the pipeline collectives.
            assert nemotron_h_model.apply()
            assert getattr(nh.NemotronHModel.__call__, "_omlx_nh_mtp", False)
            calls.clear()
            cache = model.make_cache()
            model(mx.array([[4, 5, 6, 7]], dtype=mx.uint32), cache=cache)
            model(mx.array([[8, 9, 10]], dtype=mx.uint32), cache=cache, n_confirmed=1)
            assert cache[0].rollback_state is not None
            assert calls == ["recv", "all_gather"] * 2

        # Context exit restores what was there before — the MTP body.
        assert getattr(nh.NemotronHModel.__call__, "_omlx_nh_mtp", False)

    def test_hook_body_declares_n_confirmed(self):
        """Source contract on the worker hook: a regression that drops the
        kwarg there re-breaks order 1 even if the patch's body keeps it."""
        import inspect

        from omlx.cluster.pipeline_compat import _install_nemotron_h_pipeline

        with _install_nemotron_h_pipeline(self._assignments()):
            src = inspect.getsource(nh.NemotronHModel.__call__)
        assert "n_confirmed" in src


class TestPartialRollbackStageLocal:
    """``Model.mtp_partial_rollback`` walks layers to pair each cache with its
    block type. Under pipeline the full ``backbone.layers`` carries None
    sentinels before ``start_idx`` (AttributeError on the old code) while the
    cache list holds only this rank's M/* entries — the walk must use the
    stage-local view."""

    def test_rollback_on_rank_with_start_idx_gt_zero(self, monkeypatch):
        from mlx_lm.models.cache import ArraysCache

        set_mtp_active(True)
        model = nh.Model(nh.ModelArgs.from_dict(PIPE_CONFIG))
        mx.eval(model.parameters())
        _apply_stage_assignment(model.backbone, rank=0, size=2, start=2, end=4)
        _stub_collectives(monkeypatch)

        prefill = mx.array([[4, 5, 6, 7]], dtype=mx.uint32)
        window = mx.array([[8, 9, 10]], dtype=mx.uint32)  # confirmed + 2 drafts

        # Cache A: prefill + a verify window (snapshots Mamba state).
        cache_a = _stage_cache()
        model(prefill, cache=cache_a)
        model(window, cache=cache_a, n_confirmed=1)
        mamba_a, kv_a = cache_a
        assert kv_a.offset == 4 + 3
        assert mamba_a.rollback_state is not None

        # Cache B: prefill + only the kept prefix (the confirmed token).
        cache_b = _stage_cache()
        model(prefill, cache=cache_b)
        model(window[:, :1], cache=cache_b)

        assert model.mtp_partial_rollback(cache_a, accepted=0, num_drafts=2) is True

        # KV trims the two rejected drafts; the Mamba state is restored
        # exactly to having run the kept prefix (cache B).
        assert kv_a.offset == cache_b[1].offset == 4 + 1
        assert isinstance(mamba_a, ArraysCache)
        mx.eval(mamba_a[0], mamba_a[1], cache_b[0][0], cache_b[0][1])
        assert mx.allclose(mamba_a[0], cache_b[0][0], atol=1e-5).item()
        assert mx.allclose(mamba_a[1], cache_b[0][1], atol=1e-4, rtol=1e-3).item()
        assert mamba_a.rollback_state is None
        assert mamba_a._mtp_draft_stash is None
        assert mamba_a._mtp_pos_states is None
