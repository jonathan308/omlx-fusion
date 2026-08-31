# SPDX-License-Identifier: Apache-2.0
"""Host-only Qwen4 scalar MTP hidden-capture contracts."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


class _IdentityLayer:
    is_linear = False

    def __init__(self, calls):
        self.calls = calls

    def __call__(
        self,
        hidden,
        _inputs,
        *,
        mask,
        cache,
        position_ids,
        gdn_sink,
        target_verify,
    ):
        del mask, cache, position_ids
        self.calls.append((gdn_sink, target_verify))
        if gdn_sink is not None:
            gdn_sink.append(("rollback",))
        return hidden


class _FinalMixer:
    def __init__(self, hidden_size, calls):
        self.hidden_size = hidden_size
        self.calls = calls

    def __call__(self, hidden, *, target_verify):
        self.calls.append(target_verify)
        return hidden[..., : self.hidden_size]


def _numpy_mtp_module(language, hidden_size=32, hc_count=2):
    """Duck-typed owner for the real Qwen4ExpMTPModule.fuse_inputs method."""

    def identity(value):
        return value

    return SimpleNamespace(
        hidden_size=hidden_size,
        hc_count=hc_count,
        pre_fc_norm_embedding=identity,
        pre_fc_norm_hidden=identity,
        fc_embedding=identity,
        fc_hidden=identity,
        fuse_inputs=lambda token_embeddings, hidden: (
            language.Qwen4ExpMTPModule.fuse_inputs(
                SimpleNamespace(
                    hidden_size=hidden_size,
                    hc_count=hc_count,
                    pre_fc_norm_embedding=identity,
                    pre_fc_norm_hidden=identity,
                    fc_embedding=identity,
                    fc_hidden=identity,
                ),
                token_embeddings,
                hidden,
            )
        ),
    )


@pytest.fixture
def scalar_harness(monkeypatch):
    """Exercise the real Qwen4 model/wrapper control flow with NumPy tensors."""

    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    layer_calls = []
    mixer_calls = []
    backbone = SimpleNamespace(
        args=SimpleNamespace(hc_count=2),
        embed_tokens=lambda inputs: np.zeros(
            (*inputs.shape, 32),
            dtype=np.float32,
        ),
        layers=[_IdentityLayer(layer_calls)],
        ssm_idx=0,
        fa_idx=0,
        hyper_connection_mixer=_FinalMixer(32, mixer_calls),
    )
    host = language.LanguageModel.__new__(language.LanguageModel)
    object.__setattr__(host, "model", backbone)

    # The real method needs only tile/isinstance from mx for this no-kernel
    # one-layer harness.  Replacing the module binding keeps the test strictly
    # host-only: no Metal graph is built or evaluated.
    monkeypatch.setattr(
        language,
        "mx",
        SimpleNamespace(array=np.ndarray, tile=np.tile),
    )
    monkeypatch.setattr(
        language,
        "_create_qwen3_5_attention_mask",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        language,
        "_create_qwen3_5_ssm_mask",
        lambda *_args, **_kwargs: None,
    )

    base_calls = []

    def base_call(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        **kwargs,
    ):
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        return_hidden = bool(kwargs.pop("return_hidden", False))
        hidden_sink = [] if capture_layer_ids is not None else None
        gdn_sink = [] if capture_layer_ids is not None else None
        base_calls.append((capture_layer_ids, hidden_sink, gdn_sink))
        mixed = language.Qwen4ExpModel.__call__(
            self.model,
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
            gdn_sink=gdn_sink,
        )
        if return_hidden:
            if hidden_sink is None:
                hidden_sink = []
            hidden_sink.append(mixed)
        return SimpleNamespace(
            logits=mixed,
            hidden_states=hidden_sink,
            gdn_states=gdn_sink,
        )

    monkeypatch.setattr(language.Qwen3_5LanguageModel, "__call__", base_call)
    return SimpleNamespace(
        language=language,
        host=host,
        backbone=backbone,
        base_calls=base_calls,
        layer_calls=layer_calls,
        mixer_calls=mixer_calls,
    )


def test_scalar_return_hidden_keeps_raw_mtp_width_without_verify_capture(
    scalar_harness,
    monkeypatch,
):
    harness = scalar_harness
    language = harness.language
    from omlx.patches.mlx_lm_mtp import prompt_priming

    prime_calls = []
    monkeypatch.setattr(prompt_priming, "capture_eligible", lambda *_args: True)
    monkeypatch.setattr(
        prompt_priming,
        "maybe_capture",
        lambda *_args: prime_calls.append(True),
    )
    harness.backbone._omlx_mtp_prime_host = weakref.ref(harness.host)

    output = harness.host(
        np.array([[7]], dtype=np.int32),
        cache=[None],
        return_hidden=True,
    )

    assert harness.base_calls == [(None, None, None)]
    assert harness.layer_calls == [(None, False)]
    assert harness.mixer_calls == [False]
    assert output.gdn_states is None
    assert len(output.hidden_states) == 1
    assert output.hidden_states[0].shape == (1, 1, 64)
    assert output.logits.shape == (1, 1, 32)
    assert prime_calls == []

    canonical_raw = []
    canonical_mixed = language.Qwen4ExpModel.__call__(
        harness.backbone,
        np.array([[7]], dtype=np.int32),
        cache=[None],
        capture_layer_ids=[],
        hidden_sink=canonical_raw,
        gdn_sink=None,
    )
    assert len(canonical_raw) == 1
    assert np.array_equal(output.hidden_states[0], canonical_raw[0])
    assert np.array_equal(output.logits, canonical_mixed)

    # This is the actual Qwen4 head input contract reached from post-init,
    # depth-0, and boundary materialization.  The old raw residual succeeds;
    # the final mixed H-wide tensor produced by a simple L>1 gate does not.
    mtp = _numpy_mtp_module(language)
    token_embeddings = np.zeros((1, 1, 32), dtype=np.float32)
    fused = mtp.fuse_inputs(token_embeddings, output.hidden_states[0])
    assert fused.shape == (1, 1, 64)
    with pytest.raises(ValueError, match=r"hc_count \* hidden_size"):
        mtp.fuse_inputs(token_embeddings, output.logits)


def test_multitoken_and_explicit_scalar_verify_capture_are_unchanged(
    scalar_harness,
):
    harness = scalar_harness

    multi = harness.host(
        np.array([[7, 8]], dtype=np.int32),
        cache=[None],
        return_hidden=True,
    )
    assert harness.base_calls[-1][0] == []
    assert harness.layer_calls[-1][1] is True
    assert multi.gdn_states == [("rollback",)]
    assert len(multi.hidden_states) == 1
    assert multi.hidden_states[0].shape == (1, 2, 64)

    explicit = harness.host(
        np.array([[9]], dtype=np.int32),
        cache=[None],
        return_hidden=True,
        capture_layer_ids=[],
    )
    assert harness.base_calls[-1][0] == []
    assert harness.layer_calls[-1][1] is True
    assert explicit.gdn_states == [("rollback",)]
    assert [value.shape[-1] for value in explicit.hidden_states] == [64, 32]


def test_post_init_passes_raw_scalar_hidden_into_actual_mtp_forward(
    scalar_harness,
    monkeypatch,
):
    """The real post-init seam reaches the real Qwen4 head width contract."""

    import mlx.core as real_mx

    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    harness = scalar_harness
    language = harness.language
    numpy_mtp = _numpy_mtp_module(language)
    observed = []

    class NumpyHead:
        layers = []

        def __call__(self, hidden, next_ids, embed_tokens, cache):
            del cache
            observed.append(hidden.shape)
            fused = numpy_mtp.fuse_inputs(embed_tokens(next_ids), hidden)
            return fused[..., :32], fused

    object.__setattr__(harness.host, "mtp", NumpyHead())
    object.__setattr__(
        harness.host,
        "args",
        SimpleNamespace(tie_word_embeddings=False),
    )
    object.__setattr__(
        harness.host,
        "lm_head",
        lambda hidden: np.zeros((*hidden.shape[:-1], 16), dtype=np.float32),
    )
    object.__setattr__(harness.host, "_omlx_mtp_chain", True)
    object.__setattr__(harness.host, "_omlx_mtp_depth", 1)

    monkeypatch.setattr(bg, "_ensure_uint32", lambda value: value)
    monkeypatch.setattr(bg, "_mtp_logprobs", lambda _batch, value: value)
    monkeypatch.setattr(
        bg,
        "_mtp_sample",
        lambda _batch, _sampler, _value: np.array([8], dtype=np.uint32),
    )
    monkeypatch.setattr(
        bg,
        "_materialize_distributed_hidden_sibling",
        lambda *_args, **_kwargs: False,
    )
    # _post_init_mtp's two evals are synchronization only.  The harness uses
    # already-materialized NumPy arrays, so a no-op preserves the host-only
    # test while leaving the production path untouched.
    monkeypatch.setattr(real_mx, "eval", lambda *_args, **_kwargs: None)

    def chain_next_drafts(gen_batch, state, hidden_rows, committed, _prev_buf):
        logits, head_hidden = gen_batch.model.mtp_forward(
            hidden_rows,
            committed.reshape(1, -1),
            state.mtp_cache,
            return_hidden=True,
        )
        assert logits.shape == (1, 1, 16)
        assert head_hidden.shape == (1, 1, 64)
        state.hist_offset += int(committed.shape[0])
        state.drafts = np.array([9], dtype=np.uint32)

    monkeypatch.setattr(bg, "_chain_next_drafts", chain_next_drafts)

    batch = SimpleNamespace(
        model=harness.host,
        prompt_cache=[None],
        uids=[17],
        samplers=[None],
        fallback_sampler=lambda values: values,
        logits_processors=[],
        tokens=[[2, 3, 4]],
        _next_tokens=np.array([7], dtype=np.uint32),
        _next_logprobs=[np.zeros((32,), dtype=np.float32)],
        _token_context=[None],
    )
    bg._post_init_mtp(batch)

    state = batch._omlx_mtp_state
    assert observed == [(1, 1, 64)]
    assert state.hist_offset == 1
    assert [item[0] for item in state.queue] == [7, 8]


def test_scalar_capture_context_is_restored_after_backbone_exception(
    scalar_harness,
    monkeypatch,
):
    harness = scalar_harness
    language = harness.language
    outer_sink = ["outer"]
    outer_token = language._SCALAR_MTP_HIDDEN_SINK.set(outer_sink)

    def fail(*_args, **_kwargs):
        assert language._SCALAR_MTP_HIDDEN_SINK.get() == []
        raise RuntimeError("controlled scalar failure")

    monkeypatch.setattr(language.Qwen3_5LanguageModel, "__call__", fail)
    try:
        with pytest.raises(RuntimeError, match="controlled scalar failure"):
            harness.host(
                np.array([[7]], dtype=np.int32),
                cache=[None],
                return_hidden=True,
            )
        assert language._SCALAR_MTP_HIDDEN_SINK.get() is outer_sink
    finally:
        language._SCALAR_MTP_HIDDEN_SINK.reset(outer_token)
    assert language._SCALAR_MTP_HIDDEN_SINK.get() is None
