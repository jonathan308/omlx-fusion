# SPDX-License-Identifier: Apache-2.0
"""Qwen4 scalar-row target output-head diagnostic contracts."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from test_mlx_vlm_qwen4_exp_compat import _tiny_config

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import language  # noqa: E402


def _model(*, tied: bool = False):
    config = _tiny_config()
    config.text_config.tie_word_embeddings = tied
    model = language.LanguageModel(config.text_config, config)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("tied", [False, True], ids=["lm-head", "tied-embedding"])
@pytest.mark.parametrize("width", [2, 6, 9])
def test_qwen4_tokenwise_lm_head_matches_exact_scalar_projection(tied, width):
    model = _model(tied=tied)
    mx.random.seed(20260831 + width + int(tied))
    hidden = mx.random.normal((1, width, model.args.hidden_size)).astype(
        mx.bfloat16
    )
    projection = (
        model.model.embed_tokens.as_linear if tied else model.lm_head
    )
    expected = mx.concatenate(
        [projection(hidden[:, row : row + 1]) for row in range(width)],
        axis=1,
    )
    actual = language._tokenwise_lm_head_projection(model, hidden)
    mx.eval(expected, actual)

    assert actual.shape == (1, width, model.args.vocab_size)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("tied", [False, True], ids=["lm-head", "tied-embedding"])
def test_qwen4_tokenwise_lm_head_preserves_quantized_full_logprobs(tied):
    model = _model(tied=tied)
    if tied:
        model.model.embed_tokens = nn.QuantizedEmbedding.from_embedding(
            model.model.embed_tokens,
            group_size=32,
            bits=4,
        )
        assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
        projection = model.model.embed_tokens.as_linear
    else:
        model.lm_head = nn.QuantizedLinear.from_linear(
            model.lm_head,
            group_size=32,
            bits=4,
        )
        assert isinstance(model.lm_head, nn.QuantizedLinear)
        projection = model.lm_head

    mx.random.seed(20260902 + int(tied))
    width = 6
    hidden = mx.random.normal((1, width, model.args.hidden_size)).astype(
        mx.bfloat16
    )
    expected = mx.concatenate(
        [projection(hidden[:, row : row + 1]) for row in range(width)],
        axis=1,
    )
    actual = language._tokenwise_lm_head_projection(model, hidden)
    logprobs = actual.astype(mx.float32) - mx.logsumexp(
        actual.astype(mx.float32),
        axis=-1,
        keepdims=True,
    )
    probability_mass = mx.exp(logprobs).sum(axis=-1)
    mx.eval(actual, expected, probability_mass)

    assert actual.shape == (1, width, model.args.vocab_size)
    assert logprobs.shape == (1, width, model.args.vocab_size)
    assert mx.array_equal(actual, expected).item()
    assert mx.allclose(
        probability_mass,
        mx.ones((1, width), dtype=mx.float32),
        rtol=1e-5,
        atol=1e-5,
    ).item()


@pytest.mark.parametrize("tied", [False, True], ids=["lm-head", "tied-embedding"])
def test_qwen4_target_verify_replaces_full_logits_through_scoped_sink(
    monkeypatch,
    tied,
):
    model = _model(tied=tied)
    width = 6
    tokens = mx.arange(2, 2 + width, dtype=mx.int32)[None]
    sentinel = mx.full((1, width, model.args.vocab_size), 7, dtype=mx.float32)
    calls = []

    def replacement(owner, hidden):
        calls.append((owner, hidden.shape))
        return sentinel

    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_LM_HEAD_VERIFY", "1")
    monkeypatch.setattr(language, "_tokenwise_lm_head_projection", replacement)
    output = model(tokens, cache=model.make_cache(), return_hidden=True)
    mx.eval(output.logits)

    assert calls == [(model, (1, width, model.args.hidden_size))]
    assert mx.array_equal(output.logits, sentinel).item()
    assert language._TOKENWISE_LM_HEAD_HIDDEN_SINK.get() is None


@pytest.mark.parametrize(
    ("env", "batch", "width", "return_hidden", "skip_logits"),
    [
        (None, 1, 2, True, False),
        ("0", 1, 2, True, False),
        ("1", 1, 1, True, False),
        ("1", 1, 10, True, False),
        ("1", 2, 2, True, False),
        ("1", 1, 2, False, False),
        ("1", 1, 2, True, True),
    ],
)
def test_qwen4_tokenwise_lm_head_is_strictly_gated(
    monkeypatch,
    env,
    batch,
    width,
    return_hidden,
    skip_logits,
):
    model = _model()
    tokens = mx.zeros((batch, width), dtype=mx.int32)
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("ineligible tokenwise LM-head route was called")

    monkeypatch.setattr(language, "_tokenwise_lm_head_projection", forbidden)
    if env is None:
        monkeypatch.delenv("OMLX_QWEN4_TOKENWISE_LM_HEAD_VERIFY", raising=False)
    else:
        monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_LM_HEAD_VERIFY", env)
    output = model(
        tokens,
        cache=model.make_cache(),
        return_hidden=return_hidden,
        skip_logits=skip_logits,
    )
    if output.logits is not None:
        mx.eval(output.logits)

    assert calls == []
    assert output.logits is None if skip_logits else output.logits is not None
    assert language._TOKENWISE_LM_HEAD_HIDDEN_SINK.get() is None


def test_qwen4_tokenwise_lm_head_failure_retains_default_logits(monkeypatch):
    model = _model(tied=True)
    tokens = mx.array([[2, 3, 4, 5, 6, 7]], dtype=mx.int32)
    baseline = model(tokens, cache=model.make_cache(), return_hidden=True).logits
    mx.eval(baseline)

    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_LM_HEAD_VERIFY", "1")
    monkeypatch.setattr(
        language,
        "_tokenwise_lm_head_projection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic scalar projection failure")
        ),
    )
    fallback = model(tokens, cache=model.make_cache(), return_hidden=True).logits
    mx.eval(fallback)

    assert mx.array_equal(fallback, baseline).item()
    assert language._TOKENWISE_LM_HEAD_HIDDEN_SINK.get() is None


def test_qwen4_tokenwise_lm_head_restores_nested_context(monkeypatch):
    model = _model()
    tokens = mx.array([[2, 3]], dtype=mx.int32)
    outer_sink = []
    outer_token = language._TOKENWISE_LM_HEAD_HIDDEN_SINK.set(outer_sink)
    monkeypatch.setenv("OMLX_QWEN4_TOKENWISE_LM_HEAD_VERIFY", "1")
    try:
        output = model(tokens, cache=model.make_cache(), return_hidden=True)
        mx.eval(output.logits)
        assert language._TOKENWISE_LM_HEAD_HIDDEN_SINK.get() is outer_sink
        assert outer_sink == []
    finally:
        language._TOKENWISE_LM_HEAD_HIDDEN_SINK.reset(outer_token)
