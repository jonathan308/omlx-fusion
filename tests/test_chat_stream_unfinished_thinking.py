# SPDX-License-Identifier: Apache-2.0
"""Regression contract for EOS inside a prompt-opened thinking block."""

import json
from types import SimpleNamespace

import pytest

from omlx import server
from omlx.api.openai_models import ChatCompletionRequest, Message
from omlx.api.responses_models import ResponsesRequest


class _ThinkingTokenizer:
    think_start_id = 41
    think_end_id = 42
    eos_token_id = 2

    def apply_chat_template(self, _messages, **_kwargs):
        return "<think>"

    def encode(self, _text, add_special_tokens=False):
        return [self.think_start_id]


class _ReasoningOnlyEngine:
    tokenizer = _ThinkingTokenizer()
    model_type = "qwen4_exp"
    is_diffusion_model = False

    async def stream_chat(self, **_kwargs):
        yield SimpleNamespace(
            new_text='The user just said "Right!"',
            text='The user just said "Right!"',
            tool_calls=None,
            finish_reason="stop",
            finished=True,
            prompt_tokens=20,
            completion_tokens=7,
            cached_tokens=19,
        )


class _PlainTokenizer(_ThinkingTokenizer):
    def apply_chat_template(self, _messages, **_kwargs):
        return "assistant:"

    def encode(self, _text, add_special_tokens=False):
        return [11]


class _PlainEngine(_ReasoningOnlyEngine):
    tokenizer = _PlainTokenizer()

    async def stream_chat(self, **_kwargs):
        yield SimpleNamespace(
            new_text="hello",
            text="hello",
            tool_calls=None,
            finish_reason="stop",
            finished=True,
            prompt_tokens=10,
            completion_tokens=1,
            cached_tokens=0,
        )


def _sse_payload(event: str) -> dict | None:
    if not event.startswith("data: ") or event == "data: [DONE]\n\n":
        return None
    return json.loads(event[6:])


@pytest.mark.asyncio
async def test_unfinished_reasoning_never_becomes_content_and_is_incomplete():
    request = ChatCompletionRequest(
        model="qwen4-exp-test",
        messages=[Message(role="user", content="Right!")],
        stream=True,
    )
    events = [
        event
        async for event in server.stream_chat_completion(
            _ReasoningOnlyEngine(),
            [{"role": "user", "content": "Right!"}],
            request,
        )
    ]
    payloads = [payload for event in events if (payload := _sse_payload(event))]

    deltas = [choice["delta"] for payload in payloads for choice in payload["choices"]]
    reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
    content = "".join(delta.get("content", "") for delta in deltas)
    finish_reasons = [
        choice.get("finish_reason")
        for payload in payloads
        for choice in payload["choices"]
        if choice.get("finish_reason") is not None
    ]

    assert reasoning == 'The user just said "Right!"'
    assert content == ""
    assert finish_reasons == ["length"]


@pytest.mark.asyncio
async def test_responses_finalizer_never_reparses_unfinished_reasoning_as_text():
    request = ResponsesRequest(model="qwen4-exp-test", input="Right!", stream=True)
    events = [
        event
        async for event in server.stream_responses_api(
            _ReasoningOnlyEngine(),
            [{"role": "user", "content": "Right!"}],
            request,
            store_response=False,
        )
    ]
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[6:]))

    reasoning = "".join(
        payload.get("delta", "")
        for payload in payloads
        if payload.get("type") == "response.reasoning_summary_text.delta"
    )
    content_deltas = "".join(
        payload.get("delta", "")
        for payload in payloads
        if payload.get("type") == "response.output_text.delta"
    )
    done_text = next(
        payload["text"]
        for payload in payloads
        if payload.get("type") == "response.output_text.done"
    )
    incomplete = next(
        payload["response"]
        for payload in payloads
        if payload.get("type") == "response.incomplete"
    )

    assert reasoning == 'The user just said "Right!"'
    assert content_deltas == ""
    assert done_text == ""
    assert incomplete["status"] == "incomplete"
    assert incomplete["output"][-1]["content"][0]["text"] == ""


@pytest.mark.asyncio
async def test_responses_actual_prompt_overrides_native_reasoning_capability():
    """enable_thinking=False remains a normal visible response."""
    request = ResponsesRequest(model="qwen4-exp-test", input="Hi", stream=True)
    events = [
        event
        async for event in server.stream_responses_api(
            _PlainEngine(),
            [{"role": "user", "content": "Hi"}],
            request,
            store_response=False,
            native_reasoning=True,
        )
    ]
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[6:]))

    assert not any(
        payload.get("type") == "response.reasoning_summary_text.delta"
        for payload in payloads
    )
    assert next(
        payload["text"]
        for payload in payloads
        if payload.get("type") == "response.output_text.done"
    ) == "hello"
    assert any(payload.get("type") == "response.completed" for payload in payloads)
