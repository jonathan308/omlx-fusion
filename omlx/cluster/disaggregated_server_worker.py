# SPDX-License-Identifier: Apache-2.0
"""Persistent OpenAI-compatible full-replica prefill/decode rank worker."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from queue import Empty, Queue
import struct
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterator

from .cache_transfer import recv_cache_transfer, send_cache_transfer, prepare_cache_transfer


_PROGRESS = struct.Struct("!II")
_LOGITS_HEADER = struct.Struct("!III")
_DTYPE_TO_CODE = {"float16": 1, "bfloat16": 2, "float32": 3}
_CODE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CODE.items()}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("ring", "jaccl", "jaccl-ring"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--plan-hash", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--state-dir", default="~/.omlx/cluster/runtime")
    parser.add_argument("--control-host", required=True)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--control-token", required=True)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--prompt-cache-size", type=int, default=1)
    parser.add_argument("--decode-concurrency", type=int, default=1)
    parser.add_argument("--prompt-concurrency", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    args, _unknown = parser.parse_known_args()
    return args


def _dtype_name(dtype: Any) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _cache_states(cache: list[Any]) -> list[Any]:
    return [entry.state for entry in cache]


def _prefill_calls(prompt_tokens: int, step: int) -> int:
    return (max(0, prompt_tokens - 2) // step) + 2


def _prefill_logits(
    mx: Any,
    model: Any,
    cache: list[Any],
    tokens: list[int],
    *,
    step: int,
    progress: Any,
) -> tuple[Any, float]:
    started = time.perf_counter()
    values = mx.array(tokens, dtype=mx.int32)
    processed = 0
    total = len(tokens)
    while total - processed > 1:
        width = min(step, (total - processed) - 1)
        _ = model(values[None, processed : processed + width], cache=cache)
        mx.eval(_cache_states(cache))
        processed += width
        progress(processed, total)
        mx.clear_cache()
    logits = model(values[None, processed:], cache=cache)[:, -1, :]
    mx.eval(logits, _cache_states(cache))
    progress(total, total)
    return logits, time.perf_counter() - started


def _state_machine(tokenizer: Any, stop_words: list[str]):
    from mlx_lm.generate import SequenceStateMachine

    transitions: dict[str, list[tuple[tuple[int, ...], str | None]]] = {
        "normal": []
    }
    sequences: dict[tuple[int, ...], str] = {}
    common = []
    for token in tokenizer.eos_token_ids:
        sequence = (int(token),)
        common.append((sequence, None))
        sequences[sequence] = tokenizer.convert_ids_to_tokens(int(token))
    for word in stop_words:
        sequence = tuple(tokenizer.encode(word, add_special_tokens=False))
        if sequence:
            common.append((sequence, None))
            sequences[sequence] = word
    transitions["normal"].extend(common)
    if tokenizer.has_thinking:
        start = tuple(tokenizer.think_start_tokens)
        end = tuple(tokenizer.think_end_tokens)
        transitions["normal"].append((start, "reasoning"))
        transitions["reasoning"] = [(end, "normal"), *common]
        sequences[start] = tokenizer.think_start
        sequences[end] = tokenizer.think_end
    if tokenizer.has_tool_calling:
        start = tuple(tokenizer.tool_call_start_tokens)
        end = tuple(tokenizer.tool_call_end_tokens or ())
        transitions["normal"].append((start, "tool"))
        transitions["tool"] = ([(end, "normal")] if end else []) + common
        sequences[start] = tokenizer.tool_call_start
        if end:
            sequences[end] = tokenizer.tool_call_end
    return SequenceStateMachine(transitions, initial="normal"), sequences


@dataclass
class _ServingRequest:
    prompt: list[int]
    args: Any
    context: Any
    state_machine: Any
    output: Queue
    progress: Any


class _PhaseResponseGenerator:
    """APIHandler adapter and online two-stage request broker."""

    def __init__(
        self,
        *,
        mx: Any,
        model: Any,
        tokenizer: Any,
        group: Any,
        control: Any,
        model_identity: str,
        prefill_rank: int,
        decode_rank: int,
        prefill_step_size: int,
        cli_args: Any,
        stream: Any,
    ) -> None:
        if decode_rank != 0 or prefill_rank != 1:
            raise RuntimeError(
                "persistent phase-split serving currently requires rank 0 decode "
                "and rank 1 prefill"
            )
        self.mx = mx
        self.model = model
        self.tokenizer = tokenizer
        self.group = group
        self.control = control
        self.model_identity = model_identity
        self.prefill_rank = prefill_rank
        self.decode_rank = decode_rank
        self.prefill_step_size = prefill_step_size
        self.cli_args = cli_args
        self.stream = stream
        self.requests: Queue = Queue()
        self._stopping = threading.Event()
        self._decode_thread: threading.Thread | None = None
        self._broker = threading.Thread(
            target=self._broker_loop,
            name="omlx-phase-broker",
            daemon=True,
        )
        self._broker.start()

    def _tokenize(self, request: Any, args: Any) -> list[int]:
        from mlx_lm.server import convert_chat, process_message_content

        if request.request_type == "text":
            return list(self.tokenizer.encode(request.prompt))
        messages = copy.deepcopy(request.messages)
        if self.tokenizer.has_chat_template:
            process_message_content(messages)
            template_kwargs = dict(
                tools=request.tools,
                tokenize=True,
                **dict(self.cli_args.chat_template_args or {}),
            )
            if args.chat_template_kwargs:
                template_kwargs.update(args.chat_template_kwargs)
            return list(
                self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            )
        return list(self.tokenizer.encode(convert_chat(messages, request.role_mapping)))

    def generate(self, request: Any, args: Any, progress_callback: Any = None):
        from mlx_lm.server import GenerationContext

        if args.num_draft_tokens:
            raise ValueError("phase-split MTP/speculative decode is not enabled yet")
        prompt = self._tokenize(request, args)
        machine, sequences = _state_machine(self.tokenizer, args.stop_words)
        context = GenerationContext(
            has_tool_calling=self.tokenizer.has_tool_calling,
            has_thinking=self.tokenizer.has_thinking,
            tool_parser=self.tokenizer.tool_parser,
            sequences=sequences,
            prompt=prompt,
            prompt_cache_count=0,
        )
        output: Queue = Queue()
        self.requests.put(
            _ServingRequest(
                prompt=prompt,
                args=args,
                context=context,
                state_machine=machine,
                output=output,
                progress=progress_callback or (lambda *_: None),
            )
        )

        def responses() -> Iterator[Any]:
            while True:
                value = output.get()
                if value is None:
                    return
                if isinstance(value, BaseException):
                    raise value
                yield value

        return context, responses()

    def _recv_progress(self, request: _ServingRequest) -> bytes:
        packet = self.control.broadcast_owned_bytes(
            None,
            source_rank=self.prefill_rank,
            expected_size=_PROGRESS.size,
        )
        processed, total = _PROGRESS.unpack(packet)
        request.progress(int(processed), int(total))
        return packet

    def _decode(self, request: _ServingRequest, cache: list[Any], logits: Any) -> None:
        from mlx_lm.server import Response, _format_top_logprobs, _make_logits_processors, _make_sampler

        self.mx.set_default_stream(self.stream)
        args = request.args
        if args.seed is not None:
            self.mx.random.seed(args.seed)
        sampler = _make_sampler(args, self.tokenizer)
        processors = _make_logits_processors(args)
        history = self.mx.array(request.prompt, dtype=self.mx.int32)
        state = request.state_machine.make_state()
        detokenizer = self.tokenizer.detokenizer
        try:
            for index in range(args.max_tokens):
                adjusted = logits
                for processor in processors:
                    adjusted = processor(history, adjusted)
                logprobs = adjusted - self.mx.logsumexp(adjusted, keepdims=True)
                sampled = sampler(logprobs)
                self.mx.eval(sampled, logprobs)
                token = int(sampled.item())
                state, matched, current_state = request.state_machine.match(state, token)
                detokenizer.add_token(token)
                finish_reason = None
                if matched is not None and current_state is None:
                    finish_reason = "stop"
                elif index + 1 >= args.max_tokens:
                    finish_reason = "length"
                request.output.put(
                    Response(
                        detokenizer.last_segment,
                        token,
                        current_state,
                        matched,
                        float(logprobs[0, token].item()),
                        finish_reason,
                        _format_top_logprobs(
                            logprobs.squeeze(0),
                            args.top_logprobs,
                            self.tokenizer,
                        ),
                    )
                )
                if finish_reason is not None or request.context._should_stop:
                    break
                history = self.mx.concatenate(
                    [history, self.mx.array([token], dtype=self.mx.int32)]
                )
                logits = self.model(
                    self.mx.array([[token]], dtype=self.mx.int32),
                    cache=cache,
                )[:, -1, :]
                self.mx.eval(logits)
        except BaseException as exc:
            request.output.put(exc)
        finally:
            detokenizer.finalize()
            request.output.put(None)

    def _broker_loop(self) -> None:
        self.mx.set_default_stream(self.stream)
        previous: threading.Thread | None = None
        while not self._stopping.is_set():
            try:
                request = self.requests.get(timeout=0.05)
            except Empty:
                continue
            if request is None:
                break
            try:
                self.control.broadcast_object(
                    {
                        "op": "prefill",
                        "prompt": request.prompt,
                        "prefill_step_size": self.prefill_step_size,
                    }
                )
                for _ in range(_prefill_calls(len(request.prompt), self.prefill_step_size)):
                    self._recv_progress(request)
                header = self.control.broadcast_owned_bytes(
                    None,
                    source_rank=self.prefill_rank,
                    expected_size=_LOGITS_HEADER.size,
                )
                dtype_code, rows, columns = _LOGITS_HEADER.unpack(header)
                dtype_name = _CODE_TO_DTYPE.get(dtype_code)
                if dtype_name is None or rows != 1 or columns <= 0:
                    raise RuntimeError("prefill rank returned an invalid logits header")

                # Leave rank 1 waiting on the TCP barrier until the previous
                # decode releases rank 0's Metal stream. No RDMA receive is
                # posted during an arbitrarily long decode.
                if previous is not None:
                    previous.join()
                self.control.barrier()
                cache, _manifest, _stats = recv_cache_transfer(
                    self.mx,
                    src=self.prefill_rank,
                    group=self.group,
                    expected_model_identity=self.model_identity,
                )
                logits = self.mx.distributed.recv(
                    (rows, columns),
                    getattr(self.mx, dtype_name),
                    self.prefill_rank,
                    group=self.group,
                )
                self.mx.eval(logits)
                self.mx.synchronize()
                previous = threading.Thread(
                    target=self._decode,
                    args=(request, cache, logits),
                    name="omlx-phase-decode",
                    daemon=True,
                )
                previous.start()
            except BaseException as exc:
                request.output.put(exc)
                request.output.put(None)
        if previous is not None:
            previous.join()
        with threading.Lock():
            try:
                self.control.broadcast_object({"op": "stop"})
            except Exception:
                pass

    def stop_and_join(self) -> None:
        self._stopping.set()
        self.requests.put(None)
        self._broker.join(timeout=10.0)

    def join(self) -> None:
        self._broker.join()


def _prefill_rank_loop(
    *,
    mx: Any,
    model: Any,
    group: Any,
    control: Any,
    model_identity: str,
) -> None:
    from mlx_lm.models.cache import make_prompt_cache

    while True:
        request = control.broadcast_object(None)
        if not isinstance(request, dict):
            raise RuntimeError("phase broker sent an invalid request")
        if request.get("op") == "stop":
            return
        if request.get("op") != "prefill":
            raise RuntimeError("phase broker sent an unknown operation")
        prompt = [int(value) for value in request.get("prompt") or ()]
        step = int(request.get("prefill_step_size") or 2048)
        if len(prompt) < 2 or step < 1:
            raise RuntimeError("phase broker sent an invalid prompt")
        cache = make_prompt_cache(model)

        def progress(processed: int, total: int) -> None:
            control.broadcast_owned_bytes(
                _PROGRESS.pack(processed, total),
                source_rank=1,
                expected_size=_PROGRESS.size,
            )

        logits, _seconds = _prefill_logits(
            mx,
            model,
            cache,
            prompt,
            step=step,
            progress=progress,
        )
        dtype_code = _DTYPE_TO_CODE.get(_dtype_name(logits.dtype))
        if dtype_code is None or logits.ndim != 2:
            raise RuntimeError("phase prefill produced unsupported logits")
        control.broadcast_owned_bytes(
            _LOGITS_HEADER.pack(dtype_code, int(logits.shape[0]), int(logits.shape[1])),
            source_rank=1,
            expected_size=_LOGITS_HEADER.size,
        )
        control.barrier()
        prepared = prepare_cache_transfer(
            cache,
            model_identity=model_identity,
            prompt_tokens=len(prompt),
        )
        send_cache_transfer(mx, prepared, dst=0, group=group)
        mx.eval(mx.distributed.send(mx.contiguous(logits), 0, group=group))
        mx.synchronize()


def run_worker(args: argparse.Namespace) -> int:
    from omlx._torch_stub import install as install_torch_stub

    install_torch_stub()
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.server import _run_http_server

    from .control_plane import RankControlPlane
    from .deployment import (
        decode_worker_contract,
        decode_worker_path_map,
        decode_worker_serving_mode,
    )
    from .inference_worker import (
        _apply_rank_wired_limit,
        _emit_event,
        _install_signal_handlers,
        _release_metal_memory,
        _runtime_assignment,
        _wait_for_serve_release,
    )
    from .jaccl_lease import acquire_jaccl_communicator_lease
    from .jaccl_side_channel import init_cluster_group
    from .memory_guard import (
        admission_budget,
        assignment_memory_safety,
        guard_rank_load,
    )
    from .staging import model_identity_digest
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    _install_signal_handlers()
    plan_hash, assignments, _profiles, tensor_parallel_size = decode_worker_contract(
        args.plan
    )
    mode, prefill_rank, decode_rank = decode_worker_serving_mode(args.plan)
    if mode != "disaggregated" or tensor_parallel_size != 1:
        raise RuntimeError("phase server received a non-disaggregated plan")
    if args.plan_hash != plan_hash:
        raise RuntimeError("worker plan hash does not match launch contract")

    init_backend = "jaccl" if args.backend.startswith("jaccl") else "ring"
    lease = (
        acquire_jaccl_communicator_lease(
            deployment_id=args.deployment_id,
            state_dir=args.state_dir,
        )
        if init_backend == "jaccl"
        else None
    )
    try:
        group = init_cluster_group(mx, backend=init_backend, strict=True)
        rank = int(group.rank())
        if group.size() != 2:
            raise RuntimeError("phase server currently requires two ranks")
        assignment = sorted(assignments, key=lambda item: item.rank)[rank]
        admission_ceiling = guard_rank_load(
            assignment,
            rank=rank,
            role=assignment.role,
            memory_guard_tier=assignment.memory_guard_tier,
        )
        load_budget = admission_budget(
            admission_ceiling,
            role=assignment.role,
            safety=assignment_memory_safety(assignment),
        )
        _apply_rank_wired_limit(load_budget)
        path_map = decode_worker_path_map(args.plan)
        model_path = Path(path_map.get(assignment.node_id, args.model)).expanduser()
        identity = model_identity_digest(model_path)
        maybe_apply_pre_load_patches(
            model_path,
            model_settings=SimpleNamespace(
                mtp_enabled=False,
                mtp_num_draft_tokens=0,
            ),
        )
        stream = mx.new_thread_unsafe_stream(mx.default_device())
        mx.set_default_stream(stream)
        before = int(mx.get_active_memory())
        model, tokenizer = load(
            str(model_path),
            lazy=False,
            trust_remote_code=args.trust_remote_code,
        )
        model.eval()
        measured = max(0, int(mx.get_active_memory()) - before)
        _emit_event(
            {
                "type": "rank_ready",
                "protocol_version": 1,
                "deployment_id": args.deployment_id,
                "plan_hash": plan_hash,
                "world_size": 2,
                "measured_weight_bytes": measured,
                **_runtime_assignment(assignment),
            }
        )
        _wait_for_serve_release(
            args.state_dir,
            args.deployment_id,
            plan_hash,
            2,
        )
        with RankControlPlane(
            rank=rank,
            world_size=2,
            host=args.control_host,
            port=args.control_port,
            token=args.control_token,
            connect_timeout=120.0,
            io_timeout=3600.0,
        ) as control:
            control.barrier()
            if rank == 0:
                cli_args = SimpleNamespace(
                    allowed_origins=None,
                    num_draft_tokens=0,
                    max_tokens=4096,
                    temp=0.0,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    model=str(model_path),
                    chat_template_args={},
                )
                generator = _PhaseResponseGenerator(
                    mx=mx,
                    model=model,
                    tokenizer=tokenizer,
                    group=group,
                    control=control,
                    model_identity=identity,
                    prefill_rank=int(prefill_rank),
                    decode_rank=int(decode_rank),
                    prefill_step_size=args.prefill_step_size,
                    cli_args=cli_args,
                    stream=stream,
                )
                _emit_event(
                    {
                        "type": "ready",
                        "protocol_version": 1,
                        "deployment_id": args.deployment_id,
                        "plan_hash": plan_hash,
                        "rank": 0,
                        "world_size": 2,
                        "port": args.port,
                        "serving_mode": "disaggregated",
                        "prefill_rank": prefill_rank,
                        "decode_rank": decode_rank,
                    }
                )
                _run_http_server("127.0.0.1", args.port, generator)
            else:
                _prefill_rank_loop(
                    mx=mx,
                    model=model,
                    group=group,
                    control=control,
                    model_identity=identity,
                )
        return 0
    finally:
        if lease is not None:
            lease.close()
        try:
            _release_metal_memory("phase server shutdown")
        except Exception:
            pass


def main() -> int:
    return run_worker(_arguments())


if __name__ == "__main__":
    raise SystemExit(main())
