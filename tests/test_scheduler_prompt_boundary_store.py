# SPDX-License-Identifier: Apache-2.0
"""Tests for parser-stop prompt-boundary cache storage."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


def _scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_aware_cache = object()
    scheduler.config = SchedulerConfig(paged_cache_block_size=4)
    return scheduler


def _request(prompt_tokens):
    return SimpleNamespace(
        prompt_token_ids=prompt_tokens,
        specprefill_indices=None,
    )


def test_prompt_boundary_store_fills_only_sliceable_snapshot_placeholders():
    scheduler = _scheduler()
    prompt_tokens = list(range(10))
    boundary_tokens = prompt_tokens[:8]
    boundary_cache = [
        {"state": (), "class_name": "KVCache", "cache_type": "KVCache"},
        {
            "state": ("rotating-at-boundary",),
            "class_name": "RotatingKVCache",
            "cache_type": "RotatingKVCache",
        },
    ]
    live_cache = [
        {"state": ("kv-live",), "class_name": "KVCache", "cache_type": "KVCache"},
        {
            "state": ("rotating-live-tail",),
            "class_name": "RotatingKVCache",
            "cache_type": "RotatingKVCache",
        },
    ]

    scheduler._get_boundary_store_override = MagicMock(
        return_value=(boundary_tokens, boundary_cache, None, {})
    )
    scheduler._extract_live_request_cache_for_store = MagicMock(
        return_value=(live_cache, "live-config")
    )

    result = scheduler._prepare_prompt_boundary_cache_store(
        "req-parser-stop",
        _request(prompt_tokens),
        uid=7,
    )

    assert result is not None
    token_sequence, cache_to_store, model_config, intermediate_snapshots = result
    assert token_sequence == boundary_tokens
    assert cache_to_store == [live_cache[0], boundary_cache[1]]
    assert model_config == "live-config"
    assert intermediate_snapshots == {}
    scheduler._extract_live_request_cache_for_store.assert_called_once_with(
        "req-parser-stop",
        7,
        boundary_tokens,
    )


def test_prompt_boundary_store_refills_blanked_cachelist_members():
    """Member-filtered snapshots (#2550 follow-up): a promoted CacheList
    layer with a blanked sliceable sub must be refilled from the live
    cache, keeping the snapshot's boundary state for the other members."""
    scheduler = _scheduler()
    prompt_tokens = list(range(10))
    boundary_tokens = prompt_tokens[:8]
    boundary_cache = [
        {
            "state": [(), ("conv-at-boundary",)],
            "meta_state": (["KVCache", "ArraysCache"], [(), ()]),
            "class_name": "CacheList",
            "cache_type": "CacheList",
        },
    ]
    live_cache = [
        {
            "state": [("kv-live-keys", "kv-live-values"), ("conv-live-tail",)],
            "meta_state": (["KVCache", "ArraysCache"], [(), ()]),
            "class_name": "CacheList",
            "cache_type": "CacheList",
        },
    ]

    scheduler._get_boundary_store_override = MagicMock(
        return_value=(boundary_tokens, boundary_cache, None, {})
    )
    scheduler._extract_live_request_cache_for_store = MagicMock(
        return_value=(live_cache, "live-config")
    )

    result = scheduler._prepare_prompt_boundary_cache_store(
        "req-parser-stop",
        _request(prompt_tokens),
        uid=7,
    )

    assert result is not None
    token_sequence, cache_to_store, model_config, _ = result
    assert token_sequence == boundary_tokens
    layer = cache_to_store[0]
    assert layer["state"][0] == ("kv-live-keys", "kv-live-values")
    assert layer["state"][1] == ("conv-at-boundary",)
    assert model_config == "live-config"


def test_prompt_boundary_store_skips_unfillable_blanked_members():
    """When the live cache cannot supply a blanked CacheList member, the
    store must be skipped instead of persisting a partial composite."""
    scheduler = _scheduler()
    prompt_tokens = list(range(10))
    boundary_cache = [
        {
            "state": [(), ("conv-at-boundary",)],
            "class_name": "CacheList",
            "cache_type": "CacheList",
        },
    ]
    live_cache = [
        {"state": ("kv-live",), "class_name": "KVCache", "cache_type": "KVCache"},
    ]

    scheduler._get_boundary_store_override = MagicMock(
        return_value=(prompt_tokens[:8], boundary_cache, None, {})
    )
    scheduler._extract_live_request_cache_for_store = MagicMock(
        return_value=(live_cache, "live-config")
    )

    result = scheduler._prepare_prompt_boundary_cache_store(
        "req-parser-stop",
        _request(prompt_tokens),
        uid=7,
    )

    assert result is None


def test_prompt_boundary_store_skips_missing_snapshot_for_snapshot_models():
    scheduler = _scheduler()
    scheduler._get_boundary_store_override = MagicMock(return_value=None)
    scheduler._detect_boundary_snapshot_need = MagicMock(return_value=True)
    scheduler._extract_live_request_cache_for_store = MagicMock()

    result = scheduler._prepare_prompt_boundary_cache_store(
        "req-parser-stop",
        _request(list(range(10))),
        uid=7,
    )

    assert result is None
    scheduler._extract_live_request_cache_for_store.assert_not_called()


def test_prompt_boundary_store_uses_live_cache_for_sliceable_models():
    scheduler = _scheduler()
    prompt_tokens = list(range(10))
    boundary_tokens = prompt_tokens[:8]
    live_cache = [
        {"state": ("kv-live",), "class_name": "KVCache", "cache_type": "KVCache"},
        {
            "state": ("batch-kv-live",),
            "class_name": "BatchKVCache",
            "cache_type": "BatchKVCache",
        },
    ]
    scheduler._get_boundary_store_override = MagicMock(return_value=None)
    scheduler._detect_boundary_snapshot_need = MagicMock(return_value=False)
    scheduler._extract_live_request_cache_for_store = MagicMock(
        return_value=(live_cache, "live-config")
    )

    result = scheduler._prepare_prompt_boundary_cache_store(
        "req-parser-stop",
        _request(prompt_tokens),
        uid=7,
    )

    assert result == (boundary_tokens, live_cache, "live-config", None)
    scheduler._extract_live_request_cache_for_store.assert_called_once_with(
        "req-parser-stop",
        7,
        boundary_tokens,
    )


def test_live_store_uses_exact_terminal_source_after_uid_retirement():
    scheduler = _scheduler()
    request_id = "req-terminal-candidate"
    expected_tokens = [1, 2, 3, 4]
    candidate_tokens = expected_tokens + [5, 6]
    candidate_cache = [object()]
    request = _request(expected_tokens)
    request._terminal_prompt_boundary_source = (
        candidate_tokens,
        candidate_cache,
    )
    scheduler.requests = {request_id: request}
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.extract_cache.return_value = {}
    scheduler._stream = object()
    scheduler._resident_cache_matches_token_count = MagicMock(return_value=True)
    extracted = [
        {"state": ("qsa",), "class_name": "QSAKVCache", "cache_type": "QSAKVCache"}
    ]
    scheduler._extract_cache_states = MagicMock(
        return_value=(extracted, "candidate-config")
    )

    with (
        patch("omlx.scheduler._safe_sync_stream"),
        patch("omlx.scheduler.mx.stream", return_value=nullcontext()),
    ):
        result = scheduler._extract_live_request_cache_for_store(
            request_id,
            uid=7,
            expected_tokens=expected_tokens,
        )

    assert result == (extracted, "candidate-config")
    scheduler._resident_cache_matches_token_count.assert_called_once_with(
        candidate_cache,
        len(candidate_tokens),
    )
    scheduler._extract_cache_states.assert_called_once_with(candidate_cache)


def test_prompt_boundary_store_refills_qsa_from_retired_terminal_source():
    scheduler = _scheduler()
    request_id = "req-terminal-boundary"
    prompt_tokens = list(range(10))
    boundary_tokens = prompt_tokens[:8]
    boundary_cache = [
        {"state": (), "class_name": "QSAKVCache", "cache_type": "QSAKVCache"},
        {
            "state": ("arrays-at-boundary",),
            "class_name": "ArraysCache",
            "cache_type": "ArraysCache",
        },
    ]
    terminal_cache = [object(), object()]
    terminal_extracted = [
        {
            "state": ("qsa-terminal-tail",),
            "class_name": "QSAKVCache",
            "cache_type": "QSAKVCache",
        },
        {
            "state": ("arrays-terminal-tail",),
            "class_name": "ArraysCache",
            "cache_type": "ArraysCache",
        },
    ]
    request = _request(prompt_tokens)
    request._terminal_prompt_boundary_source = (
        prompt_tokens + [100, 101],
        terminal_cache,
    )
    scheduler.requests = {request_id: request}
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.extract_cache.return_value = {}
    scheduler._stream = object()
    scheduler._get_boundary_store_override = MagicMock(
        return_value=(boundary_tokens, boundary_cache, None, {})
    )
    scheduler._resident_cache_matches_token_count = MagicMock(return_value=True)
    scheduler._extract_cache_states = MagicMock(
        return_value=(terminal_extracted, "terminal-config")
    )

    with (
        patch("omlx.scheduler._safe_sync_stream"),
        patch("omlx.scheduler.mx.stream", return_value=nullcontext()),
    ):
        result = scheduler._prepare_prompt_boundary_cache_store(
            request_id,
            request,
            uid=7,
        )

    assert result is not None
    token_sequence, cache_to_store, model_config, snapshots = result
    assert token_sequence == boundary_tokens
    assert cache_to_store == [terminal_extracted[0], boundary_cache[1]]
    assert model_config == "terminal-config"
    assert snapshots == {}


def test_terminal_prompt_source_rejects_wrong_prefix_or_timeline():
    scheduler = _scheduler()
    request_id = "req-terminal-candidate"
    candidate_cache = [object()]
    request = _request([1, 2, 3, 4])
    request._terminal_prompt_boundary_source = ([1, 9, 3, 4, 5], candidate_cache)
    scheduler.requests = {request_id: request}
    scheduler._resident_cache_matches_token_count = MagicMock(return_value=True)
    scheduler._extract_cache_states = MagicMock()

    assert (
        scheduler._extract_terminal_prompt_source_for_store(
            request_id,
            [1, 2, 3, 4],
        )
        is None
    )
    scheduler._resident_cache_matches_token_count.assert_not_called()
    scheduler._extract_cache_states.assert_not_called()

    request._terminal_prompt_boundary_source = ([1, 2, 3, 4, 5], candidate_cache)
    scheduler._resident_cache_matches_token_count.return_value = False
    assert (
        scheduler._extract_terminal_prompt_source_for_store(
            request_id,
            [1, 2, 3, 4],
        )
        is None
    )
    scheduler._extract_cache_states.assert_not_called()


def test_terminal_prompt_source_stages_without_exact_resident_tier():
    scheduler = _scheduler()
    cache = [object()]
    request = _request([1, 2, 3, 4])
    request.request_id = "req-terminal-source"
    request._mtp_exact_terminal_proved = "qwen4-target-only-v1"
    request.skip_cache_store = False
    request.images = None
    request.videos = None
    request.vlm_inputs_embeds = None
    request.vlm_extra_keys_for_cache = None
    scheduler._request_is_text_only_for_resident_cache = MagicMock(return_value=True)
    scheduler._invalidate_resident_pool_with_telemetry = MagicMock(return_value=True)
    scheduler._resident_cache_matches_token_count = MagicMock(return_value=True)

    scheduler._stage_terminal_prompt_boundary_source(
        request,
        cache,
        [1, 2, 3, 4, 5],
    )

    assert request._terminal_prompt_boundary_source == (
        [1, 2, 3, 4, 5],
        cache,
    )
    scheduler._invalidate_resident_pool_with_telemetry.assert_called_once_with(
        cache,
        phase="durable-source",
    )
    scheduler._resident_cache_matches_token_count.assert_called_once_with(cache, 5)


def test_cleanup_finished_stores_prompt_boundary_without_extracted_cache(
    mock_model,
    mock_tokenizer,
):
    scheduler = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4),
    )
    scheduler.block_aware_cache = MagicMock()
    scheduler.paged_cache_manager = None

    request = Request(
        request_id="req-parser-stop",
        prompt="prompt",
        sampling_params=SamplingParams(),
    )
    request.prompt_token_ids = list(range(10))
    request.num_prompt_tokens = 10
    request.output_token_ids = [100, 101]
    request._extracted_cache = None

    boundary_tokens = list(range(8))
    boundary_cache = [
        {"state": ("kv-at-boundary",), "class_name": "KVCache", "cache_type": "KVCache"}
    ]
    scheduler.running[request.request_id] = request
    scheduler.requests[request.request_id] = request
    scheduler.request_id_to_uid[request.request_id] = 7
    scheduler.uid_to_request_id[7] = request.request_id

    with (
        patch.object(
            scheduler,
            "_prepare_prompt_boundary_cache_store",
            return_value=(boundary_tokens, boundary_cache, "boundary-config", None),
        ) as prepare,
        patch.object(scheduler, "_remove_uid_from_active_batch"),
    ):
        scheduler._cleanup_finished({request.request_id})

    prepare.assert_called_once_with(request.request_id, request, 7)
    scheduler.block_aware_cache.store_cache.assert_called_once()
    args, kwargs = scheduler.block_aware_cache.store_cache.call_args
    assert args[0] == request.request_id
    assert args[1] == boundary_tokens
    assert args[2] == boundary_cache
    assert kwargs["model_cache_config"] == "boundary-config"


def test_cleanup_finished_skip_cache_store_takes_leak_guard_branch(
    mock_model,
    mock_tokenizer,
):
    """A skip_cache_store request must not prep or submit a store, but its
    blocks still go through the leak-guard release path."""
    scheduler = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(paged_cache_block_size=4),
    )
    scheduler.block_aware_cache = MagicMock()
    scheduler.paged_cache_manager = None

    request = Request(
        request_id="req-ctx-probe",
        prompt="prompt",
        sampling_params=SamplingParams(),
        skip_cache_store=True,
    )
    request.prompt_token_ids = list(range(10))
    request.num_prompt_tokens = 10
    request.output_token_ids = [100]
    request._extracted_cache = ["kv-live"]

    scheduler.running[request.request_id] = request
    scheduler.requests[request.request_id] = request
    scheduler.request_id_to_uid[request.request_id] = 7
    scheduler.uid_to_request_id[7] = request.request_id

    with (
        patch.object(scheduler, "_prepare_prompt_boundary_cache_store") as prepare,
        patch.object(scheduler, "_remove_uid_from_active_batch"),
    ):
        scheduler._cleanup_finished({request.request_id})

    prepare.assert_not_called()
    scheduler.block_aware_cache.store_cache.assert_not_called()
    scheduler.block_aware_cache.clear_request_entry.assert_called_once_with(
        request.request_id
    )
    assert request.request_id not in scheduler.running
    assert request.request_id not in scheduler.requests
