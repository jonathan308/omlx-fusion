# Instant cached turns (experimental)

The Advanced **Instant cached turns** toggle controls oMLX's opt-in latency
stack for exact cached text follow-ups. Internally the setting remains
`latent_metal_keepwarm_enabled` for configuration compatibility. It combines:

1. a tiny asynchronous Metal pulse that keeps the command path responsive;
2. a bounded exact-resident handoff that retains validated prompt state in RAM;
3. an exact prompt-tail materializer that reconstructs an existing durable
   prefix, evaluates only a bounded uncached suffix through the target model,
   and atomically publishes a stable fallback.

The second mechanism is useful when an API client re-renders an assistant or
tool transcript differently from the raw terminal token stream. oMLX retains
the newest validated terminal cache and a stable input-prompt prefix when they
fit under the same byte ceiling. The next request acquires the longest exact
token prefix that actually matches; older branches use the durable fallback.

The toggle never changes logits, sampling, or accepted output. Unsupported,
multimodal, concurrent, or mismatched cache state falls back to normal serving.
Enabling it can use additional RAM and a small amount of idle power.

The mechanism is adapted from the Apache-2.0
[ThunderMLX](https://github.com/jonathan308/ThunderMLX) keepwarm design, with
oMLX-specific continuous-batching, cache, model-load, and live-settings gates.
The scheduler facility is model-agnostic for text-only, non-speculative cache
families that pass exact timeline validation. Speculative families fail closed;
Qwen4 is the first qualified exception because its target-only terminal commit,
QSA K/V, raw index keys, MRoPE positions, and recurrent state can all be proven
at the same token boundary. Image/video/media-keyed requests remain on the
existing multimodal cache path and never enter prompt-tail L0.

Fusion also preserves the optional distributed data-plane ping for cluster
ranks; the local Advanced toggle controls the shared master switch without
removing cluster telemetry or JACCL/RDMA keepwarm behavior.

Enable it in **Settings → Advanced → Performance**. The switch applies to
loaded Batched and VLM engines immediately and is saved for engines loaded
later. It is disabled by default because it can use slightly more idle power,
GPU time, and resident cache memory.

Safety behavior:

- no touch runs before a real request completes or a resident cache is seen;
- active requests and queued admissions always win; only one hidden
  materializer runs process-wide, and new work cancels it between bounded
  chunks;
- request-start, post-response, and periodic touches all execute on the same
  one-worker MLX executor and stream used by inference;
- pulse width/repeat count and prompt-tail token/chunk counts are bounded;
- hidden forwards are target-only: no sampler, BatchGenerator row, output,
  MTP draft, prompt-priming capture, or durable cache store is created;
- every retained cache must pass exact token, offset, recurrent-state, QSA
  index, and text-position validation before atomic publication;
- a two-candidate L0 remains under the configured byte ceiling and never
  evicts a longer matching terminal merely to install a shorter fallback;
- the durable paged/SSD tier may be read and promoted, but the hidden pass
  performs no SSD write and does not alter user cache-rate or throughput
  telemetry;
- a failed or slow touch enters a bounded backoff;
- clearing the in-memory cache, including the exact-resident L0 tier, disarms
  local warming until the next real request;
- unloading an engine waits behind in-flight maintenance, then releases its
  controller, cache references, model graph, and Metal arrays in order.

The optional environment controls are:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `OMLX_KEEPWARM` | `0` | Master switch |
| `OMLX_KEEPWARM_INTERVAL_SECONDS` | local `2`; cluster `10` | Periodic idle cadence |
| `OMLX_KEEPWARM_IDLE_AFTER_SECONDS` | `2` | Idle time before periodic touch |
| `OMLX_KEEPWARM_MATRIX_SIZE` | `1` | Periodic matrix width |
| `OMLX_KEEPWARM_REQUEST_START` | `1` | Request-boundary warm after idle |
| `OMLX_KEEPWARM_REQUEST_START_IDLE_SECONDS` | `2` | Request-boundary idle gate |
| `OMLX_KEEPWARM_REQUEST_START_MATRIX_SIZE` | `128` | Request-boundary width |
| `OMLX_KEEPWARM_POST_RESPONSE` | `1` | Follow-up-turn warm |
| `OMLX_KEEPWARM_POST_RESPONSE_DELAY_SECONDS` | local `1`; cluster `5` | Follow-up warm delay |
| `OMLX_KEEPWARM_POST_RESPONSE_MATRIX_SIZE` | `128` | Follow-up width |
| `OMLX_KEEPWARM_LARGE_CACHE_TOKENS` | `8192` | Long-cache cadence threshold |
| `OMLX_KEEPWARM_LARGE_CACHE_INTERVAL_SECONDS` | local `2`; cluster `60` | Long-cache cadence |
| `OMLX_KEEPWARM_SLOW_THRESHOLD_SECONDS` | `1` | Slow-operation threshold |
| `OMLX_KEEPWARM_SLOW_BACKOFF_SECONDS` | `60` | Backoff after a failed/slow touch |
| `OMLX_CLUSTER_KEEPWARM_DATAPLANE_PING` | `1` | Cluster-only rank data-plane ping |
| `OMLX_KEEPWARM_PROMPT_TAIL` | `0` | Exact idle prompt-tail materialization |
| `OMLX_KEEPWARM_PROMPT_TAIL_DELAY_SECONDS` | `1` | Delay after a completed request |
| `OMLX_KEEPWARM_PROMPT_TAIL_MIN_TOKENS` | `256` | Smallest prompt eligible for materialization |
| `OMLX_KEEPWARM_PROMPT_TAIL_MAX_SUFFIX_TOKENS` | `4096` | Maximum uncached suffix evaluated while idle |
| `OMLX_KEEPWARM_PROMPT_TAIL_MAX_TOKENS` | `262144` | Maximum complete prompt considered |
| `OMLX_KEEPWARM_PROMPT_TAIL_CHUNK_SIZE` | `128` | Cancellation granularity for hidden target forwards |
| `OMLX_EXACT_RESIDENT_MAX_ENTRIES` | explicit `0`; UI keepwarm uses `2` | Resident exact-prefix slots (`0` is a hard disable) |
| `OMLX_EXACT_RESIDENT_MAX_BYTES` | `8 GiB` | Shared terminal/fallback byte ceiling |

For qualification, compare cached-turn TTFT after 0, 5, 15, and 60 seconds of
idle with the toggle off and on. Also compare B1/B2/B4/B6 throughput,
cancellation, cache reuse and SSD writes, cross-session restores, and process
footprint over repeated touches. Include a transcript-divergence case where the
longer raw terminal misses but the exact input-prefix fallback matches. Cold
prefill, sustained generated-token rate, tool calls, output hashes, and MTP
acceptance must remain equivalent; keepwarm changes readiness and cache
availability only, never model math.

## Model-specific tuning: Qwen3.8-27B-oQ4e-mtp

Validated on Qwen3.8-Next (80B-A3B) with instant cached turns at ~0.3s TTFT. For the dense 27B `Jundot/Qwen3.8-27B-oQ4e-mtp` (`Qwen3_5ForConditionalGeneration`, `mtp_num_hidden_layers=1`, `qwen3_5`), the Lightning MTP draft+verify path (`mtp_enabled true`) blocks the exact-resident L0 handoff with `speculative-terminal-unproved` (`omlx/scheduler.py:9504` `self._resident_cache_spec_decode_active()`). Result: cached turns fall back to paged `hot_cache`/`SSD` (`Prefix cache restore source=paged cached=4096` 3–5s TTFT) instead of `source=exact-resident cached=5067 suffix=1 lookup 0.28ms retained 0.45GiB` 0.41s/0.77s TTFT.

**Tuning (proven on M5 Pro 48GB, block 2048, hot 4GB write-through, SSD 398GB):**

- Disable MTP for this model when `latent_metal_keepwarm_enabled` is desired:
  ```bash
  curl -s -b /tmp/cookie.jar -X PUT -H "Content-Type: application/json" \
    -d '{"mtp_enabled": false}' \
    http://127.0.0.1:8000/admin/api/models/Qwen3.8-27B-oQ4e-mtp/settings
  # then reload
  curl -s -X POST -H "Authorization: Bearer test" http://127.0.0.1:8000/v1/models/Qwen3.8-27B-oQ4e-mtp/unload
  curl -s -X POST -H "Authorization: Bearer test" http://127.0.0.1:8000/v1/models/Qwen3.8-27B-oQ4e-mtp/load
  ```
  Persisted in `~/.omlx/model_settings.json` `Qwen3.8-27B-oQ4e-mtp.mtp_enabled false`. After restart, `exact_resident_cache` shows `entries 2 max_entries 2 hits 1` vs `0` with MTP on, and `stream_model_ttft 0.41s stream_visible_ttft 0.77s` for 5067→5067 hit vs 3.24s/5.65s paged.

- Trade-off: MTP gives 2.5–3.3 tok/cycle for cold/long generation (26 tok/s) but disables instant L0. With MTP off, cold generation is ~15 tok/s but cached TTFT is 0.3–0.4s (vs 3–5s paged) and `cache_efficiency` 44.8%→68.6% for opencode-style 7635-token system prompts. For opencode (long system+tools, short follow-ups), instant 0.3s dominates.

- Alternative (future): enable Qwen3.5 target-only proof similar to Qwen4 `qwen4-target-only-v1` (`omlx/scheduler.py:8540` `_resident_cache_qwen4_target_only_enabled`) to allow MTP + exact resident. Until proven, the per-model disable is the validated tuning.
