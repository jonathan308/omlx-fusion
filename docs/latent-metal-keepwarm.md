# Latent Metal keepwarm (experimental)

Latent Metal keepwarm is an opt-in latency feature for cached follow-up turns.
It submits a tiny, bounded fp16 matrix multiplication on each loaded engine's
existing serialized MLX lane while that engine is idle. It never reads or
writes model weights, KV state, prompt tokens, or SSD cache artifacts.

The mechanism is adapted from the Apache-2.0
[ThunderMLX](https://github.com/jonathan308/ThunderMLX) keepwarm design, with
oMLX-specific continuous-batching, cache-clear, model-load, and live-settings
gates. Fusion also preserves the existing optional distributed data-plane ping
for cluster ranks; the local Advanced toggle controls the shared master switch
without removing cluster telemetry or JACCL/RDMA keepwarm behavior.

Enable it in **Settings → Advanced → Performance**. The switch applies to
loaded Batched and VLM engines immediately and is saved for engines loaded
later. It is disabled by default because keeping the GPU command path active
can use slightly more idle power.

Safety behavior:

- no touch runs before a real request completes or a resident cache is seen;
- active requests and queued admissions on that engine always win; an idle
  touch skips rather than entering that engine's scheduler;
- request-start, post-response, and periodic touches all execute on the same
  one-worker MLX executor and stream used by inference;
- matrix width and repeat count are bounded;
- a failed or slow touch enters a bounded backoff;
- clearing the in-memory cache, including the exact-resident L0 tier, disarms
  local warming until the next real request;
- unloading an engine shuts down its controller and retains no Metal arrays.

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

For qualification, compare cached-turn TTFT after 0, 5, 15, and 60 seconds of
idle with the toggle off and on. Also compare B1/B2/B4/B6 throughput,
cancellation, cache reuse and SSD writes, cross-session restores, and process
footprint over repeated touches. Cold-prefill, generated-token, tool-call, and
MTP acceptance results must remain equivalent; keepwarm changes readiness
only, never model math.
