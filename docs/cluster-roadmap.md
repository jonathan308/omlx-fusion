# Distributed & RDMA serving — roadmap

Status: planning document for contributors. Sizes are rough: **S** days,
**M** weeks, **L** a month or more. Every "today" claim cites the module that
implements it, so the plan can be re-grounded as the code moves.

## Phase 0 — merged foundation (done)

PR #2423 plus follow-ups delivered the experimental cluster:

- read-only Thunderbolt, RDMA, IP, route, memory, and runtime probes
  (`omlx/cluster/probe.py` — `rdma_ctl status`, `ibv_devices`,
  `system_profiler`, `sysctl`);
- untrusted Bonjour discovery and advertisement (`omlx/cluster/discovery.py`,
  `_omlx._tcp` and SSH service types) with a **Detected nearby** list;
- shared-secret pairing (HMAC-SHA256) with short-lived SSH key exchange
  tokens and a dedicated managed identity (`omlx/cluster/ssh_keys.py`,
  `token_auth.py`; routes `/pairing-token`, `/ssh-key/exchange-token`,
  `/ssh-key/exchange`);
- prompt-free `accept-new` SSH trust with keepalive and a fixed non-interactive
  option set (`omlx/cluster/ssh_policy.py`);
- exact oMLX/MLX/MLX-LM/cluster-protocol preflight and peer import checks
  (`omlx/cluster/launch.py`, `omlx/cluster/autoconfigure.py`);
- safetensors-header planning across unequal memory budgets, including hybrid
  TP+PP plans, with a SHA-256 plan hash every worker validates
  (`omlx/cluster/planner.py`, `omlx/cluster/deployment.py`);
- Ring, JACCL, and JACCL Ring launch through MLX's official launcher with
  isolated rank processes and hard process-group teardown
  (`omlx/cluster/launch.py`);
- capability-gated tensor parallelism: explicit adapters for `qwen3_next` and
  `nemotron_h`, plus a source-gated native `shard()` fallback
  (`omlx/cluster/tensor_strategies.py`, `progressive_loading.py`);
- node roles, admission checks, and a load-time memory watchdog
  (`omlx/cluster/node_role.py`, `memory_guard.py`);
- heartbeat markers with 45 s staleness read against the peer's own clock, a
  failure-tolerant watchdog, and named peer-loss errors
  (`omlx/cluster/liveness.py`);
- per-rank shard staging with atomic `.part` installs and size verification
  (`omlx/cluster/staging.py`);
- bounded compute/collective calibration, headroom-aware auto-tuning,
  execution profiles, and a durable end-to-end strategy benchmark store
  feeding automatic parallelism selection (`omlx/cluster/performance.py`,
  `strategy_benchmarks.py`, `link_bandwidth.py`,
  `autoconfigure.order_hosts_for_topology`);
- capability-gated runtime optimizations, including the experimental
  token-only output path (`omlx/cluster/runtime_optimizations.py`);
- the Cluster dashboard tab with live shard map, per-rank KV ownership, TTFT,
  prefill/decode throughput, pipeline utilization, measured collective
  bandwidth, and failure guidance (`omlx/admin/templates/dashboard/_cluster.html`,
  `omlx/cluster/routes.py`, `guidance.py`, `telemetry.py`);
- follow-up fixes: removed the unwired keychain verify route (e4328e71),
  peer-probe backoff after SSH failures (d2b45479), planner memory accounting
  (739a0b63), heartbeat reads against the peer clock (8996d4c9), and fabric
  probe failure surfacing with redaction (9700e79e).

## Phase 1 — Onboarding (S–M)

Today a second Mac needs a source checkout and venv by hand
(`staging.DEFAULT_REMOTE_PYTHON` points at `~/omlx-distributed/.venv`), the
shared secret is copied manually, and a version skew is only *reported* —
the fix is the operator's problem.

Deliverables:

- **Peer enrollment and guided join.** A companion PR is in flight adding
  join-token enrollment (`omlx/cluster/enrollment.py`, a `omlx cluster join`
  CLI command, and a dashboard "Add a Mac" panel). Land it, then make the
  Cluster tab's pairing flow delegate to it.
- **Peer environment version-sync.** Preflight already names the exact
  mismatching packages (`autoconfigure.preflight_issues`,
  `launch.probe_remote_host`). Add a one-click action that installs or pins
  the matching MLX/MLX-LM build on the peer and re-probes until
  `runtime_compatible` is true.
- **Remove the manual-install cliff.** Bootstrap a worker-only environment
  over SSH from the coordinator: detect what the peer has, install what the
  model's import check (`autoconfigure.peer_import_issues`) says it needs,
  and report progress in the dashboard.

Dependencies: the in-flight enrollment PR. Acceptance: a factory-fresh Mac
with Remote Login enabled becomes a ready rank without anyone opening a
terminal on the peer; a version-skewed peer is repaired or explicitly blocked
from the coordinator's dashboard.

## Phase 2 — Resilience (M)

Today detection works but recovery is manual and total. The watchdog notices
a lost peer within seconds of serving traffic (45 s marker staleness, 15 s
probe cadence while loading, 3 s once ready, two consecutive failures
required — `liveness.py`), and then the whole job is torn down
(SIGTERM → SIGKILL on the launcher process group, `launch.py`); the operator
re-activates by hand.

Deliverables:

- **Rank auto-restart and rejoin.** Restart a dead rank's process and
  re-handshake the MLX group in place of full teardown, bounded by a retry
  budget so a genuinely gone Mac still fails the deployment.
- **SIGHUP-immune remote supervision.** A remote rank's lifetime is still
  tied to the SSH control channel that launched it — keepalive
  (`ssh_policy.py`, applied in `launch.py`) reduces drops but supervises
  nothing. Launch ranks under a detached supervisor on the peer so a dropped
  channel is a probe failure, not a dead rank.
- **Chaos test.** Kill a remote rank mid-decode on physical hardware; assert
  the deployment fails (or recovers, once rejoin lands) with a named error
  and no hung client request.
- **Automated soak gate.** `scripts/cluster_context_gate.py` already runs an
  exact-token long-context gate against a live endpoint, but it is manual.
  Schedule it against a standing two-Mac pair.
- **Two-Mac CI job.** The merged suite is entirely mocked or loopback
  (`collective-smoke` is a two-rank loopback all-sum; `pipeline-smoke` runs
  locally). Add a CI job on two physical Macs that runs the real smokes over
  a Thunderbolt cable.

Dependencies: none beyond Phase 0; supervision design should land before
auto-restart so restarts survive channel loss. Acceptance: a killed rank
mid-decode produces a stated failure or an automatic rejoin within the retry
budget; the soak and chaos gates run unattended and gate cluster changes.

## Phase 3 — Performance (M)

Deliverables:

- **Promote the token-only output path.** The capability-gated
  sampling-rank-only path (`runtime_optimizations.py`,
  `ExecutionSettings.sampling_rank_only`) removes the final hidden-state
  all-gather, but seeded single-request generation is rejected while it is
  active because MLX-LM routes seeded requests outside its continuous-batch
  path (`omlx/engine/distributed.py`). Teach that path seeded sampling, then
  make the optimization the default for models that pass source validation.
- **True 1F1B pipeline scheduling.** Today's coalesced microbatch target caps
  MLX-LM's continuous prefill/completion batches; it is explicitly not a
  1F1B scheduler, so decode latency still includes every stage and
  inter-stage send with bubbles between them. Interleave prefill and decode
  microbatches across stages to cut the bubbles.
- **Collective-bandwidth-aware strategy selection.** The strategy benchmark
  store already records context-bucketed end-to-end samples and
  `/autoconfigure` consumes them (`strategy_benchmarks.py`, `routes.py`);
  measured link profiles already steer host placement
  (`link_bandwidth.py`, `autoconfigure.order_hosts_for_topology`). Feed the
  measured collective bandwidth into `choose_parallelism` itself so the
  TP-vs-pipeline decision uses the wire, not only capability heuristics and
  end-to-end history.

Dependencies: Phase 2's chaos coverage before changing the default output
path. Acceptance: seeded chat requests succeed with the token-only path
active; a measured bubble reduction on a two-Mac pipeline; on a mixed
TB/Ethernet fabric the automatic strategy choice picks the measured-fastest
layout.

## Phase 4 — Coverage & scale (M–L)

Deliverables:

- **More TP adapters.** Only `qwen3_next` and `nemotron_h` have explicit
  adapters (`tensor_strategies.py`); mainstream dense models (Llama, Qwen3)
  rely on the source-gated native `shard()` fallback. Add first-party
  adapters with per-architecture tests.
- **Hybrid TP+PP at runtime.** `planner.plan_hybrid` plans it and
  `catalogue.py` reports it, but the runtime is fail-closed:
  `inference_worker.py` installs pipeline edges only when
  `tensor_parallel_size == 1`, and `progressive_loading.py` refuses combined
  pipeline+tensor groups. Wire the hybrid path through worker launch,
  loading, and validation.
- **3+ Mac GUI.** The dashboard flow is built around a single trusted peer;
  the schema, planner, and launcher already allow up to 64 nodes (route
  validation caps hosts at 64). Extend the Cluster tab to multi-peer
  enrollment, placement review, and per-rank live view.
- **Topology-aware rank ordering everywhere.** `order_hosts_for_topology`
  already places TP groups on the fastest links, but only inside
  `/autoconfigure`. Extend placement to order pipeline stages across
  heterogeneous links and cover the manual planning flow.

Dependencies: hybrid runtime work gates the 3+ Mac GUI being useful beyond
pipeline-only layouts. Acceptance: a dense Llama/Qwen3 model runs TP through
a first-party adapter; a three-Mac hybrid plan activates from the GUI and
serves; manual and automatic flows produce the same topology-aware rank
order.

## Phase 5 — Heterogeneous fleet (L)

Deliverables:

- **Rebase and land PR #2591** — CUDA workers, ConnectX-7 NCCL pairs, and
  join-keys — on the merged cluster base. Pairing, preflight, planning, and
  staging interfaces changed under it; the rebase is the work.
- **Hierarchical Ring → NCCL supernode gateway.** PR #2591's own design doc
  admits the outer ring is the only cross-supernode transport today. Add a
  gateway that bridges per-supernode NCCL domains over the Mac-side ring so a
  mixed fleet is one deployment instead of two.

Dependencies: Phase 1 enrollment (join-keys overlap), Phase 2 resilience
before fleets grow. Acceptance: a mixed Mac + CUDA deployment passes
preflight, plans, stages, and serves; cross-supernode bandwidth is measured
and reported alongside the Thunderbolt collectives.

## Phase 6 — Security & polish (S–M)

Deliverables:

- **Pairing confirmation instead of manual secret copy.** Today the operator
  types the same ≥16-character secret on both dashboards and it
  authenticates short-lived exchange tokens with HMAC-SHA256
  (`token_auth.py`, `ssh_keys.py`). Add a short-authentication-string or
  scanned-code confirmation so the secret never travels by hand.
- **SSH host-key pinning with guided recovery.** `accept-new` records
  first-seen keys and refuses changed ones (`ssh_policy.py`); recovery is a
  manual `ssh-keygen -R` following dashboard guidance. Pin the fingerprint at
  pairing time, display it for confirmation, and offer a guided replace flow
  when it legitimately changes.
- **i18n for the Cluster tab.** `_cluster.html` (~2,400 lines) is hardcoded
  English except five `cluster.pairing.*` strings, which are already
  translated in all nine locales. Key the remaining strings and translate
  them.
- **Multi-user / multi-tenancy review.** Cluster routes ride the single admin
  API key and the registry is per-install. Define what pairing, activation,
  and diagnostics should mean when more than one principal administers a
  Mac, including audit trails for enrollment and teardown.

Dependencies: none for i18n; pinning builds on Phase 1's enrollment flow.
Acceptance: pairing completes without copying a secret; a changed host key
produces a guided in-dashboard recovery; the Cluster tab renders fully in
every shipped locale; the multi-user model is documented even if the answer
is "single admin only" for now.
