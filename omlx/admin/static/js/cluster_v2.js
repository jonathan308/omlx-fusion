// oMLX Cluster v2 wizard — Alpine.js component backing
// omlx/admin/templates/dashboard/_cluster_v2.html.
//
// Contract (ops/notes/omlx_cluster_v2_spec.md, Module C): consume ONLY the
// Module A/B endpoints plus the pre-existing planner/activate API:
//
//   Module A/B (cluster v2):
//     GET    /api/cluster/devices               — {paired, discovered, self}, polled at 1 Hz
//     POST   /api/cluster/pair/approve          — {node_id, code}
//     POST   /api/cluster/pair/deny             — {node_id}
//     POST   /api/cluster/pair/join             — {coordinator_addr} — joiner: mint + show the code
//     GET    /api/cluster/pair/join             — local join snapshot, polled at 1 Hz (drives approval)
//     POST   /api/cluster/pair/join/cancel      — abandon the join in progress
//     POST   /api/cluster/devices/manual        — {ip, port} seed + probe a peer by address
//     DELETE /api/cluster/devices/{node_id}     — unpair
//     GET    /api/cluster/discovery/health      — {multicast_rx_within_5s: bool,
//           last_multicast_rx_at: float|null, mdns_active: bool, transport: str}
//           Powers the "Local network permission" self-test check row.
//           Implemented at discovery_routes.py; a 404 from an older build
//           degrades the row to "skipped", never a red failure (beacon loss
//           only affects discovery, not pairing via Add by IP).
//   Existing planner/activate API (omlx/cluster/routes.py, unchanged):
//     POST   /admin/api/cluster/models          — per-node model inventory
//     POST   /admin/api/cluster/catalogue       — model fit across nodes
//     POST   /admin/api/cluster/peer-probe      — SSH reachability / versions / RDMA
//     POST   /admin/api/cluster/autoconfigure   — benchmark probe (measure_performance)
//     POST   /admin/api/cluster/plan            — signed shard plan
//     GET    /admin/api/cluster/deployments     — active deployments
//     POST   /admin/api/cluster/deployments     — activate an approved plan
//     DELETE /admin/api/cluster/deployments/{id}— deactivate
//
// State machine (wizardState()): empty → discovering → device_card → pairing →
// checks → plan → active, with error as an overlay state (banner + toasts,
// never a modal). N device cards are rendered; there is no 2-Mac cap.
//
// This file is loaded as a plain script before Alpine initializes, exactly
// like dashboard.js, so `clusterV2Wizard` is a global factory referenced from
// x-data in _cluster_v2.html.

function clusterV2Wizard() {
    const CLUSTER_V2_API = {
        devices: '/api/cluster/devices',
        discoveryHealth: '/api/cluster/discovery/health',
        pairApprove: '/api/cluster/pair/approve',
        pairDeny: '/api/cluster/pair/deny',
        pairJoin: '/api/cluster/pair/join',
        pairJoinCancel: '/api/cluster/pair/join/cancel',
        manualDevice: '/api/cluster/devices/manual',
        unpair: (nodeId) =>
            `/api/cluster/devices/${encodeURIComponent(nodeId)}`,
        models: '/admin/api/cluster/models',
        catalogue: '/admin/api/cluster/catalogue',
        peerProbe: '/admin/api/cluster/peer-probe',
        autoconfigure: '/admin/api/cluster/autoconfigure',
        plan: '/admin/api/cluster/plan',
        nodeRoles: '/admin/api/cluster/node-roles',
        deployments: '/admin/api/cluster/deployments',
        deployment: (id) =>
            `/admin/api/cluster/deployments/${encodeURIComponent(id)}`,
    };

    // exo-style data layer: one snapshot endpoint polled once per second
    // while the tab is visible; the UI is a pure function of the snapshot.
    const CLUSTER_V2_POLL_MS = 1000;
    const CLUSTER_V2_DEPLOYMENTS_EVERY_TICKS = 5;
    // A missing v2 backend (404) flips the error state immediately; flaky
    // networks get a few grace failures first.
    const CLUSTER_V2_FAILURE_GRACE = 3;

    const CLUSTER_V2_LINK_META = {
        tb: { label: 'Thunderbolt', icon: 'zap' },
        ethernet: { label: 'Ethernet', icon: 'cable' },
        wifi: { label: 'Wi-Fi', icon: 'wifi' },
        tailscale: { label: 'Tailscale', icon: 'globe' },
        unknown: { label: 'Network', icon: 'help-circle' },
    };

    // Offline mirror of omlx/cluster/node_role.py (NodeRole.reserve_for):
    // a workstation keeps max(32 GiB, 50%) of its Mac, a headless node keeps
    // 10%. The wizard prefers GET /admin/api/cluster/node-roles, which exposes
    // these same numbers from the server; this mirror exists only so the
    // usable-budget labels still render when that endpoint is unreachable.
    // If node_role.py changes, this mirror must change with it.
    const CLUSTER_V2_ROLE_FALLBACK = {
        workstation: {
            key: 'workstation',
            label: 'Workstation',
            reserve_bytes: 32 * 1024 ** 3,
            reserve_fraction: 0.5,
        },
        headless: {
            key: 'headless',
            label: 'Headless',
            reserve_bytes: 0,
            reserve_fraction: 0.1,
        },
    };

    // English fallbacks for the cluster.v2.* strings the wizard adds. The
    // dashboard resolves window.t against en.json-filled locale_json, so these
    // only matter when window.t is unavailable (offline component tests).
    // Keep in sync with omlx/admin/i18n/en.json.
    const CLUSTER_V2_STRINGS = {
        'cluster.v2.strategy.title': 'How the model is split',
        'cluster.v2.strategy.auto': 'Auto',
        'cluster.v2.strategy.tensor': 'Tensor',
        'cluster.v2.strategy.pipeline': 'Pipeline',
        'cluster.v2.strategy.recommended': 'Recommended',
        'cluster.v2.strategy.hint.auto':
            'oMLX picks the split that fits this model and your link',
        'cluster.v2.strategy.hint.tensor':
            'Every Mac works on every token — needs a fast link',
        'cluster.v2.strategy.hint.pipeline':
            'Each Mac holds a different slice of the layers',
        'cluster.v2.strategy.tensor_needs_two':
            'Tensor parallelism needs 2+ Macs',
        'cluster.v2.strategy.tensor_unsupported':
            "This model's attention heads cannot be split across Macs",
        'cluster.v2.strategy.pipeline_unsupported':
            'This model does not support pipeline stages — use Tensor instead',
        'cluster.v2.split.tensor_caption':
            'Tensor split — each Mac holds 1/{count} of every layer',
        'cluster.v2.split.tensor_share': '1/{count} of every layer',
        'cluster.v2.models.on_every_mac': 'on every Mac',
        'cluster.v2.models.partial':
            'on {have} of {total} Macs — copied at activation',
        'cluster.v2.checks.beacon_label':
            'Local network permission (discovery only)',
        'cluster.v2.steps.plan_hint_tensor': 'Every layer, on every Mac',
    };
    const t = (key) =>
        typeof window !== 'undefined' && typeof window.t === 'function'
            ? window.t(key)
            : CLUSTER_V2_STRINGS[key] || key;

    return {
        // ---- snapshot state -------------------------------------------------
        devicesPayload: null,
        devicesLoaded: false,
        devicesError: '',
        devicesFailureCount: 0,
        devicesUnreachable: false,
        deploymentsPayload: [],
        deploymentsLoaded: false,
        discoveryHealth: null,
        discoveryHealthUnsupported: false,

        // ---- wizard cursor ---------------------------------------------------
        // Null = derive from the snapshot. Explicit values: 'checks', 'plan'.
        stage: null,
        pairing: { target: null, code: '', busy: false, error: '' },

        // ---- joiner side (this Mac shows the code, the other Mac approves) ---
        // Server-driven snapshot from GET /api/cluster/pair/join; polled in
        // tick() so the panel survives reloads and approval completes by
        // itself. target_name is client-local context for friendly toasts.
        join: {
            state: 'idle',
            code: null,
            expires_at: null,
            coordinator_addr: null,
            seconds_remaining: 0,
            error: null,
            busy: false,
            target_name: '',
        },
        joinApprovedNotified: false,
        joinDeniedNotified: false,

        // ---- add by IP (when multicast discovery is unavailable) -------------
        manualAddr: '',
        manualBusy: false,
        manualError: '',
        checks: {
            started: false,
            running: false,
            probes: {},
            benchmark: null,
            benchmarkRunning: false,
            ranAt: null,
        },

        // ---- plan ------------------------------------------------------------
        modelOptions: [],
        modelsLoading: false,
        modelsError: '',
        selectedModelPath: '',
        modelSearch: '',
        // Execution strategy for the split: 'auto' | 'tensor' | 'pipeline'.
        // 'tensor' threads tensor_parallel_size = planNodes().length into both
        // /plan and /deployments; anything else plans pipeline-only (TP=1).
        planStrategy: 'auto',
        // Per-model strategy advice from POST /admin/api/cluster/catalogue
        // (null = not attempted yet). catalogueFailed switches the
        // recommendation badge to the fast-transport heuristic.
        catalogueModels: null,
        catalogueLoading: false,
        catalogueFailed: false,
        plan: null,
        planLoading: false,
        planError: '',
        // Explicit per-node role picks (node_id → 'workstation' | 'headless').
        // Nodes without an entry keep the long-standing default: this Mac is
        // a workstation, every peer is headless.
        nodeRoles: {},
        roleOptions: [],
        // Parsed from a /plan 400: { shortfallBytes, canFixWithHeadless }.
        planFitFailure: null,
        activateBusy: false,
        confirmUnpairFor: '',
        confirmDeactivateFor: '',

        // ---- feedback ----------------------------------------------------------
        toasts: [],
        toastSeq: 0,
        installCommandCopied: false,

        pollTimer: null,
        tickCount: 0,

        // =====================================================================
        // Lifecycle
        // =====================================================================
        init() {
            this.tick();
            this.pollTimer = setInterval(() => this.tick(), CLUSTER_V2_POLL_MS);
        },

        wizardVisible() {
            // mainTab and clusterLegacyView live on ancestor scopes
            // (dashboard() and the dashboard.html wrapper respectively).
            return (
                this.mainTab === 'cluster' &&
                !this.clusterLegacyView &&
                !document.hidden
            );
        },

        async tick() {
            if (!this.wizardVisible()) return;
            await this.refreshDevices();
            await this.refreshJoinState();
            this.tickCount += 1;
            if (
                !this.deploymentsLoaded ||
                this.tickCount % CLUSTER_V2_DEPLOYMENTS_EVERY_TICKS === 0
            ) {
                await this.refreshDeployments();
            }
        },

        // =====================================================================
        // API helpers
        // =====================================================================
        async apiFetch(url, options = {}) {
            const response = await fetch(url, {
                headers: { 'Content-Type': 'application/json' },
                ...options,
            });
            if (response.status === 401) {
                window.location.href = '/admin';
                throw new Error('Sign-in required');
            }
            if (!response.ok) {
                let detail = `${response.status} ${response.statusText}`;
                try {
                    const payload = await response.json();
                    if (payload && payload.detail) {
                        detail =
                            typeof payload.detail === 'string'
                                ? payload.detail
                                : JSON.stringify(payload.detail);
                    }
                } catch (ignored) {
                    /* non-JSON error body */
                }
                const error = new Error(detail);
                error.status = response.status;
                throw error;
            }
            if (response.status === 204) return null;
            return response.json();
        },

        async refreshDevices() {
            try {
                const payload = await this.apiFetch(CLUSTER_V2_API.devices);
                this.devicesPayload = payload || {};
                this.devicesLoaded = true;
                this.devicesError = '';
                this.devicesFailureCount = 0;
                this.devicesUnreachable = false;
            } catch (error) {
                this.devicesFailureCount += 1;
                this.devicesError = error?.message || 'Cluster API unreachable';
                // 404 means the v2 discovery backend is not serving — that is
                // a hard, actionable failure, not network noise.
                if (
                    error?.status === 404 ||
                    this.devicesFailureCount >= CLUSTER_V2_FAILURE_GRACE
                ) {
                    this.devicesUnreachable = true;
                }
            }
        },

        async refreshDeployments() {
            try {
                const payload = await this.apiFetch(CLUSTER_V2_API.deployments);
                this.deploymentsPayload = payload?.deployments || [];
                this.deploymentsLoaded = true;
            } catch (error) {
                // Deployments are the pre-existing API; a failure here should
                // not tear down the discovery UI, just surface a toast once.
                if (this.deploymentsLoaded) {
                    this.notify(
                        'warning',
                        error?.message || 'Could not refresh deployments',
                    );
                }
            }
        },

        async refreshDiscoveryHealth() {
            try {
                this.discoveryHealth = await this.apiFetch(
                    CLUSTER_V2_API.discoveryHealth,
                );
                this.discoveryHealthUnsupported = false;
            } catch (error) {
                if (error?.status === 404) {
                    // Older builds predate the endpoint (implemented at
                    // discovery_routes.py); the check row degrades to
                    // "skipped" instead of a misleading red failure.
                    this.discoveryHealthUnsupported = true;
                    this.discoveryHealth = null;
                } else {
                    this.discoveryHealth = null;
                }
            }
        },

        // The joiner snapshot is server-owned, so a page reload mid-join
        // restores the panel and the coordinator's approval completes on the
        // next tick without any user action.
        async refreshJoinState() {
            try {
                const snapshot = await this.apiFetch(CLUSTER_V2_API.pairJoin);
                if (!snapshot) return;
                const previous = this.join.state;
                this.join = { ...this.join, ...snapshot, busy: false };
                if (
                    snapshot.state === 'approved' &&
                    !this.joinApprovedNotified
                ) {
                    this.joinApprovedNotified = true;
                    this.notify(
                        'success',
                        `This Mac joined ${this.joinTargetName()}'s cluster.`,
                    );
                    await this.refreshDevices();
                    this.startChecks();
                } else if (
                    snapshot.state === 'denied' &&
                    previous !== 'denied' &&
                    !this.joinDeniedNotified
                ) {
                    this.joinDeniedNotified = true;
                    this.notify(
                        'error',
                        `${this.joinTargetName()} denied the join request.`,
                    );
                    // Denied is terminal server-side; reset locally so the
                    // panel clears instead of sticking on the refusal.
                    await this.cancelJoin({ silent: true });
                }
            } catch (error) {
                // A 404 means this backend predates the joiner endpoints —
                // stay idle rather than tearing down the rest of the wizard.
                if (error?.status !== 404) {
                    this.join = {
                        ...this.join,
                        error: error?.message || 'Join status unavailable',
                    };
                }
            }
        },

        // =====================================================================
        // Snapshot selectors
        // =====================================================================
        selfDevice() {
            return this.devicesPayload?.self || null;
        },

        pairedDevices() {
            const paired = this.devicesPayload?.paired;
            return Array.isArray(paired) ? paired : [];
        },

        discoveredDevices() {
            const discovered = this.devicesPayload?.discovered;
            if (!Array.isArray(discovered)) return [];
            return discovered.filter(
                (device) =>
                    device && !device.paired && device.state !== 'awaiting_approval',
            );
        },

        pendingApprovals() {
            const discovered = this.devicesPayload?.discovered;
            if (!Array.isArray(discovered)) return [];
            return discovered.filter(
                (device) => device && device.state === 'awaiting_approval',
            );
        },

        allDevices() {
            // N device cards — the v1 GUI's 2-Mac cap is gone.
            const devices = [];
            const self = this.selfDevice();
            if (self) devices.push({ ...self, is_self: true, paired: true });
            for (const device of this.pairedDevices()) devices.push(device);
            for (const device of this.discoveredDevices()) devices.push(device);
            for (const device of this.pendingApprovals()) devices.push(device);
            return devices;
        },

        activeDeployment() {
            return this.deploymentsPayload.length
                ? this.deploymentsPayload[0]
                : null;
        },

        // =====================================================================
        // State machine — empty / discovering / device_card / pairing / checks /
        // plan / active / error
        // =====================================================================
        wizardState() {
            if (this.devicesUnreachable) return 'error';
            if (this.activeDeployment()) return 'active';
            if (this.stage === 'plan' && this.pairedDevices().length) {
                return 'plan';
            }
            if (this.stage === 'checks' && this.pairedDevices().length) {
                return 'checks';
            }
            if (
                this.pairing.target ||
                this.pendingApprovals().length ||
                this.joinActive()
            ) {
                return 'pairing';
            }
            if (
                this.discoveredDevices().length ||
                this.pairedDevices().length
            ) {
                return 'device_card';
            }
            // Identity exists but nobody else is on the network yet — and the
            // pre-first-load skeleton also lands here (it animates either way).
            if (!this.devicesLoaded || this.selfDevice()) return 'discovering';
            return 'empty';
        },

        wizardSteps() {
            const steps = [
                { key: 'discover', title: 'Find devices', hint: 'Automatic on your network' },
                { key: 'pair', title: 'Pair', hint: 'One code, both Macs' },
                { key: 'checks', title: 'Check', hint: 'SSH · model · RDMA' },
                { key: 'plan', title: 'Split the model', hint: this.planStepHint() },
                { key: 'active', title: 'Activate', hint: 'Run across the pool' },
            ];
            // Map the 8 UI states onto the 5 step slots.
            const slotFor = {
                empty: 0,
                discovering: 0,
                device_card: 1,
                pairing: 1,
                checks: 2,
                plan: 3,
                active: 4,
            };
            const activeSlot = slotFor[this.wizardState()] ?? 0;
            return steps.map((step, index) => ({
                ...step,
                state:
                    this.wizardState() === 'error'
                        ? 'todo'
                        : index < activeSlot ||
                          (this.wizardState() === 'active' && index === 4)
                        ? 'done'
                        : index === activeSlot
                        ? 'active'
                        : 'todo',
            }));
        },

        // Step-4 hint is strategy-aware: a tensor split gives every Mac every
        // layer, so "Layers per Mac" would describe the wrong thing.
        planStepHint() {
            return this.planStrategy === 'tensor'
                ? t('cluster.v2.steps.plan_hint_tensor')
                : 'Layers per Mac';
        },

        // =====================================================================
        // Version parity — actionable banner, device stays visible
        // =====================================================================
        versionMismatches() {
            const self = this.selfDevice();
            if (!self || !self.version) return [];
            return this.allDevices()
                .filter(
                    (device) =>
                        !device.is_self &&
                        device.version &&
                        device.version !== self.version,
                )
                .map((device) => ({
                    name: this.deviceName(device),
                    peerVersion: device.version,
                    selfVersion: self.version,
                }));
        },

        // =====================================================================
        // Device card helpers
        // =====================================================================
        deviceName(device) {
            return (
                device?.friendly_name ||
                device?.node_id?.slice(0, 8) ||
                'Unknown device'
            );
        },

        deviceRamGb(device) {
            const ram = device?.caps?.ram_gb;
            return typeof ram === 'number' && ram > 0 ? ram : null;
        },

        deviceRamLabel(device) {
            const ram = this.deviceRamGb(device);
            return ram ? `${ram} GB` : 'Memory unknown';
        },

        deviceChipLabel(device) {
            return device?.caps?.chip || 'Apple silicon';
        },

        deviceLinkMeta(device) {
            return (
                CLUSTER_V2_LINK_META[device?.link] ||
                CLUSTER_V2_LINK_META.unknown
            );
        },

        deviceStateTone(device) {
            if (device?.state === 'dead') {
                return 'bg-red-50 border-red-200 text-red-700';
            }
            if (device?.state === 'suspect') {
                return 'bg-amber-50 border-amber-200 text-amber-700';
            }
            return 'bg-green-50 border-green-200 text-green-700';
        },

        deviceStateLabel(device) {
            if (device?.is_self) return 'This Mac';
            if (device?.state === 'awaiting_approval') return 'Wants to join';
            if (device?.paired) return 'Paired';
            if (device?.state === 'dead') return 'Unreachable';
            if (device?.state === 'suspect') return 'Connection shaky';
            return 'Found nearby';
        },

        combinedMemoryLabel() {
            const total = this.allDevices().reduce(
                (sum, device) => sum + (this.deviceRamGb(device) || 0),
                0,
            );
            return total ? `${total} GB combined` : '';
        },

        // =====================================================================
        // Pairing (Module B)
        // =====================================================================
        beginPairing(device) {
            this.pairing = {
                target: device,
                code: '',
                busy: false,
                error: '',
            };
        },

        cancelPairing() {
            this.pairing = { target: null, code: '', busy: false, error: '' };
        },

        async submitPairApproval(device) {
            const code = (this.pairing.code || '').trim();
            if (!device || this.pairing.busy) return;
            if (!/^\d{6}$/.test(code)) {
                this.pairing.error = 'The code is the 6 digits shown on the other Mac.';
                return;
            }
            this.pairing.busy = true;
            this.pairing.error = '';
            try {
                await this.apiFetch(CLUSTER_V2_API.pairApprove, {
                    method: 'POST',
                    body: JSON.stringify({ node_id: device.node_id, code }),
                });
                this.notify(
                    'success',
                    `${this.deviceName(device)} joined the cluster.`,
                );
                this.cancelPairing();
                this.startChecks();
            } catch (error) {
                if (error?.status === 404 || error?.status === 409) {
                    this.pairing.error =
                        'No join request from this Mac yet. On the other Mac, open its oMLX dashboard and press Pair first — then type its code here.';
                } else if (error?.status === 403 || error?.status === 429) {
                    this.pairing.error =
                        error.message ||
                        'Wrong code too many times — wait for the lockout to lift and try a fresh code.';
                } else {
                    this.pairing.error =
                        error?.message || 'Pairing failed. Try again.';
                }
            } finally {
                this.pairing.busy = false;
            }
        },

        async submitPairDenial(device) {
            if (!device) return;
            try {
                await this.apiFetch(CLUSTER_V2_API.pairDeny, {
                    method: 'POST',
                    body: JSON.stringify({ node_id: device.node_id }),
                });
                this.notify(
                    'info',
                    `Join request from ${this.deviceName(device)} denied.`,
                );
            } catch (error) {
                this.notify(
                    'error',
                    error?.message || 'Could not deny the join request',
                );
            }
            if (this.pairing.target?.node_id === device?.node_id) {
                this.cancelPairing();
            }
        },

        async unpairDevice(device) {
            if (!device) return;
            // Two-step confirm — destructive, but never a modal.
            if (this.confirmUnpairFor !== device.node_id) {
                this.confirmUnpairFor = device.node_id;
                return;
            }
            this.confirmUnpairFor = '';
            try {
                await this.apiFetch(CLUSTER_V2_API.unpair(device.node_id), {
                    method: 'DELETE',
                });
                this.notify(
                    'info',
                    `${this.deviceName(device)} was removed from the cluster.`,
                );
                await this.refreshDevices();
            } catch (error) {
                this.notify(
                    'error',
                    error?.message || 'Could not unpair this device',
                );
            }
        },

        // =====================================================================
        // Joiner side — THIS Mac shows the code; the other Mac approves it.
        // =====================================================================
        joinActive() {
            // Anything but idle keeps the joiner panel up; approved/denied
            // clear themselves through refreshJoinState().
            return !!this.join.state && this.join.state !== 'idle';
        },

        joinTargetName() {
            return (
                this.join.target_name ||
                this.join.coordinator_name ||
                this.join.coordinator_addr ||
                'the other Mac'
            );
        },

        joinCountdownLabel() {
            const total = Math.max(0, Math.round(this.join.seconds_remaining || 0));
            const minutes = Math.floor(total / 60);
            const seconds = String(total % 60).padStart(2, '0');
            return `${minutes}:${seconds}`;
        },

        // exo-style address choice: prefer a routable IPv4 on a direct or
        // known interface; a bare link-local fe80:: (no scope zone) can never
        // be dialed, so it ranks below everything.
        bestDeviceAddr(device) {
            const addrs = Array.isArray(device?.addrs) ? device.addrs : [];
            const preferred = ['manual', 'tb', 'thunderbolt', 'ethernet', 'tailscale'];
            const scored = addrs
                .filter((addr) => addr && addr.ip)
                .map((addr) => {
                    const ip = String(addr.ip);
                    let score = 0;
                    const rank = preferred.indexOf(String(addr.if_type || ''));
                    if (rank >= 0) score += 100 - rank;
                    if (/^\d{1,3}(\.\d{1,3}){3}$/.test(ip)) score += 50;
                    if (ip.toLowerCase().startsWith('fe80:')) score -= 1000;
                    return { addr, score };
                })
                .sort((a, b) => b.score - a.score);
            return scored.length ? scored[0].addr : null;
        },

        coordinatorAddrFor(device) {
            const addr = this.bestDeviceAddr(device);
            if (!addr) return null;
            return `${addr.ip}:${device.http_port || 8000}`;
        },

        async beginJoinAsJoiner(device) {
            const target = this.coordinatorAddrFor(device);
            if (!target) {
                this.notify(
                    'error',
                    `No usable address for ${this.deviceName(device)} yet — try Add by IP.`,
                );
                return;
            }
            await this.beginJoinAddr(target, this.deviceName(device));
        },

        async beginJoinAddr(coordinatorAddr, targetName) {
            if (this.join.busy) return;
            this.join.busy = true;
            try {
                const snapshot = await this.apiFetch(CLUSTER_V2_API.pairJoin, {
                    method: 'POST',
                    body: JSON.stringify({ coordinator_addr: coordinatorAddr }),
                });
                this.join = {
                    ...this.join,
                    ...snapshot,
                    busy: false,
                    target_name: targetName || coordinatorAddr,
                };
                this.joinApprovedNotified = false;
                this.joinDeniedNotified = false;
                this.cancelPairing();
            } catch (error) {
                this.join.busy = false;
                this.notify(
                    'error',
                    error?.message || 'Could not reach that Mac',
                );
            }
        },

        async restartJoin() {
            // "Code expired — start again": same coordinator, fresh code.
            const addr = this.join.coordinator_addr;
            if (!addr) return;
            await this.beginJoinAddr(addr, this.join.target_name);
        },

        async cancelJoin(options = {}) {
            try {
                const snapshot = await this.apiFetch(
                    CLUSTER_V2_API.pairJoinCancel,
                    { method: 'POST' },
                );
                this.join = {
                    ...this.join,
                    ...(snapshot || { state: 'idle' }),
                    busy: false,
                    target_name: '',
                };
                if (!options.silent) {
                    this.notify('info', 'Join cancelled.');
                }
            } catch (error) {
                if (!options.silent) {
                    this.notify(
                        'error',
                        error?.message || 'Could not cancel the join',
                    );
                }
            }
        },

        // =====================================================================
        // Add by IP — the deterministic path when multicast can't reach the
        // other Mac (Thunderbolt pairs, filtered routers, Local Network off).
        // =====================================================================
        async submitManualPeer() {
            if (this.manualBusy) return;
            const raw = (this.manualAddr || '').trim();
            const match = raw.match(/^(\d{1,3}(?:\.\d{1,3}){3})(?::(\d{1,5}))?$/);
            if (!match) {
                this.manualError = 'Enter an IPv4 address like 192.168.1.50 or 192.168.1.50:8000.';
                return;
            }
            const ip = match[1];
            const port = match[2] ? parseInt(match[2], 10) : 8000;
            const octetsOk = ip.split('.').every((part) => Number(part) <= 255);
            if (!octetsOk || !(port >= 1 && port <= 65535)) {
                this.manualError = 'That address or port is out of range.';
                return;
            }
            this.manualBusy = true;
            this.manualError = '';
            try {
                const result = await this.apiFetch(CLUSTER_V2_API.manualDevice, {
                    method: 'POST',
                    body: JSON.stringify({ ip, port }),
                });
                await this.refreshDevices();
                if (result && result.verified) {
                    const name = result.peer?.friendly_name || ip;
                    this.notify('success', `Found ${name} at ${ip}.`);
                    // One click: this Mac shows the code, the other approves.
                    await this.beginJoinAddr(`${ip}:${port}`, name);
                } else {
                    this.notify(
                        'warning',
                        'No oMLX node answered at that address yet — it stays on the list while we keep trying.',
                    );
                }
                this.manualAddr = '';
            } catch (error) {
                this.manualError =
                    error?.message || 'Could not add that address';
            } finally {
                this.manualBusy = false;
            }
        },

        // =====================================================================
        // Automatic checks — SSH, model presence, version parity, rdma_ctl,
        // benchmark, multicast self-test. Each is a row: spinner / pass / fail
        // with a fix.
        // =====================================================================
        startChecks() {
            this.stage = 'checks';
            this.runChecks();
        },

        async runChecks() {
            if (this.checks.running) return;
            this.stage = 'checks';
            this.checks.started = true;
            this.checks.running = true;
            const peers = this.pairedDevices();
            await Promise.all([
                ...peers.map((peer) => this.probePeer(peer)),
                this.refreshDiscoveryHealth(),
            ]);
            this.checks.running = false;
            this.checks.ranAt = Date.now();
        },

        sshTargetFor(device) {
            // Pairing enrollment records the SSH target; the devices payload
            // surfaces it as ssh_target on paired rows. Fall back to the
            // first verified probe address when no enrollment exists yet.
            if (device?.ssh_target) return String(device.ssh_target);
            const addrs = Array.isArray(device?.addrs) ? device.addrs : [];
            // A bare fe80:: link-local address has no scope id here, so SSH
            // to it has no route — prefer any routable address first.
            const usable = addrs.filter(
                (addr) => addr && addr.ip && !String(addr.ip).startsWith('fe80::'),
            );
            const first = usable[0] || addrs.find((addr) => addr && addr.ip);
            return first ? String(first.ip) : this.deviceName(device);
        },

        async probePeer(peer) {
            const ssh = this.sshTargetFor(peer);
            try {
                const result = await this.apiFetch(CLUSTER_V2_API.peerProbe, {
                    method: 'POST',
                    body: JSON.stringify({ ssh }),
                });
                this.checks.probes = {
                    ...this.checks.probes,
                    [peer.node_id]: { ok: true, ssh, result },
                };
            } catch (error) {
                this.checks.probes = {
                    ...this.checks.probes,
                    [peer.node_id]: {
                        ok: false,
                        ssh,
                        error: error?.message || 'Probe failed',
                    },
                };
            }
        },

        async runBenchmark() {
            if (this.checks.benchmarkRunning) return;
            this.checks.benchmarkRunning = true;
            try {
                const result = await this.apiFetch(
                    CLUSTER_V2_API.autoconfigure,
                    {
                        method: 'POST',
                        body: JSON.stringify({
                            nodes: this.planNodes(),
                            hosts: this.deploymentHosts(),
                            // The autoconfigure contract demands exactly one
                            // of model_path / model_size_bytes. The wizard has
                            // no model picked at benchmark time, so a 16 GiB
                            // placeholder satisfies the contract — the probe
                            // measurements (compute + link speeds) are what
                            // actually matter here.
                            model_size_bytes: 16 * 1024 ** 3,
                            measure_performance: true,
                            preflight: false,
                            detect_transports: true,
                        }),
                    },
                );
                this.checks.benchmark = { ok: true, result };
            } catch (error) {
                this.checks.benchmark = {
                    ok: false,
                    error: error?.message || 'Benchmark failed',
                };
            } finally {
                this.checks.benchmarkRunning = false;
            }
        },

        checkRows() {
            const peers = this.pairedDevices();
            const rows = [];
            const allPass = (list) => list.every(Boolean);

            // 1. SSH reachability (peer-probe per paired device).
            {
                const probed = peers.map(
                    (peer) => this.checks.probes[peer.node_id],
                );
                const running = this.checks.running && probed.some((p) => !p);
                const failures = peers.filter(
                    (peer) => this.checks.probes[peer.node_id]?.ok === false,
                );
                rows.push({
                    key: 'ssh',
                    label: 'SSH connection',
                    status: !this.checks.started
                        ? 'pending'
                        : running
                        ? 'running'
                        : peers.length && allPass(probed.map((p) => p?.ok))
                        ? 'pass'
                        : failures.length
                        ? 'fail'
                        : 'running',
                    detail: failures.length
                        ? `Can't reach ${failures
                              .map((peer) => this.deviceName(peer))
                              .join(', ')} over SSH.`
                        : 'Each Mac accepts the cluster key.',
                    fix: `On the failing Mac: System Settings → General → Sharing → turn on Remote Login, then press Re-run checks.`,
                });
            }

            // 2. Model presence (resolved once a model is chosen in the plan
            //    step; the wizard annotates presence per device there).
            {
                const model = this.selectedModel();
                let status = 'skipped';
                let detail = 'Checked automatically when you pick a model.';
                if (model) {
                    const holders = new Set(
                        (model.locations || []).map((loc) => loc.node_id),
                    );
                    const missing = this.allDevices().filter(
                        (device) =>
                            device.paired !== false &&
                            !holders.has(device.node_id) &&
                            !device.is_self &&
                            !(this.selfDevice() && holders.has('127.0.0.1')),
                    );
                    if (missing.length) {
                        status = 'fail';
                        detail = `${this.shortModelName(
                            model,
                        )} is missing on ${missing
                            .map((device) => this.deviceName(device))
                            .join(', ')}.`;
                    } else {
                        status = 'pass';
                        detail = `${this.shortModelName(model)} is on every Mac.`;
                    }
                }
                rows.push({
                    key: 'model',
                    label: 'Model on every Mac',
                    status,
                    detail,
                    fix: 'Open the model on the missing Mac and download it there, or let activation stage the files for you.',
                });
            }

            // 3. Version parity (from the discovery snapshot, live).
            {
                const mismatches = this.versionMismatches();
                rows.push({
                    key: 'version',
                    label: 'Matching oMLX versions',
                    status: mismatches.length ? 'fail' : 'pass',
                    detail: mismatches.length
                        ? mismatches
                              .map(
                                  (m) =>
                                      `${m.name} runs v${m.peerVersion}, this Mac runs v${m.selfVersion}.`,
                              )
                              .join(' ')
                        : 'Every Mac runs the same build.',
                    fix: 'Update the older Mac to the same oMLX build. App: it auto-updates. Brew: brew upgrade omlx. Source: pull the same commit on both Macs.',
                });
            }

            // 4. rdma_ctl / Thunderbolt fabric.
            {
                const fabricMembers = peers.filter(
                    (peer) => peer?.caps?.thunderbolt || peer?.caps?.jaccl,
                );
                if (!fabricMembers.length && !this.selfDevice()?.caps?.jaccl) {
                    rows.push({
                        key: 'rdma',
                        label: 'Thunderbolt RDMA (rdma_ctl)',
                        status: 'skipped',
                        detail:
                            'No Thunderbolt fabric detected — the TCP ring transport will be used instead.',
                        fix: '',
                    });
                } else {
                    const failing = fabricMembers.filter((peer) => {
                        const probe = this.checks.probes[peer.node_id];
                        if (!probe) return false;
                        if (!probe.ok) return true;
                        const rdma =
                            probe.result?.status?.transport?.rdma || {};
                        return !(rdma.devices || []).length;
                    });
                    rows.push({
                        key: 'rdma',
                        label: 'Thunderbolt RDMA (rdma_ctl)',
                        status: this.checks.running
                            ? 'running'
                            : failing.length
                            ? 'fail'
                            : 'pass',
                        detail: failing.length
                            ? `RDMA is not enabled on ${failing
                                  .map((peer) => this.deviceName(peer))
                                  .join(', ')}.`
                            : 'rdma_ctl reports devices on every Thunderbolt Mac.',
                        fix: 'On the failing Mac, connect the Thunderbolt cable and verify with `rdma_ctl status` in Terminal, then Re-run checks. Without it the cluster falls back to the slower TCP ring.',
                    });
                }
            }

            // 5. Benchmark (explicit — it loads the chips for a few seconds).
            {
                const bench = this.checks.benchmark;
                rows.push({
                    key: 'benchmark',
                    label: 'Speed benchmark',
                    status: this.checks.benchmarkRunning
                        ? 'running'
                        : bench
                        ? bench.ok
                            ? 'pass'
                            : 'fail'
                        : 'pending',
                    detail: bench
                        ? bench.ok
                            ? 'Measured compute and link speeds shape the layer split.'
                            : bench.error
                        : 'Optional but recommended — run it once so the split matches real speeds.',
                    fix: 'Benchmark failures are usually a sleeping peer. Wake the other Mac and press Run benchmark again.',
                });
            }

            // 6. Multicast / Local Network self-test (implemented at
            //    discovery_routes.py). Beacon loss is a warning, never a red
            //    failure: discovery degrades to Add by IP while pairing and
            //    the model split keep working — the footer only gates on
            //    SSH + versions, so a red row would contradict "All clear".
            {
                const health = this.discoveryHealth;
                let status = 'pending';
                let detail = 'Checks whether discovery beacons are arriving.';
                let fix =
                    'macOS is blocking local-network beacons. System Settings → Privacy & Security → Local Network → allow oMLX, then restart oMLX. Pairing still works via Add by IP while this is amber.';
                if (health) {
                    if (health.multicast_rx_within_5s) {
                        status = 'pass';
                        detail = 'Discovery beacons are flowing on this network.';
                    } else {
                        status = 'warn';
                        detail =
                            'No discovery beacons received in the last 5 seconds.';
                    }
                } else if (this.discoveryHealthUnsupported) {
                    status = 'skipped';
                    detail =
                        'This build does not report discovery health. Discovery itself may still work.';
                    fix = '';
                } else if (this.checks.running) {
                    status = 'running';
                }
                rows.push({
                    key: 'multicast',
                    label: t('cluster.v2.checks.beacon_label'),
                    status,
                    detail,
                    fix,
                });
            }

            return rows;
        },

        checksBlockingPass() {
            const rows = this.checkRows();
            const byKey = Object.fromEntries(rows.map((row) => [row.key, row]));
            return (
                byKey.ssh?.status === 'pass' &&
                byKey.version?.status === 'pass'
            );
        },

        // =====================================================================
        // Plan — visual layer split from the existing planner
        // =====================================================================
        enterPlan() {
            this.stage = 'plan';
            this.loadNodeRoles();
            if (!this.modelOptions.length && !this.modelsLoading) {
                this.loadModels();
            } else {
                // Models already cached — the recommendation badge still
                // needs its one catalogue call for this plan session.
                this.loadCatalogue();
            }
        },

        // =====================================================================
        // Per-node roles — how much of each Mac the cluster may take
        // =====================================================================
        async loadNodeRoles() {
            if (this.roleOptions.length) return;
            try {
                const payload = await this.apiFetch(CLUSTER_V2_API.nodeRoles);
                this.roleOptions = payload?.roles || [];
            } catch (error) {
                // The mirrored fallback constants keep the budget labels
                // rendering; planning itself re-derives roles server-side.
            }
        },

        roleSpecFor(key) {
            return (
                this.roleOptions.find((role) => role.key === key) ||
                CLUSTER_V2_ROLE_FALLBACK[key] ||
                CLUSTER_V2_ROLE_FALLBACK.headless
            );
        },

        roleLabel(key) {
            return this.roleSpecFor(key).label || key;
        },

        nodeRole(nodeId, isSelf) {
            // The default is unchanged: the Mac whose display is in front of
            // the user keeps a workstation reserve; everything else is
            // headless (also the server default).
            return this.nodeRoles[nodeId] || (isSelf ? 'workstation' : 'headless');
        },

        setNodeRole(device, role) {
            if (!device || this.planLoading) return;
            if (this.nodeRole(device.node_id, !!device.is_self) === role) return;
            this.nodeRoles = { ...this.nodeRoles, [device.node_id]: role };
            this.planFitFailure = null;
            if (this.selectedModelPath) this.runPlan();
        },

        // The paired Macs that will receive layers (this Mac included).
        planRoleDevices() {
            return this.allDevices().filter(
                (device) => device.paired && this.deviceRamGb(device),
            );
        },

        reserveBytesFor(roleKey, capacityBytes) {
            // Mirrors NodeRole.reserve_for in omlx/cluster/node_role.py using
            // the server-provided reserve_bytes/reserve_fraction pair (or the
            // synced fallback mirror above): max(absolute floor, fractional
            // headroom), always leaving the model at least 1 GiB.
            const spec = this.roleSpecFor(roleKey);
            const gib = 1024 ** 3;
            const reserve = Math.max(
                Number(spec.reserve_bytes || 0),
                Math.floor(capacityBytes * Number(spec.reserve_fraction || 0)),
            );
            return Math.min(reserve, Math.max(0, capacityBytes - gib));
        },

        usableGbLabel(device) {
            const capacity = (this.deviceRamGb(device) || 0) * 1024 ** 3;
            if (!capacity) return '';
            const role = this.nodeRole(device.node_id, !!device.is_self);
            const usable = Math.max(
                0,
                capacity - this.reserveBytesFor(role, capacity),
            );
            return `${Math.round(usable / 1024 ** 3)} GB usable as ${this.roleLabel(role)}`;
        },

        // What flipping every workstation node to headless would free up.
        headlessGainBytes() {
            return this.planRoleDevices().reduce((gain, device) => {
                const capacity = (this.deviceRamGb(device) || 0) * 1024 ** 3;
                const role = this.nodeRole(device.node_id, !!device.is_self);
                if (role !== 'workstation') return gain;
                return (
                    gain +
                    this.reserveBytesFor('workstation', capacity) -
                    this.reserveBytesFor('headless', capacity)
                );
            }, 0);
        },

        // /plan 400s with "model does not fit the supplied per-node budgets
        // (at least N additional bytes required)" — turn the number into
        // guidance instead of a dead end.
        parseFitFailure(message) {
            if (!/per-node budgets/.test(message || '')) return null;
            const match = /at least (\d+) additional bytes/.exec(message || '');
            if (!match) return null;
            const shortfallBytes = Number(match[1]);
            return {
                shortfallBytes,
                canFixWithHeadless:
                    shortfallBytes > 0 &&
                    this.headlessGainBytes() >= shortfallBytes,
            };
        },

        fitShortfallLabel() {
            const bytes = this.planFitFailure?.shortfallBytes;
            return bytes ? `${(bytes / 1024 ** 3).toFixed(1)} GiB` : '';
        },

        // One explicit click: roles never flip on their own.
        async switchAllToHeadless() {
            const flipped = { ...this.nodeRoles };
            for (const device of this.planRoleDevices()) {
                flipped[device.node_id] = 'headless';
            }
            this.nodeRoles = flipped;
            this.planFitFailure = null;
            await this.runPlan();
        },

        inventoryHosts() {
            const hosts = [
                {
                    node_id: this.selfDevice()?.node_id || 'coordinator',
                    ssh: '127.0.0.1',
                },
            ];
            for (const peer of this.pairedDevices()) {
                hosts.push({
                    node_id: peer.node_id,
                    ssh: this.sshTargetFor(peer),
                    python_executable:
                        this.checks.probes[peer.node_id]?.result?.status
                            ?.runtime?.python_executable || undefined,
                });
            }
            return hosts;
        },

        async loadModels() {
            this.modelsLoading = true;
            this.modelsError = '';
            try {
                const payload = await this.apiFetch(CLUSTER_V2_API.models, {
                    method: 'POST',
                    body: JSON.stringify({ hosts: this.inventoryHosts() }),
                });
                this.modelOptions = payload?.models || [];
            } catch (error) {
                this.modelsError =
                    error?.message || 'Could not list downloaded models';
            } finally {
                this.modelsLoading = false;
            }
            // Catalogue advice (recommended strategy pill) rides on the model
            // list — fetched once per plan session, silently.
            this.loadCatalogue();
        },

        filteredModels() {
            const query = (this.modelSearch || '').trim().toLowerCase();
            if (!query) return this.modelOptions;
            return this.modelOptions.filter((model) =>
                `${model.id || ''} ${model.model_path || ''} ${
                    model.display_name || ''
                }`
                    .toLowerCase()
                    .includes(query),
            );
        },

        selectedModel() {
            return (
                this.modelOptions.find(
                    (model) => model.model_path === this.selectedModelPath,
                ) || null
            );
        },

        // A name a person recognizes: the inventory's display_name when the
        // server provides one; otherwise a model_path cleaned of Hugging Face
        // cache internals — snapshot-hash tail segments (40–64 hex chars) and
        // the models--org--name directory encoding.
        displayModelName(model) {
            const display = String(model?.display_name || '').trim();
            if (display) return display;
            const raw = String(model?.model_path || model?.id || '');
            const segments = raw.split('/').filter(Boolean);
            while (
                segments.length &&
                /^[0-9a-f]{40,64}$/i.test(segments[segments.length - 1])
            ) {
                segments.pop();
            }
            if (segments[segments.length - 1] === 'snapshots') segments.pop();
            let name = segments.pop() || raw || 'this model';
            const hub = /^models--([^/]+?)--(.+)$/.exec(name);
            if (hub) name = `${hub[1]}/${hub[2]}`;
            return name;
        },

        shortModelName(model) {
            return this.displayModelName(model);
        },

        // Denominator is the Macs that will actually run the split (this Mac
        // + paired peers) — discovered-but-unpaired devices never receive
        // layers, so counting them misreported "on 1 of 3 Macs".
        modelPresenceLabel(model) {
            const holders = new Set(
                (model?.locations || []).map((loc) => loc.node_id),
            );
            const total =
                (this.selfDevice() ? 1 : 0) + this.pairedDevices().length;
            if (!total) return '';
            const have = Math.min(holders.size, total);
            if (have >= total) return t('cluster.v2.models.on_every_mac');
            return t('cluster.v2.models.partial')
                .replace('{have}', String(have))
                .replace('{total}', String(total));
        },

        async selectModel(model) {
            if (!model || this.planLoading) return;
            this.selectedModelPath = model.model_path;
            this.normalizePlanStrategy();
            await this.runPlan();
        },

        // =====================================================================
        // Execution strategy — auto / tensor / pipeline, server-recommended
        // =====================================================================
        // Tensor parallelism must span every node (routes.py rejects TP !=
        // len(nodes)), so the size is always the full pool.
        planTensorParallelSize() {
            return this.planStrategy === 'tensor'
                ? this.planNodes().length
                : 1;
        },

        catalogueEntryForModel() {
            if (!Array.isArray(this.catalogueModels)) return null;
            const model = this.selectedModel();
            if (!model) return null;
            return (
                this.catalogueModels.find(
                    (entry) => entry.model_path === model.model_path,
                ) || null
            );
        },

        // One 'Recommended' pill, on exactly one option: the strategy the
        // catalogue names for the picked model, or — when the catalogue could
        // not advise — tensor iff every member sits on a fast (jaccl /
        // Thunderbolt) transport, mirroring resolvedBackend().
        recommendedStrategy() {
            const entry = this.catalogueEntryForModel();
            if (entry) {
                if (entry.strategy === 'tensor') return 'tensor';
                if (entry.strategy === 'pipeline') return 'pipeline';
                // 'hybrid' / 'single node' cannot be expressed by the picker.
                return 'auto';
            }
            if (
                this.catalogueFailed ||
                Array.isArray(this.catalogueModels)
            ) {
                // Fallback heuristic: tensor only pays off when every member
                // sits on a fast (jaccl / Thunderbolt) transport — the same
                // rule resolvedBackend() applies; anything else, auto.
                return this.resolvedBackend() === 'jaccl' ? 'tensor' : 'auto';
            }
            return '';
        },

        strategyOptions() {
            const entry = this.catalogueEntryForModel();
            const nodeCount = this.planNodes().length;
            const tensorUnsupported =
                !!entry && entry.supports_tensor_parallel === false;
            const pipelineUnsupported =
                !!entry && entry.supports_pipeline === false;
            const tensorDisabled = nodeCount < 2 || tensorUnsupported;
            return [
                {
                    key: 'auto',
                    label: t('cluster.v2.strategy.auto'),
                    disabled: false,
                    disabledReason: '',
                },
                {
                    key: 'tensor',
                    label: t('cluster.v2.strategy.tensor'),
                    disabled: tensorDisabled,
                    disabledReason: !tensorDisabled
                        ? ''
                        : nodeCount < 2
                        ? t('cluster.v2.strategy.tensor_needs_two')
                        : t('cluster.v2.strategy.tensor_unsupported'),
                },
                {
                    key: 'pipeline',
                    label: t('cluster.v2.strategy.pipeline'),
                    disabled: pipelineUnsupported,
                    disabledReason: pipelineUnsupported
                        ? t('cluster.v2.strategy.pipeline_unsupported')
                        : '',
                },
            ];
        },

        selectedStrategyOption() {
            return (
                this.strategyOptions().find(
                    (option) => option.key === this.planStrategy,
                ) || { key: 'auto', disabledReason: '' }
            );
        },

        strategyHint() {
            return t(`cluster.v2.strategy.hint.${this.planStrategy}`);
        },

        setPlanStrategy(key) {
            const option = this.strategyOptions().find(
                (item) => item.key === key,
            );
            if (!option || option.disabled) return;
            if (this.planStrategy === key) return;
            this.planStrategy = key;
            if (this.selectedModelPath) this.runPlan();
        },

        // A model change (or catalogue arrival) can invalidate the current
        // pick — e.g. pipeline selected for a pipeline-incapable model.
        normalizePlanStrategy() {
            const option = this.strategyOptions().find(
                (item) => item.key === this.planStrategy,
            );
            if (option?.disabled) this.planStrategy = 'auto';
        },

        // Advisory only: on failure the badge falls back to the transport
        // heuristic. No toasts — a missing recommendation is not an error.
        async loadCatalogue() {
            if (
                this.catalogueLoading ||
                this.catalogueModels !== null ||
                this.catalogueFailed
            ) {
                return;
            }
            const candidates = this.modelOptions
                .filter((model) => model?.model_path)
                .map((model) => {
                    const sourceLocation =
                        (model.locations || []).find(
                            (loc) => loc.ssh === model.model_source,
                        ) || {};
                    return {
                        id: String(model.id || model.model_path),
                        model_path: model.model_path,
                        model_source: model.model_source || '127.0.0.1',
                        model_source_python:
                            sourceLocation.python_executable || undefined,
                        source_node_id: model.source_node_id || '',
                        model_context_length:
                            model.model_context_length || undefined,
                    };
                });
            if (!candidates.length) return;
            this.catalogueLoading = true;
            try {
                const payload = await this.apiFetch(CLUSTER_V2_API.catalogue, {
                    method: 'POST',
                    body: JSON.stringify({
                        nodes: this.planNodes(),
                        models: candidates,
                        execution_profile: 'balanced',
                    }),
                });
                this.catalogueModels = payload?.models || [];
                this.normalizePlanStrategy();
            } catch (error) {
                this.catalogueFailed = true;
            } finally {
                this.catalogueLoading = false;
            }
        },

        // Tensor plans give every node ALL layers with a tensor_parallel_rank;
        // the contiguous-range split bar would lie about them.
        planIsTensor() {
            return (this.plan?.tensor_parallel_size || 1) > 1;
        },

        tensorShareLabel() {
            return t('cluster.v2.split.tensor_share').replace(
                '{count}',
                String(this.plan?.tensor_parallel_size || 1),
            );
        },

        tensorCaptionLabel() {
            return t('cluster.v2.split.tensor_caption').replace(
                '{count}',
                String(this.plan?.tensor_parallel_size || 1),
            );
        },

        planNodes() {
            const nodes = [];
            const self = this.selfDevice();
            if (self) {
                nodes.push({
                    node_id: self.node_id,
                    capacity_bytes:
                        (this.deviceRamGb(self) || 0) * 1024 ** 3,
                    // The Mac whose display is in front of the user keeps a
                    // workstation reserve unless the plan step says otherwise.
                    role: this.nodeRole(self.node_id, true),
                    memory_guard_tier: 'balanced',
                    accelerator: 'metal',
                });
            }
            for (const peer of this.pairedDevices()) {
                nodes.push({
                    node_id: peer.node_id,
                    capacity_bytes:
                        (this.deviceRamGb(peer) || 0) * 1024 ** 3,
                    role: this.nodeRole(peer.node_id, false),
                    memory_guard_tier: 'balanced',
                    accelerator: 'metal',
                });
            }
            return nodes.filter((node) => node.capacity_bytes > 0);
        },

        async runPlan() {
            if (!this.selectedModelPath) return;
            this.planLoading = true;
            this.planError = '';
            this.planFitFailure = null;
            this.plan = null;
            try {
                const model = this.selectedModel();
                const body = {
                    model_path: this.selectedModelPath,
                    nodes: this.planNodes(),
                    execution_profile: 'balanced',
                    // 'tensor' spans every node (routes.py requires TP ==
                    // len(nodes)); 'auto'/'pipeline' plan pipeline-only.
                    tensor_parallel_size: this.planTensorParallelSize(),
                };
                if (
                    model?.model_source &&
                    model.model_source !== '127.0.0.1'
                ) {
                    body.model_source = model.model_source;
                }
                this.plan = await this.apiFetch(CLUSTER_V2_API.plan, {
                    method: 'POST',
                    body: JSON.stringify(body),
                });
            } catch (error) {
                this.planError =
                    error?.message || 'Could not build the layer split';
                this.planFitFailure = this.parseFitFailure(this.planError);
            } finally {
                this.planLoading = false;
            }
        },

        planAssignments() {
            const assignments = this.plan?.assignments;
            return Array.isArray(assignments) ? assignments : [];
        },

        planTotalLayers() {
            return this.planAssignments().reduce(
                (sum, assignment) => sum + (assignment.layer_count || 0),
                0,
            );
        },

        planNodeName(assignment) {
            const device = this.allDevices().find(
                (item) => item.node_id === assignment.node_id,
            );
            return device
                ? this.deviceName(device)
                : assignment.node_id || `Rank ${assignment.rank}`;
        },

        resolvedBackend() {
            const members = [this.selfDevice(), ...this.pairedDevices()].filter(
                Boolean,
            );
            const allJaccl =
                members.length > 1 &&
                members.every((device) => device?.caps?.jaccl);
            return allJaccl ? 'jaccl' : 'ring';
        },

        backendLabel() {
            return this.resolvedBackend() === 'jaccl'
                ? 'JACCL · Thunderbolt RDMA'
                : 'TCP ring';
        },

        deploymentHosts() {
            const hosts = [];
            const self = this.selfDevice();
            if (self) {
                hosts.push({
                    node_id: self.node_id,
                    ssh: '127.0.0.1',
                    ips: (self.addrs || [])
                        .map((addr) => addr?.ip)
                        .filter(Boolean),
                    rdma: [],
                });
            }
            for (const peer of this.pairedDevices()) {
                hosts.push({
                    node_id: peer.node_id,
                    ssh: this.sshTargetFor(peer),
                    ips: (peer.addrs || [])
                        .map((addr) => addr?.ip)
                        .filter(Boolean),
                    rdma: [],
                    python_executable:
                        this.checks.probes[peer.node_id]?.result?.status
                            ?.runtime?.python_executable || undefined,
                });
            }
            return hosts.filter((host) => host.ips.length || host.ssh);
        },

        async activatePlan() {
            if (!this.plan || this.activateBusy) return;
            this.activateBusy = true;
            try {
                const model = this.selectedModel();
                const body = {
                    model_path: this.selectedModelPath,
                    backend: this.resolvedBackend(),
                    nodes: this.planNodes(),
                    hosts: this.deploymentHosts(),
                    approved_placement: this.plan.placement_signature,
                    execution_profile: 'balanced',
                    // Must match what was planned — the signed placement
                    // covers tensor_parallel_size, so a mismatch 409s.
                    tensor_parallel_size: this.planTensorParallelSize(),
                };
                if (
                    model?.model_source &&
                    model.model_source !== '127.0.0.1'
                ) {
                    body.model_source = model.model_source;
                }
                await this.apiFetch(CLUSTER_V2_API.deployments, {
                    method: 'POST',
                    body: JSON.stringify(body),
                });
                this.notify(
                    'success',
                    'Cluster activated. The model starts across your Macs on next load.',
                );
                this.stage = null;
                this.plan = null;
                await this.refreshDeployments();
            } catch (error) {
                if (error?.status === 409) {
                    this.notify(
                        'warning',
                        'The plan changed since you reviewed it — rebuilding it now.',
                    );
                    await this.runPlan();
                } else {
                    this.notify(
                        'error',
                        error?.message || 'Activation failed',
                    );
                }
            } finally {
                this.activateBusy = false;
            }
        },

        async deactivateDeployment(deployment) {
            const id = deployment?.deployment_id;
            if (!id) return;
            if (this.confirmDeactivateFor !== id) {
                this.confirmDeactivateFor = id;
                return;
            }
            this.confirmDeactivateFor = '';
            try {
                await this.apiFetch(CLUSTER_V2_API.deployment(id), {
                    method: 'DELETE',
                });
                this.notify('info', 'Cluster deactivated.');
                await this.refreshDeployments();
            } catch (error) {
                this.notify(
                    'error',
                    error?.message || 'Could not deactivate',
                );
            }
        },

        // =====================================================================
        // Feedback — toasts, never modals
        // =====================================================================
        notify(type, message, timeoutMs = 6000) {
            const id = ++this.toastSeq;
            this.toasts.push({ id, type, message });
            setTimeout(() => {
                this.toasts = this.toasts.filter((toast) => toast.id !== id);
            }, timeoutMs);
        },

        dismissToast(id) {
            this.toasts = this.toasts.filter((toast) => toast.id !== id);
        },

        toastTone(type) {
            return {
                success: 'border-green-200 bg-green-50 text-green-800',
                error: 'border-red-200 bg-red-50 text-red-800',
                warning: 'border-amber-200 bg-amber-50 text-amber-800',
                info: 'border-neutral-200 bg-white text-neutral-800',
            }[type] || 'border-neutral-200 bg-white text-neutral-800';
        },

        toastIcon(type) {
            return {
                success: 'check-circle-2',
                error: 'circle-alert',
                warning: 'triangle-alert',
                info: 'info',
            }[type] || 'info';
        },

        retryConnection() {
            this.devicesUnreachable = false;
            this.devicesFailureCount = 0;
            this.refreshDevices();
        },

        copyInstallCommand() {
            const command = 'brew install jundot/omlx/omlx';
            const done = () => {
                this.installCommandCopied = true;
                setTimeout(() => (this.installCommandCopied = false), 2000);
            };
            if (navigator.clipboard?.writeText) {
                navigator.clipboard.writeText(command).then(done, done);
            } else {
                done();
            }
        },
    };
}
