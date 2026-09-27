"""Blocked-sequential KDA recurrence for GLM-5.3 prefill.

GLM-5.3's linear-attention layers run the vector-gated delta rule (Kimi
delta attention): per token and head, with a per-channel forget gate g,

    S = S * g                 (decay, broadcast over value rows)
    p = S . k                 (128-wide dot per value row)
    S = S + k * (v - p) * beta
    y = S . q

The stock prefill evaluates it with ``gated_delta_kernel``: one simdgroup
per value row, g materialized as an fp32 [T, H, 128] tensor by
``compute_g_safe``. This kernel runs the same recurrence, restructured for
Apple GPUs:

- 64 value rows per threadgroup; each thread keeps 2 rows x 16 channels of
  the fp32 state in registers, the 8 channel segments of a row group sit in
  one simdgroup, and the k.S / q.S partials are reduced with a
  reduce-scatter over those 8 lanes (no barriers inside the time loop);
- q/k/v/beta are staged in threadgroup memory per 16-token block, read once
  per threadgroup, with the next block prefetched into registers while the
  current one is processed;
- the gate exp(lb * sigmoid(exp(A_log) * (a + dt_bias))) is evaluated while
  staging with ``compute_g_safe``'s expression tree and MLX's Sigmoid
  functor (bit-identical gate), so the fp32 gate tensor never round-trips
  through memory.

Same per-step math as ``gated_delta_kernel``; only the fp32 summation order
of the two 128-wide dots differs.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import mlx.core as mx


class RecurrenceConfig(NamedTuple):
    """Launch shape of the blocked recurrence (every variant is exact)."""

    tb: int = 16  # tokens staged per threadgroup block
    db: int = 64  # value rows per threadgroup
    rows: int = 2  # value rows per thread (1 or 2)
    prefetch: bool = True  # load the next block into registers meanwhile

    @property
    def threads(self) -> int:
        return self.db // self.rows * 8


DEFAULT_CONFIG = RecurrenceConfig()

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def _reduce_all(rows: int) -> str:
    """Sum part[rows] over the 8 lanes of a row group into P[rows] on every lane."""
    if rows == 1:
        return """
            float P[1] = {part[0]};
            P[0] += simd_shuffle_xor(P[0], 4);
            P[0] += simd_shuffle_xor(P[0], 2);
            P[0] += simd_shuffle_xor(P[0], 1);"""
    # Reduce-scatter (each half of the lanes finishes one row), then swap.
    return """
            const bool h2 = (seg & 4) != 0;
            float keep = h2 ? part[1] : part[0];
            keep += simd_shuffle_xor(h2 ? part[0] : part[1], 4);
            keep += simd_shuffle_xor(keep, 2);
            keep += simd_shuffle_xor(keep, 1);
            const float other = simd_shuffle_xor(keep, 4);
            float P[2] = {h2 ? other : keep, h2 ? keep : other};"""


def _reduce_one(rows: int) -> str:
    """Sum part[rows] over the row lanes; `writer` lanes hold row `own` in `keep`."""
    if rows == 1:
        return """
            float keep = part[0];
            keep += simd_shuffle_down(keep, 4);
            keep += simd_shuffle_down(keep, 2);
            keep += simd_shuffle_down(keep, 1);
            const int own = 0;
            const bool writer = seg == 0;"""
    return """
            const bool h2 = (seg & 4) != 0;
            float keep = h2 ? part[1] : part[0];
            keep += simd_shuffle_xor(h2 ? part[0] : part[1], 4);
            keep += simd_shuffle_xor(keep, 2);
            keep += simd_shuffle_xor(keep, 1);
            const int own = h2 ? 1 : 0;
            const bool writer = (seg & 3) == 0;"""


# compute_g_safe: exp(lb * sigmoid(exp(A_log) * (a + dt_bias))) with MLX's
# Sigmoid functor, rounded like the compiled reference -> bit-identical gate.
_GATE_EXPR = """{
                const float x = decay * (static_cast<float>(SRC) + dtb[d]);
                const float e = 1 / (1 + metal::exp(metal::abs(x)));
                const float sig = (x < 0) ? e : 1 - e;
                g_s[r][d] = metal::precise::exp(lb * sig);
            }"""


def _source(cfg: RecurrenceConfig) -> str:
    if cfg.rows not in (1, 2) or cfg.db % (cfg.rows * 4) or cfg.threads > 1024:
        raise ValueError(f"unsupported recurrence config {cfg}")
    if cfg.prefetch:
        prologue = """
    constexpr int NQK = (TB * Dk + NT - 1) / NT;
    constexpr int NV = (TB * DB + NT - 1) / NT;
    InT pk[NQK];
    InT pq[NQK];
    InT pa[NQK];
    InT pv[NV];
    InT pb = InT(0);
#define KDA_FETCH(T0N) { \\
        const int ttn = min(TB, T - (T0N)); \\
        for (int j = 0; j < NQK; ++j) { \\
            const int p = tid + j * NT; \\
            if (p < ttn * Dk) { \\
                const int r = p / Dk, d = p % Dk; \\
                const size_t off = (size_t)((T0N) + r) * qk_row + d; \\
                pk[j] = k_base[off]; \\
                pq[j] = q_base[off]; \\
                pa[j] = a_base[off]; \\
            } \\
        } \\
        for (int j = 0; j < NV; ++j) { \\
            const int p = tid + j * NT; \\
            if (p < ttn * DB) { \\
                pv[j] = v_base[(size_t)((T0N) + p / DB) * v_row + p % DB]; \\
            } \\
        } \\
        if (tid < ttn) pb = beta_base[(size_t)((T0N) + tid) * H]; \\
    }
    KDA_FETCH(0)"""
        staging = f"""
        for (int j = 0; j < NQK; ++j) {{
            const int p = tid + j * NT;
            if (p < tt * Dk) {{
                const int r = p / Dk, d = p % Dk;
                k_s[r][d] = pk[j];
                q_s[r][d] = pq[j];
                {_GATE_EXPR.replace("SRC", "pa[j]")}
            }}
        }}
        for (int j = 0; j < NV; ++j) {{
            const int p = tid + j * NT;
            if (p < tt * DB) {{
                v_s[p / DB][p % DB] = pv[j];
            }}
        }}
        if (tid < tt) b_s[tid] = static_cast<float>(pb);"""
        after_barrier = "        if (t0 + TB < T) KDA_FETCH(t0 + TB)"
    else:
        prologue = ""
        staging = f"""
        for (int p = tid; p < tt * Dk; p += NT) {{
            const int r = p / Dk, d = p % Dk;
            const size_t off = (size_t)(t0 + r) * qk_row + d;
            k_s[r][d] = k_base[off];
            q_s[r][d] = q_base[off];
            {_GATE_EXPR.replace("SRC", "a_base[off]")}
        }}
        for (int p = tid; p < tt * DB; p += NT) {{
            v_s[p / DB][p % DB] = v_base[(size_t)(t0 + p / DB) * v_row + p % DB];
        }}
        for (int p = tid; p < tt; p += NT) {{
            b_s[p] = static_cast<float>(beta_base[(size_t)(t0 + p) * H]);
        }}"""
        after_barrier = ""

    return f"""
    constexpr int TB = {cfg.tb};
    constexpr int DB = {cfg.db};
    constexpr int R = {cfg.rows};
    constexpr int NT = {cfg.threads};
    const int tid = thread_position_in_threadgroup.x;
    const int blk = threadgroup_position_in_grid.x;
    const int h = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int dv0 = blk * DB;
    // thread -> (group of R value rows, 16-channel segment); the 8 segment
    // lanes of a row group are adjacent in one simdgroup.
    const int rg = tid / 8;
    const int seg = tid % 8;
    const int d0 = seg * 16;
    threadgroup InT k_s[TB][Dk + 8];
    threadgroup InT q_s[TB][Dk + 8];
    threadgroup InT v_s[TB][DB + 8];
    threadgroup float b_s[TB];
    threadgroup float g_s[TB][Dk + 4];

    const size_t qk_row = (size_t)H * Dk;
    const size_t v_row = (size_t)H * Dv;
    const device InT* k_base = k + ((size_t)b * T * H + h) * Dk;
    const device InT* q_base = q + ((size_t)b * T * H + h) * Dk;
    const device InT* a_base = a + ((size_t)b * T * H + h) * Dk;
    const device InT* v_base = v + ((size_t)b * T * H + h) * Dv + dv0;
    const device InT* beta_base = beta + (size_t)b * T * H + h;
    const device float* dtb = dt_bias + (size_t)h * Dk;
    const float decay = metal::precise::exp(A_log[h]);
    const float lb = lower_bound[0];

    float4 st[R][4];
    for (int r = 0; r < R; ++r) {{
        const device float4* S_in = (const device float4*)(
            state_in + (((size_t)b * H + h) * Dv + dv0 + rg * R + r) * Dk + d0);
        for (int i = 0; i < 4; ++i) st[r][i] = S_in[i];
    }}
    device InT* y_base = y + ((size_t)b * T * H + h) * Dv + dv0 + rg * R;{prologue}

    for (int t0 = 0; t0 < T; t0 += TB) {{
        const int tt = min(TB, T - t0);{staging}
        threadgroup_barrier(mem_flags::mem_threadgroup);
{after_barrier}
        for (int t = 0; t < tt; ++t) {{
            const threadgroup float4* g4 = (const threadgroup float4*)&g_s[t][d0];
            const float bt = b_s[t];
            const threadgroup vec<InT, 4>* k4 = (const threadgroup vec<InT, 4>*)&k_s[t][d0];
            const threadgroup vec<InT, 4>* q4 = (const threadgroup vec<InT, 4>*)&q_s[t][d0];
            float4 kf[4];
            for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
            float part[R];
            // decay the state rows in place, then k.S
            for (int r = 0; r < R; ++r) {{
                float4 p4 = 0.0f;
                for (int i = 0; i < 4; ++i) {{
                    st[r][i] = st[r][i] * g4[i];
                    p4 += st[r][i] * kf[i];
                }}
                part[r] = (p4.x + p4.y) + (p4.z + p4.w);
            }}
            {{{_reduce_all(cfg.rows)}
                // delta rule update, then q.S
                for (int r = 0; r < R; ++r) {{
                    const float delta =
                        (static_cast<float>(v_s[t][rg * R + r]) - P[r]) * bt;
                    float4 o4 = 0.0f;
                    for (int i = 0; i < 4; ++i) {{
                        st[r][i] = st[r][i] + kf[i] * delta;
                        o4 += st[r][i] * float4(q4[i]);
                    }}
                    part[r] = (o4.x + o4.y) + (o4.z + o4.w);
                }}
            }}
            {{{_reduce_one(cfg.rows)}
                if (writer) {{
                    y_base[(size_t)(t0 + t) * v_row + own] = static_cast<InT>(keep);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (int r = 0; r < R; ++r) {{
        device float4* S_out = (device float4*)(
            state_out + (((size_t)b * H + h) * Dv + dv0 + rg * R + r) * Dk + d0);
        for (int i = 0; i < 4; ++i) S_out[i] = st[r][i];
    }}
"""


_KERNELS: dict = {}


def _kernel(cfg: RecurrenceConfig):
    kernel = _KERNELS.get(cfg)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=(
                f"omlx_glm53_kda_recurrence_tb{cfg.tb}_db{cfg.db}_r{cfg.rows}"
                f"{'_pf' if cfg.prefetch else ''}"
            ),
            input_names=[
                "q", "k", "v", "a", "beta", "A_log", "dt_bias", "lower_bound",
                "state_in", "T",
            ],
            output_names=["y", "state_out"],
            source=_source(cfg),
            header=_HEADER,
        )
        _KERNELS[cfg] = kernel
    return kernel


def kda_recurrence(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    beta: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    lower_bound: float,
    state: mx.array,
    config: Optional[RecurrenceConfig] = None,
) -> Tuple[mx.array, mx.array]:
    """Vector-gated delta rule over a prompt chunk (safe-gate variant).

    q, k, a: [B, T, H, 128]; v: [B, T, H, 128]; beta: [B, T, H] (sigmoid
    already applied); a_log: [H] fp32; dt_bias: [H * 128] fp32; state:
    [B, H, 128, 128] fp32. Returns y [B, T, H, 128] (q.dtype) and the fp32
    state, like ``gated_delta_update(..., lower_bound=lower_bound)``.
    """
    B, T, H, Dk = q.shape
    Dv = v.shape[-1]
    if Dk != 128 or Dv != 128:
        raise ValueError("kda_recurrence needs 128-wide heads")
    cfg = config or DEFAULT_CONFIG
    dtype = q.dtype
    return _kernel(cfg)(
        inputs=[
            q,
            k,
            v,
            a.astype(dtype),
            beta.astype(dtype),
            a_log.reshape(H).astype(mx.float32),
            dt_bias.reshape(H, Dk).astype(mx.float32),
            mx.array([lower_bound], dtype=mx.float32),
            state.astype(mx.float32),
            T,
        ],
        template=[("InT", dtype), ("Dk", Dk), ("Dv", Dv), ("H", H)],
        grid=(cfg.threads * (Dv // cfg.db), H, B),
        threadgroup=(cfg.threads, 1, 1),
        output_shapes=[(B, T, H, Dv), (B, H, Dv, Dk)],
        output_dtypes=[dtype, mx.float32],
    )
