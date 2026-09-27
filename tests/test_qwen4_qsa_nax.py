# SPDX-License-Identifier: Apache-2.0
"""Tensor-unit Qwen4 QSA main attention (one query per threadgroup)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.custom_kernels.nax import is_nax_available
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import qsa_fast, qsa_nax  # noqa: E402

needs_nax = pytest.mark.skipif(
    not is_nax_available(), reason="tensor-unit (NAX) GPU required"
)
needs_native = pytest.mark.skipif(
    not (
        fast.is_native_available()
        and fast.has_symbol("qwen4_qsa_sparse_gqa_attention")
    ),
    reason="native Qwen4 QSA kernel not built",
)


def _selections(rng, lq, q_offset, shared=2.0):
    """Ascending top-512 blocks per query; canonical 0..511 below the budget."""
    blocks = (q_offset + lq) // 4 + 1
    base = rng.standard_normal(blocks).astype(np.float32)
    sel = np.zeros((lq, 512), dtype=np.int32)
    for t in range(lq):
        complete = (q_offset + t + 1) // 4
        if complete <= 512:
            sel[t] = np.arange(512)
            continue
        scores = base[:complete] * shared + rng.standard_normal(complete)
        sel[t] = np.sort(np.argpartition(scores, -512)[-512:])
    return sel


def _reference(q, k, v, sel, q_offset):
    """fp64 QSA reference: each query attends its valid blocks then its tail."""
    q = np.array(q.astype(mx.float32), dtype=np.float64)[0]
    k = np.array(k.astype(mx.float32), dtype=np.float64)[0]
    v = np.array(v.astype(mx.float32), dtype=np.float64)[0]
    lq = q.shape[1]
    out = np.zeros((lq, 24, 256))
    for t in range(lq):
        p = q_offset + t
        complete = (p + 1) // 4
        valid = sel[t, : min(512, complete)]
        toks = np.concatenate(
            [(valid[:, None] * 4 + np.arange(4)).reshape(-1), np.arange(complete * 4, p + 1)]
        )
        for h in range(24):
            s = k[h // 12, toks] @ q[h, t] / 16.0
            e = np.exp(s - s.max())
            out[t, h] = (e / e.sum()) @ v[h // 12, toks]
    return out


@needs_nax
@needs_native
@pytest.mark.parametrize("pv_mode", ["half2", "bf16x3"])
@pytest.mark.parametrize(
    ("lq", "q_offset", "prefix"),
    # Consecutive queries cycle through all four tail lengths (0..3 tokens);
    # (33, 0) has selections shorter than 512 blocks; (6, 70000) is long context.
    [(37, 2100, 5), (33, 0, 0), (19, 8171, 3), (6, 70000, 1)],
)
def test_nax_attention_matches_native_and_fp64_reference(
    monkeypatch, pv_mode, lq, q_offset, prefix
):
    monkeypatch.setattr(qsa_nax, "PV_MODE", pv_mode)
    rng = np.random.default_rng(7 + lq)
    mx.random.seed(lq)
    kl = q_offset + lq
    # Strided views like production: a query slice and K/V with spare capacity.
    q_all = mx.random.normal((1, 24, prefix + lq, 256)).astype(mx.bfloat16)
    q = q_all[:, :, prefix:]
    kbuf = mx.random.normal((1, 2, kl + 29, 256)).astype(mx.bfloat16)
    vbuf = mx.random.normal((1, 2, kl + 29, 256)).astype(mx.bfloat16)
    k, v = kbuf[:, :, :kl], vbuf[:, :, :kl]
    sel = _selections(rng, lq, q_offset)
    sel_mx = mx.array(sel)[None]

    got = qsa_nax.sparse_gqa_attention(q, k, v, sel_mx, q_offset=q_offset)
    native = fast.qwen4_qsa_sparse_gqa_attention(
        q,
        k,
        v,
        mx.contiguous(sel_mx.astype(mx.uint32)[:, None]),
        256**-0.5,
        q_offset,
        key_tile=64,
        dimension_tile=64,
    ).transpose(0, 2, 1, 3)
    mx.eval(got, native)
    ref = _reference(q, k, v, sel, q_offset)
    g = np.array(got.astype(mx.float32))[0]
    n = np.array(native.astype(mx.float32))[0]
    # Both round an fp32 result to bf16: identical error against fp64 up to
    # fp32 summation order (at most one bf16 ulp apart, almost always equal).
    half_ulp = np.abs(ref) * 2.0**-8 + 1e-6
    assert np.mean(np.abs(g - ref) > half_ulp) <= 2 * np.mean(np.abs(n - ref) > half_ulp) + 1e-3
    assert np.max(np.abs(g - n) / (np.abs(n) * 2.0**-7 + 1e-6)) <= 1.0 + 1e-6
    assert np.mean(g == n) > 0.99


@needs_nax
@needs_native
@pytest.mark.parametrize("kl_mod", [1, 2, 3])
def test_rows_past_kl_are_never_consumed(kl_mod):
    """The last tail block may extend past kL; spare cache capacity holding NaN
    must not leak into any output (the native kernel never reads it)."""
    rng = np.random.default_rng(kl_mod)
    mx.random.seed(kl_mod)
    lq = 13
    q_offset = 4096 + kl_mod - lq
    kl = q_offset + lq
    assert kl % 4 == kl_mod
    q = mx.random.normal((1, 24, lq, 256)).astype(mx.bfloat16)
    spare = mx.full((1, 2, 7, 256), float("nan"), dtype=mx.bfloat16)
    kbuf = mx.concatenate([mx.random.normal((1, 2, kl, 256)).astype(mx.bfloat16), spare], axis=2)
    vbuf = mx.concatenate([mx.random.normal((1, 2, kl, 256)).astype(mx.bfloat16), spare], axis=2)
    k, v = kbuf[:, :, :kl], vbuf[:, :, :kl]
    sel = mx.array(_selections(rng, lq, q_offset))[None]
    got = qsa_nax.sparse_gqa_attention(q, k, v, sel, q_offset=q_offset)
    native = fast.qwen4_qsa_sparse_gqa_attention(
        q, k, v, mx.contiguous(sel.astype(mx.uint32)[:, None]), 256**-0.5, q_offset,
        key_tile=64, dimension_tile=64,
    ).transpose(0, 2, 1, 3)
    mx.eval(got, native)
    assert not mx.any(mx.isnan(got)).item()
    assert mx.allclose(got.astype(mx.float32), native.astype(mx.float32), atol=1e-2).item()


@needs_nax
@needs_native
def test_gathered_qsa_routes_through_nax_and_matches_native(monkeypatch):
    rng = np.random.default_rng(3)
    mx.random.seed(3)
    lq, q_offset = 45, 2200
    kl = q_offset + lq
    q = mx.random.normal((1, 24, lq, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, kl, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, kl, 256)).astype(mx.bfloat16)
    sel = mx.array(_selections(rng, lq, q_offset))[None]
    calls = []
    original = qsa_nax.sparse_gqa_attention

    def spy(*args, **kwargs):
        calls.append(kwargs["q_offset"])
        return original(*args, **kwargs)

    monkeypatch.setattr(qsa_nax, "sparse_gqa_attention", spy)
    monkeypatch.setattr(qsa_fast, "_NAX_QSA_MAIN_DISABLED", False)
    routed = qsa_fast._nax_sparse_gqa_attention(q, k, v, sel, q_offset=q_offset)
    native = qsa_fast._native_sparse_gqa_attention(q, k, v, sel, q_offset=q_offset)
    assert calls == [q_offset]
    assert routed is not None and native is not None
    mx.eval(routed, native)
    assert routed.shape == native.shape == (1, lq, 24, 256)
    diff = mx.abs(routed.astype(mx.float32) - native.astype(mx.float32))
    assert float(mx.max(diff / (mx.abs(native.astype(mx.float32)) * 2.0**-7 + 1e-6))) <= 1.0 + 1e-6


def test_nax_route_fails_closed(monkeypatch):
    q = mx.zeros((1, 24, 32, 256), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, 4096, 256), dtype=mx.bfloat16)
    sel = mx.zeros((1, 32, 512), dtype=mx.int32)
    monkeypatch.setattr(qsa_fast, "_NAX_QSA_MAIN_DISABLED", False)
    monkeypatch.setattr(qsa_nax, "nax_available", lambda: True)
    monkeypatch.setenv("OMLX_QWEN4_QSA_NATIVE_MAIN_MIN_ROWS", "0")
    qsa_fast._native_main_min_rows.cache_clear()
    try:
        # Other geometry or dtype: not handled.
        assert qsa_fast._nax_sparse_gqa_attention(
            q.astype(mx.float16), k.astype(mx.float16), k.astype(mx.float16), sel, q_offset=4000
        ) is None
        assert qsa_fast._nax_sparse_gqa_attention(
            q[:, :4], k, k, sel, q_offset=4000
        ) is None
        # Disabled by environment.
        monkeypatch.setenv("OMLX_QWEN4_QSA_NAX", "0")
        assert qsa_fast._nax_sparse_gqa_attention(q, k, k, sel, q_offset=4000) is None
        monkeypatch.delenv("OMLX_QWEN4_QSA_NAX")

        # A failing kernel disables the route instead of raising.
        def boom(*args, **kwargs):
            raise RuntimeError("no pipeline")

        monkeypatch.setattr(qsa_nax, "sparse_gqa_attention", boom)
        assert qsa_fast._nax_sparse_gqa_attention(q, k, k, sel, q_offset=4000) is None
        assert qsa_fast._NAX_QSA_MAIN_DISABLED is True
    finally:
        qsa_fast._native_main_min_rows.cache_clear()
