# SPDX-License-Identifier: Apache-2.0
"""MoE gate/up gather reading token rows through the sorted row map.

(m5_gather_qmm_nax.sorted_gather_qmm_swiglu(row_map=...),
m5_gather_qmm.sort_routes / fused_gate_up_activation(token_rows=...))

The sorted MoE prefill copies every token's row once per selected expert
([T * k, 1, D]) before the sorted gate/up gather GEMM. With the row map the
kernel reads the same rows from the token rows in place, so the copy is never
computed. The values, tiles and tensor ops are unchanged: outputs must be
bit-identical to the materialised rows for every plan and format, in every
model path that uses it (Qwen weighted-sum prefill and regrouped SwitchGLU,
GLM DSA SwitchGLU for MiMo V2, DeepSeek V4 SwitchGLU for GLM-5.3), and every
declined call must fall back to the copy.
"""

from __future__ import annotations

import io

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import omlx.patches.m5_gather_qmm as reroute
import omlx.patches.m5_gather_qmm_nax as nax
from omlx.patches.m5_gather_qmm import apply_m5_gather_qmm_workaround

ROW_MAP_ENV = "OMLX_M5_GATHER_QMM_NAX_ROW_MAP"


def _on_nax() -> bool:
    if not mx.metal.is_available():
        return False
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


needs_nax = pytest.mark.skipif(not _on_nax(), reason="needs an M5 (NAX) GPU")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "OMLX_M5_GATHER_QMM_NAX",
        "OMLX_M5_GATHER_QMM_NAX_PLAN",
        "OMLX_M5_GATHER_QMM_NAX_SWIGLU",
        ROW_MAP_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def installed(monkeypatch):
    """The m5 reroute wrapper on mx.gather_qmm (restored afterwards)."""
    was_installed = getattr(mx.gather_qmm, "_omlx_m5_reroute", False)
    raw = reroute._original_gather_qmm if was_installed else mx.gather_qmm
    mx.gather_qmm = raw
    monkeypatch.delenv("OMLX_M5_GATHER_QMM_FIX", raising=False)
    assert apply_m5_gather_qmm_workaround()
    yield raw
    mx.gather_qmm = raw
    reroute._original_gather_qmm = raw
    if was_installed:
        mx.gather_qmm = reroute._gather_qmm_rerouted


@pytest.fixture
def launches(monkeypatch):
    """Records, per epilogue launch, whether it read rows through a map."""
    calls = []
    real = nax._launch

    def spy(*args, **kwargs):
        if kwargs.get("epi"):
            calls.append(kwargs.get("row_map") is not None)
        return real(*args, **kwargs)

    monkeypatch.setattr(nax, "_launch", spy)
    return calls


def _assert_bitwise(got, ref, msg=""):
    assert got.shape == ref.shape and got.dtype == ref.dtype, msg
    view = {2: mx.uint16, 4: mx.uint32}[got.dtype.size]
    assert mx.array_equal(got.view(view), ref.view(view)).item(), msg


def _quantized(E, N, K, mode, bits, gs, dtype, seed=0):
    w = (mx.random.normal((E, N, K), key=mx.random.key(seed)) * 0.05).astype(dtype)
    if mode == "affine":
        wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
    else:
        wq, scales = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
        biases = None
    return wq, scales, biases


# Empty experts, runs spanning several tiles, partial tiles of all sizes,
# a single-row run.
_COUNTS = (70, 0, 5, 33, 64, 17, 1, 130, 0, 11)


def _sorted_indices(counts=_COUNTS):
    return mx.array(np.repeat(np.arange(len(counts)), counts).astype(np.uint32))


def _token_rows(T, K, dtype, seed=3):
    """Token rows whose scale spans 0.1x-16x (sigmoid tails, clip bounds)."""
    scale = mx.power(10.0, mx.linspace(-1.0, 1.2, T)).reshape(T, 1, 1)
    return (mx.random.normal((T, 1, K), key=mx.random.key(seed)) * scale).astype(dtype)


def _row_map(M, T):
    """A scrambled, repeating sorted row -> token row map touching every
    token row, the last one included."""
    return ((mx.arange(M, dtype=mx.uint32) * 7 + 3) % T).astype(mx.uint32)


_FORMATS = [
    ("affine", 4, 64, mx.bfloat16),  # GLM-5.3, Qwen3.8
    ("mxfp4", 4, 32, mx.bfloat16),  # MiMo-V2.6
    ("affine", 4, 32, mx.bfloat16),
    ("affine", 8, 64, mx.float16),
]
_LIMITS = [None, 10.0]

P = nax.Plan
SEG, DB = nax._SCHED_SEG, nax._SCHED_DB
_PLANS = [
    P(SEG, 64, 64, 0, 0),
    P(DB, 64, 64, 0, 0),
    P(DB, 64, 64, 32, 0),
    P(DB, 96, 64, 32, 0),
    P(SEG, 96, 128, 32, 0),
    P(SEG, 128, 128, 32, 8192),
    P(DB, 128, 64, 0, 0),
]


def _plan_id(plan):
    return plan.describe().replace(" ", "-")


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@needs_nax
@pytest.mark.parametrize("mode,bits,gs,dtype", _FORMATS)
@pytest.mark.parametrize("plan", _PLANS, ids=_plan_id)
def test_row_map_bit_identical_to_materialised_rows(mode, bits, gs, dtype, plan):
    """Every configuration, K with and without a 64-deep tail after the
    128-deep steps: rows read through the map == the copied rows, and == the
    unfused path (plain kernel + split + activation) on the copy."""
    idx = _sorted_indices()
    M = int(idx.shape[0])
    T = M // 3 + 1
    for K in (256, 384):
        wq, scales, biases = _quantized(len(_COUNTS), 256, K, mode, bits, gs, dtype)
        x_tok = _token_rows(T, K, dtype)
        row_map = _row_map(M, T)
        x = x_tok[row_map]
        kw = dict(group_size=gs, bits=bits, mode=mode, plan=plan)
        for limit in _LIMITS:
            got = nax.sorted_gather_qmm_swiglu(
                x_tok, wq, scales, biases, idx, limit=limit, row_map=row_map, **kw
            )
            assert got is not None and got.shape == (M, 1, 128)
            ref = nax.sorted_gather_qmm_swiglu(
                x, wq, scales, biases, idx, limit=limit, **kw
            )
            _assert_bitwise(got, ref, f"K={K} limit={limit}")
            gate_up = nax.sorted_gather_qmm(x, wq, scales, biases, idx, **kw)
            x_gate, x_up = mx.split(gate_up, 2, axis=-1)
            _assert_bitwise(got, nax.reference_activation(x_up, x_gate, limit))


@needs_nax
@pytest.mark.parametrize("mode,gs,limit", [("affine", 64, 10.0), ("mxfp4", 32, None)])
@pytest.mark.parametrize("tokens,top_k,E", [(300, 8, 16), (1100, 10, 64)])
def test_row_map_of_routed_tokens(mode, gs, limit, tokens, top_k, E):
    """SwitchGLU routing through sort_routes, default plans."""
    K, n = 256, 64
    wq, scales, biases = _quantized(E, 2 * n, K, mode, 4, gs, mx.bfloat16)
    scores = mx.random.uniform(shape=(1, tokens, E), key=mx.random.key(tokens))
    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    h = (mx.random.normal((1, tokens, K), key=mx.random.key(1)) * 4.0).astype(
        mx.bfloat16
    )
    x_tok, row_map, idx, _ = reroute.sort_routes(mx.expand_dims(h, (-2, -3)), inds)
    kw = dict(group_size=gs, bits=4, mode=mode, limit=limit)
    got = nax.sorted_gather_qmm_swiglu(
        x_tok, wq, scales, biases, idx, row_map=row_map, **kw
    )
    ref = nax.sorted_gather_qmm_swiglu(x_tok[row_map], wq, scales, biases, idx, **kw)
    assert got is not None and ref is not None
    _assert_bitwise(got, ref)


def test_sort_routes_matches_mlx_lm_gather_sort():
    from mlx_lm.models.switch_layers import _gather_sort

    h = mx.random.normal((2, 50, 1, 1, 32)).astype(mx.bfloat16)
    inds = mx.argpartition(
        -mx.random.uniform(shape=(2, 50, 16)), kth=5, axis=-1
    )[..., :6]
    x_tok, row_map, idx, inv = reroute.sort_routes(h, inds)
    x_ref, idx_ref, inv_ref = _gather_sort(h, inds)
    assert x_tok.shape == (100, 1, 32) and row_map.dtype == mx.uint32
    _assert_bitwise(x_tok[row_map], x_ref)
    assert mx.array_equal(idx, idx_ref).item()
    assert mx.array_equal(inv, inv_ref).item()


# ---------------------------------------------------------------------------
# Support gating, switches and fallbacks
# ---------------------------------------------------------------------------


def test_row_map_unsupported_calls_return_none(monkeypatch):
    E, K, n, T = 4, 128, 64, 40
    idx = mx.zeros((64,), dtype=mx.uint32)
    x_tok = mx.zeros((T, 1, K), dtype=mx.bfloat16)
    w = mx.zeros((E, 2 * n, K // 8), dtype=mx.uint32)
    s = mx.zeros((E, 2 * n, K // 64), dtype=mx.bfloat16)
    rmap = _row_map(64, T)

    def call(row_map, x=x_tok):
        return nax.sorted_gather_qmm_swiglu(
            x, w, s, s, idx, group_size=64, bits=4, verify=False, row_map=row_map
        )

    assert call(rmap.astype(mx.int32)) is None  # map must be uint32
    assert call(rmap[:-1]) is None  # one entry per sorted row
    assert call(rmap.reshape(8, 8)) is None
    assert call(rmap, x=x_tok.reshape(T, K)) is None  # token rows [T, 1, K]
    assert not nax.supports(x_tok, w, s, s, idx, 64, 4, "affine")  # no map
    assert nax.supports(x_tok, w, s, s, idx, 64, 4, "affine", rmap)
    # switches: the row map alone, or the epilogue it rides on
    assert nax.row_map_enabled()
    monkeypatch.setenv(ROW_MAP_ENV, "0")
    assert not nax.row_map_enabled()
    assert call(rmap) is None
    monkeypatch.delenv(ROW_MAP_ENV)
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_NAX_SWIGLU", "0")
    assert not nax.row_map_enabled()
    assert call(rmap) is None


@needs_nax
def test_failed_row_map_self_test_falls_back(monkeypatch, installed, launches):
    from omlx.patches.glm_moe_dsa.switch_layers import SwiGLU

    proj, h, inds = _dsa_problem()
    x_tok, row_map, idx, _ = reroute.sort_routes(mx.expand_dims(h, (-2, -3)), inds)
    x = x_tok[row_map]
    ref = reroute.fused_gate_up_activation(proj, x, idx, SwiGLU())
    monkeypatch.setattr(nax, "_verified", {})
    monkeypatch.setattr(nax, "_self_test_act_map", lambda key: False)
    del launches[:]
    got = reroute.fused_gate_up_activation(
        proj, x, idx, SwiGLU(), token_rows=(x_tok, row_map)
    )
    # declined map -> the copy (earlier launches: the unmapped canaries,
    # re-run after the verdict cache reset)
    assert launches[-1] is False and True not in launches
    _assert_bitwise(got, ref)


def _dsa_problem(tokens=96, E=16, n=64, K=128, top_k=8):
    from omlx.patches.glm_moe_dsa import switch_layers as dsa

    lin = dsa.SwitchLinear(K, 2 * n, E, bias=False)
    lin.weight = (
        mx.random.normal(lin.weight.shape, key=mx.random.key(9)) * 0.05
    ).astype(mx.bfloat16)
    proj = lin.to_quantized(group_size=64, bits=4)
    scores = mx.random.uniform(shape=(1, tokens, E), key=mx.random.key(2))
    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    h = (mx.random.normal((1, tokens, K), key=mx.random.key(3)) * 4.0).astype(
        mx.bfloat16
    )
    mx.eval(proj.parameters(), inds, h)
    return proj, h, inds


@needs_nax
def test_fused_gate_up_activation_reads_token_rows(installed, launches, monkeypatch):
    from omlx.patches.glm_moe_dsa.switch_layers import SwiGLU

    proj, h, inds = _dsa_problem()
    x_tok, row_map, idx, _ = reroute.sort_routes(mx.expand_dims(h, (-2, -3)), inds)
    x = x_tok[row_map]
    act = SwiGLU()
    got = reroute.fused_gate_up_activation(
        proj, x, idx, act, token_rows=(x_tok, row_map)
    )
    ref = reroute.fused_gate_up_activation(proj, x, idx, act)
    assert launches == [True, False]
    _assert_bitwise(got, ref)
    x_gate, x_up = mx.split(proj(x, idx, sorted_indices=True), 2, axis=-1)
    _assert_bitwise(got, act(x_up, x_gate))
    # Switched off, or a map the kernel does not take: the copy, same result.
    monkeypatch.setenv(ROW_MAP_ENV, "0")
    off = reroute.fused_gate_up_activation(
        proj, x, idx, act, token_rows=(x_tok, row_map)
    )
    monkeypatch.delenv(ROW_MAP_ENV)
    bad = reroute.fused_gate_up_activation(
        proj, x, idx, act, token_rows=(x_tok, row_map.astype(mx.int32))
    )
    assert launches == [True, False, False, False]
    _assert_bitwise(off, ref)
    _assert_bitwise(bad, ref)


# ---------------------------------------------------------------------------
# Model paths: row map on vs off (bitwise), and the copy leaves the graph
# ---------------------------------------------------------------------------

E, D, INTER, TOP_K = 16, 128, 64, 8


def _gathers(out) -> int:
    f = io.StringIO()
    mx.export_to_dot(f, out)
    return f.getvalue().count('label ="Gather"')


def _check_on_off(run, launches, monkeypatch, n_launch=1):
    """``run()`` with the row map (default) and without: bitwise equal, one
    Gather fewer in the output graph, and only mapped epilogue launches."""
    del launches[:]
    on = run()
    gathers_on = _gathers(on)
    mx.eval(on)
    assert launches == [True] * n_launch
    monkeypatch.setenv(ROW_MAP_ENV, "0")
    off = run()
    gathers_off = _gathers(off)
    mx.eval(off)
    monkeypatch.delenv(ROW_MAP_ENV)
    assert launches == [True] * n_launch + [False] * n_launch
    assert gathers_on == gathers_off - n_launch
    _assert_bitwise(on, off)


def _fused_glu(module, quant, family, activation=None, seed=0):
    from omlx.patches import moe_gate_up_fusion as fusion

    mx.random.seed(seed)
    kwargs = {} if activation is None else {"activation": activation}
    glu = module.SwitchGLU(D, INTER, E, **kwargs)
    for name in ("gate_proj", "up_proj", "down_proj"):
        glu[name].weight = glu[name].weight.astype(mx.bfloat16)
    group_size, bits, mode = quant
    nn.quantize(glu, group_size=group_size, bits=bits, mode=mode)
    mx.eval(glu.parameters())
    glu.eval()

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [glu]

    Holder.__module__ = f"mlx_lm.models.{family}"
    assert fusion.apply_switch_glu_gate_up_fusion(Holder()) == 1
    return glu


def _inputs(tokens):
    x = (mx.random.normal((1, tokens, D), key=mx.random.key(tokens)) * 2.0).astype(
        mx.bfloat16
    )
    s = mx.random.uniform(shape=(1, tokens, E), key=mx.random.key(1))
    inds = mx.argpartition(-s, kth=TOP_K - 1, axis=-1)[..., :TOP_K]
    w = mx.random.uniform(shape=inds.shape, key=mx.random.key(5))
    scores = (w / w.sum(axis=-1, keepdims=True)).astype(mx.float32)
    mx.eval(x, inds, scores)
    return x, inds, scores


@needs_nax
@pytest.mark.parametrize("quant", [(32, 4, "mxfp4"), (64, 4, "affine")])
def test_glm_dsa_switch_glu_row_map(installed, launches, monkeypatch, quant):
    """MiMo V2's experts (GLM DSA SwitchGLU), both inverse-order variants."""
    from omlx.patches.glm_moe_dsa import switch_layers as dsa

    glu = _fused_glu(dsa, quant, "mimo_v2")
    for tokens in (96, 300):
        x, inds, scores = _inputs(tokens)
        for inverse_scatter in (False, True):
            glu.inverse_scatter = inverse_scatter
            for weighted in (False, True):
                _check_on_off(
                    lambda: glu(x, inds, scores=scores, weighted_sum=weighted),
                    launches,
                    monkeypatch,
                )


@needs_nax
def test_deepseek_v4_switch_glu_row_map(installed, launches, monkeypatch):
    """GLM-5.3's experts (DeepSeek V4 SwitchGLU, clamped SwiGLU)."""
    from mlx_vlm.models.glm5_next import language as _  # noqa: F401

    from omlx.patches import mlx_vlm_glm5_next_compat as compat
    from omlx.patches.deepseek_v4 import switch_layers as v4

    compat.apply_mlx_vlm_glm5_next_compat_patch()
    import importlib

    lang = importlib.import_module("mlx_vlm.models.glm5_next.language")
    glu = _fused_glu(
        v4, (64, 4, "affine"), "glm5_next", activation=lang.Glm5NextClampedSwiGLU(10.0)
    )
    for tokens in (160, 300):
        x, inds, scores = _inputs(tokens)
        for weighted in (False, True):
            _check_on_off(
                lambda: glu(x, inds, scores=scores, weighted_sum=weighted),
                launches,
                monkeypatch,
            )


@pytest.fixture
def qwen_glu(monkeypatch):
    """An mlx-vlm SwitchGLU regrouped by qwen35_moe_gate_up (call patch
    restored afterwards)."""
    from mlx_lm.models.switch_layers import SwitchGLU
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    import omlx.patches.qwen35_moe_gate_up as gate_up

    monkeypatch.delenv("OMLX_QWEN35_MOE_GATE_UP", raising=False)
    verifier = Qwen3_5BatchInvariantForward
    monkeypatch.setattr(verifier, "_switch_glu", verifier._switch_glu)
    saved = {cls: cls.__dict__.get("__call__") for cls in (SwitchGLU, VLMSwitchGLU)}
    flags = ("_omlx_gate_up_fused_call", "_omlx_gate_up_original_call")

    mx.random.seed(7)
    glu = VLMSwitchGLU(D, INTER, E)
    for name in ("gate_proj", "up_proj", "down_proj"):
        glu[name].weight = glu[name].weight.astype(mx.bfloat16)
    nn.quantize(glu, group_size=64, bits=4)
    mx.eval(glu.parameters())
    glu.eval()

    class _FakeQwen4Model:
        pass

    _FakeQwen4Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("blocks.0", glu)]
    assert gate_up.apply_qwen35_moe_gate_up_fusion(model) == 1
    yield glu
    for cls, call in saved.items():
        original = cls.__dict__.get("_omlx_gate_up_original_call", call)
        cls.__call__ = original if original is not None else call
        for attr in flags:
            if attr in cls.__dict__:
                delattr(cls, attr)
    gate_up._CALL_PATCHED = False


@needs_nax
def test_qwen_weighted_sum_prefill_row_map(installed, launches, monkeypatch, qwen_glu):
    """Qwen3.8's prefill path (>= 1024 tokens: native weighted sum)."""
    from omlx.custom_kernels.qwen35_prefill import fast

    import omlx.patches.qwen35_moe_weighted_sum as ws

    if not fast.has_symbol("qwen35_moe_weighted_sum"):
        pytest.skip("qwen35_moe_weighted_sum native kernel unavailable")
    for tokens in (96, 300):
        x, inds, scores = _inputs(tokens)
        _check_on_off(
            lambda: ws._native_switch_weighted_sum(
                qwen_glu, x, inds, scores, fast.qwen35_moe_weighted_sum
            ),
            launches,
            monkeypatch,
        )


@needs_nax
def test_qwen_regrouped_call_row_map(installed, launches, monkeypatch, qwen_glu):
    """Qwen's sorted SwitchGLU call below the weighted-sum threshold."""
    for tokens in (96, 300):
        x, inds, _ = _inputs(tokens)
        _check_on_off(lambda: qwen_glu(x, inds), launches, monkeypatch)
