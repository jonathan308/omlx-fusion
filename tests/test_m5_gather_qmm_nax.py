# SPDX-License-Identifier: Apache-2.0
"""Tests for the runtime-compiled NAX sorted gather_qmm (m5_gather_qmm_nax)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import omlx.patches.m5_gather_qmm as patch_mod
import omlx.patches.m5_gather_qmm_nax as nax
from omlx.patches.m5_gather_qmm import apply_m5_gather_qmm_workaround


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
    monkeypatch.delenv("OMLX_M5_GATHER_QMM_NAX", raising=False)
    monkeypatch.delenv("OMLX_M5_GATHER_QMM_NAX_PLAN", raising=False)


def _stock():
    fn = mx.gather_qmm
    if getattr(fn, "_omlx_m5_reroute", False):
        fn = patch_mod._original_gather_qmm
    return fn


def _quantized(E, N, K, mode, bits, gs, dtype, seed=0):
    w = (mx.random.normal((E, N, K), key=mx.random.key(seed)) * 0.05).astype(dtype)
    if mode == "affine":
        wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
        wd = mx.dequantize(wq, scales, biases, group_size=gs, bits=bits)
    else:
        wq, scales = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
        biases = None
        wd = mx.dequantize(wq, scales, group_size=gs, bits=bits, mode=mode)
    return wq, scales, biases, wd


def _rows(counts, K, dtype, seed=1):
    idx = mx.array(np.repeat(np.arange(len(counts)), counts).astype(np.uint32))
    x = (mx.random.normal((int(idx.shape[0]), 1, K), key=mx.random.key(seed)) * 0.5).astype(
        dtype
    )
    return x, idx


def _routed_rows(tokens, top_k, E, K, dtype, skew, seed=2):
    """SwitchGLU-style sorted rows: top-k routing, flattened, sorted."""
    key = mx.random.key(seed)
    if skew:
        p = 1.0 / (mx.arange(E) + 5.0) ** skew
        scores = mx.log(p)[None, :] + mx.random.gumbel(shape=(tokens, E), key=key)
    else:
        scores = mx.random.uniform(shape=(tokens, E), key=key)
    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[:, :top_k].astype(mx.uint32)
    flat = inds.flatten()
    order = mx.argsort(flat)
    x = (mx.random.normal((tokens, K), key=mx.random.key(seed + 1)) * 0.5).astype(dtype)
    return x[order // top_k][:, None, :], flat[order]


def _nax(x, wq, scales, biases, idx, mode, bits, gs, plan=None):
    out = nax.sorted_gather_qmm(
        x, wq, scales, biases, idx, group_size=gs, bits=bits, mode=mode, plan=plan
    )
    assert out is not None
    return out


def _stock_sorted(x, wq, scales, biases, idx, mode, bits, gs):
    return _stock()(
        x,
        wq,
        scales,
        biases,
        rhs_indices=idx,
        transpose=True,
        group_size=gs,
        bits=bits,
        mode=mode,
        sorted_indices=True,
    )


def _fp32_ref(x, wd, idx):
    return x.astype(mx.float32) @ wd[idx].swapaxes(-1, -2).astype(mx.float32)


# ---------------------------------------------------------------------------
# Support gating (no GPU work)
# ---------------------------------------------------------------------------


def test_supports_gating():
    E, N, K, M = 4, 64, 128, 16
    x = mx.zeros((M, 1, K), dtype=mx.bfloat16)
    idx = mx.zeros((M,), dtype=mx.uint32)
    w4 = mx.zeros((E, N, K // 8), dtype=mx.uint32)
    s64 = mx.zeros((E, N, K // 64), dtype=mx.bfloat16)
    s32u8 = mx.zeros((E, N, K // 32), dtype=mx.uint8)

    assert nax.supports(x, w4, s64, s64, idx, 64, 4, "affine")
    assert nax.supports(x.astype(mx.float16), w4, s64.astype(mx.float16),
                        s64.astype(mx.float16), idx, 64, 4, "affine")
    assert nax.supports(x, w4, s32u8, None, idx, 32, 4, "mxfp4")
    w8 = mx.zeros((E, N, K // 4), dtype=mx.uint32)
    assert nax.supports(x, w8, s64, s64, idx, 64, 8, "affine")

    # fp32 activations, scale dtype mismatch, missing biases.
    assert not nax.supports(x.astype(mx.float32), w4, s64, s64, idx, 64, 4, "affine")
    assert not nax.supports(x, w4, s64.astype(mx.float16), s64, idx, 64, 4, "affine")
    assert not nax.supports(x, w4, s64, None, idx, 64, 4, "affine")
    # Unsupported bit widths / modes / group sizes.
    w3 = mx.zeros((E, N, K * 3 // 32), dtype=mx.uint32)
    assert not nax.supports(x, w3, s64, s64, idx, 64, 3, "affine")
    assert not nax.supports(x, w4, s32u8, None, idx, 32, 4, "nvfp4")
    s16 = mx.zeros((E, N, K // 16), dtype=mx.uint8)
    assert not nax.supports(x, w4, s16, None, idx, 16, 4, "mxfp4")
    # Layout: 2-D rows, rows per index != 1, int32 indices, too few rows.
    assert not nax.supports(x.reshape(M, K), w4, s64, s64, idx, 64, 4, "affine")
    x2 = mx.zeros((M // 2, 2, K), dtype=mx.bfloat16)
    assert not nax.supports(x2, w4, s64, s64, idx[: M // 2], 64, 4, "affine")
    assert not nax.supports(x, w4, s64, s64, idx.astype(mx.int32), 64, 4, "affine")
    assert not nax.supports(x[:4], w4, s64, s64, idx[:4], 64, 4, "affine")
    # Shape mismatch between w and K.
    assert not nax.supports(x, w8, s64, s64, idx, 64, 4, "affine")


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_NAX", "0")
    x = mx.zeros((16, 1, 128), dtype=mx.bfloat16)
    w = mx.zeros((4, 64, 16), dtype=mx.uint32)
    s = mx.zeros((4, 64, 2), dtype=mx.bfloat16)
    idx = mx.zeros((16,), dtype=mx.uint32)
    assert not nax.enabled()
    assert nax.sorted_gather_qmm(x, w, s, s, idx, group_size=64, bits=4) is None


# ---------------------------------------------------------------------------
# Exactness on NAX hardware
# ---------------------------------------------------------------------------

# Empty experts, runs spanning several 64-row tiles, partial tiles of all
# sizes (1..63 rows), a single-row run.
_COUNTS = (70, 0, 5, 33, 64, 17, 1, 130, 0, 11)

_ALIGNED = [
    ("affine", 4, 64, mx.bfloat16),
    ("affine", 4, 32, mx.bfloat16),
    ("affine", 4, 128, mx.bfloat16),
    ("affine", 8, 64, mx.bfloat16),
    ("affine", 8, 32, mx.float16),
    ("affine", 4, 64, mx.float16),
    ("mxfp4", 4, 32, mx.bfloat16),
    ("mxfp4", 4, 32, mx.float16),
]


P = nax.Plan
SEG, DB = nax._SCHED_SEG, nax._SCHED_DB

# Every configuration _plan picks, plus the plain layouts of each schedule
# and tile height, a 64-deep seg with 128-row tiles and a small x group.
_PLANS = [
    P(SEG, 64, 64, 0, 0),
    P(DB, 64, 64, 0, 0),
    P(DB, 64, 64, 32, 0),
    P(DB, 96, 64, 32, 0),
    P(SEG, 96, 128, 32, 0),
    P(SEG, 128, 128, 32, 8192),
    P(SEG, 128, 64, 0, 0),
    P(DB, 128, 64, 0, 0),
    P(SEG, 96, 64, 3, 0),
]


def _plan_id(plan):
    return plan.describe().replace(" ", "-")


@pytest.fixture(params=_PLANS, ids=_plan_id)
def forced_plan(request, monkeypatch):
    """Run a test with every configuration pinned through the env override."""
    plan = request.param
    sched = "seg" if plan.sched == SEG else "db"
    monkeypatch.setenv(
        "OMLX_M5_GATHER_QMM_NAX_PLAN",
        f"{sched},{plan.bm},{plan.bk},{plan.gx},{plan.pad}",
    )
    return plan


def test_plan_selection(monkeypatch):
    monkeypatch.delenv("OMLX_M5_GATHER_QMM_NAX_PLAN", raising=False)
    # Qwen3.8 (E=512 top-10; gate_up K=2560, down K=640) at 1k/2k/4k/8k
    assert nax._plan(10240, 512, 2560, 1280) == P(DB, 64, 64, 0, 0)
    assert nax._plan(20480, 512, 2560, 1280) == P(DB, 64, 64, 32, 0)
    assert nax._plan(40960, 512, 2560, 1280) == P(DB, 96, 64, 32, 0)
    assert nax._plan(81920, 512, 2560, 1280) == P(SEG, 128, 128, 32, 8192)
    assert nax._plan(40960, 512, 640, 2560) == P(DB, 64, 64, 0, 0)
    assert nax._plan(81920, 512, 640, 2560) == P(SEG, 96, 128, 32, 0)
    # GLM-5.3 (E=288 top-8) at 2k / 4k / 8k-token chunks
    assert nax._plan(16384, 288, 4096, 2048) == P(DB, 96, 64, 32, 0)
    assert nax._plan(32768, 288, 2048, 4096) == P(SEG, 128, 128, 32, 8192)
    assert nax._plan(65536, 288, 4096, 4096) == P(SEG, 128, 128, 32, 8192)
    # MiMo-V2.6 (E=256 top-8) at 1k / 2k / 8k
    assert nax._plan(8192, 256, 4096, 2048) == P(DB, 64, 64, 0, 0)
    assert nax._plan(16384, 256, 2048, 4096) == P(DB, 96, 64, 32, 0)
    assert nax._plan(65536, 256, 4096, 4096) == P(SEG, 128, 128, 32, 8192)
    # ragged K / N keep 64-row seg tiles in the plain layout
    assert nax._plan(81920, 512, 2560 + 32, 1280) == P(SEG, 64, 64, 0, 0)
    assert nax._plan(81920, 512, 2560, 1280 + 32) == P(SEG, 64, 64, 0, 0)
    # the env override pins a configuration; invalid values are ignored
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_NAX_PLAN", "seg,96,64,4,0")
    assert nax._plan(10240, 512, 2560, 1280) == P(SEG, 96, 64, 4, 0)
    for bad in ("seg,80,64,0,0", "db,64,256,0,0", "xx,64,64,0,0", "seg,64,64", "seg,64,64,0,99999"):
        monkeypatch.setenv("OMLX_M5_GATHER_QMM_NAX_PLAN", bad)
        assert nax._plan(10240, 512, 2560, 1280) == P(DB, 64, 64, 0, 0)


@needs_nax
@pytest.mark.parametrize("mode,bits,gs,dtype", _ALIGNED)
@pytest.mark.parametrize("plan", _PLANS, ids=_plan_id)
@pytest.mark.parametrize("K", [256, 384])
def test_bit_identical_to_stock_sorted_kernel(mode, bits, gs, dtype, plan, K):
    """K % 64 == 0: same dequantization and tensor-op order as mlx's kernel.

    K = 384 leaves a 64-deep tail after the 128-deep K steps.
    """
    E, N = len(_COUNTS), 128
    wq, scales, biases, _ = _quantized(E, N, K, mode, bits, gs, dtype)
    x, idx = _rows(_COUNTS, K, dtype)
    out = _nax(x, wq, scales, biases, idx, mode, bits, gs, plan)
    ref = _stock_sorted(x, wq, scales, biases, idx, mode, bits, gs)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert mx.array_equal(out, ref).item()


@needs_nax
@pytest.mark.parametrize("mode,gs", [("affine", 64), ("mxfp4", 32)])
@pytest.mark.parametrize("skew", [0.0, 1.2])
def test_routed_rows_match_stock(mode, gs, skew):
    """SwitchGLU routing (uniform and skewed, empty experts) at MoE shapes."""
    E, N, K = 64, 192, 1152
    wq, scales, biases, _ = _quantized(E, N, K, mode, 4, gs, mx.bfloat16, seed=5)
    for tokens in (160, 600, 1200):  # 20, 75 and 150 rows per expert
        x, idx = _routed_rows(tokens, 8, E, K, mx.bfloat16, skew)
        ref = _stock_sorted(x, wq, scales, biases, idx, mode, 4, gs)
        for plan in [None] + _PLANS:
            out = _nax(x, wq, scales, biases, idx, mode, 4, gs, plan)
            assert mx.array_equal(out, ref).item(), f"tokens={tokens} plan={plan}"


@needs_nax
@pytest.mark.parametrize(
    "mode,bits,dtype",
    [("affine", 4, mx.bfloat16), ("affine", 8, mx.float16), ("mxfp4", 4, mx.bfloat16)],
)
@pytest.mark.parametrize("K", [32, 96, 544])
def test_ragged_k_matches_fp32_reference(mode, bits, dtype, K, forced_plan):
    """K % 64 == 32 (group 32): the stock kernel's tail is wrong here."""
    E, N = len(_COUNTS), 128
    wq, scales, biases, wd = _quantized(E, N, K, mode, bits, 32, dtype)
    x, idx = _rows(_COUNTS, K, dtype)
    out = _nax(x, wq, scales, biases, idx, mode, bits, 32)
    ref = _fp32_ref(x, wd, idx)
    err = mx.abs(out.astype(mx.float32) - ref)
    # bf16/fp16 output rounding of an fp32-accumulated dot product.
    tol = mx.abs(ref) * (2.0**-7 if dtype == mx.bfloat16 else 2.0**-10) + 1e-3
    assert mx.all(err <= tol).item(), f"max err {err.max().item()}"


@needs_nax
def test_ragged_k_tail_never_reads_past_the_row():
    """The K tail zero-fills the weight tile instead of dequantizing past K.

    The scales/biases are views whose buffer continues with NaN right after
    the last expert's last row: a kernel that dequantizes the tail block
    past K (as mlx's fixed kernel does) turns that into NaN * 0 = NaN.
    """
    E, N, K = 4, 64, 96
    wq, scales, biases, wd = _quantized(E, N, K, "affine", 4, 32, mx.bfloat16)
    nan = mx.full((64,), float("nan"), dtype=mx.bfloat16)
    s_view = mx.concatenate([scales.reshape(-1), nan])[: scales.size].reshape(scales.shape)
    b_view = mx.concatenate([biases.reshape(-1), nan])[: biases.size].reshape(biases.shape)
    x, idx = _rows((0, 0, 0, 40), K, mx.bfloat16)
    out = _nax(x, wq, s_view, b_view, idx, "affine", 4, 32)
    assert not mx.any(mx.isnan(out)).item()
    ref = _fp32_ref(x, wd, idx)
    assert mx.abs(out.astype(mx.float32) - ref).max().item() < 0.05


@needs_nax
@pytest.mark.parametrize("mode,gs", [("affine", 64), ("mxfp4", 32)])
@pytest.mark.parametrize("K", [256, 384])
def test_ragged_n_matches_stock(mode, gs, K, forced_plan):
    E, N = len(_COUNTS), 100
    wq, scales, biases, _ = _quantized(E, N, K, mode, 4, gs, mx.bfloat16)
    x, idx = _rows(_COUNTS, K, mx.bfloat16)
    out = _nax(x, wq, scales, biases, idx, mode, 4, gs)
    ref = _stock_sorted(x, wq, scales, biases, idx, mode, 4, gs)
    assert mx.array_equal(out, ref).item()


@needs_nax
@pytest.mark.parametrize("mode,gs", [("affine", 64), ("mxfp4", 32)])
def test_more_than_32768_rows(mode, gs):
    """One call past the stock kernel's int16 row-offset limit.

    Every row is independent, so the stock kernel run on <= 32768-row
    slices (where it is correct) is an exact reference.
    """
    E, N, K = 16, 64, 128
    counts = tuple([2304] * 15 + [4000])  # 38560 rows, heavy last expert
    wq, scales, biases, _ = _quantized(E, N, K, mode, 4, gs, mx.bfloat16)
    x, idx = _rows(counts, K, mx.bfloat16)
    rows = int(idx.shape[0])
    assert rows > 32768
    half = rows // 2
    ref = mx.concatenate(
        [
            _stock_sorted(x[:half], wq, scales, biases, idx[:half], mode, 4, gs),
            _stock_sorted(x[half:], wq, scales, biases, idx[half:], mode, 4, gs),
        ]
    )
    for plan in [None] + _PLANS:
        out = _nax(x, wq, scales, biases, idx, mode, 4, gs, plan)
        assert mx.array_equal(out, ref).item(), f"plan={plan}"


@needs_nax
def test_single_expert_and_all_experts_empty_but_one():
    E, N, K = 32, 128, 128
    wq, scales, biases, _ = _quantized(E, N, K, "affine", 4, 64, mx.bfloat16)
    # >= 4 rows per expert overall, so mlx picks its sorted rhs kernel too.
    for counts in ((0,) * 31 + (300,), (129,) + (0,) * 31):
        x, idx = _rows(counts, K, mx.bfloat16)
        ref = _stock_sorted(x, wq, scales, biases, idx, "affine", 4, 64)
        for plan in [None] + _PLANS:
            out = _nax(x, wq, scales, biases, idx, "affine", 4, 64, plan)
            assert mx.array_equal(out, ref).item(), f"plan={plan}"


@needs_nax
@pytest.mark.parametrize("bm", [64, 96, 128])
def test_tile_scan_descriptors(bm):
    """(row_start, expert, rows) tiles, expert-major, <= bm rows each."""
    counts = (70, 0, 5, 33, 64, 17, 1, 130, 0, 11)
    idx = mx.array(np.repeat(np.arange(len(counts)), counts).astype(np.uint32))
    M, E = int(idx.shape[0]), len(counts)
    max_tiles = (M + bm - 1) // bm + E
    tiles, count = nax._get_kernel("scan")(
        inputs=[idx, mx.array([M, E, max_tiles], dtype=mx.int32)],
        template=[("BM", bm), ("MAXE", nax._MAX_EXPERTS)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(max_tiles * 4,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    expect = []
    start = 0
    for e, n in enumerate(counts):
        for r in range(start, start + n, bm):
            expect.append((r, e, min(bm, start + n - r), 0))
        start += n
    n_tiles = int(count[0].item())
    assert n_tiles == len(expect)
    got = np.array(tiles).reshape(-1, 4)[:n_tiles]
    assert [tuple(int(v) for v in t) for t in got] == expect


@needs_nax
@pytest.mark.parametrize("bm", [64, 96])
def test_tile_scan_randomized(bm):
    """Row counts with every M % 4, clustered and sparse expert use."""
    rng = np.random.default_rng(7)
    for trial in range(40):
        E = int(rng.integers(1, 600))
        M = int(rng.integers(8, 3000))
        pool = rng.integers(0, E, max(1, E // 5)) if trial % 2 else np.arange(E)
        idx_np = np.sort(rng.choice(pool, M)).astype(np.uint32)
        max_tiles = (M + bm - 1) // bm + min(E, M)
        tiles, count = nax._get_kernel("scan")(
            inputs=[mx.array(idx_np), mx.array([M, E, max_tiles], dtype=mx.int32)],
            template=[("BM", bm), ("MAXE", nax._MAX_EXPERTS)],
            grid=(1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(max_tiles * 4,), (1,)],
            output_dtypes=[mx.uint32, mx.uint32],
        )
        expect = []
        for e in range(E):
            lo = int(np.searchsorted(idx_np, e, "left"))
            hi = int(np.searchsorted(idx_np, e, "right"))
            expect += [(r, e, min(bm, hi - r), 0) for r in range(lo, hi, bm)]
        n_tiles = int(count[0].item())
        got = np.array(tiles).reshape(-1, 4)[:n_tiles]
        assert [tuple(int(v) for v in t) for t in got] == expect, (E, M)


@needs_nax
def test_tile_scan_bounded_on_unsorted_indices():
    """Unsorted input breaks the contract but must not overrun the buffer."""
    idx = mx.array(np.tile(np.arange(8, dtype=np.uint32), 50))
    M, E = int(idx.shape[0]), 8
    max_tiles = (M + 63) // 64 + E
    _, count = nax._get_kernel("scan")(
        inputs=[idx, mx.array([M, E, max_tiles], dtype=mx.int32)],
        template=[("BM", 64), ("MAXE", nax._MAX_EXPERTS)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(max_tiles * 4,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    assert int(count[0].item()) <= max_tiles


# ---------------------------------------------------------------------------
# Wrapper routing
# ---------------------------------------------------------------------------


@pytest.fixture
def _installed(monkeypatch):
    was_installed = getattr(mx.gather_qmm, "_omlx_m5_reroute", False)
    raw = patch_mod._original_gather_qmm if was_installed else mx.gather_qmm
    mx.gather_qmm = raw
    monkeypatch.delenv("OMLX_M5_GATHER_QMM_FIX", raising=False)
    assert apply_m5_gather_qmm_workaround()
    yield raw
    mx.gather_qmm = raw
    patch_mod._original_gather_qmm = raw
    if was_installed:
        mx.gather_qmm = patch_mod._gather_qmm_rerouted


@needs_nax
def test_wrapper_routes_sorted_calls_to_nax(_installed, monkeypatch):
    calls = []
    real = nax.sorted_gather_qmm

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(nax, "sorted_gather_qmm", spy)
    E, N, K = 8, 128, 96  # ragged K: the stock sorted kernel is wrong here
    wq, scales, biases, wd = _quantized(E, N, K, "affine", 4, 32, mx.bfloat16)
    x, idx = _rows((20, 0, 50, 7, 64, 1, 30, 9), K, mx.bfloat16)
    out = mx.gather_qmm(
        x, wq, scales, biases, rhs_indices=idx, transpose=True, group_size=32, bits=4,
        sorted_indices=True,
    )
    assert calls == [True]
    ref = _fp32_ref(x, wd, idx)
    assert mx.abs(out.astype(mx.float32) - ref).max().item() < 0.05

    # Positional scales/biases and default group size / bits (affine 64/4).
    wq, scales, biases, _ = _quantized(E, N, 128, "affine", 4, 64, mx.bfloat16)
    x, idx = _rows((20, 0, 50, 7, 64, 1, 30, 9), 128, mx.bfloat16)
    out = mx.gather_qmm(x, wq, scales, biases, None, idx, sorted_indices=True)
    ref = _stock_sorted(x, wq, scales, biases, idx, "affine", 4, 64)
    assert calls == [True, True]
    assert mx.array_equal(out, ref).item()

    # Fewer than 4 rows per expert: mlx's qmv path, not the NAX route.
    xs, idxs = _rows((3, 0, 5, 7, 2, 1, 4, 9), 128, mx.bfloat16)
    mx.gather_qmm(xs, wq, scales, biases, rhs_indices=idxs, sorted_indices=True)
    assert calls == [True, True]

    # Unsorted calls and lhs gathers never take the route.
    mx.gather_qmm(x, wq, scales, biases, rhs_indices=idx, group_size=64, bits=4)
    mx.gather_qmm(
        x, wq, scales, biases, lhs_indices=mx.arange(x.shape[0]).astype(mx.uint32),
        rhs_indices=idx, sorted_indices=True,
    )
    assert calls == [True, True]


@needs_nax
def test_wrapper_kill_switch_keeps_stock_path(_installed, monkeypatch):
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_NAX", "0")
    seen = []
    monkeypatch.setattr(
        nax, "_launch", lambda *a, **k: seen.append(1) or None
    )
    E, N, K = 8, 128, 128
    wq, scales, biases, _ = _quantized(E, N, K, "affine", 4, 64, mx.bfloat16)
    x, idx = _rows((20, 0, 50, 7, 64, 1, 30, 9), K, mx.bfloat16)
    out = mx.gather_qmm(
        x, wq, scales, biases, rhs_indices=idx, group_size=64, bits=4, sorted_indices=True
    )
    ref = _stock_sorted(x, wq, scales, biases, idx, "affine", 4, 64)
    assert not seen
    assert mx.array_equal(out, ref).item()


@needs_nax
def test_failed_self_test_falls_back(_installed, monkeypatch):
    monkeypatch.setattr(nax, "_verified", {})
    monkeypatch.setattr(nax, "_self_test", lambda key: False)
    E, N, K = 8, 128, 128
    wq, scales, biases, _ = _quantized(E, N, K, "affine", 4, 64, mx.bfloat16)
    x, idx = _rows((20, 0, 50, 7, 64, 1, 30, 9), K, mx.bfloat16)
    assert (
        nax.sorted_gather_qmm(x, wq, scales, biases, idx, group_size=64, bits=4) is None
    )
    out = mx.gather_qmm(
        x, wq, scales, biases, rhs_indices=idx, group_size=64, bits=4, sorted_indices=True
    )
    ref = _stock_sorted(x, wq, scales, biases, idx, "affine", 4, 64)
    assert mx.array_equal(out, ref).item()


@needs_nax
@pytest.mark.parametrize("plan", _PLANS, ids=_plan_id)
def test_self_test_passes_for_supported_instantiations(plan):
    for key in [
        (mx.bfloat16, "affine", 4, 64, True, True),
        (mx.bfloat16, "affine", 4, 32, True, False),
        (mx.float16, "affine", 8, 64, False, True),
        (mx.bfloat16, "mxfp4", 4, 32, True, True),
        (mx.bfloat16, "mxfp4", 4, 32, False, False),
    ]:
        if plan.sched == DB and not (key[4] and key[5]):
            continue  # db runs aligned shapes only (seg covers the rest)
        dtype, mode, bits, gs, align_n, align_k = key
        full = (dtype, mode, bits, gs, plan, align_n, align_k)
        assert nax._self_test(full) is True, full
