# SPDX-License-Identifier: Apache-2.0
"""Fused GLM-5.3 prefill hyper-connection kernels: exactness and fallbacks."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import mlx_vlm_glm5_next_compat as compat

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal kernels need a GPU"
)


@pytest.fixture(autouse=True)
def _apply_glm5_next_compat():
    compat.apply_mlx_vlm_glm5_next_compat_patch()


def _modules():
    from mlx_vlm.models.deepseek_v4 import hyper_connection as dsv4_hc
    from mlx_vlm.models.glm5_next import hc_prefill, language

    return dsv4_hc, hc_prefill, language


def _connection(hidden=4096, seed=0, cls=None):
    dsv4_hc, _, language = _modules()
    cls = cls or language.HyperConnection
    config = SimpleNamespace(
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
        hidden_size=hidden,
    )
    connection = cls(config)
    key = mx.random.key(seed)
    k1, k2, k3 = mx.random.split(key, 3)
    connection.fn = mx.random.normal((24, 4 * hidden), key=k1) * 0.02
    connection.base = mx.random.normal((24,), key=k2) * 0.5
    connection.scale = mx.random.uniform(0.5, 2.0, (3,), key=k3)
    connection.eval()
    mx.eval(connection.parameters())
    return connection


def _stream(length, hidden=4096, seed=1, batch=1):
    x = mx.random.normal((batch, length, 4, hidden), key=mx.random.key(seed))
    return x.astype(mx.bfloat16)


def _fp32_reference_pre(connection, x):
    """Canonical math with the mixes in full fp32 (CPU matmul)."""
    dsv4_hc, _, _ = _modules()
    y = x.astype(mx.float32)
    z = mx.fast.rms_norm(y.flatten(-2), None, connection.norm_eps)
    mixes = mx.matmul(z, connection.fn.T, stream=mx.cpu)
    return dsv4_hc._hc_kernel(
        x,
        y,
        mixes,
        connection.scale,
        connection.base,
        connection.hc_mult,
        connection.sinkhorn_iters,
        connection.hc_eps,
    )


def _bf16_ulps(a, b):
    a = np.array(a.astype(mx.float32)).view(np.int32) >> 16
    b = np.array(b.astype(mx.float32)).view(np.int32) >> 16
    a = np.where(a < 0, -(a & 0x7FFF), a).astype(np.int64)
    b = np.where(b < 0, -(b & 0x7FFF), b).astype(np.int64)
    return np.abs(a - b)


def test_fused_pre_matches_fp32_reference_up_to_summation_order():
    _, hc_prefill, _ = _modules()
    connection = _connection()
    x = _stream(96)
    fused = hc_prefill.hc_pre(connection, x)
    assert fused is not None
    reference = _fp32_reference_pre(connection, x)
    mx.eval(fused, reference)
    for got, want in zip(fused[1:], reference[1:]):
        assert got.shape == want.shape and got.dtype == want.dtype
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-5, atol=1e-5)
    assert fused[0].shape == reference[0].shape
    assert fused[0].dtype == mx.bfloat16
    # Only rounding flips (and near-cancellation) from the fp32 mix order.
    ulps = _bf16_ulps(fused[0], reference[0])
    assert (ulps > 0).mean() < 0.01
    np.testing.assert_allclose(
        np.array(fused[0].astype(mx.float32)),
        np.array(reference[0].astype(mx.float32)),
        rtol=2**-7,
        atol=1e-3,
    )


@pytest.mark.parametrize("length", [2048, 777])
def test_fused_pre_is_batch_invariant(length):
    _, hc_prefill, _ = _modules()
    connection = _connection()
    x = _stream(length)
    full = hc_prefill.hc_pre(connection, x)
    pieces = [hc_prefill.hc_pre(connection, x[:, s : s + 256]) for s in range(0, length, 256)]
    tiled = [mx.concatenate([p[i] for p in pieces], axis=1) for i in range(3)]
    shifted = hc_prefill.hc_pre(connection, x[:, 3 : length - 5])
    mx.eval(full, tiled, shifted)
    for a, b, c in zip(full, tiled, shifted):
        assert mx.array_equal(a, b)
        assert mx.array_equal(a[:, 3 : length - 5], c)


def test_fused_pre_batch_rows_match_single_requests():
    _, hc_prefill, _ = _modules()
    connection = _connection()
    x = _stream(40, batch=3)
    batched = hc_prefill.hc_pre(connection, x)
    singles = [hc_prefill.hc_pre(connection, x[i : i + 1]) for i in range(3)]
    mx.eval(batched, singles)
    for i, single in enumerate(singles):
        for a, b in zip(batched, single):
            assert mx.array_equal(a[i : i + 1], b)


@pytest.mark.parametrize("rows, threads", [(8, 1024), (32, 1024), (16, 512), (8, 256)])
def test_fused_pre_does_not_depend_on_tile_shape(monkeypatch, rows, threads):
    """The reduction order is fixed, so any tile height / thread count gives
    the same bits as the default configuration."""
    _, hc_prefill, _ = _modules()
    connection = _connection()
    x = _stream(203)
    default = hc_prefill.hc_pre(connection, x)
    mx.eval(default)
    monkeypatch.setattr(hc_prefill, "_ROWS", rows)
    monkeypatch.setattr(hc_prefill, "_THREADS", threads)
    other = hc_prefill.hc_pre(connection, x)
    if other is None:
        # The kernel fails closed where the device cannot launch this
        # threadgroup (e.g. virtual GPUs capped below 1024 threads).
        pytest.skip(f"this GPU cannot run {rows} rows x {threads} threads per threadgroup")
    mx.eval(other)
    for a, b in zip(default, other):
        assert mx.array_equal(a, b)


@pytest.mark.parametrize("length", [64, 37])
def test_fused_expand_matches_exact_short_block_kernel(length):
    from mlx_vlm.models.fast_ops import exact_hc_expand

    _, hc_prefill, _ = _modules()
    connection = _connection()
    x = _stream(length)
    branch = (mx.random.normal((1, length, 4096), key=mx.random.key(7)) * 0.5).astype(
        mx.bfloat16
    )
    _, post, comb = hc_prefill.hc_pre(connection, x)
    fused = hc_prefill.hc_expand(branch, x, post, comb)
    exact = exact_hc_expand(branch, x, post, comb)
    assert fused is not None and exact is not None
    mx.eval(fused, exact)
    assert fused.shape == x.shape and fused.dtype == x.dtype
    assert mx.array_equal(fused, exact)


def test_fused_expand_is_batch_invariant():
    _, hc_prefill, _ = _modules()
    connection = _connection()
    x = _stream(600)
    branch = (mx.random.normal((1, 600, 4096), key=mx.random.key(9))).astype(mx.bfloat16)
    _, post, comb = hc_prefill.hc_pre(connection, x)
    full = hc_prefill.hc_expand(branch, x, post, comb)
    pieces = [
        hc_prefill.hc_expand(
            branch[:, s : s + 256], x[:, s : s + 256], post[:, s : s + 256], comb[:, s : s + 256]
        )
        for s in range(0, 600, 256)
    ]
    assert mx.array_equal(full, mx.concatenate(pieces, axis=1))


def test_bitwise_equal_to_canonical_path_without_tf32():
    """With MLX_ENABLE_TF32=0 the canonical prefill path runs its matmuls in
    fp32; the fused kernels then reproduce it bit for bit on this data."""
    script = textwrap.dedent(
        """
        import mlx.core as mx
        from omlx.patches import mlx_vlm_glm5_next_compat as compat
        compat.apply_mlx_vlm_glm5_next_compat_patch()
        from mlx_vlm.models.deepseek_v4 import hyper_connection as dsv4_hc
        from mlx_vlm.models.glm5_next import hc_prefill
        from tests.test_glm5_next_hc_prefill import _connection, _stream
        c = _connection(cls=dsv4_hc.HyperConnection)
        x = _stream(300)
        branch = (mx.random.normal((1, 300, 4096), key=mx.random.key(3))).astype(mx.bfloat16)
        ref = c(x)
        fused = hc_prefill.hc_pre(c, x)
        ref_e = dsv4_hc.hc_expand(branch, x, ref[1], ref[2])
        fused_e = hc_prefill.hc_expand(branch, x, fused[1], fused[2])
        mx.eval(ref, fused, ref_e, fused_e)
        ok = all(bool(mx.array_equal(a, b)) for a, b in zip(ref, fused))
        ok = ok and bool(mx.array_equal(ref_e, fused_e))
        print("BITWISE", ok)
        """
    )
    env = dict(os.environ, MLX_ENABLE_TF32="0")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "BITWISE True" in result.stdout, result.stdout + result.stderr[-2000:]


def test_unsupported_inputs_fall_back():
    _, hc_prefill, _ = _modules()
    connection = _connection(hidden=512)
    x = _stream(32, hidden=512)
    assert hc_prefill.hc_pre(connection, x) is None  # 4 * 512 is not a 4096 multiple
    connection = _connection()
    assert hc_prefill.hc_pre(connection, _stream(32).astype(mx.float32)) is None
    connection.train()
    assert hc_prefill.hc_pre(connection, _stream(32)) is None
    connection.eval()
    connection.fn = connection.fn.astype(mx.bfloat16)
    assert hc_prefill.hc_pre(connection, _stream(32)) is None
    branch = mx.zeros((1, 32, 4096), dtype=mx.bfloat16)
    post = mx.zeros((1, 32, 4), dtype=mx.float32)
    comb = mx.zeros((1, 32, 4, 4), dtype=mx.float32)
    assert hc_prefill.hc_expand(branch, _stream(32), post.astype(mx.bfloat16), comb) is None
    assert hc_prefill.hc_expand(branch[..., :4088], _stream(32)[..., :4088], post, comb) is None


def test_layer_routes_only_prefill_blocks_to_fused_kernels(monkeypatch):
    _, hc_prefill, language = _modules()
    connection = _connection()
    calls = {"pre": 0, "expand": 0}
    real_pre, real_expand = hc_prefill.hc_pre, hc_prefill.hc_expand

    def count_pre(*args):
        calls["pre"] += 1
        return real_pre(*args)

    def count_expand(*args):
        calls["expand"] += 1
        return real_expand(*args)

    monkeypatch.setattr(hc_prefill, "hc_pre", count_pre)
    monkeypatch.setattr(hc_prefill, "hc_expand", count_expand)
    for length, expected in ((1, 0), (8, 0), (9, 1)):
        x = _stream(length)
        before = dict(calls)
        collapsed, post, comb = connection(x)
        out = language.hc_expand(collapsed, x, post, comb)
        mx.eval(out)
        assert out.shape == x.shape
        assert calls["pre"] - before["pre"] == expected
        assert calls["expand"] - before["expand"] == expected


def test_disabled_env_keeps_canonical_path(monkeypatch):
    _, hc_prefill, _ = _modules()
    monkeypatch.setattr(hc_prefill, "_DISABLED", True)
    connection = _connection()
    assert hc_prefill.hc_pre(connection, _stream(32)) is None
    x = _stream(32)
    post = mx.zeros((1, 32, 4), dtype=mx.float32)
    comb = mx.zeros((1, 32, 4, 4), dtype=mx.float32)
    assert hc_prefill.hc_expand(x[:, :, 0], x, post, comb) is None
