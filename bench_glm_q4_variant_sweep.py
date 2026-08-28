from __future__ import annotations

import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v4.switch_layers import _build_mxfp4_blocks


MODEL = "/Users/jonathanspangler/.lmstudio/models/Jundot/GLM-5.3-Flash-oQ4e"
PREFIX = "language_model.model.layers.3.mlp.switch_mlp"
E_SUB = 32
CONFIGS = {
    1: (16, 32),
    2: (32, 32),
    3: (16, 64),
    4: (32, 64),
    5: (64, 32),
    6: (64, 64),
    7: (128, 32),
    8: (128, 64),
}


def load_projection(shard: int, name: str):
    tensors = mx.load(f"{MODEL}/model-{shard:05d}-of-00034.safetensors")
    base = f"{PREFIX}.{name}"
    out = tuple(
        mx.contiguous(tensors[f"{base}.{suffix}"][:E_SUB])
        for suffix in ("weight", "scales", "biases")
    )
    mx.eval(*out)
    return out


def stock(x, params, indices):
    w, s, b = params
    return mx.gather_qmm(
        x,
        w,
        s,
        b,
        rhs_indices=indices,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
        sorted_indices=True,
    )


def native(x, params, plan, variant):
    w, s, b = params
    meta, count = plan
    return fast.deepseek_affine_gather_qmm_blocks(
        x, w, s, b, meta, count, 64, 4, variant
    )


def measure(fn, repeats=7):
    for _ in range(3):
        mx.eval(fn())
    mx.synchronize()
    out = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        out.append((time.perf_counter() - start) * 1000)
    return statistics.median(out), min(out), out


def main():
    gate = load_projection(2, "gate_proj")
    up = load_projection(3, "up_proj")
    down = load_projection(3, "down_proj")
    rows_per_expert = 228
    routes = E_SUB * rows_per_expert
    indices = mx.repeat(mx.arange(E_SUB, dtype=mx.int32), rows_per_expert)
    x = mx.random.normal((routes, 1, 4096), dtype=mx.bfloat16)
    gate_ref = stock(x, gate, indices)
    up_ref = stock(x, up, indices)
    activated = nn.silu(mx.minimum(gate_ref, 10.0)) * mx.clip(up_ref, -10.0, 10.0)
    down_ref = stock(activated, down, indices)
    mx.eval(x, indices, gate_ref, up_ref, activated, down_ref)

    cases = (("gate", x, gate, gate_ref), ("up", x, up, up_ref), ("down", activated, down, down_ref))
    stock_total = 0.0
    for name, xx, params, _ in cases:
        med, low, raw = measure(lambda xx=xx, p=params: stock(xx, p, indices))
        stock_total += med
        print(f"stock {name:4s} med={med:.3f} min={low:.3f} raw={raw}")
    print(f"stock total={stock_total:.3f}ms")

    for variant, (bm, bn) in CONFIGS.items():
        plan = _build_mxfp4_blocks(indices, E_SUB, bm)
        mx.eval(*plan)
        total = 0.0
        parity = []
        for name, xx, params, ref in cases:
            y = native(xx, params, plan, variant)
            mx.eval(y)
            parity.append(bool(mx.array_equal(ref.view(mx.uint16), y.view(mx.uint16)).item()))
            med, low, raw = measure(
                lambda xx=xx, p=params, pl=plan, v=variant: native(xx, p, pl, v)
            )
            total += med
            print(
                f"v{variant} bm{bm} bn{bn} {name:4s} parity={parity[-1]} "
                f"med={med:.3f} min={low:.3f} raw={raw}"
            )
        print(
            f"v{variant} bm{bm} bn{bn} total={total:.3f}ms "
            f"speedup={stock_total/total:.3f}x parity={parity}"
        )


if __name__ == "__main__":
    main()
