from __future__ import annotations

import gc
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v4.switch_layers import _build_mxfp4_blocks


MODEL = "/Users/jonathanspangler/.lmstudio/models/Jundot/GLM-5.3-Flash-oQ4e"
PREFIX = "language_model.model.layers.3.mlp.switch_mlp"
E_SUB = 32
H = 4096
I = 2048


def load_projection(shard: int, name: str):
    tensors = mx.load(f"{MODEL}/model-{shard:05d}-of-00034.safetensors")
    base = f"{PREFIX}.{name}"
    out = tuple(
        mx.contiguous(tensors[f"{base}.{suffix}"][:E_SUB])
        for suffix in ("weight", "scales", "biases")
    )
    mx.eval(*out)
    del tensors
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


def native_pair(x, p0, p1, plan, variant):
    w0, s0, b0 = p0
    w1, s1, b1 = p1
    meta, count = plan
    return fast.deepseek_affine_gather_qmm_pair_concat_blocks(
        x, w0, s0, b0, w1, s1, b1, meta, count, 64, 4, variant
    )


def bench(fn, warmup=2, repeats=7):
    for _ in range(warmup):
        y = fn()
        mx.eval(y)
    mx.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        y = fn()
        mx.eval(y)
        mx.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), min(times), times


def bits(x):
    return x.view(mx.uint16) if x.dtype == mx.bfloat16 else x.view(mx.uint16)


def main():
    print("device", mx.device_info())
    print("native", fast.is_native_available())
    gate = load_projection(2, "gate_proj")
    up = load_projection(3, "up_proj")
    down = load_projection(3, "down_proj")
    print("weights", gate[0].shape, up[0].shape, down[0].shape)

    # Preserve the real E=288 average routes/expert for M={256,2048,8192},
    # while holding only 32 actual expert banks in this isolated process.
    for tokens in (256, 2048, 8192):
        rows_per_expert = max(1, round(tokens * 8 / 288))
        routes = E_SUB * rows_per_expert
        indices = mx.repeat(mx.arange(E_SUB, dtype=mx.int32), rows_per_expert)
        x = mx.random.normal((routes, 1, H), dtype=mx.bfloat16)
        mx.eval(indices, x)
        print(
            f"\nM={tokens} simulated_routes={routes} rows/expert={rows_per_expert} "
            f"(real routes={tokens*8})"
        )

        plans = {
            1: _build_mxfp4_blocks(indices, E_SUB, 16),
            2: _build_mxfp4_blocks(indices, E_SUB, 32),
        }
        for plan in plans.values():
            mx.eval(*plan)

        # Projection parity and timing. Gate and up share shape; down uses the
        # exact activation produced from the actual gate/up banks.
        gate_ref = stock(x, gate, indices)
        up_ref = stock(x, up, indices)
        activated = nn.silu(mx.minimum(gate_ref, 10.0)) * mx.clip(
            up_ref, -10.0, 10.0
        )
        down_ref = stock(activated, down, indices)
        mx.eval(gate_ref, up_ref, activated, down_ref)

        for variant, plan in plans.items():
            gate_y = native(x, gate, plan, variant)
            up_y = native(x, up, plan, variant)
            pair_y = native_pair(x, gate, up, plan, variant)
            act_y = nn.silu(mx.minimum(gate_y, 10.0)) * mx.clip(
                up_y, -10.0, 10.0
            )
            down_y = native(act_y, down, plan, variant)
            mx.eval(gate_y, up_y, pair_y, down_y)
            print(
                "parity",
                "bm", 16 if variant == 1 else 32,
                "gate", bool(mx.array_equal(bits(gate_ref), bits(gate_y)).item()),
                "up", bool(mx.array_equal(bits(up_ref), bits(up_y)).item()),
                "pair-gate", bool(mx.array_equal(bits(gate_ref), bits(pair_y[..., :I])).item()),
                "pair-up", bool(mx.array_equal(bits(up_ref), bits(pair_y[..., I:])).item()),
                "down", bool(mx.array_equal(bits(down_ref), bits(down_y)).item()),
            )

        repeats = 9 if tokens < 8192 else 5
        cases = [
            ("stock gate", lambda: stock(x, gate, indices)),
            ("stock up", lambda: stock(x, up, indices)),
            ("stock down", lambda: stock(activated, down, indices)),
        ]
        for variant, plan in plans.items():
            bm = 16 if variant == 1 else 32
            cases.extend(
                [
                    (f"native bm{bm} gate", lambda p=plan, v=variant: native(x, gate, p, v)),
                    (f"native bm{bm} up", lambda p=plan, v=variant: native(x, up, p, v)),
                    (f"native bm{bm} down", lambda p=plan, v=variant: native(activated, down, p, v)),
                    (f"native bm{bm} pair", lambda p=plan, v=variant: native_pair(x, gate, up, p, v)),
                ]
            )
        timings = {}
        for name, fn in cases:
            med, low, raw = bench(fn, repeats=repeats)
            timings[name] = med
            print(f"{name:24s} median={med:9.3f} min={low:9.3f} ms raw={raw}")
        stock_total = timings["stock gate"] + timings["stock up"] + timings["stock down"]
        for bm in (16, 32):
            native_total = (
                timings[f"native bm{bm} gate"]
                + timings[f"native bm{bm} up"]
                + timings[f"native bm{bm} down"]
            )
            pair_total = timings[f"native bm{bm} pair"] + timings[f"native bm{bm} down"]
            print(
                f"total bm{bm}: stock3={stock_total:.3f} native3={native_total:.3f} "
                f"speedup={stock_total/native_total:.3f}x pair+down={pair_total:.3f} "
                f"pair_speedup={stock_total/pair_total:.3f}x"
            )
        del x, indices, plans, gate_ref, up_ref, activated, down_ref
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
