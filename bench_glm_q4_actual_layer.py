from __future__ import annotations

import gc
import math
import statistics
import time

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v4.switch_layers import (
    _build_mxfp4_blocks,
    _gather_sort,
)
from omlx.patches.glm_moe_dsa.deepseek_v32 import group_expert_select


MODEL = "/Users/jonathanspangler/.lmstudio/models/Jundot/GLM-5.3-Flash-oQ4e"
LP = "language_model.model.layers.3.mlp"
SP = f"{LP}.switch_mlp"
E = 288
H = 4096
I = 2048
TOPK = 8
VARIANT = 6
BM = 64


def load_shard(number: int):
    return mx.load(f"{MODEL}/model-{number:05d}-of-00034.safetensors")


def route_inputs(tokens: int, router):
    x = mx.random.normal((1, tokens, H), dtype=mx.bfloat16)
    logits = x.astype(mx.float32) @ router[0].astype(mx.float32).T
    indices, scores = group_expert_select(
        logits, router[1], TOPK, 1, 1, 2.5, True
    )
    x_sorted, idx, inv = _gather_sort(mx.expand_dims(x, (-2, -3)), indices)
    plan = _build_mxfp4_blocks(idx, E, BM)
    mx.eval(x_sorted, idx, inv, scores, *plan)
    counts = mx.sum(
        idx[:, None] == mx.arange(E, dtype=idx.dtype)[None, :],
        axis=0,
    )
    mx.eval(counts)
    counts_list = counts.tolist()
    print(
        f"M={tokens} routes={idx.size} count min/mean/max="
        f"{min(counts_list)}/{statistics.mean(counts_list):.2f}/{max(counts_list)} "
        f"sd={statistics.pstdev(counts_list):.2f} x={x_sorted.shape}"
    )
    return x_sorted, idx, plan


def projection(shard: int, name: str):
    tensors = load_shard(shard)
    prefix = f"{SP}.{name}"
    params = tuple(
        mx.contiguous(tensors[f"{prefix}.{suffix}"])
        for suffix in ("weight", "scales", "biases")
    )
    mx.eval(*params)
    del tensors
    return params


def stock(x, params, idx):
    w, s, b = params
    return mx.gather_qmm(
        x,
        w,
        s,
        b,
        rhs_indices=idx,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
        sorted_indices=True,
    )


def native(x, params, plan):
    w, s, b = params
    meta, count = plan
    return fast.deepseek_affine_gather_qmm_blocks(
        x, w, s, b, meta, count, 64, 4, VARIANT
    )


def timed(fn):
    start = time.perf_counter()
    mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - start) * 1000


def interleaved(stock_fn, native_fn, repeats: int):
    for _ in range(3):
        mx.eval(stock_fn(), native_fn())
    mx.synchronize()
    stock_times = []
    native_times = []
    for rep in range(repeats):
        if rep % 2:
            native_times.append(timed(native_fn))
            stock_times.append(timed(stock_fn))
        else:
            stock_times.append(timed(stock_fn))
            native_times.append(timed(native_fn))
    return stock_times, native_times


def summarize(name: str, tokens: int, stock_times, native_times):
    sm = statistics.median(stock_times)
    nm = statistics.median(native_times)
    ratios = [s / n for s, n in zip(stock_times, native_times)]
    print(
        f"{name} M={tokens}: stock median={sm:.3f} min={min(stock_times):.3f} "
        f"native median={nm:.3f} min={min(native_times):.3f} "
        f"ratio={sm/nm:.3f}x paired_median={statistics.median(ratios):.3f}x\n"
        f"  stock={stock_times}\n  native={native_times}"
    )
    return sm, nm


def main():
    shard2 = load_shard(2)
    router = (
        mx.contiguous(shard2[f"{LP}.gate.weight"]),
        mx.contiguous(shard2[f"{LP}.gate.e_score_correction_bias"]),
    )
    mx.eval(*router)
    del shard2

    routes = {tokens: route_inputs(tokens, router) for tokens in (256, 2048, 8192)}
    del router
    gc.collect()
    mx.clear_cache()

    totals = {tokens: [0.0, 0.0] for tokens in routes}
    for shard, name in ((2, "gate_proj"), (3, "up_proj"), (3, "down_proj")):
        params = projection(shard, name)
        print(f"\n{name} params={params[0].shape}/{params[1].shape}/{params[2].shape}")
        for tokens, (routed_x, idx, plan) in routes.items():
            if name == "down_proj":
                x = mx.random.normal((idx.size, 1, I), dtype=mx.bfloat16)
                mx.eval(x)
            else:
                x = routed_x
            ref = stock(x, params, idx)
            got = native(x, params, plan)
            mx.eval(ref, got)
            equal = bool(
                mx.array_equal(ref.view(mx.uint16), got.view(mx.uint16)).item()
            )
            max_abs = float(mx.max(mx.abs(ref - got)).item())
            print(f"{name} M={tokens}: bitwise={equal} max_abs={max_abs}")
            repeats = 13 if tokens == 256 else (11 if tokens == 2048 else 9)
            st, nt = interleaved(
                lambda x=x, p=params, i=idx: stock(x, p, i),
                lambda x=x, p=params, pl=plan: native(x, p, pl),
                repeats,
            )
            sm, nm = summarize(name, tokens, st, nt)
            totals[tokens][0] += sm
            totals[tokens][1] += nm
            del ref, got
            if name == "down_proj":
                del x
        del params
        gc.collect()
        mx.clear_cache()

    print("\nTOTAL THREE PROJECTIONS")
    for tokens, (stock_ms, native_ms) in totals.items():
        # 42 routed layers per model chunk; model calls one such projection
        # triplet per layer.
        saved_layer = stock_ms - native_ms
        print(
            f"M={tokens}: stock={stock_ms:.3f} native={native_ms:.3f} "
            f"speedup={stock_ms/native_ms:.3f}x saved/layer={saved_layer:.3f}ms "
            f"saved/42layers={saved_layer*42/1000:.3f}s"
        )


if __name__ == "__main__":
    main()
