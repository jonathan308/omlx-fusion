from __future__ import annotations

import statistics
import time

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v4.switch_layers import _build_mxfp4_blocks, _gather_sort
from omlx.patches.glm_moe_dsa.deepseek_v32 import group_expert_select


MODEL = "/Users/jonathanspangler/.lmstudio/models/Jundot/GLM-5.3-Flash-oQ4e"
LP = "language_model.model.layers.3.mlp"
CONFIGS = {1:(16,32),2:(32,32),3:(16,64),4:(32,64),5:(64,32),6:(64,64),7:(128,32),8:(128,64)}


def timed(fn):
    start=time.perf_counter(); mx.eval(fn()); mx.synchronize(); return (time.perf_counter()-start)*1000


def main():
    d=mx.load(f"{MODEL}/model-00002-of-00034.safetensors")
    rw=mx.contiguous(d[f"{LP}.gate.weight"]); rb=mx.contiguous(d[f"{LP}.gate.e_score_correction_bias"])
    pfx=f"{LP}.switch_mlp.gate_proj"
    w,s,b=(mx.contiguous(d[f"{pfx}.{k}"]) for k in ("weight","scales","biases"))
    mx.eval(rw,rb,w,s,b); del d
    x0=mx.random.normal((1,8192,4096),dtype=mx.bfloat16)
    inds,_=group_expert_select(x0.astype(mx.float32)@rw.astype(mx.float32).T,rb,8,1,1,2.5,True)
    x,idx,_=_gather_sort(mx.expand_dims(x0,(-2,-3)),inds); mx.eval(x,idx)
    def stock(): return mx.gather_qmm(x,w,s,b,rhs_indices=idx,transpose=True,group_size=64,bits=4,mode='affine',sorted_indices=True)
    ref=stock(); mx.eval(ref)
    for _ in range(3): mx.eval(stock())
    st=[timed(stock) for _ in range(15)]
    sm=statistics.median(st)
    print('stock',sm,min(st),st)
    for v,(bm,bn) in CONFIGS.items():
        plan=_build_mxfp4_blocks(idx,288,bm); mx.eval(*plan)
        def fn(v=v,p=plan): return fast.deepseek_affine_gather_qmm_blocks(x,w,s,b,p[0],p[1],64,4,v)
        y=fn(); mx.eval(y)
        eq=bool(mx.array_equal(ref.view(mx.uint16),y.view(mx.uint16)).item())
        for _ in range(3): mx.eval(fn())
        ts=[timed(fn) for _ in range(15)]
        med=statistics.median(ts)
        print(v,bm,bn,'parity',eq,'median',med,'min',min(ts),'speedup',sm/med,'raw',ts)


if __name__=='__main__': main()
