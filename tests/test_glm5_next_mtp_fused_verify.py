# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 Lightning MTP verify forwards on the fused decode/verify kernels.

The MTP runtime's verify forward (``return_hidden=True``, i.e. ``gdn_sink``)
runs the attention half-layer on the vendor's exact fused kernels and records
one ``KdaStepCapture`` per KDA layer, which the rollback replays through the
same kernel. Against the reference verify path (``_FUSED_VERIFY`` off: stock
HC ops and the captured reference KDA body) these tests require, bit for bit:
the verify logits and hidden states, every cache after the rollback for each
accept count 0..depth, and the following verify/decode forwards; and the MTP
head forward against the reference ops.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_vlm_glm5_next_compat as compat


def _nax() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


# The fused kernels replay the reference graph bit for bit on M5 (NAX) GPUs,
# where they are enabled; other GPUs keep the reference path.
pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or not _nax(),
    reason="fused GLM decode kernels are validated and enabled on M5 (NAX) GPUs",
)

VOCAB = 256
# (bits of q/k/f_a/g_a/b/f_b/g_b, bits of v_proj) per KDA layer: 8-bit gate
# rows replay in the kernel (qmv_quad); 5-bit ones run outside it for verify
# blocks (a_pre/gate_pre, replayed from the capture); a 5-bit v_proj among
# 8-bit projections takes the grouped input projection (GLM-5.3 layer 40).
KDA_BITS = {
    "8-bit/5-bit": ((8, 8), (5, 5)),
    "mixed-v/4-bit": ((8, 5), (4, 4)),
}


def _runtime():
    compat.apply_mlx_vlm_glm5_next_compat_patch()
    from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime as rt

    assert rt.apply()
    from mlx_vlm.models.glm5_next import language

    return rt, language


def _rand_affine(out_dims, in_dims, bits, group_size=64, scale=0.004):
    words = in_dims * bits // 32
    w = mx.random.randint(0, 2**31 - 1, (out_dims, words)).astype(mx.uint32)
    w = w * 2 + mx.random.randint(0, 2, w.shape).astype(mx.uint32)
    groups = in_dims // group_size
    s = (mx.random.uniform(0.5, 1.5, (out_dims, groups)) * scale).astype(mx.bfloat16)
    b = (-mx.random.uniform(0.5, 1.5, (out_dims, groups)) * scale * 2**bits / 2).astype(
        mx.bfloat16
    )
    return w, s, b


def _quantized_linear(out_dims, in_dims, bits):
    layer = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=bits)
    layer.weight, layer.scales, layer.biases = _rand_affine(out_dims, in_dims, bits)
    return layer


def _model(seed, kda_bits, heads=16):
    """A small quantized GLM-5.3 with one MTP layer whose shapes engage every
    fused decode/verify path (KDA: 8 heads of 128)."""
    from mlx_vlm.models import glm5_next

    from omlx.patches.deepseek_v4.switch_layers import SwitchLinear

    from omlx.patches.mlx_vlm_mtp import is_mtp_attach_enabled, set_mtp_attach_enabled

    _, language = _runtime()
    text = glm5_next.TextConfig(
        model_type="glm5_next_text", vocab_size=VOCAB, hidden_size=1024,
        intermediate_size=2048, moe_intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=heads, num_key_value_heads=heads, n_shared_experts=1,
        n_routed_experts=128, routed_scaling_factor=2.5, kv_lora_rank=512,
        q_lora_rank=256, qk_rope_head_dim=0, v_head_dim=64, qk_nope_head_dim=64,
        num_experts_per_tok=8, first_k_dense_replace=1, max_position_embeddings=8192,
        rms_norm_eps=1e-5, index_topk=2048, index_head_dim=128, index_n_heads=32,
        layer_types=[
            "linear_attention", "deepseek_sparse_attention",
            "linear_attention", "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        linear_attn_config={
            "num_heads": 8, "head_dim": 128, "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=4, hc_mult=4, hc_sinkhorn_iters=20, num_nextn_predict_layers=1,
    )
    mx.random.seed(seed)
    attach = is_mtp_attach_enabled()
    set_mtp_attach_enabled(True)
    try:
        model = language.LanguageModel(text)
    finally:
        set_mtp_attach_enabled(attach)
    assert hasattr(model, "mtp")
    nn.quantize(
        model, group_size=64, bits=4,
        class_predicate=lambda _, m: isinstance(m, (nn.Linear, SwitchLinear)),
    )
    params = []
    for name, value in nn.utils.tree_flatten(model.parameters()):
        if value.dtype == mx.uint32:
            continue
        if language.glm5_next_cast_predicate(name):
            value = (mx.random.normal(value.shape) * 0.05).astype(mx.bfloat16)
            if name.endswith("scales"):
                value = mx.abs(value) * 0.1 + 0.002
            if "norm" in name and name.endswith("weight"):
                value = (1.0 + value).astype(mx.bfloat16)
        else:
            value = mx.random.normal(value.shape) * 0.05
            if name.endswith("hc.scale") or name.endswith("hc.base"):
                value = value * 10
        params.append((name, value))
    model.load_weights(params, strict=False)
    kda_layers = [layer.self_attn for layer in model.model.layers if layer.is_linear]
    for attn, (bits, v_bits) in zip(kda_layers, kda_bits):
        qkv, hidden = attn.qkv_dim, attn.hidden_size
        for name, (out_dims, in_dims) in {
            "q_proj": (qkv, hidden), "k_proj": (qkv, hidden), "v_proj": (qkv, hidden),
            "g_a_proj": (128, hidden), "b_proj": (attn.num_heads, hidden),
            "g_b_proj": (qkv, 128),
        }.items():
            setattr(
                attn, name, _quantized_linear(out_dims, in_dims, v_bits if name == "v_proj" else bits)
            )
        attn.forget_gate.f_a_proj = _quantized_linear(128, hidden, bits)
        attn.forget_gate.f_b_proj = _quantized_linear(qkv, 128, bits)
    model.eval()
    mx.eval(model.parameters())
    return model


def _bitwise_equal(a: mx.array, b: mx.array) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32}.get(
        a.dtype
    )
    if view is not None:
        a, b = a.view(view), b.view(view)
    return bool(mx.array_equal(a, b).item())


def _cache_items(cache):
    """Every array and position scalar that defines a model cache's state."""
    items = []
    for i, layer in enumerate(cache):
        subs = getattr(layer, "caches", None) or [layer]
        for j, c in enumerate(subs):
            state = c.state
            for k, value in enumerate(state if isinstance(state, (list, tuple)) else [state]):
                items.append(((i, j, k), value))
            for attr in ("offset", "remainder", "_processed", "lengths", "left_padding"):
                items.append(((i, j, attr), getattr(c, attr, None)))
    return items


def _assert_items_equal(left, right, where):
    assert [k for k, _ in left] == [k for k, _ in right], where
    mx.eval([v for _, v in left + right if isinstance(v, mx.array)])
    for (key, a), (_, b) in zip(left, right):
        if isinstance(a, mx.array) or isinstance(b, mx.array):
            assert isinstance(a, mx.array) and isinstance(b, mx.array), (where, key)
            assert _bitwise_equal(a, b), (where, key)
        else:
            assert a == b, (where, key, a, b)


def _assert_caches_equal(fused, reference, where):
    _assert_items_equal(_cache_items(fused), _cache_items(reference), where)


def check_verify_matches_reference(
    seed, prompt_len, cycles, kda_bits, head_steps=2, armed=False
):
    """Fused vs reference MTP verify cycles on two identical models.

    ``cycles`` lists ``(width, accepted)``: a verify block of ``width`` tokens
    (1 + depth), then the rollback keeping ``accepted`` drafts. ``armed``
    arms the verify-shape qmm routing around both verify forwards, as the
    batch generator does in production. Returns the fused kernel families
    used by the fused model's verify forwards.
    """
    from collections import Counter

    from omlx.patches import qwen35_verify_qmm as verify_qmm
    from omlx.patches.mlx_lm_mtp import cache_rollback
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    if armed:
        assert verify_qmm.apply_verify_qmm_patch()
    rt, language = _runtime()
    fused_model = _model(seed, kda_bits)
    reference_model = _model(seed, kda_bits)
    prompt = mx.random.randint(0, VOCAB, (1, prompt_len)).astype(mx.int32)
    caches = []
    for model in (fused_model, reference_model):
        cache = model.make_cache()
        for start in range(0, prompt_len, 512):
            logits = model(prompt[:, start : start + 512], cache=cache).logits
            mx.eval(logits)
        caches.append(cache)
    fused_cache, reference_cache = caches
    head_caches = [fused_model.make_mtp_cache(), reference_model.make_mtp_cache()]
    next_ids = mx.argmax(logits[:, -1:], axis=-1).astype(mx.int32)
    used = Counter()
    saved = rt._FUSED_VERIFY, language._DECODE_FUSION
    try:
        for step, (width, accepted) in enumerate(cycles):
            where = f"seed {seed} ctx {prompt_len} step {step} width {width} accept {accepted}"
            block = mx.concatenate(
                [next_ids, (next_ids * 7 + mx.arange(1, width)[None]) % VOCAB], axis=1
            )[:, :width]
            outs = []
            for model, cache, fused in (
                (fused_model, fused_cache, True),
                (reference_model, reference_cache, False),
            ):
                rt._FUSED_VERIFY = fused
                before = Counter(dk.STATS)
                cache_rollback.set_undo_armed(True)
                verify_qmm.set_verify_qmm_armed(armed)
                try:
                    out = model(block, cache=cache, return_hidden=True)
                finally:
                    verify_qmm.set_verify_qmm_armed(False)
                    cache_rollback.set_undo_armed(False)
                mx.eval(out.logits, out.hidden_states)
                if fused:
                    used += Counter(dk.STATS) - before
                outs.append(out)
            rt._FUSED_VERIFY = True
            fused_out, reference_out = outs
            kda = [l.self_attn for l in fused_model.model.layers if l.is_linear]
            for attn, entry in zip(kda, fused_out.gdn_states):
                # Under armed routes a layer with separately quantized
                # projections keeps the reference body for 3+ row blocks.
                reference_body = armed and width >= 3 and not attn._fused_ready
                assert isinstance(entry, tuple if reference_body else language.KdaStepCapture)
            assert all(isinstance(e, tuple) for e in reference_out.gdn_states)
            assert mx.all(mx.isfinite(reference_out.logits)).item(), where
            assert _bitwise_equal(fused_out.logits, reference_out.logits), where
            assert _bitwise_equal(
                fused_out.hidden_states[-1], reference_out.hidden_states[-1]
            ), where

            kept = []
            for model, cache, out in (
                (fused_model, fused_cache, fused_out),
                (reference_model, reference_cache, reference_out),
            ):
                m = model.mtp_clamp_accept(cache, accepted, width - 1)
                model.rollback_speculative_cache(cache, out.gdn_states, m, width)
                kept.append(m)
            assert kept[0] == kept[1], where
            _assert_caches_equal(fused_cache, reference_cache, where)

            # MTP head: fold the committed rows, then chain drafts; fused
            # kernels against the reference ops (identical inputs).
            m = kept[0]
            correction = mx.argmax(reference_out.logits[:, m : m + 1], axis=-1).astype(
                mx.int32
            )
            committed = mx.concatenate([block[:, 1 : m + 1], correction], axis=1)
            hidden = reference_out.hidden_states[-1][:, : m + 1]
            heads = []
            for model, head_cache, fusion in (
                (fused_model, head_caches[0], True),
                (reference_model, head_caches[1], False),
            ):
                language._DECODE_FUSION = fusion
                results = []
                logits, h = model.mtp_forward(
                    hidden, committed, head_cache, return_hidden=True, logits_keep=1
                )
                results += [logits, h]
                for _ in range(head_steps):
                    tok = mx.argmax(logits[:, -1:], axis=-1).astype(mx.int32)
                    logits, h = model.mtp_forward(
                        h[:, -1:], tok, head_cache, return_hidden=True
                    )
                    results += [logits, h]
                mx.eval(results)
                heads.append(results)
            language._DECODE_FUSION = saved[1]
            for i, (a, b) in enumerate(zip(*heads)):
                assert _bitwise_equal(a, b), (where, "mtp head", i)
            next_ids = correction
    finally:
        rt._FUSED_VERIFY, language._DECODE_FUSION = saved
    # A plain decode step on the rolled-back caches (fused vs reference ops).
    x = fused_model(next_ids, cache=fused_cache).logits
    language._DECODE_FUSION = False
    try:
        y = reference_model(next_ids, cache=reference_cache).logits
    finally:
        language._DECODE_FUSION = saved[1]
    mx.eval(x, y)
    assert _bitwise_equal(x, y), f"seed {seed} ctx {prompt_len} plain decode"
    return {k for k, v in used.items() if v}


# Every (width, accept) pair of depths 1..3 plus deeper and one-token blocks.
_CYCLES = [
    (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 3), (4, 0), (4, 2), (4, 1),
    (1, 0), (8, 7), (8, 2), (8, 0), (5, 4), (2, 1), (6, 3), (7, 6), (7, 0),
]


def _native_indexer_available() -> bool:
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast

        return bool(
            fast.has_symbol("dsa_indexer_scores") and fast.has_symbol("dsa_topk_indices")
        )
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize("kda", sorted(KDA_BITS))
@pytest.mark.parametrize("seed", [3, 17])
def test_fused_verify_cycles_are_bitwise_reference(seed, kda):
    used = check_verify_matches_reference(seed, 300, _CYCLES, KDA_BITS[kda])
    assert {"kda", "hc_mix"} <= used, used


def test_fused_verify_cycles_with_sparse_selection_are_bitwise_reference():
    """Past index_topk the sparse layers select (indexer + pooling undo)."""
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    used = check_verify_matches_reference(
        29, 2101, _CYCLES[:10], KDA_BITS["8-bit/5-bit"]
    )
    assert {"kda", "hc_mix", "dsa_scores", "dsa_topk"} <= used, used


def test_fused_verify_cycles_under_armed_verify_qmm_routing_are_bitwise_reference():
    """The batch generator arms the verify-shape qmm routes around every MTP
    verify forward. A KDA layer whose projections do not share one
    quantization (a 5-bit v_proj among 8-bit ones) runs them as separate
    QuantizedLinear calls in the reference body, which the routes take for
    3+ rows; the fused verify path must give the same values there too."""
    used = check_verify_matches_reference(
        23, 300, _CYCLES, KDA_BITS["mixed-v/4-bit"], armed=True
    )
    assert {"kda", "hc_mix"} <= used, used


def test_fused_verify_cycles_are_bitwise_reference_with_nax_tf32():
    """The production default runs fp32 GEMMs on NAX (TF32), where the
    one-token HC expand and the verify router kernels also engage."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    code = (
        "import sys; sys.path[:0] = [%r, %r]\n"
        "import test_glm5_next_mtp_fused_verify as t\n"
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "used = []\n"
        "for seed, ctx in ((5, 300), (11, 2101)):\n"
        "    for bits in t.KDA_BITS.values():\n"
        "        used.append(sorted(\n"
        "            t.check_verify_matches_reference(seed, ctx, t._CYCLES[:12], bits)))\n"
        "used.append(sorted(t.check_verify_matches_reference(\n"
        "    19, 300, t._CYCLES, t.KDA_BITS['mixed-v/4-bit'], armed=True)))\n"
        "print(dk.nax_relaxed_fp32_matmul(), used)\n" % (str(here), str(here.parent))
    )
    env = dict(os.environ, MLX_ENABLE_TF32="1", OMLX_GLM5_DECODE_DISABLE="")
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1200
    )
    assert done.returncode == 0, done.stderr[-4000:]
    nax_tf32, used = done.stdout.strip().splitlines()[-1].split(" ", 1)
    for families in eval(used):
        assert {"kda", "hc_mix"} <= set(families), families
        if nax_tf32 == "True":
            # The compiled verify FFN traces the fused router (its first-use
            # check ran eagerly), and one-token blocks the fused HC expand.
            assert {"router_rows", "hc_expand"} <= set(families), families


def _generate(model, prompt, count, fused, use_mtp):
    """Greedy tokens from mlx-lm's BatchGenerator with oMLX's MTP loop."""
    from types import SimpleNamespace

    from mlx_lm.generate import BatchGenerator

    from omlx.models.vlm import VLMModelAdapter

    rt, _ = _runtime()
    saved = rt._FUSED_VERIFY
    rt._FUSED_VERIFY = fused
    model._omlx_mtp_decode_enabled = use_mtp
    adapter = VLMModelAdapter(
        SimpleNamespace(config=SimpleNamespace(model_type="glm5_next"), language_model=model)
    )
    generator = BatchGenerator(
        adapter, max_tokens=count, prefill_step_size=512,
        sampler=lambda lp: mx.argmax(lp, -1),
    )
    tokens = []
    try:
        generator.insert([prompt])
        for _ in range(4 * count):
            _, responses = generator.next()
            tokens.extend(response.token for response in responses)
            if len(tokens) >= count:
                break
    finally:
        generator.close()
        rt._FUSED_VERIFY = saved
    return tokens[:count]


@pytest.mark.parametrize(
    "seed,kda,depth", [(3, "8-bit/5-bit", 1), (17, "mixed-v/4-bit", 3)]
)
def test_mtp_generation_emits_the_reference_path_tokens(seed, kda, depth, monkeypatch):
    """End to end through the MTP loop (verify, accept, rollback, head fold
    and chain): the fused verify path emits the reference path's tokens."""
    from mlx_vlm.models.cache import ArraysCache

    from omlx.patches import mlx_lm_mtp
    from omlx.patches.mlx_lm_mtp import batch_generator
    from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk

    _, language = _runtime()
    assert batch_generator.apply()
    # The engine's scheduler hands a single sequence's recurrent caches over
    # unpadded (no SSM mask); a bare BatchGenerator merges them with a zero
    # left padding whose all-true mask keeps the fused KDA step off. Same
    # values either way; emulate the engine.
    make_mask = ArraysCache.make_mask

    def unpadded(self, n):
        pad = self.left_padding
        if pad is not None and self.lengths is None and pad.size == 1 and pad.item() <= 0:
            return None
        return make_mask(self, n)

    monkeypatch.setattr(ArraysCache, "make_mask", unpadded)
    monkeypatch.setattr(mlx_lm_mtp, "_MTP_ACTIVE", True)
    mlx_lm_mtp_depth = mlx_lm_mtp.get_mtp_depth(), mlx_lm_mtp.is_mtp_depth_fixed()
    mlx_lm_mtp.set_mtp_depth(depth, fixed=True)
    try:
        models = {name: _model(seed, KDA_BITS[kda]) for name in ("fused", "reference")}
    finally:
        mlx_lm_mtp.set_mtp_depth(*mlx_lm_mtp_depth)
    prompt = [int(v) for v in mx.random.randint(0, VOCAB, (300,), key=mx.random.key(seed)).tolist()]
    rollbacks = []
    rollback = language.LanguageModel.rollback_speculative_cache

    def counted(self, caches, gdn_states, accepted, block_size):
        rollbacks.append((int(accepted), int(block_size)))
        return rollback(self, caches, gdn_states, accepted, block_size)

    monkeypatch.setattr(language.LanguageModel, "rollback_speculative_cache", counted)
    before = dk.STATS["kda"]
    fused = _generate(models["fused"], prompt, 20, True, True)
    fused_kda = dk.STATS["kda"] - before
    rollbacks_fused = list(rollbacks)
    before = dk.STATS["kda"]
    reference = _generate(models["reference"], prompt, 20, False, True)
    reference_kda = dk.STATS["kda"] - before
    assert fused == reference
    assert any(b > 1 for _, b in rollbacks_fused), rollbacks_fused
    assert rollbacks[len(rollbacks_fused):] == rollbacks_fused
    assert fused_kda > reference_kda, (fused_kda, reference_kda)


def test_rollback_of_a_fused_block_replays_only_partial_accepts(monkeypatch):
    """A fully kept block leaves the forward's own states in the cache; a
    partial accept replays the kept rows through the fused kernel."""
    rt, language = _runtime()
    model = _model(7, KDA_BITS["8-bit/5-bit"])
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3, 4, 5]]), cache=cache).logits)
    replays = []
    original = language.KdaStepCapture.replay

    def replay(self, n):
        replays.append(n)
        return original(self, n)

    monkeypatch.setattr(language.KdaStepCapture, "replay", replay)
    for accepted, expected in ((3, []), (1, [2, 2])):
        from omlx.patches.mlx_lm_mtp import cache_rollback

        cache_rollback.set_undo_armed(True)
        try:
            out = model(mx.array([[6, 7, 8, 9]]), cache=cache, return_hidden=True)
        finally:
            cache_rollback.set_undo_armed(False)
        kda = [c for c in cache if not hasattr(c, "caches")]
        final = [(c[0], c[1]) for c in kda]
        replays.clear()
        model.rollback_speculative_cache(cache, out.gdn_states, accepted, 4)
        assert replays == expected
        if not expected:
            assert all(c[0] is a and c[1] is b for c, (a, b) in zip(kda, final))


def test_fused_capture_refuses_a_mismatched_block_before_touching_caches():
    """A capture whose width is not the verify block's is refused before
    any layer (sparse ones included) is trimmed or rewritten."""
    from omlx.patches.mlx_lm_mtp import cache_rollback

    _runtime()
    model = _model(9, KDA_BITS["8-bit/5-bit"])
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3, 4, 5]]), cache=cache).logits)
    cache_rollback.set_undo_armed(True)
    try:
        out = model(mx.array([[6, 7, 8, 9]]), cache=cache, return_hidden=True)
    finally:
        cache_rollback.set_undo_armed(False)
    mx.eval(out.logits)
    before = _cache_items(cache)
    with pytest.raises(RuntimeError, match="KDA capture"):
        model.rollback_speculative_cache(cache, out.gdn_states, 0, 3)
    _assert_items_equal(before, _cache_items(cache), "refused rollback")


def test_batched_rollback_rejects_a_fused_capture():
    _, language = _runtime()
    from omlx.patches.mlx_vlm_mtp.glm5_next_batch_rollback import rollback_rows
    from mlx_vlm.models.cache import ArraysCache

    cache = ArraysCache(size=2)
    cache[0], cache[1] = mx.zeros((2, 3, 8)), mx.zeros((2, 1, 1, 1))
    capture = language.KdaStepCapture(None, mx.zeros((2, 4, 8)), None, None)
    with pytest.raises(ValueError):
        rollback_rows(language, [cache], [capture], [0, 1], 4)
