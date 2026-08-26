from __future__ import annotations

import subprocess
import sys

import pytest

from omlx.patches.glm5_next.kda import (
    OFFICIAL_DSA_LAYERS,
    OFFICIAL_KDA_LAYERS,
    KDACache,
    KDAConfig,
    KDAContractError,
    make_kda_class,
    validate_kda_config,
    validate_kda_weights,
)
from omlx.patches.glm5_next.mhc import (
    MHCConfig,
    MHCContractError,
    apply_mhc_residual,
    make_hyper_head_class,
    make_mhc_class,
    validate_mhc_config,
    validate_mhc_weights,
)


def _text_config():
    return {
        "model_type": "glm5_next_text",
        "hidden_size": 4096,
        "num_hidden_layers": 45,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "mhc": True,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        "layer_types": [
            "linear_attention"
            if i in OFFICIAL_KDA_LAYERS
            else "deepseek_sparse_attention"
            for i in range(45)
        ],
        "linear_attn_config": {
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": list(OFFICIAL_KDA_LAYERS),
            "full_attn_layers": list(OFFICIAL_DSA_LAYERS),
        },
    }


def test_core_modules_do_not_eagerly_import_mlx():
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'mlx' or name.startswith('mlx.') or name == 'mlx_lm' or name.startswith('mlx_lm.'):
        raise AssertionError('eager MLX import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import omlx.patches.glm5_next.kda
import omlx.patches.glm5_next.mhc
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_exact_zero_based_34_layer_kda_placement():
    assert len(OFFICIAL_KDA_LAYERS) == 34
    assert OFFICIAL_KDA_LAYERS[:6] == (0, 1, 2, 4, 5, 6)
    assert OFFICIAL_KDA_LAYERS[-4:] == (40, 41, 42, 44)
    assert tuple(range(3, 45, 4)) == OFFICIAL_DSA_LAYERS
    assert set(OFFICIAL_KDA_LAYERS) | set(OFFICIAL_DSA_LAYERS) == set(range(45))
    validate_kda_config({"text_config": _text_config()})


def test_kda_schedule_validator_rejects_kimi_one_based_alias():
    config = _text_config()
    config["linear_attn_config"]["kda_layers"] = [i + 1 for i in OFFICIAL_KDA_LAYERS]
    with pytest.raises(KDAContractError, match="kda_layers changed"):
        validate_kda_config(config)


def _kda_headers():
    specs = {
        "A_log": ([64], "F32"),
        "b_proj.weight": ([64, 4096], "BF16"),
        "dt_bias": ([8192], "F32"),
        "f_a_proj.weight": ([128, 4096], "BF16"),
        "f_b_proj.weight": ([8192, 128], "BF16"),
        "g_a_proj.weight": ([128, 4096], "BF16"),
        "g_b_proj.weight": ([8192, 128], "BF16"),
        "k_conv1d.weight": ([8192, 1, 4], "BF16"),
        "k_proj.weight": ([8192, 4096], "BF16"),
        "o_norm.weight": ([128], "BF16"),
        "o_proj.weight": ([4096, 8192], "BF16"),
        "q_conv1d.weight": ([8192, 1, 4], "BF16"),
        "q_proj.weight": ([8192, 4096], "BF16"),
        "v_conv1d.weight": ([8192, 1, 4], "BF16"),
        "v_proj.weight": ([8192, 4096], "BF16"),
    }
    return {
        f"model.language_model.layers.{layer}.self_attn.{suffix}": {
            "shape": shape,
            "dtype": dtype,
        }
        for layer in OFFICIAL_KDA_LAYERS
        for suffix, (shape, dtype) in specs.items()
    }


def test_kda_weight_validator_checks_placement_shape_and_dtype():
    headers = _kda_headers()
    validate_kda_weights(headers)
    bad = dict(headers)
    key = "model.language_model.layers.0.self_attn.q_conv1d.weight"
    bad[key] = {"shape": [8192, 4, 1], "dtype": "BF16"}
    with pytest.raises(KDAContractError, match="shape changed"):
        validate_kda_weights(bad)
    bad = dict(headers)
    bad["model.language_model.layers.3.self_attn.A_log"] = {
        "shape": [64],
        "dtype": "F32",
    }
    with pytest.raises(KDAContractError, match=r"extra=\[3\]"):
        validate_kda_weights(bad)
    bad = dict(headers)
    bad["model.language_model.layers.0.self_attn.not_official"] = (
        "some-shard.safetensors"
    )
    with pytest.raises(KDAContractError, match="unexpected KDA tensor"):
        validate_kda_weights(bad)


def test_mhc_contract_and_all_45_main_layers_but_not_mtp():
    validate_mhc_config(_text_config())
    specs = {
        "hc_attn_fn": ([24, 16384], "BF16"),
        "hc_attn_base": ([24], "F32"),
        "hc_attn_scale": ([3], "F32"),
        "hc_ffn_fn": ([24, 16384], "BF16"),
        "hc_ffn_base": ([24], "F32"),
        "hc_ffn_scale": ([3], "F32"),
    }
    headers = {
        f"model.language_model.layers.{layer}.{suffix}": {
            "shape": shape,
            "dtype": dtype,
        }
        for layer in range(45)
        for suffix, (shape, dtype) in specs.items()
    }
    validate_mhc_weights(headers)
    headers["model.language_model.layers.45.hc_attn_fn"] = {
        "shape": [24, 16384],
        "dtype": "BF16",
    }
    with pytest.raises(MHCContractError, match=r"extra=\[45\]"):
        validate_mhc_weights(headers)


def test_kda_cache_update_snapshot_commit_and_exact_rollback():
    cache = KDACache()
    original = tuple(object() for _ in range(4))
    cache.update(*original, tokens=7)
    checkpoint = cache.begin_update()
    changed = tuple(object() for _ in range(4))
    cache.update(*changed, tokens=2)
    assert cache.offset == 9
    assert tuple(cache.state) == changed
    cache.rollback(checkpoint)
    assert cache.offset == 7
    assert tuple(cache.state) == original

    checkpoint = cache.begin_update()
    cache.update(*changed, tokens=1)
    cache.commit(checkpoint)
    assert cache.offset == 8
    assert cache.trim(1) == 0
    assert cache.is_trimmable() is False
    assert cache.size() == 8
    assert cache.meta_state == ("8",)


def test_lazy_mlx_mhc_shapes_sinkhorn_and_residual_equation():
    mx = pytest.importorskip("mlx.core")
    mhc_class = make_mhc_class()
    hyper_head_class = make_hyper_head_class()
    config = MHCConfig(hidden_size=8, streams=2, eps=1e-6, sinkhorn_iters=8)
    module = mhc_class(config)
    streams = mx.random.normal((2, 3, 2, 8)).astype(mx.float16)
    post, comb, collapsed = module(streams)
    branch = mx.random.normal((2, 3, 8)).astype(mx.float16)
    updated = apply_mhc_residual(post, comb, branch, streams)
    head = hyper_head_class()(updated)
    mx.eval(post, comb, collapsed, updated, head)

    assert post.shape == (2, 3, 2)
    assert comb.shape == (2, 3, 2, 2)
    assert collapsed.shape == (2, 3, 8)
    assert updated.shape == streams.shape
    assert head.shape == (2, 3, 8)
    assert collapsed.dtype == mx.float16
    assert comb.dtype == mx.float32
    assert mx.allclose(comb.sum(axis=-1), mx.ones((2, 3, 2)), atol=2e-4).item()
    assert mx.allclose(comb.sum(axis=-2), mx.ones((2, 3, 2)), atol=2e-4).item()


def test_lazy_mlx_kda_shape_fp32_state_and_chunk_update_equivalence():
    mx = pytest.importorskip("mlx.core")
    kda_class = make_kda_class()
    config = KDAConfig(hidden_size=8, num_heads=2, head_dim=4, conv_kernel_size=3)
    module = kda_class(config)
    x = mx.random.normal((1, 5, 8)).astype(mx.float16)

    full_cache = KDACache()
    full = module(x, cache=full_cache, use_kernel=False)
    step_cache = KDACache()
    pieces = [
        module(x[:, i : i + 1], cache=step_cache, use_kernel=False) for i in range(5)
    ]
    stepped = mx.concatenate(pieces, axis=1)
    mx.eval(full, stepped, *full_cache.state, *step_cache.state)

    assert full.shape == (1, 5, 8)
    assert full.dtype == mx.float16
    assert full_cache[0].shape == (1, 2, 8)
    assert full_cache[3].shape == (1, 2, 4, 4)
    assert full_cache[3].dtype == mx.float32
    assert full_cache.offset == step_cache.offset == 5
    assert mx.allclose(full, stepped, atol=2e-3, rtol=2e-3).item()
