# SPDX-License-Identifier: Apache-2.0
"""Fail-closed tests for Qwen3.8 Flash Next core primitives."""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest


def _config(**overrides):
    values = {
        "hidden_size": 2560,
        "linear_num_value_heads": 48,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "hidden_act": "silu",
        "output_gate_type": "sigmoid",
        "mamba_ssm_dtype": "float32",
        "layer_types": ["linear_attention", "qwen_sparse_attention"],
        "hc_count": 4,
        "hc_lowrank": 320,
        "rms_norm_eps": 1e-6,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "norm_topk_prob": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@dataclass(frozen=True)
class _Tensor:
    shape: tuple[int, ...]
    marker: str = ""

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 3 or key[0] is not Ellipsis:
            raise AssertionError(f"unexpected packed-MoE slice: {key!r}")
        intermediate = key[1]
        if intermediate == slice(None, 640, None):
            marker = "gate"
        elif intermediate == slice(640, None, None):
            marker = "up"
        else:
            raise AssertionError(f"unexpected intermediate slice: {intermediate!r}")
        return _Tensor((self.shape[0], 640, self.shape[2]), marker)


def test_core_modules_do_not_eagerly_import_mlx():
    code = """
import importlib, sys
for name in (
    'omlx.patches.qwen4_exp.gdn',
    'omlx.patches.qwen4_exp.hc',
    'omlx.patches.qwen4_exp.moe',
):
    importlib.import_module(name)
print(any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("linear_num_value_heads", 32),
        ("linear_num_key_heads", 8),
        ("linear_key_head_dim", 64),
        ("linear_value_head_dim", 64),
        ("linear_conv_kernel_dim", 3),
        ("output_gate_type", "silu"),
        ("mamba_ssm_dtype", "bfloat16"),
    ],
)
def test_gdn_rejects_nearby_qwen_layouts(field, bad):
    gdn = importlib.import_module("omlx.patches.qwen4_exp.gdn")
    with pytest.raises(ValueError, match="GatedDeltaNet config"):
        gdn.validate_gdn_config(_config(**{field: bad}), layer_idx=0)


def test_gdn_accepts_only_linear_layer_and_official_split_projection():
    gdn = importlib.import_module("omlx.patches.qwen4_exp.gdn")
    gdn.validate_gdn_config(_config(), layer_idx=0)
    with pytest.raises(ValueError, match="cannot serve layer type"):
        gdn.validate_gdn_config(_config(), layer_idx=1)

    prefix = "model.language_model.layers.0.linear_attn"
    weights = {
        f"{prefix}.in_proj_qkv.weight": _Tensor((10240, 2560)),
        f"{prefix}.in_proj_z.weight": _Tensor((6144, 2560)),
        f"{prefix}.in_proj_b.weight": _Tensor((48, 2560)),
        f"{prefix}.in_proj_a.weight": _Tensor((48, 2560)),
        f"{prefix}.conv1d.weight": _Tensor((10240, 1, 4)),
        f"{prefix}.dt_bias": _Tensor((48,)),
        f"{prefix}.A_log": _Tensor((48,)),
        f"{prefix}.norm.weight": _Tensor((128,)),
        f"{prefix}.out_proj.weight": _Tensor((2560, 6144)),
    }
    gdn.validate_gdn_weight_layout(weights, prefix)

    fused = dict(weights)
    fused[f"{prefix}.in_proj_qkvz.weight"] = _Tensor((1,))
    with pytest.raises(ValueError, match="split in_proj_qkv/z/b/a"):
        gdn.validate_gdn_weight_layout(fused, prefix)


def test_gdn_forwards_nonzero_n_confirmed_only_to_rollback_capable_base():
    gdn = importlib.import_module("omlx.patches.qwen4_exp.gdn")
    seen = {}

    class Cache(list):
        rollback_state = None
        _mtp_draft_stash = None

    class PatchedOwner:
        def _process_chunk(self):
            raise AssertionError("dispatch must not execute replay itself")

    def patched_base(inputs, *, mask=None, cache=None, n_confirmed=0):
        seen.update(
            inputs=inputs,
            mask=mask,
            cache=cache,
            n_confirmed=n_confirmed,
        )
        cache.rollback_state = ("conv-before", "ssm-before-fp32")
        cache._mtp_draft_stash = ("qkv", "a", "b")
        return "verified"

    cache = Cache([None, None])
    output = gdn._call_pinned_gdn(
        PatchedOwner(),
        patched_base,
        "window",
        mask="mask",
        cache=cache,
        n_confirmed=1,
    )

    assert output == "verified"
    assert seen == {
        "inputs": "window",
        "mask": "mask",
        "cache": cache,
        "n_confirmed": 1,
    }
    assert cache.rollback_state == ("conv-before", "ssm-before-fp32")
    assert cache._mtp_draft_stash == ("qkv", "a", "b")


def test_gdn_fails_closed_if_nonzero_n_confirmed_reaches_unpatched_base():
    gdn = importlib.import_module("omlx.patches.qwen4_exp.gdn")
    calls = []

    class UnpatchedOwner:
        pass

    def unpatched_base(inputs, *, mask=None, cache=None):
        calls.append((inputs, mask, cache))
        return "ordinary"

    # Ordinary forwards omit the kwarg and remain compatible with an
    # unpatched/DFlash-shaped base.
    assert (
        gdn._call_pinned_gdn(
            UnpatchedOwner(),
            unpatched_base,
            "token",
            mask="mask",
            cache="cache",
            n_confirmed=0,
        )
        == "ordinary"
    )
    assert calls == [("token", "mask", "cache")]

    with pytest.raises(RuntimeError, match="refusing to advance recurrent state"):
        gdn._call_pinned_gdn(
            UnpatchedOwner(),
            unpatched_base,
            "verify-window",
            n_confirmed=1,
        )
    assert calls == [("token", "mask", "cache")]


def test_gdn_recurrent_cache_is_coerced_to_fp32_before_snapshot():
    gdn = importlib.import_module("omlx.patches.qwen4_exp.gdn")
    fp32 = object()

    class FakeMX:
        float32 = fp32

    class State:
        def __init__(self, dtype):
            self.dtype = dtype
            self.cast_targets = []

        def astype(self, dtype):
            self.cast_targets.append(dtype)
            return State(dtype)

    original = State("bfloat16")
    cache = [None, original]
    gdn._coerce_recurrent_state_fp32(cache, FakeMX)

    assert original.cast_targets == [fp32]
    assert cache[1].dtype is fp32


def test_hc_requires_four_streams_rank_320_and_exact_weight_names():
    hc = importlib.import_module("omlx.patches.qwen4_exp.hc")
    hc.validate_hc_config(_config())
    with pytest.raises(ValueError, match="hc_count"):
        hc.validate_hc_config(_config(hc_count=2))
    with pytest.raises(ValueError, match="hc_lowrank"):
        hc.validate_hc_config(_config(hc_lowrank=256))

    prefix = "model.language_model.layers.0.attn_hyper_connection"
    weights = {
        f"{prefix}.hc_norm.weight": _Tensor((10240,)),
        f"{prefix}.input_mix_weight_down.weight": _Tensor((320, 10240)),
        f"{prefix}.input_mix_weight_up.weight": _Tensor((10240, 320)),
        f"{prefix}.block_inject_weight.weight": _Tensor((4, 10240)),
    }
    hc.validate_hc_weight_layout(weights, prefix)


def test_final_hc_mixer_forbids_block_injection_weights():
    hc = importlib.import_module("omlx.patches.qwen4_exp.hc")
    prefix = "model.language_model.hyper_connection_mixer"
    weights = {
        f"{prefix}.hc_norm.weight": _Tensor((10240,)),
        f"{prefix}.input_mix_weight_down.weight": _Tensor((320, 10240)),
        f"{prefix}.input_mix_weight_up.weight": _Tensor((10240, 320)),
    }
    hc.validate_hc_weight_layout(weights, prefix, use_combine=False)
    weights[f"{prefix}.block_inject_weight.weight"] = _Tensor((4, 10240))
    with pytest.raises(ValueError, match="must not contain"):
        hc.validate_hc_weight_layout(weights, prefix, use_combine=False)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("num_experts", 256),
        ("num_experts_per_tok", 8),
        ("moe_intermediate_size", 512),
        ("shared_expert_intermediate_size", 512),
    ],
)
def test_moe_rejects_non_flash_next_dimensions(field, bad):
    moe = importlib.import_module("omlx.patches.qwen4_exp.moe")
    with pytest.raises(ValueError, match="MoE config"):
        moe.validate_moe_config(_config(**{field: bad}))


def _official_moe_weights(prefix: str):
    return {
        f"{prefix}.gate.weight": _Tensor((512, 2560)),
        f"{prefix}.experts.gate_up_proj": _Tensor((512, 1280, 2560), "packed"),
        f"{prefix}.experts.down_proj": _Tensor((512, 2560, 640), "down"),
        f"{prefix}.shared_expert.gate_proj.weight": _Tensor((640, 2560)),
        f"{prefix}.shared_expert.up_proj.weight": _Tensor((640, 2560)),
        f"{prefix}.shared_expert.down_proj.weight": _Tensor((2560, 640)),
        f"{prefix}.shared_expert_gate.weight": _Tensor((1, 2560)),
    }


def test_moe_validates_official_packed_512_top10_layout():
    moe = importlib.import_module("omlx.patches.qwen4_exp.moe")
    moe.validate_moe_config(_config())
    prefix = "model.language_model.layers.0.mlp"
    moe.validate_moe_weight_layout(_official_moe_weights(prefix), prefix)


def test_moe_sanitize_splits_backbone_and_mtp_without_dropping_mtp():
    moe = importlib.import_module("omlx.patches.qwen4_exp.moe")
    backbone = "model.language_model.layers.0.mlp"
    mtp = "model.language_model.mtp.layers.0.mlp"
    keep_key = "model.language_model.mtp.fc_embedding.weight"
    keep_value = object()
    weights = {
        **_official_moe_weights(backbone),
        **_official_moe_weights(mtp),
        keep_key: keep_value,
    }

    sanitized = moe.sanitize_moe_weights(weights)

    assert keep_key in sanitized
    assert sanitized[keep_key] is keep_value
    for prefix in (backbone, mtp):
        assert f"{prefix}.experts.gate_up_proj" not in sanitized
        assert f"{prefix}.experts.down_proj" not in sanitized
        assert sanitized[f"{prefix}.switch_mlp.gate_proj.weight"].marker == "gate"
        assert sanitized[f"{prefix}.switch_mlp.up_proj.weight"].marker == "up"
        assert sanitized[f"{prefix}.switch_mlp.down_proj.weight"].marker == "down"


def test_moe_sanitize_fails_closed_on_malformed_packed_weight():
    moe = importlib.import_module("omlx.patches.qwen4_exp.moe")
    prefix = "model.language_model.layers.0.mlp"
    weights = _official_moe_weights(prefix)
    weights[f"{prefix}.experts.gate_up_proj"] = _Tensor((512, 640, 2560))
    with pytest.raises(ValueError, match="Invalid qwen4_exp packed MoE"):
        moe.sanitize_moe_weights(weights)
