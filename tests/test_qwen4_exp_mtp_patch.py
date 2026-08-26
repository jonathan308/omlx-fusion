# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

import pytest

from omlx.patches.mlx_lm_mtp import qwen4_exp_model as patch


@dataclass
class Args:
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    hc_count: int = 4
    hc_lowrank: int = 320
    rms_norm_eps: float = 1e-6
    mtp_num_hidden_layers: int = 1
    tie_word_embeddings: bool = False
    mtp: dict = field(
        default_factory=lambda: {
            "hybrid": True,
            "layer_types": ["full_attention"],
            "num_hidden_layers": 1,
        }
    )


class Tensor:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, _index):
        return self


def official_weights():
    return {key: Tensor(shape) for key, shape in patch._REQUIRED_MTP_SHAPES.items()}


def test_official_depth_one_config_is_required():
    patch.validate_mtp_config(Args())
    bad = Args(mtp_num_hidden_layers=5)
    with pytest.raises(patch.Qwen4ExpMTPContractError, match="expected 1"):
        patch.validate_mtp_config(bad)


def test_exact_official_weight_contract_and_shapes():
    weights = official_weights()
    patch.validate_mtp_weights(weights)

    weights.pop("mtp.fc_hidden.weight")
    with pytest.raises(patch.Qwen4ExpMTPContractError, match="missing"):
        patch.validate_mtp_weights(weights)

    weights = official_weights()
    weights["mtp.fc_hidden.weight"] = Tensor((1, 1))
    with pytest.raises(patch.Qwen4ExpMTPContractError, match="shape mismatch"):
        patch.validate_mtp_weights(weights)


def test_lossless_switch_moe_layout_is_accepted():
    weights = official_weights()
    weights.pop("mtp.layers.0.mlp.experts.gate_up_proj")
    weights.pop("mtp.layers.0.mlp.experts.down_proj")
    weights.update(
        {
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight": Tensor((512, 640, 2560)),
            "mtp.layers.0.mlp.switch_mlp.up_proj.weight": Tensor((512, 640, 2560)),
            "mtp.layers.0.mlp.switch_mlp.down_proj.weight": Tensor((512, 2560, 640)),
        }
    )
    patch.validate_mtp_weights(weights)


def test_packed_and_switch_experts_together_fail_closed():
    weights = official_weights()
    weights.update(
        {
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight": Tensor((512, 640, 2560)),
            "mtp.layers.0.mlp.switch_mlp.up_proj.weight": Tensor((512, 640, 2560)),
            "mtp.layers.0.mlp.switch_mlp.down_proj.weight": Tensor((512, 2560, 640)),
        }
    )
    with pytest.raises(patch.Qwen4ExpMTPContractError, match="exactly one"):
        patch.validate_mtp_weights(weights)


class FakeModule:
    pass


class FakeLinear:
    def __init__(self, input_dims, output_dims, bias=False):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.bias = bias


class FakeKVCache:
    pass


class SafeMtpLayer:
    layer_type = "full_attention"
    _omlx_qwen4_exp_mtp_safe = True


def _install_fake_runtime(monkeypatch, *, safe_layer=True, include_factory=True):
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.zeros = lambda shape: Tensor(shape)
    fake_nn = types.ModuleType("mlx.nn")
    fake_nn.Module = FakeModule
    fake_nn.Linear = FakeLinear
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mx
    fake_mlx.nn = fake_nn
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx.nn", fake_nn)

    models = types.ModuleType("mlx_lm.models")
    cache = types.ModuleType("mlx_lm.models.cache")
    cache.KVCache = FakeKVCache
    q4 = types.ModuleType("mlx_lm.models.qwen4_exp")

    class TextModel:
        def __init__(self, args):
            self.args = args

        def __call__(self, inputs, cache=None, return_hidden=False):
            return inputs

        def sanitize(self, weights):
            return dict(weights)

    q4.TextModel = TextModel
    q4.create_attention_mask = lambda hidden, cache: (hidden, cache)
    q4.make_qsa_cache = FakeKVCache
    if include_factory:
        layer = (
            SafeMtpLayer()
            if safe_layer
            else types.SimpleNamespace(layer_type="full_attention")
        )
        q4.build_mtp_decoder_layer = lambda args: layer
    models.qwen4_exp = q4
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.models = models
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.qwen4_exp", q4)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)
    return q4


def test_apply_requires_explicit_native_factory(monkeypatch):
    _install_fake_runtime(monkeypatch, include_factory=False)
    with pytest.raises(patch.Qwen4ExpMTPContractError, match="build_mtp_decoder_layer"):
        patch.apply()


def test_apply_is_idempotent_and_disabled_load_strips_mtp(monkeypatch):
    q4 = _install_fake_runtime(monkeypatch)
    from omlx.patches import mlx_lm_mtp

    monkeypatch.setattr(mlx_lm_mtp, "_MTP_ACTIVE", False)
    assert patch.apply() is True
    assert patch.apply() is True
    model = q4.TextModel(Args())
    assert model._omlx_mtp_decode_enabled is False
    assert model._omlx_mtp_depth == 1
    assert model._omlx_mtp_chain is False
    assert model.make_mtp_cache() == []
    assert model.sanitize(official_weights()) == {}


def test_enabled_load_builds_one_safe_qsa_moe_layer_and_keeps_head(monkeypatch):
    q4 = _install_fake_runtime(monkeypatch)
    from omlx.patches import mlx_lm_mtp

    monkeypatch.setattr(mlx_lm_mtp, "_MTP_ACTIVE", True)
    assert patch.apply() is True
    model = q4.TextModel(Args())
    assert model._omlx_mtp_decode_enabled is True
    assert model._omlx_mtp_depth == 1
    assert len(model.mtp.layers) == 1
    assert model.mtp.layers[0].layer_type == "full_attention"
    assert len(model.make_mtp_cache()) == 1

    sanitized = model.sanitize(official_weights())
    assert "mtp.fc_embedding.weight" in sanitized
    assert "mtp.layers.0.mlp.experts.gate_up_proj" not in sanitized
    assert "mtp.layers.0.mlp.switch_mlp.gate_proj.weight" in sanitized


def test_outer_model_prefix_is_validated_and_preserved(monkeypatch):
    q4 = _install_fake_runtime(monkeypatch)
    from omlx.patches import mlx_lm_mtp

    monkeypatch.setattr(mlx_lm_mtp, "_MTP_ACTIVE", True)
    patch.apply()
    model = q4.TextModel(Args())
    outer_weights = {
        "language_model." + key: value for key, value in official_weights().items()
    }
    sanitized = model.sanitize(outer_weights)
    assert "language_model.mtp.fc_hidden.weight" in sanitized
    assert "language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight" in sanitized
    assert not any(key.startswith("mtp.") for key in sanitized), (
        "outer-model sanitize must not lose the language_model path"
    )


def test_unsafe_decoder_factory_fails_closed_at_construction(monkeypatch):
    q4 = _install_fake_runtime(monkeypatch, safe_layer=False)
    from omlx.patches import mlx_lm_mtp

    monkeypatch.setattr(mlx_lm_mtp, "_MTP_ACTIVE", True)
    patch.apply()
    with pytest.raises(patch.Qwen4ExpMTPContractError, match="mtp_safe"):
        q4.TextModel(Args())
