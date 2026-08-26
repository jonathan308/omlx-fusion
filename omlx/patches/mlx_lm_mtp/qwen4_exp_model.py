# SPDX-License-Identifier: Apache-2.0
"""Native depth-1 MTP binding for Qwen3.8 Flash Next (``qwen4_exp``).

This is deliberately a sibling of, not an alias for, :mod:`qwen35_model`.
The published head has a different graph and checkpoint layout:

* the backbone supplies its four-stream (10,240 feature) hidden state;
* ``pre_fc_norm_hidden`` and the head's hyper-connection mixer reduce it to
  2,560 features;
* ``pre_fc_norm_embedding`` independently normalizes the sampled-token
  embedding;
* ``fc_hidden`` and ``fc_embedding`` are added;
* the fused state is expanded to four streams and passed through exactly one
  full-attention QSA + MoE decoder layer; and
* the dedicated ``mtp.hyper_connection_mixer`` returns the 2,560-feature head
  output consumed by the shared language-model projection.

There is no dense-attention fallback here.  The native ``qwen4_exp`` module
must provide ``build_mtp_decoder_layer(args)`` and that factory must construct
the same fail-closed sparse-QSA layer used by the backbone.  This keeps model
loading honest while the native micro-block sparse kernel is still optional.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

OFFICIAL_MTP_DEPTH = 1
OFFICIAL_HIDDEN_SIZE = 2_560
OFFICIAL_HC_COUNT = 4
OFFICIAL_HYPER_SIZE = OFFICIAL_HIDDEN_SIZE * OFFICIAL_HC_COUNT


class Qwen4ExpMTPContractError(RuntimeError):
    """The model module, config, or checkpoint is not the official MTP graph."""


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else None


_REQUIRED_MTP_SHAPES: dict[str, tuple[int, ...]] = {
    "mtp.pre_fc_norm_embedding.weight": (2_560,),
    "mtp.pre_fc_norm_hidden.weight": (10_240,),
    "mtp.fc_embedding.weight": (2_560, 2_560),
    "mtp.fc_hidden.weight": (2_560, 2_560),
    "mtp.hyper_connection_mixer.hc_norm.weight": (10_240,),
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (320, 10_240),
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (10_240, 320),
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": (4, 10_240),
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": (10_240,),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": (320, 10_240),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": (10_240, 320),
    # QSA projects both query and its sigmoid output gate (24*256*2).
    "mtp.layers.0.self_attn.q_proj.weight": (12_288, 2_560),
    "mtp.layers.0.self_attn.k_proj.weight": (512, 2_560),
    "mtp.layers.0.self_attn.v_proj.weight": (512, 2_560),
    "mtp.layers.0.self_attn.o_proj.weight": (2_560, 6_144),
    "mtp.layers.0.self_attn.q_norm.weight": (256,),
    "mtp.layers.0.self_attn.k_norm.weight": (256,),
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": (640, 2_560),
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": (128,),
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": (128,),
    "mtp.layers.0.mlp.gate.weight": (512, 2_560),
    "mtp.layers.0.mlp.experts.gate_up_proj": (512, 1_280, 2_560),
    "mtp.layers.0.mlp.experts.down_proj": (512, 2_560, 640),
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (640, 2_560),
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": (640, 2_560),
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": (2_560, 640),
    "mtp.layers.0.mlp.shared_expert_gate.weight": (1, 2_560),
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": (4, 10_240),
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": (10_240,),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": (320, 10_240),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": (10_240, 320),
}


def validate_mtp_config(args: Any) -> None:
    """Require the official one-layer hybrid QSA+MoE MTP contract."""

    exact = {
        "model_type": "qwen4_exp_text",
        "hidden_size": OFFICIAL_HIDDEN_SIZE,
        "hc_count": OFFICIAL_HC_COUNT,
        "hc_lowrank": 320,
        "mtp_num_hidden_layers": OFFICIAL_MTP_DEPTH,
    }
    errors = [
        f"{name}={_value(args, name)!r} (expected {expected!r})"
        for name, expected in exact.items()
        if _value(args, name) != expected
    ]
    mtp = _value(args, "mtp", {}) or {}
    if not (
        _value(mtp, "hybrid") is True
        and list(_value(mtp, "layer_types", [])) == ["full_attention"]
        and int(_value(mtp, "num_hidden_layers", 0) or 0) == 1
    ):
        errors.append("mtp must be hybrid depth 1 with layer_types=['full_attention']")
    if errors:
        raise Qwen4ExpMTPContractError(
            "Unsupported Qwen3.8 Flash Next MTP config: " + "; ".join(errors)
        )


def validate_mtp_weights(weights: Mapping[str, Any]) -> None:
    """Validate the published head or its lossless MLX MoE representation.

    Raw HF weights contain two packed expert tensors.  Fusion's converter
    losslessly splits those into three SwitchGLU tensors, and quantized MLX
    artifacts may add ``.scales`` / ``.biases`` beside ordinary ``.weight``
    tensors.  Both layouts represent the same graph and must remain loadable.
    """

    mtp_keys = {key for key in weights if key.startswith("mtp.")}
    raw_experts = {
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.experts.down_proj",
    }
    switch_experts = {
        "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
        "mtp.layers.0.mlp.switch_mlp.up_proj.weight",
        "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
    }
    expected_keys = set(_REQUIRED_MTP_SHAPES)
    base_required = expected_keys - raw_experts
    uses_raw = raw_experts <= mtp_keys
    uses_switch = switch_experts <= mtp_keys
    if uses_raw == uses_switch:
        raise Qwen4ExpMTPContractError(
            "Qwen4-Exp MTP must contain exactly one routed-expert layout: "
            "official packed experts or sanitized SwitchGLU experts"
        )
    required = base_required | (raw_experts if uses_raw else switch_experts)

    allowed = set(required)
    for key in tuple(required):
        if key.endswith(".weight"):
            stem = key[: -len(".weight")]
            allowed.update((f"{stem}.scales", f"{stem}.biases"))
    missing = sorted(required - mtp_keys)
    unexpected = sorted(mtp_keys - allowed)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise Qwen4ExpMTPContractError(
            "Qwen4-Exp MTP checkpoint contract mismatch: " + "; ".join(details)
        )

    # Exact shape checking applies to published unquantized tensors.  MLX
    # quantized weights have packed dimensions plus side metadata and are
    # validated by the quantized module loader instead.
    has_quant_metadata = any(key.endswith((".scales", ".biases")) for key in mtp_keys)
    bad_shapes = []
    shape_contract = (
        _REQUIRED_MTP_SHAPES
        if uses_raw
        else {
            key: shape
            for key, shape in _REQUIRED_MTP_SHAPES.items()
            if key not in raw_experts
        }
    )
    if uses_switch:
        shape_contract = {
            **shape_contract,
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight": (512, 640, 2_560),
            "mtp.layers.0.mlp.switch_mlp.up_proj.weight": (512, 640, 2_560),
            "mtp.layers.0.mlp.switch_mlp.down_proj.weight": (512, 2_560, 640),
        }
    for key, expected in shape_contract.items():
        actual = _shape(weights[key])
        # Streaming quantizers sometimes expose key-only placeholders.  Shape
        # validation is deferred for those placeholders, but never relaxed for
        # real tensors.
        if not has_quant_metadata and actual is not None and actual != expected:
            bad_shapes.append(f"{key}={actual}, expected {expected}")
    if bad_shapes:
        raise Qwen4ExpMTPContractError(
            "Qwen4-Exp MTP tensor shape mismatch: " + "; ".join(bad_shapes)
        )


def _require_module_contract(q4: Any) -> None:
    required = (
        "TextModel",
        "build_mtp_decoder_layer",
        "create_attention_mask",
        "make_qsa_cache",
    )
    missing = [name for name in required if not hasattr(q4, name)]
    if missing:
        raise Qwen4ExpMTPContractError(
            "mlx_lm.models.qwen4_exp is missing the native MTP integration API: "
            + ", ".join(missing)
        )

    signature = inspect.signature(q4.TextModel.__call__)
    if "return_hidden" not in signature.parameters:
        raise Qwen4ExpMTPContractError(
            "qwen4_exp.TextModel.__call__ must accept return_hidden and return "
            "the pre-mixer 10,240-wide state for MTP verification"
        )


def _register_mtp_module(q4: Any) -> None:
    if hasattr(q4, "Qwen4ExpMTPModule"):
        return

    import mlx.core as mx
    import mlx.nn as nn

    from ..qwen4_exp.hc import Qwen4ExpHyperConnectionMixer

    class ZeroCenteredRMSNorm(nn.Module):
        """Qwen4-Exp RMSNorm: checkpoint weights are zero-centered."""

        def __init__(self, dimensions: int, eps: float):
            super().__init__()
            self.weight = mx.zeros((dimensions,))
            self.eps = eps

        def __call__(self, hidden):
            input_dtype = hidden.dtype
            hidden_f32 = hidden.astype(mx.float32)
            variance = mx.mean(mx.square(hidden_f32), axis=-1, keepdims=True)
            normalized = hidden_f32 * mx.rsqrt(variance + self.eps)
            return (normalized * (1.0 + self.weight)).astype(input_dtype)

    class Qwen4ExpMTPModule(nn.Module):
        """Official depth-1 QSA+MoE prediction head."""

        def __init__(self, args: Any):
            super().__init__()
            validate_mtp_config(args)
            eps = float(_value(args, "rms_norm_eps", 1e-6))
            self.pre_fc_norm_embedding = ZeroCenteredRMSNorm(2_560, eps)
            self.pre_fc_norm_hidden = ZeroCenteredRMSNorm(10_240, eps)
            self.fc_embedding = nn.Linear(2_560, 2_560, bias=False)
            self.fc_hidden = nn.Linear(2_560, 2_560, bias=False)
            self.hyper_connection_mixer = Qwen4ExpHyperConnectionMixer(args)
            layer = q4.build_mtp_decoder_layer(args)
            if getattr(layer, "layer_type", None) != "full_attention":
                raise Qwen4ExpMTPContractError(
                    "build_mtp_decoder_layer must return a full_attention layer"
                )
            if not bool(getattr(layer, "_omlx_qwen4_exp_mtp_safe", False)):
                raise Qwen4ExpMTPContractError(
                    "build_mtp_decoder_layer must mark the exact QSA+MoE layer "
                    "with _omlx_qwen4_exp_mtp_safe=True"
                )
            self.layers = [layer]

        def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
            if hidden_states.shape[-1] != OFFICIAL_HYPER_SIZE:
                raise Qwen4ExpMTPContractError(
                    "Qwen4-Exp MTP requires the pre-mixer 10,240-wide backbone "
                    f"state, got {hidden_states.shape[-1]}"
                )
            embedding = self.pre_fc_norm_embedding(embed_tokens(next_token_ids))
            hidden = self.pre_fc_norm_hidden(hidden_states)
            hidden = self.hyper_connection_mixer(hidden)
            fused = self.fc_embedding(embedding) + self.fc_hidden(hidden)
            hyper = mx.concatenate([fused] * OFFICIAL_HC_COUNT, axis=-1)

            if cache is None:
                cache = [None]
            if len(cache) != OFFICIAL_MTP_DEPTH:
                raise Qwen4ExpMTPContractError(
                    f"Qwen4-Exp MTP cache must contain one layer, got {len(cache)}"
                )
            mask = q4.create_attention_mask(fused, cache[0])
            hyper = self.layers[0](hyper, mask=mask, cache=cache[0])
            return self.hyper_connection_mixer(hyper)

    Qwen4ExpMTPModule.__name__ = "Qwen4ExpMTPModule"
    Qwen4ExpMTPModule.__qualname__ = "Qwen4ExpMTPModule"
    Qwen4ExpMTPModule.__module__ = q4.__name__
    q4.Qwen4ExpMTPModule = Qwen4ExpMTPModule


def _patch_text_model(q4: Any) -> None:
    cls = q4.TextModel
    if getattr(cls, "_omlx_qwen4_exp_mtp_patched", False):
        return
    if hasattr(cls, "mtp_forward"):
        raise Qwen4ExpMTPContractError(
            "qwen4_exp.TextModel already has an unrecognized mtp_forward; "
            "refusing to combine incompatible MTP implementations"
        )

    original_init = cls.__init__
    original_sanitize = cls.sanitize

    def __init__(self, args: Any):  # noqa: N807 - runtime __init__ wrapper
        original_init(self, args)
        validate_mtp_config(args)
        from . import is_mtp_active

        active = bool(is_mtp_active())
        self._omlx_mtp_decode_enabled = active
        self._omlx_mtp_chain = False
        self._omlx_mtp_depth = OFFICIAL_MTP_DEPTH
        if active:
            self.mtp = q4.Qwen4ExpMTPModule(args)

    def sanitize(self, weights):
        root_mtp = {key for key in weights if key.startswith("mtp.")}
        outer_mtp = {key for key in weights if key.startswith("language_model.mtp.")}
        if root_mtp and outer_mtp:
            raise Qwen4ExpMTPContractError(
                "Qwen4-Exp MTP weights mix root and language_model prefixes"
            )
        mtp_keys = root_mtp or outer_mtp
        prefix = "language_model." if outer_mtp else ""
        mtp_weights = {
            key.removeprefix(prefix): value
            for key, value in weights.items()
            if key in mtp_keys
        }
        backbone_weights = {
            key: value for key, value in weights.items() if key not in mtp_keys
        }
        sanitized = dict(original_sanitize(self, backbone_weights))
        if not hasattr(self, "mtp"):
            return sanitized
        if not mtp_weights:
            raise Qwen4ExpMTPContractError(
                "Lightning MTP is enabled but this Qwen3.8 Flash Next artifact "
                "contains no root mtp.* tensors"
            )
        validate_mtp_weights(mtp_weights)
        from ..qwen4_exp.moe import sanitize_moe_weights

        mtp_weights = sanitize_moe_weights(mtp_weights)
        if prefix:
            mtp_weights = {prefix + key: value for key, value in mtp_weights.items()}
        overlap = set(sanitized).intersection(mtp_weights)
        if overlap:
            raise Qwen4ExpMTPContractError(
                "MTP sanitize collided with backbone keys: "
                + ", ".join(sorted(overlap))
            )
        sanitized.update(mtp_weights)
        return sanitized

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        if not hasattr(self, "mtp"):
            raise Qwen4ExpMTPContractError(
                "Qwen4-Exp MTP was not enabled when this model was loaded"
            )
        backbone = getattr(self, "model", self)
        embed_tokens = getattr(backbone, "embed_tokens", None)
        if not callable(embed_tokens):
            raise Qwen4ExpMTPContractError(
                "qwen4_exp.TextModel must expose model.embed_tokens"
            )
        head_hidden = self.mtp(hidden_states, next_token_ids, embed_tokens, mtp_cache)
        logits_source = head_hidden
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:, :]
        if bool(_value(self.args, "tie_word_embeddings", False)):
            logits = embed_tokens.as_linear(logits_source)
        else:
            logits = self.lm_head(logits_source)
        return (logits, head_hidden) if return_hidden else logits

    def make_mtp_cache(self):
        return [q4.make_qsa_cache()] if hasattr(self, "mtp") else []

    cls.__init__ = __init__
    cls.sanitize = sanitize
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls._omlx_qwen4_exp_mtp_patched = True


def apply() -> bool:
    """Install the native Qwen4-Exp MTP binding, idempotently.

    Returns ``False`` only when the Qwen4-Exp module itself is not importable.
    A present but incomplete module raises :class:`Qwen4ExpMTPContractError`
    so a bad architecture cannot be mistaken for an unsupported optional
    model.
    """

    try:
        from mlx_lm.models import qwen4_exp as q4
    except ImportError:
        logger.debug("mlx_lm.models.qwen4_exp not importable; MTP patch skipped")
        return False

    _require_module_contract(q4)
    _register_mtp_module(q4)
    _patch_text_model(q4)
    return True


__all__ = [
    "OFFICIAL_MTP_DEPTH",
    "Qwen4ExpMTPContractError",
    "apply",
    "validate_mtp_config",
    "validate_mtp_weights",
]
