# SPDX-License-Identifier: Apache-2.0
"""Native mlx-lm text runtime for the pinned GLM5-Next architecture.

The outer checkpoint is multimodal, but this module deliberately exposes only
the text graph.  Image and video keyword arguments are rejected instead of
being ignored.  The module is registered solely as ``mlx_lm.models.glm5_next``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Final

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.pipeline import PipelineMixin

from omlx.patches.glm5_next.dsa import (
    MAIN_DSA_LAYERS,
    Glm5NextDsa,
    Glm5NextDsaCache,
    Glm5NextDsaConfig,
)
from omlx.patches.glm5_next.kda import KDACache, KDAConfig, make_kda_class
from omlx.patches.glm5_next.mhc import (
    MHCConfig,
    apply_mhc_residual,
    make_hyper_head_class,
    make_mhc_class,
)
from omlx.patches.glm5_next.moe import make_sparse_moe_class, sanitize_moe_weights
from omlx.patches.glm5_next.mtp import (
    make_mtp_block_class,
    sanitize_mtp_weights,
    validate_mtp_config,
)
from omlx.patches.glm5_next.mtp import (
    make_mtp_cache as make_glm5_next_mtp_cache,
)
from omlx.patches.glm5_next.mtp import (
    mtp_partial_rollback as rollback_glm5_next_mtp_cache,
)
from omlx.patches.glm5_next.nvfp4 import bind_glm5_next_nvfp4
from omlx.patches.glm5_next.vision import (
    VISION_PREFIX,
    reject_unsupported_media,
    sanitize_vision_weights,
    validate_vision_config,
)

GLM5_NEXT_STRICT_GRAPH = True
MAIN_LAYER_COUNT: Final = 45
KDA_LAYERS: Final = tuple(
    index for index in range(MAIN_LAYER_COUNT) if index not in MAIN_DSA_LAYERS
)
DSA_LAYERS: Final = tuple(MAIN_DSA_LAYERS)

logger = logging.getLogger(__name__)
_TRACED_LENGTHS: set = set()


class Glm5NextModelContractError(ValueError):
    """The supplied configuration/artifact cannot execute as GLM5-Next."""


class Glm5NextRuntimeUnavailableError(RuntimeError):
    """The graph is registered, but one or more exact runtime seams are absent."""


def runtime_gaps() -> tuple[str, ...]:
    """Return unresolved primitive capabilities without allocating a model.

    These feature flags are intentionally affirmative.  A similarly named
    class/function is not enough to claim correctness for a 320B checkpoint.
    """

    from omlx.patches.glm5_next import dsa, moe, mtp

    checks = (
        (
            "loadable DSA nn.Module and full mlx-lm cache ABI",
            getattr(dsa, "GLM5_NEXT_DSA_MODULE_READY", False),
        ),
        (
            "block-FP8 expert execution with weight_scale_inv sidecars",
            getattr(moe, "GLM5_NEXT_BLOCK_FP8_RUNTIME_READY", False),
        ),
        (
            "executable layer-45 depth-1 MTP head",
            getattr(mtp, "GLM5_NEXT_MTP_RUNTIME_READY", False),
        ),
    )
    return tuple(description for description, ready in checks if ready is not True)


def require_runtime_ready() -> None:
    gaps = runtime_gaps()
    if gaps:
        raise Glm5NextRuntimeUnavailableError(
            "GLM5-Next strict graph is registered but live execution remains "
            "disabled: " + "; ".join(gaps)
        )


def native_vision_ready() -> bool:
    """Use the vision module's affirmative flag and structural gap helper."""

    from omlx.patches.glm5_next import vision

    return (
        vision.GLM5_NEXT_VISION_RUNTIME_READY is True
        and not vision.vision_runtime_gaps()
    )


GLM5_NEXT_NATIVE_TEXT_READY: Final = not runtime_gaps()


def _known(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


def _official_layer_types() -> list[str]:
    return [
        "deepseek_sparse_attention" if index in DSA_LAYERS else "linear_attention"
        for index in range(MAIN_LAYER_COUNT)
    ]


@dataclass
class TextModelArgs:
    model_type: str = "glm5_next_text"
    vocab_size: int = 154_880
    hidden_size: int = 4_096
    intermediate_size: int = 12_288
    num_hidden_layers: int = MAIN_LAYER_COUNT
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    max_position_embeddings: int = 1_048_576
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    first_k_dense_replace: int = 3
    moe_intermediate_size: int = 2_048
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.5
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    moe_router_dtype: str = "float32"
    n_group: int = 1
    topk_group: int = 1
    swiglu_limit: float = 10.0

    mhc: bool = True
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20

    mla_use_nope: bool = True
    q_lora_rank: int = 1_536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2_048
    index_kpool: int = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True
    index_share_for_mtp_iteration: bool = True
    num_nextn_predict_layers: int = 1

    layer_types: list[str] = field(default_factory=_official_layer_types)
    mlp_layer_types: list[str] = field(
        default_factory=lambda: ["dense"] * 3 + ["sparse"] * 42
    )
    linear_attn_config: dict[str, Any] = field(default_factory=dict)
    quantization: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TextModelArgs:
        if not isinstance(values, Mapping):
            raise Glm5NextModelContractError("glm5_next text_config must be an object")
        instance = cls(**_known(cls, values))
        instance.validate_architecture()
        return instance

    def validate_architecture(self) -> None:
        exact = {
            "model_type": "glm5_next_text",
            "vocab_size": 154_880,
            "hidden_size": 4_096,
            "intermediate_size": 12_288,
            "num_hidden_layers": MAIN_LAYER_COUNT,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "first_k_dense_replace": 3,
            "moe_intermediate_size": 2_048,
            "n_routed_experts": 288,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "moe_router_dtype": "float32",
            "n_group": 1,
            "topk_group": 1,
            "mhc": True,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "mla_use_nope": True,
            "q_lora_rank": 1_536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 2_048,
            "index_kpool": 4,
            "index_kpool_compress": True,
            "index_kpool_always_select_tail": True,
            "index_share_for_mtp_iteration": True,
            "num_nextn_predict_layers": 1,
        }
        errors = [
            f"{name}={getattr(self, name)!r} (expected {wanted!r})"
            for name, wanted in exact.items()
            if getattr(self, name) != wanted
        ]
        if self.layer_types != _official_layer_types():
            errors.append("layer_types must contain exactly 34 KDA and 11 DSA layers")
        if self.mlp_layer_types != ["dense"] * 3 + ["sparse"] * 42:
            errors.append("mlp_layer_types must contain 3 dense then 42 sparse layers")
        linear = self.linear_attn_config
        expected_linear = {
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": list(KDA_LAYERS),
            "full_attn_layers": list(DSA_LAYERS),
        }
        if not isinstance(linear, Mapping) or any(
            linear.get(key) != value for key, value in expected_linear.items()
        ):
            errors.append(
                "linear_attn_config does not match the zero-based KDA schedule"
            )
        if errors:
            raise Glm5NextModelContractError(
                "Unsupported GLM5-Next text architecture: " + "; ".join(errors)
            )

    def kda_config(self) -> KDAConfig:
        return KDAConfig(
            hidden_size=self.hidden_size,
            num_heads=64,
            head_dim=128,
            conv_kernel_size=4,
            gate_lower_bound=-5.0,
            rms_norm_eps=self.rms_norm_eps,
            hidden_act=self.hidden_act,
        )

    def mhc_config(self) -> MHCConfig:
        return MHCConfig(
            hidden_size=self.hidden_size,
            streams=self.hc_mult,
            eps=self.hc_eps,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            rms_norm_eps=self.rms_norm_eps,
        )

    def dsa_config(self) -> Glm5NextDsaConfig:
        return Glm5NextDsaConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            index_kpool=self.index_kpool,
            index_kpool_compress=self.index_kpool_compress,
            index_kpool_always_select_tail=self.index_kpool_always_select_tail,
            rms_norm_eps=self.rms_norm_eps,
        )


@dataclass
class ModelArgs:
    model_type: str
    text_config: dict[str, Any]
    vision_config: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    quantization: dict[str, Any] | None = None
    quantization_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ModelArgs:
        if values.get("model_type") != "glm5_next":
            raise Glm5NextModelContractError("model_type must be 'glm5_next'")
        text = values.get("text_config")
        TextModelArgs.from_dict(text)
        if bool(values.get("tie_word_embeddings", False)):
            raise Glm5NextModelContractError("GLM5-Next has an independent final head")
        vision = values.get("vision_config")
        if vision is not None:
            if not isinstance(vision, Mapping):
                raise Glm5NextModelContractError("vision_config must be an object")
            validate_vision_config(vision)
        return cls(
            model_type="glm5_next",
            text_config=dict(text),
            vision_config=dict(vision) if isinstance(vision, Mapping) else None,
            tie_word_embeddings=False,
            quantization=(
                dict(values["quantization"])
                if isinstance(values.get("quantization"), Mapping)
                else None
            ),
            quantization_config=(
                dict(values["quantization_config"])
                if isinstance(values.get("quantization_config"), Mapping)
                else None
            ),
        )


def layer_kind(index: int) -> str:
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < MAIN_LAYER_COUNT
    ):
        raise Glm5NextModelContractError("main layer index must be in 0..44")
    return "dsa" if index in DSA_LAYERS else "kda"


class Glm5NextKDACache(KDACache):
    """KDA cache with the batch operations expected by mlx-lm schedulers."""

    def prepare(
        self,
        *,
        lengths=None,
        right_padding=None,
        left_padding=None,
        **_kwargs,
    ) -> None:
        del right_padding
        self.lengths = None if lengths is None else mx.array(lengths, dtype=mx.int32)
        self.left_padding = (
            None if left_padding is None else mx.array(left_padding, dtype=mx.int32)
        )

    def filter(self, batch_indices) -> None:
        self.cache = [
            None if value is None else mx.take(value, batch_indices, axis=0)
            for value in self.cache
        ]
        if self.lengths is not None:
            self.lengths = mx.take(self.lengths, batch_indices, axis=0)
        if self.left_padding is not None:
            self.left_padding = mx.take(self.left_padding, batch_indices, axis=0)

    def extract(self, index: int) -> Glm5NextKDACache:
        result = type(self)()
        result.cache = [
            None if value is None else value[index : index + 1] for value in self.cache
        ]
        result.offset = self.offset
        if self.lengths is not None:
            result.lengths = self.lengths[index : index + 1]
        if self.left_padding is not None:
            result.left_padding = self.left_padding[index : index + 1]
        return result

    def extend(self, other: Glm5NextKDACache) -> None:
        if not isinstance(other, Glm5NextKDACache):
            raise ValueError("can only extend with a GLM5-Next KDA cache")
        self.cache = [
            right
            if left is None
            else left
            if right is None
            else mx.concatenate((left, right), axis=0)
            for left, right in zip(self.cache, other.cache)
        ]
        # Recurrent and short-convolution states already summarize each row's
        # complete history, so unlike KV caches they need no sequence padding.
        self.offset = max(self.offset, other.offset)
        self.lengths = None
        self.left_padding = None

    @classmethod
    def merge(cls, caches: Sequence[Glm5NextKDACache]) -> Glm5NextKDACache:
        if not caches:
            return cls()
        result = cls()
        result.cache = list(caches[0].cache)
        result.offset = caches[0].offset
        for cache in caches[1:]:
            result.extend(cache)
        return result

    def finalize(self) -> None:
        # Prompt validity is consumed by KDA during prefill. Every subsequent
        # decode token is real even when merged rows had different histories.
        self.lengths = None
        self.left_padding = None


def make_layer_cache(index: int, dsa_config: Glm5NextDsaConfig | None = None):
    if layer_kind(index) == "dsa":
        return Glm5NextDsaCache(dsa_config or Glm5NextDsaConfig.official())
    return Glm5NextKDACache()


class Glm5NextDenseMLP(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.limit = args.swiglu_limit

    def __call__(self, hidden):
        gate = mx.minimum(
            self.gate_proj(hidden), mx.array(self.limit, dtype=hidden.dtype)
        )
        up = mx.clip(self.up_proj(hidden), -self.limit, self.limit)
        return self.down_proj(nn.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = layer_kind(layer_idx) == "kda"
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.self_attn = (
            make_kda_class()(args.kda_config())
            if self.is_linear
            else Glm5NextDsa(args.dsa_config(), layer_idx=layer_idx)
        )
        self.mlp = (
            Glm5NextDenseMLP(args)
            if layer_idx < args.first_k_dense_replace
            else make_sparse_moe_class()(args)
        )
        mhc_class = make_mhc_class()
        self.hc_attn = mhc_class(args.mhc_config())
        self.hc_ffn = mhc_class(args.mhc_config())

    def __call__(self, streams, mask, cache=None):
        post, comb, collapsed = _profiled("mhc.attn", self.hc_attn, streams)
        branch_input = self.input_layernorm(collapsed)
        kind = "kda" if self.is_linear else "dsa"
        if self.is_linear:
            branch = _profiled(
                "attn.kda", self.self_attn, branch_input, mask=mask, cache=cache
            )
        else:
            branch = _profiled(
                "attn.dsa", self.self_attn, branch_input, mask, cache=cache
            )
        streams = apply_mhc_residual(post, comb, branch, streams)
        post, comb, collapsed = _profiled("mhc.ffn", self.hc_ffn, streams)
        branch = _profiled(
            f"mlp.{kind}", self.mlp, self.post_attention_layernorm(collapsed)
        )
        return apply_mhc_residual(post, comb, branch, streams)


_PROFILE_ENABLED: Final = os.environ.get("GLM5_NEXT_PROFILE") == "1"
_PROFILE_LOCK = threading.Lock()
_PROFILE_STATS: dict[str, list[float]] = {}
_PROFILE_CALLS = [0]


def _profile_record(section: str, elapsed: float) -> None:
    with _PROFILE_LOCK:
        bucket = _PROFILE_STATS.setdefault(section, [])
        bucket.append(elapsed)
        _PROFILE_CALLS[0] += 1
        calls = _PROFILE_CALLS[0]
    if calls % 200 == 0:
        _profile_report()


def _profile_report() -> None:
    with _PROFILE_LOCK:
        snapshot = {key: list(value) for key, value in _PROFILE_STATS.items()}
        _PROFILE_STATS.clear()
    parts = []
    for key in sorted(snapshot):
        values = snapshot[key]
        parts.append(f"{key}={1000 * sum(values) / len(values):.2f}ms x{len(values)}")
    logger.info("GLM5_NEXT_PROFILE %s", " ".join(parts))


def _profiled(section, fn, *args, **kwargs):
    if not _PROFILE_ENABLED:
        return fn(*args, **kwargs)
    from mlx.utils import tree_flatten

    start = time.perf_counter()
    result = fn(*args, **kwargs)
    built = time.perf_counter()
    mx.eval(*[value for _path, value in tree_flatten(result)])
    end = time.perf_counter()
    _profile_record(section, end - start)
    _profile_record(section + ".build", built - start)
    return result


Glm5NextMTPBlock = make_mtp_block_class()


class Glm5NextTextBackbone(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, index) for index in range(MAIN_LAYER_COUNT)]
        self.hyper_head = make_hyper_head_class()()
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self, inputs, cache=None, input_embeddings=None, *, return_hidden=False
    ):
        hidden = (
            self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        )
        hidden = mx.repeat(hidden[..., None, :], self.args.hc_mult, axis=2)
        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        if len(cache) != len(self.pipeline_layers):
            raise ValueError("GLM5-Next cache must contain exactly 45 layer entries")
        for layer, layer_cache in zip(self.pipeline_layers, cache):
            batch, length = hidden.shape[:2]
            if length not in _TRACED_LENGTHS:
                _TRACED_LENGTHS.add(length)
                logger.info(
                    "GLM5-Next backbone call: batch=%d length=%d cache0=%s",
                    batch,
                    length,
                    type(cache[0]).__name__ if cache else None,
                )
            if isinstance(layer_cache, Glm5NextDsaCache):
                # The final DSA module derives prepared batch validity from its
                # own cache. Passing a KDA-style mask here would discard its
                # right-padding/finalize semantics.
                mask = None
            elif isinstance(layer_cache, KDACache):
                # A None mask means fully-valid rows; forcing an all-ones mask
                # here would defeat the mask-free compiled decode fast path.
                mask = layer_cache.make_mask(length)
            else:
                mask = mx.ones((batch, length), dtype=mx.bool_)
            kind = "kda" if layer.is_linear else "dsa"
            hidden = _profiled(f"layer.{kind}", layer, hidden, mask, cache=layer_cache)
        hidden = _profiled("head", lambda h: self.norm(self.hyper_head(h)), hidden)
        return hidden


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Glm5NextTextBackbone(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.mtp = [Glm5NextMTPBlock(args, dsa_config=args.dsa_config())]

    @property
    def layers(self):
        return self.model.layers

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden=False,
        **media,
    ):
        reject_unsupported_media(media)
        if _PROFILE_ENABLED:
            start = time.perf_counter()
            hidden = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
            logits = self.lm_head(hidden)
            # Wall time here is pure Python graph construction; evaluation
            # happens later at the sampler.  No barrier, no distortion.
            _profile_record("forward.build", time.perf_counter() - start)
        else:
            hidden = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
            logits = self.lm_head(hidden)
        return (logits, hidden) if return_hidden else logits

    def make_cache(self):
        dsa_config = self.args.dsa_config()
        return [
            make_layer_cache(index, dsa_config) for index in range(MAIN_LAYER_COUNT)
        ]

    def prepare_dsa_kv_projections(self) -> int:
        """Materialize the 11 main-layer latent K/V decompositions."""

        prepared = 0
        for layer in self.model.layers:
            if isinstance(layer.self_attn, Glm5NextDsa):
                layer.self_attn.prepare_kv_b_projections()
                prepared += 1
        return prepared

    def make_mtp_cache(self):
        return [make_glm5_next_mtp_cache(self.args.dsa_config())]

    def mtp_forward(self, hidden, inputs, cache=None, *, logits_keep=0):
        if cache is None:
            cache = self.make_mtp_cache()
        if len(cache) != 1 or not isinstance(cache[0], Glm5NextDsaCache):
            raise ValueError("GLM5-Next MTP cache must contain one two-array DSA cache")
        embedding = self.model.embed_tokens(inputs)
        head = self.mtp[0]
        output = head(
            hidden,
            embedding,
            None,
            cache[0],
            normalize=True,
        )
        if logits_keep:
            output = output[:, -int(logits_keep) :]
        return self.lm_head(output), output

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        return rollback_glm5_next_mtp_cache(
            cache, accepted=accepted, num_drafts=num_drafts
        )

    @property
    def quant_predicate(self):
        def predicate(path, _module):
            return not _converter_dense_path(path)

        return predicate

    @property
    def cast_predicate(self):
        return lambda path: not _converter_dense_path(path)


def _converter_dense_path(path: str) -> bool:
    """Mirror convert._must_remain_dense after source-to-runtime remapping."""

    lower = path.lower()
    return (
        ".indexer." in lower
        or ".hc_" in lower
        or ".hc_attn" in lower
        or ".hc_ffn" in lower
        or ".mlp.gate" in lower
        or "shared_expert_gate" in lower
        or "a_log" in lower
        or "dt_bias" in lower
        or "conv1d" in lower
        or "norm" in lower
        or lower.endswith(".bias")
        or "index_kpool" in lower
    )


_MHC_KEY = re.compile(r"^(.*\.layers\.\d+)\.(hc_(?:attn|ffn))_(fn|base|scale)$")


def sanitize_weight_name(name: str) -> str | None:
    """Map one source key to the native text module tree without aliasing GLM-5.2."""

    if name.startswith("model.layers."):
        raise Glm5NextModelContractError("glm_moe_dsa weight aliases are forbidden")
    if name.startswith(VISION_PREFIX):
        return None
    if name.startswith("model.language_model."):
        name = "language_model.model." + name[len("model.language_model.") :]
    elif name.startswith("lm_head.") or name.startswith("mtp."):
        name = "language_model." + name
    match = _MHC_KEY.match(name)
    if match:
        name = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return name


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Fail before embeddings, 45 layers, or any 288-expert bank allocate.
        require_runtime_ready()
        self.args = args
        self.model_type = args.model_type
        text_args = TextModelArgs.from_dict(args.text_config)
        validate_mtp_config(args.text_config)
        self.language_model = TextModel(text_args)
        quantization = args.quantization or args.quantization_config
        self._nvfp4 = bind_glm5_next_nvfp4(self, quantization)
        self._converted_affine = isinstance(quantization, Mapping) and not self._nvfp4

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden=False,
        **media,
    ):
        reject_unsupported_media(media)
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
        )

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def model(self):
        return self.language_model.model

    def make_cache(self):
        return self.language_model.make_cache()

    def prepare_dsa_kv_projections(self) -> int:
        return self.language_model.prepare_dsa_kv_projections()

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def mtp_forward(self, *args, **kwargs):
        return self.language_model.mtp_forward(*args, **kwargs)

    def mtp_partial_rollback(self, *args, **kwargs):
        return self.language_model.mtp_partial_rollback(*args, **kwargs)

    def sanitize(self, weights):
        # A present vision family must be complete, even though the text runtime
        # subsequently drops it.  Partial multimodal checkpoints never degrade
        # silently into a text model.
        sanitized = dict(weights)
        if not self._converted_affine:
            sanitized = sanitize_vision_weights(sanitized)
        # Official source checkpoints store 288 individual block-FP8 experts;
        # conversion packs them. Converted q4/q8 artifacts already contain
        # SwitchGLU ``weight/scales/biases`` triples and must not be mistaken
        # for the source block-FP8 ``weight/weight_scale_inv`` representation.
        if any(".mlp.experts." in key for key in sanitized):
            sanitized = sanitize_moe_weights(sanitized)
        if not self._converted_affine or any(
            key.startswith("model.language_model.layers.45.") for key in sanitized
        ):
            sanitized = sanitize_mtp_weights(sanitized)
        output = {}
        for source, value in sanitized.items():
            target = sanitize_weight_name(source)
            if target is None:
                continue
            if (
                target.endswith("conv1d.weight")
                and getattr(value, "shape", (1,))[-1] != 1
            ):
                value = value.moveaxis(2, 1)
            output[target] = value
        return output

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate


__all__ = [
    "DSA_LAYERS",
    "DecoderLayer",
    "GLM5_NEXT_STRICT_GRAPH",
    "GLM5_NEXT_NATIVE_TEXT_READY",
    "Glm5NextDsaCache",
    "Glm5NextKDACache",
    "Glm5NextMTPBlock",
    "Glm5NextModelContractError",
    "Glm5NextRuntimeUnavailableError",
    "KDA_LAYERS",
    "MAIN_LAYER_COUNT",
    "Model",
    "ModelArgs",
    "TextModel",
    "TextModelArgs",
    "layer_kind",
    "make_layer_cache",
    "native_vision_ready",
    "require_runtime_ready",
    "runtime_gaps",
    "sanitize_weight_name",
]
