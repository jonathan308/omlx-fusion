# SPDX-License-Identifier: Apache-2.0
"""Exact GLM-5.3-Flash layer-45 MTP checkpoint/cache boundary.

This is intentionally not an alias of the older ``glm_moe_dsa`` MTP patch.
GLM5-Next has 45 trunk layers and exactly one separately trained DSA+MoE head
at source layer 45.  The head owns a latent-KV cache and an indexer cache.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, Final

OFFICIAL_MAIN_LAYER_COUNT: Final = 45
OFFICIAL_MTP_LAYER: Final = 45
OFFICIAL_MTP_DEPTH: Final = 1
MTP_CACHE_SLOTS_PER_LAYER: Final = 2
MTP_SOURCE_PREFIX: Final = "model.language_model.layers.45."
MTP_TARGET_PREFIX: Final = "mtp.0."
GLM5_NEXT_MTP_RUNTIME_READY: Final = True


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _text_config(config: Any) -> Any:
    nested = _get(config, "text_config")
    return config if nested is None else nested


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def validate_mtp_config(config: Any) -> None:
    config = _text_config(config)
    if _get(config, "num_hidden_layers") != OFFICIAL_MAIN_LAYER_COUNT:
        raise ValueError("GLM5-Next MTP requires exactly 45 trunk layers")
    if _get(config, "num_nextn_predict_layers") != OFFICIAL_MTP_DEPTH:
        raise ValueError("GLM5-Next ships exactly one layer-45 MTP head")
    if _get(config, "hidden_size") != 4_096:
        raise ValueError("GLM5-Next MTP hidden_size must be 4096")
    if _get(config, "index_share_for_mtp_iteration") is not True:
        raise ValueError("GLM5-Next MTP index sharing contract changed")


def validate_mtp_weight_layout(
    weights: Mapping[str, Any], *, prefix: str = MTP_SOURCE_PREFIX
) -> None:
    """Validate the layer-45 fusion head and its independent DSA boundary."""

    expected = {
        "eh_proj.weight": (4_096, 8_192),
        "enorm.weight": (4_096,),
        "hnorm.weight": (4_096,),
        "shared_head.norm.weight": (4_096,),
        "input_layernorm.weight": (4_096,),
        "post_attention_layernorm.weight": (4_096,),
        "self_attn.q_a_layernorm.weight": (1_536,),
        "self_attn.kv_a_layernorm.weight": (512,),
        "self_attn.indexer.wk.weight": (128, 4_096),
        "self_attn.indexer.weights_proj.weight": (32, 4_096),
        "self_attn.indexer.wq_b.weight": (4_096, 1_536),
        "self_attn.indexer.k_norm.weight": (128,),
        "self_attn.indexer.k_norm.bias": (128,),
        "self_attn.indexer.index_kpool_compress_ape": (4, 128),
        "self_attn.indexer.index_kpool_compress_gate": (128, 4_096),
    }
    for suffix, wanted in expected.items():
        key = prefix + suffix
        if key not in weights:
            raise ValueError(f"GLM5-Next MTP is missing checkpoint tensor: {key}")
        actual = _shape(weights[key])
        if actual != wanted:
            raise ValueError(
                f"Invalid GLM5-Next MTP tensor {key}: found {actual}, expected {wanted}"
            )


def sanitize_mtp_weights(weights: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve and remap raw source layer 45 under the dedicated MTP tree.

    All non-MTP tensors pass through unchanged.  A partial layer-45 family is
    rejected rather than silently discarded.  The function also accepts an
    already-remapped artifact and is then idempotent.
    """

    sanitized = dict(weights)
    raw = [key for key in sanitized if key.startswith(MTP_SOURCE_PREFIX)]
    mapped = [key for key in sanitized if key.startswith(MTP_TARGET_PREFIX)]
    if raw and mapped:
        raise ValueError(
            "GLM5-Next checkpoint contains both raw and remapped MTP weights"
        )
    if not raw:
        if mapped:
            _validate_remapped_mtp_weights(sanitized)
        return sanitized

    validate_mtp_weight_layout(sanitized)
    special = {
        "eh_proj.weight": "eh_proj.weight",
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "shared_head.norm.weight": "norm.weight",
    }
    for key in sorted(raw):
        value = sanitized.pop(key)
        suffix = key[len(MTP_SOURCE_PREFIX) :]
        target = special.get(suffix)
        if target is None:
            target = "block." + suffix
        sanitized[MTP_TARGET_PREFIX + target] = value
    return sanitized


def _validate_remapped_mtp_weights(weights: Mapping[str, Any]) -> None:
    expected = {
        "mtp.0.enorm.weight": (4_096,),
        "mtp.0.hnorm.weight": (4_096,),
        "mtp.0.norm.weight": (4_096,),
        "mtp.0.block.self_attn.indexer.wk.weight": (128, 4_096),
        "mtp.0.block.self_attn.indexer.weights_proj.weight": (32, 4_096),
    }
    for key, wanted in expected.items():
        if key not in weights:
            raise ValueError(f"GLM5-Next remapped MTP is missing tensor: {key}")
        if _shape(weights[key]) != wanted:
            raise ValueError(
                f"Invalid GLM5-Next remapped MTP tensor {key}: "
                f"found {_shape(weights[key])}, expected {wanted}"
            )
    eh_weight = "mtp.0.eh_proj.weight"
    if eh_weight not in weights:
        raise ValueError(f"GLM5-Next remapped MTP is missing tensor: {eh_weight}")
    if "mtp.0.eh_proj.scales" in weights:
        scales = _shape(weights["mtp.0.eh_proj.scales"])
        packed = _shape(weights[eh_weight])
        biases = _shape(weights.get("mtp.0.eh_proj.biases"))
        if (
            len(packed) != 2
            or packed[0] != 4_096
            or len(scales) != 2
            or scales[0] != 4_096
            or biases != scales
            or 8_192 % scales[-1]
        ):
            raise ValueError("Invalid GLM5-Next affine MTP eh_proj geometry")
        bits = 32 * packed[-1] // 8_192
        if bits not in (4, 8):
            raise ValueError("GLM5-Next converted MTP eh_proj must be affine Q4/Q8")
    elif _shape(weights[eh_weight]) != (4_096, 8_192):
        raise ValueError("Invalid GLM5-Next remapped MTP tensor mtp.0.eh_proj.weight")


def partition_mtp_cache(
    cache: Sequence[Any] | None, *, depth: int = OFFICIAL_MTP_DEPTH
) -> tuple[tuple[Any, Any], ...]:
    """Validate and partition the exact two-slot layer-45 cache boundary."""

    if depth != OFFICIAL_MTP_DEPTH:
        raise ValueError("GLM5-Next MTP cache depth must remain exactly 1")
    if cache is None:
        return ((None, None),)
    state = getattr(cache, "state", None)
    if state is not None and not isinstance(cache, Sequence):
        if len(state) != MTP_CACHE_SLOTS_PER_LAYER:
            raise ValueError("GLM5-Next MTP DSA cache must expose two arrays")
        return ((state[0], state[1]),)
    if len(cache) != MTP_CACHE_SLOTS_PER_LAYER:
        raise ValueError(
            "GLM5-Next MTP cache requires exactly two slots "
            "(latent KV and DSA indexer KV)"
        )
    return ((cache[0], cache[1]),)


split_mtp_cache = partition_mtp_cache


def make_mtp_cache(config: Any = None):
    """Create the layer-45 DSA cache whose state is exactly two arrays."""

    from .dsa import Glm5NextDsaCache

    return Glm5NextDsaCache(config)


def _unwrap_mtp_cache(cache: Any):
    from .dsa import Glm5NextDsaCache

    if isinstance(cache, Glm5NextDsaCache):
        return cache
    if (
        isinstance(cache, Sequence)
        and len(cache) == 1
        and isinstance(cache[0], Glm5NextDsaCache)
    ):
        return cache[0]
    raise ValueError("GLM5-Next MTP cache must be one two-array DSA cache")


def mtp_partial_rollback(cache: Any, *, accepted: int, num_drafts: int) -> bool:
    """Trim rejected draft rows from both latent and indexer arrays exactly."""

    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (accepted, num_drafts)
        )
        or accepted > num_drafts
    ):
        raise ValueError("accepted and num_drafts must satisfy 0 <= accepted <= drafts")
    rejected = num_drafts - accepted
    if rejected == 0:
        return True
    target = _unwrap_mtp_cache(cache)
    return target.trim(rejected) == rejected


def _runtime_value(config: Any, name: str, default: Any = None) -> Any:
    config = _text_config(config)
    return _get(config, name, default)


@lru_cache(maxsize=2)
def _mtp_implementation_class(validate_official: bool = True):
    import mlx.core as mx
    import mlx.nn as nn

    from .dsa import Glm5NextDsa, Glm5NextDsaConfig
    from .moe import make_sparse_moe_class

    class _DecoderLayer(nn.Module):
        def __init__(self, config, dsa):
            super().__init__()
            hidden = dsa.hidden_size
            eps = _runtime_value(config, "rms_norm_eps", dsa.rms_norm_eps)
            self.input_layernorm = nn.RMSNorm(hidden, eps=eps)
            self.self_attn = Glm5NextDsa(dsa, layer_idx=OFFICIAL_MTP_LAYER)
            self.post_attention_layernorm = nn.RMSNorm(hidden, eps=eps)
            self.mlp = make_sparse_moe_class(validate_official=validate_official)(
                config
            )

        def __call__(self, hidden, mask, cache):
            hidden = hidden + self.self_attn(
                self.input_layernorm(hidden), mask, cache=cache
            )
            return hidden + self.mlp(self.post_attention_layernorm(hidden))

    class _Glm5NextMTPBlock(nn.Module):
        def __init__(self, config, *, dsa_config=None):
            super().__init__()
            if validate_official:
                validate_mtp_config(config)
            if dsa_config is None:
                config_object = _text_config(config)
                builder = getattr(config_object, "dsa_config", None)
                if callable(builder):
                    dsa_config = builder()
                else:
                    dsa_config = Glm5NextDsaConfig.from_model_config(config_object)
            if not isinstance(dsa_config, Glm5NextDsaConfig):
                raise ValueError("GLM5-Next MTP requires a Glm5NextDsaConfig")
            hidden = dsa_config.hidden_size
            eps = _runtime_value(config, "rms_norm_eps", dsa_config.rms_norm_eps)
            self.enorm = nn.RMSNorm(hidden, eps=eps)
            self.hnorm = nn.RMSNorm(hidden, eps=eps)
            self.eh_proj = nn.Linear(2 * hidden, hidden, bias=False)
            self.norm = nn.RMSNorm(hidden, eps=eps)
            self.block = _DecoderLayer(config, dsa_config)
            self.dsa_config = dsa_config

        def __call__(self, hidden, embedding, mask, cache=None, *, normalize=False):
            fused = self.eh_proj(
                mx.concatenate((self.enorm(embedding), self.hnorm(hidden)), axis=-1)
            )
            output = self.block(fused, mask, cache)
            return self.norm(output) if normalize else output

        def make_cache(self):
            return make_mtp_cache(self.dsa_config)

    _Glm5NextMTPBlock.__name__ = "Glm5NextMTPBlock"
    _Glm5NextMTPBlock.__qualname__ = "Glm5NextMTPBlock"
    _Glm5NextMTPBlock.__module__ = __name__
    return _Glm5NextMTPBlock


def make_mtp_block_class(*, validate_official: bool = True):
    """Return the executable fusion+DSA+MoE head class lazily."""

    return _mtp_implementation_class(validate_official)


class Glm5NextMTPBlock:
    """Lazy official depth-1 layer-45 MTP constructor."""

    def __new__(cls, config: Any):
        return _mtp_implementation_class(True)(config)


__all__ = [
    "MTP_CACHE_SLOTS_PER_LAYER",
    "MTP_SOURCE_PREFIX",
    "MTP_TARGET_PREFIX",
    "OFFICIAL_MAIN_LAYER_COUNT",
    "OFFICIAL_MTP_DEPTH",
    "OFFICIAL_MTP_LAYER",
    "GLM5_NEXT_MTP_RUNTIME_READY",
    "Glm5NextMTPBlock",
    "make_mtp_block_class",
    "make_mtp_cache",
    "mtp_partial_rollback",
    "partition_mtp_cache",
    "sanitize_mtp_weights",
    "split_mtp_cache",
    "validate_mtp_config",
    "validate_mtp_weight_layout",
]
