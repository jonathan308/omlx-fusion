"""Exact no-RoPE DSA/MLA correctness backend for GLM-5.3-Flash.

This module is derived from ``Glm5NextTextIndexer`` and
``Glm5NextTextAttention`` in the official Transformers implementation merged
as huggingface/transformers commit ``eb4d9e2a64a013bec12289288b85d0b1210ba0aa``.
The checkpoint contract is pinned separately to zai-org/GLM-5.3-Flash revision
``84c6a6aa9497188e15a635ba793b0f95a79b1033``.

The implementation is intentionally a correctness backend.  In particular,
``sparse_mla_attention`` gathers only selected latent rows and never builds a
``[batch, heads, queries, kv_length]`` dense score tensor.  Faster Metal
kernels may replace that function later without changing the cache or weight
ABI below.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import _BaseCache, create_causal_mask, dynamic_roll

OFFICIAL_MODEL_REVISION = "84c6a6aa9497188e15a635ba793b0f95a79b1033"
UPSTREAM_TRANSFORMERS_COMMIT = "eb4d9e2a64a013bec12289288b85d0b1210ba0aa"
UPSTREAM_TRANSFORMERS_SHA256 = (
    "2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b"
)
UPSTREAM_TRANSFORMERS_PATH = "src/transformers/models/glm5_next/modeling_glm5_next.py"

MAIN_DSA_LAYERS = tuple(range(3, 45, 4))
MTP_DSA_LAYER = 45
ALL_DSA_LAYERS = MAIN_DSA_LAYERS + (MTP_DSA_LAYER,)


class Glm5NextDsaContractError(ValueError):
    """The supplied geometry, layer, cache, or weights are not GLM5-Next DSA."""


def _strict_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Glm5NextDsaContractError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class Glm5NextDsaConfig:
    """Shape-only DSA contract.

    Tiny geometries are supported for parity tests, while all architectural
    relationships are fail-closed.  The official checkpoint values are
    available from :meth:`official`.
    """

    hidden_size: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    qk_rope_head_dim: int = 0
    attention_bias: bool = False
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        integer_fields = (
            "hidden_size",
            "num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "v_head_dim",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "index_kpool",
        )
        for name in integer_fields:
            _strict_positive_int(getattr(self, name), name)
        if self.qk_rope_head_dim != 0:
            raise Glm5NextDsaContractError(
                "GLM5-Next DSA requires qk_rope_head_dim == 0"
            )
        if self.attention_bias is not False:
            raise Glm5NextDsaContractError("official DSA projections are bias-free")
        if self.index_kpool_compress is not True:
            raise Glm5NextDsaContractError("index k-pool compression must be enabled")
        if self.index_kpool_always_select_tail is not True:
            raise Glm5NextDsaContractError(
                "the visible incomplete tail must be selected"
            )
        if self.index_topk < self.index_kpool:
            raise Glm5NextDsaContractError("index_topk must be at least index_kpool")
        if self.index_topk % self.index_kpool:
            raise Glm5NextDsaContractError(
                "index_topk must be divisible by index_kpool"
            )
        if not isinstance(self.rms_norm_eps, (int, float)) or self.rms_norm_eps <= 0:
            raise Glm5NextDsaContractError("rms_norm_eps must be positive")

    @classmethod
    def official(cls) -> Glm5NextDsaConfig:
        return cls(
            hidden_size=4096,
            num_attention_heads=64,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=256,
            v_head_dim=256,
            index_n_heads=32,
            index_head_dim=128,
            index_topk=2048,
            index_kpool=4,
        )

    @classmethod
    def from_model_config(cls, config: object) -> Glm5NextDsaConfig:
        """Copy the required fields from an mlx-lm/Hugging Face config object."""

        names = (
            "hidden_size",
            "num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "v_head_dim",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "index_kpool",
        )
        missing = [name for name in names if not hasattr(config, name)]
        if missing:
            raise Glm5NextDsaContractError(
                f"model config is missing DSA fields: {missing}"
            )
        return cls(
            **{name: getattr(config, name) for name in names},
            qk_rope_head_dim=getattr(config, "qk_rope_head_dim", None),
            attention_bias=getattr(config, "attention_bias", False),
            index_kpool_compress=getattr(config, "index_kpool_compress", True),
            index_kpool_always_select_tail=getattr(
                config, "index_kpool_always_select_tail", True
            ),
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        )


def validate_dsa_layer_index(layer_idx: int) -> None:
    if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
        raise Glm5NextDsaContractError("DSA layer index must be an integer")
    if layer_idx not in ALL_DSA_LAYERS:
        raise Glm5NextDsaContractError(
            f"layer {layer_idx} is not DSA; expected {ALL_DSA_LAYERS}"
        )


_WEIGHT_NAMES = (
    "q_a_proj.weight",
    "q_a_layernorm.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
    "indexer.wq_b.weight",
    "indexer.wk.weight",
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    "indexer.weights_proj.weight",
    "indexer.index_kpool_compress_ape",
    "indexer.index_kpool_compress_gate",
)


def dsa_weight_shapes(config: Glm5NextDsaConfig) -> dict[str, tuple[int, ...]]:
    h = config.hidden_size
    nh = config.num_attention_heads
    qd = config.qk_nope_head_dim
    vd = config.v_head_dim
    ih = config.index_n_heads
    idim = config.index_head_dim
    return {
        "q_a_proj.weight": (config.q_lora_rank, h),
        "q_a_layernorm.weight": (config.q_lora_rank,),
        "q_b_proj.weight": (nh * qd, config.q_lora_rank),
        "kv_a_proj_with_mqa.weight": (config.kv_lora_rank, h),
        "kv_a_layernorm.weight": (config.kv_lora_rank,),
        "kv_b_proj.weight": (nh * (qd + vd), config.kv_lora_rank),
        "o_proj.weight": (h, nh * vd),
        "indexer.wq_b.weight": (ih * idim, config.q_lora_rank),
        "indexer.wk.weight": (idim, h),
        "indexer.k_norm.weight": (idim,),
        "indexer.k_norm.bias": (idim,),
        "indexer.weights_proj.weight": (ih, h),
        "indexer.index_kpool_compress_ape": (config.index_kpool, idim),
        "indexer.index_kpool_compress_gate": (idim, h),
    }


@dataclass(frozen=True, slots=True)
class Glm5NextDsaWeights:
    """Validated, dequantized tensors using official checkpoint suffix names."""

    tensors: Mapping[str, mx.array]

    def __getitem__(self, name: str) -> mx.array:
        return self.tensors[name]


def validate_dsa_weights(
    weights: Mapping[str, mx.array] | Glm5NextDsaWeights,
    config: Glm5NextDsaConfig,
) -> Glm5NextDsaWeights:
    """Validate the complete dequantized DSA tensor ABI, rejecting extras."""

    if isinstance(weights, Glm5NextDsaWeights):
        weights = weights.tensors
    if not isinstance(weights, Mapping):
        raise Glm5NextDsaContractError("DSA weights must be a mapping")
    names = set(weights)
    expected_names = set(_WEIGHT_NAMES)
    if names != expected_names:
        missing = sorted(expected_names - names)
        extra = sorted(names - expected_names)
        raise Glm5NextDsaContractError(
            f"DSA weight names changed; missing={missing}, extra={extra}"
        )
    expected_shapes = dsa_weight_shapes(config)
    copied: dict[str, mx.array] = {}
    for name in _WEIGHT_NAMES:
        tensor = weights[name]
        if not isinstance(tensor, mx.array):
            raise Glm5NextDsaContractError(f"{name} must be an MLX array")
        if tuple(tensor.shape) != expected_shapes[name]:
            raise Glm5NextDsaContractError(
                f"{name} shape changed: expected {expected_shapes[name]}, "
                f"found {tuple(tensor.shape)}"
            )
        if not mx.issubdtype(tensor.dtype, mx.floating):
            raise Glm5NextDsaContractError(f"{name} must have floating dtype")
        copied[name] = tensor
    return Glm5NextDsaWeights(copied)


class Glm5NextDsaCache(_BaseCache):
    """mlx-lm cache ABI for compressed MLA and packed indexer state.

    Physical rows are right-aligned when independent request caches are
    merged.  The packed valid channel remains authoritative for the indexer;
    ``left_padding`` additionally supplies mlx-lm's causal-mask interface.
    """

    def __init__(self, config: Glm5NextDsaConfig | None = None) -> None:
        self.kv_latent = None
        self.indexer_states = None
        self.kv_lora_rank = None if config is None else config.kv_lora_rank
        self.index_state_dim = None if config is None else 2 * config.index_head_dim + 1
        self._idx = 0
        self._batch_size = None
        self.left_padding = None
        self._prepared_left_padding = None
        self._right_padding = None
        self._prepared_lengths = None

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def offset(self) -> int:
        """Physical cache width, matching mlx-lm ``BatchKVCache.size``."""

        return self._idx

    @property
    def offsets(self) -> mx.array:
        """Per-row non-padding lengths after batch preparation/finalization."""

        if self._batch_size is None:
            return mx.array([], dtype=mx.int32)
        left = (
            mx.zeros((self._batch_size,), dtype=mx.int32)
            if self.left_padding is None
            else self.left_padding
        )
        return mx.full((self._batch_size,), self._idx, dtype=mx.int32) - left

    def _set_batch(self, batch_size: int) -> None:
        if self._batch_size is None:
            self._batch_size = batch_size
            if self._prepared_left_padding is None:
                self.left_padding = mx.zeros((batch_size,), dtype=mx.int32)
            else:
                if len(self._prepared_left_padding) != batch_size:
                    raise Glm5NextDsaContractError(
                        "prepared padding batch differs from cache input"
                    )
                self.left_padding = mx.array(
                    self._prepared_left_padding, dtype=mx.int32
                )
        elif batch_size != self._batch_size:
            raise Glm5NextDsaContractError(
                f"cache batch changed: expected {self._batch_size}, found {batch_size}"
            )

    def append(
        self,
        kv_latent: mx.array,
        indexer_states: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if kv_latent.ndim != 3 or indexer_states.ndim != 3:
            raise Glm5NextDsaContractError("cache inputs must be rank-3 [B, L, D]")
        if kv_latent.shape[:2] != indexer_states.shape[:2]:
            raise Glm5NextDsaContractError("latent and indexer cache geometry differs")
        self._set_batch(kv_latent.shape[0])
        if self.kv_lora_rank is None:
            self.kv_lora_rank = kv_latent.shape[2]
        if self.index_state_dim is None:
            self.index_state_dim = indexer_states.shape[2]
        if kv_latent.shape[2] != self.kv_lora_rank:
            raise Glm5NextDsaContractError("cached MLA latent width changed")
        if indexer_states.shape[2] != self.index_state_dim:
            raise Glm5NextDsaContractError("cached indexer state width changed")
        if (self.kv_latent is None) != (self.indexer_states is None):
            raise Glm5NextDsaContractError("partial DSA cache state is forbidden")
        if self.kv_latent is None:
            self.kv_latent = kv_latent
            self.indexer_states = indexer_states
        else:
            self.kv_latent = mx.concatenate(
                (self.kv_latent[:, : self._idx], kv_latent), axis=1
            )
            self.indexer_states = mx.concatenate(
                (self.indexer_states[:, : self._idx], indexer_states), axis=1
            )
        self._idx += kv_latent.shape[1]
        return self.kv_latent[:, : self._idx], self.indexer_states[:, : self._idx]

    update_and_fetch = append

    def prepare(self, *, lengths=None, right_padding=None, left_padding=None) -> None:
        if left_padding is not None:
            if self._idx:
                raise Glm5NextDsaContractError(
                    "left padding can only prepare an empty DSA cache"
                )
            self._prepared_left_padding = [int(value) for value in left_padding]
        if right_padding is not None:
            padding = [int(value) for value in right_padding]
            if any(value < 0 for value in padding):
                raise Glm5NextDsaContractError("right padding cannot be negative")
            self._right_padding = padding if any(padding) else None
        if lengths is not None:
            lengths = [int(value) for value in lengths]
            if any(value < 0 for value in lengths):
                raise Glm5NextDsaContractError("cache lengths cannot be negative")
            expected_batch = len(lengths)
            if self._batch_size is not None and self._batch_size != expected_batch:
                raise Glm5NextDsaContractError("prepared lengths batch changed")
            if right_padding is not None and len(right_padding) != expected_batch:
                raise Glm5NextDsaContractError("padding and lengths batch differs")
            # Equal-length preparation is also used by some batching paths;
            # only an actual right-padded prompt needs per-chunk validity.
            self._prepared_lengths = (
                lengths if self._right_padding is not None else None
            )

    def current_valid_mask(self, length: int) -> mx.array:
        """Return bool validity for the next chunk prepared by mlx-lm batching."""

        if self._batch_size is None:
            batch = (
                len(self._prepared_lengths)
                if self._prepared_lengths is not None
                else len(self._prepared_left_padding or ())
            )
        else:
            batch = self._batch_size
        if batch <= 0:
            raise Glm5NextDsaContractError(
                "attention_mask is required before cache batch is known"
            )
        if self._prepared_lengths is None and self._prepared_left_padding is None:
            return mx.ones((batch, length), dtype=mx.bool_)
        positions = self._idx + mx.arange(length)[None, :]
        if self._prepared_lengths is None:
            starts = mx.array(self._prepared_left_padding, dtype=mx.int32)[:, None]
            return positions >= starts
        limits = mx.array(self._prepared_lengths, dtype=mx.int32)[:, None]
        if self._prepared_left_padding is None:
            return positions < limits
        starts = mx.array(self._prepared_left_padding, dtype=mx.int32)[:, None]
        return (positions >= starts) & (positions < starts + limits)

    def finalize(self) -> None:
        if self._right_padding is None:
            return
        if self.kv_latent is None or self.indexer_states is None:
            raise Glm5NextDsaContractError("cannot finalize padding on an empty cache")
        if len(self._right_padding) != self._batch_size:
            raise Glm5NextDsaContractError("right padding batch changed")
        padding = mx.array(self._right_padding, dtype=mx.int32)
        self.kv_latent = dynamic_roll(self.kv_latent, padding, axis=1)
        self.indexer_states = dynamic_roll(self.indexer_states, padding, axis=1)
        positions = mx.arange(self._idx)[None, :]
        prefix_valid = positions >= padding[:, None]
        valid = self.indexer_states[..., -1].astype(mx.bool_) & prefix_valid
        self.indexer_states = mx.concatenate(
            (
                self.indexer_states[..., :-1],
                valid[..., None].astype(self.indexer_states.dtype),
            ),
            axis=-1,
        )
        self.left_padding = self.left_padding + padding
        self._right_padding = None
        self._prepared_lengths = None

    @property
    def state(self):
        if self.kv_latent is None:
            return None, None
        return (
            self.kv_latent[:, : self._idx],
            self.indexer_states[:, : self._idx],
        )

    @state.setter
    def state(self, value) -> None:
        self.kv_latent, self.indexer_states = value
        if (self.kv_latent is None) != (self.indexer_states is None):
            raise Glm5NextDsaContractError("partial restored DSA cache is forbidden")
        if self.kv_latent is None:
            self._idx = 0
            self._batch_size = None
            return
        if self.kv_latent.ndim != 3 or self.indexer_states.ndim != 3:
            raise Glm5NextDsaContractError(
                "restored DSA cache must contain rank-3 arrays"
            )
        if self.kv_latent.shape[:2] != self.indexer_states.shape[:2]:
            raise Glm5NextDsaContractError("restored DSA cache geometry differs")
        self._batch_size, self._idx = self.kv_latent.shape[:2]
        self.kv_lora_rank = self.kv_latent.shape[2]
        self.index_state_dim = self.indexer_states.shape[2]
        if not hasattr(self, "left_padding") or self.left_padding is None:
            self.left_padding = mx.zeros((self._batch_size,), dtype=mx.int32)
        self._right_padding = None
        self._prepared_left_padding = None
        self._prepared_lengths = None

    @property
    def meta_state(self):
        left = () if self.left_padding is None else tuple(self.left_padding.tolist())
        return (
            str(self._idx),
            left,
            "" if self.kv_lora_rank is None else str(self.kv_lora_rank),
            "" if self.index_state_dim is None else str(self.index_state_dim),
        )

    @meta_state.setter
    def meta_state(self, value) -> None:
        idx, left, kv_width, index_width = value
        self._idx = int(idx)
        self.left_padding = mx.array(left, dtype=mx.int32)
        self._batch_size = len(left) or getattr(self, "_batch_size", None)
        self.kv_lora_rank = None if kv_width == "" else int(kv_width)
        self.index_state_dim = None if index_width == "" else int(index_width)
        self._right_padding = None
        self._prepared_left_padding = None
        self._prepared_lengths = None

    def size(self) -> int:
        return self._idx

    def empty(self) -> bool:
        return self.kv_latent is None or self._idx == 0

    @property
    def nbytes(self) -> int:
        if self.kv_latent is None:
            return 0
        return self.kv_latent.nbytes + self.indexer_states.nbytes

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise Glm5NextDsaContractError("trim count must be a non-negative integer")
        removed = min(self._idx, n)
        if removed and self.kv_latent is not None:
            self._idx -= removed
            self.kv_latent = self.kv_latent[:, : self._idx]
            self.indexer_states = self.indexer_states[:, : self._idx]
            self.left_padding = mx.minimum(self.left_padding, self._idx)
        return removed

    def make_mask(self, n: int, return_array: bool = False, **kwargs):
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise Glm5NextDsaContractError("mask query length must be positive")
        return create_causal_mask(
            n,
            offset=self._idx,
            left_padding=self.left_padding,
            **kwargs,
        )

    def filter(self, batch_indices) -> None:
        if not isinstance(batch_indices, mx.array):
            batch_indices = mx.array(batch_indices, dtype=mx.int32)
        if batch_indices.ndim != 1 or not mx.issubdtype(
            batch_indices.dtype, mx.integer
        ):
            raise Glm5NextDsaContractError(
                "batch_indices must be a rank-1 integer array"
            )
        if self.kv_latent is not None:
            self.kv_latent = mx.take(self.kv_latent, batch_indices, axis=0)
            self.indexer_states = mx.take(self.indexer_states, batch_indices, axis=0)
        if self.left_padding is not None:
            self.left_padding = mx.take(self.left_padding, batch_indices, axis=0)
            minimum = int(mx.min(self.left_padding).item())
            if minimum > 0 and self.kv_latent is not None:
                self.kv_latent = self.kv_latent[:, minimum:]
                self.indexer_states = self.indexer_states[:, minimum:]
                self._idx -= minimum
                self.left_padding -= minimum
        self._batch_size = batch_indices.shape[0]

    def reorder(self, batch_indices: mx.array) -> None:
        self.filter(batch_indices)

    @staticmethod
    def _right_aligned(cache: Glm5NextDsaCache, width: int):
        batch = cache._batch_size or 0
        left = width - cache._idx
        kv = cache.kv_latent
        packed = cache.indexer_states
        if kv is None:
            if cache.kv_lora_rank is None or cache.index_state_dim is None:
                raise Glm5NextDsaContractError(
                    "cannot batch an unshaped empty DSA cache"
                )
            kv = mx.zeros((batch, 0, cache.kv_lora_rank), dtype=mx.float32)
            packed = mx.zeros((batch, 0, cache.index_state_dim), dtype=mx.float32)
        if left:
            kv = mx.pad(kv[:, : cache._idx], ((0, 0), (left, 0), (0, 0)))
            packed = mx.pad(packed[:, : cache._idx], ((0, 0), (left, 0), (0, 0)))
        else:
            kv = kv[:, : cache._idx]
            packed = packed[:, : cache._idx]
        old_left = (
            mx.zeros((batch,), dtype=mx.int32)
            if cache.left_padding is None
            else cache.left_padding
        )
        return kv, packed, old_left + left

    def extend(self, other: Glm5NextDsaCache) -> None:
        if not isinstance(other, Glm5NextDsaCache):
            raise Glm5NextDsaContractError("can only extend with a DSA cache")
        if self._right_padding is not None or other._right_padding is not None:
            raise Glm5NextDsaContractError("finalize caches before extending them")
        if self._batch_size is None:
            restored = Glm5NextDsaCache.from_state(other.state, other.meta_state)
            self.__dict__.update(restored.__dict__)
            return
        if other._batch_size is None:
            return
        if (
            self.kv_lora_rank != other.kv_lora_rank
            or self.index_state_dim != other.index_state_dim
        ):
            raise Glm5NextDsaContractError("cannot extend caches with different widths")
        width = max(self._idx, other._idx)
        left = self._right_aligned(self, width)
        right = self._right_aligned(other, width)
        self.kv_latent = mx.concatenate((left[0], right[0]), axis=0)
        self.indexer_states = mx.concatenate((left[1], right[1]), axis=0)
        self.left_padding = mx.concatenate((left[2], right[2]), axis=0)
        self._idx = width
        self._batch_size += other._batch_size

    def extract(self, idx: int) -> Glm5NextDsaCache:
        if self._batch_size is None or not 0 <= idx < self._batch_size:
            raise Glm5NextDsaContractError("cache extraction index out of range")
        cache = Glm5NextDsaCache()
        cache.kv_lora_rank = self.kv_lora_rank
        cache.index_state_dim = self.index_state_dim
        padding = int(self.left_padding[idx].item())
        if self.kv_latent is not None:
            cache.kv_latent = mx.contiguous(
                self.kv_latent[idx : idx + 1, padding : self._idx]
            )
            cache.indexer_states = mx.contiguous(
                self.indexer_states[idx : idx + 1, padding : self._idx]
            )
            cache._idx = cache.kv_latent.shape[1]
        cache._batch_size = 1
        cache.left_padding = mx.zeros((1,), dtype=mx.int32)
        return cache

    @classmethod
    def merge(cls, caches) -> Glm5NextDsaCache:
        if not caches:
            raise Glm5NextDsaContractError("cannot merge an empty cache list")
        rows = []
        for cache in caches:
            if not isinstance(cache, cls):
                raise Glm5NextDsaContractError("cache merge type changed")
            if cache._batch_size is None:
                shaped = cls()
                shaped.kv_lora_rank = cache.kv_lora_rank
                shaped.index_state_dim = cache.index_state_dim
                shaped._batch_size = 1
                shaped.left_padding = mx.zeros((1,), dtype=mx.int32)
                rows.append(shaped)
            else:
                rows.extend(cache.extract(i) for i in range(cache._batch_size))
        merged = rows[0]
        for cache in rows[1:]:
            merged.extend(cache)
        return merged

    def reset(self) -> None:
        config = None
        if self.kv_lora_rank is not None and self.index_state_dim is not None:
            kv_width = self.kv_lora_rank
            index_width = self.index_state_dim
        else:
            kv_width = index_width = None
        self.__init__(config)
        self.kv_lora_rank = kv_width
        self.index_state_dim = index_width


def _linear(x: mx.array, weight: mx.array) -> mx.array:
    return mx.matmul(x, weight.T)


def _masked_softmax(logits: mx.array, valid: mx.array, axis: int) -> mx.array:
    """Float32 softmax returning zero for wholly invalid rows."""

    logits = logits.astype(mx.float32)
    floor = mx.array(mx.finfo(mx.float32).min, dtype=mx.float32)
    masked = mx.where(valid, logits, floor)
    maximum = mx.max(masked, axis=axis, keepdims=True)
    exponent = mx.where(valid, mx.exp(masked - maximum), mx.array(0.0))
    denominator = mx.sum(exponent, axis=axis, keepdims=True)
    return exponent / mx.maximum(denominator, mx.array(1.0))


def _batch_take_rows(states: mx.array, indices: mx.array) -> mx.array:
    """Gather [B, ...] row indices without broadcasting over kv_length."""

    if states.ndim != 3 or indices.ndim < 2 or states.shape[0] != indices.shape[0]:
        raise Glm5NextDsaContractError("invalid batched row gather geometry")
    rows = []
    flat_size = 1
    for size in indices.shape[1:]:
        flat_size *= size
    for batch in range(states.shape[0]):
        gathered = mx.take(states[batch], indices[batch].reshape(flat_size), axis=0)
        rows.append(gathered.reshape(indices.shape[1:] + (states.shape[-1],)))
    return mx.stack(rows, axis=0)


def _pooled_indexer_states(
    packed_states: mx.array,
    config: Glm5NextDsaConfig,
    index_kpool_compress_ape: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Rebuild complete compressed pools, aligned to each row's first real key."""

    idim = config.index_head_dim
    keys = packed_states[..., :idim]
    gate_scores = packed_states[..., idim : 2 * idim]
    valid_keys = packed_states[..., -1].astype(mx.bool_)
    batch, kv_length = keys.shape[:2]
    pool = config.index_kpool
    pool_count = (kv_length + pool - 1) // pool

    any_valid = mx.any(valid_keys, axis=-1)
    first_key = mx.where(
        any_valid,
        mx.argmax(valid_keys.astype(mx.int32), axis=-1),
        mx.array(kv_length, dtype=mx.int32),
    ).astype(mx.int32)
    offsets = mx.arange(pool_count * pool, dtype=mx.int32).reshape(1, pool_count, pool)
    pool_indices = first_key[:, None, None] + offsets
    within = pool_indices < kv_length
    safe_indices = mx.clip(pool_indices, 0, max(kv_length - 1, 0))

    grouped_keys = _batch_take_rows(keys, safe_indices)
    grouped_gate = _batch_take_rows(gate_scores, safe_indices)
    grouped_valid = _batch_take_rows(
        valid_keys[..., None].astype(mx.float32), safe_indices
    )[..., 0].astype(mx.bool_)
    grouped_valid = grouped_valid & within
    pool_valid = mx.all(grouped_valid, axis=-1)
    raw_indices = mx.where(grouped_valid, pool_indices, mx.array(-1, mx.int32))

    logits = (
        grouped_gate.astype(mx.float32)
        + index_kpool_compress_ape.astype(mx.float32)[None, None, :, :]
    )
    probabilities = _masked_softmax(logits, grouped_valid[..., None], axis=2).astype(
        grouped_keys.dtype
    )
    pool_keys = mx.sum(probabilities * grouped_keys, axis=2)
    return pool_keys, raw_indices, pool_valid


def _append_visible_tail(
    topk_indices: mx.array,
    token_visible: mx.array,
    key_valid: mx.array,
    pool: int,
) -> mx.array:
    max_tail = pool - 1
    if max_tail == 0:
        return topk_indices
    batch, _, kv_length = token_visible.shape
    any_key = mx.any(key_valid, axis=-1)
    first_key = mx.where(
        any_key,
        mx.argmax(key_valid.astype(mx.int32), axis=-1),
        mx.array(kv_length, mx.int32),
    ).astype(mx.int32)
    visible_count = mx.sum(token_visible.astype(mx.int32), axis=-1)
    tail_count = visible_count % pool
    offsets = mx.arange(max_tail, dtype=mx.int32)
    tail_start = first_key[:, None] + visible_count - tail_count
    tail_indices = tail_start[..., None] + offsets
    tail_valid = (offsets[None, None, :] < tail_count[..., None]) & (
        tail_indices < kv_length
    )
    safe = mx.clip(tail_indices, 0, max(kv_length - 1, 0))
    tail_visible = mx.take_along_axis(token_visible, safe, axis=-1)
    tail_indices = mx.where(tail_valid & tail_visible, tail_indices, mx.array(-1))
    return mx.concatenate((topk_indices, tail_indices.astype(mx.int32)), axis=-1)


def _select_topk_from_projected(
    hidden_states: mx.array,
    q: mx.array,
    head_weights: mx.array,
    attention_mask: mx.array,
    packed_states: mx.array,
    config: Glm5NextDsaConfig,
    index_kpool_compress_ape: mx.array,
) -> mx.array:
    """Select raw keys from already projected index queries/head weights."""

    if hidden_states.ndim != 3:
        raise Glm5NextDsaContractError("hidden_states must be [B, L, hidden_size]")
    batch, q_length, hidden = hidden_states.shape
    if hidden != config.hidden_size:
        raise Glm5NextDsaContractError("hidden state width changed")
    if q.shape != (
        batch,
        q_length,
        config.index_n_heads,
        config.index_head_dim,
    ):
        raise Glm5NextDsaContractError("projected index query shape changed")
    if head_weights.shape != (batch, q_length, config.index_n_heads):
        raise Glm5NextDsaContractError("projected index weight shape changed")
    if attention_mask.shape != (batch, q_length) or attention_mask.dtype != mx.bool_:
        raise Glm5NextDsaContractError("attention_mask must be bool [B, L]")
    expected_packed = 2 * config.index_head_dim + 1
    if (
        packed_states.ndim != 3
        or packed_states.shape[0] != batch
        or packed_states.shape[2] != expected_packed
        or packed_states.shape[1] < q_length
    ):
        raise Glm5NextDsaContractError("packed indexer cache shape changed")

    pool_keys, pool_indices, pool_valid = _pooled_indexer_states(
        packed_states, config, index_kpool_compress_ape
    )
    # [B,L,I,D] x [B,P,D] -> [B,L,I,P], still pool-compressed.
    scores = mx.einsum(
        "blid,bpd->blip", q.astype(mx.float32), pool_keys.astype(mx.float32)
    )
    scores = mx.maximum(scores * (config.index_head_dim**-0.5), mx.array(0.0))
    scaled_weights = head_weights.astype(mx.float32) * config.index_n_heads**-0.5
    index_scores = mx.sum(scaled_weights[..., None] * scores, axis=2)

    kv_length = packed_states.shape[1]
    valid_keys = packed_states[..., -1].astype(mx.bool_)
    q_positions = kv_length - q_length + mx.arange(q_length, dtype=mx.int32)
    kv_positions = mx.arange(kv_length, dtype=mx.int32)
    visible_tokens = (
        kv_positions[None, None, :] <= q_positions[None, :, None]
    ) & valid_keys[:, None, :]

    pool_end = mx.clip(pool_indices[..., -1], 0, max(kv_length - 1, 0))
    pool_visible = mx.take_along_axis(
        visible_tokens,
        mx.broadcast_to(pool_end[:, None, :], (batch, q_length, pool_end.shape[1])),
        axis=-1,
    )
    valid_candidates = pool_visible & pool_valid[:, None, :]
    floor = mx.array(mx.finfo(mx.float32).min, mx.float32)
    index_scores = mx.where(valid_candidates, index_scores, floor)

    select_k = min(config.index_topk // config.index_kpool, pool_indices.shape[1])
    if select_k:
        selected = mx.argsort(-index_scores, axis=-1)[..., :select_k].astype(mx.int32)
        selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)
        # Gather pool rows separately for each batch without a kv-length expansion.
        chosen = []
        for b in range(batch):
            flat = selected[b].reshape(-1)
            chosen.append(
                mx.take(pool_indices[b], flat, axis=0).reshape(
                    q_length, select_k, config.index_kpool
                )
            )
        selected_indices = mx.stack(chosen, axis=0)
        expanded_valid = mx.broadcast_to(
            selected_valid[..., None], selected_indices.shape
        )
        topk_indices = mx.where(expanded_valid, selected_indices, mx.array(-1))
        topk_indices = topk_indices.reshape(batch, q_length, -1)
    else:
        topk_indices = mx.full((batch, q_length, 0), -1, dtype=mx.int32)

    topk_indices = _append_visible_tail(
        topk_indices, visible_tokens, valid_keys, config.index_kpool
    )
    output_width = config.index_topk + config.index_kpool - 1
    current_width = topk_indices.shape[-1]
    if current_width < output_width:
        topk_indices = mx.concatenate(
            (
                topk_indices,
                mx.full(
                    (batch, q_length, output_width - current_width),
                    -1,
                    dtype=mx.int32,
                ),
            ),
            axis=-1,
        )
    topk_indices = topk_indices[..., :output_width]
    topk_indices = mx.where(attention_mask[..., None], topk_indices, mx.array(-1))
    return topk_indices.astype(mx.int32)


def select_topk_indices(
    hidden_states: mx.array,
    q_resid: mx.array,
    attention_mask: mx.array,
    packed_states: mx.array,
    config: Glm5NextDsaConfig,
    weights: Mapping[str, mx.array] | Glm5NextDsaWeights,
) -> mx.array:
    """Run the exact compressed-pool indexer from standalone weight arrays."""

    weights = validate_dsa_weights(weights, config)
    batch, q_length = hidden_states.shape[:2]
    if q_resid.shape != (batch, q_length, config.q_lora_rank):
        raise Glm5NextDsaContractError("q_resid shape changed")
    q = _linear(q_resid, weights["indexer.wq_b.weight"]).reshape(
        batch, q_length, config.index_n_heads, config.index_head_dim
    )
    head_weights = _linear(
        hidden_states.astype(weights["indexer.weights_proj.weight"].dtype),
        weights["indexer.weights_proj.weight"],
    )
    return _select_topk_from_projected(
        hidden_states,
        q,
        head_weights,
        attention_mask,
        packed_states,
        config,
        weights["indexer.index_kpool_compress_ape"],
    )


def sparse_mla_attention(
    query_states: mx.array,
    kv_latent: mx.array,
    topk_indices: mx.array,
    kv_b_proj_weight: mx.array,
    config: Glm5NextDsaConfig,
) -> mx.array:
    """Exact selected-token MLA returning ``[B, L, H * v_head_dim]``.

    The implementation absorbs each head's K projection into the query, gathers
    only selected latent rows, and applies the V projection after the sparse
    weighted reduction.  No dense attention score fallback exists here.
    """

    if query_states.ndim != 4:
        raise Glm5NextDsaContractError("query_states must be [B, H, L, D]")
    batch, heads, q_length, qdim = query_states.shape
    if (heads, qdim) != (config.num_attention_heads, config.qk_nope_head_dim):
        raise Glm5NextDsaContractError("query head geometry changed")
    if kv_latent.ndim != 3 or kv_latent.shape[0] != batch:
        raise Glm5NextDsaContractError("kv_latent must be [B, K, kv_lora_rank]")
    if kv_latent.shape[-1] != config.kv_lora_rank:
        raise Glm5NextDsaContractError("MLA latent width changed")
    if topk_indices.ndim != 3 or topk_indices.shape[:2] != (batch, q_length):
        raise Glm5NextDsaContractError("topk_indices must be [B, L, selected]")
    if not mx.issubdtype(topk_indices.dtype, mx.integer):
        raise Glm5NextDsaContractError("topk_indices must have integer dtype")
    expected_kvb = (
        config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
        config.kv_lora_rank,
    )
    if tuple(kv_b_proj_weight.shape) != expected_kvb:
        raise Glm5NextDsaContractError("kv_b_proj.weight shape changed")
    if topk_indices.shape[-1] <= 0:
        raise Glm5NextDsaContractError("sparse attention requires selected slots")

    kv_length = kv_latent.shape[1]
    valid = (topk_indices >= 0) & (topk_indices < kv_length)
    safe = mx.clip(topk_indices, 0, max(kv_length - 1, 0)).astype(mx.int32)
    selected_latent = _batch_take_rows(kv_latent, safe)

    projection = kv_b_proj_weight.reshape(
        config.num_attention_heads,
        config.qk_nope_head_dim + config.v_head_dim,
        config.kv_lora_rank,
    )
    key_projection = projection[:, : config.qk_nope_head_dim]
    value_projection = projection[:, config.qk_nope_head_dim :]

    # Equivalent to expanding selected keys, but keeps the largest intermediate
    # at [B,H,L,C] / [B,L,W,C] rather than [B,H,L,K,D].
    latent_query = mx.einsum(
        "bhld,hdc->bhlc",
        query_states.astype(mx.float32),
        key_projection.astype(mx.float32),
    )
    logits = mx.einsum(
        "bhlc,blwc->bhlw",
        latent_query,
        selected_latent.astype(mx.float32),
    ) * (config.qk_nope_head_dim**-0.5)
    probabilities = _masked_softmax(logits, valid[:, None, :, :], axis=-1)
    context_latent = mx.einsum(
        "bhlw,blwc->bhlc", probabilities, selected_latent.astype(mx.float32)
    )
    output = mx.einsum(
        "bhlc,hdc->bhld", context_latent, value_projection.astype(mx.float32)
    )
    output = output.transpose(0, 2, 1, 3).reshape(
        batch, q_length, config.num_attention_heads * config.v_head_dim
    )
    return output.astype(query_states.dtype)


def _sparse_mla_attention_module(
    query_states: mx.array,
    kv_latent: mx.array,
    topk_indices: mx.array,
    kv_b_proj: nn.Module,
    config: Glm5NextDsaConfig,
) -> mx.array:
    """Selected-token MLA through a load-time replaceable projection module."""

    batch, heads, q_length, _ = query_states.shape
    kv_length = kv_latent.shape[1]
    valid = (topk_indices >= 0) & (topk_indices < kv_length)
    safe = mx.clip(topk_indices, 0, max(kv_length - 1, 0)).astype(mx.int32)
    selected_latent = _batch_take_rows(kv_latent, safe)
    selected_kv = (
        kv_b_proj(selected_latent)
        .reshape(
            batch,
            q_length,
            topk_indices.shape[-1],
            heads,
            config.qk_nope_head_dim + config.v_head_dim,
        )
        .transpose(0, 3, 1, 2, 4)
    )
    selected_keys = selected_kv[..., : config.qk_nope_head_dim]
    selected_values = selected_kv[..., config.qk_nope_head_dim :]
    logits = mx.sum(
        query_states[..., None, :].astype(mx.float32)
        * selected_keys.astype(mx.float32),
        axis=-1,
    ) * (config.qk_nope_head_dim**-0.5)
    probabilities = _masked_softmax(logits, valid[:, None], axis=-1)
    output = mx.sum(
        probabilities[..., None] * selected_values.astype(mx.float32), axis=-2
    )
    return (
        output.transpose(0, 2, 1, 3)
        .reshape(batch, q_length, heads * config.v_head_dim)
        .astype(query_states.dtype)
    )


class _Glm5NextDsaIndexer(nn.Module):
    """Official indexer parameter tree; selection math remains in shared helpers."""

    def __init__(self, config: Glm5NextDsaConfig) -> None:
        super().__init__()
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
        )
        self.wk = nn.Linear(config.hidden_size, config.index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(
            config.index_head_dim, eps=1e-6, affine=True, bias=True
        )
        self.weights_proj = nn.Linear(
            config.hidden_size, config.index_n_heads, bias=False
        )
        self.index_kpool_compress_ape = mx.zeros(
            (config.index_kpool, config.index_head_dim)
        )
        self.index_kpool_compress_gate = mx.zeros(
            (config.index_head_dim, config.hidden_size)
        )


class Glm5NextDsa(nn.Module):
    """Lazy-loadable exact DSA module with official checkpoint parameter names."""

    def __init__(
        self,
        config: Glm5NextDsaConfig | object,
        weights: Mapping[str, mx.array] | Glm5NextDsaWeights | None = None,
        *,
        layer_idx: int,
    ) -> None:
        super().__init__()
        if not isinstance(config, Glm5NextDsaConfig):
            config = Glm5NextDsaConfig.from_model_config(config)
        validate_dsa_layer_index(layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * config.qk_nope_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, config.kv_lora_rank, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            bias=False,
        )
        self.indexer = _Glm5NextDsaIndexer(config)
        if weights is not None:
            validated = validate_dsa_weights(weights, config)
            self.load_weights(list(validated.tensors.items()), strict=True)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        cache: Glm5NextDsaCache | None = None,
        *,
        return_topk: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        if hidden_states.ndim != 3:
            raise Glm5NextDsaContractError("hidden_states must be rank 3")
        batch, length, hidden = hidden_states.shape
        if hidden != self.config.hidden_size:
            raise Glm5NextDsaContractError("hidden state width changed")
        if attention_mask is None:
            attention_mask = (
                cache.current_valid_mask(length)
                if cache is not None
                and (
                    cache.batch_size is not None
                    or cache._prepared_lengths is not None
                    or cache._prepared_left_padding is not None
                )
                else mx.ones((batch, length), dtype=mx.bool_)
            )
        if attention_mask.shape != (batch, length) or attention_mask.dtype != mx.bool_:
            raise Glm5NextDsaContractError("attention_mask must be bool [B, L]")
        if not mx.issubdtype(hidden_states.dtype, mx.floating):
            raise Glm5NextDsaContractError("hidden_states must have floating dtype")

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = (
            self.q_b_proj(q_resid)
            .reshape(
                batch,
                length,
                self.config.num_attention_heads,
                self.config.qk_nope_head_dim,
            )
            .transpose(0, 2, 1, 3)
        )

        kv_latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))
        index_key = self.indexer.k_norm(self.indexer.wk(hidden_states))
        gate_scores = _linear(hidden_states, self.indexer.index_kpool_compress_gate)
        packed = mx.concatenate(
            (index_key, gate_scores, attention_mask[..., None].astype(index_key.dtype)),
            axis=-1,
        )
        if cache is None:
            full_latent, full_packed = kv_latent, packed
        elif isinstance(cache, Glm5NextDsaCache):
            full_latent, full_packed = cache.append(kv_latent, packed)
        else:
            raise Glm5NextDsaContractError("cache must be Glm5NextDsaCache or None")

        index_query = self.indexer.wq_b(q_resid).reshape(
            batch,
            length,
            self.config.index_n_heads,
            self.config.index_head_dim,
        )
        index_weights = self.indexer.weights_proj(hidden_states)
        topk_indices = _select_topk_from_projected(
            hidden_states,
            index_query,
            index_weights,
            attention_mask,
            full_packed,
            self.config,
            self.indexer.index_kpool_compress_ape,
        )
        output = _sparse_mla_attention_module(
            query_states,
            full_latent,
            topk_indices,
            self.kv_b_proj,
            self.config,
        )
        output = self.o_proj(output)
        if return_topk:
            return output, topk_indices
        return output


GLM5_NEXT_DSA_MODULE_READY = True


__all__ = [
    "ALL_DSA_LAYERS",
    "GLM5_NEXT_DSA_MODULE_READY",
    "MAIN_DSA_LAYERS",
    "MTP_DSA_LAYER",
    "OFFICIAL_MODEL_REVISION",
    "UPSTREAM_TRANSFORMERS_COMMIT",
    "UPSTREAM_TRANSFORMERS_PATH",
    "UPSTREAM_TRANSFORMERS_SHA256",
    "Glm5NextDsa",
    "Glm5NextDsaCache",
    "Glm5NextDsaConfig",
    "Glm5NextDsaContractError",
    "Glm5NextDsaWeights",
    "dsa_weight_shapes",
    "select_topk_indices",
    "sparse_mla_attention",
    "validate_dsa_layer_index",
    "validate_dsa_weights",
]
