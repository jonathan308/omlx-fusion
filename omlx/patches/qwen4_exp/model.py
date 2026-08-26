# SPDX-License-Identifier: Apache-2.0
"""Native mlx-lm model surface for Qwen3.8 Flash Next.

This file is loaded as ``mlx_lm.models.qwen4_exp`` by the registration shim.
It models the published GDN, MoE, four-stream hyper-connections, layer-1 PLE
and QSA parameter tree. QSA executes through Fusion's exact portable MLX
micro-block sparse adapter and never falls back to dense attention or DSA.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
from mlx_lm.models.pipeline import PipelineMixin

from omlx.patches.qwen4_exp.config import ModelArgs, TextModelArgs
from omlx.patches.qwen4_exp.gdn import (
    Qwen4ExpGatedDeltaNet,
    validate_gdn_weight_layout,
)
from omlx.patches.qwen4_exp.hc import (
    Qwen4ExpGatedResidual,
    Qwen4ExpHyperConnectionMixer,
    expand_hyper_residual,
    validate_hc_weight_layout,
)
from omlx.patches.qwen4_exp.moe import (
    Qwen4ExpSparseMoeBlock,
    sanitize_moe_weights,
    validate_moe_weight_layout,
)
from omlx.patches.qwen4_exp.ple import (
    PLE_MLX_Q8_DTYPE,
    PLE_PREFIX,
    Qwen4ExpPLESSDPool,
)
from omlx.patches.qwen4_exp.qsa import (
    Qwen4ExpQSAExecutor,
    validate_qsa_weights,
)
from omlx.patches.qwen4_exp.qsa_mlx import (
    Qwen4ExpMLXQSABackend,
    Qwen4ExpQSAKVCache,
    prepare_qsa_request,
)

QWEN4_EXP_STRICT_QSA = True


class ZeroCenteredRMSNorm(nn.Module):
    """Official Qwen4-Exp RMSNorm: multiply by ``1 + weight``."""

    def __init__(self, dim: int, eps: float = 1e-6, group_size: int | None = None):
        super().__init__()
        if group_size is not None and dim % group_size:
            raise ValueError(f"{dim=} must be divisible by {group_size=}")
        self.weight = mx.zeros((dim,))
        self.eps = eps
        self.group_size = group_size

    def __call__(self, value):
        dtype = value.dtype
        work = value.astype(mx.float32)
        if self.group_size is not None:
            work = work.reshape(*work.shape[:-1], -1, self.group_size)
        work = work * mx.rsqrt(
            mx.mean(mx.square(work), axis=-1, keepdims=True) + self.eps
        )
        if self.group_size is not None:
            work = work.reshape(*value.shape)
        return (work * (1.0 + self.weight.astype(mx.float32))).astype(dtype)


class Qwen4ExpQSAIndexer(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.index_n_heads = args.indexer_n_heads
        self.index_kv_heads = args.indexer_kv_heads
        self.index_head_dim = args.indexer_head_dim
        output = (args.indexer_n_heads + args.indexer_kv_heads) * args.indexer_head_dim
        self.index_qk_proj = nn.Linear(args.hidden_size, output, bias=False)
        self.q_layernorm = ZeroCenteredRMSNorm(args.indexer_head_dim, args.rms_norm_eps)
        self.k_layernorm = ZeroCenteredRMSNorm(args.indexer_head_dim, args.rms_norm_eps)


class Qwen4ExpAttention(nn.Module):
    """Published QSA projections with true micro-block sparse execution."""

    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * args.head_dim * 2,
            bias=False,
        )
        self.k_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * args.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, args.num_key_value_heads * args.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * args.head_dim, args.hidden_size, bias=False
        )
        self.q_norm = ZeroCenteredRMSNorm(args.head_dim, args.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(args.head_dim, args.rms_norm_eps)
        self.indexer = Qwen4ExpQSAIndexer(args)
        self.qsa = Qwen4ExpQSAExecutor(
            args,
            backend=Qwen4ExpMLXQSABackend(index_key_norm=self.indexer.k_layernorm),
        )
        self.rope_theta = float(args.rope_theta)
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)

    def _position_embeddings(self, batch: int, length: int, offset, dtype):
        steps = mx.arange(length, dtype=mx.float32)
        if isinstance(offset, mx.array):
            positions = mx.maximum(offset.astype(mx.float32)[:, None] + steps, 0)
        else:
            positions = mx.broadcast_to((float(offset) + steps)[None], (batch, length))
        dimensions = mx.arange(0, self.rotary_dim, 2, dtype=mx.float32)
        inverse = mx.power(self.rope_theta, -dimensions / self.rotary_dim)
        frequencies = positions[..., None] * inverse[None, None, :]
        frequencies = mx.concatenate((frequencies, frequencies), axis=-1)
        cos = mx.cos(frequencies).astype(dtype)
        sin = mx.sin(frequencies).astype(dtype)
        return cos, sin

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        projected = self.q_proj(x).reshape(
            batch, length, self.num_attention_heads, self.head_dim * 2
        )
        queries, gate = mx.split(projected, 2, axis=-1)
        gate = gate.reshape(batch, length, -1)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            self.k_proj(x).reshape(
                batch, length, self.num_key_value_heads, self.head_dim
            )
        ).transpose(0, 2, 1, 3)
        values = (
            self.v_proj(x)
            .reshape(batch, length, self.num_key_value_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        index_qk = self.indexer.index_qk_proj(x).reshape(
            batch,
            length,
            self.indexer.index_n_heads + self.indexer.index_kv_heads,
            self.indexer.index_head_dim,
        )
        index_queries = self.indexer.q_layernorm(
            index_qk[:, :, : self.indexer.index_n_heads, :]
        )
        index_keys = index_qk[:, :, self.indexer.index_n_heads, :]

        if isinstance(cache, CacheList):
            main_cache, auxiliary_cache = cache.caches
            offset = main_cache.offset
        else:
            main_cache = auxiliary_cache = None
            offset = cache.offset if isinstance(cache, Qwen4ExpQSAKVCache) else 0
        cos, sin = self._position_embeddings(batch, length, offset, queries.dtype)
        if main_cache is not None:
            prepared_mask = (
                mask
                if isinstance(mask, mx.array)
                else main_cache.make_mask(length, return_array=True)
            )
            keys, values = main_cache.update_and_fetch(keys, values)
            auxiliary_values = mx.concatenate((cos, sin), axis=-1)[:, None]
            index_keys, auxiliary_values = auxiliary_cache.update_and_fetch(
                index_keys[:, None], auxiliary_values
            )
            index_keys = index_keys[:, 0]
            cos, sin = mx.split(auxiliary_values[:, 0], 2, axis=-1)
            request = prepare_qsa_request(
                queries=queries,
                keys=keys,
                values=values,
                index_queries=index_queries,
                index_keys=index_keys,
                position_cos=cos,
                position_sin=sin,
                attention_mask=prepared_mask,
            )
        else:
            request = prepare_qsa_request(
                queries=queries,
                keys=keys,
                values=values,
                index_queries=index_queries,
                index_keys=index_keys,
                position_cos=cos,
                position_sin=sin,
                attention_mask=mask if isinstance(mask, mx.array) else None,
                cache=cache,
            )
        output = self.qsa(request).reshape(batch, length, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class Qwen4ExpPLELayer(nn.Module):
    """Projection/state side of layer-1 PLE backed by the SSD pool."""

    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        if layer_idx != 1:
            raise ValueError("Qwen3.8 Flash Next PLE weights are bound to layer 1")
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        self.context_len = args.ngram_size - 1
        self.short_conv_state_len = (args.ple_conv_kernel_size - 1) * args.ngram_size
        hc_hidden = args.hidden_size * args.hc_count
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)
        self.norm_key = ZeroCenteredRMSNorm(
            hc_hidden, args.rms_norm_eps, group_size=args.hidden_size
        )
        self.norm_query = ZeroCenteredRMSNorm(
            hc_hidden, args.rms_norm_eps, group_size=args.hidden_size
        )
        self.norm_conv = ZeroCenteredRMSNorm(
            hc_hidden, args.rms_norm_eps, group_size=args.hidden_size
        )
        self.conv1d = nn.Conv1d(
            hc_hidden,
            hc_hidden,
            args.ple_conv_kernel_size,
            groups=hc_hidden,
            dilation=args.ngram_size,
            bias=False,
        )
        self._pool: Qwen4ExpPLESSDPool | None = None

    def _get_pool(self) -> Qwen4ExpPLESSDPool:
        if self._pool is None:
            from omlx.patches.qwen4_exp import get_model_dir

            model_dir = get_model_dir()
            if model_dir is None:
                raise RuntimeError("qwen4_exp PLE has no bound checkpoint directory")
            self._pool = Qwen4ExpPLESSDPool(model_dir)
        return self._pool

    def __call__(self, hidden_states, input_ids, cache=None, mask=None):
        import numpy as np

        mx.eval(input_ids)
        token_ids = np.asarray(input_ids, dtype=np.int64)
        previous = None
        if cache is not None and cache[3] is not None:
            mx.eval(cache[3])
            previous = np.asarray(cache[3], dtype=np.int64)
        pool = self._get_pool()
        if pool.layout.table_dtype == PLE_MLX_Q8_DTYPE:
            embedding = mx.array(pool.lookup(token_ids, previous_context=previous))
        else:
            raw_embedding = pool.lookup_raw(token_ids, previous_context=previous)
            embedding = mx.array(raw_embedding)
            if raw_embedding.dtype == np.dtype("uint16"):
                embedding = embedding.view(mx.bfloat16)
        embedding = embedding.astype(hidden_states.dtype)

        history = (
            token_ids
            if previous is None
            else np.concatenate((previous, token_ids), axis=1)
        )
        next_context = history[:, -self.context_len :]
        if cache is not None:
            cache[3] = mx.array(next_context)

        key = self.norm_key(self.key_proj(embedding)).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        value = self.value_proj(embedding)
        query = self.norm_query(hidden_states).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        gate = mx.sum(key * query, axis=-1, keepdims=True) / (self.hidden_size**0.5)
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*hidden_states.shape)
        normalized = self.norm_conv(gated)
        if mask is not None:
            normalized = mx.where(mask[..., None], normalized, 0)
            gated = mx.where(mask[..., None], gated, 0)

        if cache is not None and cache[2] is not None:
            state = cache[2]
        else:
            state = mx.zeros(
                (
                    *normalized.shape[:-2],
                    self.short_conv_state_len,
                    normalized.shape[-1],
                ),
                dtype=normalized.dtype,
            )
        conv_input = mx.concatenate((state, normalized), axis=-2)
        if cache is not None:
            cache[2] = mx.contiguous(conv_input[..., -self.short_conv_state_len :, :])
        return gated + nn.silu(self.conv1d(conv_input))


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = args.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = Qwen4ExpGatedDeltaNet(args, layer_idx)
        else:
            self.self_attn = Qwen4ExpAttention(args, layer_idx)
        self.mlp = Qwen4ExpSparseMoeBlock(args)
        self.ple = Qwen4ExpPLELayer(args, layer_idx) if layer_idx == 1 else None
        self.attn_hyper_connection = Qwen4ExpGatedResidual(args)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(args)
        self.hc_count = args.hc_count

    def __call__(
        self,
        hidden_states,
        mask=None,
        cache=None,
        input_ids=None,
        n_confirmed: int = 0,
    ):
        if self.ple is not None:
            if input_ids is None:
                raise ValueError("Qwen4-Exp PLE requires original input_ids")
            if n_confirmed and cache is not None:
                cache._qwen4_ple_rollback_state = (cache[2], cache[3])
                cache._qwen4_ple_draft_stash = (hidden_states, input_ids)
            hidden_states = hidden_states + self.ple(
                hidden_states, input_ids, cache=cache, mask=mask
            )

        mixed, residual, injection = self.attn_hyper_connection(hidden_states)
        if self.is_linear:
            block = self.linear_attn(
                mixed,
                mask=mask,
                cache=cache,
                n_confirmed=n_confirmed,
            )
        else:
            block = self.self_attn(mixed, mask=mask, cache=cache)
        hidden_states = expand_hyper_residual(
            block, residual, injection, hc_count=self.hc_count
        )

        mixed, residual, injection = self.mlp_hyper_connection(hidden_states)
        block = self.mlp(mixed)
        return expand_hyper_residual(block, residual, injection, hc_count=self.hc_count)


class Qwen4ExpMTPDecoderLayer(nn.Module):
    """Official one-layer MTP block: QSA + MoE, no PLE."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self._omlx_qwen4_exp_mtp_safe = True
        self.layer_type = "full_attention"
        self.is_linear = False
        # Reuse the published QSA geometry and an existing QSA layer index;
        # the MTP tensor path is independent of the backbone layer number.
        self.self_attn = Qwen4ExpAttention(args, args.qsa_layer_indices[0])
        self.mlp = Qwen4ExpSparseMoeBlock(args)
        self.attn_hyper_connection = Qwen4ExpGatedResidual(args)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(args)
        self.hc_count = args.hc_count

    def __call__(self, hidden_states, mask=None, cache=None, input_ids=None):
        del input_ids
        mixed, residual, injection = self.attn_hyper_connection(hidden_states)
        block = self.self_attn(mixed, mask=mask, cache=cache)
        hidden_states = expand_hyper_residual(
            block, residual, injection, hc_count=self.hc_count
        )
        mixed, residual, injection = self.mlp_hyper_connection(hidden_states)
        block = self.mlp(mixed)
        return expand_hyper_residual(block, residual, injection, hc_count=self.hc_count)


def build_mtp_decoder_layer(args: TextModelArgs):
    """Factory consumed by the isolated Qwen4-Exp MTP patch."""

    return Qwen4ExpMTPDecoderLayer(args)


def make_qsa_cache():
    """Batch-compatible QSA cache: main K/V plus index/position state."""

    return CacheList(KVCache(), KVCache())


class Qwen4ExpTextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, index) for index in range(args.num_hidden_layers)
        ]
        self.hyper_connection_mixer = Qwen4ExpHyperConnectionMixer(args)
        self.ssm_idx = 0
        self.qsa_idx = args.qsa_layer_indices[0]

    def pipeline(self, group):
        super().pipeline(group)
        self.ssm_idx = next(
            (idx for idx, layer in enumerate(self.pipeline_layers) if layer.is_linear),
            None,
        )
        self.qsa_idx = next(
            (
                idx
                for idx, layer in enumerate(self.pipeline_layers)
                if not layer.is_linear
            ),
            None,
        )

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        *,
        return_hyper: bool = False,
        n_confirmed: int = 0,
    ):
        hidden = (
            self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        )
        hidden = mx.repeat(hidden, self.args.hc_count, axis=-1)
        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        ssm_mask = (
            create_ssm_mask(hidden, cache[self.ssm_idx])
            if self.ssm_idx is not None
            else None
        )
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        if pipeline_rank < pipeline_size - 1:
            hidden = mx.distributed.recv_like(hidden, pipeline_rank + 1)

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            mask = ssm_mask if layer.is_linear else None
            hidden = layer(
                hidden,
                mask=mask,
                cache=layer_cache,
                input_ids=inputs,
                n_confirmed=n_confirmed,
            )

        if pipeline_rank != 0:
            hidden = mx.distributed.send(hidden, (pipeline_rank - 1) % pipeline_size)
        if pipeline_size > 1:
            hidden = mx.distributed.all_gather(hidden)[: hidden.shape[0]]
        return hidden if return_hyper else self.hyper_connection_mixer(hidden)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
        skip_lm_head: bool = False,
    ):
        hidden = self.model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hyper=return_hidden,
            n_confirmed=n_confirmed,
        )
        logits_source = (
            self.model.hyper_connection_mixer(hidden) if return_hidden else hidden
        )
        if skip_lm_head:
            return (None, hidden) if return_hidden else None
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(logits_source)
        else:
            logits = self.lm_head(logits_source)
        return (logits, hidden) if return_hidden else logits

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [
            ArraysCache(size=4 if layer.ple is not None else 2)
            if layer.is_linear
            else make_qsa_cache()
            for layer in self.layers
        ]

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Keep the confirmed token and discard rejected QSA/GDN/PLE state."""

        layers = self.model.pipeline_layers
        if len(cache) != len(layers):
            return False
        trim_n = int(num_drafts) - int(accepted)
        if trim_n <= 0:
            return True
        keep = 1 + int(accepted)

        for layer, layer_cache in zip(layers, cache):
            if layer.is_linear:
                if getattr(layer_cache, "rollback_state", None) is None:
                    return False
                if getattr(layer_cache, "_mtp_draft_stash", None) is None:
                    return False
                if layer.ple is not None and (
                    getattr(layer_cache, "_qwen4_ple_rollback_state", None) is None
                    or getattr(layer_cache, "_qwen4_ple_draft_stash", None) is None
                ):
                    return False
            elif not (
                hasattr(layer_cache, "is_trimmable") and layer_cache.is_trimmable()
            ):
                return False

        for layer, layer_cache in zip(layers, cache):
            if layer.is_linear:
                conv_initial, recurrent_initial = layer_cache.rollback_state
                qkv, a, b = layer_cache._mtp_draft_stash
                _, conv_kept, recurrent_kept = layer.linear_attn._process_chunk(
                    qkv[:, :keep],
                    a[:, :keep],
                    b[:, :keep],
                    conv_initial,
                    recurrent_initial,
                    None,
                )
                layer_cache[0] = conv_kept
                layer_cache[1] = recurrent_kept
                layer_cache.rollback_state = None
                layer_cache._mtp_draft_stash = None

                if layer.ple is not None:
                    ple_conv, ple_context = layer_cache._qwen4_ple_rollback_state
                    ple_hidden, ple_inputs = layer_cache._qwen4_ple_draft_stash
                    layer_cache[2] = ple_conv
                    layer_cache[3] = ple_context
                    layer.ple(
                        ple_hidden[:, :keep],
                        ple_inputs[:, :keep],
                        cache=layer_cache,
                    )
                    layer_cache._qwen4_ple_rollback_state = None
                    layer_cache._qwen4_ple_draft_stash = None
            else:
                layer_cache.trim(trim_n)
        return True

    def sanitize(self, weights):
        sanitized = sanitize_moe_weights(weights)
        if self.args.tie_word_embeddings:
            sanitized.pop("language_model.lm_head.weight", None)
        for key, value in list(sanitized.items()):
            if "conv1d.weight" in key and getattr(value, "shape", (1,))[-1] != 1:
                sanitized[key] = value.moveaxis(2, 1)
        return sanitized

    @property
    def quant_predicate(self):
        def predicate(path, _module):
            if path.endswith(("A_log", "dt_bias")) or ".ple.ple_embedding" in path:
                return False
            if (
                path.endswith(("mlp.gate", "shared_expert_gate"))
                or ".ple.conv1d" in path
            ):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        return lambda path: not path.endswith(("A_log", "dt_bias"))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_args = TextModelArgs.from_dict(args.text_config)
        self.language_model = TextModel(text_args)
        self._quantized_checkpoint = bool(args.quantization or args.quantization_config)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings: mx.array | None = None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
        skip_lm_head: bool = False,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
            n_confirmed=n_confirmed,
            skip_lm_head=skip_lm_head,
        )

    def mtp_forward(self, *args, **kwargs):
        if not hasattr(self.language_model, "mtp_forward"):
            raise RuntimeError("Qwen4-Exp MTP patch is not installed")
        return self.language_model.mtp_forward(*args, **kwargs)

    def make_mtp_cache(self):
        if not hasattr(self.language_model, "make_mtp_cache"):
            return []
        return self.language_model.make_mtp_cache()

    def mtp_partial_rollback(self, *args, **kwargs):
        return self.language_model.mtp_partial_rollback(*args, **kwargs)

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        # Validate and bind all 128 SSD-backed PLE ranges during load, before
        # ordinary MLX strict-loading discards the table tensors.
        ple_layer = self.language_model.model.layers[1].ple
        if ple_layer is None:
            raise ValueError("Qwen4-Exp layer 1 must own the PLE pool")
        ple_layer._get_pool()
        if not self._quantized_checkpoint:
            self._validate_source_layout(weights)
        sanitized = {}
        ple_prefix = PLE_PREFIX + "."
        for key, value in weights.items():
            if key.startswith(("model.visual.", "vision_tower.")):
                continue
            if key.startswith(ple_prefix):
                # The SSD pool validates and reads these tensors directly
                # from their source safetensor ranges.
                continue
            if key.startswith("model.language_model."):
                key = "language_model.model." + key[len("model.language_model.") :]
            elif key.startswith("mtp.") or not key.startswith("language_model."):
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    def _validate_source_layout(self, weights):
        """Prove the published BF16 graph before remapping any tensor."""

        for index, layer_type in enumerate(self.language_model.args.layer_types):
            prefix = f"model.language_model.layers.{index}"
            if layer_type == "linear_attention":
                validate_gdn_weight_layout(weights, f"{prefix}.linear_attn")
            else:
                validate_qsa_weights(weights, prefix=f"{prefix}.self_attn")
            validate_hc_weight_layout(weights, f"{prefix}.attn_hyper_connection")
            validate_hc_weight_layout(weights, f"{prefix}.mlp_hyper_connection")
            validate_moe_weight_layout(weights, f"{prefix}.mlp")
        validate_hc_weight_layout(
            weights,
            "model.language_model.hyper_connection_mixer",
            use_combine=False,
        )
        ple_expected = {
            "model.language_model.layers.1.ple.key_proj.weight": (10_240, 2_560),
            "model.language_model.layers.1.ple.value_proj.weight": (2_560, 2_560),
            "model.language_model.layers.1.ple.norm_key.weight": (10_240,),
            "model.language_model.layers.1.ple.norm_query.weight": (10_240,),
            "model.language_model.layers.1.ple.norm_conv.weight": (10_240,),
        }
        for key, expected in ple_expected.items():
            if (
                key not in weights
                or tuple(getattr(weights[key], "shape", ())) != expected
            ):
                actual = tuple(getattr(weights.get(key), "shape", ()))
                raise ValueError(
                    f"Qwen4-Exp PLE tensor {key!r} must have shape "
                    f"{expected}, got {actual}"
                )
        conv_key = "model.language_model.layers.1.ple.conv1d.weight"
        conv_shape = tuple(getattr(weights.get(conv_key), "shape", ()))
        if conv_shape not in {(10_240, 1, 4), (10_240, 4, 1)}:
            raise ValueError(
                f"Qwen4-Exp PLE tensor {conv_key!r} has invalid shape {conv_shape}"
            )

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate


__all__ = [
    "DecoderLayer",
    "Model",
    "ModelArgs",
    "QWEN4_EXP_STRICT_QSA",
    "Qwen4ExpAttention",
    "Qwen4ExpPLELayer",
    "Qwen4ExpMTPDecoderLayer",
    "Qwen4ExpTextModel",
    "TextModel",
    "TextModelArgs",
    "build_mtp_decoder_layer",
    "create_attention_mask",
    "make_qsa_cache",
]

# Transformers-style spelling used by the MTP integration layer.
Qwen4ExpTextDecoderLayer = DecoderLayer
