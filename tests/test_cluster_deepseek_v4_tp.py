# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 tensor parallelism: planner eligibility and the native shard.

The planner used to refuse every TP degree for DeepSeek-V4: its config
declares num_key_value_heads=1, which cannot be halved. But this
architecture's cache is one shared KV row plus pooled compressor/indexer
state per layer — none of it per-head and none of it touched by the vendored
``shard()`` — so the KV head count does not bound the split. These tests pin
the planner change and prove the native shard numerically reproduces the
whole model without loading real weights.
"""

import json
import struct

import pytest

from omlx.cluster.autoconfigure import candidate_tensor_parallel_sizes
from omlx.cluster.planner import (
    ModelLayout,
    _kv_bytes_per_token_per_layer,
    _kv_cache_replicated_across_tp,
    _tensor_parallel_divisors,
    inspect_safetensors_layout,
)


@pytest.fixture(scope="module")
def dsv4():
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    import mlx_lm.models.deepseek_v4 as module

    return module


def _deepseek_v4_config() -> dict:
    """The config.json shape the planner reads for DeepSeek-V4."""

    return {
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "o_groups": 8,
        "o_lora_rank": 1024,
    }


def _tiny_args(dsv4, compress_ratios=(0, 0)):
    """A DeepSeek-V4 small enough to run on the GPU in a unit test.

    Four attention heads over two o_groups mirror the checkpoint's 64/8:
    shard() splits wq_b per o_group, so heads-per-group must halve cleanly.
    """

    return dsv4.ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 64,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_hidden_layers": len(compress_ratios),
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "num_hash_layers": 0,
            "q_lora_rank": 4,
            "qk_rope_head_dim": 4,
            "head_dim": 4,
            "o_groups": 2,
            "o_lora_rank": 4,
            "index_n_heads": 2,
            "index_head_dim": 4,
            "index_topk": 2,
            "compress_ratios": list(compress_ratios),
            "num_nextn_predict_layers": 0,
        }
    )


class _FakeGroup:
    """Just enough group for weight slicing; forwards stub the collectives."""

    def __init__(self, rank: int, size: int = 2):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _write_model_dir(root, config: dict, layer_count: int) -> None:
    (root / "config.json").write_text(json.dumps(config))
    header = {}
    offset = 0
    for index in range(layer_count):
        header[f"model.layers.{index}.attn.wq_a.weight"] = {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [offset, offset + 2],
        }
        offset += 2
    header["model.embed_tokens.weight"] = {
        "dtype": "BF16",
        "shape": [1],
        "data_offsets": [offset, offset + 2],
    }
    blob = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\x00" * (offset + 2)
    )


class TestPlannerEligibility:
    def test_kv_cache_is_replicated_for_deepseek_v4(self):
        assert _kv_cache_replicated_across_tp(_deepseek_v4_config()) is True

    def test_kv_heads_do_not_bound_the_tp_degree(self):
        assert _tensor_parallel_divisors(_deepseek_v4_config()) == (64,)

    def test_kv_byte_reservation_is_unchanged(self):
        # One shared 512-wide KV row per layer: the standard formula prices
        # it as num_key_value_heads=1 * head_dim=512 * 2 * 2 bytes, exactly
        # as before the replication flag covered this family.
        assert _kv_bytes_per_token_per_layer(_deepseek_v4_config()) == 2048

    def test_dense_families_are_unchanged(self):
        llama = {
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
        }
        assert _kv_cache_replicated_across_tp(llama) is False
        assert _tensor_parallel_divisors(llama) == (32, 8)

        qwen3_next = {
            "model_type": "qwen3_next",
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
        }
        assert _kv_cache_replicated_across_tp(qwen3_next) is False
        assert _tensor_parallel_divisors(qwen3_next) == (16, 2, 32)

    def test_mla_latent_accounting_is_unchanged(self):
        mla = {
            "model_type": "deepseek_v3",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "num_attention_heads": 128,
        }
        assert _kv_cache_replicated_across_tp(mla) is True
        # num_key_value_heads defaults to the attention head count, so the
        # divisor tuple was already collapsed for MLA configs.
        assert _tensor_parallel_divisors(mla) == (128,)
        assert _kv_bytes_per_token_per_layer(mla) == (512 + 64) * 2


class TestPlannerLayout:
    def test_tp2_is_offered_for_the_deepseek_v4_layout(self, dsv4, tmp_path):
        _write_model_dir(tmp_path, _deepseek_v4_config(), layer_count=43)

        layout = inspect_safetensors_layout(str(tmp_path))

        assert layout.layer_count == 43
        assert layout.tensor_parallel_heads == 64
        assert layout.tensor_parallel_divisors == (64,)
        assert layout.kv_replicated_across_tp is True
        assert layout.supports_tensor_parallel is True
        assert layout.kv_bytes_per_token_per_layer == 2048
        assert candidate_tensor_parallel_sizes(layout, 2) == (2, 1)
        assert (
            ModelLayout.from_dict(layout.to_dict()).kv_replicated_across_tp
            is True
        )

    def test_gqa_layout_still_bounds_the_split_by_kv_heads(self, tmp_path):
        config = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
        _write_model_dir(tmp_path, config, layer_count=2)

        layout = inspect_safetensors_layout(str(tmp_path))

        assert layout.tensor_parallel_divisors == (32, 8)
        assert layout.kv_replicated_across_tp is False
        assert candidate_tensor_parallel_sizes(layout, 4) == (4, 1)
        assert candidate_tensor_parallel_sizes(layout, 3) == (1,)


class TestRankZeroLogitsContract:
    def test_real_model_satisfies_the_runtime_checker(self, dsv4):
        from omlx.cluster.runtime_optimizations import _supports_rank_zero_logits

        model = dsv4.Model(_tiny_args(dsv4))
        supported, vocab_size, reason = _supports_rank_zero_logits(model)

        assert supported is True, reason
        assert vocab_size == 64

    def test_skip_logits_returns_backbone_hidden_without_lm_head(self, dsv4):
        import mlx.core as mx

        model = dsv4.Model(_tiny_args(dsv4))
        head = model.lm_head
        head_calls = []

        class HeadSpy:
            def __call__(self, value):
                head_calls.append(value.shape)
                return head(value)

        model.lm_head = HeadSpy()
        inputs = mx.array([[1, 2, 3]])

        hidden = model(inputs, skip_logits=True)
        mx.eval(hidden)
        assert head_calls == []
        assert hidden.shape == (1, 3, model.args.hidden_size)

        logits = model(inputs)
        mx.eval(logits)
        assert head_calls == [(1, 3, model.args.hidden_size)]
        assert logits.shape == (1, 3, model.args.vocab_size)

    def test_mtp_patched_call_keeps_the_contract(self, dsv4):
        """The MTP patch always replaces Model.__call__ in real serving (it
        owns sanitize correctness), so the runtime checker inspects *its*
        signature — it must accept skip_logits too."""

        import mlx.core as mx

        from omlx.cluster.runtime_optimizations import _supports_rank_zero_logits
        from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

        apply_mlx_lm_mtp_patch()
        assert getattr(dsv4.Model.__call__, "_omlx_mtp_call_marker", False) is True

        model = dsv4.Model(_tiny_args(dsv4))
        supported, vocab_size, reason = _supports_rank_zero_logits(model)
        assert supported is True, reason
        assert vocab_size == 64

        inputs = mx.array([[1, 2, 3]])
        hidden = model(inputs, skip_logits=True)
        mx.eval(hidden)
        assert hidden.shape == (1, 3, model.args.hidden_size)
        logits = model(inputs)
        mx.eval(logits)
        assert logits.shape == (1, 3, model.args.vocab_size)


class TestNativeShard:
    def test_native_shard_passes_the_layer_local_gate(self, dsv4):
        from omlx.cluster.tensor_strategies import native_shard_is_layer_local

        supported, reason = native_shard_is_layer_local(dsv4.Model.shard)
        assert supported is True, reason

    def _sharded_pair(self, dsv4, args):
        import mlx.core as mx

        from omlx.cluster.tensor_strategies import apply_tensor_strategy

        mx.random.seed(7)
        reference = dsv4.Model(args)
        sharded = []
        for rank in (0, 1):
            mx.random.seed(7)
            model = dsv4.Model(args)
            strategy = apply_tensor_strategy(
                model, _FakeGroup(rank), mx_module=mx
            )
            assert strategy == "native"
            sharded.append(model)
        return reference, sharded

    def test_every_projection_is_sharded_or_replicated_as_designed(self, dsv4):
        args = _tiny_args(dsv4, compress_ratios=(0, 4, 128))
        reference, (rank0, _) = self._sharded_pair(dsv4, args)

        for full_layer, layer in zip(
            reference.model.layers, rank0.model.layers
        ):
            full_attn, attn = full_layer.attn, layer.attn
            assert attn.n_heads * 2 == full_attn.n_heads
            assert attn.wq_b.weight.shape[0] * 2 == full_attn.wq_b.weight.shape[0]
            assert attn.wo_a.weight.shape[-1] * 2 == full_attn.wo_a.weight.shape[-1]
            assert attn.attn_sink.shape[0] * 2 == full_attn.attn_sink.shape[0]
            # The shared KV projection and the low-rank Q path are replicated.
            assert attn.wkv.weight.shape == full_attn.wkv.weight.shape
            assert attn.wq_a.weight.shape == full_attn.wq_a.weight.shape
            # The forward reduces through the group both places.
            assert attn.sharding_group is not None
            assert layer.ffn.sharding_group is not None

            full_ffn, ffn = full_layer.ffn, layer.ffn
            assert (
                ffn.switch_mlp.gate_proj.weight.shape[1] * 2
                == full_ffn.switch_mlp.gate_proj.weight.shape[1]
            )
            assert (
                ffn.switch_mlp.down_proj.weight.shape[-1] * 2
                == full_ffn.switch_mlp.down_proj.weight.shape[-1]
            )
            assert (
                ffn.shared_experts.gate_proj.weight.shape[0] * 2
                == full_ffn.shared_experts.gate_proj.weight.shape[0]
            )
            assert (
                ffn.shared_experts.down_proj.weight.shape[-1] * 2
                == full_ffn.shared_experts.down_proj.weight.shape[-1]
            )

        # The ratio-4 layer's indexer is replicated whole: its head count and
        # projections are never divided, and neither is its compressor pool.
        full_sparse = reference.model.layers[1].attn
        sparse = rank0.model.layers[1].attn
        assert sparse.indexer.n_heads == full_sparse.indexer.n_heads
        assert sparse.indexer.wq_b.weight.shape == full_sparse.indexer.wq_b.weight.shape
        assert (
            sparse.indexer.weights_proj.weight.shape
            == full_sparse.indexer.weights_proj.weight.shape
        )
        assert (
            sparse.indexer.compressor.wkv.weight.shape
            == full_sparse.indexer.compressor.wkv.weight.shape
        )
        assert sparse.compressor.wkv.weight.shape == full_sparse.compressor.wkv.weight.shape

    def test_attention_partials_sum_to_the_full_layer(self, dsv4, monkeypatch):
        import mlx.core as mx

        from omlx.cluster.tensor_strategies import apply_tensor_strategy

        args = _tiny_args(dsv4)
        models = []
        for _ in range(3):
            mx.random.seed(7)
            model = dsv4.Model(args)
            # Distinct per-head sinks: the checkpoint's are not zeros, and a
            # sink split that ignores the per-o_group head layout is silently
            # wrong. They must be in place before shard() slices them.
            for layer in model.model.layers:
                layer.attn.attn_sink = mx.arange(
                    args.num_attention_heads, dtype=mx.float32
                )
            models.append(model)
        reference, rank0, rank1 = models
        apply_tensor_strategy(rank0, _FakeGroup(0), mx_module=mx)
        apply_tensor_strategy(rank1, _FakeGroup(1), mx_module=mx)

        collectives = []

        def identity_all_sum(value, group=None):
            collectives.append(group)
            return value

        monkeypatch.setattr(mx.distributed, "all_sum", identity_all_sum)

        x = mx.random.normal((1, 4, args.hidden_size))
        expected = reference.model.layers[0].attn(x)
        partial0 = rank0.model.layers[0].attn(x)
        partial1 = rank1.model.layers[0].attn(x)
        actual = partial0 + partial1
        mx.eval(expected, actual)

        assert float(mx.max(mx.abs(actual - expected)).item()) < 1e-5
        # Exactly one output collective per sharded attention forward, on the
        # rank's own group — never a global default group.
        assert collectives == [rank0.model.layers[0].attn.sharding_group] + [
            rank1.model.layers[0].attn.sharding_group
        ]

    def test_moe_partials_sum_to_the_full_layer(self, dsv4, monkeypatch):
        import mlx.core as mx

        args = _tiny_args(dsv4)
        reference, (rank0, rank1) = self._sharded_pair(dsv4, args)

        collectives = []

        def identity_all_sum(value, group=None):
            collectives.append(group)
            return value

        monkeypatch.setattr(mx.distributed, "all_sum", identity_all_sum)

        x = mx.random.normal((1, 4, args.hidden_size))
        input_ids = mx.array([[1, 2, 3, 4]])
        expected = reference.model.layers[0].ffn(x, input_ids)
        partial0 = rank0.model.layers[0].ffn(x, input_ids)
        partial1 = rank1.model.layers[0].ffn(x, input_ids)
        actual = partial0 + partial1
        mx.eval(expected, actual)

        assert float(mx.max(mx.abs(actual - expected)).item()) < 1e-5
        assert collectives == [rank0.model.layers[0].ffn.sharding_group] + [
            rank1.model.layers[0].ffn.sharding_group
        ]

    def test_mxfp4_switch_linear_shards_dequantize_consistently(self):
        """The expert projections ship mxfp4; the packed weight and its
        e8m0 scales must split along the same axis on every rank."""

        import mlx.core as mx
        from mlx.nn.layers.distributed import shard_inplace

        from omlx.patches.deepseek_v4.switch_layers import QuantizedSwitchLinear

        def seeded_projection():
            mx.random.seed(7)
            return QuantizedSwitchLinear(
                64,
                64,
                num_experts=2,
                bias=False,
                group_size=32,
                bits=4,
                mode="mxfp4",
            )

        x = mx.random.normal((4, 1, 64))
        indices = mx.array([0, 1, 0, 1])

        # all-to-sharded (gate/up): each rank produces half the output width.
        full = seeded_projection()
        ranks = [seeded_projection(), seeded_projection()]
        for rank, projection in enumerate(ranks):
            shard_inplace(projection, "all-to-sharded", group=_FakeGroup(rank))
            assert projection.weight.shape[1] * 2 == full.weight.shape[1]
            assert projection.scales.shape[1] * 2 == full.scales.shape[1]
        expected = full(x, indices)
        halves = [projection(x, indices) for projection in ranks]
        actual = mx.concatenate(halves, axis=-1)
        mx.eval(expected, actual)
        assert float(mx.max(mx.abs(actual - expected)).item()) < 1e-5

        # sharded-to-all (down): each rank consumes half the input width and
        # the collective sums the partial products.
        full = seeded_projection()
        ranks = [seeded_projection(), seeded_projection()]
        for rank, projection in enumerate(ranks):
            shard_inplace(projection, "sharded-to-all", group=_FakeGroup(rank))
            assert projection.weight.shape[-1] * 2 == full.weight.shape[-1]
            assert projection.scales.shape[-1] * 2 == full.scales.shape[-1]
        expected = full(x, indices)
        partials = [
            projection(x[..., rank * 32 : (rank + 1) * 32], indices)
            for rank, projection in enumerate(ranks)
        ]
        actual = partials[0] + partials[1]
        mx.eval(expected, actual)
        assert float(mx.max(mx.abs(actual - expected)).item()) < 1e-5
