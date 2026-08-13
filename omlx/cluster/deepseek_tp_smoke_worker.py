# SPDX-License-Identifier: Apache-2.0
"""Tiny two-rank tensor-parallel DeepSeek-V4 forward used by the TP diagnostic.

Sharding bugs do not crash — a dropped all-reduce or a mis-sliced projection
still emits fluent text. Each rank therefore runs the same tiny model twice:
once whole (the reference) and once sharded across the real two-member ring
through ``apply_tensor_strategy``, the exact path the cluster worker takes.
The sharded forward must pick the same next token and stay within float noise
of the reference logits.
"""

from __future__ import annotations

import json
import sys

# One of each attention flavor the checkpoint ships: an uncompressed local
# layer, a ratio-4 sparse layer with the (replicated, never sharded) indexer,
# and a ratio-128 compressed layer. Four attention heads over two o_groups
# halve cleanly: shard() splits wq_b per o_group, so heads-per-group — like
# the checkpoint's 64/8 — must stay divisible by the rank count.
_CONFIG = {
    "model_type": "deepseek_v4",
    "vocab_size": 64,
    "hidden_size": 8,
    "intermediate_size": 16,
    "moe_intermediate_size": 4,
    "num_hidden_layers": 3,
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
    "compress_ratios": [0, 4, 128],
}


def main() -> int:
    try:
        from omlx._torch_stub import install as install_torch_stub

        install_torch_stub()

        import mlx.core as mx

        from omlx.cluster.tensor_strategies import apply_tensor_strategy
        from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

        group = mx.distributed.init(backend="ring", strict=True)
        if group.size() != 2:
            raise RuntimeError("DeepSeek TP smoke requires exactly two ranks")

        apply_deepseek_v4_patch()
        import mlx_lm.models.deepseek_v4 as dsv4

        mx.random.seed(7)
        model = dsv4.Model(dsv4.ModelArgs.from_dict(_CONFIG))
        inputs = mx.array([[1, 2, 3, 4]])

        reference = model(inputs)
        mx.eval(reference)
        reference_token = int(mx.argmax(reference[0, -1]).item())

        strategy = apply_tensor_strategy(model, group, mx_module=mx)
        sharded = model(inputs)
        mx.eval(sharded)
        if sharded.shape != reference.shape:
            raise RuntimeError(
                f"sharded output shape {sharded.shape} != {reference.shape}"
            )
        max_abs_diff = float(
            mx.max(mx.abs(sharded - reference)).astype(mx.float32).item()
        )
        sharded_token = int(mx.argmax(sharded[0, -1]).item())
        if sharded_token != reference_token or max_abs_diff > 1e-3:
            raise RuntimeError(
                "tensor-parallel forward diverged from the whole-model "
                f"reference: token {sharded_token} != {reference_token}, "
                f"max abs diff {max_abs_diff}"
            )
        checksum = float(mx.sum(sharded.astype(mx.float32)).item())
        record = {
            "type": "deepseek_tp_result",
            "model_type": "deepseek_v4",
            "rank": group.rank(),
            "size": group.size(),
            "strategy": strategy,
            "layers": len(model.model.layers),
            "heads_per_rank": model.model.layers[0].attn.n_heads,
            "reference_token": reference_token,
            "sharded_token": sharded_token,
            "max_abs_diff": max_abs_diff,
            "checksum": checksum,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(
            f"oMLX DeepSeek TP smoke failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
