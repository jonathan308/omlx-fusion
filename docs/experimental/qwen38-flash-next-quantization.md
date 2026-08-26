# Qwen3.8 Flash Next: Fusion conversion and storage plan

Status: architecture bring-up; do not use this document to load the live DS4
deployment.

## Source artifacts

- Source of truth: `Qwen/Qwen3.8-Flash-Next` (BF16, 131 safetensor shards).
- The index reports approximately 360 GB decimal (335 GiB) of tensor data,
  including the PLE tables and the official depth-one MTP head.
- `Qwen/Qwen3.8-Flash-Next-FP8` is an NVIDIA-oriented comparison artifact. It
  is not an MLX 8-bit checkpoint and must not be relabelled as one.
- `RadixArk/Qwen3.8-Flash-Next-NVFP4` is a secondary interoperability target,
  not the source for validating the native architecture.

## Runtime layout

Qwen3.8 Flash Next has three independently managed pools:

1. Compute weights: embeddings, 48 decoder layers, MoE experts, QSA/GDN,
   hyper-connections, final norm, and language head.
2. PLE: 128 checkpoint shards under
   `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.*`.
   Fusion maps these from SSD and gathers only requested rows.
3. MTP: the official one-layer hybrid QSA+MoE head under `mtp.*`.

The PLE pool is not LongCat n-gram speculation and must not be folded into the
ordinary MLX model tree. The config's `ple_layer_ids=[2]` is one-indexed; the
weight path is decoder layer 1.

## Conversion matrix

| Artifact | Compute weights | PLE table | Purpose |
| --- | --- | --- | --- |
| Source | BF16 | BF16 | Validation and future conversions; disk only |
| Fusion Q8 | MLX 8-bit, BF16 activations | SSD-backed BF16 | First correctness/quality serving build |
| Fusion Q8 compact | MLX 8-bit, BF16 activations | SSD-backed affine Q8 | Optional lower-I/O/storage bake-off |
| Fusion Q4 bake-off | MLX 4-bit / MXFP4 / oQ4 | SSD-backed 8-bit, then validated 4-bit | Throughput and fit comparison |
| Official FP8 | Published FP8 layout | Published layout | Documentation/compatibility comparison only |
| RadixArk NVFP4 | Imported only after contract audit | Keep separately mapped | Later experimental compatibility path |

Do not quantize `A_log`, `dt_bias`, normalization weights, PLE hashing metadata,
or GDN/PLE state. Keep GDN and PLE convolutions at 8-bit or higher. The first
validated artifact keeps PLE BF16 and SSD-backed (about 102.4 GB decimal on
disk, not resident). Affine Q8 PLE is optional after it matches the BF16 lookup
reference. PLE must never go below 4-bit and a 4-bit table is accepted only
after hash-by-hash output comparison against BF16/8-bit.

## Acceptance gates

- Native `model_type=qwen4_exp`; never alias to qwen3_5 or qwen3_next.
- Exact config validation: 48 layers, 36 GDN + 12 QSA, QSA dimensions,
  512/10+1 MoE, four hyper-connection streams, 128 PLE shards, and MTP depth 1.
- Generation requires Fusion's real micro-block sparse QSA backend, which is
  included in the overlay. Dense SDPA and DeepSeek DSA are not accepted
  substitutes; contract/backend mismatches fail closed.
- The Q8 conversion must round-trip tokenizer metadata and preserve all
  `mtp.*` tensors.
- Compare logits and generated tokens on fixed prompts before performance
  benchmarking. Record memory, PLE hit/miss latency, prefill, decode, and MTP
  acceptance separately.
- Only transfer a validated serving artifact to the Studio; never convert or
  compile it while the live DS4 cluster is resident.
