// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
using namespace metal;

constant uint kRows = 6;
constant uint kBlocksPerRow = 512;
constant uint kUnionCapacity = 3072;
constant uint kTokensPerRow = 2051;
constant uint kBlockTokensPerRow = 2048;
constant uint kTailPerRow = 3;
constant uint kBlockTokenCapacity = 12288;
constant uint kTokenCapacity = 12306;
constant uint kInvalid = 0xffffffffu;
constant uint kStatusNonChronological = 1u;
constant uint kStatusNonCausal = 2u;
constant uint kStatusTokenOverflow = 4u;

struct Qwen4CompactStageParams {
  uint q_offset;
  uint key_tokens;
};

struct Qwen4CompactGatherParams {
  uint key_tokens;
  uint reserved;
  ulong key_head_stride;
  ulong key_token_stride;
  ulong value_head_stride;
  ulong value_token_stride;
};
static_assert(sizeof(Qwen4CompactStageParams) == 8,
              "compact-stage parameter ABI changed");
static_assert(sizeof(Qwen4CompactGatherParams) == 40,
              "compact-gather parameter ABI changed");

// Deterministic six-way merge over six already-sorted, unique selector rows.
// Lane zero performs the bounded merge (at most 3,072 iterations); all lanes
// initialize the fixed-capacity outputs in parallel first.  There is no host
// read or dynamic allocation, and the 3,072 capacity is the mathematical
// maximum, so no selector membership can be truncated.
[[kernel]] void qwen4_qsa_compact_stage_plan(
    const device uint *selected [[buffer(0)]],
    device uint *union_blocks [[buffer(1)]],
    device uint *union_count [[buffer(2)]],
    device uint *row_token_slots [[buffer(3)]],
    device uint *source_tokens [[buffer(4)]],
    device uint *validation_status [[buffer(5)]],
    constant Qwen4CompactStageParams &params [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint status;
  for (uint index = lane; index < kUnionCapacity; index += 32) {
    union_blocks[index] = kInvalid;
  }
  for (uint index = lane; index < kRows * kTokensPerRow; index += 32) {
    row_token_slots[index] = kInvalid;
  }
  for (uint index = lane; index < kTokenCapacity; index += 32) {
    source_tokens[index] = kInvalid;
  }
  if (lane == 0) {
    union_count[0] = 0;
    validation_status[0] = 0;
    atomic_store_explicit(&status, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_device);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Fail closed before any block-to-token narrowing.  The union merge is
  // exact only for the sorted, unique, fully causal rows produced by Qwen's
  // long-context selector.  A nonzero status leaves every map sentinel-filled
  // and union_count zero so no invalid plan can be consumed accidentally.
  for (uint index = lane; index < kRows * kBlocksPerRow; index += 32) {
    const uint row = index / kBlocksPerRow;
    const uint column = index - row * kBlocksPerRow;
    const uint raw_block = selected[index];
    uint local_status = 0;
    if (column > 0 && raw_block <= selected[index - 1]) {
      local_status |= kStatusNonChronological;
    }
    const uint q_abs = params.q_offset + row;
    const uint complete_blocks = (q_abs + 1) / 4;
    const ulong source_last = ulong(raw_block) * ulong(4) + ulong(3);
    if (raw_block >= complete_blocks ||
        source_last >= ulong(params.key_tokens)) {
      local_status |= kStatusNonCausal;
    }
    if (source_last > ulong(kInvalid)) {
      local_status |= kStatusTokenOverflow;
    }
    if (local_status != 0) {
      atomic_fetch_or_explicit(
          &status, local_status, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint checked_status =
      atomic_load_explicit(&status, memory_order_relaxed);
  if (lane == 0) {
    validation_status[0] = checked_status;
  }

  if (lane != 0 || checked_status != 0) {
    return;
  }

  uint cursors[kRows] = {0, 0, 0, 0, 0, 0};
  uint count = 0;
  while (count < kUnionCapacity) {
    uint next_block = kInvalid;
    for (uint row = 0; row < kRows; ++row) {
      if (cursors[row] < kBlocksPerRow) {
        next_block = min(
            next_block,
            selected[row * kBlocksPerRow + cursors[row]]);
      }
    }
    if (next_block == kInvalid) {
      break;
    }

    union_blocks[count] = next_block;
    for (uint token = 0; token < 4; ++token) {
      const ulong source = ulong(next_block) * ulong(4) + ulong(token);
      source_tokens[count * 4 + token] = uint(source);
    }
    for (uint row = 0; row < kRows; ++row) {
      if (cursors[row] < kBlocksPerRow &&
          selected[row * kBlocksPerRow + cursors[row]] == next_block) {
        const uint column = cursors[row];
        for (uint token = 0; token < 4; ++token) {
          row_token_slots[row * kTokensPerRow + column * 4 + token] =
              count * 4 + token;
        }
        cursors[row] = column + 1;
      }
    }
    ++count;
  }
  union_count[0] = count;

  for (uint row = 0; row < kRows; ++row) {
    const uint q_abs = params.q_offset + row;
    const uint complete_blocks = (q_abs + 1) / 4;
    const uint tail_count = (q_abs + 1) - complete_blocks * 4;
    for (uint tail = 0; tail < tail_count; ++tail) {
      const uint stage_slot = kBlockTokenCapacity + row * kTailPerRow + tail;
      const uint source = complete_blocks * 4 + tail;
      if (source < params.key_tokens) {
        source_tokens[stage_slot] = source;
        row_token_slots[
            row * kTokensPerRow + kBlockTokensPerRow + tail] = stage_slot;
      }
    }
  }
}

template <typename T>
[[kernel]] void qwen4_qsa_compact_stage_gather(
    const device T *keys [[buffer(0)]],
    const device T *values [[buffer(1)]],
    const device uint *source_tokens [[buffer(2)]],
    device T *staged_keys [[buffer(3)]],
    device T *staged_values [[buffer(4)]],
    constant Qwen4CompactGatherParams &params [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  constexpr uint kDim = 256;
  constexpr uint kWordsPerRow = kDim / 8;
  constexpr uint kHeads = 2;
  const uint words = kHeads * kTokenCapacity * kWordsPerRow;
  if (index >= words) {
    return;
  }
  const uint d8 = index % kWordsPerRow;
  const uint token_and_head = index / kWordsPerRow;
  const uint stage_token = token_and_head % kTokenCapacity;
  const uint head = token_and_head / kTokenCapacity;
  const uint source_token = source_tokens[stage_token];

  uint4 key_word = uint4(0);
  uint4 value_word = uint4(0);
  if (source_token < params.key_tokens) {
    const ulong key_source_offset =
        ulong(head) * params.key_head_stride +
        ulong(source_token) * params.key_token_stride;
    const ulong value_source_offset =
        ulong(head) * params.value_head_stride +
        ulong(source_token) * params.value_token_stride;
    key_word = *((const device uint4 *)(keys + key_source_offset) + d8);
    value_word = *((const device uint4 *)(values + value_source_offset) + d8);
  }
  const ulong stage_offset =
      (ulong(head) * ulong(kTokenCapacity) + ulong(stage_token)) * kDim;
  *((device uint4 *)(staged_keys + stage_offset) + d8) = key_word;
  *((device uint4 *)(staged_values + stage_offset) + d8) = value_word;
}

template [[host_name("qwen4_qsa_compact_stage_gather_float16")]]
[[kernel]] decltype(qwen4_qsa_compact_stage_gather<float16_t>)
    qwen4_qsa_compact_stage_gather<float16_t>;

template [[host_name("qwen4_qsa_compact_stage_gather_bfloat16")]]
[[kernel]] decltype(qwen4_qsa_compact_stage_gather<bfloat16_t>)
    qwen4_qsa_compact_stage_gather<bfloat16_t>;
