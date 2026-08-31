// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

constexpr int kQwen4StageRows = 6;
constexpr int kQwen4StageBlocksPerRow = 512;
constexpr int kQwen4StageMinimumQOffset =
    kQwen4StageBlocksPerRow * 4 - 1;
constexpr int kQwen4StageUnionCapacity = 3072;
constexpr int kQwen4StageTokensPerRow = 2051;
constexpr int kQwen4StageTailSlots = 18;
constexpr int kQwen4StageTokenCapacity =
    kQwen4StageUnionCapacity * 4 + kQwen4StageTailSlots;

// Outputs: union block IDs [3072], union count [1], chronological explicit
// token slots [1,1,6,2051], source-token IDs [12306], and validation status
// [1]. Status zero is required before the maps may be consumed.
std::vector<mx::array> qwen4_qsa_compact_stage_plan(
    const mx::array &selected_blocks, int q_offset, int key_tokens,
    mx::StreamOrDevice s = {});

// Outputs fixed-capacity staged K and V [1,2,12306,256].  Invalid source
// slots are zero-filled and never referenced by the chronological slot map.
std::vector<mx::array> qwen4_qsa_compact_stage_gather(
    const mx::array &keys, const mx::array &values,
    const mx::array &source_tokens, mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
