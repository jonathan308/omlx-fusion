#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

mx::array qwen4_qsa_sparse_gqa_attention(
    const mx::array &queries, const mx::array &keys, const mx::array &values,
    const mx::array &selected_blocks, float scale, int q_offset,
    int key_tile = 128, int dimension_tile = 32, mx::StreamOrDevice s = {});

mx::array qwen4_qsa_sparse_gqa_attention_tokens(
    const mx::array &queries, const mx::array &keys, const mx::array &values,
    const mx::array &selected_tokens, float scale, int key_tile = 64,
    int dimension_tile = 64, mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
// SPDX-License-Identifier: Apache-2.0
