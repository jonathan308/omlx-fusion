// SPDX-License-Identifier: Apache-2.0

#include "qwen4_qsa_compact_stage.h"

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;

std::string compact_stage_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void *>(&compact_stage_binary_dir), &info)) {
      throw std::runtime_error("Unable to get compact-stage binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

struct Qwen4CompactStageParams {
  uint32_t q_offset;
  uint32_t key_tokens;
};
static_assert(sizeof(Qwen4CompactStageParams) == 2 * sizeof(uint32_t));

struct alignas(uint64_t) Qwen4CompactGatherParams {
  uint32_t key_tokens;
  uint32_t reserved;
  uint64_t key_head_stride;
  uint64_t key_token_stride;
  uint64_t value_head_stride;
  uint64_t value_token_stride;
};
static_assert(offsetof(Qwen4CompactGatherParams, key_head_stride) == 8);
static_assert(offsetof(Qwen4CompactGatherParams, value_token_stride) == 32);
static_assert(sizeof(Qwen4CompactGatherParams) == 40);

bool compact_stage_gather_layout(const array &value) {
  const auto token_stride = value.strides(-2);
  const auto head_stride = value.strides(-3);
  return value.offset() % 16 == 0 && value.strides(-1) == 1 &&
         token_stride >= 256 &&
         token_stride % 8 == 0 && head_stride > 0 && head_stride % 8 == 0 &&
         static_cast<uint64_t>(head_stride) >=
             static_cast<uint64_t>(value.shape(-2)) *
                 static_cast<uint64_t>(token_stride);
}

class Qwen4CompactStagePlanPrimitive : public Primitive {
public:
  Qwen4CompactStagePlanPrimitive(Stream stream, int q_offset, int key_tokens)
      : Primitive(stream), q_offset_(q_offset), key_tokens_(key_tokens) {}

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error("Qwen4 compact-stage plan has no CPU path.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    const auto &selected = inputs[0];
    if (!selected.flags().row_contiguous || selected.strides(-1) != 1) {
      throw std::runtime_error(
          "Qwen4 compact-stage plan requires a realized row-contiguous "
          "selector.");
    }
    for (auto &output : outputs) {
      output.set_data(allocator::malloc(output.nbytes()));
    }

    Qwen4CompactStageParams params{
        static_cast<uint32_t>(q_offset_), static_cast<uint32_t>(key_tokens_)};
    auto library =
        device.get_library("omlx_glm_kernels", compact_stage_binary_dir());
    auto kernel = device.get_kernel("qwen4_qsa_compact_stage_plan", library);
    auto &encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(selected, 0);
    encoder.set_output_array(outputs[0], 1);
    encoder.set_output_array(outputs[1], 2);
    encoder.set_output_array(outputs[2], 3);
    encoder.set_output_array(outputs[3], 4);
    encoder.set_output_array(outputs[4], 5);
    encoder.set_bytes(params, 6);
    encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
  }

  DEFINE_NAME(OMLXQwen4CompactStagePlan)
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs =
        static_cast<const Qwen4CompactStagePlanPrimitive &>(other);
    return q_offset_ == rhs.q_offset_ && key_tokens_ == rhs.key_tokens_;
  }

private:
  int q_offset_;
  int key_tokens_;
};

class Qwen4CompactStageGatherPrimitive : public Primitive {
public:
  explicit Qwen4CompactStageGatherPrimitive(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error("Qwen4 compact-stage gather has no CPU path.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    const auto &keys = inputs[0];
    const auto &values = inputs[1];
    const auto &sources = inputs[2];
    if (!compact_stage_gather_layout(keys) ||
        !compact_stage_gather_layout(values) ||
        !sources.flags().row_contiguous || sources.strides(-1) != 1) {
      throw std::runtime_error(
          "Qwen4 compact-stage gather requires unit-stride K/V vectors, "
          "128-bit-aligned head/token strides, and a contiguous source map.");
    }
    for (auto &output : outputs) {
      output.set_data(allocator::malloc(output.nbytes()));
    }

    const std::string dtype = keys.dtype() == bfloat16 ? "bfloat16" : "float16";
    const std::string kernel_name =
        "qwen4_qsa_compact_stage_gather_" + dtype;
    auto library =
        device.get_library("omlx_glm_kernels", compact_stage_binary_dir());
    auto kernel = device.get_kernel(kernel_name, library);
    auto &encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(keys, 0);
    encoder.set_input_array(values, 1);
    encoder.set_input_array(sources, 2);
    encoder.set_output_array(outputs[0], 3);
    encoder.set_output_array(outputs[1], 4);
    Qwen4CompactGatherParams params{
        static_cast<uint32_t>(keys.shape(2)),
        0,
        static_cast<uint64_t>(keys.strides(1)),
        static_cast<uint64_t>(keys.strides(2)),
        static_cast<uint64_t>(values.strides(1)),
        static_cast<uint64_t>(values.strides(2))};
    encoder.set_bytes(params, 5);
    constexpr size_t words =
        2 * kQwen4StageTokenCapacity * (256 / 8);
    encoder.dispatch_threads(MTL::Size(words, 1, 1), MTL::Size(256, 1, 1));
  }

  DEFINE_NAME(OMLXQwen4CompactStageGather)
  bool is_equivalent(const Primitive & /* other */) const override {
    return true;
  }
};

} // namespace

std::vector<array> qwen4_qsa_compact_stage_plan(
    const array &selected_blocks, int q_offset, int key_tokens,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  if (stream.device == Device::cpu || selected_blocks.dtype() != uint32 ||
      selected_blocks.shape() !=
          Shape{1, 1, kQwen4StageRows, kQwen4StageBlocksPerRow} ||
      selected_blocks.strides(-1) != 1 ||
      q_offset < kQwen4StageMinimumQOffset || key_tokens <= 0 ||
      key_tokens < kQwen4StageRows ||
      q_offset > key_tokens - kQwen4StageRows) {
    std::ostringstream msg;
    msg << "[qwen4_qsa_compact_stage_plan] expected GPU uint32 selected="
        << "[1,1,6,512], q_offset>=2047, q_offset+6<=key_tokens; got "
        << selected_blocks.shape() << ", q_offset=" << q_offset
        << ", key_tokens=" << key_tokens << ".";
    throw std::invalid_argument(msg.str());
  }
  return array::make_arrays(
      {{kQwen4StageUnionCapacity},
       {1},
       {1, 1, kQwen4StageRows, kQwen4StageTokensPerRow},
       {kQwen4StageTokenCapacity},
       {1}},
      {uint32, uint32, uint32, uint32, uint32},
      std::make_shared<Qwen4CompactStagePlanPrimitive>(
          stream, q_offset, key_tokens),
      {selected_blocks});
}

std::vector<array> qwen4_qsa_compact_stage_gather(
    const array &keys, const array &values, const array &source_tokens,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  if (stream.device == Device::cpu || keys.shape() != values.shape() ||
      keys.ndim() != 4 || keys.shape(0) != 1 || keys.shape(1) != 2 ||
      keys.shape(2) <= 0 || keys.shape(3) != 256 ||
      keys.dtype() != values.dtype() ||
      (keys.dtype() != float16 && keys.dtype() != bfloat16) ||
      source_tokens.dtype() != uint32 ||
      source_tokens.shape() != Shape{kQwen4StageTokenCapacity} ||
      !compact_stage_gather_layout(keys) ||
      !compact_stage_gather_layout(values)) {
    std::ostringstream msg;
    msg << "[qwen4_qsa_compact_stage_gather] expected GPU matching BF16/F16 "
        << "k/v=[1,2,K,256] and uint32 sources=[12306]; got " << keys.shape()
        << ", " << values.shape() << ", " << source_tokens.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  Shape staged_shape{1, 2, kQwen4StageTokenCapacity, 256};
  return array::make_arrays(
      {staged_shape, staged_shape},
      {keys.dtype(), keys.dtype()},
      std::make_shared<Qwen4CompactStageGatherPrimitive>(stream),
      {keys, values, source_tokens});
}

} // namespace omlx::glm_kernels
