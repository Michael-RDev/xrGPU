// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <xrgpu/operators.h>
#include <array>
#include <stdexcept>
namespace xrgpu::operators::detail {
constexpr uint32_t kMaxRank = 16;
constexpr uint32_t kParameterWords = 8 + 3 * kMaxRank;
struct Plan {
  std::string operation;
  std::string shader;
  std::vector<Tensor> inputs;
  Tensor output;
  std::array<uint32_t, kParameterWords> parameters{};
  uint32_t output_words = 0;
};
uint64_t elements(const Tensor& tensor);
DType storage(const Tensor& tensor);
Plan make_plan(const std::string& op_id, const std::vector<Tensor>& inputs,
               const std::vector<Tensor>& requested_outputs);
std::string json_quote(const std::string& value);
}
