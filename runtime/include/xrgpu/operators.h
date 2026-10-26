// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xrgpu::operators {
enum class DType { Float32, Float16, Int32, UInt32, Int16, Int8, UInt8, Int64, Bool, Unspecified };
std::string dtype_name(DType dtype);
DType parse_dtype(const std::string& name);
uint32_t dtype_bytes(DType dtype);
// Owned, little-endian bytes in contiguous row-major order. An empty shape is a
// scalar; dimensions of zero represent empty tensors. Unspecified storage uses
// the logical dtype. The alpha kernels require storage_dtype == dtype.
struct Tensor {
  std::string name;
  std::vector<int64_t> shape;
  DType dtype = DType::Float32;
  DType storage_dtype = DType::Unspecified;
  std::string layout = "contiguous";
  std::vector<uint8_t> data;
};
struct Options {
  bool profiling = false;
  uint64_t timeout_ns = 15000000000ULL;
  uint64_t memory_limit_bytes = 268435456ULL;
};
// Metadata-only validation and inference; does not access Vulkan or input data.
// Requested outputs may be omitted; otherwise their names/types/shapes must match.
std::vector<Tensor> plan(const std::string& op_id, const std::vector<Tensor>& inputs,
                         const std::vector<Tensor>& requested_outputs = {});
// JSON with available/backend/device/reason. Never silently uses CPU compute.
std::string probe();
class Operator final {
public:
  static std::unique_ptr<Operator> prepare(const std::string& package_directory,
      const std::string& op_id, const std::vector<Tensor>& inputs,
      Options options = {}, const std::vector<Tensor>& requested_outputs = {});
  ~Operator();
  Operator(const Operator&) = delete;
  Operator& operator=(const Operator&) = delete;
  // Synchronous submission, recording/submit/wait timing only. Inputs are fixed
  // for this prepared call. read_outputs returns owned bytes outside that timing.
  void run();
  std::vector<Tensor> read_outputs() const;
  // JSON: execution_ms, gpu_ms (null when unavailable), dispatch_count, device,
  // backend, operation, storage and input/output metadata, executed/closed state.
  std::string info() const;
  void close();
private:
  struct Impl;
  explicit Operator(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
} // namespace xrgpu::operators
