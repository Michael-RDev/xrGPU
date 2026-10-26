#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xrgpu {
enum class DType { Float32, Int64 };
// Tensors own their bytes; outputs remain valid across later calls and close().
struct Tensor {
  std::string name;
  std::vector<int64_t> shape;
  DType dtype = DType::Float32;
  std::vector<uint8_t> data;
};
struct SessionOptions {
  bool force_cpu = false;
  bool profiling = false;
  // Development builds only. Release builds reject these options.
  std::string experiment;
  bool experimental_repairs = false;
  std::string experimental_rules = "[]";
  bool repairs_only = false;
};
class Session final {
public:
  static std::unique_ptr<Session> load(const std::string &bundle_directory,
                                       SessionOptions options = {});
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  std::vector<Tensor> forward(const std::vector<Tensor> &inputs);
  // JSON includes input/output specs, backend, device, decisions and fallback
  // reason.
  std::string info() const;
  void close();

private:
  struct Impl;
  explicit Session(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
} // namespace xrgpu
