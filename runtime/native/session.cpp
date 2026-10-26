#include "dispatch.h"
#include "runtime/integration/xrgpu_kernel_package.h"
#include "runtime_capabilities.h"
#include "sha256.h"
#include "trace.h"
#include "xrgpu_build_config.h"
#include <cstring>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <xrgpu/session.h>

namespace xrgpu {
namespace {
using executorch::extension::Module;
using executorch::runtime::EValue;
namespace fs = std::filesystem;
using Json = nlohmann::json;
std::once_flag runtime_initialization;
Json read_json(const fs::path &path) {
  std::ifstream input(path);
  if (!input)
    throw std::invalid_argument("Cannot read model manifest: " + path.string());
  return Json::parse(input);
}
fs::path contained_file(const fs::path &root, const std::string &name) {
  fs::path relative(name);
  if (name.empty() || relative.is_absolute() ||
      name.find('\\') != std::string::npos)
    throw std::invalid_argument("Invalid bundle path: " + name);
  for (const auto &part : relative)
    if (part == ".." || part == "." || part.empty())
      throw std::invalid_argument("Invalid bundle path: " + name);
  auto path = root;
  for (const auto &part : relative) {
    path /= part;
    if (fs::is_symlink(path))
      throw std::invalid_argument("Bundle symlinks are not supported: " + name);
  }
  if (!fs::is_regular_file(path))
    throw std::invalid_argument("Missing bundle file: " + name);
  return path;
}
DType dtype(const std::string &value) {
  if (value == "float32")
    return DType::Float32;
  if (value == "int64")
    return DType::Int64;
  throw std::invalid_argument("Only float32 and int64 tensors are supported");
}
size_t tensor_bytes(const Json &spec) {
  size_t bytes = dtype(spec.at("dtype")) == DType::Float32 ? 4 : 8;
  for (const auto &item : spec.at("shape")) {
    if (!item.is_number_integer())
      throw std::invalid_argument("Tensor shapes must be static integers");
    auto value = item.get<int64_t>();
    if (value <= 0 || value > std::numeric_limits<int32_t>::max() ||
        bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(value))
      throw std::invalid_argument("Invalid tensor shape");
    bytes *= static_cast<size_t>(value);
  }
  if (bytes > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    throw std::invalid_argument("Tensor exceeds the v1 Session size limit");
  return bytes;
}
void validate_specs(const Json &specs) {
  if (!specs.is_array() || specs.empty())
    throw std::invalid_argument("Tensor specifications are required");
  std::set<std::string> names;
  for (const auto &spec : specs) {
    auto name = spec.at("name").get<std::string>();
    if (name.empty() || !names.insert(name).second)
      throw std::invalid_argument("Duplicate tensor name");
    if (!spec.at("shape").is_array())
      throw std::invalid_argument("Invalid tensor shape");
    tensor_bytes(spec);
  }
}
void validate_method(Module &module, const Json &manifest) {
  auto metadata = module.method_meta("forward");
  if (!metadata.ok())
    throw std::runtime_error("Cannot inspect model forward method");
  if (metadata->num_inputs() != manifest.at("inputs").size() ||
      metadata->num_outputs() != manifest.at("outputs").size())
    throw std::invalid_argument(
        "Program tensor count differs from bundle specification");
  for (bool output : {false, true}) {
    const auto &specs = manifest.at(output ? "outputs" : "inputs");
    for (size_t i = 0; i < specs.size(); ++i) {
      auto tensor = output ? metadata->output_tensor_meta(i)
                           : metadata->input_tensor_meta(i);
      if (!tensor.ok())
        throw std::invalid_argument(
            "Model interface must contain tensors only");
      auto expected = dtype(specs[i].at("dtype")) == DType::Float32
                          ? executorch::aten::ScalarType::Float
                          : executorch::aten::ScalarType::Long;
      std::vector<int64_t> sizes(tensor->sizes().begin(),
                                 tensor->sizes().end());
      if (tensor->scalar_type() != expected ||
          sizes != specs[i].at("shape").get<std::vector<int64_t>>())
        throw std::invalid_argument(
            "Program tensor specification differs from bundle");
    }
  }
}
} // namespace
struct Session::Impl {
  mutable std::mutex mutex;
  Json manifest, device, package, qualification;
  fs::path directory;
  SessionOptions options;
  xrgpu_package::Context selection;
  std::unique_ptr<Module> module;
  Trace *trace = nullptr;
  std::string backend, fallback_reason;
  double load_ms = 0, last_ms = 0;
  bool closed = false;
  void load_program(const std::string &name) {
    const auto file =
        manifest.at("programs").at(name).at("file").get<std::string>();
    std::unique_ptr<Trace> recorder;
    if (options.profiling)
      recorder = std::make_unique<Trace>();
    auto *new_trace = recorder.get();
    xrgpu_package::Scope scope(selection);
    gpu_timestamps_supported =
        options.profiling && device.value("gpu_timestamps", false);
    auto new_module =
        std::make_unique<Module>(contained_file(directory, file).string(),
                                 Module::LoadMode::Mmap, std::move(recorder));
    auto error = new_module->load_forward();
    if (error != executorch::runtime::Error::Ok)
      throw std::runtime_error("ExecuTorch load failed, code " +
                               std::to_string(static_cast<int>(error)));
    validate_method(*new_module, manifest);
    // Commit only a fully loaded program. A throwing Vulkan load must leave
    // module null so the caller can load the CPU fallback safely.
    module = std::move(new_module);
    trace = new_trace;
    backend = name;
  }
};
Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Session::~Session() = default;
std::unique_ptr<Session> Session::load(const std::string &directory,
                                       SessionOptions options) {
#ifndef XRGPU_ENABLE_EXPERIMENTS
  if (!options.experiment.empty() || options.experimental_repairs ||
      options.repairs_only || options.experimental_rules != "[]")
    throw std::invalid_argument(
        "Candidate overrides are disabled in production");
#else
  if (!options.experiment.empty() && options.experiment != "repairs" &&
      options.experiment != "gelu" && options.experiment != "grouped_conv2d") {
    if (options.experiment.rfind("linear:", 0) != 0 ||
        !xrgpu_package::known_variant("linear", options.experiment.substr(7)))
      throw std::invalid_argument("Unknown development candidate");
  }
#endif
  auto state = std::make_unique<Impl>();
  state->options = std::move(options);
  state->directory = fs::canonical(directory);
  state->manifest = read_json(contained_file(state->directory, "bundle.json"));
  const auto &manifest = state->manifest;
  if (manifest.at("schema_version") != 2 ||
      manifest.at("bundle_type") != "xrgpu.model" ||
      manifest.at("precision") != "fp32" ||
      manifest.at("runtime_revision") != XRGPU_ET_REVISION)
    throw std::invalid_argument(
        "Unsupported or incompatible xrGPU model bundle");
  if (manifest.contains("qualification") || manifest.contains("selections"))
    throw std::invalid_argument(
        "Model bundles cannot supply qualification policy");
  validate_specs(manifest.at("inputs"));
  validate_specs(manifest.at("outputs"));
  if (!manifest.at("files").is_object() || manifest.at("files").empty())
    throw std::invalid_argument("Model bundle file hashes are required");
  for (auto it = manifest.at("files").begin(); it != manifest.at("files").end();
       ++it) {
    if (file_sha256(contained_file(state->directory, it.key()).string()) !=
        it.value())
      throw std::invalid_argument("Bundle checksum mismatch: " + it.key());
  }
  for (auto it = manifest.at("programs").begin();
       it != manifest.at("programs").end(); ++it) {
    if (it.key() != "cpu" && it.key() != "vulkan")
      throw std::invalid_argument("Unknown model backend");
    if (!manifest.at("files").contains(
            it.value().at("file").get<std::string>()))
      throw std::invalid_argument("Unhashed model program");
  }
  if (!manifest.at("programs").contains("cpu"))
    throw std::invalid_argument("CPU fallback program required");
  state->package = Json::parse(XRGPU_PACKAGE_JSON);
  state->qualification = Json::parse(XRGPU_QUALIFICATION_JSON);
  if (manifest.contains("kernel_package")) {
    for (const auto *key : {"package_id", "package_sha256"})
      if (!state->package.contains(key) ||
          manifest.at("kernel_package").at(key) != state->package.at(key))
        throw std::invalid_argument(
            "Model kernel package does not match the installed runtime");
  }
  state->device = runtime_capabilities();
  state->selection.rules = xrgpu_package::eligible_rules(
      state->qualification, state->package, state->device, XRGPU_ET_REVISION);
  state->selection.require_repairs = true;
  state->selection.repairs_only = state->options.repairs_only;
#ifdef XRGPU_ENABLE_EXPERIMENTS
  state->selection.experiment = state->options.experiment;
  state->selection.experimental_repairs = state->options.experimental_repairs;
  const auto rules = Json::parse(state->options.experimental_rules);
  if (!rules.is_array())
    throw std::invalid_argument("Development rules must be an array");
  std::set<std::string> seen;
  for (const auto &rule : rules) {
    const auto &request = rule.at("request");
    const auto operation = request.at("operation").get<std::string>();
    const auto kind = rule.at("kind").get<std::string>();
    if (!xrgpu_package::known_variant(operation,
                                      rule.at("variant").get<std::string>()) ||
        (kind != "performance" && kind != "correctness_repair") ||
        request.at("dtype") != "float32" ||
        !request.at("attributes").is_object() ||
        (request.at("layout") != "buffer" &&
         request.at("layout") != "texture3d") ||
        !seen.insert(request.dump()).second)
      throw std::invalid_argument("Invalid or ambiguous development rule");
    for (const char *key : {"input_shape", "output_shape"}) {
      if (!request.at(key).is_array() || request.at(key).empty())
        throw std::invalid_argument("Invalid development rule shape");
      for (const auto &dimension : request.at(key))
        if (!dimension.is_number_integer() || dimension.get<int64_t>() <= 0)
          throw std::invalid_argument("Static development rule shape required");
    }
  }
  if (!rules.empty())
    state->selection.rules = rules;

#endif
  std::call_once(runtime_initialization,
                 [] { executorch::runtime::runtime_init(); });
  const auto started = milliseconds();
  const bool target =
      state->device.value("model", std::string{}) == "Quest 3" &&
      state->device.value("vendor_id", 0) == 0x5143 &&
      state->device.value("gpu", std::string{}).find("740") !=
          std::string::npos;
  if (state->options.force_cpu)
    state->fallback_reason = "CPU explicitly requested";
  else if (!manifest.at("programs").contains("vulkan"))
    state->fallback_reason = "Bundle has no supported Vulkan program";
  else if (!state->device.value("vulkan", false))
    state->fallback_reason =
        state->device.value("reason", std::string("Vulkan unavailable"));
  else if (!target)
    state->fallback_reason =
        "Device is outside the Quest 3 / Adreno 740 target";
  else if (state->package.empty())
    state->fallback_reason = "No verified kernel package is installed";
  else {
    try {
      state->load_program("vulkan");
    } catch (const xrgpu_package::MissingQualifiedRepair &error) {
      state->fallback_reason = error.what();
    } catch (const std::runtime_error &error) {
      state->fallback_reason =
          std::string("Vulkan load unavailable: ") + error.what();
    }
  }
  if (!state->module)
    state->load_program("cpu");
  state->load_ms = milliseconds() - started;
  return std::unique_ptr<Session>(new Session(std::move(state)));
}
std::vector<Tensor> Session::forward(const std::vector<Tensor> &inputs) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->closed)
    throw std::runtime_error("Closed session");
  const auto &specs = impl_->manifest.at("inputs");
  if (inputs.size() != specs.size())
    throw std::invalid_argument("Input tensor count mismatch");
  std::vector<executorch::extension::TensorPtr> tensors;
  std::vector<EValue> values;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    const auto &spec = specs[i];
    if (input.name != spec.at("name") ||
        input.shape != spec.at("shape").get<std::vector<int64_t>>() ||
        input.dtype != dtype(spec.at("dtype")) ||
        input.data.size() != tensor_bytes(spec))
      throw std::invalid_argument("Input tensor specification mismatch: " +
                                  spec.at("name").get<std::string>());
    std::vector<executorch::aten::SizesType> sizes(input.shape.begin(),
                                                   input.shape.end());
    tensors.push_back(executorch::extension::make_tensor_ptr(
        sizes, const_cast<uint8_t *>(input.data.data()),
        input.dtype == DType::Float32 ? executorch::aten::ScalarType::Float
                                      : executorch::aten::ScalarType::Long,
        executorch::aten::TensorShapeDynamism::STATIC));
    values.emplace_back(*tensors.back());
  }
  xrgpu_package::Scope scope(impl_->selection);
  gpu_timestamps_supported =
      impl_->options.profiling && impl_->device.value("gpu_timestamps", false);
  if (impl_->trace)
    impl_->trace->reset();
  const auto started = milliseconds();
  auto result = impl_->module->forward(values);
  impl_->last_ms = milliseconds() - started;
  if (!result.ok())
    throw std::runtime_error("ExecuTorch forward failed, code " +
                             std::to_string(static_cast<int>(result.error())));
  const auto &output_specs = impl_->manifest.at("outputs");
  if (result->size() != output_specs.size())
    throw std::runtime_error("Output tensor count mismatch");
  std::vector<Tensor> outputs;
  for (size_t i = 0; i < result->size(); ++i) {
    if (!result->at(i).isTensor())
      throw std::runtime_error("Expected tensor output");
    const auto tensor = result->at(i).toTensor();
    const auto &spec = output_specs[i];
    Tensor output;
    output.name = spec.at("name");
    output.dtype = dtype(spec.at("dtype"));
    output.shape.assign(tensor.sizes().begin(), tensor.sizes().end());
    const auto expected = output.dtype == DType::Float32
                              ? executorch::aten::ScalarType::Float
                              : executorch::aten::ScalarType::Long;
    if (tensor.scalar_type() != expected ||
        output.shape != spec.at("shape").get<std::vector<int64_t>>())
      throw std::runtime_error("Output tensor specification mismatch");
    output.data.resize(tensor_bytes(spec));
    const size_t item_size = output.dtype == DType::Float32 ? 4 : 8;
    bool contiguous = true;
    size_t expected_stride = 1;
    const auto strides = tensor.strides();
    for (size_t dimension = output.shape.size(); dimension-- > 0;) {
      if (strides[dimension] < 0)
        throw std::runtime_error("Negative output stride unsupported");
      if (output.shape[dimension] > 1 &&
          static_cast<size_t>(strides[dimension]) != expected_stride)
        contiguous = false;
      expected_stride *= static_cast<size_t>(output.shape[dimension]);
    }
    const auto *source = static_cast<const uint8_t *>(tensor.const_data_ptr());
    if (contiguous)
      std::memcpy(output.data.data(), source, output.data.size());
    else
      for (size_t element = 0; element < output.data.size() / item_size;
           ++element) {
        size_t remaining = element, offset = 0;
        for (size_t dimension = output.shape.size(); dimension-- > 0;) {
          const size_t coordinate =
              remaining % static_cast<size_t>(output.shape[dimension]);
          remaining /= static_cast<size_t>(output.shape[dimension]);
          const size_t stride = static_cast<size_t>(strides[dimension]);
          if (stride &&
              coordinate >
                  (std::numeric_limits<size_t>::max() / item_size - offset) /
                      stride)
            throw std::runtime_error("Output stride overflow");
          offset += coordinate * stride;
        }
        std::memcpy(output.data.data() + element * item_size,
                    source + offset * item_size, item_size);
      }
    outputs.push_back(std::move(output));
  }
  return outputs;
}
std::string Session::info() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  Json result = {{"model_id", impl_->manifest.at("model_id")},
                 {"inputs", impl_->manifest.at("inputs")},
                 {"outputs", impl_->manifest.at("outputs")},
                 {"backend", impl_->backend},
                 {"fallback_reason", impl_->fallback_reason},
                 {"device", impl_->device},
                 {"runtime_revision", XRGPU_ET_REVISION},
                 {"decisions", impl_->backend == "vulkan"
                                   ? impl_->selection.decisions
                                   : Json::object()},
                 {"closed", impl_->closed},
                 {"load_ms", impl_->load_ms},
                 {"last_ms", impl_->last_ms}};
  if (impl_->backend != "vulkan" && !impl_->selection.decisions.empty())
    result["attempted_vulkan_decisions"] = impl_->selection.decisions;
  result["kernel_package"] = Json::object();
  for (const auto *key : {"package_id", "package_sha256"})
    if (impl_->package.contains(key))
      result["kernel_package"][key] = impl_->package.at(key);
  result["qualification_status"] =
      impl_->qualification.value("status", std::string("unqualified"));
#ifdef XRGPU_ENABLE_EXPERIMENTS
  result["build_type"] = "development";
#else
  result["build_type"] = "production";
#endif
  result["qualification_eligible"] = !impl_->selection.rules.empty();
  result["qualification_sha256"] =
      impl_->qualification.value("qualification_sha256", std::string{});
  if (impl_->trace)
    result["trace"] = {{"gpu", impl_->trace->gpu},
                       {"host", impl_->trace->host},
                       {"cpu_delegate", impl_->trace->cpu_delegate}};
  return result.dump();
}
void Session::close() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->module.reset();
  impl_->trace = nullptr;
  impl_->closed = true;
}
} // namespace xrgpu
