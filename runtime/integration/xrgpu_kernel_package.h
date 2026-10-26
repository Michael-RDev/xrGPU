#pragma once
// Shared by the library and xrBench. Qualification is bundled at build time.
#include <nlohmann/json.hpp>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
namespace xrgpu_package {
using Json = nlohmann::json;
struct MissingQualifiedRepair : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Context {
  Json rules = Json::array();
  Json decisions = Json::object();
  std::string experiment;
  bool require_repairs = true;
  bool repairs_only = false;
  bool experimental_repairs = false;
  std::map<const void*, size_t> graphs;
};
inline thread_local Context* current = nullptr;
inline thread_local Context development;
inline thread_local unsigned selections = 0;
inline thread_local int tile_m = 1;
inline thread_local unsigned wg_x = 8, wg_y = 4;
inline thread_local bool buffer_weights = false;
inline std::string node_identity(const void* graph, int64_t output) {
  if (!current) return "";
  auto inserted = current->graphs.emplace(graph, current->graphs.size());
  return "graph" + std::to_string(inserted.first->second) + ":value" + std::to_string(output);
}
struct Scope {
  Context* previous;
  explicit Scope(Context& context) : previous(current) { current = &context; }
  ~Scope() { current = previous; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};
inline void linear_variant(const std::string& variant) {
  for (int m : {1, 2, 4})
    for (auto wg : {std::pair<unsigned, unsigned>{8, 4}, {16, 4}, {8, 8}})
      for (bool buffer : {false, true}) {
        auto id = "m" + std::to_string(m) + "_wg" + std::to_string(wg.first) + "x" +
            std::to_string(wg.second) + (buffer ? "_buffer" : "_texture2d");
        if (variant == id) {
          tile_m = m; wg_x = wg.first; wg_y = wg.second; buffer_weights = buffer;
          return;
        }
      }
  throw std::invalid_argument("Unknown xrGPU linear variant: " + variant);
}
inline Json descriptor(const char* operation, const std::vector<int64_t>& input,
                       const std::vector<int64_t>& output, const char* layout,
                       Json attributes = Json::object()) {
  return {{"operation", operation}, {"input_shape", input}, {"output_shape", output},
          {"dtype", "float32"}, {"layout", layout}, {"attributes", attributes}};
}
inline bool known_variant(const std::string& operation, const std::string& variant) {
  if (operation == "gelu") return variant == "semantics_v1";
  if (operation == "grouped_conv2d") return variant == "grouped_v1";
  if (operation != "linear") return false;
  for (int m : {1, 2, 4})
    for (auto wg : {std::pair<unsigned, unsigned>{8, 4}, {16, 4}, {8, 8}})
      for (const char* storage : {"buffer", "texture2d"})
        if (variant == "m" + std::to_string(m) + "_wg" + std::to_string(wg.first) +
            "x" + std::to_string(wg.second) + "_" + storage) return true;
  return false;
}
// Records are verified and embedded at library build time. A model cannot
// supply its own eligibility. Device/driver identities never use wildcards.
inline Json eligible_rules(const Json& record, const Json& package,
                           const Json& device, const std::string& runtime) {
  if (record.is_null() || record.empty()) return Json::array();
  if (record.value("schema_version", 0) != 2 ||
      record.value("kind", "") != "xrgpu.qualification")
    throw std::invalid_argument("Unsupported xrGPU qualification record");
  for (const char* key : {"package_id", "package_sha256", "runtime_revision"})
    if (!package.contains(key) || record.at(key) != package.at(key))
      throw std::invalid_argument("Qualification package/runtime mismatch");
  const auto& rules = record.at("rules");
  if (!rules.is_array()) throw std::invalid_argument("Invalid qualification rules");
  if (record.value("status", "") == "pending") {
    if (!rules.empty()) throw std::invalid_argument("Pending rules cannot authorize kernels");
    return Json::array();
  }
  if (record.value("status", "") != "qualified" || record.value("synthetic", true) ||
      record.at("report_sha256").empty() || rules.empty())
    throw std::invalid_argument("Qualified hardware evidence required");
  const auto& target = record.at("target");
  if (package.contains("device"))
    for (auto it = package.at("device").begin(); it != package.at("device").end(); ++it)
      if (target.contains(it.key()) && target.at(it.key()) != it.value())
        throw std::invalid_argument("Qualification target is outside the kernel package");
  bool matches = record.at("runtime_revision") == runtime;
  for (const char* key : {"model", "gpu", "vendor_id", "device_id", "driver", "api_version"}) {
    if (!target.contains(key) || target.at(key).is_null())
      throw std::invalid_argument("Incomplete qualification target");
    matches = matches && device.contains(key) && device.at(key) == target.at(key);
  }
  Json seen = Json::object();
  for (const auto& rule : rules) {
    const auto& request = rule.at("request");
    const auto operation = request.at("operation").get<std::string>();
    const auto kind = rule.at("kind").get<std::string>();
    if (!known_variant(operation, rule.at("variant").get<std::string>()) ||
        (kind != "performance" && kind != "correctness_repair") ||
        (kind == "correctness_repair" && operation == "linear") ||
        request.at("dtype") != "float32" || !request.at("attributes").is_object() ||
        (request.at("layout") != "buffer" && request.at("layout") != "texture3d"))
      throw std::invalid_argument("Invalid qualification rule");
    for (const char* key : {"input_shape", "output_shape"}) {
      const auto& shape = request.at(key);
      if (!shape.is_array() || shape.empty() || shape.size() > 8)
        throw std::invalid_argument("Static qualification shape required");
      for (const auto& d : shape)
        if (!d.is_number_integer() || d.get<int64_t>() <= 0)
          throw std::invalid_argument("Positive static qualification shape required");
    }
    const auto key = request.dump();
    if (seen.contains(key)) throw std::invalid_argument("Ambiguous xrGPU qualification");
    seen[key] = true;
    const auto& gates = record.at("gates");
    for (const char* gate : {"hardware", "provenance", "correctness", "controls", "holdout"})
      if (!gates.contains(gate) || gates.at(gate) != true)
        throw std::invalid_argument("Missing hardware correctness qualification gate");
    if (kind == "performance")
      for (const char* gate : {"performance", "models", "sustained"})
        if (!gates.contains(gate) || gates.at(gate) != true)
          throw std::invalid_argument("Missing performance qualification gate");
  }
  return matches ? rules : Json::array();
}
inline std::string choose(const Json& request, bool supported = true, bool unsafe_stock = false,
                          const std::string& node = "") {
  if (!current) return "";
  std::string selected;
  const auto operation = request.at("operation").get<std::string>();
  if (supported) {
    for (const auto& rule : current->rules) {
      if (rule.at("request") == request &&
          (!current->repairs_only || rule.at("kind") == "correctness_repair")) {
        if (!selected.empty()) throw std::runtime_error("Ambiguous xrGPU qualification");
        selected = rule.at("variant").get<std::string>();
      }
    }
#ifdef XRGPU_ENABLE_EXPERIMENTS
    const auto& experiment = current->experiment;
    if (experiment == operation || ((experiment == "repairs" || current->experimental_repairs) && operation != "linear"))
      selected = operation == "gelu" ? "semantics_v1" : "grouped_v1";
    if (operation == "linear" && experiment.rfind("linear:", 0) == 0)
      selected = experiment.substr(7);
#endif
  }
  if (!selected.empty() && !known_variant(operation, selected))
    throw std::invalid_argument("Unknown qualified kernel variant");
  if (!selected.empty() && operation == "linear") linear_variant(selected);
  auto key = node.empty() ? request.dump() : node;
  if (!current->decisions.contains(key) && !selected.empty()) ++selections;
  current->decisions[key] = {{"request", request},
      {"implementation", selected.empty() ? "stock" : "xrgpu"}, {"variant", selected},
      {"reason", !supported ? "Unsupported operation layout or shape" :
          selected.empty() ? "No matching qualified variant" : "Exact eligible request"}};
  if (selected.empty() && unsafe_stock && current->require_repairs)
    throw MissingQualifiedRepair("xrGPU requires a qualified correctness repair: " + request.dump());
  return selected;
}
// Explicit candidates are available only in the instrumented development runner.
inline void configure(const std::string& request) {
  current = nullptr; development = Context{}; selections = 0;
  if (request == "stock" || request.empty()) return;
#ifdef XRGPU_ENABLE_EXPERIMENTS
  if (request.rfind("xrgpu:", 0) != 0) throw std::invalid_argument("Unknown xrGPU request");
  auto token = request.substr(6);
  if (token.rfind("linear:", 0) == 0) linear_variant(token.substr(7));
  else if (token != "gelu" && token != "grouped_conv2d" && token != "repairs")
    throw std::invalid_argument("Unknown xrGPU candidate");
  development.experiment = token;
  development.require_repairs = false;
  current = &development;
#else
  throw std::invalid_argument("Candidate overrides are disabled in production");
#endif
}
}
