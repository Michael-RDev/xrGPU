#pragma once
// Shared by the library and xrBench. Qualification is bundled at build time.
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
namespace xrgpu_package {
using Json = nlohmann::json;
struct Context {
  Json rules = Json::array();
  Json decisions = Json::object();
  std::string experiment;
  bool require_repairs = true;
  bool repairs_only = false;
};
inline thread_local Context* current = nullptr;
inline thread_local Context development;
inline thread_local unsigned selections = 0;
inline thread_local int tile_m = 1;
inline thread_local unsigned wg_x = 8, wg_y = 4;
inline thread_local bool buffer_weights = false;
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
inline std::string choose(const Json& request, bool supported = true, bool unsafe_stock = false) {
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
    if (experiment == operation || (experiment == "repairs" && operation != "linear"))
      selected = operation == "gelu" ? "semantics_v1" : "grouped_v1";
    if (operation == "linear" && experiment.rfind("linear:", 0) == 0)
      selected = experiment.substr(7);
#endif
  }
  if (selected.empty() && unsafe_stock && current->require_repairs)
    throw std::runtime_error("xrGPU requires a qualified correctness repair: " + request.dump());
  if (!selected.empty() && operation == "linear") linear_variant(selected);
  auto key = request.dump();
  if (!current->decisions.contains(key) && !selected.empty()) ++selections;
  current->decisions[key] = {{"request", request},
      {"implementation", selected.empty() ? "stock" : "xrgpu"}, {"variant", selected},
      {"reason", selected.empty() ? "No matching qualified variant" : "Exact eligible request"}};
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
