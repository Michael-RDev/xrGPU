#include "runtime/integration/xrgpu_kernel_package.h"
#include "devices/registry.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <thread>

using namespace xrgpu_package;

template <class F> void rejects(F action) {
  bool failed = false;
  try { action(); } catch (const std::exception&) { failed = true; }
  assert(failed);
}

int main(int argc, char** argv) {
  assert(argc == 2);
  std::ifstream stream(argv[1]);
  const auto contract = Json::parse(stream);
  for (const auto& v : contract.at("variants")) {
    linear_variant(v.at("id"));
    assert(tile_m == v.at("tile_m"));
    assert(wg_x == v.at("workgroup").at(0));
    assert(wg_y == v.at("workgroup").at(1));
    assert(buffer_weights == (v.at("weight_storage") == "buffer"));
  }
  for (const auto& v : xrgpu::quest3::linear_variants) {
    linear_variant(v.label);
    assert(tile_m == v.tile_m && wg_x == v.workgroup_x && wg_y == v.workgroup_y);
    assert(buffer_weights == v.buffer_weights && v.tile_n == 4);
  }
  rejects([] { linear_variant("unknown"); });
  auto linear = descriptor("linear", {1, 128, 384}, {1, 128, 1536}, "texture3d");
  auto gelu = descriptor("gelu", {1, 128, 384}, {1, 128, 384}, "texture3d",
                         {{"approximate", "none"}});
  assert(choose(linear).empty());
  Context outer;
  {
    Scope scope(outer);
    assert(choose(linear).empty());
    rejects([&] { choose(gelu, true, true); });
    outer.rules.push_back({{"request", gelu}, {"variant", "semantics_v1"},
                           {"kind", "correctness_repair"}});
    assert(choose(gelu, true, true) == "semantics_v1");
    rejects([&] { choose(gelu, false, true); });
    outer.rules.push_back({{"request", linear}, {"variant", "m4_wg8x4_buffer"},
                           {"kind", "performance"}});
    assert(choose(linear) == "m4_wg8x4_buffer");
    auto different = linear;
    different["input_shape"][1] = 127;
    assert(choose(different).empty());
    outer.repairs_only = true;
    assert(choose(linear).empty());
    assert(choose(gelu) == "semantics_v1");
    outer.repairs_only = false;
    outer.rules.push_back(outer.rules.back());
    rejects([&] { choose(linear); });
    outer.rules.erase(outer.rules.size() - 1);
    Context inner;
    try { Scope nested(inner); assert(choose(linear).empty()); throw 1; }
    catch (int) {}
    assert(current == &outer && choose(linear) == "m4_wg8x4_buffer");
    std::thread independent([&] { assert(current == nullptr && choose(linear).empty()); });
    independent.join();
  }
  assert(current == nullptr);
#ifdef XRGPU_ENABLE_EXPERIMENTS
  configure("xrgpu:linear:m2_wg16x4_buffer");
  assert(choose(linear) == "m2_wg16x4_buffer" && tile_m == 2 && wg_x == 16);
  assert(choose(linear, false).empty());
  configure("xrgpu:gelu");
  assert(choose(gelu) == "semantics_v1");
  rejects([] { configure("xrgpu:unknown"); });
#else
  rejects([] { configure("xrgpu:linear:m2_wg16x4_buffer"); });
#endif
  configure("stock");
  assert(current == nullptr && choose(linear).empty());
  std::cout << "PASS: 18 variants, demo mapping, stock/repair policy, scopes and threads\n";
}
