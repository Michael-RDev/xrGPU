#include "devices/registry.h"
#include "runtime/native/dispatch.h"
#include <cassert>
#include <iostream>

using namespace xrgpu;
static_assert(find_device_profile({"Quest 3", 0x5143, 32, 8, 4}) == &quest3::profile);
static_assert(find_device_profile({"Quest 3S", 0x5143, 1024, 1024, 1024}) == nullptr);
static_assert(find_device_profile({"Quest 2", 0x5143, 1024, 1024, 1024}) == nullptr);
static_assert(find_device_profile({"Phone", 0x5143, 1024, 1024, 1024}) == nullptr);
static_assert(find_device_profile({"", 0x5143, 1024, 1024, 1024}) == nullptr);
static_assert(find_device_profile({"Quest 3", 0, 32, 8, 4}) == nullptr);
static_assert(find_device_profile({"Quest 3", 0x5143, 31, 8, 4}) == nullptr);
static_assert(find_device_profile({"Quest 3", 0x5143, 32, 7, 4}) == nullptr);
static_assert(find_device_profile({"Quest 3", 0x5143, 32, 8, 3}) == nullptr);
static_assert(profile_by_id("unregistered") == nullptr);

int main() {
  experiment_variant = 1;
  assert(!eligible(1, 128, 384, 1536, true));  // No profile, no custom dispatch.
  experiment_profile = &quest3::profile;
  for (int variant : {1, 2, 3, 4}) {
    experiment_variant = variant;
    for (int rows : {32, 128, 256}) {
      assert(eligible(1, rows, 384, 1536, true));
      assert(eligible(1, rows, 1536, 384, true));
    }
    assert(!eligible(1, 31, 384, 1536, true));
    assert(!eligible(1, 128, 384, 1535, true));
    assert(!eligible(1, 128, 13, 17, true));
    assert(!eligible(2, 128, 384, 1536, true));
    assert(!eligible(1, 128, 384, 1536, false));
    assert(eligible(1,128,384,384,true)==(variant==4));
    assert(eligible(1,128,384,1152,true)==(variant==4));
  }
  for (int variant : {0, -1, 5}) {
    experiment_variant = variant;
    assert(!eligible(1, 128, 384, 1536, true));
  }

  // This unregistered fixture proves dispatch does not assume Quest 3 tuning.
  const LinearVariant other_variant{7, "test", 2, 16, 2, 8, 1, true};
  const LinearKernelSet other_set{"test.linear", "test_shader", &other_variant, 1,
      [](int64_t b, int64_t m, int64_t k, int64_t n, bool fp32) {
        return b == 1 && m == 7 && k == 13 && n == 17 && fp32;
      }};
  const DeviceProfile other{"test-only", "test", nullptr, &other_set};
  experiment_profile = &other; experiment_variant = 7;
  assert(eligible(1, 7, 13, 17, true));
  assert(!eligible(1, 128, 384, 1536, true));
  assert(tile_m() == 2 && tile_n() == 16 && workgroup_x() == 2 && workgroup_y() == 8);
  assert(buffer_weights() && std::string_view(shader_prefix()) == "test_shader");

  // Compare the compiled policy with the human-readable manifest in Python.
  std::cout << quest3::profile.id << '\n' << quest3::linear.id << '\n'
            << quest3::linear.shader_prefix << '\n';
  for (const auto& v : quest3::linear_variants)
    std::cout << v.id << ' ' << v.tile_m << ' ' << v.tile_n << ' '
              << v.workgroup_x << ' ' << v.workgroup_y << ' ' << v.workgroup_z
              << ' ' << v.buffer_weights << '\n';
}
