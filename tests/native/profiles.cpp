#include "devices/registry.h"
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
  const auto& linear = *quest3::profile.linear;
  for (int variant : {1, 2, 3}) {
    assert(linear.variant(variant));
    for (int rows : {32, 128, 256}) {
      assert(linear.supports(1, rows, 384, 1536, true));
      assert(linear.supports(1, rows, 1536, 384, true));
    }
    assert(linear.supports(1, 31, 384, 1536, true));
    assert(linear.supports(1, 128, 384, 1535, true));
    assert(linear.supports(1, 128, 13, 17, true));
    assert(!linear.supports(1, 0, 384, 1536, true));
    assert(!linear.supports(2, 128, 384, 1536, true));
    assert(!linear.supports(1, 128, 384, 1536, false));
  }
  for (int variant : {0, -1, 4}) {
    assert(!linear.variant(variant));
  }

  // This unregistered fixture proves dispatch does not assume Quest 3 tuning.
  const LinearVariant other_variant{7, "test", 2, 16, 2, 8, 1, true};
  const LinearKernelSet other_set{"test.linear", "test_shader", &other_variant, 1,
      [](int64_t b, int64_t m, int64_t k, int64_t n, bool fp32) {
        return b == 1 && m == 7 && k == 13 && n == 17 && fp32;
      }};
  const DeviceProfile other{"test-only", "test", nullptr, &other_set};
  assert(other.linear->supports(1, 7, 13, 17, true));
  assert(!other.linear->supports(1, 128, 384, 1536, true));
  assert(other.linear->variant(7)->tile_m == 2);
  assert(other.linear->variant(7)->buffer_weights);

  // Compare the compiled demo IDs with the current package contract in Python.
  std::cout << quest3::profile.id << '\n' << quest3::linear.id << '\n'
            << quest3::linear.shader_prefix << '\n';
  for (const auto& v : quest3::linear_variants)
    std::cout << v.id << ' ' << v.label << ' ' << v.tile_m << ' ' << v.tile_n << ' '
              << v.workgroup_x << ' ' << v.workgroup_y << ' ' << v.workgroup_z
              << ' ' << v.buffer_weights << '\n';
}
