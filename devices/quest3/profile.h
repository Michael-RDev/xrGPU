#pragma once
#include "runtime/native/profiles.h"

namespace xrgpu::quest3 {
// Stable demo IDs map to candidates in kernels/linear/contract.json. The current
// shaders produce four columns per invocation, not the old eight-column tile.
inline constexpr bool supports_linear(int64_t batch, int64_t m, int64_t k,
                                      int64_t n, bool fp32) {
  return fp32 && batch == 1 && m > 0 && k > 0 && n > 0;
}
inline constexpr LinearVariant linear_variants[] = {
    {1, "m4_wg8x4_texture2d", 4, 4, 8, 4, 1, false},
    {2, "m2_wg8x4_texture2d", 2, 4, 8, 4, 1, false},
    {3, "m4_wg8x4_buffer", 4, 4, 8, 4, 1, true},
};
inline constexpr LinearKernelSet linear = {
    "quest3.linear.vulkan.fp32.v2", "xrgpu_linear", linear_variants,
    sizeof(linear_variants) / sizeof(linear_variants[0]), supports_linear};
// This permits experiments, not automatic use in the search demo. Measurements
// must still pass independently for this device, driver, and workload.
inline constexpr bool matches(const DeviceCapabilities& caps) {
  return caps.model == "Quest 3" && caps.vendor == 0x5143 &&
      caps.max_invocations >= 32 && caps.max_workgroup_x >= 8 &&
      caps.max_workgroup_y >= 4;
}
inline constexpr DeviceProfile profile = {"quest3", "Quest 3", matches, &linear};
}
