#pragma once
#include "kernels/linear/vulkan/quest3/dispatch.h"

namespace xrgpu::quest3 {
// This permits experiments, not automatic use in the search demo. Measurements
// must still pass independently for this device, driver, and workload.
inline constexpr bool matches(const DeviceCapabilities& caps) {
  return caps.model == "Quest 3" && caps.vendor == 0x5143 &&
      caps.max_invocations >= 32 && caps.max_workgroup_x >= 8 &&
      caps.max_workgroup_y >= 4;
}
inline constexpr DeviceProfile profile = {"quest3", "Quest 3", matches, &linear};
}
