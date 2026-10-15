#pragma once
#include "devices/quest3/profile.h"

namespace xrgpu {
// Explicit registration: adding shader files alone never enables a device.
inline constexpr const DeviceProfile* device_profiles[] = {&quest3::profile};

inline constexpr const DeviceProfile* find_device_profile(const DeviceCapabilities& caps) {
  const DeviceProfile* match = nullptr;
  for (const auto* profile : device_profiles) {
    if (!profile->matches(caps)) continue;
    if (match) return nullptr;  // Ambiguous device identity must use stock.
    match = profile;
  }
  return match;
}

inline constexpr const DeviceProfile* profile_by_id(std::string_view id) {
  for (const auto* profile : device_profiles)
    if (id == profile->id) return profile;
  return nullptr;
}
}
