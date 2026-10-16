#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace xrgpu {
struct DeviceCapabilities {
  std::string_view model;
  uint32_t vendor, max_invocations, max_workgroup_x, max_workgroup_y;
  uint32_t device_id = 0, api_version = 0;
};

// Variant IDs are local to a kernel set; zero always means stock dispatch.
struct LinearVariant {
  int id;
  const char* label;
  int tile_m, tile_n;
  uint32_t workgroup_x, workgroup_y, workgroup_z;
  bool buffer_weights;
  bool (*shape_override)(int64_t, int64_t, int64_t, int64_t, bool) = nullptr;
};

struct LinearKernelSet {
  const char* id;
  const char* shader_prefix;
  const LinearVariant* variants;
  size_t variant_count;
  bool (*supports)(int64_t batch, int64_t m, int64_t k, int64_t n, bool fp32);

  constexpr const LinearVariant* variant(int id) const {
    if (id <= 0) return nullptr;
    for (size_t i = 0; i < variant_count; ++i)
      if (variants[i].id == id) return &variants[i];
    return nullptr;
  }
};

struct DeviceProfile {
  const char* id;
  const char* label;
  bool (*matches)(const DeviceCapabilities&);
  const LinearKernelSet* linear;
};
}
