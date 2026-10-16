#pragma once
#include <cstdint>
#include "runtime/native/profiles.h"

// Set only on the inference worker, before loading a model. Each graph captures
// its dispatch at load time. The rendering thread never changes this state.
namespace xrgpu {
inline thread_local int experiment_variant = 0;
inline thread_local const DeviceProfile* experiment_profile = nullptr;
inline thread_local uint64_t selected_nodes = 0;
inline thread_local bool gpu_timestamps_supported = false;
inline const LinearVariant* selected_variant() {
  return experiment_profile && experiment_profile->linear
      ? experiment_profile->linear->variant(experiment_variant) : nullptr;
}
inline bool eligible(int64_t batch, int64_t m, int64_t k, int64_t n, bool fp32) {
  const auto* variant=selected_variant();
  return variant && (variant->shape_override ? variant->shape_override(batch,m,k,n,fp32) :
      experiment_profile->linear->supports(batch, m, k, n, fp32));
}
// Called by the delegate only after eligible() passes.
inline int tile_m() { return selected_variant()->tile_m; }
inline int tile_n() { return selected_variant()->tile_n; }
inline uint32_t workgroup_x() { return selected_variant()->workgroup_x; }
inline uint32_t workgroup_y() { return selected_variant()->workgroup_y; }
inline uint32_t workgroup_z() { return selected_variant()->workgroup_z; }
inline bool buffer_weights() { return selected_variant()->buffer_weights; }
inline const char* shader_prefix() { return experiment_profile->linear->shader_prefix; }
}
