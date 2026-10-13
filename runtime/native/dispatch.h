#pragma once
#include <cstdint>

// Set only on the inference worker, before loading a model. Each graph captures
// its dispatch at load time. The rendering thread never changes this state.
namespace xrgpu {
inline thread_local int experiment_variant = 0;
inline thread_local uint64_t selected_nodes = 0;
inline thread_local bool gpu_timestamps_supported = false;
inline bool eligible(int64_t batch, int64_t m, int64_t k, int64_t n, bool fp32) {
  return experiment_variant >= 1 && experiment_variant <= 3 && fp32 && batch == 1 &&
      (m == 32 || m == 128 || m == 256) &&
      ((k == 384 && n == 1536) || (k == 1536 && n == 384));
}
inline int tile_m() { return experiment_variant == 2 ? 2 : 4; }
inline int workgroup_x() { return experiment_variant == 1 ? 4 : 8; }
}
