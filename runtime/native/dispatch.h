#pragma once
// The legacy filename is retained for the pinned backend's timestamp hook.
// Kernel selection lives exclusively in integration/xrgpu_kernel_package.h.
namespace xrgpu {
inline thread_local bool gpu_timestamps_supported = false;
}
