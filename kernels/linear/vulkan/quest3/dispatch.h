#pragma once
#include "runtime/native/profiles.h"

namespace xrgpu::quest3 {
inline constexpr bool supports_linear(int64_t batch, int64_t m, int64_t k,
                                      int64_t n, bool fp32) {
  return fp32 && batch == 1 && (m == 32 || m == 128 || m == 256) &&
      ((k == 384 && n == 1536) || (k == 1536 && n == 384));
}
inline constexpr LinearVariant linear_variants[] = {
    {1, "4×8 tiles", 4, 8, 4, 4, 1, false},
    {2, "2×8 tiles", 2, 8, 8, 4, 1, false},
    {3, "4×8 tiles, buffer weights", 4, 8, 8, 4, 1, true},
};
inline constexpr LinearKernelSet linear = {
    "quest3.linear.vulkan.fp32.v1", "xrgpu_linear", linear_variants,
    sizeof(linear_variants) / sizeof(linear_variants[0]), supports_linear};
}
