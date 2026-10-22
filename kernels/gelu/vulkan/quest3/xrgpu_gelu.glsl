/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define T ${buffer_scalar_type(DTYPE)}

#define op(X, A, B) ${OPERATOR}

${define_active_storage_type(STORAGE)}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_in", DTYPE, STORAGE)}

layout(push_constant) uniform restrict Block {
$if STORAGE == "buffer":
  int numel;
$else:
  ivec4 out_limits;
float minimum;
float maximum;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "activations.h"
// Abramowitz-Stegun 7.1.26, maximum erf approximation error < 1.5e-7.
// Saturate negative tails to avoid cancellation; finite inputs are the v1 contract.
float xrgpu_erf_gelu(float x) {
  if (x < -10.0) return 0.0;
  if (x > 10.0) return x;
  float a = abs(x) * 0.7071067811865475;
  float t = 1.0 / (1.0 + 0.3275911 * a);
  float p = (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t
               - 0.284496736)*t + 0.254829592)*t;
  float tail = p * exp(-a*a);
  return x < 0.0 ? 0.5*x*tail : x*(1.0 - 0.5*tail);
}
vec4 xrgpu_erf_gelu(vec4 x) {
  return vec4(xrgpu_erf_gelu(x.x), xrgpu_erf_gelu(x.y),
              xrgpu_erf_gelu(x.z), xrgpu_erf_gelu(x.w));
}

#ifdef USING_BUFFER

void main() {
  const int i = int(gl_GlobalInvocationID.x);
  if (i >= numel) {
    return;
  }

  float in_val = float(t_in[i]);
  t_out[i] = T(op(in_val, minimum, maximum));
}

#else

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits.xyz))) {
    return;
  }

  VEC4_T in_texel = texelFetch(t_in, pos, 0);
  imageStore(t_out, pos, VEC4_T(op(in_texel, minimum, maximum)));
}

#endif
