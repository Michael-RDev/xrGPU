#version 450 core
${define_required_extensions(STORAGE, DTYPE)}
#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}
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

// Abramowitz-Stegun 7.1.26 evaluates the erf GELU used by BERT.
// FP32 arithmetic throughout, including when storage uses half precision.
// The erf approximation's absolute error is below 1.5e-7; the full
// transformer is additionally checked against its FP32 CPU checkpoints.
vec4 gelu(vec4 x) {
  vec4 z = abs(x) * 0.7071067811865475;
  vec4 t = 1.0 / (1.0 + 0.3275911 * z);
  vec4 p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t;
  vec4 erf_z = sign(x) * (1.0 - p * exp(-z * z));
  return 0.5 * x * (1.0 + erf_z);
}
void main() {
$if STORAGE == "buffer":
  int i = int(gl_GlobalInvocationID.x);
  if (i >= numel) return;
  t_out[i] = T(gelu(vec4(float(t_in[i]))).x);
$else:
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, out_limits.xyz))) return;
  imageStore(t_out, pos, gelu(vec4(texelFetch(t_in, pos, 0))));
}
