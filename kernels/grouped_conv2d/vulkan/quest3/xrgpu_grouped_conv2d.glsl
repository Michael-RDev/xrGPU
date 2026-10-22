#version 450 core
#define PRECISION ${PRECISION}
#include "indexing_utils.h"
layout(std430) buffer;
${layout_declare_tensor(0, "w", "t_out", "float", "texture3d")}
${layout_declare_tensor(1, "r", "t_in", "float", "texture3d")}
${layout_declare_tensor(2, "r", "t_kernel", "float", "texture2d")}
${layout_declare_tensor(3, "r", "t_bias", "float", "texture2d")}
${layout_declare_ubo(4, "ivec3", "out_limits")}
${layout_declare_ubo(5, "ivec4", "in_sizes")}
${layout_declare_ubo(6, "ivec2", "kernel_size", "ivec2", "stride", "ivec2", "padding", "ivec2", "dilation")}
${layout_declare_ubo(7, "ivec2", "overlay_region", "int", "in_group_size")}
${layout_declare_ubo(8, "float", "out_min", "float", "out_max")}
layout(push_constant) uniform restrict Block { int input_channels; int output_channels; };
layout(local_size_x_id=0, local_size_y_id=1, local_size_z_id=2) in;
layout(constant_id=3) const int ngroups=1;
void main() {
  ivec3 pos=ivec3(gl_GlobalInvocationID);
  if(any(greaterThanEqual(pos,out_limits))) return;
  ivec2 origin=pos.xy*stride-padding;
  vec4 result=vec4(0);
  // Each output lane can belong to a different group. Weight prepacking pads
  // the group-local input channels, not the global input-channel coordinate.
  int ci_group=input_channels/ngroups;
  int co_group=output_channels/ngroups;
  for(int lane=0;lane<4;lane++) {
    int oc=4*pos.z+lane;
    if(oc>=output_channels) continue;
    int group=oc/co_group;
    float sum=texelFetch(t_bias,ivec2(pos.z,0),0)[lane];
    for(int ic=0;ic<ci_group;ic++) {
      int channel=group*ci_group+ic;
      for(int ky=0;ky<kernel_size.y;ky++) {
        int y=origin.y+ky*dilation.y;
        if(y<0||y>=in_sizes.y) continue;
        for(int kx=0;kx<kernel_size.x;kx++) {
          int x=origin.x+kx*dilation.x;
          if(x<0||x>=in_sizes.x) continue;
          float a=texelFetch(t_in,ivec3(x,y,channel/4),0)[channel%4];
          ivec2 wp=ivec2((ic/4)*kernel_size.x*4+kx*4+ic%4,pos.z*kernel_size.y+ky);
          sum=fma(a,texelFetch(t_kernel,wp,0)[lane],sum);
        }
      }
    }
    result[lane]=${OUTPUT};
  }
  imageStore(t_out,pos,result);
}
