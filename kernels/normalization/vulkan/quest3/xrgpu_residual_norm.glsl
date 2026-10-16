#version 450 core
#define PRECISION ${PRECISION}
layout(std430) buffer;
${layout_declare_tensor(B, "w", "out_data", "float", "buffer")}
${layout_declare_tensor(B, "r", "in_data", "float", "buffer")}
${layout_declare_tensor(B, "r", "residual", "float", "buffer")}
${layout_declare_tensor(B, "r", "weight", "float", "buffer")}
${layout_declare_tensor(B, "r", "bias", "float", "buffer")}
layout(push_constant) uniform Params { int rows; int width; float eps; int unused; };
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
shared float values[384];
shared float sums[64];
void reduce_sum(int lane) {
  barrier();
  for(int step=32;step>0;step>>=1) {
    if(lane<step) sums[lane]+=sums[lane+step];
    barrier();
  }
}
void main() {
  int lane=int(gl_LocalInvocationID.x), row=int(gl_WorkGroupID.y), base=row*width;
  float sum=0.0;
  for(int i=lane;i<width;i+=64) {
    float value=in_data[base+i]+residual[base+i];
    values[i]=value; sum+=value;
  }
  sums[lane]=sum; reduce_sum(lane);
  float mean=sums[0]/float(width), square_sum=0.0;
  for(int i=lane;i<width;i+=64) {
    float diff=values[i]-mean; square_sum=fma(diff,diff,square_sum);
  }
  barrier();
  sums[lane]=square_sum; reduce_sum(lane);
  float inv_std=inversesqrt(sums[0]/float(width)+eps);
  for(int i=lane;i<width;i+=64)
    out_data[base+i]=(values[i]-mean)*inv_std*weight[i]+bias[i];
}
