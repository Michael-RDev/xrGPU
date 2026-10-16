#version 450 core
#define PRECISION ${PRECISION}
// Static batch-one MiniLM attention. One workgroup owns one query and head.
// QK, masked stable softmax, and AV stay in shared memory in one dispatch.
layout(std430) buffer;
${layout_declare_tensor(B, "w", "output_data", "float", "buffer")}
${layout_declare_tensor(B, "r", "qkv", "float", "buffer")}
${layout_declare_tensor(B, "r", "mask", "float", "buffer")}
layout(push_constant) uniform Params { int length; int width; int heads; int unused; };
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
shared float query[32];
shared float scores[256];
shared float reduction[64];
void main() {
  int lane=int(gl_LocalInvocationID.x);
  int row=int(gl_WorkGroupID.y), head=int(gl_WorkGroupID.z);
  int base=head*32, stride=3*width;
  if(lane<32) query[lane]=qkv[row*stride+base+lane];
  barrier();
  float local_max=-1.0e30;
  for(int key=lane;key<length;key+=64) {
    float value=0.0;
    for(int d=0;d<32;++d) value=fma(query[d],qkv[key*stride+width+base+d],value);
    value=value*0.1767766952966369+(1.0-mask[key])*-10000.0;
    scores[key]=value; local_max=max(local_max,value);
  }
  reduction[lane]=local_max;
  barrier();
  for(int step=32;step>0;step>>=1) {
    if(lane<step) reduction[lane]=max(reduction[lane],reduction[lane+step]);
    barrier();
  }
  float max_score=reduction[0], total=0.0;
  for(int key=lane;key<length;key+=64) {
    float value=exp(scores[key]-max_score);
    scores[key]=value; total+=value;
  }
  // All lanes finish reading the shared maximum before overwriting it.
  barrier();
  reduction[lane]=total;
  barrier();
  for(int step=32;step>0;step>>=1) {
    if(lane<step) reduction[lane]+=reduction[lane+step];
    barrier();
  }
  if(lane<32) {
    float result=0.0;
    for(int key=0;key<length;++key)
      result=fma(scores[key]/reduction[0],qkv[key*stride+2*width+base+lane],result);
    output_data[row*width+base+lane]=result;
  }
}
