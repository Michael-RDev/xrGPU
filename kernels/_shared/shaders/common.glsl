// SPDX-License-Identifier: Apache-2.0
// Original Vulkan implementation of the named tensor contracts. No WebGPU runtime.
layout(set=0,binding=0,std430) readonly buffer A { uint wordsA[]; };
layout(set=0,binding=1,std430) readonly buffer B { uint wordsB[]; };
layout(set=0,binding=2,std430) writeonly buffer C { uint wordsC[]; };
layout(set=0,binding=3,std430) readonly buffer P { uint params[]; };
// DTYPE: 0 f32, 1 f16, 2 i32, 3 u32, 4 i8, 5 u8, 6 i16.
#if DTYPE == 1 || DTYPE == 6
#define LANES 2
#define BITS 16u
#define MASK 65535u
#elif DTYPE == 4 || DTYPE == 5
#define LANES 4
#define BITS 8u
#define MASK 255u
#else
#define LANES 1
#define BITS 32u
#define MASK 4294967295u
#endif
uint readA(uint i) {
#if LANES == 1
 return wordsA[i];
#else
 return (wordsA[i/LANES] >> ((i%LANES)*BITS)) & MASK;
#endif
}
uint readB(uint i) {
#if LANES == 1
 return wordsB[i];
#else
 return (wordsB[i/LANES] >> ((i%LANES)*BITS)) & MASK;
#endif
}
#if DTYPE == 1
#include "half.glsl"
#endif
float decodeFloat(uint x) {
#if DTYPE == 1
 return xrgpuHalfToFloat(x);
#else
 return uintBitsToFloat(x);
#endif
}
uint encodeFloat(float x) {
#if DTYPE == 1
 return xrgpuFloatToHalf(x);
#else
 return floatBitsToUint(x);
#endif
}
uint broadcastOffset(uint index,uint base) {
 uint offset=0u;
 for(uint r=params[1];r>0u;--r) {
  uint d=params[8u+r-1u];
  offset+=(index%d)*params[base+r-1u];index/=d;
 }
 return offset;
}
