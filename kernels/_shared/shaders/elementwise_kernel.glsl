// SPDX-License-Identifier: Apache-2.0
#include "common.glsl"
#ifdef UNARY
#include "unary_ops.glsl"
#else
#include "binary_ops.glsl"
#endif
layout(local_size_x=64) in;
void main() {
 uint word=(gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x)*64u+gl_LocalInvocationIndex;
 uint first=word*LANES;
 if(first>=params[0])return;
 uint result=0u;
 for(uint lane=0u;lane<LANES&&first+lane<params[0];++lane) {
  uint i=first+lane;
#ifdef UNARY
  uint value=evaluateUnary(readA(i));
#else
  uint value=evaluateBinary(readA(broadcastOffset(i,24u)),readB(broadcastOffset(i,40u)));
#endif
#if LANES == 1
  result=value;
#else
  result|=(value&MASK)<<(lane*BITS);
#endif
 }
 wordsC[word]=result;
}
