// SPDX-License-Identifier: Apache-2.0
#include "common.glsl"
layout(local_size_x=64) in;
uint multiply(uint index) {
 uint m=params[2],n=params[3],k=params[4];
 uint batch=index/(m*n),row=(index/n)%m,col=index%n;
 uint aBase=broadcastOffset(batch,24u)+row*k,bBase=broadcastOffset(batch,40u)+col;
#if DTYPE == 0 || DTYPE == 1
 precise float sum=0.0;
 for(uint j=0u;j<k;++j)sum+=decodeFloat(readA(aBase+j))*decodeFloat(readB(bBase+j*n));
 return encodeFloat(sum);
#else
 uint sum=0u;
 for(uint j=0u;j<k;++j)sum+=readA(aBase+j)*readB(bBase+j*n);
 return sum;
#endif
}
void main() {
 uint word=(gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x)*64u+gl_LocalInvocationIndex;
 uint first=word*LANES;
 if(first>=params[0])return;
 uint result=0u;
 for(uint lane=0u;lane<LANES&&first+lane<params[0];++lane) {
  uint value=multiply(first+lane);
#if LANES == 1
  result=value;
#else
  result|=(value&MASK)<<(lane*BITS);
#endif
 }
 wordsC[word]=result;
}
