// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstring>
namespace xrgpu::operators::extended::detail::half_host {
using uint=uint32_t;
inline float uintBitsToFloat(uint32_t v){float x;std::memcpy(&x,&v,4);return x;}
inline uint32_t floatBitsToUint(float x){uint32_t v;std::memcpy(&v,&x,4);return v;}
// SPDX-License-Identifier: Apache-2.0
// Original IEEE 754 binary16 storage conversions. Integer rounding avoids
// implementation-dependent packHalf overflow/NaN handling in GPU drivers.
inline float xrgpuHalfToFloat(uint halfBits) {
 uint sign=(halfBits&0x8000u)<<16u;
 uint exponent=(halfBits>>10u)&31u;
 uint fraction=halfBits&1023u;
 if(exponent==0u) {
  if(fraction==0u)return uintBitsToFloat(sign);
  uint shifts=0u;
  while((fraction&1024u)==0u){fraction<<=1u;++shifts;}
  return uintBitsToFloat(sign|((113u-shifts)<<23u)|((fraction&1023u)<<13u));
 }
 if(exponent==31u)return uintBitsToFloat(sign|0x7f800000u|(fraction<<13u));
 return uintBitsToFloat(sign|((exponent+112u)<<23u)|(fraction<<13u));
}
inline uint xrgpuFloatToHalf(float value) {
 uint bits=floatBitsToUint(value);
 uint sign=(bits>>16u)&0x8000u;
 uint exponent=(bits>>23u)&255u;
 uint fraction=bits&0x7fffffu;
 if(exponent==255u) {
  if(fraction==0u)return sign|0x7c00u;
  // Keep representable payload bits and produce a nonzero quiet NaN payload.
  return sign|0x7c00u|(fraction>>13u)|0x0200u;
 }
 int halfExponent=int(exponent)-112;
 if(halfExponent>=31)return sign|0x7c00u;
 if(halfExponent<=0) {
  if(halfExponent< -10)return sign;
  uint mantissa=fraction|0x800000u;
  uint shift=uint(14-halfExponent);
  uint truncated=mantissa>>shift;
  uint remainder=mantissa&((1u<<shift)-1u);
  uint halfway=1u<<(shift-1u);
  if(remainder>halfway||(remainder==halfway&&(truncated&1u)!=0u))++truncated;
  // Rounding the largest subnormal may produce the smallest normal (0x0400).
  return sign|truncated;
 }
 uint rounded=fraction>>13u;
 uint remainder=fraction&0x1fffu;
 if(remainder>0x1000u||(remainder==0x1000u&&(rounded&1u)!=0u))++rounded;
 // Carry through the exponent, including finite overflow to infinity.
 return sign|(uint(halfExponent)<<10u)+rounded;
}

// JSON literals are binary64 numbers; round them directly to binary16 to avoid
// double rounding at a half midpoint after an intermediate binary32 cast.
inline uint32_t double_to_half(double value){
 uint64_t bits;std::memcpy(&bits,&value,8);uint32_t sign=uint32_t((bits>>48)&0x8000u);
 uint32_t exponent=uint32_t((bits>>52)&2047u);uint64_t fraction=bits&0x000fffffffffffffULL;
 if(exponent==2047u)return sign|0x7c00u|(fraction?uint32_t(fraction>>42)|0x200u:0u);
 int e=int(exponent)-1008;if(e>=31)return sign|0x7c00u;
 if(e<=0){if(e< -10)return sign;uint64_t m=fraction|0x0010000000000000ULL;uint32_t shift=uint32_t(43-e);uint64_t rounded=m>>shift,remainder=m&((uint64_t(1)<<shift)-1),halfway=uint64_t(1)<<(shift-1);if(remainder>halfway||(remainder==halfway&&(rounded&1)))++rounded;return sign|uint32_t(rounded);}
 uint64_t rounded=fraction>>42,remainder=fraction&((uint64_t(1)<<42)-1),halfway=uint64_t(1)<<41;if(remainder>halfway||(remainder==halfway&&(rounded&1)))++rounded;return sign|((uint32_t(e)<<10)+uint32_t(rounded));
}

}
