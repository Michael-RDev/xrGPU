// SPDX-License-Identifier: Apache-2.0
#ifndef XRGPU_EXTENDED_MATH
#define XRGPU_EXTENDED_MATH
#define DTYPE 0
#define OPCODE 43
#define BITS 32u
#define MASK 4294967295u
float decodeFloat(uint a){return uintBitsToFloat(a);}
uint encodeFloat(float x){return floatBitsToUint(x);}
#define evaluateUnary extendedLegacyUnary
#include "unary_ops.glsl"
#undef evaluateUnary
#undef DTYPE
#undef OPCODE
#undef BITS
#undef MASK
float unaryLog(float x){return unaryLogBits(floatBitsToUint(x),0);}
#endif
