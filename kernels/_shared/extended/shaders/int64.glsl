// SPDX-License-Identifier: Apache-2.0
// Exact two-word integer casts; no shaderInt64 or shaderFloat64 feature needed.
float extendedInt64ToFloat(uvec2 value){
 uint sign=value.y&0x80000000u;
 if(sign!=0u){value.x=~value.x+1u;value.y=~value.y+(value.x==0u?1u:0u);}
 int leading=value.y!=0u?32+findMSB(value.y):findMSB(value.x);
 if(leading<0)return 0.0;
 if(leading<=23){float result=float(value.x);return sign!=0u?-result:result;}
 uint shift=uint(leading-23),mantissa,remainderHi=0u,remainderLo=0u,halfHi=0u,halfLo=0u;
 if(shift<32u){mantissa=(value.x>>shift)|(value.y<<(32u-shift));remainderLo=value.x&((1u<<shift)-1u);halfLo=1u<<(shift-1u);}
 else{uint highShift=shift-32u;mantissa=value.y>>highShift;remainderLo=value.x;if(highShift==0u)halfLo=0x80000000u;else{remainderHi=value.y&((1u<<highShift)-1u);halfHi=1u<<(highShift-1u);}}
 bool greater=remainderHi>halfHi||(remainderHi==halfHi&&remainderLo>halfLo);
 bool equal=remainderHi==halfHi&&remainderLo==halfLo;
 if(greater||(equal&&(mantissa&1u)!=0u))++mantissa;
 if(mantissa==0x01000000u){mantissa>>=1u;++leading;}
 return uintBitsToFloat(sign|((uint(leading)+127u)<<23u)|(mantissa&0x7fffffu));
}
uvec2 extendedFloatToInt64(float value){
 uint bits=floatBitsToUint(value),sign=bits>>31u,rawExponent=(bits>>23u)&255u,mantissa=(bits&0x7fffffu)|0x800000u;
 int exponent=int(rawExponent)-127;
 if(rawExponent==255u||exponent>63||(exponent==63&&(sign==0u||mantissa!=0x800000u))){fail(3u);return uvec2(0u);}
 if(exponent<0)return uvec2(0u);
 int shift=exponent-23;uvec2 result;
 if(shift>=32)result=uvec2(0u,mantissa<<uint(shift-32));
 else if(shift>=0)result=uvec2(mantissa<<uint(shift),shift==0?0u:mantissa>>uint(32-shift));
 else result=uvec2(mantissa>>uint(-shift),0u);
 if(sign!=0u){result.x=~result.x+1u;result.y=~result.y+(result.x==0u?1u:0u);}
 return result;
}
