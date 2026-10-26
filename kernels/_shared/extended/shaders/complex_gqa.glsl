// SPDX-License-Identifier: Apache-2.0
// Original three-pass native GQA. Precomputed Q/K/V avoids recursively expanding
// normalization, rotary and cache quantization inside every attention product.
#ifndef XRGPU_COMPLEX_GQA
#define XRGPU_COMPLEX_GQA
float gqaRead(uint slot,uint index){uint bits=readU(slot,index);return dtype(slot)==D_F16?xrgpuHalfToFloat(bits):uintBitsToFloat(bits);}
void gqaWrite(uint slot,uint index,float value){if(outDtype(slot)==D_F16)writeU(slot,index,xrgpuFloatToHalf(value));else writeU(slot,index,floatBitsToUint(value));}
float gqaRaw(uint role,uint b,uint h,uint token,uint d){uint heads=role==0u?arg(1u):arg(2u),width=role==2u?arg(9u):arg(8u),seq=role==0u?arg(6u):arg(7u);return gqaRead(role,((b*seq+token)*heads+h)*width+d);}
uint gqaOffset(uint role,uint b,uint h,uint token,uint d){uint heads=role==0u?arg(1u):arg(2u),width=role==2u?arg(9u):arg(8u),seq=role==0u?arg(6u):arg(7u);return arg(28u+role)+((b*seq+token)*heads+h)*width+d;}
void gqaTransform(){
 uint row=gid(),qRows=arg(10u)*arg(6u)*arg(1u),kvRows=arg(10u)*arg(7u)*arg(2u),role;
 if(row<qRows)role=0u;else if(row<qRows+kvRows){role=1u;row-=qRows;}else if(row<qRows+2u*kvRows){role=2u;row-=qRows+kvRows;}else return;
 uint heads=role==0u?arg(1u):arg(2u),seq=role==0u?arg(6u):arg(7u),width=role==2u?arg(9u):arg(8u),h=row%heads,token=(row/heads)%seq,b=row/(heads*seq),normSlot=role==0u?13u:14u;
 float inverse=1.0;
 if(role<2u&&has(normSlot)){float square=0.0;for(uint d=0u;d<width;++d){float x=gqaRaw(role,b,h,token,d);square+=x*x;}inverse=inversesqrt(square/float(width)+argF(13u));}
 uint rotary=0u,position=complexCachePast(b)+token;
 if(role<2u&&arg(14u)!=0u){if(!has(7u)||!has(8u)||position>=dim(7u,0u)||position>=dim(8u,0u)){fail(54u);return;}rotary=min(width,2u*dim(7u,1u));if((rotary&1u)!=0u){fail(54u);return;}}
 for(uint d=0u;d<width;++d){float value=gqaRaw(role,b,h,token,d);if(role<2u&&has(normSlot))value*=inverse*gqaRead(normSlot,d);
  if(d<rotary){uint pair=arg(26u)!=0u?d/2u:d%(rotary/2u),mate=arg(26u)!=0u?(d^1u):(d<rotary/2u?d+rotary/2u:d-rotary/2u);float other=gqaRaw(role,b,h,token,mate);if(has(normSlot))other*=inverse*gqaRead(normSlot,mate);float signValue=arg(26u)!=0u?((d&1u)==0u?-1.0:1.0):(d<rotary/2u?-1.0:1.0);value=value*gqaRead(7u,position*dim(7u,1u)+pair)+signValue*other*gqaRead(8u,position*dim(8u,1u)+pair);}
  if(role>0u)value=complexCacheRoundtrip(role,h,d,value);
  setScratchF(gqaOffset(role,b,h,token,d),value);
 }
}
float gqaKeyValue(uint role,uint b,uint h,uint absoluteT,uint d,uint past){if(absoluteT<past)return complexCacheRead(role,b,h,absoluteT,d);return scratchF(gqaOffset(role,b,h,absoluteT-past,d));}
float gqaWeight(float score,float maximum){return isinf(maximum)?(maximum>0.0&&score==maximum?1.0:0.0):exp(score-maximum);}
void gqaAttend(){
 uint row=gid(),heads=arg(1u),queries=arg(6u);if(row>=arg(10u)*heads*queries)return;
 uint h=row%heads,qi=(row/heads)%queries,b=row/(heads*queries),kh=h/(heads/arg(2u)),past=complexCachePast(b),keys=complexCacheTotal(b),first=0u,scoreBase=arg(31u)+row*arg(32u);
 if(arg(24u)!=0u&&has(3u)&&past>dim(3u,2u))first=past-dim(3u,2u);
 if(int(arg(15u))>=0&&past+qi>arg(15u))first=max(first,past+qi-arg(15u));
 if(keys<first||keys-first>arg(32u)){fail(55u);return;}
 float maximum=uintBitsToFloat(0xff800000u);
 for(uint t=first;t<keys;++t){float score=uintBitsToFloat(0xff800000u);
  if((arg(4u)==0u||t<=past+qi)&&(int(arg(15u))<0||t+arg(15u)>=past+qi)){
   score=0.0;for(uint d=0u;d<arg(8u);++d)score+=scratchF(gqaOffset(0u,b,h,qi,d))*gqaKeyValue(1u,b,kh,t,d,past);score*=argF(3u);
   if(argF(12u)>0.0)score=argF(12u)*tanh(score/argF(12u));
   if(has(9u))score+=gqaRead(9u,broad4(9u,uvec4(b,h,qi,t)));
  }
  setScratchF(scoreBase+t-first,score);maximum=max(maximum,score);
 }
 if(has(10u))maximum=max(maximum,gqaRead(10u,h));if(arg(25u)==1u)maximum=max(maximum,0.0);
 float denominator=0.0;
 for(uint t=first;t<keys;++t){float weight=gqaWeight(scratchF(scoreBase+t-first),maximum);setScratchF(scoreBase+t-first,weight);denominator+=weight;}
 if(has(10u))denominator+=gqaWeight(gqaRead(10u,h),maximum);if(arg(25u)==1u)denominator+=gqaWeight(0.0,maximum);
 for(uint d=0u;d<arg(9u);++d){float value=0.0;for(uint t=first;t<keys;++t)value+=scratchF(scoreBase+t-first)*gqaKeyValue(2u,b,kh,t,d,past);gqaWrite(os(0u),row*arg(9u)+d,denominator>0.0?value/denominator:0.0);}
}
void gqaCaches(){if(gid()!=0u)return;for(uint role=1u;role<=2u;++role){complexCacheCopy(role);uint width=role==1u?arg(8u):arg(9u);for(uint b=0u;b<arg(10u);++b){uint past=complexCachePast(b),total=complexCacheTotal(b);for(uint h=0u;h<arg(2u);++h)for(uint t=0u;t<arg(7u)&&past+t<total;++t)for(uint d=0u;d<width;++d)complexCacheWrite(role,b,h,past+t,d,scratchF(gqaOffset(role,b,h,t,d)));}}}
void gqaMain(){
#if defined(XRGPU_GQA_PHASE) && XRGPU_GQA_PHASE == 0
 gqaTransform();
#elif defined(XRGPU_GQA_PHASE) && XRGPU_GQA_PHASE == 1
 gqaAttend();
#elif defined(XRGPU_GQA_PHASE) && XRGPU_GQA_PHASE == 2
 gqaCaches();
#else
 if(arg(27u)==0u)gqaTransform();else if(arg(27u)==1u)gqaAttend();else if(arg(27u)==2u)gqaCaches();else fail(55u);
#endif
}
#endif
