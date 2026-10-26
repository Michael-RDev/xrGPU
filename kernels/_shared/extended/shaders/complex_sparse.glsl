// SPDX-License-Identifier: Apache-2.0
// Original native sparse attention: GPU QKV transform, CSR attention, cache copy.
#ifndef XRGPU_COMPLEX_SPARSE
#define XRGPU_COMPLEX_SPARSE
float sparseRead(uint slot,uint index){uint bits=readU(slot,index);return dtype(slot)==D_F16?xrgpuHalfToFloat(bits):uintBitsToFloat(bits);}
void sparseWrite(uint slot,uint index,float value){writeU(slot,index,outDtype(slot)==D_F16?xrgpuFloatToHalf(value):floatBitsToUint(value));}
uint sparseKeys(uint b){int value=readI(8u,b);if(value<0){fail(56u);return 0u;}return uint(value);}
uint sparsePast(uint b){uint keys=sparseKeys(b);if(readU(7u,0u)==arg(6u))return 0u;return keys>arg(6u)?keys-arg(6u):0u;}
float sparseRaw(uint role,uint b,uint h,uint token,uint d){uint heads=role==0u?arg(1u):arg(2u),width=arg(8u),seq=arg(6u);if(has(1u))return sparseRead(role,((b*seq+token)*heads+h)*width+d);uint offset=role==0u?0u:role==1u?arg(1u)*width:(arg(1u)+arg(2u))*width;return sparseRead(0u,(b*seq+token)*dim(0u,2u)+offset+h*width+d);}
uint sparseOffset(uint role,uint b,uint h,uint token,uint d){uint heads=role==0u?arg(1u):arg(2u);return arg(28u+role)+((b*arg(6u)+token)*heads+h)*arg(8u)+d;}
void sparseTransform(){uint row=gid(),qRows=arg(10u)*arg(6u)*arg(1u),kvRows=arg(10u)*arg(6u)*arg(2u),role;
 if(row<qRows)role=0u;else if(row<qRows+kvRows){role=1u;row-=qRows;}else if(row<qRows+2u*kvRows){role=2u;row-=qRows+kvRows;}else return;
 uint heads=role==0u?arg(1u):arg(2u),h=row%heads,token=(row/heads)%arg(6u),b=row/(heads*arg(6u)),width=arg(8u),rotary=0u,position=sparsePast(b)+token;
 if(role<2u&&arg(14u)!=0u){if(!has(9u)||!has(10u)||position>=dim(9u,0u)||position>=dim(10u,0u)){fail(54u);return;}rotary=min(width,2u*dim(9u,1u));if((rotary&1u)!=0u){fail(54u);return;}}
 for(uint d=0u;d<width;++d){float value=sparseRaw(role,b,h,token,d);if(d<rotary){uint pair=arg(26u)!=0u?d/2u:d%(rotary/2u),mate=arg(26u)!=0u?(d^1u):(d<rotary/2u?d+rotary/2u:d-rotary/2u);float signValue=arg(26u)!=0u?((d&1u)==0u?-1.0:1.0):(d<rotary/2u?-1.0:1.0);value=value*sparseRead(9u,position*dim(9u,1u)+pair)+signValue*sparseRaw(role,b,h,token,mate)*sparseRead(10u,position*dim(10u,1u)+pair);}
  uint targetType=role==0u?dtype(0u):outDtype(os(role));if(targetType==D_F16)value=xrgpuHalfToFloat(xrgpuFloatToHalf(value));setScratchF(sparseOffset(role,b,h,token,d),value);
 }
}
float sparseValue(uint role,uint b,uint h,uint absoluteT,uint d,uint past){if(absoluteT<past){uint slot=role+2u;if(absoluteT>=dim(slot,2u)){fail(56u);return 0.0;}return sparseRead(slot,((b*arg(2u)+h)*dim(slot,2u)+absoluteT)*arg(8u)+d);}return scratchF(sparseOffset(role,b,h,absoluteT-past,d));}
void sparseAttend(){uint row=gid(),heads=arg(1u),queries=arg(6u);if(row>=arg(10u)*heads*queries)return;uint h=row%heads,qi=(row/heads)%queries,b=row/(heads*queries),kh=h/(heads/arg(2u)),past=sparsePast(b),keys=sparseKeys(b),block=max(1u,arg(20u)),csrRow=(past+qi)/block,layoutId=h%dim(5u,0u),scoreBase=arg(31u)+row*arg(32u);
 if(keys>arg(32u)||csrRow+1u>=dim(5u,1u)){fail(56u);return;}int beginI=readI(5u,layoutId*dim(5u,1u)+csrRow),endI=readI(5u,layoutId*dim(5u,1u)+csrRow+1u);if(beginI<0||endI<beginI||uint(endI)>dim(6u,1u)){fail(56u);return;}uint begin=uint(beginI),end=uint(endI);float maximum=uintBitsToFloat(0xff800000u);
 for(uint token=0u;token<keys;++token){bool permitted=false;for(uint j=begin;j<end;++j){int column=readI(6u,layoutId*dim(6u,1u)+j);if(column<0){fail(56u);return;}permitted=permitted||uint(column)==token/block;}float score=uintBitsToFloat(0xff800000u);if(permitted&&token<=past+qi){score=0.0;for(uint d=0u;d<arg(8u);++d)score+=scratchF(sparseOffset(0u,b,h,qi,d))*sparseValue(1u,b,kh,token,d,past);score*=argF(3u);}setScratchF(scoreBase+token,score);maximum=max(maximum,score);}
 float denominator=0.0;for(uint token=0u;token<keys;++token){float score=scratchF(scoreBase+token),weight=isinf(maximum)?(maximum>0.0&&score==maximum?1.0:0.0):exp(score-maximum);setScratchF(scoreBase+token,weight);denominator+=weight;}
 for(uint d=0u;d<arg(9u);++d){float result=0.0;for(uint token=0u;token<keys;++token)result+=scratchF(scoreBase+token)*sparseValue(2u,b,kh,token,d,past);sparseWrite(os(0u),row*arg(9u)+d,denominator>0.0?result/denominator:0.0);}
}
void sparseCaches(){if(gid()!=0u)return;for(uint role=1u;role<=2u;++role){uint outputSlot=os(role),inputSlot=role+2u;if(outputSlot==0xffffffffu)continue;for(uint i=0u;i<outSize(outputSlot);++i){if(i<size(inputSlot))writeRaw(outputSlot,i,readRaw(inputSlot,i));else writeU(outputSlot,i,0u);}for(uint b=0u;b<arg(10u);++b){uint past=sparsePast(b),keys=sparseKeys(b);if(keys>outDim(outputSlot,2u)){fail(56u);return;}for(uint token=0u;token<arg(6u)&&past+token<keys;++token)for(uint h=0u;h<arg(2u);++h)for(uint d=0u;d<arg(8u);++d)sparseWrite(outputSlot,((b*arg(2u)+h)*outDim(outputSlot,2u)+past+token)*arg(8u)+d,scratchF(sparseOffset(role,b,h,token,d)));}}}
void sparseMain(){
#if XRGPU_SPARSE_PHASE == 0
 sparseTransform();
#elif XRGPU_SPARSE_PHASE == 1
 sparseAttend();
#elif XRGPU_SPARSE_PHASE == 2
 sparseCaches();
#else
 fail(56u);
#endif
}
#endif
