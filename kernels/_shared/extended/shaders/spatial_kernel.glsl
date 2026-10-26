// SPDX-License-Identifier: Apache-2.0
#ifndef SPATIAL_OPCODE
#define SPATIAL_OPCODE arg(0u)
#endif
// SPDX-License-Identifier: Apache-2.0
#include "framework.glsl"
#include "../spatial_ids.h"
#include "spatial_math.glsl"
layout(local_size_x=64) in;
void coordinates(uint index,uint slot,bool outputTensor,out uint coords[16]) {
 uint r=outputTensor?outRank(slot):rank(slot);
 for(uint d=0u;d<16u;++d)coords[d]=0u;
 for(uint d=r;d>0u;--d){uint n=outputTensor?outDim(slot,d-1u):dim(slot,d-1u);coords[d-1u]=n==0u?0u:index%n;index=n==0u?0u:index/n;}
}
uint broadcastIndex(uint slot,uint outputSlot,uint index) {
 uint coords[16];coordinates(index,outputSlot,true,coords);uint result=0u,r=rank(slot),outR=outRank(outputSlot);
 for(uint d=0u;d<r;++d){uint c=outR>=r?coords[outR-r+d]:0u;result+=dim(slot,d)==1u?0u:c*stride(slot,d);}return result;
}
bool signedType(uint t){return t==D_I32||t==D_I16||t==D_I8||t==D_I64;}
bool floatingType(uint t){return t==D_F32||t==D_F16;}
uint reductionCount(){uint count=1u;for(uint d=0u;d<rank(0u);++d)if(arg(16u+d)!=0u)count*=dim(0u,d);return count;}
uint reductionBase(uint outputSlot,uint index) {
 uint coords[16];coordinates(index,outputSlot,true,coords);uint base=0u,j=0u;
 for(uint d=0u;d<rank(0u);++d)if(arg(16u+d)==0u){uint c=arg(2u)!=0u?coords[d]:coords[j++];base+=c*stride(0u,d);}return base;
}
uint reductionOffset(uint base,uint index) {
 for(uint d=rank(0u);d>0u;--d)if(arg(16u+d-1u)!=0u){uint n=dim(0u,d-1u);base+=(n==0u?0u:index%n)*stride(0u,d-1u);index=n==0u?0u:index/n;}return base;
}
vec2 spatialIntegerDF(int value){return unaryDFAdd(vec2(float(value>>16)*65536.0,0.0),vec2(float(uint(value)&65535u),0.0));}
vec2 spatialSqrtDF(vec2 value){float root=sqrt(value.x);if(root==0.0||isinf(root)||isnan(root))return vec2(root,0.0);vec2 residual=unaryDFAdd(value,-unaryDFMul(vec2(root,0.0),vec2(root,0.0)));return unaryDFNormalize(root,(residual.x+residual.y)/(2.0*root));}
// Keep a significand and binary exponent separate, including subnormal inputs.
// This avoids driver FTZ and intermediate overflow in quantization and scans.
vec2 spatialFloatParts(float value,out int power){uint bits=floatBitsToUint(value),fraction=bits&0x7fffffu;int e=int((bits>>23u)&255u);power=e-127;if(e==0){if(fraction==0u){power=0;return vec2(value,0.0);}int shift=23-findMSB(fraction);fraction=(fraction<<uint(shift))&0x7fffffu;power=-126-shift;}return vec2(uintBitsToFloat((bits&0x80000000u)|0x3f800000u|fraction),0.0);}
void reduceValue(uint outputSlot,uint index) {
 uint op=SPATIAL_OPCODE,count=reductionCount(),base=reductionBase(outputSlot,index),type=dtype(0u);
 if(!floatingType(type)&&(op==S_REDUCE_SUM||op==S_REDUCE_PROD||op==S_REDUCE_SQUARE||op==S_REDUCE_L1||op==S_REDUCE_MEAN||op==S_REDUCE_MIN||op==S_REDUCE_MAX)){
  if(op==S_REDUCE_MIN||op==S_REDUCE_MAX){uint bits=type==D_I8||type==D_U8?8u:type==D_I16?16u:32u;uint hi=signedType(type)?(1u<<(bits-1u))-1u:bits==32u?0xffffffffu:(1u<<bits)-1u;uint lo=signedType(type)?~hi:0u;uint accum=op==S_REDUCE_MIN?hi:lo;
   for(uint j=0u;j<count;++j){uint at=reductionOffset(base,j),v=signedType(type)?uint(readI(0u,at)):readU(0u,at);accum=op==S_REDUCE_MIN?(signedType(type)?uint(min(int(accum),int(v))):min(accum,v)):(signedType(type)?uint(max(int(accum),int(v))):max(accum,v));}writeU(outputSlot,index,accum);return;}
  vec2 accum=vec2(op==S_REDUCE_PROD?1.0:0.0,0.0);
  for(uint j=0u;j<count;++j){uint at=reductionOffset(base,j);vec2 v=signedType(type)?spatialIntegerDF(readI(0u,at)):unaryDFAdd(vec2(float(readU(0u,at)>>16u)*65536.0,0.0),vec2(float(readU(0u,at)&65535u),0.0));
   if(op==S_REDUCE_PROD){if(isinf(accum.x)){if(v.x==0.0)accum=vec2(0.0);else if(v.x<0.0)accum=-accum;}else{if(v.x!=0.0&&abs(accum.x)>3.402823466e38/abs(v.x))accum=vec2((accum.x<0.0)!=(v.x<0.0)?-unaryInfinity():unaryInfinity(),0.0);else accum=unaryDFMul(accum,v);if(isinf(accum.x))accum.y=0.0;}}else if(op==S_REDUCE_SQUARE)accum=unaryDFAdd(accum,unaryDFMul(v,v));else accum=unaryDFAdd(accum,op==S_REDUCE_L1&&v.x<0.0?-v:v);
  }
  if(op==S_REDUCE_MEAN)accum=count==0u?vec2(0.0):unaryDFDiv(accum,vec2(float(count),0.0));
  // ONNX Runtime's widened integer reduction stages saturate the final cast.
  if(type==D_I32){if(accum.x>2147483648.0||(accum.x==2147483648.0&&accum.y>=-1.0)){writeU(outputSlot,index,0x7fffffffu);return;}if(accum.x<-2147483648.0||(accum.x==-2147483648.0&&accum.y<=0.0)){writeU(outputSlot,index,0x80000000u);return;}}
  if(type==D_U32&&(accum.x>4294967296.0||(accum.x==4294967296.0&&accum.y>=-1.0))){writeU(outputSlot,index,0xffffffffu);return;}if(signedType(type)){float high=trunc(accum.x/65536.0);vec2 tail=unaryDFAdd(accum,vec2(-high*65536.0,0.0));int low=int(tail.x);float fraction=(tail.x-float(low))+tail.y;if(accum.x>0.0&&fraction<0.0)--low;else if(accum.x<0.0&&fraction>0.0)++low;writeU(outputSlot,index,uint(int(high))*65536u+uint(low));}else{uint high=uint(accum.x/65536.0);vec2 tail=unaryDFAdd(accum,vec2(-float(high)*65536.0,0.0));writeU(outputSlot,index,high*65536u+uint(tail.x));}return;
 }
 float maximum=-unaryInfinity();if(op==S_REDUCE_LOGSUMEXP){for(uint j=0u;j<count;++j){float x=readF(0u,reductionOffset(base,j));maximum=isnan(x)?x:max(maximum,x);}if(isinf(maximum)||isnan(maximum)){writeF(outputSlot,index,maximum);return;}}
 if(op==S_REDUCE_MIN||op==S_REDUCE_MAX){float result=count==0u?(op==S_REDUCE_MIN?unaryInfinity():-unaryInfinity()):readF(0u,base);for(uint j=1u;j<count;++j){float x=readF(0u,reductionOffset(base,j));if(op==S_REDUCE_MIN?x<result:x>result)result=x;}writeF(outputSlot,index,result);return;}
 float result=op==S_REDUCE_PROD?1.0:0.0;
 for(uint j=0u;j<count;++j){float x=readF(0u,reductionOffset(base,j));if(isnan(x)){result=x;break;}if(op==S_REDUCE_PROD)result*=x;else if(op==S_REDUCE_L1)result+=abs(x);else if(op==S_REDUCE_L2||op==S_REDUCE_SQUARE)result+=x*x;else if(op==S_REDUCE_LOGSUMEXP)result+=unaryExp(x-maximum);else result+=x;}
 if(op==S_REDUCE_MEAN)result=count==0u?unaryNaN():result/float(count);else if(op==S_REDUCE_L2)result=sqrt(result);else if(op==S_REDUCE_LOGSUM)result=unaryLogBits(floatBitsToUint(result),0);else if(op==S_REDUCE_LOGSUMEXP)result=maximum+unaryLogBits(floatBitsToUint(result),0);
 writeF(outputSlot,index,result);
}
float biasSoftmaxInput(uint index){uint block=arg(9u),rows=size(0u)/block,biasRows=size(1u)/block,row=index/block;uint selected=arg(12u)!=0u?row/(rows/biasRows):row%biasRows;return readF(0u,index)+readF(1u,selected*block+index%block);}
void axisValue(uint outputSlot,uint index) {
 uint op=SPATIAL_OPCODE,axisIndex=arg(8u),n=dim(0u,axisIndex),inner=stride(0u,axisIndex),base=0u,position=0u;
 if(op==S_ARGMAX||op==S_ARGMIN){uint coords[16];coordinates(index,outputSlot,true,coords);uint j=0u;bool keep=outRank(outputSlot)==rank(0u);for(uint d=0u;d<rank(0u);++d)if(d!=axisIndex){uint c=keep?coords[d]:coords[j++];base+=c*stride(0u,d);}}
 else{position=inner==0u?0u:(index/inner)%n;base=index-position*inner;}
 if(op==S_BIASSOFTMAX){n=arg(9u);inner=1u;position=index%n;base=index-position;}
 if(op==S_CUMSUM||op==S_CUMPROD){uint first=arg(3u)!=0u?position:0u,last=arg(3u)!=0u?n:position+1u;if(arg(2u)!=0u){if(arg(3u)!=0u)++first;else --last;}if(floatingType(dtype(0u))){float v=op==S_CUMPROD?1.0:0.0;if(op==S_CUMPROD){vec2 product=vec2(1.0,0.0);int power=0;bool special=false;for(uint j=first;j<last;++j){float x=readF(0u,base+j*inner);if(special||x==0.0||isnan(x)||isinf(x)){if(!special)v=unaryScalePowerOfTwo(product,clamp(power,-10000,10000));v*=x;special=true;}else{int e;vec2 part=spatialFloatParts(x,e);product=unaryDFMul(product,part);power=clamp(power+e,-100000000,100000000);if(abs(product.x)>=2.0){product*=0.5;++power;}}}if(!special)v=unaryScalePowerOfTwo(product,clamp(power,-10000,10000));}else for(uint j=first;j<last;++j)v+=readF(0u,base+j*inner);writeF(outputSlot,index,v);}else{uint v=op==S_CUMPROD?1u:0u;for(uint j=first;j<last;++j){uint x=readU(0u,base+j*inner);v=op==S_CUMPROD?v*x:v+x;}writeU(outputSlot,index,v);}return;}
 if(op==S_ARGMAX||op==S_ARGMIN||op==S_HARDMAX){uint selected=0u;float best=readF(0u,base);int bestI=readI(0u,base);uint bestU=readU(0u,base);for(uint j=1u;j<n;++j){uint at=base+j*inner;bool equal=false,better=false;if(floatingType(dtype(0u))){float x=readF(0u,at);better=op==S_ARGMIN?x<best:x>best;equal=x==best;if(better){best=x;}}else if(signedType(dtype(0u))){int x=readI(0u,at);better=op==S_ARGMIN?x<bestI:x>bestI;equal=x==bestI;if(better)bestI=x;}else{uint x=readU(0u,at);better=op==S_ARGMIN?x<bestU:x>bestU;equal=x==bestU;if(better)bestU=x;}if(better||(equal&&arg(11u)!=0u))selected=j;}
  if(op==S_HARDMAX)writeF(outputSlot,index,!isnan(best)&&position==selected?1.0:0.0);else writeRaw(outputSlot,index,uvec2(selected,0u));return;
 }
 float maximum=-unaryInfinity();for(uint j=0u;j<n;++j){uint at=base+j*inner;float x=op==S_BIASSOFTMAX?biasSoftmaxInput(at):readF(0u,at);maximum=isnan(x)?x:max(maximum,x);}
 if(op==S_BIASSOFTMAX&&maximum==-unaryInfinity()){writeF(outputSlot,index,0.0);return;}
 float denominator=0.0;for(uint j=0u;j<n;++j){uint at=base+j*inner;float x=op==S_BIASSOFTMAX?biasSoftmaxInput(at):readF(0u,at);denominator+=unaryExp(x-maximum);}
 float x=op==S_BIASSOFTMAX?biasSoftmaxInput(index):readF(0u,index);writeF(outputSlot,index,op==S_LOGSOFTMAX?x-maximum-unaryLogBits(floatBitsToUint(denominator),0):unaryExp(x-maximum)/denominator);
}
float stashRound(float value){return arg(13u)==10u?xrgpuHalfToFloat(xrgpuFloatToHalf(value)):value;}
float normInput(uint index){float v=readF(0u,index);uint op=SPATIAL_OPCODE;if(arg(13u)==10u&&(op==S_GROUPNORM||op==S_RMSNORM))v=stashRound(v);if(op==S_SKIPLAYERNORM||op==S_SKIPSIMPLENORM||op==S_SKIPGROUPNORM){uint skipIndex=broadcastIndex(3u,0u,index);if(op==S_SKIPGROUPNORM&&rank(3u)==2u){uint channels=dim(0u,arg(7u)!=0u?rank(0u)-1u:1u),channel=arg(7u)!=0u?index%channels:(index/stride(0u,1u))%channels,batch=index/(size(0u)/dim(0u,0u));skipIndex=batch*channels+channel;}v+=readF(3u,skipIndex);if(has(4u)){uint channel=arg(7u)!=0u?index%dim(0u,rank(0u)-1u):(index/stride(0u,1u))%dim(0u,1u);uint at=op==S_SKIPGROUPNORM?channel:index%dim(0u,rank(0u)-1u);v+=readF(4u,at%size(4u));}}return v;}
void normalizationValue(uint outputSlot,uint index){
 uint op=SPATIAL_OPCODE,role=arg(200u+outputSlot);if(size(0u)==0u){writeF(outputSlot,index,unaryNaN());return;}uint logical=role==1u||role==2u?index*arg(9u):index;float x=size(0u)==0u?unaryNaN():normInput(logical),epsilon=argF(4u);uint r=rank(0u),channel=size(0u)==0u?0u:arg(7u)!=0u?logical%dim(0u,r-1u):(r>=2u?(logical/stride(0u,1u))%dim(0u,1u):logical%dim(0u,0u));
 if(role==3u){writeF(outputSlot,index,x);return;}
 if(op==S_BATCHNORM){float mean=readF(3u,channel),variance=readF(4u,channel);writeF(outputSlot,index,(x-mean)*readF(1u,channel)/sqrt(variance+epsilon)+readF(2u,channel));return;}
 if(op==S_LRN){uint spatial=stride(0u,1u),channels=dim(0u,1u),base=logical-channel*spatial,length=arg(5u);int first=int(channel)-int((length-1u)/2u),last=int(channel)+int(length/2u);float sum=0.0;for(int c=max(0,first);c<=min(int(channels)-1,last);++c){float v=readF(0u,base+uint(c)*spatial);sum+=v*v;}float denominator=argF(34u)+argF(32u)*sum/float(length);writeF(outputSlot,index,x/unaryExp(argF(33u)*unaryLogBits(floatBitsToUint(denominator),0)));return;}
 uint start=0u,count=arg(9u),step=1u,group=0u,spatial=1u,channels=0u;bool groupNorm=op==S_GROUPNORM||op==S_MSGROUPNORM||op==S_SKIPGROUPNORM;
 if(op==S_INSTANCENORM){count=stride(0u,1u);start=(logical/count)*count;}
 else if(groupNorm){channels=dim(0u,arg(7u)!=0u?r-1u:1u);uint per=channels/arg(5u);group=channel/per;spatial=size(0u)/(dim(0u,0u)*channels);count=per*spatial;uint batch=logical/(channels*spatial);start=batch*channels*spatial+(arg(7u)!=0u?group*per:group*count);}
 else if(op==S_LPNORM){uint a=arg(8u);step=stride(0u,a);count=dim(0u,a);start=logical-((logical/step)%count)*step;}
 else if(op==S_MEANVARNORM){uint coords[16];coordinates(logical,0u,false,coords);count=reductionCount();for(uint d=0u;d<r;++d)if(arg(16u+d)==0u)start+=coords[d]*stride(0u,d);}
 else start=count==0u?0u:(logical/count)*count;
 bool rms=op==S_RMSNORM||op==S_SIMPLENORM||op==S_SKIPSIMPLENORM||op==S_GATEDRMS;float mean=0.0;vec2 sumDF=vec2(0.0),squareDF=vec2(0.0);bool preciseStats=(op==S_LAYERNORM||op==S_RMSNORM)&&arg(13u)!=10u;
 for(uint j=0u;j<count;++j){uint at=start+j*step;if(groupNorm&&arg(7u)!=0u)at=start+(j/(channels/arg(5u)))*channels+j%(channels/arg(5u));if(op==S_MEANVARNORM)at=reductionOffset(start,j);float v=normInput(at);precise float square=v*v;float term=op==S_LPNORM?(arg(6u)==1u?abs(v):square):rms?stashRound(square):v;mean=stashRound(mean+term);if(preciseStats){sumDF=unaryDFAdd(sumDF,vec2(v,0.0));squareDF=unaryDFAdd(squareDF,unaryDFMul(vec2(v,0.0),vec2(v,0.0)));}}
 if(op==S_LPNORM){float norm=arg(6u)==2u?sqrt(mean):mean;writeF(outputSlot,index,norm==0.0?0.0:x/norm);return;}
 vec2 meanDF=count==0u?vec2(unaryNaN(),0.0):unaryDFDiv(sumDF,vec2(float(count),0.0));
 mean=count==0u?unaryNaN():preciseStats?(rms?unaryDFDiv(squareDF,vec2(float(count),0.0)).x:meanDF.x):stashRound(mean/float(count));float variance=rms?mean:0.0;
 vec2 varianceDF=vec2(0.0);
 if(!rms)for(uint j=0u;j<count;++j){uint at=start+j*step;if(groupNorm&&arg(7u)!=0u)at=start+(j/(channels/arg(5u)))*channels+j%(channels/arg(5u));if(op==S_MEANVARNORM)at=reductionOffset(start,j);float inputValue=normInput(at);float v=op==S_GROUPNORM?inputValue:stashRound(inputValue-mean);precise float square=v*v;variance=stashRound(variance+stashRound(square));if(preciseStats){vec2 delta=unaryDFAdd(vec2(inputValue,0.0),-meanDF);varianceDF=unaryDFAdd(varianceDF,unaryDFMul(delta,delta));}}
 if(!rms){variance=count==0u?unaryNaN():stashRound(variance/float(count));if(op==S_GROUPNORM){precise float meanSquare=stashRound(mean*mean);variance=stashRound(variance-meanSquare);}if(preciseStats){varianceDF=unaryDFDiv(varianceDF,vec2(float(count),0.0));variance=varianceDF.x;}}
 float standard=stashRound(sqrt(stashRound(variance+(op==S_MEANVARNORM?0.0:stashRound(epsilon)))));vec2 standardDF=vec2(standard,0.0);if(op==S_LAYERNORM&&arg(13u)!=10u)standardDF=spatialSqrtDF(unaryDFAdd(varianceDF,vec2(epsilon,0.0)));float inv=1.0/standard;if(role==1u){writeF(outputSlot,index,rms?0.0:mean);return;}if(role==2u){writeF(outputSlot,index,inv);return;}
 vec2 delta=preciseStats&&!rms?unaryDFAdd(vec2(x,0.0),-meanDF):vec2(rms?x:stashRound(x-mean),0.0);
 float result=arg(13u)==10u?stashRound(delta.x/standard):unaryDFDiv(delta,standardDF).x;if(op!=S_MEANVARNORM){uint parameter=op==S_GATEDRMS?logical%count:groupNorm?channel:op==S_INSTANCENORM?channel:broadcastIndex(1u,outputSlot,logical);if((op==S_LAYERNORM||op==S_GROUPNORM||op==S_RMSNORM)&&dtype(0u)==D_F16)result=xrgpuHalfToFloat(xrgpuFloatToHalf(result));precise float product=result*readF(1u,parameter);result=product;if((op==S_LAYERNORM||op==S_GROUPNORM)&&dtype(0u)==D_F16)result=xrgpuHalfToFloat(xrgpuFloatToHalf(result));if(has(2u)){uint biasIndex=groupNorm||op==S_INSTANCENORM?parameter:broadcastIndex(2u,outputSlot,logical);result+=readF(2u,biasIndex);}}
 if(op==S_GATEDRMS)result*=readF(3u,logical)/(1.0+unaryExp(-readF(3u,logical)));if((op==S_MSGROUPNORM||op==S_SKIPGROUPNORM)&&arg(10u)!=0u)result/=1.0+unaryExp(-result);writeF(outputSlot,index,result);
}
#include "spatial_pool_conv.glsl"
#include "spatial_grid.glsl"
#include "spatial_signal.glsl"
void main(){uint index=gid(),slot=0u;for(;slot<arg(216u);++slot){if(index<outSize(slot))break;index-=outSize(slot);}if(slot>=arg(216u))return;uint op=SPATIAL_OPCODE;if(op>=S_REDUCE_SUM&&op<=S_REDUCE_SQUARE)reduceValue(slot,index);else if(op<=S_HARDMAX)axisValue(slot,index);else if(op<=S_LRN)normalizationValue(slot,index);else if(op<=S_COL2IM)poolConvValue(slot,index);else if(op<=S_UPSAMPLE)gridValue(slot,index);else signalValue(slot,index);}
