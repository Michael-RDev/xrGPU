// SPDX-License-Identifier: Apache-2.0
#define XRGPU_NO_INT64_FLOAT_WRITE
#define XRGPU_NO_INT64_FLOAT_READ
#include "framework.glsl"
#include "math.glsl"
#include "tensor_codes.h"
layout(local_size_x=64) in;
// Frozen numerical contracts permit floating storage only on these paths.
#ifdef XRGPU_TENSOR_FLOAT32_INPUT
float tensorNativeFloat(uint slot,uint index){return uintBitsToFloat(readU(slot,index));}
#define readF tensorNativeFloat
#elif defined(XRGPU_TENSOR_FLOAT_INPUT)
float tensorNativeFloat(uint slot,uint index){uint bits=readU(slot,index);return dtype(slot)==D_F16?xrgpuHalfToFloat(bits):uintBitsToFloat(bits);}
#define readF tensorNativeFloat
#endif

bool floating(uint t){return t==D_F32||t==D_F16;}
bool unsignedType(uint t){return t==D_U32||t==D_U8||t==D_BOOL;}
uint bytesOf(uint t){return t==D_I64?8u:(t==D_F16||t==D_I16)?2u:(t==D_I8||t==D_U8||t==D_BOOL)?1u:4u;}
uint coordinate(uint index,uint d,uint outputSlot){uint s=outStride(outputSlot,d);return s==0u?0u:(index/s)%outDim(outputSlot,d);}
uint coordInput(uint index,uint d,uint inputSlot){uint s=stride(inputSlot,d);return s==0u?0u:(index/s)%dim(inputSlot,d);}
uint bc(uint slot,uint index,uint outputSlot){uint offset=0u,r=rank(slot),q=outRank(outputSlot);for(uint d=0u;d<r;++d){uint od=q-r+d;uint c=q<r?0u:coordinate(index,od,outputSlot);offset+=(dim(slot,d)==1u?0u:c)*stride(slot,d);}return offset;}
uint bcIn(uint slot,uint index,uint referenceSlot){uint offset=0u,r=rank(slot),q=rank(referenceSlot);for(uint d=0u;d<r;++d){uint od=q-r+d;uint c=q<r?0u:coordInput(index,od,referenceSlot);offset+=(dim(slot,d)==1u?0u:c)*stride(slot,d);}return offset;}
void copyIn(uint inputSlot,uint source,uint outputSlot,uint dest){writeRaw(outputSlot,dest,readRaw(inputSlot,source));}
void indexOut(uint slot,uint index,uint value){writeRaw(slot,index,uvec2(value,0u));}
bool equalValue(uint a,uint ai,uint b,uint bi){if(floating(dtype(a))){float x=readF(a,ai),y=readF(b,bi);return x==y||(isnan(x)&&isnan(y));}uvec2 x=readRaw(a,ai),y=readRaw(b,bi);return all(equal(x,y));}
bool lessValue(uint a,uint ai,uint b,uint bi){uint t=dtype(a);if(floating(t)){float x=readF(a,ai),y=readF(b,bi);return isnan(y)?!isnan(x):!isnan(x)&&x<y;}if(t==D_I64){uvec2 x=read64(a,ai),y=read64(b,bi);return x.y==y.y?x.x<y.x:int(x.y)<int(y.y);}if(unsignedType(t))return readU(a,ai)<readU(b,bi);return readI(a,ai)<readI(b,bi);}
// Published gather contracts zero-fill invalid indices. Other indexing contracts fail.
uint boundedIndex(uint slot,uint index,uint extent){
 uint type=dtype(slot);uvec2 raw=readRaw(slot,index);
 if(type==D_I64){
  if(raw.y==0u)return raw.x<extent?raw.x:0xffffffffu;
  if(raw.y!=0xffffffffu||(raw.x&0x80000000u)==0u)return 0xffffffffu;
 }
 if(unsignedType(type))return raw.x<extent?raw.x:0xffffffffu;
 int value=readI(slot,index);if(value<0)value+=int(extent);
 return value<0||uint(value)>=extent?0xffffffffu:uint(value);
}
uint checkedIndex(uint slot,uint index,uint extent){uint value=boundedIndex(slot,index,extent);if(value==0xffffffffu){fail(1u);return 0u;}return value;}
uint checkedLength(uint slot,uint index,uint maximum){uvec2 raw=readRaw(slot,index);if(dtype(slot)==D_I64&&raw.y!=0u){fail(1u);return 0u;}if(!unsignedType(dtype(slot))&&readI(slot,index)<0){fail(1u);return 0u;}if(raw.x>maximum){fail(1u);return 0u;}return raw.x;}
float gelu(float x,bool fast){return fast?0.5*x*(1.0+unaryTanh(0.7978845608028654*(x+0.044715*x*x*x))):0.5*x*(1.0+unaryErf(x*0.7071067811865475));}
float stableExpm1(float x){
if(isnan(x)||x==0.0)return x;
if(abs(x)<2.98023223876953125e-8)return x;
if(abs(x)>0.5)return unaryExp(x)-1.0;
vec2 term=vec2(x,0.0),sum=term;
for(uint k=2u;k<=18u;++k){term=unaryDFDiv(unaryDFMul(term,vec2(x,0.0)),vec2(float(k),0.0));sum=unaryDFAdd(sum,term);}
return sum.x;
}
float sigmoid(float x){float e=unaryExp(-abs(x));return x<0.0?e/(1.0+e):1.0/(1.0+e);}
float activation(float x,uint code){if(code==1u)return max(x,0.0);if(code==2u)return x<0.0?argF(18)*x:x;if(code==3u)return unaryTanh(x);if(code==4u)return sigmoid(x);if(code==5u)return clamp(x,argF(18),argF(19));if(code==6u)return clamp(argF(18)*x+argF(19),0.0,1.0);return x;}
uint floatIntegerBits(float value){uint bits=floatBitsToUint(value),exponent=(bits>>23u)&255u;int e=int(exponent)-127;if(e<0||exponent==255u)return 0u;uint mantissa=(bits&0x7fffffu)|0x800000u;uint result=e>=55?0u:e>=23?mantissa<<uint(e-23):mantissa>>uint(23-e);return (bits&0x80000000u)!=0u?0u-result:result;}
void numericCast(uint inputSlot,uint i,uint outputSlot,uint j){uint sourceType=dtype(inputSlot),target=outDtype(outputSlot);if(sourceType==target){copyIn(inputSlot,i,outputSlot,j);return;}if(floating(target)){writeF(outputSlot,j,readF(inputSlot,i));return;}if(target==D_BOOL){writeU(outputSlot,j,floating(sourceType)?(readF(inputSlot,i)!=0.0?1u:0u):(any(notEqual(readRaw(inputSlot,i),uvec2(0)))?1u:0u));return;}if(target==D_I64){if(floating(sourceType))writeRaw(outputSlot,j,writeInt64Float(readF(inputSlot,i)));else if(unsignedType(sourceType))writeRaw(outputSlot,j,uvec2(readU(inputSlot,i),0u));else{int value=readI(inputSlot,i);writeRaw(outputSlot,j,uvec2(uint(value),value<0?0xffffffffu:0u));}return;}uint low=floating(sourceType)?floatIntegerBits(readF(inputSlot,i)):readU(inputSlot,i);writeU(outputSlot,j,low);}
// Binary significand remainder avoids cancellation for large dividends.
float exactFmod(float x,float y){
 uint xb=floatBitsToUint(x),yb=floatBitsToUint(y),sign=xb&0x80000000u;
 uint ax=xb&0x7fffffffu,ay=yb&0x7fffffffu;
 if(ay==0u||ax>=0x7f800000u||ay>0x7f800000u)return uintBitsToFloat(0x7fc00000u);
 if(ax<ay)return x;if(ax==ay)return uintBitsToFloat(sign);
 int ex=int(ax>>23u),ey=int(ay>>23u);uint mx=ax&0x7fffffu,my=ay&0x7fffffu;
 if(ex==0){ex=1;while((mx&0x800000u)==0u){mx<<=1u;--ex;}}else mx|=0x800000u;
 if(ey==0){ey=1;while((my&0x800000u)==0u){my<<=1u;--ey;}}else my|=0x800000u;
 while(ex>ey){if(mx>=my)mx-=my;if(mx==0u)return uintBitsToFloat(sign);mx<<=1u;--ex;}
 if(mx>=my)mx-=my;if(mx==0u)return uintBitsToFloat(sign);
 while((mx&0x800000u)==0u){mx<<=1u;--ex;}
 uint result=ex>0?((uint(ex)<<23u)|(mx&0x7fffffu)):mx>>uint(1-ex);
 return uintBitsToFloat(sign|result);
}
uint inputByte(uint slot,uint byteIndex){uint width=bytesOf(dtype(slot)),element=byteIndex/width,offset=byteIndex%width;uvec2 raw=readRaw(slot,element);uint word=offset<4u?raw.x:raw.y;return (word>>((offset%4u)*8u))&255u;}
uint scaleIndex(uint scaleSlot,uint index){if(size(scaleSlot)<=1u)return 0u;uint a=arg(1),block=arg(2);uint c=coordInput(index,a,0u);if(rank(scaleSlot)==1u)return block>0u?c/block:c;uint offset=0u;for(uint d=0u;d<rank(0u);++d){uint x=coordInput(index,d,0u);if(d==a&&block>0u)x/=block;offset+=(dim(scaleSlot,d)==1u?0u:x)*stride(scaleSlot,d);}return offset;}
float integerDifference(uint slot,uint i,uint zeroSlot,uint zi){if(unsignedType(dtype(slot))){uint a=readU(slot,i),b=has(zeroSlot)?readU(zeroSlot,zi):0u;return a>=b?float(a-b):-float(b-a);}int a=readI(slot,i),b=has(zeroSlot)?readI(zeroSlot,zi):0;if((a<0)==(b<0))return float(a-b);uint magnitude=a>=0?uint(a)+uint(-(b+1))+1u:uint(-(a+1))+1u+uint(b);return a>=0?float(magnitude):-float(magnitude);}
// Pair arithmetic keeps declared int64 scatter reductions exact without shaderInt64.
bool less64(uvec2 a,uvec2 b){return int(a.y)==int(b.y)?a.x<b.x:int(a.y)<int(b.y);}
uvec2 add64(uvec2 a,uvec2 b){uint low=a.x+b.x;return uvec2(low,a.y+b.y+uint(low<a.x));}
uvec2 mul64(uvec2 a,uvec2 b){uint hi,lo;umulExtended(a.x,b.x,hi,lo);return uvec2(lo,hi+a.y*b.x+a.x*b.y);}
void scatterStore(uint dest,uint source){uint reduction=arg(2),t=outDtype(0u);if(reduction==0u){copyIn(2u,source,0u,dest);return;}
 if(t==D_I64){uint at=outputAddress(0u,dest);uvec2 a=uvec2(outputHeap[at],outputHeap[at+1u]);uvec2 b=readRaw(2u,source),value;if(reduction==1u)value=add64(a,b);else if(reduction==2u)value=mul64(a,b);else if(reduction==3u)value=less64(a,b)?b:a;else value=less64(a,b)?a:b;writeRaw(0u,dest,value);return;}
 if(floating(t)){float a=outF(0u,dest),b=readF(2u,source);writeF(0u,dest,reduction==1u?a+b:reduction==2u?a*b:reduction==3u?max(a,b):min(a,b));}
 else{uint a=readOutU(0u,dest),b=readU(2u,source);if(reduction==1u)writeU(0u,dest,a+b);else if(reduction==2u)writeU(0u,dest,a*b);else if(unsignedType(t))writeU(0u,dest,reduction==3u?max(a,b):min(a,b));else{int x=int(a),y=readI(2u,source);writeI(0u,dest,reduction==3u?max(x,y):min(x,y));}}
}

// One raw load per value. IEEE bit ordering preserves subnormals, treats
// signed zeros as equal and groups all NaNs after finite/infinite values.
int uniqueElementCompare(uint left,uint right){uint a=readU(0u,left),b=readU(0u,right),type=dtype(0u);
 if(type==D_F32||type==D_F16){uint sign=type==D_F32?0x80000000u:0x8000u,mask=type==D_F32?0xffffffffu:0xffffu,infinity=type==D_F32?0x7f800000u:0x7c00u;a&=mask;b&=mask;uint absoluteA=a&(sign-1u),absoluteB=b&(sign-1u);bool nanA=absoluteA>infinity,nanB=absoluteB>infinity;if(nanA||nanB)return nanA?(nanB?0:1):-1;if(absoluteA==0u&&absoluteB==0u)return 0;uint keyA=(a&sign)!=0u?(~a&mask):(a^sign),keyB=(b&sign)!=0u?(~b&mask):(b^sign);return keyA==keyB?0:keyA<keyB?-1:1;}
 if(type==D_U32||type==D_U8||type==D_BOOL)return a==b?0:a<b?-1:1;
 int x=type==D_I8?int(a<<24u)>>24:type==D_I16?int(a<<16u)>>16:int(a),y=type==D_I8?int(b<<24u)>>24:type==D_I16?int(b<<16u)>>16:int(b);return x==y?0:x<y?-1:1;
}
int uniqueSliceCompare(uint left,uint right){if(arg(2u)==0u)return uniqueElementCompare(left,right);if(size(0u)==0u)return 0;uint axis=arg(1u),n=dim(0u,axis),inner=stride(0u,axis),outer=size(0u)/(n*inner);for(uint o=0u;o<outer;++o)for(uint j=0u;j<inner;++j){int value=uniqueElementCompare((o*n+left)*inner+j,(o*n+right)*inner+j);if(value!=0)return value;}return 0;}
bool uniqueEqual(uint left,uint right){return uniqueSliceCompare(left,right)==0;}
bool uniqueLess(uint left,uint right){return uniqueSliceCompare(left,right)<0;}
void runUnique(){uint capacity=arg(2)!=0u?dim(0u,arg(1)):size(0u),count=0u;for(uint i=0u;i<capacity;++i){uint match=count;for(uint j=0u;j<count;++j)if(size(0u)==0u||uniqueEqual(i,scratchHeap[1u+j])){match=j;break;}if(match==count){scratchHeap[1u+count]=i;scratchHeap[1u+capacity+count]=0u;++count;}++scratchHeap[1u+capacity+match];}
 if(arg(3)!=0u)for(uint i=1u;i<count;++i){uint j=i;while(j>0u&&size(0u)!=0u&&uniqueLess(scratchHeap[1u+j],scratchHeap[j])){uint x=scratchHeap[1u+j];scratchHeap[1u+j]=scratchHeap[j];scratchHeap[j]=x;x=scratchHeap[1u+capacity+j];scratchHeap[1u+capacity+j]=scratchHeap[capacity+j];scratchHeap[capacity+j]=x;--j;}}
 setCount(0u,count);for(uint i=0u;i<outSize(0u);++i){uint a=arg(2)!=0u?arg(1):0u,c=coordinate(i,a,0u);if(c>=count)continue;uint source=0u;if(arg(2)==0u)source=scratchHeap[1u+c];else for(uint d=0u;d<rank(0u);++d)source+=(d==a?scratchHeap[1u+c]:coordinate(i,d,0u))*stride(0u,d);copyIn(0u,source,0u,i);}
 if(arg(9)!=0xffffffffu)for(uint j=0u;j<count;++j)indexOut(arg(9),j,scratchHeap[1u+j]);if(arg(10)!=0xffffffffu)for(uint i=0u;i<capacity;++i)for(uint j=0u;j<count;++j)if(size(0u)==0u||uniqueEqual(i,scratchHeap[1u+j])){indexOut(arg(10),i,j);break;}if(arg(11)!=0xffffffffu)for(uint j=0u;j<count;++j)indexOut(arg(11),j,scratchHeap[1u+capacity+j]);}
#ifdef XRGPU_UNIQUE_PHASE
// Stable GPU index mergesort bounds large sorted Unique to O(N log^2 N).
bool uniqueIndexLess(uint left,uint right){int comparison=uniqueSliceCompare(left,right);return comparison==0?left<right:comparison<0;}
bool uniqueIndexEqual(uint left,uint right){return size(0u)==0u||uniqueEqual(left,right);}
bool uniqueGroupLess(uint left,uint right){uint count=getCount(0u),a=left<count?scratchHeap[1u+left]:0xffffffffu,b=right<count?scratchHeap[1u+right]:0xffffffffu;return a==b?left<right:a<b;}
void runUniqueSort(uint index){
 uint n=arg(2)!=0u?dim(0u,arg(1)):size(0u),phase=uint(XRGPU_UNIQUE_PHASE);
 if(phase==1u){if(index<n)scratchHeap[arg(21u)+index]=index;return;}
 if(phase==2u){if(index>=n)return;uint width=arg(23u),begin=(index/(2u*width))*(2u*width),middle=min(begin+width,n),end=min(begin+2u*width,n),otherBegin=index<middle?middle:begin,lo=otherBegin,hi=index<middle?end:middle,original=scratchHeap[arg(21u)+index];
  while(lo<hi){uint m=lo+(hi-lo)/2u;if(uniqueIndexLess(scratchHeap[arg(21u)+m],original))lo=m+1u;else hi=m;}
  uint ownBegin=index<middle?begin:middle,dest=begin+(index-ownBegin)+(lo-otherBegin);scratchHeap[arg(22u)+dest]=original;return;
 }
 if(phase==3u){if(index!=0u)return;uint count=0u,previous=0u,inverseBase=1u+4u*n;
  for(uint i=0u;i<n;++i){uint original=scratchHeap[arg(21u)+i];if(i==0u||!uniqueIndexEqual(original,previous)){scratchHeap[1u+count]=original;scratchHeap[1u+n+count]=0u;++count;}++scratchHeap[1u+n+count-1u];scratchHeap[inverseBase+original]=count-1u;previous=original;}
  setCount(0u,count);if(arg(3u)==0u)return;
  if(arg(9u)!=0xffffffffu)for(uint i=0u;i<count;++i)indexOut(arg(9u),i,scratchHeap[1u+i]);
  if(arg(10u)!=0xffffffffu)for(uint i=0u;i<n;++i)indexOut(arg(10u),i,scratchHeap[inverseBase+i]);
  if(arg(11u)!=0xffffffffu)for(uint i=0u;i<count;++i)indexOut(arg(11u),i,scratchHeap[1u+n+i]);return;
 }
 if(phase==6u){if(index<n)scratchHeap[arg(21u)+index]=index;return;}
 if(phase==7u){if(index>=n)return;uint width=arg(23u),begin=(index/(2u*width))*(2u*width),middle=min(begin+width,n),end=min(begin+2u*width,n),otherBegin=index<middle?middle:begin,lo=otherBegin,hi=index<middle?end:middle,original=scratchHeap[arg(21u)+index];while(lo<hi){uint m=lo+(hi-lo)/2u;if(uniqueGroupLess(scratchHeap[arg(21u)+m],original))lo=m+1u;else hi=m;}uint ownBegin=index<middle?begin:middle,dest=begin+(index-ownBegin)+(lo-otherBegin);scratchHeap[arg(22u)+dest]=original;return;}
 if(phase==8u){uint count=getCount(0u);if(index>=count)return;uint group=scratchHeap[arg(21u)+index];scratchHeap[arg(24u)+group]=index;if(arg(9u)!=0xffffffffu)indexOut(arg(9u),index,scratchHeap[1u+group]);if(arg(11u)!=0xffffffffu)indexOut(arg(11u),index,scratchHeap[1u+n+group]);return;}
 if(phase!=4u){fail(6u);return;}
 if(arg(3u)==0u&&index<n&&arg(10u)!=0xffffffffu)indexOut(arg(10u),index,scratchHeap[arg(24u)+scratchHeap[1u+4u*n+index]]);
 if(index>=outSize(0u))return;uint axis=arg(2u)!=0u?arg(1u):0u,coordinateAxis=coordinate(index,axis,0u),count=getCount(0u);if(coordinateAxis>=count)return;
 uint group=arg(3u)!=0u?coordinateAxis:scratchHeap[arg(21u)+coordinateAxis],original=scratchHeap[1u+group],source=original;
 if(arg(2u)!=0u){source=0u;for(uint d=0u;d<rank(0u);++d)source+=(d==axis?original:coordinate(index,d,0u))*stride(0u,d);}
 copyIn(0u,source,0u,index);
}
#endif
float overlap(uint batch,uint left,uint right){uint a=(batch*dim(0u,1u)+left)*4u,b=(batch*dim(0u,1u)+right)*4u;vec4 x=vec4(readF(0u,a),readF(0u,a+1u),readF(0u,a+2u),readF(0u,a+3u)),y=vec4(readF(0u,b),readF(0u,b+1u),readF(0u,b+2u),readF(0u,b+3u));if(arg(1)!=0u){x=vec4(x.x-x.z/2.0,x.y-x.w/2.0,x.x+x.z/2.0,x.y+x.w/2.0);y=vec4(y.x-y.z/2.0,y.y-y.w/2.0,y.x+y.z/2.0,y.y+y.w/2.0);}else{x=vec4(min(x.x,x.z),min(x.y,x.w),max(x.x,x.z),max(x.y,x.w));y=vec4(min(y.x,y.z),min(y.y,y.w),max(y.x,y.z),max(y.y,y.w));}float intersection=max(0.0,min(x.z,y.z)-max(x.x,y.x))*max(0.0,min(x.w,y.w)-max(x.y,y.y));float area=max(0.0,x.z-x.x)*max(0.0,x.w-x.y)+max(0.0,y.z-y.x)*max(0.0,y.w-y.y)-intersection;return area>0.0?intersection/area:0.0;}
void runNms(){uint total=0u,n=dim(0u,1u),classes=dim(1u,1u),batch=dim(0u,0u);float threshold=has(3u)?readF(3u,0u):0.0,scoreThreshold=has(4u)?readF(4u,0u):0.0;for(uint b=0u;b<batch;++b)for(uint c=0u;c<classes;++c){for(uint i=0u;i<n;++i)scratchHeap[1u+i]=0u;for(uint k=0u;k<arg(2);++k){uint best=n;float score=-uintBitsToFloat(0x7f800000u);for(uint i=0u;i<n;++i){float s=readF(1u,(b*classes+c)*n+i);if(scratchHeap[1u+i]==0u&&(!has(4u)||s>scoreThreshold)&&(best==n||s>score)){best=i;score=s;}}if(best==n)break;indexOut(0u,total*3u,b);indexOut(0u,total*3u+1u,c);indexOut(0u,total*3u+2u,best);++total;scratchHeap[1u+best]=1u;for(uint j=0u;j<n;++j)if(overlap(b,best,j)>threshold)scratchHeap[1u+j]=1u;}}setCount(0u,total);}
// TopK has no Int64 data contract. Decode each key once, preserving uint32 order.
int topCompare(uint candidate,uint current,bool largest){uint type=dtype(0u),a=readU(0u,candidate),b=readU(0u,current);if(floating(type)){float x=type==D_F32?uintBitsToFloat(a):xrgpuHalfToFloat(a),y=type==D_F32?uintBitsToFloat(b):xrgpuHalfToFloat(b);bool nx=isnan(x),ny=isnan(y);if(nx||ny)return nx==ny?0:nx?1:-1;return x==y?0:(largest?x>y:x<y)?-1:1;}if(unsignedType(type))return a==b?0:(largest?a>b:a<b)?-1:1;int x=type==D_I8?int(a<<24u)>>24:type==D_I16?int(a<<16u)>>16:int(a),y=type==D_I8?int(b<<24u)>>24:type==D_I16?int(b<<16u)>>16:int(b);return x==y?0:(largest?x>y:x<y)?-1:1;}
bool topBetter(uint candidate,uint current,bool largest){return topCompare(candidate,current,largest)<0;}
uint modulo64(uvec2 value,uint divisor){bool negative=(value.y&0x80000000u)!=0u;if(negative){value.x=~value.x+1u;value.y=~value.y+uint(value.x==0u);}uint result=0u;for(uint j=64u;j>0u;--j){uint bit=j>32u?(value.y>>(j-33u))&1u:(value.x>>(j-1u))&1u;uint doubled=(result<<1u)|bit;bool carry=(result&0x80000000u)!=0u;result=carry||doubled>=divisor?doubled-divisor:doubled;}return negative&&result!=0u?divisor-result:result;}
float typedHalf(float x){return xrgpuHalfToFloat(xrgpuFloatToHalf(x));}
void runTopK(){uint a=arg(1),k=arg(2),n=dim(0u,a),inner=stride(0u,a),outer=n==0u?0u:size(0u)/(n*inner);for(uint o=0u;o<outer;++o)for(uint q=0u;q<inner;++q){for(uint j=0u;j<n;++j)scratchHeap[j]=0u;for(uint r=0u;r<k;++r){uint best=n;for(uint j=0u;j<n;++j)if(scratchHeap[j]==0u&&(best==n||topBetter((o*n+j)*inner+q,(o*n+best)*inner+q,arg(3)!=0u)))best=j;scratchHeap[best]=1u;uint dest=(o*k+r)*inner+q;copyIn(0u,(o*n+best)*inner+q,0u,dest);indexOut(1u,dest,best);}}}
// Stable index mergesort makes vocabulary-sized TopK practical on native GPUs.
bool topIndexLess(uint row,uint left,uint right){uint n=dim(0u,arg(1)),inner=stride(0u,arg(1)),base=(row/inner)*n*inner+row%inner;uint a=base+left*inner,b=base+right*inner;int comparison=topCompare(a,b,arg(3)!=0u);return comparison<0||(comparison==0&&left<right);}
void runTopKSort(uint index){
#ifdef XRGPU_TOPK_PHASE
const uint phase=XRGPU_TOPK_PHASE;
#else
const uint phase=0u;
#endif
uint n=dim(0u,arg(1)),inner=stride(0u,arg(1));if(phase==1u){if(index<size(0u))scratchHeap[index]=index%n;return;}
 if(phase==2u){if(index>=size(0u))return;uint row=index/n,j=index%n,width=arg(23),begin=(j/(2u*width))*(2u*width),middle=min(begin+width,n),end=min(begin+2u*width,n),otherBegin=j<middle?middle:begin,lo=otherBegin,hi=j<middle?end:middle,original=scratchHeap[arg(21)+index];while(lo<hi){uint m=lo+(hi-lo)/2u;if(topIndexLess(row,scratchHeap[arg(21)+row*n+m],original))lo=m+1u;else hi=m;}uint ownBegin=j<middle?begin:middle,dest=begin+(j-ownBegin)+(lo-otherBegin);scratchHeap[arg(22)+row*n+dest]=original;return;}
 if(index>=outSize(0u))return;uint k=arg(2),row=(index/(k*inner))*inner+index%inner,ordinal=(index/inner)%k,original=scratchHeap[arg(21)+row*n+ordinal],source=(row/inner)*n*inner+original*inner+row%inner;copyIn(0u,source,0u,index);indexOut(1u,index,original);
}
void runEinsum(uint i){uint count=arg(5),nlabels=arg(1),nout=arg(2),nreduce=arg(3);uint coordinates[64];for(uint d=0u;d<nlabels;++d)coordinates[d]=0u;for(uint d=0u;d<nout;++d)coordinates[arg(96u+d)]=coordinate(i,d,0u);float sum=0.0;for(uint k=0u;k<arg(4);++k){uint remainder=k;for(uint reverse=nreduce;reverse>0u;--reverse){uint l=arg(112u+reverse-1u),extent=arg(32u+l);coordinates[l]=remainder%extent;remainder/=extent;}float product=1.0;for(uint s=0u;s<count;++s){uint offset=0u;for(uint d=0u;d<rank(s);++d)offset+=(dim(s,d)==1u?0u:coordinates[arg(176u+s*16u+d)])*stride(s,d);product*=readF(s,offset);}sum+=product;}writeF(0u,i,sum);}
void runGemm(uint i){uint outRows=arg(7),outCols=arg(8),k=arg(3);uint row=(i/outCols)%outRows,col=i%outCols,batch=i/(outCols*outRows);uint aRank=rank(0u),bRank=rank(1u),aBase=0u,bBase=0u;uint batchIndex=batch;uint bcoords[16];for(uint d=0u;d<16u;++d)bcoords[d]=0u;for(uint reverse=arg(10)-2u;reverse>0u;--reverse){uint d=reverse-1u,e=arg(32u+d);bcoords[d]=batchIndex%e;batchIndex/=e;}for(uint s=0u;s<2u;++s){uint r=rank(s),base=0u;for(uint d=0u;d+2u<r;++d){uint od=(arg(10)-r)+d;uint original=d;if((s==0u?arg(4):arg(5))!=0u&&d==r-3u)original=r-2u;base+=(dim(s,original)==1u?0u:bcoords[od])*stride(s,original);}if(s==0u)aBase=base;else bBase=base;}
 uint ar=aRank-2u,ac=aRank-1u,br=bRank-2u,bc_=bRank-1u;if(arg(4)!=0u&&aRank>=3u)ar=aRank-3u;if(arg(5)!=0u&&bRank>=3u)br=bRank-3u;float sum=0.0;for(uint j=0u;argF(16)!=0.0&&j<k;++j){uint ai=aBase+(arg(1)!=0u?j:row)*stride(0u,ar)+(arg(1)!=0u?row:j)*stride(0u,ac);uint bi=bBase+(arg(2)!=0u?col:j)*stride(1u,br)+(arg(2)!=0u?j:col)*stride(1u,bc_);sum+=readF(0u,ai)*readF(1u,bi);}sum*=argF(16);if(has(2u)&&argF(17)!=0.0)sum+=argF(17)*readF(2u,bc(2u,i,0u));if(XRGPU_TENSOR_OPCODE==TX_M_GEMM_FAST_GELU)sum=gelu(sum,true);else if(XRGPU_TENSOR_OPCODE==TX_M_FUSED_GEMM)sum=activation(sum,arg(6));writeF(0u,i,sum);}

void main(){uint threadIndex=gid();uint code=XRGPU_TENSOR_OPCODE;
#ifdef XRGPU_TOPK_PHASE
runTopKSort(threadIndex);
#elif defined(XRGPU_UNIQUE_PHASE)
runUniqueSort(threadIndex);
#else
bool serial=code==TX_UNIQUE||code==TX_NON_MAX_SUPPRESSION||code==TX_TOP_K||code==TX_NON_ZERO||code==TX_COMPRESS||code==TX_DYNAMIC_QUANTIZE_LINEAR||code==TX_LOOP||code==TX_SCAN||code==TX_SPLIT||code==TX_SPLIT_TO_SEQUENCE||code==TX_SCATTER||code==TX_SCATTER_ELEMENTS||code==TX_SCATTER_N_D||code==TX_TENSOR_SCATTER;
if(serial){if(threadIndex!=0u)return;}else if(threadIndex>=outSize(0u))return;
 if(code==TX_UNIQUE){runUnique();return;}if(code==TX_NON_MAX_SUPPRESSION){runNms();return;}if(code==TX_TOP_K){runTopK();return;}if(code==TX_EINSUM){runEinsum(threadIndex);return;}
 if(code==TX_GEMM||code==TX_M_FUSED_GEMM||code==TX_M_FUSED_MAT_MUL||code==TX_M_TRANSPOSE_MAT_MUL||code==TX_M_GEMM_FAST_GELU){runGemm(threadIndex);return;}
 if(code==TX_NON_ZERO){uint count=0u;for(uint i=0u;i<size(0u);++i){bool nonzero=floating(dtype(0u))?readF(0u,i)!=0.0:any(notEqual(readRaw(0u,i),uvec2(0)));if(nonzero){for(uint d=0u;d<rank(0u);++d)indexOut(0u,d*size(0u)+count,coordInput(i,d,0u));++count;}}setCount(0u,count);return;}
 if(code==TX_COMPRESS){uint a=arg(1),n=min(size(1u),arg(2)!=0u?dim(0u,a):size(0u)),count=0u;for(uint j=0u;j<n;++j)if(readU(1u,j)!=0u)++count;setCount(0u,count);for(uint i=0u;i<outSize(0u);++i){uint c=coordinate(i,a,0u);if(c>=count)continue;uint sourceAxis=0u,ordinal=0u;for(;sourceAxis<n;++sourceAxis)if(readU(1u,sourceAxis)!=0u){if(ordinal==c)break;++ordinal;}uint source=sourceAxis;if(arg(2)!=0u){source=0u;for(uint d=0u;d<rank(0u);++d)source+=(d==a?sourceAxis:coordinate(i,d,0u))*stride(0u,d);}copyIn(0u,source,0u,i);}return;}
 if(code==TX_DYNAMIC_QUANTIZE_LINEAR){float lo=0.0,hi=0.0;for(uint i=0u;i<size(0u);++i){float x=readF(0u,i);if(!isnan(x)){lo=min(lo,x);hi=max(hi,x);}}float scale=hi==lo?1.0:(hi-lo)/255.0;float zp=clamp(roundEven(-lo/scale),0.0,255.0);writeF(1u,0u,scale);writeU(2u,0u,uint(zp));for(uint i=0u;i<size(0u);++i)writeU(0u,i,uint(clamp(roundEven(readF(0u,i)/scale)+zp,0.0,255.0)));return;}
 if(code==TX_LOOP){for(uint j=0u;j<size(2u);++j){float state=readF(2u,j);for(uint t=0u;t<arg(2);++t){if(t<arg(1))state+=readF(3u,j);writeF(1u,t*size(2u)+j,t<arg(1)?state:0.0);}writeF(0u,j,state);}return;}
 if(code==TX_SCAN){uint width=size(0u),length=dim(1u,0u);for(uint j=0u;j<width;++j){float state=readF(0u,j);for(uint t=0u;t<length;++t){uint source=arg(1)!=0u?length-1u-t:t;state+=readF(1u,source*width+j);writeF(1u,source*width+j,state);}writeF(0u,j,state);}return;}
 if(code==TX_SPLIT||code==TX_SPLIT_TO_SEQUENCE){uint a=arg(1);for(uint o=0u;o<outputCount();++o)for(uint i=0u;i<outSize(o);++i){uint source=0u;for(uint d=0u;d<rank(0u);++d){uint c;if(arg(2)!=0u)c=d==a?0u:coordinate(i,d<a?d:d-1u,o);else c=coordinate(i,d,o);if(d==a)c+=arg(32u+o);source+=c*stride(0u,d);}copyIn(0u,source,o,i);}return;}
 if(code==TX_SCATTER||code==TX_SCATTER_ELEMENTS||code==TX_SCATTER_N_D){for(uint i=0u;i<size(0u);++i)copyIn(0u,i,0u,i);for(uint j=0u;j<size(2u);++j){uint dest=0u;if(code==TX_SCATTER_N_D){uint k=dim(1u,rank(1u)-1u),slice=1u;for(uint d=k;d<rank(0u);++d)slice*=dim(0u,d);uint tuple=j/slice;dest=j%slice;for(uint d=0u;d<k;++d){uint v=checkedIndex(1u,tuple*k+d,dim(0u,d));dest+=v*stride(0u,d);}}else{uint a=arg(1);for(uint d=0u;d<rank(0u);++d){uint v=d==a?checkedIndex(1u,j,dim(0u,d)):coordInput(j,d,2u);if(v>=dim(0u,d)){fail(1u);return;}dest+=v*stride(0u,d);}}scatterStore(dest,j);}return;}
 if(code==TX_TENSOR_SCATTER){for(uint i=0u;i<size(0u);++i)copyIn(0u,i,0u,i);uint a=arg(1),n=dim(0u,a),updates=dim(1u,a);if(n==0u)return;for(uint i=0u;i<size(1u);++i){uint batch=coordInput(i,0u,1u),u=coordInput(i,a,1u);bool perBatch=!has(2u)||size(2u)==dim(0u,0u);uvec2 start=has(2u)?readRaw(2u,perBatch?batch:batch*updates+u):uvec2(0u);if(dtype(2u)!=D_I64&&int(start.x)<0&&!unsignedType(dtype(2u)))start.y=0xffffffffu;uint target;if(arg(2)!=0u){target=modulo64(start,n);if(perBatch){uint increment=u%n;target=target>=n-increment?target-(n-increment):target+increment;}}else{uvec2 value=perBatch?add64(start,uvec2(u,0u)):start;if(value.y!=0u||value.x>=n)continue;target=value.x;}uint dest=0u;for(uint d=0u;d<rank(0u);++d)dest+=(d==a?target:coordInput(i,d,1u))*stride(0u,d);copyIn(1u,i,0u,dest);}return;}

 {uint i=threadIndex;
  if(code==TX_IDENTITY||code==TX_RESHAPE||code==TX_FLATTEN||code==TX_SQUEEZE||code==TX_UNSQUEEZE||code==TX_CONSTANT){copyIn(0u,i,0u,i);return;}
  if(code==TX_DROPOUT){copyIn(0u,i,0u,i);if(outputCount()>1u)writeU(1u,i,1u);return;}
  if(code==TX_CONSTANT_OF_SHAPE){if(arg(23)!=0u){copyIn(1u,0u,0u,i);return;}if(arg(22)!=0u)writeU(0u,i,arg(21));else writeF(0u,i,argF(16));return;}
  if(code==TX_CAST||code==TX_CAST_LIKE){numericCast(0u,i,0u,i);return;}
  if(code==TX_BIT_CAST){uint width=bytesOf(outDtype(0u));uvec2 value=uvec2(0u);for(uint j=0u;j<width;++j){uint b=inputByte(0u,i*width+j);if(j<4u)value.x|=b<<(j*8u);else value.y|=b<<((j-4u)*8u);}writeRaw(0u,i,value);return;}
  if(code==TX_SHAPE){indexOut(0u,i,dim(0u,arg(1)+i));return;}if(code==TX_SIZE){indexOut(0u,0u,size(0u));return;}
  if(code==TX_IF){copyIn(readU(0u,0u)!=0u?1u:2u,i,0u,i);return;}
  if(code==TX_EYE_LIKE){writeF(0u,i,arg(2)!=0u?0.0:int(coordinate(i,1u,0u))-int(coordinate(i,0u,0u))==int(arg(1))?1.0:0.0);return;}
  if(code==TX_RANGE){if(floating(dtype(0u)))writeF(0u,i,readF(0u,0u)+float(i)*readF(2u,0u));else writeU(0u,i,readU(0u,0u)+i*readU(2u,0u));return;}
  if(code==TX_TRANSPOSE){uint source=0u;for(uint d=0u;d<outRank(0u);++d)source+=coordinate(i,d,0u)*stride(0u,arg(32u+d));copyIn(0u,source,0u,i);return;}
  if(code==TX_EXPAND){copyIn(0u,bc(0u,i,0u),0u,i);return;}
  if(code==TX_TILE){uint source=0u;for(uint d=0u;d<outRank(0u);++d)source+=(coordinate(i,d,0u)%dim(0u,d))*stride(0u,d);copyIn(0u,source,0u,i);return;}
  if(code==TX_CONCAT){uint a=arg(1),c=coordinate(i,a,0u),slot=0u;while(slot<inputCount()&&(!has(slot)||c>=dim(slot,a))){if(has(slot))c-=dim(slot,a);++slot;}if(slot>=inputCount()){fail(2u);return;}uint source=0u;for(uint d=0u;d<outRank(0u);++d)source+=(d==a?c:coordinate(i,d,0u))*stride(slot,d);copyIn(slot,source,0u,i);return;}
  if(code==TX_SLICE){uint source=0u;for(uint d=0u;d<outRank(0u);++d)source+=uint(int(arg(32u+d))+int(coordinate(i,d,0u))*int(arg(48u+d)))*stride(0u,d);copyIn(0u,source,0u,i);return;}
  if(code==TX_PAD){uint source=0u;bool inside=true;for(uint d=0u;d<outRank(0u);++d){int begin=int(arg(32u+d)),end=int(arg(32u+rank(0u)+d)),crop=max(-begin,0),n=int(dim(0u,d))-crop-max(-end,0),c=int(coordinate(i,d,0u))-max(begin,0);if(c<0||c>=n){if(arg(1)==0u)inside=false;else if(n<=0){fail(3u);return;}else if(arg(1)==1u)c=clamp(c,0,n-1);else if(arg(1)==2u){if(n<=1){fail(3u);return;}int period=2*(n-1);c=((c%period)+period)%period;if(c>=n)c=period-c;}else c=((c%n)+n)%n;}source+=uint(max(c+crop,0))*stride(0u,d);}if(inside)copyIn(0u,source,0u,i);else if(has(1u))copyIn(1u,0u,0u,i);else writeRaw(0u,i,uvec2(0u));return;}
  if(code==TX_GATHER){uint a=arg(1),index=0u,source=0u;for(uint d=0u;d<rank(1u);++d)index+=coordinate(i,a+d,0u)*stride(1u,d);uint v=boundedIndex(1u,index,dim(0u,a));if(v==0xffffffffu){writeRaw(0u,i,uvec2(0u));return;}for(uint d=0u;d<rank(0u);++d){uint c=d<a?coordinate(i,d,0u):d==a?uint(v):coordinate(i,d+rank(1u)-1u,0u);source+=c*stride(0u,d);}copyIn(0u,source,0u,i);return;}
  if(code==TX_GATHER_ELEMENTS){uint source=0u,a=arg(1);for(uint d=0u;d<rank(0u);++d){uint c=d==a?checkedIndex(1u,i,dim(0u,d)):coordinate(i,d,0u);if(c>=dim(0u,d)){fail(1u);return;}source+=c*stride(0u,d);}copyIn(0u,source,0u,i);return;}
  if(code==TX_GATHER_N_D){uint b=arg(1),k=arg(2),prefix=rank(1u)-1u,tuple=0u,source=0u;for(uint d=0u;d<prefix;++d)tuple+=coordinate(i,d,0u)*stride(1u,d);for(uint d=0u;d<b;++d)source+=coordinate(i,d,0u)*stride(0u,d);for(uint d=0u;d<k;++d){uint c=boundedIndex(1u,tuple+d,dim(0u,b+d));if(c==0xffffffffu){writeRaw(0u,i,uvec2(0u));return;}source+=c*stride(0u,b+d);}for(uint d=b+k;d<rank(0u);++d)source+=coordinate(i,prefix+d-b-k,0u)*stride(0u,d);copyIn(0u,source,0u,i);return;}
  if(code==TX_REVERSE_SEQUENCE){uint a=arg(1),time=arg(2),batch=coordinate(i,a,0u),t=coordinate(i,time,0u);uint length=checkedLength(1u,batch,dim(0u,time));uint source=0u;for(uint d=0u;d<rank(0u);++d){uint c=coordinate(i,d,0u);if(d==time&&t<uint(length))c=uint(length)-1u-t;source+=c*stride(0u,d);}copyIn(0u,source,0u,i);return;}
  if(code==TX_SPACE_TO_DEPTH||code==TX_DEPTH_TO_SPACE){uint n=coordinate(i,0u,0u),c=coordinate(i,1u,0u),h=coordinate(i,2u,0u),w=coordinate(i,3u,0u),b=arg(1),source=0u;if(code==TX_SPACE_TO_DEPTH){uint inC=dim(0u,1u),dh=c/inC/b,dw=(c/inC)%b;source=((n*inC+c%inC)*dim(0u,2u)+(h*b+dh))*dim(0u,3u)+(w*b+dw);}else{uint outC=outDim(0u,1u),inC=arg(2)==0u?(h%b*b+w%b)*outC+c:c*b*b+(h%b*b+w%b);source=((n*dim(0u,1u)+inC)*dim(0u,2u)+h/b)*dim(0u,3u)+w/b;}copyIn(0u,source,0u,i);return;}
  if(code==TX_TRILU){int row=int(coordinate(i,outRank(0u)-2u,0u)),col=int(coordinate(i,outRank(0u)-1u,0u)),difference=col-row;uvec2 k=has(1u)?readRaw(1u,0u):uvec2(0u),value=uvec2(uint(difference),difference<0?0xffffffffu:0u);bool keep=arg(1)!=0u?!less64(value,k):!less64(k,value);if(keep)copyIn(0u,i,0u,i);else writeRaw(0u,i,uvec2(0u));return;}
  if(code==TX_ONE_HOT){uint a=arg(1),depth=arg(2),index=0u;for(uint d=0u;d<rank(0u);++d)index+=coordinate(i,d<a?d:d+1u,0u)*stride(0u,d);int value=0;bool valid=true;if(unsignedType(dtype(0u))){uint v=readU(0u,index);valid=v<depth;value=int(v);}else if(floating(dtype(0u))){float v=trunc(readF(0u,index));valid=!isnan(v)&&!isinf(v)&&v>=-float(depth)&&v<float(depth);if(valid)value=int(v);}else value=readI(0u,index);if(value<0)value+=int(depth);copyIn(2u,valid&&value==int(coordinate(i,a,0u))?1u:0u,0u,i);return;}
  if(code==TX_QUANTIZE_LINEAR||code==TX_DEQUANTIZE_LINEAR){uint s=scaleIndex(1u,i),z=has(2u)?scaleIndex(2u,i):0u;float scale=readF(1u,s);if((code==TX_QUANTIZE_LINEAR&&scale==0.0)||isinf(scale)||isnan(scale)){fail(4u);return;}if(code==TX_DEQUANTIZE_LINEAR)writeF(0u,i,integerDifference(0u,i,2u,z)*scale);else{float zero=has(2u)?readF(2u,z):0.0,x=readF(0u,i);if(arg(24)!=0u){x=typedHalf(x);scale=typedHalf(scale);}float quotient=x/scale;if(arg(24)!=0u)quotient=typedHalf(quotient);float q=roundEven(quotient)+zero;uint t=outDtype(0u);q=clamp(q,t==D_I8?-128.0:0.0,t==D_I8?127.0:255.0);writeF(0u,i,q);}return;}
  uint a=bc(0u,i,0u),b=has(1u)?bc(1u,i,0u):0u;
  if(code==TX_AND||code==TX_OR||code==TX_XOR){bool x=readU(0u,a)!=0u,y=readU(1u,b)!=0u;writeU(0u,i,(code==TX_AND?x&&y:code==TX_OR?x||y:x!=y)?1u:0u);return;}
  if(code==TX_NOT){writeU(0u,i,readU(0u,a)==0u?1u:0u);return;}
  if(code==TX_BITWISE_NOT){writeU(0u,i,~readU(0u,a));return;}
  if(code==TX_EQUAL||code==TX_GREATER||code==TX_GREATER_OR_EQUAL||code==TX_LESS||code==TX_LESS_OR_EQUAL){bool eq=equalValue(0u,a,1u,b),lt=lessValue(0u,a,1u,b),gt=lessValue(1u,b,0u,a);if(floating(dtype(0u))&&(isnan(readF(0u,a))||isnan(readF(1u,b))))eq=lt=gt=false;bool v=code==TX_EQUAL?eq:code==TX_GREATER?gt:code==TX_GREATER_OR_EQUAL?gt||eq:code==TX_LESS?lt:lt||eq;writeU(0u,i,v?1u:0u);return;}
  if(code==TX_BIT_SHIFT){uint shift=readU(1u,b),width=bytesOf(dtype(0u))*8u;writeU(0u,i,shift>=width?0u:arg(1)!=0u?readU(0u,a)>>shift:readU(0u,a)<<shift);return;}
  if(code==TX_WHERE){copyIn(readU(0u,a)!=0u?1u:2u,readU(0u,a)!=0u?bc(1u,i,0u):bc(2u,i,0u),0u,i);return;}
  if(code==TX_CLIP){if(floating(dtype(0u))){float x=readF(0u,a);if(isnan(x)){copyIn(0u,a,0u,i);return;}if(has(1u))x=max(x,readF(1u,0u));if(has(2u))x=min(x,readF(2u,0u));writeF(0u,i,x);}else if(unsignedType(dtype(0u))){uint x=readU(0u,a);if(has(1u))x=max(x,readU(1u,0u));if(has(2u))x=min(x,readU(2u,0u));writeU(0u,i,x);}else{int x=readI(0u,a);if(has(1u))x=max(x,readI(1u,0u));if(has(2u))x=min(x,readI(2u,0u));writeI(0u,i,x);}return;}
  if(code==TX_MAX||code==TX_MIN){uint chosen=0u,at=a;for(uint s=1u;s<inputCount();++s)if(has(s)){if(floating(dtype(chosen))&&isnan(readF(chosen,at)))break;uint j=bc(s,i,0u);bool take=code==TX_MAX?lessValue(chosen,at,s,j):lessValue(s,j,chosen,at);if(floating(dtype(s))&&isnan(readF(s,j)))take=true;if(take){chosen=s;at=j;}}copyIn(chosen,at,0u,i);return;}
  if(code==TX_SUM||code==TX_MEAN){float sum=0.0;uint n=0u;for(uint s=0u;s<inputCount();++s)if(has(s)){sum+=readF(s,bc(s,i,0u));++n;}writeF(0u,i,code==TX_MEAN?sum/float(n):sum);return;}
  if(code==TX_MOD){
   if(floating(dtype(0u))){float x=readF(0u,a),y=readF(1u,b),r=exactFmod(x,y);if(arg(1)==0u&&!isnan(r)){if(r!=0.0&&((floatBitsToUint(r)^floatBitsToUint(y))&0x80000000u)!=0u)r+=y;else if(r==0.0)r=uintBitsToFloat(floatBitsToUint(y)&0x80000000u);}writeF(0u,i,r);}
   else if(unsignedType(dtype(0u))){uint x=readU(0u,a),y=readU(1u,b);if(y==0u){fail(4u);return;}writeU(0u,i,x%y);}
   else{int x=readI(0u,a),y=readI(1u,b);if(y==0){fail(4u);return;}uint ax=x<0?0u-uint(x):uint(x),ay=y<0?0u-uint(y):uint(y),m=ax%ay;int r=x<0?int(0u-m):int(m);if(arg(1)==0u&&r!=0&&((r<0)!=(y<0)))r+=y;writeI(0u,i,r);}return;
  }
  if(code==TX_POW){float x=readF(0u,a),y=readF(1u,b),value;if(!floating(dtype(1u))||(!isnan(y)&&abs(y)<=2147483520.0&&y==trunc(y))){int exponent=floating(dtype(1u))?int(y):readI(1u,b);uint n=exponent<0?0u-uint(exponent):uint(exponent);if(dtype(0u)==D_I32){uint product=1u,base=readU(0u,a);while(n>0u){if((n&1u)!=0u)product*=base;base*=base;n>>=1u;}writeU(0u,i,exponent<0?(x==1.0?1u:x==-1.0?product:0u):product);return;}value=1.0;float base=x;while(n>0u){if((n&1u)!=0u)value*=base;base*=base;n>>=1u;}if(exponent<0)value=1.0/value;}else value=pow(x,y);writeF(0u,i,value);return;}
  if(code==TX_IS_INF||code==TX_IS_NA_N){float x=readF(0u,a);bool v=code==TX_IS_NA_N?isnan(x):isinf(x)&&((x<0.0&&arg(1)!=0u)||(x>0.0&&arg(2)!=0u));writeU(0u,i,v?1u:0u);return;}
  float x=readF(0u,a),y=x;
  if(code==TX_CELU)y=max(x,0.0)+min(0.0,argF(16)*(stableExpm1(x/argF(16))));
  else if(code==TX_ELU)y=x>=0.0?x:argF(16)*(stableExpm1(x));
  else if(code==TX_GELU)y=gelu(x,arg(1)!=0u);
  else if(code==TX_M_GELU)y=gelu(x,false);
  else if(code==TX_HARD_SIGMOID){float value=argF(16)*x+argF(17);y=isnan(value)?value:clamp(value,0.0,1.0);}
  else if(code==TX_LEAKY_RELU)y=x>=0.0?x:argF(16)*x;
  else if(code==TX_SELU)y=argF(18)*(x>0.0?x:argF(16)*(stableExpm1(x)));
  else if(code==TX_SHRINK)y=x< -argF(20)?x+argF(19):x>argF(20)?x-argF(19):0.0;
  else if(code==TX_SWISH||code==TX_M_QUICK_GELU)y=x*sigmoid(argF(16)*x);
  else if(code==TX_THRESHOLDED_RELU)y=x>argF(16)?x:0.0;
  else if(code==TX_M_BIAS_ADD){y=x+readF(1u,b);if(has(2u))y+=readF(2u,bc(2u,i,0u));}
  else if(code==TX_M_BIAS_GELU)y=gelu(x+readF(1u,b),false);
  else if(code==TX_M_FAST_GELU)y=gelu(x+(has(1u)?readF(1u,b):0.0),true);
  else if(code==TX_M_GATED_ADD){float product=readF(1u,b)*readF(2u,bc(2u,i,0u));y=x+(dtype(0u)==D_F16?typedHalf(product):product);}
  else if(code==TX_SWI_G_L_U){if(dtype(0u)==D_F16){float scaled=typedHalf(typedHalf(argF(16))*x),gate=typedHalf(sigmoid(scaled)),swish=typedHalf(x*gate);y=swish*readF(1u,b);}else y=x*sigmoid(argF(16)*x)*readF(1u,b);}
  else if(code==TX_M_BIAS_SPLIT_GELU){uint width=outDim(0u,outRank(0u)-1u),index=(i/width)*(2u*width)+i%width;float left=readF(0u,index)+(has(1u)?readF(1u,bcIn(1u,index,0u)):0.0);float right=readF(0u,index+width)+(has(1u)?readF(1u,bcIn(1u,index+width,0u)):0.0);y=left*gelu(right,false);}
  else {fail(99u);return;}
  writeF(0u,i,y);
 }
#endif
}
