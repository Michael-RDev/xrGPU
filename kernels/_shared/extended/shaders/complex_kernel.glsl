// SPDX-License-Identifier: Apache-2.0
// Original, correctness-first serial Vulkan algorithms. All numerical work is
// performed in this shader; host code only validates and plans tensor storage.
// This family has no numerical Int64 outputs. Its only Int64 inputs are
// exact rotary positions and gather indices read through read64/controlIndex.
#define XRGPU_NO_INT64_FLOAT_WRITE
#define XRGPU_NO_INT64_FLOAT_READ
#include "framework.glsl"
#if XRGPU_COMPLEX_OPCODE >= 1020 && XRGPU_COMPLEX_OPCODE <= 1022
// Recurrent published contracts require FP32. Preserve that concrete type in
// the shader rather than carrying the generic half/integer conversion tree
// into every gate product; this also bounds software Vulkan compilation.
float recurrentReadF(uint slot,uint index){return uintBitsToFloat(inputHeap[parameters[inDesc(slot)]+index]);}
void recurrentWriteF(uint slot,uint index,float value){outputHeap[parameters[outDesc(slot)]+index]=floatBitsToUint(value);}
#define readF recurrentReadF
#define writeF recurrentWriteF
#else
// Floating arithmetic contracts use F32/F16. Integer calls here are quantized
// zero points; 64-bit indexing remains an exact separate read64/readI path.
float complexReadF(uint slot,uint index){uint v=readU(slot,index),t=dtype(slot);if(t==D_F32)return uintBitsToFloat(v);if(t==D_F16)return xrgpuHalfToFloat(v);if(t==D_I8)return float(int(v<<24u)>>24);if(t==D_I16)return float(int(v<<16u)>>16);return t==D_I32?float(int(v)):float(v);}
void complexWriteF(uint slot,uint index,float value){uint t=outDtype(slot);if(t==D_F32)writeU(slot,index,floatBitsToUint(value));else if(t==D_F16)writeU(slot,index,xrgpuFloatToHalf(value));else if(t==D_I32||t==D_I8||t==D_I16)writeI(slot,index,int(value));else writeU(slot,index,uint(value));}
#define readF complexReadF
#define writeF complexWriteF
#endif
#if XRGPU_COMPLEX_OPCODE == 1012 || XRGPU_COMPLEX_OPCODE == 1027 || XRGPU_COMPLEX_OPCODE == 1029
#include "complex_math.glsl"
#endif
layout(local_size_x=64) in;
const uint NONE=0xffffffffu;
uint op(){return uint(XRGPU_COMPLEX_OPCODE);}
int controlIndex(uint slot,uint index){if(dtype(slot)!=D_I64)return readI(slot,index);uvec2 x=read64(slot,index);if(x.y==0u&&x.x<=0x7fffffffu)return int(x.x);if(x.y==0xffffffffu&&x.x>=0x80000000u)return int(x.x);return (x.y&0x80000000u)!=0u?int(0x80000000u):0x7fffffff;}

uint os(uint canonical){return arg(64u+canonical);}
void put(uint canonical,uint i,float x){uint s=os(canonical);if(s!=NONE&&i<outSize(s))writeF(s,i,x);}
void putI(uint canonical,uint i,int x){uint s=os(canonical);if(s!=NONE&&i<outSize(s))writeI(s,i,x);}
float get(uint s,uint i,float fallback){return s!=NONE&&has(s)&&i<size(s)?readF(s,i):fallback;}
int geti(uint s,uint i,int fallback){return s!=NONE&&has(s)&&i<size(s)?readI(s,i):fallback;}
uint length(uint s){return rank(s)>0?dim(s,rank(s)-1u):1u;}
float logistic(float x){return x>=0.0?1.0/(1.0+exp(-x)):exp(x)/(1.0+exp(x));}
float softplus(float x){return max(x,0.0)+log(1.0+exp(-abs(x)));}
float fused(float x,uint kind){if(kind==1u)return max(x,0.0);if(kind==2u)return x*logistic(x);if(kind==3u){
#if XRGPU_COMPLEX_OPCODE == 1027 || XRGPU_COMPLEX_OPCODE == 1029
 return .5*x*(1.0+unaryErf(x*.7071067811865475));
#else
 return .5*x*(1.0+tanh(.7978845608028654*(x+.044715*x*x*x)));
#endif
 }if(kind==4u)return tanh(x);if(kind==5u)return logistic(x);return x;}
float halfRound(float x){return xrgpuHalfToFloat(xrgpuFloatToHalf(x));}
uint broad4(uint slot,uvec4 c){uint n=rank(slot),idx=0u;for(uint r=0u;r<n;++r){uint axis=4u-n+r;idx+=(dim(slot,r)==1u?0u:c[axis])*stride(slot,r);}return idx;}
#include "complex_quant_cache.glsl"
#if XRGPU_COMPLEX_OPCODE == 1002
#include "complex_gqa.glsl"
#endif
#if XRGPU_COMPLEX_OPCODE == 1004
#include "complex_sparse.glsl"
#endif


float embeddingSum(uint b,uint t,uint j){uint hidden=dim(2u,1u),seq=dim(0u,1u),row=b*seq+t;int wi=readI(0u,row),pi=geti(8u,(has(8u)&&dim(8u,0u)==1u)?t:row,int(t)),si=geti(1u,row,0);if(wi<0||uint(wi)>=dim(2u,0u)||pi<0||uint(pi)>=dim(3u,0u)||(has(4u)&&(si<0||uint(si)>=dim(4u,0u)))){fail(55u);return 0.0;}float w=readF(2u,uint(wi)*hidden+j),p=readF(3u,uint(pi)*hidden+j),s=get(4u,uint(si)*hidden+j,0.0);return dtype(2u)==D_F16?halfRound(halfRound(w+s)+p):(w+p)+s;}

void gates(){
 if(op()==1015u){uint h=length(0u);for(uint i=0u;i<size(0u);++i){float x=readF(0u,i)+readF(1u,i%h);put(0u,i,softplus(x)*readF(2u,i%h));if(has(3u))put(1u,i,logistic(readF(3u,i)));}return;}
 if(op()==1033u){uint d=dim(0u,3u),hc=dim(0u,2u),rows=size(0u)/max(1u,d);for(uint row=0u;row<rows;++row){float k2=0.0,q2=0.0,dot=0.0;for(uint j=0u;j<d;++j){float k=readF(0u,row*d+j),q=readF(1u,row*d+j);k2+=k*k;q2+=q*q;dot+=k*q*readF(3u,(row%hc)*d+j)*readF(4u,(row%hc)*d+j);}dot*=inversesqrt(k2/float(d)+argF(1))*inversesqrt(q2/float(d)+argF(1))/sqrt(float(d));float gate=logistic(sign(dot)*sqrt(max(abs(dot),1e-6)));for(uint j=0u;j<d;++j)put(0u,row*d+j,gate*readF(2u,(row/hc)*d+j));}return;}
 if(op()==1034u){uint batch=dim(0u,0u),seq=dim(0u,1u),heads=arg(1),h=dim(0u,2u)/heads,d=dim(3u,1u);for(uint b=0u;b<batch;++b)for(uint head=0u;head<heads;++head)for(uint t=0u;t<seq;++t){float u=0.0,r=0.0;for(uint j=0u;j<d;++j){float a=readF(4u,j);for(uint k=0u;k<h;++k)a+=(readF(0u,(b*seq+t)*heads*h+head*h+k)+readF(1u,head*h+k))*readF(3u,k*d+j);if(j<d/2u)u+=a;else r+=a;}float gate=logistic(u)*(logistic(r)*readF(5u,head)-1.0)+2.0;for(uint k=0u;k<seq;++k)put(0u,((b*heads+head)*seq+t)*seq+k,gate*readF(2u,(head*seq+t)*seq+k));}return;}
 if(op()==1032u){uint batch=dim(0u,0u),seq=dim(0u,1u),hidden=dim(2u,1u);for(uint b=0u;b<batch;++b){int mask=has(7u)?int(seq):0;for(uint t=0u;t<seq;++t){uint row=b*seq+t;float mean=0.0,variance=0.0;for(uint j=0u;j<hidden;++j){float x=embeddingSum(b,t,j);mean+=x;put(2u,row*hidden+j,x);}mean/=float(hidden);for(uint j=0u;j<hidden;++j){float x=embeddingSum(b,t,j);variance+=(x-mean)*(x-mean);}float inv=inversesqrt(variance/float(hidden)+argF(1));for(uint j=0u;j<hidden;++j)put(0u,row*hidden+j,(embeddingSum(b,t,j)-mean)*inv*readF(5u,j)+readF(6u,j));if(has(7u)&&readI(7u,row)==0&&mask==int(seq))mask=int(t);}putI(1u,b,mask);}return;}
}

int historyId(uint b,int t){uint seq=dim(0u,1u),history=arg(1)-1u;if(t>=0&&uint(t)<seq)return readI(0u,b*seq+uint(t));int old=t+int(history);return old>=0&&has(3u)?readI(3u,b*history+uint(old)):int(arg(3));}
void ngram(){uint batch=dim(0u,0u),seq=dim(0u,1u),orders=arg(1),heads=arg(2),count=(orders-1u)*heads;for(uint b=0u;b<batch;++b){for(uint t=0u;t<seq;++t){uint mix=0u;for(uint shift=0u;shift<orders;++shift){mix^=uint(historyId(b,int(t)-int(shift)))*readU(1u,shift);if(shift>0u)for(uint h=0u;h<heads;++h){uint outputHead=(shift-1u)*heads+h;int vocab=readI(2u,outputHead),v=vocab>0?int(mix)%vocab:0;if(v<0)v+=vocab;putI(0u,(b*seq+t)*count+outputHead,v);}}}for(uint j=0u;j<orders-1u;++j)putI(1u,b*(orders-1u)+j,historyId(b,int(seq)-int(orders-1u)+int(j)));}}

void rotary(){
 if(op()==1012u){uint batch=dim(1u,0u),heads=dim(1u,1u),seq=dim(1u,2u),d=dim(1u,3u);for(uint b=0u;b<batch;++b)for(uint h=0u;h<heads;++h)for(uint t=0u;t<seq;++t)for(uint j=0u;j<d;++j){uint i=((b*heads+h)*seq+t)*d+j;float angle=readF(0u,(b*seq+t)*d+j),c=halfRound(cos(angle)),s=halfRound(sin(angle));put(0u,i,halfRound(readF(1u,i)*c)+halfRound(readF(2u,i)*s));put(1u,i,halfRound(readF(3u,i)*c)+halfRound(readF(4u,i)*s));}return;}
 uint heads=max(1u,arg(1)),d=rank(0u)==4u?dim(0u,3u):dim(0u,2u)/heads,seq=rank(0u)==4u?dim(0u,2u):dim(0u,1u),batch=dim(0u,0u),rd=arg(3)==0u?d:arg(3);uint cs=op()==1009u?1u:2u,ss=cs+1u,ps=op()==1009u?3u:1u;
 for(uint b=0u;b<batch;++b)for(uint h=0u;h<heads;++h)for(uint t=0u;t<seq;++t)for(uint j=0u;j<d;++j){uint base=rank(0u)==4u?((b*heads+h)*seq+t)*d:((b*seq+t)*heads+h)*d,i=base+j;float x=readF(0u,i);if(j>=rd){put(0u,i,x);continue;}uint pair=arg(2)!=0u?j/2u:j%(rd/2u),mate=arg(2)!=0u?(j^1u):(j<rd/2u?j+rd/2u:j-rd/2u);float signValue=arg(2)!=0u?(j%2u==0u?-1.0:1.0):(j<rd/2u?-1.0:1.0);int position=int(t);if(has(ps)){if(op()==1011u){uint axis=0u,offset=0u;for(uint section=0u;section<arg(7);++section){offset+=arg(8u+section);if(pair<offset){axis=section%3u;break;}}position=readI(ps,(axis*batch+b)*seq+t);}else position=size(ps)==1u?readI(ps,0u)+int(t):readI(ps,b*seq+t);}if(op()==1011u&&(position<0||uint(position)>=dim(cs,0u))){put(0u,i,x);continue;}if(position<0)position=0;uint cache=uint(position)*dim(cs,1u)+pair;float result=x*readF(cs,cache)+signValue*readF(0u,base+mate)*readF(ss,cache);put(0u,i,result*argF(4));}
}

uint matrixOffset(uint slot,uint batch,uint row,uint col,bool left){uint n=rank(slot);if(n==1u)return left?col:row;uint index=row*dim(slot,n-1u)+col;uint rankBatch=n-2u,outBatch=outRank(os(0u))-(arg(4)==0u?1u:0u)-(arg(5)==0u?1u:0u);for(uint rev=0u;rev<outBatch;++rev){uint width=outDim(os(0u),outBatch-1u-rev),coordinate=batch%width;batch/=width;if(rev<rankBatch){uint d=n-3u-rev;index+=(dim(slot,d)==1u?0u:coordinate)*stride(slot,d);}}return index;}
void integerMatmul(){uint bs=op()==1023u?1u:3u,k=arg(1),m=arg(2),n=arg(3),count=outSize(os(0u));for(uint linearIndex=0u;linearIndex<count;++linearIndex){uint col=arg(5)!=0u?0u:linearIndex%n,row=(linearIndex/max(1u,n))%max(1u,m),batch=linearIndex/max(1u,m*n);uint sum=0u;for(uint j=0u;j<k;++j){int a=readI(0u,matrixOffset(0u,batch,row,j,true)),b=readI(bs,matrixOffset(bs,batch,j,col,false));int az=op()==1023u?geti(2u,size(2u)>1u?row:0u,0):readI(2u,0u),bz=op()==1023u?geti(3u,size(3u)>1u?col:0u,0):readI(5u,size(5u)>1u?col:0u);sum+=uint(a-az)*uint(b-bz);}if(op()==1023u)putI(0u,linearIndex,int(sum));else{float scale=readF(1u,0u)*readF(4u,size(4u)>1u?col:0u)/readF(6u,0u),x=roundEven(float(int(sum))*scale)+readF(7u,0u);uint type=outDtype(os(0u));put(0u,linearIndex,clamp(x,type==D_I8?-128.0:0.0,type==D_I8?127.0:255.0));}}}

float nbitWeight(uint bs,uint scales,uint zero,uint column,uint k,uint columns){uint bits=arg(4),block=arg(3),blocks=(arg(1)+block-1u)/block,blockBytes=(block*bits+7u)/8u,element=k%block,shift=(element*bits)%8u;uint packed=readU(bs,(column*blocks+k/block)*blockBytes+(element*bits)/8u);uint q=(packed>>shift)&((1u<<bits)-1u);float z=zero!=NONE&&has(zero)?readF(zero,column*blocks+k/block):float(1u<<(bits-1u));return (float(q)-z)*readF(scales,column*blocks+k/block);}
float packedInput(uint row,uint k,bool normalized){float x=readF(0u,row*arg(1)+k);if(normalized){x+=get(1u,row*arg(1)+k,0.0);if(has(2u)){float square=0.0;for(uint j=0u;j<arg(1);++j){float a=readF(0u,row*arg(1)+j)+get(1u,row*arg(1)+j,0.0);square+=a*a;}x*=inversesqrt(square/float(arg(1))+argF(7))*readF(2u,k);}}return x;}
float bnbWeight(uint k,uint n){const float nf4[16]=float[16](-1.0,-.6961928009986877,-.5250730514526367,-.3949174880981445,-.28444138169288635,-.18477343022823334,-.09105003625154495,0.0,.07958029955625534,.16093020141124725,.24611230194568634,.33791524171829224,.44070982933044434,.5626170039176941,.7229568362236023,1.0);const float fp4[16]=float[16](0.0,.0052083333333333,.6666666666666666,1.0,.3333333333333333,.5,.1666666666666666,.25,-0.0,-.0052083333333333,-.6666666666666666,-1.0,-.3333333333333333,-.5,-.1666666666666666,-.25);uint idx=arg(6)!=0u?n*arg(1)+k:k*arg(2)+n,byte=readU(1u,idx/2u),code=idx%2u==0u?byte>>4u:byte&15u;return (arg(5)==1u?nf4[code]:fp4[code])*readF(2u,idx/arg(3));}
void packedMatmul(){uint k=arg(1),rows=size(0u)/k;bool norm=op()==1027u||op()==1028u;for(uint row=0u;row<rows;++row){if(norm)for(uint j=0u;j<k;++j)put(op()==1028u?3u:1u,row*k+j,readF(0u,row*k+j)+get(1u,row*k+j,0.0));uint branches=op()==1028u?3u:1u;for(uint branch=0u;branch<branches;++branch){uint n=op()==1028u&&branch>0u?arg(9):arg(2);for(uint col=0u;col<n;++col){float sum=0.0,up=0.0;for(uint j=0u;j<k;++j){float a=packedInput(row,j,norm),w;if(op()==1026u)w=bnbWeight(j,col);else if(op()==1028u)w=nbitWeight(3u+2u*branch,4u+2u*branch,NONE,col,j,n);else if(op()==1027u)w=nbitWeight(3u,4u,NONE,col,j,n);else w=nbitWeight(1u,2u,3u,col,j,n);sum+=a*w;if(op()==1027u)up+=a*nbitWeight(6u,7u,NONE,col,j,n);}if(op()==1027u){sum=fused(sum+get(5u,col,0.0),arg(8))*(up+get(8u,col,0.0));}else if(op()==1025u)sum+=get(4u,col,0.0);put(branch,row*n+col,sum);}}}}

void gatherQuant(){uint bits=arg(1),block=arg(2),rows=dim(0u,0u),width=dim(0u,1u)*(8u/bits),blocks=(width+block-1u)/block;for(uint row=0u;row<dim(1u,0u);++row){int selected=controlIndex(1u,row);if(selected<0||uint(selected)>=rows){fail(55u);continue;}for(uint j=0u;j<width;++j){uint shift=(j*bits)%8u,packed=readU(0u,uint(selected)*dim(0u,1u)+(j*bits)/8u),q=(packed>>shift)&((1u<<bits)-1u),z=has(3u)?(readU(3u,uint(selected)*((blocks*bits+7u)/8u)+(j/block*bits)/8u)>>((j/block*bits)%8u))&((1u<<bits)-1u):(1u<<(bits-1u));float scale=readF(2u,uint(selected)*blocks+j/block);put(0u,row*width+j,(float(q)-float(z))*scale);}}}

float recurrentActivation(float x,uint number){float clip=argF(4);if(clip>0.0)x=clamp(x,-clip,clip);uint code=arg(16u+number);float a=argF(24u+number),b=argF(32u+number);if(code==0u)return logistic(x);if(code==1u)return tanh(x);if(code==2u)return max(x,0.0);if(code==3u)return a*x+b;if(code==4u)return x>=0.0?x:a*x;if(code==5u)return x>a?x:0.0;if(code==6u)return a*tanh(b*x);if(code==7u)return clamp(a*x+b,0.0,1.0);if(code==8u)return x>=0.0?x:a*(exp(x)-1.0);if(code==9u)return x/(1.0+abs(x));return softplus(x);}
uint recurrentStateIndex(uint d,uint b,uint j){return arg(1)==0u?(d*arg(6)+b)*arg(7)+j:(b*arg(8)+d)*arg(7)+j;}
uint recurrentYIndex(uint t,uint d,uint b,uint j){return arg(1)==0u?((t*arg(8)+d)*arg(6)+b)*arg(7)+j:((b*arg(5)+t)*arg(8)+d)*arg(7)+j;}
float recurrentLinear(uint d,uint b,uint t,uint gate,uint j,uint count,bool recurrent){uint hidden=arg(7),inputSize=arg(9),gates=op()==1020u?1u:op()==1021u?3u:4u;float value=get(3u,d*(2u*gates*hidden)+gate*hidden+j,0.0);if(recurrent)value+=get(3u,d*(2u*gates*hidden)+(gates+gate)*hidden+j,0.0);for(uint i=0u;i<inputSize;++i){uint xi=arg(1)==0u?(t*arg(6)+b)*inputSize+i:(b*arg(5)+t)*inputSize+i;value+=readF(0u,xi)*readF(1u,(d*gates*hidden+gate*hidden+j)*inputSize+i);}if(recurrent)for(uint i=0u;i<hidden;++i)value+=scratchF(i)*readF(2u,(d*gates*hidden+gate*hidden+j)*hidden+i);return value;}
void recurrent(){uint hidden=arg(7),seq=arg(5),batch=arg(6),dirs=arg(8);for(uint d=0u;d<dirs;++d)for(uint b=0u;b<batch;++b){for(uint j=0u;j<hidden;++j){setScratchF(j,get(5u,recurrentStateIndex(d,b,j),0.0));setScratchF(hidden+j,get(6u,recurrentStateIndex(d,b,j),0.0));}uint valid=uint(max(0,geti(4u,b,int(seq))));valid=min(valid,seq);bool reverse=arg(2)==1u||(arg(2)==2u&&d==1u);for(uint step=0u;step<valid;++step){uint t=reverse?valid-1u-step:step;if(op()==1020u){for(uint j=0u;j<hidden;++j)setScratchF(2u*hidden+j,recurrentActivation(recurrentLinear(d,b,t,0u,j,hidden,true),d));}else if(op()==1021u){for(uint j=0u;j<hidden;++j){setScratchF(3u*hidden+j,recurrentActivation(recurrentLinear(d,b,t,0u,j,hidden,true),2u*d));setScratchF(4u*hidden+j,recurrentActivation(recurrentLinear(d,b,t,1u,j,hidden,true),2u*d));}for(uint j=0u;j<hidden;++j){float candidate=recurrentLinear(d,b,t,2u,j,hidden,false),rec=get(3u,d*6u*hidden+5u*hidden+j,0.0);for(uint i=0u;i<hidden;++i)rec+=(arg(3)==0u?scratchF(4u*hidden+i):1.0)*scratchF(i)*readF(2u,(d*3u*hidden+2u*hidden+j)*hidden+i);candidate+=arg(3)!=0u?scratchF(4u*hidden+j)*rec:rec;float h=recurrentActivation(candidate,2u*d+1u),z=scratchF(3u*hidden+j);setScratchF(2u*hidden+j,(1.0-z)*h+z*scratchF(j));}}else{for(uint j=0u;j<hidden;++j){float oldc=scratchF(hidden+j),peepholeI=get(7u,d*3u*hidden+j,0.0),peepholeF=get(7u,d*3u*hidden+2u*hidden+j,0.0);float i=recurrentActivation(recurrentLinear(d,b,t,0u,j,hidden,true)+peepholeI*oldc,3u*d),f=arg(3)!=0u?1.0-i:recurrentActivation(recurrentLinear(d,b,t,2u,j,hidden,true)+peepholeF*oldc,3u*d),candidate=recurrentActivation(recurrentLinear(d,b,t,3u,j,hidden,true),3u*d+1u);float c=f*oldc+i*candidate;float o=recurrentActivation(recurrentLinear(d,b,t,1u,j,hidden,true)+get(7u,d*3u*hidden+hidden+j,0.0)*c,3u*d);setScratchF(2u*hidden+j,o*recurrentActivation(c,3u*d+2u));setScratchF(5u*hidden+j,c);}}for(uint j=0u;j<hidden;++j){setScratchF(j,scratchF(2u*hidden+j));if(op()==1022u)setScratchF(hidden+j,scratchF(5u*hidden+j));put(0u,recurrentYIndex(t,d,b,j),scratchF(j));}}for(uint j=0u;j<hidden;++j){put(1u,recurrentStateIndex(d,b,j),scratchF(j));if(op()==1022u)put(2u,recurrentStateIndex(d,b,j),scratchF(hidden+j));}}}

float causalInput(uint b,uint c,int t,uint start,uint seq){uint channels=arg(7),length=arg(9);bool variable=op()==1019u;if(t>=0&&uint(t)<seq){uint index=variable?(start+uint(t))*channels+c:arg(1)!=0u?(b*seq+uint(t))*channels+c:(b*channels+c)*seq+uint(t);return readF(0u,index);}uint stateSlot=variable?4u:3u;int history=t+int(length);if(history>=0&&uint(history)<length&&has(stateSlot)){uint index=!variable&&arg(1)!=0u?(b*length+uint(history))*channels+c:(b*channels+c)*length+uint(history);if(!variable&&arg(3)>0u)index+=(arg(3)-1u)*arg(6)*channels*length;return readF(stateSlot,index);}return 0.0;}
void causal(){uint batch=arg(6),channels=arg(7),kernel=arg(8),history=arg(9);bool variable=op()==1019u;for(uint b=0u;b<batch;++b){uint start=variable?readU(2u,b):0u,seq=variable?readU(2u,b+1u)-start:dim(0u,arg(1)!=0u?1u:2u),bias=variable?3u:2u;for(uint c=0u;c<channels;++c){for(uint t=0u;t<seq;++t){float result=get(bias,c,0.0);for(uint k=0u;k<kernel;++k)result+=causalInput(b,c,int(t)-int((kernel-1u-k)*arg(2)),start,seq)*readF(1u,c*kernel+k);uint index=variable?(start+t)*channels+c:arg(1)!=0u?(b*seq+t)*channels+c:(b*channels+c)*seq+t;put(0u,index,fused(result,arg(4)));}uint windows=variable?1u:max(1u,arg(3));for(uint window=0u;window<windows;++window)for(uint j=0u;j<history;++j){int at=int(seq)-int(windows)+1+int(window)-int(history)+int(j);uint dest=!variable&&arg(1)!=0u?((window*batch+b)*history+j)*channels+c:((window*batch+b)*channels+c)*history+j;put(1u,dest,causalInput(b,c,at,start,seq));}if(variable&&os(2u)!=NONE){uint cap=arg(5),count=uint(max(0,geti(5u,b,int(cap))));for(uint j=0u;j<cap;++j)put(2u,(b*cap+j)*channels+c,j<min(count,seq)?causalInput(b,c,int(j),start,seq):0.0);}}}}

// Additional family algorithms are defined below before the single invocation.
uint attentionStart(uint b){if(op()==1005u)return readU(4u,b);if(op()==1006u)return readU(5u,b);if(op()==1007u)return readU(5u,b);return 0u;}
uint attentionQueries(uint b){if(op()==1005u||op()==1006u||op()==1007u){uint cu=op()==1005u?4u:5u;return readU(cu,b+1u)-readU(cu,b);}return arg(6);}
uint attentionPast(uint b){if(op()==1002u)return uint(max(0,geti(5u,b,int(arg(7)))-int(arg(7))+1));if(op()==1004u)return uint(max(0,geti(8u,b,int(arg(6)))-int(arg(6))));if(op()==1007u)return uint(max(0,geti(6u,b,0)));return arg(5);}
uint attentionKeys(uint b){if(op()==1005u||op()==1006u)return attentionQueries(b);if(op()==1007u)return attentionPast(b)+(has(1u)?attentionQueries(b):0u);if(op()==1002u)return uint(max(0,geti(5u,b,int(arg(7))-1)+1));if(op()==1004u)return uint(max(0,geti(8u,b,int(arg(6)))));return attentionPast(b)+arg(7);}
float projected(uint role,uint b,uint h,uint t,uint j){uint hidden=arg(1)*arg(8),column=role*hidden+h*arg(8)+j;float value=readF(2u,column);uint width=dim(0u,2u);for(uint k=0u;k<width;++k){int a=readI(0u,(b*arg(6)+t)*width+k)-geti(6u,0u,0),w=readI(1u,k*3u*hidden+column)-geti(7u,size(7u)>1u?column:0u,0);value+=float(a*w)*readF(3u,0u)*readF(4u,size(4u)>1u?column:0u);}return value;}
float attentionRaw(uint role,uint b,uint h,uint t,uint j){
 uint qh=arg(1),kh=arg(2),d=role==2u?arg(9):arg(8),past=attentionPast(b),slot=role;
 if(op()==1008u){if(role>0u&&t<past){uint offset=((((role-1u)*arg(10)+b)*qh+h)*past+t)*d+j;return readF(8u,offset);}return projected(role,b,h,role==0u?t:t-past,j);}
 if(op()==1005u){uint token=attentionStart(b)+t;return scratchF(arg(28u+role)+token*qh*d+h*d+j);}
 if(op()==1006u){uint token=attentionStart(b)+t;if(!has(1u)){uint index=rank(0u)==4u?((token*qh+h)*3u+role)*d+j:(token*3u*qh+role*qh+h)*d+j;return readF(0u,index)+get(3u,role*qh*d+h*d+j,0.0);}return readF(slot,token*(role==0u?qh:kh)*d+h*d+j)+get(3u,role*qh*arg(8)+h*d+j,0.0);}
 if(op()==1007u){if(role==0u)return readF(0u,(attentionStart(b)+t)*dim(0u,1u)+h*d+j);uint cache=role==1u?3u:4u,blockSize=dim(cache,1u),physical=readU(7u,b*dim(7u,1u)+t/blockSize)*blockSize+t%blockSize;return scratchF(arg(27u+role)+(physical*kh+h)*d+j);}
 if(role>0u){uint cache=NONE;if(op()==1000u)cache=role+3u;else if(op()==1002u||op()==1004u)cache=role+2u;else if(op()==1003u)cache=role+4u;if(t<past&&op()==1002u)return complexCacheRead(role,b,h,t,j);if(t<past&&cache!=NONE&&has(cache)){float v=readF(cache,((b*kh+h)*dim(cache,2u)+t)*d+j);return v;}t-=past;}
 if(op()==1004u&&!has(1u)){uint hidden=qh*arg(8),r=role==0u?h:h*kh/qh;return readF(0u,((b*arg(6)+t)*3u*hidden)+role*hidden+r*d+j);}
 uint heads=role==0u?qh:kh;uint index=rank(slot)==4u?((b*heads+h)*dim(slot,2u)+t)*d+j:((b*dim(slot,1u)+t)*heads+h)*d+j;
 float value=readF(slot,index);if(op()==1001u)value+=get(3u,role*qh*arg(8)+h*d+j,0.0);return value;
}
float attentionNormalized(uint role,uint b,uint h,uint t,uint j){float value=attentionRaw(role,b,h,t,j);if(op()!=1002u||role==2u||(role==1u&&t<attentionPast(b)))return value;uint normSlot=role==0u?13u:14u,d=arg(8);if(has(normSlot)){float square=0.0;for(uint k=0u;k<d;++k){float x=attentionRaw(role,b,h,t,k);square+=x*x;}value*=inversesqrt(square/float(d)+argF(13))*readF(normSlot,j);}return value;}
float attentionQuantize(uint role,uint b,uint h,uint t,uint j,float x){return op()==1002u&&role>0u&&t>=attentionPast(b)?complexCacheRoundtrip(role,h,j,x):x;}
float attentionValue(uint role,uint b,uint h,uint t,uint j){float x=attentionNormalized(role,b,h,t,j);if((op()!=1002u&&op()!=1004u)||arg(14)==0u||role==2u||(role==1u&&t<attentionPast(b)))return attentionQuantize(role,b,h,t,j,x);uint cs=op()==1002u?7u:9u,ss=cs+1u;if(!has(cs)||!has(ss)){fail(54u);return 0.0;}uint rd=min(arg(8),2u*dim(cs,1u));if(j>=rd)return attentionQuantize(role,b,h,t,j,x);uint mate=arg(26)!=0u?(j^1u):(j<rd/2u?j+rd/2u:j-rd/2u),pair=arg(26)!=0u?j/2u:j%(rd/2u),position=role==0u?attentionPast(b)+t:t;if(position>=dim(cs,0u)){fail(54u);return 0.0;}float signValue=arg(26)!=0u?(j%2u==0u?-1.0:1.0):(j<rd/2u?-1.0:1.0);float rotated=x*readF(cs,position*dim(cs,1u)+pair)+signValue*attentionNormalized(role,b,h,t,mate)*readF(ss,position*dim(ss,1u)+pair);return attentionQuantize(role,b,h,t,j,rotated);}

float attentionScore(uint b,uint h,uint qi,uint kj){uint kh=h/(arg(1)/arg(2));float sum=0.0;for(uint d=0u;d<arg(8);++d)sum+=attentionValue(0u,b,h,qi,d)*attentionValue(1u,b,kh,kj,d);sum*=argF(3);if(argF(12)>0.0)sum=argF(12)*tanh(sum/argF(12));
 uint keys=attentionKeys(b),queries=attentionQueries(b);if(arg(4)!=0u&&kj>attentionPast(b)+qi)return uintBitsToFloat(0xff800000u);if(int(arg(15))>=0&&kj+arg(15)<attentionPast(b)+qi)return uintBitsToFloat(0xff800000u);
 if(op()==1000u&&has(3u)){uint i=rank(3u)==3u?((dim(3u,0u)==1u?0u:b)*dim(3u,1u)+qi)*dim(3u,2u)+kj:broad4(3u,uvec4(b,h,qi,kj));if(dtype(3u)==D_BOOL){if(readU(3u,i)==0u)return uintBitsToFloat(0xff800000u);}else sum+=readF(3u,i);}
 uint bias=op()==1001u?4u:op()==1002u?9u:op()==1003u?4u:op()==1005u?5u:op()==1006u?6u:NONE;if(bias!=NONE&&has(bias)){uint pqi=qi,pkj=kj;if(op()==1005u||op()==1006u){uint offsetSlot=op()==1005u?3u:4u,width=dim(offsetSlot,1u);pqi=readU(offsetSlot,attentionStart(b)+qi)%width;pkj=readU(offsetSlot,attentionStart(b)+kj)%width;}sum+=readF(bias,broad4(bias,uvec4(b,h,pqi,pkj)));}
 uint mask=op()==1003u?3u:op()==1008u?5u:NONE;if(mask!=NONE&&has(mask)){if(rank(mask)==1u){int end=readI(mask,b);if(int(kj)>=end)return uintBitsToFloat(0xff800000u);}else if(readI(mask,b*dim(mask,1u)+kj)==0){if(op()==1003u)sum+=argF(16);else return uintBitsToFloat(0xff800000u);}}
 if(op()==1004u){uint block=uint(int(arg(20))>0?arg(20):16u),row=(attentionPast(b)+qi)/block,head=h%dim(5u,0u),rowWidth=dim(5u,1u);uint begin=readU(5u,head*rowWidth+row),end=readU(5u,head*rowWidth+row+1u);bool permitted=false;for(uint i=begin;i<end;++i)permitted=permitted||readU(6u,head*dim(6u,1u)+i)==kj/block;if(!permitted)return uintBitsToFloat(0xff800000u);}
 return sum;}
void attention(){uint heads=arg(1),kh=arg(2),d=arg(8),vd=arg(9);
 for(uint b=0u;b<arg(10);++b){uint queries=attentionQueries(b),keys=attentionKeys(b);
  for(uint h=0u;h<heads;++h)for(uint qi=0u;qi<queries;++qi){uint first=0u;if(op()==1002u){uint past=attentionPast(b);if(arg(24)!=0u&&has(3u)&&past>dim(3u,2u))first=past-dim(3u,2u);if(int(arg(15))>=0&&past+qi>arg(15))first=max(first,past+qi-arg(15));}float maximum=uintBitsToFloat(0xff800000u);
   for(uint j=first;j<keys;++j){float score=attentionScore(b,h,qi,j);setScratchF(j-first,score);maximum=max(maximum,score);if(op()==1003u)put(3u,((b*heads+h)*queries+qi)*keys+j,score);}
   float denominator=0.0;for(uint j=first;j<keys;++j){float score=scratchF(j-first),weight=isinf(maximum)?(maximum>0.0&&score==maximum?1.0:0.0):exp(score-maximum);setScratchF(j-first,weight);denominator+=weight;}
   if(op()==1002u&&has(10u))denominator+=exp(readF(10u,h)-maximum);if(op()==1002u&&arg(25)==1u)denominator+=exp(-maximum);
   for(uint v=0u;v<vd;++v){float sum=0.0;for(uint j=first;j<keys;++j)sum+=scratchF(j-first)*attentionValue(2u,b,h/(heads/kh),j,v);uint outputIndex=(op()==1005u||op()==1006u||op()==1007u)?((attentionStart(b)+qi)*heads+h)*vd+v:op()==1000u&&rank(0u)==4u?((b*heads+h)*queries+qi)*vd+v:((b*queries+qi)*heads+h)*vd+v;put(0u,outputIndex,denominator>0.0?sum/denominator:0.0);}
  }
  if(op()==1000u||op()==1003u||op()==1004u||op()==1008u)for(uint role=1u;role<=2u;++role){uint outputSlot=op()==1008u?os(1u):os(role);if(outputSlot==NONE)continue;uint width=role==1u?d:vd,capacity=op()==1008u?keys:outDim(outputSlot,2u);for(uint h=0u;h<kh;++h)for(uint t=0u;t<capacity;++t)for(uint j=0u;j<width;++j){uint target=op()==1008u?((((role-1u)*arg(10)+b)*kh+h)*capacity+t)*width+j:((b*kh+h)*capacity+t)*width+j;float x=t<keys?attentionValue(role,b,h,t,j):0.0;if((op()==1003u||op()==1004u)&&t>=keys){uint cache=op()==1003u?role+4u:role+2u;x=get(cache,((b*kh+h)*capacity+t)*width+j,0.0);}writeF(outputSlot,target,x);}}
 }
 if(op()==1002u)for(uint role=1u;role<=2u;++role){complexCacheCopy(role);for(uint b=0u;b<arg(10);++b){uint past=attentionPast(b),keys=attentionKeys(b),width=role==1u?d:vd;for(uint h=0u;h<kh;++h)for(uint t=past;t<keys;++t)for(uint j=0u;j<width;++j)complexCacheWrite(role,b,h,t,j,attentionValue(role,b,h,t,j));}}

}

void packedProjection(){uint i=gid(),heads=arg(1),dk=arg(8),dv=arg(9),qcount=dim(0u,0u)*heads*dk,role=i<qcount?0u:i<2u*qcount?1u:2u,relative=role==0u?i:role==1u?i-qcount:i-2u*qcount,width=heads*(role==2u?dv:dk);if(relative>=dim(0u,0u)*width)return;uint token=relative/width,column=relative%width,projectionColumn=(role==2u?2u*heads*dk:role*heads*dk)+column;float sum=readF(2u,projectionColumn);for(uint k=0u;k<dim(0u,1u);++k)sum+=readF(0u,token*dim(0u,1u)+k)*readF(1u,k*dim(1u,1u)+projectionColumn);setScratchF(arg(28u+role)+relative,dtype(0u)==D_F16?halfRound(sum):sum);}

void pagedCacheElement(){uint i=gid(),kcount=size(3u),role=i<kcount?1u:2u,cache=role+2u;if(role==2u)i-=kcount;if(i>=size(cache))return;uint d=arg(8u),kh=arg(2u),heads=arg(1u),physical=i/(kh*d),h=(i/d)%kh,j=i%d,blockSize=dim(cache,1u);float value=readF(cache,i);for(uint b=0u;b<arg(10u);++b)for(uint t=0u;t<attentionQueries(b);++t){uint token=attentionStart(b)+t,position=attentionPast(b)+t;int target=has(8u)?readI(8u,token):int(readU(7u,b*dim(7u,1u)+position/blockSize)*blockSize+position%blockSize);if(target>=0&&uint(target)==physical)value=has(role)?readF(role,(token*kh+h)*d+j):readF(0u,token*dim(0u,1u)+heads*d+(role-1u)*kh*d+h*d+j);}setScratchF(arg(27u+role)+i,value);put(role,i,value);}

void attentionElement(){uint i=gid();if(i>=outSize(os(0u)))return;uint vd=arg(9),heads=arg(1),token=i/(heads*vd),h=(i/vd)%heads,v=i%vd,b=0u;while(b+1u<arg(10)&&token>=attentionStart(b+1u))++b;uint qi=token-attentionStart(b),keys=attentionKeys(b);float maximum=uintBitsToFloat(0xff800000u);for(uint j=0u;j<keys;++j)maximum=max(maximum,attentionScore(b,h,qi,j));float numerator=0.0,denominator=0.0;for(uint j=0u;j<keys;++j){float score=attentionScore(b,h,qi,j),weight=isinf(maximum)?(maximum>0.0&&score==maximum?1.0:0.0):exp(score-maximum);denominator+=weight;numerator+=weight*attentionValue(2u,b,h/(heads/arg(2)),j,v);}put(0u,i,denominator>0.0?numerator/denominator:0.0);}

uint linearStart(uint b){return uint(clamp(readI(3u,b),0,int(dim(0u,0u))));}
uint linearEnd(uint b){return uint(clamp(readI(3u,b+1u),0,int(dim(0u,0u))));}

float linearQ(uint slot,uint b,uint t,uint head,uint j,uint heads,uint d){uint at;if(op()==1016u){if(rank(slot)==4u)at=((b*arg(7)+t)*heads+head)*d+j;else{uint token=has(3u)?linearStart(b)+t:b*arg(7)+t;at=(token*heads+head)*d+j;}}else at=((b*arg(7)+t)*heads+head)*d+j;float x=readF(slot,at);if(op()==1016u&&arg(10)!=0u&&slot<2u){float square=0.0;for(uint k=0u;k<d;++k){float v=readF(slot,at-j+k);square+=v*v;}x*=inversesqrt(square+1e-12);}return x;}
float linearGate(uint slot,uint b,uint t,uint h,uint fallbackHeads,float fallback){if(!has(slot))return fallback;uint token=op()==1016u&&rank(0u)==3u?(has(3u)?linearStart(b)+t:b*arg(7)+t):b*arg(7)+t;uint n=length(slot);return readF(slot,token*n+(n==1u?0u:h%n));}
void linearAttention(){bool delta=op()==1016u;uint qh=arg(1),kh=arg(2),vh=arg(3),dk=arg(4),dv=arg(5),batches=arg(6),stateSlot=delta?6u:3u,heads=delta?vh:kh;for(uint b=0u;b<batches;++b)for(uint h=0u;h<heads;++h){uint stateSize=dk*dv,seq=delta&&has(3u)?uint(max(0,int(linearEnd(b))-int(linearStart(b)))):arg(7);for(uint k=0u;k<dk;++k)for(uint v=0u;v<dv;++v){uint i=delta?((b*heads+h)*dv+v)*dk+k:((b*heads+h)*dk+k)*dv+v;if(!delta&&arg(11)>0u)i+=(arg(11)-1u)*batches*heads*stateSize;setScratchF(k*dv+v,get(stateSlot,i,0.0));}
 if(!delta&&arg(11)>seq)for(uint window=0u;window<arg(11)-seq;++window)for(uint k=0u;k<dk;++k)for(uint v=0u;v<dv;++v){uint destination=(((window*batches+b)*heads+h)*dk+k)*dv+v,source=((((window+seq)*batches+b)*heads+h)*dk+k)*dv+v;put(1u,destination,get(stateSlot,source,0.0));}
 for(uint t=0u;t<seq;++t){float decay=linearGate(4u,b,t,h,heads,0.0),beta=linearGate(5u,b,t,h,heads,1.0);if(delta&&arg(12)!=0u)decay=-exp(readF(7u,h))*softplus(decay+readF(8u,h));if(delta&&arg(13)!=0u)beta=logistic(beta);for(uint i=0u;i<stateSize;++i){float logDecay=decay;if(!delta&&has(4u)&&length(4u)==heads*dk){uint token=b*arg(7)+t;logDecay=readF(4u,token*heads*dk+h*dk+i/dv);}float factor=(arg(8)==1u||arg(8)==3u)?exp(logDecay):1.0;setScratchF(i,scratchF(i)*factor);}
 for(uint v=0u;v<dv;++v){float value=linearQ(2u,b,t,h,v,vh,dv),projection=0.0;if(arg(8)>=2u)for(uint k=0u;k<dk;++k)projection+=linearQ(1u,b,t,h/(heads/arg(14)),k,arg(14),dk)*scratchF(k*dv+v);float update=arg(8)>=2u?beta*(value-projection):value;for(uint k=0u;k<dk;++k)setScratchF(k*dv+v,scratchF(k*dv+v)+linearQ(1u,b,t,h/(heads/arg(14)),k,arg(14),dk)*update);}
 uint outputHeads=delta?vh:max(qh,kh);for(uint oh=0u;oh<outputHeads;++oh){uint stateHead=delta?oh:oh/(outputHeads/kh);if(stateHead!=h)continue;uint queryHead=delta?oh/(vh/qh):oh/(outputHeads/qh);for(uint v=0u;v<dv;++v){float result=0.0;for(uint k=0u;k<dk;++k)result+=linearQ(0u,b,t,queryHead,k,qh,dk)*scratchF(k*dv+v);float scale=argF(9)==0.0?inversesqrt(float(dk)):argF(9);uint token=delta&&rank(0u)==3u?(has(3u)?linearStart(b)+t:b*arg(7)+t):b*arg(7)+t;put(0u,(token*outputHeads+oh)*dv+v,result*scale);}}
 if(!delta&&arg(11)>0u&&t+arg(11)>=seq){uint window=t+arg(11)-seq;for(uint k=0u;k<dk;++k)for(uint v=0u;v<dv;++v)put(1u,(((window*batches+b)*heads+h)*dk+k)*dv+v,scratchF(k*dv+v));}}
 if(delta||arg(11)==0u)for(uint k=0u;k<dk;++k)for(uint v=0u;v<dv;++v){uint index=delta?((b*heads+h)*dv+v)*dk+k:((b*heads+h)*dk+k)*dv+v;put(1u,index,scratchF(k*dv+v));}}
}

float expertWeight(uint slot,uint scales,uint expert,uint row,uint col,uint rows,uint cols){if(op()==1029u)return readF(slot,(expert*rows+row)*cols+col);uint bits=arg(5),packedCols=(cols*bits+7u)/8u,byte=readU(slot,(expert*rows+row)*packedCols+(col*bits)/8u),q=(byte>>((col*bits)%8u))&((1u<<bits)-1u);float value=float(q)-float(1u<<(bits-1u));uint scalesPerRow=size(scales)/(dim(1u,1u)*rows);uint scaleIndex=(expert*rows+row)*scalesPerRow+(scalesPerRow>1u?col/max(1u,arg(6)):0u);return value*readF(scales,scaleIndex);}
uint expertRank(uint token,uint expert){float score=readF(1u,token*dim(1u,1u)+expert);uint rankValue=0u;for(uint e=0u;e<dim(1u,1u);++e){float other=readF(1u,token*dim(1u,1u)+e);if(other>score||(other==score&&e>expert))++rankValue;}return rankValue;}
void moe(){uint i=gid();if(i>=size(0u))return;uint hidden=length(0u),row=i/hidden,column=i%hidden,experts=dim(1u,1u),fc1rows=dim(2u,1u),intermediate=fc1rows/((arg(8)==6u&&arg(3)!=0u)?2u:1u);float maximum=uintBitsToFloat(0xff800000u);for(uint e=0u;e<experts;++e)if((op()==1029u&&arg(2)==0u)||expertRank(row,e)<arg(1))maximum=max(maximum,readF(1u,row*experts+e));float denominator=0.0;for(uint e=0u;e<experts;++e)if((op()==1029u&&arg(2)==0u)||expertRank(row,e)<arg(1))denominator+=exp(readF(1u,row*experts+e)-maximum);float result=0.0;for(uint e=0u;e<experts;++e){if(expertRank(row,e)>=arg(1))continue;float expert=get(op()==1029u?5u:NONE,e*hidden+column,0.0);for(uint j=0u;j<intermediate;++j){uint gateRow=arg(8)==6u&&arg(3)==1u?2u*j:j,linearRow=arg(3)==1u?2u*j+1u:intermediate+j;float gate=get(op()==1029u?3u:NONE,e*fc1rows+gateRow,0.0),linear=0.0;if(arg(8)==6u&&arg(3)!=0u)linear=get(op()==1029u?3u:NONE,e*fc1rows+linearRow,0.0);else if(op()==1029u&&has(6u))linear=get(7u,e*intermediate+j,0.0);for(uint k=0u;k<hidden;++k){float x=readF(0u,row*hidden+k);gate+=x*expertWeight(2u,3u,e,gateRow,k,fc1rows,hidden);if(arg(8)==6u&&arg(3)!=0u)linear+=x*expertWeight(2u,3u,e,linearRow,k,fc1rows,hidden);else if(op()==1029u&&has(6u))linear+=x*readF(6u,(e*intermediate+j)*hidden+k);}float activation;if(arg(8)==6u){if(arg(12)!=0u){gate=min(gate,argF(11));linear=clamp(linear,-argF(11),argF(11));}activation=gate*logistic(argF(9)*gate)*(linear+argF(10));}else{activation=fused(gate,arg(8));if(op()==1029u&&has(6u))activation*=linear;}expert+=activation*expertWeight(4u,5u,e,column,j,hidden,intermediate);}result+=exp(readF(1u,row*experts+e)-maximum)/denominator*expert;}put(0u,i,result);}



void rotaryElement(){uint i=gid();
#if XRGPU_COMPLEX_OPCODE == 1012
 if(i>=size(1u))return;uint d=dim(1u,3u),seq=dim(1u,2u),heads=dim(1u,1u),j=i%d,t=(i/d)%seq,b=i/(d*seq*heads);vec2 sc=unarySinCos(readF(0u,(b*seq+t)*d+j));float c=halfRound(sc.y),sn=halfRound(sc.x);put(0u,i,halfRound(readF(1u,i)*c)+halfRound(readF(2u,i)*sn));put(1u,i,halfRound(readF(3u,i)*c)+halfRound(readF(4u,i)*sn));
#else
 if(i>=size(0u))return;uint heads=max(1u,arg(1)),d=rank(0u)==4u?dim(0u,3u):dim(0u,2u)/heads,seq=rank(0u)==4u?dim(0u,2u):dim(0u,1u),batch=dim(0u,0u),rd=arg(3)==0u?d:arg(3),j=i%d,t=rank(0u)==4u?(i/d)%seq:(i/(d*heads))%seq,b=i/(heads*seq*d),base=i-j;float x=readF(0u,i);if(j>=rd){put(0u,i,x);return;}uint cs=op()==1009u?1u:2u,ss=cs+1u,ps=op()==1009u?3u:1u,pair=arg(2)!=0u?j/2u:j%(rd/2u),mate=arg(2)!=0u?(j^1u):(j<rd/2u?j+rd/2u:j-rd/2u);float signValue=arg(2)!=0u?(j%2u==0u?-1.0:1.0):(j<rd/2u?-1.0:1.0);int position=int(t);
 if(has(ps)){if(op()==1011u){uint axis=0u;if(arg(6)==1u){if(pair%3u==1u&&pair<3u*arg(9))axis=1u;else if(pair%3u==2u&&pair<3u*arg(10))axis=2u;}else{uint offset=0u;for(uint section=0u;section<arg(7);++section){offset+=arg(8u+section);if(pair<offset){axis=section%3u;break;}}}position=controlIndex(ps,(axis*batch+b)*seq+t);}else position=size(ps)==1u?controlIndex(ps,0u)+int(t):controlIndex(ps,b*seq+t);}
 uint cache=0u;if(op()==1009u&&rank(cs)==3u&&!has(ps)){if(b>=dim(cs,0u)||t>=dim(cs,1u)){fail(54u);return;}cache=(b*dim(cs,1u)+t)*dim(cs,2u)+pair;}else{if(position<0||uint(position)>=dim(cs,0u)){if(op()==1011u){put(0u,i,x);return;}fail(54u);return;}cache=uint(position)*dim(cs,1u)+pair;}float cosine=readF(cs,cache)*argF(4),sine=readF(ss,cache)*argF(4);if(dtype(0u)==D_F16){cosine=halfRound(cosine);sine=halfRound(sine);}put(0u,i,x*cosine+signValue*readF(0u,base+mate)*sine);
#endif
}

void main(){
#if XRGPU_COMPLEX_OPCODE == 1004
 sparseMain();return;
#endif
#if XRGPU_COMPLEX_OPCODE == 1002
 gqaMain();return;
#endif
#if XRGPU_COMPLEX_OPCODE >= 1009 && XRGPU_COMPLEX_OPCODE <= 1012
 rotaryElement();return;
#endif
#if XRGPU_COMPLEX_OPCODE == 1005 || XRGPU_COMPLEX_OPCODE == 1006 || XRGPU_COMPLEX_OPCODE == 1007
 #if XRGPU_COMPLEX_OPCODE == 1005 && XRGPU_PACKED_PHASE == 0
 packedProjection();
#elif XRGPU_COMPLEX_OPCODE == 1007 && XRGPU_PAGED_PHASE == 0
 pagedCacheElement();
#else
 attentionElement();
#endif
 return;
#endif
#if XRGPU_COMPLEX_OPCODE == 1029 || XRGPU_COMPLEX_OPCODE == 1030
 moe();return;
#endif
 if(gid()!=0u)return;for(uint outputSlot=0u;outputSlot<outputCount();++outputSlot)for(uint i=0u;i<outSize(outputSlot);++i)writeF(outputSlot,i,0.0);uint code=op();if(code<=1008u)attention();else if(code<=1012u)rotary();else if(code==1013u||code==1014u||code==1016u)linearAttention();else if(code==1015u||code==1032u||code==1033u||code==1034u)gates();else if(code>=1017u&&code<=1019u)causal();else if(code>=1020u&&code<=1022u)recurrent();else if(code==1023u||code==1024u)integerMatmul();else if(code>=1025u&&code<=1028u)packedMatmul();else if(code==1029u||code==1030u)moe();else if(code==1031u)gatherQuant();else if(code==1035u)ngram();}
