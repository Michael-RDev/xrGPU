// SPDX-License-Identifier: Apache-2.0
#ifndef XRGPU_EXTENDED_FRAMEWORK
#define XRGPU_EXTENDED_FRAMEWORK
layout(set=0,binding=0,std430) readonly buffer InputHeap {uint inputHeap[];};
layout(set=0,binding=1,std430) buffer OutputHeap {uint outputHeap[];};
layout(set=0,binding=2,std430) readonly buffer ParameterHeap {uint parameters[];};
layout(set=0,binding=3,std430) buffer ScratchHeap {uint scratchHeap[];};
#include "half.glsl"
const uint D_F32=0u,D_F16=1u,D_I32=2u,D_U32=3u,D_I16=4u,D_I8=5u,D_U8=6u,D_I64=7u,D_BOOL=8u;
uint gid(){return (gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x)*64u+gl_LocalInvocationIndex;}
uint arg(uint i){return parameters[1744u+i];}
float argF(uint i){return uintBitsToFloat(arg(i));}
uint inDesc(uint slot){return 16u+slot*36u;}
uint outDesc(uint slot){return 1168u+slot*36u;}
bool has(uint slot){if(slot>=parameters[1]||slot>=32u)return false;return parameters[inDesc(slot)+2u]!=0xffffffffu;}
uint dtype(uint slot){return parameters[inDesc(slot)+1u];}
uint rank(uint slot){return has(slot)?parameters[inDesc(slot)+2u]:0u;}
uint size(uint slot){return has(slot)?parameters[inDesc(slot)+3u]:0u;}
uint dim(uint slot,uint d){return parameters[inDesc(slot)+4u+d];}
uint stride(uint slot,uint d){return parameters[inDesc(slot)+20u+d];}
uint outDtype(uint slot){return parameters[outDesc(slot)+1u];}
uint outRank(uint slot){return parameters[outDesc(slot)+2u];}
uint outSize(uint slot){return parameters[outDesc(slot)+3u];}
uint outDim(uint slot,uint d){return parameters[outDesc(slot)+4u+d];}
uint outStride(uint slot,uint d){return parameters[outDesc(slot)+20u+d];}
uint inputAddress(uint slot,uint index){return parameters[inDesc(slot)]+index*(dtype(slot)==D_I64?2u:1u);}
uint outputAddress(uint slot,uint index){return parameters[outDesc(slot)]+index*(outDtype(slot)==D_I64?2u:1u);}
void fail(uint code);
#include "int64.glsl"
float readInt64Float(uvec2 value){
#ifdef XRGPU_NO_INT64_FLOAT_READ
 fail(3u);return 0.0;
#else
 return extendedInt64ToFloat(value);
#endif
}
uvec2 writeInt64Float(float value){
#ifdef XRGPU_NO_INT64_FLOAT_WRITE
 fail(3u);return uvec2(0u);
#else
 return extendedFloatToInt64(value);
#endif
}
uint readU(uint slot,uint index){if(!has(slot)||index>=size(slot)){fail(1u);return 0u;}return inputHeap[inputAddress(slot,index)];}
uvec2 read64(uint slot,uint index){if(!has(slot)||index>=size(slot)){fail(1u);return uvec2(0u);}uint a=inputAddress(slot,index);return dtype(slot)==D_I64?uvec2(inputHeap[a],inputHeap[a+1u]):uvec2(inputHeap[a],0u);}
int readI(uint slot,uint index){uint v=readU(slot,index),t=dtype(slot);if(t==D_I8)return int(v<<24u)>>24;if(t==D_I16)return int(v<<16u)>>16;return int(v);}
float readF(uint slot,uint index){uint v=readU(slot,index),t=dtype(slot);if(t==D_F32)return uintBitsToFloat(v);if(t==D_F16)return xrgpuHalfToFloat(v);if(t==D_U32||t==D_U8||t==D_BOOL)return float(v);if(t==D_I64)return readInt64Float(read64(slot,index));return float(readI(slot,index));}
void writeU(uint slot,uint index,uint value){if(slot>=parameters[2]||index>=outSize(slot)){fail(2u);return;}uint address=outputAddress(slot,index);outputHeap[address]=value;if(outDtype(slot)==D_I64)outputHeap[address+1u]=0u;}
void write64(uint slot,uint index,uvec2 value);
void writeI(uint slot,uint index,int value){if(outDtype(slot)==D_I64)write64(slot,index,uvec2(uint(value),value<0?0xffffffffu:0u));else writeU(slot,index,uint(value));}
void write64(uint slot,uint index,uvec2 value){if(slot>=parameters[2]||index>=outSize(slot)){fail(2u);return;}uint a=outputAddress(slot,index);outputHeap[a]=value.x;if(outDtype(slot)==D_I64)outputHeap[a+1u]=value.y;}
void writeF(uint slot,uint index,float value){uint t=outDtype(slot);if(t==D_F32)writeU(slot,index,floatBitsToUint(value));else if(t==D_F16)writeU(slot,index,xrgpuFloatToHalf(value));else if(t==D_I64)write64(slot,index,writeInt64Float(value));else if(t==D_BOOL)writeU(slot,index,value!=0.0?1u:0u);else if(t==D_U32||t==D_U8)writeU(slot,index,uint(value));else writeI(slot,index,int(value));}
float readOutF(uint slot,uint index){if(slot>=parameters[2]||index>=outSize(slot)){fail(2u);return 0.0;}uint address=outputAddress(slot,index),v=outputHeap[address],t=outDtype(slot);if(t==D_F32)return uintBitsToFloat(v);if(t==D_F16)return xrgpuHalfToFloat(v);if(t==D_I64)return readInt64Float(uvec2(v,outputHeap[address+1u]));if(t==D_U32||t==D_U8||t==D_BOOL)return float(v);if(t==D_I8)return float(int(v<<24u)>>24);if(t==D_I16)return float(int(v<<16u)>>16);return float(int(v));}
uint readOutU(uint slot,uint index){if(slot>=parameters[2]||index>=outSize(slot)){fail(2u);return 0u;}return outputHeap[outputAddress(slot,index)];}
uvec2 readRaw(uint slot,uint index){return read64(slot,index);}
void writeRaw(uint slot,uint index,uvec2 value){write64(slot,index,value);}
float outF(uint slot,uint index){return readOutF(slot,index);}
float scratchF(uint index){return uintBitsToFloat(scratchHeap[index]);}
void setScratchF(uint index,float value){scratchHeap[index]=floatBitsToUint(value);}
void setCount(uint index,uint value){scratchHeap[index]=value;}
uint getCount(uint index){return scratchHeap[index];}
uint inputCount(){return parameters[1];}
uint outputCount(){return parameters[2];}
void fail(uint code){atomicMax(scratchHeap[parameters[15]],code);}
#endif
