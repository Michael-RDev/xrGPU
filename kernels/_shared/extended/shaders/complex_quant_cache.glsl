// SPDX-License-Identifier: Apache-2.0
// Original native GQA cache storage implementation. All scale/quantization and
// window movement execute on the GPU. Host code provides only validated shapes.
#ifndef XRGPU_COMPLEX_QUANT_CACHE
#define XRGPU_COMPLEX_QUANT_CACHE
uint complexCacheInput(uint role){return role==1u?3u:4u;}
uint complexCacheScaleInput(uint role){return role==1u?11u:12u;}
uint complexCacheWidth(uint role){return role==1u?arg(8u):arg(9u);}
uint complexCacheMode(uint role){return role==1u?arg(22u):arg(23u);}
uint complexCacheTotal(uint b){int last=readI(5u,b);if(last< -1){fail(51u);return 0u;}return uint(last+1);}
uint complexCachePast(uint b){uint total=complexCacheTotal(b),added=arg(7u);return total>added?total-added:0u;}
float complexCacheScale(uint role,uint h,uint d){
 uint slot=complexCacheScaleInput(role),mode=complexCacheMode(role);
 if(mode==0u||!has(slot)){fail(52u);return 1.0;}
 uint index=mode==1u?0u:h*complexCacheWidth(role)+d;
 if(mode>2u||index>=size(slot)){fail(52u);return 1.0;}
 float scale=readF(slot,index);
 if(!(scale>0.0)||isnan(scale)||isinf(scale)){fail(52u);return 1.0;}
 return scale;
}
int complexCacheQuantize(uint role,uint h,uint d,float value){
 uint bits=arg(21u);if(bits!=4u&&bits!=8u){fail(53u);return 0;}
 float scaled=value/complexCacheScale(role,h,d);if(isnan(scaled)){fail(53u);return 0;}
 return int(roundEven(clamp(scaled,bits==4u?-8.0:-128.0,bits==4u?7.0:127.0)));
}
float complexCacheRoundtrip(uint role,uint h,uint d,float value){
 // Direct unquantized prompt variants attend original Q/K/V independently of
 // the requested cache storage type. Quantized routes attend the newly encoded
 // cache, so their new K/V must pass through the exact same quantizer as stores.
 if(complexCacheMode(role)==0u)return value;
 return float(complexCacheQuantize(role,h,d,value))*complexCacheScale(role,h,d);
}
float complexCacheRead(uint role,uint b,uint h,uint absoluteT,uint d){
 uint cache=complexCacheInput(role),width=complexCacheWidth(role);
 if(!has(cache)||rank(cache)!=4u||b>=dim(cache,0u)||h>=dim(cache,1u)||d>=width){fail(51u);return 0.0;}
 uint capacity=dim(cache,2u),past=complexCachePast(b),start=arg(24u)!=0u&&past>capacity?past-capacity:0u;
 if(absoluteT<start||absoluteT>=past){fail(51u);return 0.0;}
 uint row=absoluteT-start;if(row>=capacity){fail(51u);return 0.0;}
 uint physicalWidth=dim(cache,3u),base=((b*dim(cache,1u)+h)*capacity+row)*physicalWidth;
 if(dtype(cache)==D_F32||dtype(cache)==D_F16){if(d>=physicalWidth){fail(51u);return 0.0;}return readF(cache,base+d);}
 uint bits=arg(21u),code;
 if(bits==4u){if(d/2u>=physicalWidth){fail(51u);return 0.0;}uint byte=readU(cache,base+d/2u);code=(byte>>((d&1u)*4u))&15u;return float(int(code)-8)*complexCacheScale(role,h,d);}
 if(bits!=8u||d>=physicalWidth){fail(53u);return 0.0;}
 code=readU(cache,base+d)&255u;int signedCode=int(code<<24u)>>24;
 return float(signedCode)*complexCacheScale(role,h,d);
}
void complexCacheWrite(uint role,uint b,uint h,uint absoluteT,uint d,float value){
 uint slot=os(role);if(slot==0xffffffffu)return;
 uint width=complexCacheWidth(role);if(outRank(slot)!=4u||b>=outDim(slot,0u)||h>=outDim(slot,1u)||d>=width){fail(51u);return;}
 uint capacity=outDim(slot,2u),total=complexCacheTotal(b);
 uint start=arg(24u)!=0u&&total>capacity?total-capacity:0u;
 // Current chunks may include rows that have already fallen outside the final
 // window, or right padding beyond this batch item's live sequence length.
 if(absoluteT<start||absoluteT>=total)return;
 uint row=absoluteT-start;if(row>=capacity){fail(51u);return;}
 uint physicalWidth=outDim(slot,3u),base=((b*outDim(slot,1u)+h)*capacity+row)*physicalWidth;
 if(outDtype(slot)==D_F32||outDtype(slot)==D_F16){if(d>=physicalWidth){fail(51u);return;}writeF(slot,base+d,value);return;}
 uint bits=arg(21u);if(bits!=4u&&bits!=8u){fail(53u);return;}
 int quantized=complexCacheQuantize(role,h,d,value);
 if(bits==8u){if(d>=physicalWidth){fail(51u);return;}writeU(slot,base+d,uint(quantized)&255u);return;}
 if(d/2u>=physicalWidth){fail(51u);return;}
 uint index=base+d/2u,shift=(d&1u)*4u,mask=15u<<shift,nibble=uint(quantized+8)<<shift;
 // Each byte is one expanded heap word. CAS also permits a future parallel
 // dispatcher to write even and odd channels without losing either nibble.
 uint address=outputAddress(slot,index),old=outputHeap[address];
 for(;;){uint replacement=(old&~mask)|nibble;uint observed=atomicCompSwap(outputHeap[address],old,replacement);if(observed==old)break;old=observed;}
}
void complexCacheCopy(uint role){
 uint slot=os(role);if(slot==0xffffffffu)return;
 uint cache=complexCacheInput(role);if(outRank(slot)!=4u){fail(51u);return;}
 uint batches=outDim(slot,0u),heads=outDim(slot,1u),capacity=outDim(slot,2u),physicalWidth=outDim(slot,3u);
 for(uint b=0u;b<batches;++b){uint past=complexCachePast(b),total=complexCacheTotal(b),oldCapacity=has(cache)?dim(cache,2u):0u;
  uint oldStart=arg(24u)!=0u&&past>oldCapacity?past-oldCapacity:0u;
  uint newStart=arg(24u)!=0u&&total>capacity?total-capacity:0u;
  for(uint h=0u;h<heads;++h)for(uint row=0u;row<capacity;++row){uint absoluteT=newStart+row;
   bool retained=has(cache)&&b<dim(cache,0u)&&h<dim(cache,1u)&&absoluteT>=oldStart&&absoluteT-oldStart<oldCapacity;
   for(uint d=0u;d<physicalWidth;++d){uint target=((b*heads+h)*capacity+row)*physicalWidth+d;
    if(retained&&d<dim(cache,3u)){uint source=((b*dim(cache,1u)+h)*oldCapacity+absoluteT-oldStart)*dim(cache,3u)+d;
     if(dtype(cache)==outDtype(slot))writeRaw(slot,target,readRaw(cache,source));else if((dtype(cache)==D_F32||dtype(cache)==D_F16)&&(outDtype(slot)==D_F32||outDtype(slot)==D_F16))writeF(slot,target,readF(cache,source));else{fail(53u);writeU(slot,target,0u);}
    }else writeU(slot,target,0u);
   }
  }
 }
}
#endif
