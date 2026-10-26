// SPDX-License-Identifier: Apache-2.0
uint windowTaps(){uint n=1u;for(uint d=0u;d<arg(1u);++d)n*=arg(96u+d);return n;}
void tapCoordinates(uint index,out uint taps[16]){for(uint d=0u;d<16u;++d)taps[d]=0u;for(uint d=arg(1u);d>0u;--d){uint n=arg(96u+d-1u);taps[d-1u]=index%n;index/=n;}}
float spatialActivation(float x){uint mode=arg(10u);if(mode==1u)return isnan(x)?x:max(x,0.0);if(mode==2u)return unaryTanh(x);if(mode==3u)return 1.0/(1.0+unaryExp(-x));if(mode==4u)return x>=0.0?x:argF(224u)*x;if(mode==5u)return min(max(x,argF(224u)),argF(225u));if(mode==6u)return clamp(argF(224u)*x+argF(225u),0.0,1.0);if(mode==7u)return x>=3.0?x:x<=-3.0?(isinf(x)?unaryNaN():unarySignedZero(x)):x*((x+3.0)/6.0);return x;}
float convBilinear(uint n,uint c,float y,float x){int y0=int(floor(y)),x0=int(floor(x));float wy=y-float(y0),wx=x-float(x0),result=0.0;for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){int iy=y0+dy,ix=x0+dx;if(iy>=0&&ix>=0&&iy<int(dim(0u,2u))&&ix<int(dim(0u,3u))){uint at=n*stride(0u,0u)+c*stride(0u,1u)+uint(iy)*stride(0u,2u)+uint(ix);result+=readF(0u,at)*(dy==0?1.0-wy:wy)*(dx==0?1.0-wx:wx);}}return result;}
void poolValue(uint outputSlot,uint index){
 uint op=SPATIAL_OPCODE;if(op==S_UNPOOL){float value=0.0;for(uint j=0u;j<size(0u);++j)if(read64(1u,j).y==0u&&readU(1u,j)==index)value=readF(0u,j);writeF(outputSlot,index,value);return;}
 uint coords[16];coordinates(index,outputSlot,true,coords);uint taps=windowTaps(),count=0u,bestIndex=0u;bool maximum=op==S_MAXPOOL||op==S_GLOBALMAX;float result=maximum?(floatingType(dtype(0u))?-unaryInfinity():dtype(0u)==D_I8?-128.0:0.0):0.0;
 for(uint k=0u;k<taps;++k){uint tc[16];tapCoordinates(k,tc);bool inside=true,inPad=true;uint at=coords[0]*stride(0u,0u)+coords[1]*stride(0u,1u),columnMajor=0u,columnStride=1u;
  for(uint d=0u;d<arg(1u);++d){int pos=int(coords[d+2u]*arg(32u+d)+tc[d]*arg(48u+d))-int(arg(64u+d));inside=inside&&pos>=0&&pos<int(dim(0u,d+2u));inPad=inPad&&pos>=-int(arg(64u+d))&&pos<int(dim(0u,d+2u)+arg(80u+d));if(pos>=0){at+=uint(pos)*stride(0u,d+2u);columnMajor+=uint(pos)*columnStride;}columnStride*=dim(0u,d+2u);}
  if(inside){float x=readF(0u,at);++count;if(maximum){if(x>result||isnan(x)){result=x;bestIndex=arg(3u)==0u?at:(coords[0]*dim(0u,1u)+coords[1])*columnStride+columnMajor;}}else if(op==S_LPPOOL||op==S_GLOBALLP)result+=arg(6u)==1u?abs(x):pow(abs(x),float(arg(6u)));else result+=x;}
  else if(arg(2u)!=0u&&inPad)++count;
 }
 if(arg(200u+outputSlot)==4u){writeRaw(outputSlot,index,uvec2(bestIndex,0u));return;}
 if(op==S_AVGPOOL||op==S_GLOBALAVG)result=count==0u?unaryNaN():result/float(count);if(op==S_LPPOOL||op==S_GLOBALLP)result=arg(6u)==1u?result:arg(6u)==2u?sqrt(result):unaryExp(unaryLogBits(floatBitsToUint(result),0)/float(arg(6u)));writeF(outputSlot,index,result);
}
void col2imValue(uint outputSlot,uint index){uint coords[16];coordinates(index,outputSlot,true,coords);uint taps=windowTaps(),columns=dim(0u,2u);vec2 sum=vec2(0.0);
 for(uint k=0u;k<taps;++k){uint tc[16];tapCoordinates(k,tc);bool inside=true;uint col=0u;for(uint d=0u;d<arg(1u);++d){int numerator=int(coords[d+2u]+arg(64u+d))-int(tc[d]*arg(48u+d));int step=int(arg(32u+d));if(numerator<0||numerator%step!=0){inside=false;break;}uint pos=uint(numerator/step);if(pos>=arg(112u+d)){inside=false;break;}col=col*arg(112u+d)+pos;}if(inside){uint channels=outDim(outputSlot,1u);uint at=((coords[0]*channels+coords[1])*taps+k)*columns+col;sum=unaryDFAdd(sum,vec2(readF(0u,at),0.0));}}
 writeF(outputSlot,index,sum.x);
}
void convValue(uint outputSlot,uint index){
 uint op=SPATIAL_OPCODE,coords[16];coordinates(index,outputSlot,true,coords);uint d=arg(1u),channelAxis=arg(7u)!=0u?rank(0u)-1u:1u,co=coords[channelAxis],n=coords[0],ciCount=dim(0u,channelAxis),coCount=outDim(outputSlot,channelAxis),groups=arg(5u),ciPer=ciCount/groups,coPer=coCount/groups,group=co/coPer,taps=windowTaps();
 vec2 total=vec2(0.0);uint integerTotal=0u;
 for(uint c=0u;c<ciPer;++c){uint ci=group*ciPer+c;for(uint k=0u;k<taps;++k){uint tc[16];tapCoordinates(k,tc);uint at=n*stride(0u,0u)+ci*stride(0u,channelAxis);bool inside=true;for(uint j=0u;j<d;++j){uint ax=arg(7u)!=0u?j+1u:j+2u;int pos=int(coords[ax]*arg(32u+j)+tc[j]*arg(48u+j))-int(arg(64u+j));if(op==S_TRANSPOSE){int numerator=int(coords[ax]+arg(64u+j))-int(tc[j]*arg(48u+j));if(numerator<0||numerator%int(arg(32u+j))!=0){inside=false;break;}pos=numerator/int(arg(32u+j));}if(pos<0||pos>=int(dim(0u,ax))){inside=false;break;}at+=uint(pos)*stride(0u,ax);}
  uint wi=op==S_TRANSPOSE?(ci*coPer+(co%coPer))*taps+k:(co*ciPer+c)*taps+k;
  if(op==S_DEFORMCONV){uint og=ci/(ciCount/arg(12u)),oh=coords[2],ow=coords[3],height=outDim(outputSlot,2u),width=outDim(outputSlot,3u),offsetChannel=(og*taps+k)*2u;uint oi=((n*dim(9u,1u)+offsetChannel)*height+oh)*width+ow;float y=float(int(oh*arg(32u)+tc[0]*arg(48u))-int(arg(64u)))+readF(9u,oi);float x=float(int(ow*arg(33u)+tc[1]*arg(49u))-int(arg(65u)))+readF(9u,oi+height*width);float v=convBilinear(n,ci,y,x);if(has(10u))v*=readF(10u,((n*dim(10u,1u)+og*taps+k)*height+oh)*width+ow);total=unaryDFAdd(total,unaryDFMul(vec2(v,0.0),vec2(readF(1u,wi),0.0)));}
  else if(inside){if(op==S_CONVINTEGER||op==S_QCONV){int zx=has(3u)?readI(3u,0u):0,zw=has(4u)?readI(4u,size(4u)==1u?0u:co):0;integerTotal+=uint((readI(0u,at)-zx)*(readI(1u,wi)-zw));}else total=unaryDFAdd(total,unaryDFMul(vec2(readF(0u,at),0.0),vec2(readF(1u,wi),0.0)));}
 }}
 if(op==S_CONVINTEGER){writeU(outputSlot,index,integerTotal);return;}
 float value=total.x;if(op==S_QCONV){if(has(2u))integerTotal+=uint(readI(2u,co));int ex,ew,ey;vec2 sx=spatialFloatParts(readF(5u,0u),ex),sw=spatialFloatParts(readF(6u,size(6u)==1u?0u:co),ew),sy=spatialFloatParts(readF(7u,0u),ey);vec2 scale=unaryDFDiv(unaryDFMul(sx,sw),sy);vec2 quantized=unaryDFMul(spatialIntegerDF(int(integerTotal)),scale);int power=ex+ew-ey;float scaled=unaryScalePowerOfTwo(quantized,power);quantized=vec2(scaled,power>=-120&&power<=120?unaryScalePowerOfTwo(vec2(quantized.y,0.0),power):0.0);if(isinf(scaled))quantized=vec2(scaled>0.0?65536.0:-65536.0,0.0);float rounded=roundEven(quantized.x);if(quantized.x-floor(quantized.x)==0.5&&quantized.y!=0.0)rounded=quantized.y>0.0?ceil(quantized.x):floor(quantized.x);value=rounded+float(readI(8u,0u));value=outDtype(outputSlot)==D_U8?clamp(value,0.0,255.0):clamp(value,-128.0,127.0);}
 else{if(has(2u))value+=readF(2u,co);if(op==S_FUSEDCONV){if(has(11u))value+=readF(11u,broadcastIndex(11u,outputSlot,index));value=spatialActivation(value);}}
 writeF(outputSlot,index,value);
}
void poolConvValue(uint outputSlot,uint index){uint op=SPATIAL_OPCODE;if(op<=S_UNPOOL)poolValue(outputSlot,index);else if(op==S_COL2IM)col2imValue(outputSlot,index);else convValue(outputSlot,index);}
