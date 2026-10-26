// SPDX-License-Identifier: Apache-2.0
// Metadata/control planning only. All numerical tensor evaluation runs on Vulkan.
#include "framework.h"
#include "spatial_ids.h"
#include <set>
namespace xrgpu::operators::extended::detail {
namespace {
struct Record {const char* id;uint32_t opcode;};
constexpr Record records[]={
 {"ai.onnx.ReduceSum",S_REDUCE_SUM},{"ai.onnx.ReduceProd",S_REDUCE_PROD},{"ai.onnx.ReduceMean",S_REDUCE_MEAN},{"ai.onnx.ReduceMin",S_REDUCE_MIN},{"ai.onnx.ReduceMax",S_REDUCE_MAX},{"ai.onnx.ReduceL1",S_REDUCE_L1},{"ai.onnx.ReduceL2",S_REDUCE_L2},{"ai.onnx.ReduceLogSum",S_REDUCE_LOGSUM},{"ai.onnx.ReduceLogSumExp",S_REDUCE_LOGSUMEXP},{"ai.onnx.ReduceSumSquare",S_REDUCE_SQUARE},
 {"ai.onnx.CumSum",S_CUMSUM},{"ai.onnx.CumProd",S_CUMPROD},{"ai.onnx.Softmax",S_SOFTMAX},{"ai.onnx.LogSoftmax",S_LOGSOFTMAX},{"com.microsoft.BiasSoftmax",S_BIASSOFTMAX},{"ai.onnx.ArgMax",S_ARGMAX},{"ai.onnx.ArgMin",S_ARGMIN},{"ai.onnx.Hardmax",S_HARDMAX},
 {"ai.onnx.BatchNormalization",S_BATCHNORM},{"ai.onnx.InstanceNormalization",S_INSTANCENORM},{"ai.onnx.GroupNormalization",S_GROUPNORM},{"com.microsoft.GroupNorm",S_MSGROUPNORM},{"com.microsoft.SkipGroupNorm",S_SKIPGROUPNORM},{"ai.onnx.LayerNormalization",S_LAYERNORM},{"ai.onnx.RMSNormalization",S_RMSNORM},{"ai.onnx.SimplifiedLayerNormalization",S_SIMPLENORM},{"com.microsoft.SkipLayerNormalization",S_SKIPLAYERNORM},{"com.microsoft.SkipSimplifiedLayerNormalization",S_SKIPSIMPLENORM},{"com.microsoft.GatedRMSNorm",S_GATEDRMS},{"ai.onnx.MeanVarianceNormalization",S_MEANVARNORM},{"ai.onnx.LpNormalization",S_LPNORM},{"ai.onnx.LRN",S_LRN},
 {"ai.onnx.AveragePool",S_AVGPOOL},{"ai.onnx.MaxPool",S_MAXPOOL},{"ai.onnx.LpPool",S_LPPOOL},{"ai.onnx.GlobalAveragePool",S_GLOBALAVG},{"ai.onnx.GlobalMaxPool",S_GLOBALMAX},{"ai.onnx.GlobalLpPool",S_GLOBALLP},{"ai.onnx.MaxUnpool",S_UNPOOL},
 {"ai.onnx.Conv",S_CONV},{"ai.onnx.ConvTranspose",S_TRANSPOSE},{"ai.onnx.ConvInteger",S_CONVINTEGER},{"ai.onnx.QLinearConv",S_QCONV},{"com.microsoft.QLinearConv",S_QCONV},{"com.microsoft.FusedConv",S_FUSEDCONV},{"ai.onnx.DeformConv",S_DEFORMCONV},{"ai.onnx.Col2Im",S_COL2IM},
 {"ai.onnx.GridSample",S_GRIDSAMPLE},{"ai.onnx.AffineGrid",S_AFFINEGRID},{"ai.onnx.Resize",S_RESIZE},{"ai.onnx.Upsample",S_UPSAMPLE},{"ai.onnx.RoiAlign",S_ROIALIGN},
 {"ai.onnx.DFT",S_DFT},{"ai.onnx.STFT",S_STFT},{"ai.onnx.BlackmanWindow",S_BLACKMAN},{"ai.onnx.HammingWindow",S_HAMMING},{"ai.onnx.HannWindow",S_HANN}};
void check(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
bool wants(const Request& r,const std::string& name){return std::any_of(r.requested_outputs.begin(),r.requested_outputs.end(),[&](const Tensor& t){return t.name==name;});}
void emit(ExtendedPlan& p,const Request& r,const std::string& name,const std::vector<int64_t>& shape,DType type,uint32_t role=0,bool optional=false){if(optional&&!wants(r,name))return;check(p.outputs.size()<16,"Too many spatial outputs");p.args[200+p.outputs.size()]=role;add_output(p,name,shape,type);}
void setf(ExtendedPlan& p,size_t index,double value){check(std::isfinite(value),"Nonfinite spatial attribute");p.args.at(index)=fword(float(value));}
std::vector<int64_t> vec(const Request&r,const std::string& name,size_t count,int64_t fallback,bool positive=false){auto v=attr_is(r,name,std::vector<int64_t>(count,fallback));check(v.size()==count,"Spatial attribute length mismatch");for(auto x:v)check(x>=(positive?1:0),"Spatial attribute value out of range");return v;}
void setvec(ExtendedPlan&p,size_t base,const std::vector<int64_t>&v){for(size_t i=0;i<v.size();++i)p.args.at(base+i)=uint32_t(v[i]);}
std::vector<int64_t> controls(const Tensor&t){std::vector<int64_t>v;for(size_t i=0;i<numel(t);++i)v.push_back(control_i(t,i));return v;}
DType onnx_type(int64_t type){switch(type){case 1:return DType::Float32;case 10:return DType::Float16;case 2:return DType::UInt8;case 3:return DType::Int8;case 5:return DType::Int16;case 6:return DType::Int32;case 12:return DType::UInt32;default:throw std::invalid_argument("Unsupported window output datatype");}}
uint32_t selection(const std::string&s,std::initializer_list<const char*>values){uint32_t i=0;for(auto v:values){if(s==v)return i;++i;}throw std::invalid_argument("Unknown spatial string attribute: "+s);}
std::vector<int64_t> requested_shape(const Request&r,const std::string&name){for(const auto&t:r.requested_outputs)if(t.name==name)return t.shape;return {};}
void finish(ExtendedPlan&p){uint64_t work=0;for(const auto&t:p.outputs)work+=numel(t);check(work<=UINT32_MAX,"Spatial work count exceeds uint32");p.work_items=uint32_t(std::max<uint64_t>(1,work));p.args[216]=uint32_t(p.outputs.size());}
}
bool plan_spatial(const Request&r,ExtendedPlan&p){
 auto rec=std::find_if(std::begin(records),std::end(records),[&](const Record& x){return r.operation==x.id;});if(rec==std::end(records))return false;
 p.operation=r.operation;p.shader="spatial_"+std::to_string(rec->opcode)+".spv";p.args.assign(256,0);p.args[0]=rec->opcode;uint32_t op=rec->opcode;
 setf(p,4,attr_f(r,"epsilon",op==S_SKIPLAYERNORM||op==S_SKIPSIMPLENORM?1e-12:1e-5));check(attr_f(r,"epsilon",1e-5)>=0,"Negative epsilon");
 p.args[6]=uint32_t(attr_i(r,"p",2));p.args[7]=uint32_t(attr_i(r,"channels_last",op==S_MSGROUPNORM||op==S_SKIPGROUPNORM?1:0));p.args[10]=uint32_t(attr_i(r,"activation",0));
 if(op>=S_REDUCE_SUM&&op<=S_REDUCE_SQUARE){
  bind_inputs(p,r,{"x"},{"x"});const auto&x=p.inputs[0];auto axes=attr_is(r,"axes");bool noop=axes.empty()&&attr_i(r,"noop_with_empty_axes",0)!=0;std::set<int64_t> selected;
  if(axes.empty()&&!noop){for(size_t i=0;i<x.shape.size();++i)selected.insert(int64_t(i));}else for(auto a:axes)check(selected.insert(axis(a,x.shape.size())).second,"Duplicate reduction axis");
  auto shape=x.shape;bool keep=attr_i(r,"keepdims",1)!=0;p.args[2]=keep?1:0;for(auto a:selected)p.args[16+size_t(a)]=1;
  shape.clear();for(size_t i=0;i<x.shape.size();++i)if(!selected.count(i))shape.push_back(x.shape[i]);else if(keep)shape.push_back(1);
  emit(p,r,"y",shape,x.dtype);finish(p);return true;
 }
 if(op==S_CUMSUM||op==S_CUMPROD||op==S_ARGMAX||op==S_ARGMIN||op==S_HARDMAX||op==S_SOFTMAX||op==S_LOGSOFTMAX||op==S_BIASSOFTMAX){
  if(op==S_BIASSOFTMAX)bind_inputs(p,r,{"data","bias"},{"data","bias"});else bind_inputs(p,r,{"x"},{"x"});const auto&x=p.inputs[0];auto ax=axis(attr_i(r,"axis",op==S_ARGMAX||op==S_ARGMIN?0:op==S_BIASSOFTMAX?1:-1),x.shape.size());p.args[8]=uint32_t(ax);
  p.args[2]=uint32_t(attr_i(r,"exclusive",0));p.args[3]=uint32_t(attr_i(r,"reverse",0));p.args[11]=uint32_t(attr_i(r,"select_last_index",0));p.args[12]=uint32_t(attr_i(r,"is_inner_broadcast",0));
  auto shape=x.shape;DType type=x.dtype;if(op==S_ARGMAX||op==S_ARGMIN){type=DType::Int64;check(shape[size_t(ax)]>0,"Arg reduction requires nonempty axis");if(attr_i(r,"keepdims",1))shape[size_t(ax)]=1;else shape.erase(shape.begin()+ax);}
  if(op==S_BIASSOFTMAX){uint64_t block=1;for(size_t d=size_t(ax);d<x.shape.size();++d)block*=uint64_t(x.shape[d]);uint64_t rows=block?numel(x)/block:0,biasCount=numel(p.inputs[1]);check((numel(x)==0&&biasCount==0)||(block>0&&biasCount%block==0&&biasCount/block>0&&rows%(biasCount/block)==0),"BiasSoftmax row-block broadcast mismatch");p.args[9]=uint32_t(block);}emit(p,r,op==S_BIASSOFTMAX?"output":"y",shape,type);finish(p);return true;
 }
 if(op>=S_BATCHNORM&&op<=S_LRN){
  if(op==S_BATCHNORM)bind_inputs(p,r,{"x","scale","b","inputMean","inputVar"},{"x","scale","b","inputMean","inputVar"});
  else if(op==S_INSTANCENORM)bind_inputs(p,r,{"input","scale","b"},{"input","scale","b"});
  else if(op==S_MSGROUPNORM)bind_inputs(p,r,{"X","gamma","beta"},{"X","gamma","beta"});
  else if(op==S_SKIPGROUPNORM)bind_inputs(p,r,{"X","gamma","beta","skip","bias"},{"X","gamma","beta","skip"});
  else if(op==S_SKIPLAYERNORM)bind_inputs(p,r,{"inputT","gammaT","betaT","skipT","biasT"},{"inputT","skipT","gammaT"});
  else if(op==S_SKIPSIMPLENORM)bind_inputs(p,r,{"inputT","gammaT","unused","skipT","biasT"},{"inputT","skipT","gammaT"});
  else if(op==S_GATEDRMS)bind_inputs(p,r,{"xT","scaleT","unused","gateT"},{"xT","scaleT","gateT"});
  else if(op==S_GROUPNORM)bind_inputs(p,r,{"x","scale","bias"},{"x","scale","bias"});
  else if(op==S_LAYERNORM)bind_inputs(p,r,{"x","scale","b"},{"x","scale"});
  else if(op==S_RMSNORM||op==S_SIMPLENORM)bind_inputs(p,r,{"x","scale"},{"x","scale"});
  else if(op==S_LPNORM)bind_inputs(p,r,{"input"},{"input"});else bind_inputs(p,r,{"x"},{"x"});
  const auto&x=p.inputs[0];check(x.dtype==DType::Float32||x.dtype==DType::Float16,"Normalization needs float tensors");check(!x.shape.empty(),"Normalization needs tensor rank");auto shape=x.shape;
  int64_t ax=axis(attr_i(r,"axis",-1),shape.size());p.args[8]=uint32_t(ax);uint64_t count=1;for(size_t i=size_t(ax);i<shape.size();++i)count*=uint64_t(shape[i]);p.args[9]=uint32_t(count);p.args[13]=uint32_t(attr_i(r,"stash_type",1));check(p.args[13]==1||p.args[13]==10,"Normalization stash_type must be float32 or float16");
  if(op==S_GATEDRMS){uint64_t normSize=numel(p.inputs[1]);check(!p.inputs[1].shape.empty()&&normSize>0&&uint64_t(shape.back())%normSize==0,"GatedRMSNorm scale must divide last dimension");check(p.inputs[3].shape==shape,"GatedRMSNorm gate shape mismatch");p.args[9]=uint32_t(normSize);}
  if(op==S_BATCHNORM){check(shape.size()>=2,"BatchNormalization rank");check(attr_i(r,"training_mode",0)==0,"Training BatchNormalization outside contract");for(size_t i=1;i<5;++i)check(numel(p.inputs[i])==uint64_t(shape[1]),"BatchNormalization parameter size");}
  if(op==S_INSTANCENORM){check(shape.size()>=3,"InstanceNormalization rank");check(numel(p.inputs[1])==uint64_t(shape[1])&&numel(p.inputs[2])==uint64_t(shape[1]),"InstanceNormalization parameter size");}
  if(op==S_GROUPNORM||op==S_MSGROUPNORM||op==S_SKIPGROUPNORM){check(shape.size()>=3,"GroupNormalization rank");auto channels=shape[p.args[7]?shape.size()-1:1];auto groups=attr_i(r,op==S_GROUPNORM?"num_groups":"groups",0);check(groups>0&&channels%groups==0,"Invalid groups");p.args[5]=uint32_t(groups);uint64_t needed=uint64_t(channels);check(numel(p.inputs[1])==needed&&numel(p.inputs[2])==needed,"GroupNormalization scale/bias size");}
  if(op==S_MEANVARNORM){auto axes=attr_is(r,"axes",{0,2,3});std::set<int64_t>seen;for(auto a:axes){a=axis(a,shape.size());check(seen.insert(a).second,"Duplicate normalization axes");p.args[16+size_t(a)]=1;}}
  if(op==S_LRN){check(shape.size()==4,"LRN rank");auto size=attr_i(r,"size",0);check(size>0,"LRN size required");p.args[5]=uint32_t(size);setf(p,32,attr_f(r,"alpha",1e-4));setf(p,33,attr_f(r,"beta",0.75));setf(p,34,attr_f(r,"bias",1));}
  if(op==S_LPNORM)check(p.args[6]==1||p.args[6]==2,"LpNormalization p");
  bool skip=op==S_SKIPLAYERNORM||op==S_SKIPSIMPLENORM;bool simple=op==S_SIMPLENORM||op==S_RMSNORM||op==S_GATEDRMS||op==S_SKIPSIMPLENORM;
  if(skip)check(broadcast(shape,p.inputs[3].shape)==shape,"Skip normalization broadcast shape mismatch");if(op==S_SKIPGROUPNORM){auto c=shape[p.args[7]?shape.size()-1:1];bool matrix=p.inputs[3].shape==std::vector<int64_t>{shape[0],c};check(matrix||broadcast(shape,p.inputs[3].shape)==shape,"SkipGroupNorm broadcast shape mismatch");}
  if(op!=S_GATEDRMS&&(op==S_LAYERNORM||simple||skip)){check(broadcast(shape,p.inputs[1].shape)==shape,"Normalization scale shape mismatch");if(p.inputs.size()>2&&p.inputs[2].dtype!=DType::Unspecified)check(broadcast(shape,p.inputs[2].shape)==shape,"Normalization bias shape mismatch");}
  std::string outputName=op==S_INSTANCENORM||op==S_LPNORM?"output":op==S_MSGROUPNORM||op==S_SKIPGROUPNORM?"Y":skip?"outputT":op==S_GATEDRMS?"yT":"y";
  DType ytype=(op==S_SIMPLENORM||op==S_RMSNORM)?p.inputs[1].dtype:x.dtype;emit(p,r,outputName,shape,ytype);
  auto statistic=shape;for(size_t i=size_t(ax);i<statistic.size();++i)statistic[i]=1;
  if(op==S_LAYERNORM){emit(p,r,"mean",statistic,DType::Float32,1,true);emit(p,r,"invStdDev",statistic,DType::Float32,2,true);}
  if(op==S_SIMPLENORM)emit(p,r,"invStdVar",statistic,DType::Float32,2,true);
  if(skip){emit(p,r,"meanT",statistic,DType::Float32,1,true);emit(p,r,"invStdT",statistic,DType::Float32,2,true);emit(p,r,"residualT",shape,x.dtype,3,true);}
  if(op==S_SKIPGROUPNORM)emit(p,r,"S",shape,x.dtype,3,true);
  finish(p);return true;
 }
 if(op>=S_AVGPOOL&&op<=S_UNPOOL){
  if(op==S_UNPOOL)bind_inputs(p,r,{"x","indices","output_shape"},{"x","indices"});else bind_inputs(p,r,{"x"},{"x"});const auto&x=p.inputs[0];check(x.shape.size()>=3,"Pooling rank must include N,C,spatial");size_t d=x.shape.size()-2;p.args[1]=uint32_t(d);auto shape=x.shape;
  bool global=op==S_GLOBALAVG||op==S_GLOBALMAX||op==S_GLOBALLP;std::vector<int64_t> kernel;
  if(global)kernel.assign(x.shape.begin()+2,x.shape.end());else kernel=attr_is(r,"kernel_shape");check(kernel.size()==d,"Pooling kernel_shape required");for(auto k:kernel)check(k>0||global&&k==0,"Pooling kernel positive");
  auto strides=vec(r,"strides",d,1,true),dilations=vec(r,"dilations",d,1,true),pads=vec(r,"pads",2*d,0);std::string autopad=attr_s(r,"auto_pad","NOTSET");selection(autopad,{"NOTSET","SAME_UPPER","SAME_LOWER","VALID"});p.args[2]=uint32_t(attr_i(r,"count_include_pad",0));p.args[3]=uint32_t(attr_i(r,"storage_order",0));
  for(size_t j=0;j<d;++j){int64_t in=x.shape[j+2],eff=(kernel[j]-1)*dilations[j]+1;
   if(global){shape[j+2]=1;pads[j]=pads[j+d]=0;}
   else if(op==S_UNPOOL)shape[j+2]=(in-1)*strides[j]+kernel[j]-pads[j]-pads[j+d];
   else if(autopad=="SAME_UPPER"||autopad=="SAME_LOWER"){shape[j+2]=(in+strides[j]-1)/strides[j];int64_t total=std::max<int64_t>(0,(shape[j+2]-1)*strides[j]+eff-in);pads[j]=autopad=="SAME_UPPER"?total/2:(total+1)/2;pads[j+d]=total-pads[j];}
   else {if(autopad=="VALID")pads[j]=pads[j+d]=0;double value=double(in+pads[j]+pads[j+d]-eff)/double(strides[j]);shape[j+2]=int64_t(attr_i(r,"ceil_mode",0)?std::ceil(value):std::floor(value))+1;if(attr_i(r,"ceil_mode",0)&&shape[j+2]>0&&(shape[j+2]-1)*strides[j]>=in+pads[j])--shape[j+2];shape[j+2]=std::max<int64_t>(0,shape[j+2]);}
  }
  if(op==S_UNPOOL&&p.inputs[2].dtype!=DType::Unspecified){shape=controls(p.inputs[2]);check(shape.size()==x.shape.size()&&shape[0]==x.shape[0]&&shape[1]==x.shape[1],"MaxUnpool output_shape mismatch");}
  setvec(p,32,strides);setvec(p,48,dilations);setvec(p,64,std::vector<int64_t>(pads.begin(),pads.begin()+d));setvec(p,80,std::vector<int64_t>(pads.begin()+d,pads.end()));setvec(p,96,kernel);
  if(op==S_UNPOOL){check(p.inputs[1].shape==x.shape,"MaxUnpool indices shape");}
  emit(p,r,op==S_UNPOOL?"output":"y",shape,x.dtype);if(op==S_MAXPOOL)emit(p,r,"indices",shape,DType::Int64,4,true);finish(p);return true;
 }
 if(op>=S_CONV&&op<=S_COL2IM){
  if(op==S_COL2IM){bind_inputs(p,r,{"input","image_shape","block_shape"},{"input","image_shape","block_shape"});const auto&x=p.inputs[0];check(x.shape.size()==3,"Col2Im input rank");auto image=controls(p.inputs[1]),block=controls(p.inputs[2]);check(!image.empty()&&image.size()<=14&&image.size()==block.size(),"Col2Im spatial controls");size_t d=image.size();p.args[1]=uint32_t(d);uint64_t taps=1,columns=1;auto strides=vec(r,"strides",d,1,true),dilations=vec(r,"dilations",d,1,true),pads=vec(r,"pads",2*d,0);std::vector<int64_t> grid;
   for(size_t j=0;j<d;++j){check(image[j]>0&&block[j]>0,"Col2Im controls positive");taps*=uint64_t(block[j]);auto length=(image[j]+pads[j]+pads[j+d]-(block[j]-1)*dilations[j]-1)/strides[j]+1;check(length>=0,"Col2Im invalid image/block");grid.push_back(length);columns*=uint64_t(length);}
   check(taps>0&&uint64_t(x.shape[1])%taps==0&&uint64_t(x.shape[2])==columns,"Col2Im folded input shape mismatch");std::vector<int64_t>shape{x.shape[0],int64_t(uint64_t(x.shape[1])/taps)};shape.insert(shape.end(),image.begin(),image.end());setvec(p,32,strides);setvec(p,48,dilations);setvec(p,64,std::vector<int64_t>(pads.begin(),pads.begin()+d));setvec(p,96,block);setvec(p,112,grid);emit(p,r,"output",shape,x.dtype);finish(p);return true;
  }
  if(op==S_QCONV)bind_inputs(p,r,{"x","w","B","x_zero_point","w_zero_point","x_scale","w_scale","y_scale","y_zero_point","unused","unused","unused"},{"x","w","x_scale","w_scale","y_scale","y_zero_point"});
  else if(op==S_CONVINTEGER)bind_inputs(p,r,{"x","w","unused","x_zero_point","w_zero_point"},{"x","w"});
  else if(op==S_DEFORMCONV)bind_inputs(p,r,{"x","w","bias","unused","unused","unused","unused","unused","unused","offset","mask"},{"x","w","offset"});
  else bind_inputs(p,r,{"x","w","bias","unused","unused","unused","unused","unused","unused","unused","unused","zResidual"},{"x","w"});
  const auto&x=p.inputs[0];const auto&w=p.inputs[1];check(x.shape.size()>=3&&x.shape.size()==w.shape.size(),"Convolution rank mismatch");size_t d=x.shape.size()-2;p.args[1]=uint32_t(d);bool last=op==S_QCONV&&r.operation.rfind("com.microsoft.",0)==0&&attr_i(r,"channels_last",0)!=0;p.args[7]=last?1:0;
  int64_t groups=attr_i(r,"group",1),ci=x.shape[last?x.shape.size()-1:1],co=op==S_TRANSPOSE?w.shape[1]*groups:w.shape[0];check(groups>0&&ci%groups==0&&co%groups==0,"Invalid convolution groups");check(op==S_TRANSPOSE?w.shape[0]==ci:w.shape[1]*groups==ci,"Convolution channel mismatch");p.args[5]=uint32_t(groups);
  std::vector<int64_t> kernel(w.shape.begin()+2,w.shape.end());auto supplied=attr_is(r,"kernel_shape",kernel);check(supplied==kernel,"Convolution kernel_shape mismatch");auto strides=vec(r,"strides",d,1,true),dilations=vec(r,"dilations",d,1,true),pads=vec(r,"pads",2*d,0),outpad=vec(r,"output_padding",d,0);auto explicitshape=attr_is(r,"output_shape");if(!explicitshape.empty())check(explicitshape.size()==d,"ConvTranspose output_shape length");std::string autopad=attr_s(r,"auto_pad","NOTSET");selection(autopad,{"NOTSET","SAME_UPPER","SAME_LOWER","VALID"});auto shape=x.shape;shape[last?shape.size()-1:1]=co;
  for(size_t j=0;j<d;++j){size_t axisIndex=last?j+1:j+2;int64_t in=x.shape[axisIndex],eff=(kernel[j]-1)*dilations[j]+1;check(kernel[j]>0,"Convolution kernel positive");
   if(op==S_TRANSPOSE){check(outpad[j]<std::max(strides[j],dilations[j]),"ConvTranspose output_padding invalid");if(!explicitshape.empty())shape[axisIndex]=explicitshape[j];else if(autopad=="SAME_UPPER"||autopad=="SAME_LOWER")shape[axisIndex]=in*strides[j];else shape[axisIndex]=(in-1)*strides[j]+eff+outpad[j]-pads[j]-pads[j+d];
    if(!explicitshape.empty()||autopad=="SAME_UPPER"||autopad=="SAME_LOWER"){int64_t total=(in-1)*strides[j]+eff+outpad[j]-shape[axisIndex];check(total>=0,"ConvTranspose output_shape cannot require negative pads");pads[j]=autopad=="SAME_LOWER"?(total+1)/2:total/2;pads[j+d]=total-pads[j];}}
   else if(autopad=="SAME_UPPER"||autopad=="SAME_LOWER"){shape[axisIndex]=(in+strides[j]-1)/strides[j];int64_t total=std::max<int64_t>(0,(shape[axisIndex]-1)*strides[j]+eff-in);pads[j]=autopad=="SAME_UPPER"?total/2:(total+1)/2;pads[j+d]=total-pads[j];}
   else{if(autopad=="VALID")pads[j]=pads[j+d]=0;shape[axisIndex]=std::max<int64_t>(0,int64_t(std::floor(double(in+pads[j]+pads[j+d]-eff)/double(strides[j])))+1);}
  }
  if(p.inputs[2].dtype!=DType::Unspecified)check(numel(p.inputs[2])==uint64_t(co),"Convolution bias length");
  if(op==S_QCONV){check(numel(p.inputs[5])==1&&numel(p.inputs[7])==1&&numel(p.inputs[8])==1,"QLinearConv scalar scales/zero points");check(numel(p.inputs[6])==1||numel(p.inputs[6])==uint64_t(co),"QLinearConv per-channel scales");for(size_t slot=5;slot<=7;++slot)for(size_t j=0;j<numel(p.inputs[slot]);++j){double value=control_f(p.inputs[slot],j);check(std::isfinite(value)&&value>0,"QLinearConv scales must be finite and positive");}}
  if(op==S_FUSEDCONV){p.args[10]=selection(attr_s(r,"activation",""),{"","Relu","Tanh","Sigmoid","LeakyRelu","Clip","HardSigmoid","HardSwish"});auto a=attr_fs(r,"activation_params");setf(p,224,a.empty()?0.01:a[0]);setf(p,225,a.size()>1?a[1]:0);if(p.inputs[11].dtype!=DType::Unspecified)check(broadcast(shape,p.inputs[11].shape)==shape,"FusedConv residual shape");}
  if(op==S_DEFORMCONV){check(d==2&&x.dtype==DType::Float32,"DeformConv needs rank4 float32");auto og=attr_i(r,"offset_group",1);check(og>0&&ci%og==0,"DeformConv offset groups");p.args[12]=uint32_t(og);auto expected=shape;expected[1]=og*kernel[0]*kernel[1]*2;check(p.inputs[9].shape==expected,"DeformConv offset shape");if(p.inputs[10].dtype!=DType::Unspecified){expected[1]/=2;check(p.inputs[10].shape==expected,"DeformConv mask shape");}}
  setvec(p,32,strides);setvec(p,48,dilations);setvec(p,64,std::vector<int64_t>(pads.begin(),pads.begin()+d));setvec(p,80,std::vector<int64_t>(pads.begin()+d,pads.end()));setvec(p,96,kernel);setvec(p,112,outpad);DType type=op==S_CONVINTEGER?DType::Int32:op==S_QCONV?p.inputs[8].dtype:x.dtype;emit(p,r,"y",shape,type);finish(p);return true;
 }
 if(op>=S_GRIDSAMPLE&&op<=S_UPSAMPLE){
  if(op==S_GRIDSAMPLE){bind_inputs(p,r,{"x","grid"},{"x","grid"});const auto&x=p.inputs[0];const auto&g=p.inputs[1];size_t d=x.shape.size()-2;check((d==2||d==3)&&g.shape.size()==x.shape.size()&&g.shape[0]==x.shape[0]&&g.shape.back()==int64_t(d),"GridSample input/grid shapes");auto shape=x.shape;for(size_t j=0;j<d;++j)shape[j+2]=g.shape[j+1];p.args[1]=uint32_t(d);p.args[2]=selection(attr_s(r,"mode","linear"),{"linear","nearest","cubic"});p.args[3]=selection(attr_s(r,"padding_mode","zeros"),{"zeros","border","reflection"});p.args[5]=uint32_t(attr_i(r,"align_corners",0));emit(p,r,"y",shape,x.dtype);}
  else if(op==S_AFFINEGRID){bind_inputs(p,r,{"theta"},{"theta"});const auto&t=p.inputs[0];check(t.shape.size()==3&&(t.shape[1]==2||t.shape[1]==3)&&t.shape[2]==t.shape[1]+1,"AffineGrid theta shape");size_t d=size_t(t.shape[1]);auto shape=requested_shape(r,"grid");auto sizes=attr_is(r,"size");if(shape.empty()&&!sizes.empty()){check(sizes.size()==d+2&&sizes[0]==t.shape[0],"AffineGrid size");shape={sizes[0]};shape.insert(shape.end(),sizes.begin()+2,sizes.end());shape.push_back(int64_t(d));}check(shape.size()==d+2&&shape[0]==t.shape[0]&&shape.back()==int64_t(d),"AffineGrid needs output grid shape or size control");p.args[1]=uint32_t(d);p.args[5]=uint32_t(attr_i(r,"align_corners",0));emit(p,r,"grid",shape,t.dtype);}
  else if(op==S_ROIALIGN){bind_inputs(p,r,{"x","rois","batch_indices"},{"x","rois","batch_indices"});const auto&x=p.inputs[0];const auto&roi=p.inputs[1];check(x.shape.size()==4&&roi.shape.size()==2&&roi.shape[1]==4&&numel(p.inputs[2])==uint64_t(roi.shape[0]),"RoiAlign shapes");int64_t height=attr_i(r,"output_height",1),width=attr_i(r,"output_width",1);check(height>0&&width>0&&attr_i(r,"sampling_ratio",0)>=0,"RoiAlign sampling/output size");for(size_t j=0;j<numel(p.inputs[2]);++j){auto n=control_i(p.inputs[2],j);check(n>=0&&n<x.shape[0],"RoiAlign batch index");}p.args[2]=selection(attr_s(r,"mode","avg"),{"avg","max"});p.args[3]=selection(attr_s(r,"coordinate_transformation_mode","half_pixel"),{"half_pixel","output_half_pixel"});p.args[5]=uint32_t(attr_i(r,"sampling_ratio",0));setf(p,32,attr_f(r,"spatial_scale",1));emit(p,r,"y",{roi.shape[0],x.shape[1],height,width},x.dtype);}
  else{if(op==S_UPSAMPLE)bind_inputs(p,r,{"x","scales"},{"x","scales"});else bind_inputs(p,r,{"x"},{"x"});const auto&x=p.inputs[0];size_t d=x.shape.size();check(d>0,"Resize scalar input not supported by contract");p.args[1]=uint32_t(d);auto axes=attr_is(r,"axes");if(axes.empty())for(size_t j=0;j<d;++j)axes.push_back(int64_t(j));std::set<int64_t> seen;for(auto&a:axes){a=axis(a,d);check(seen.insert(a).second,"Resize duplicate axes");}
   std::vector<double> scales;if(op==S_UPSAMPLE){for(size_t j=0;j<numel(p.inputs[1]);++j)scales.push_back(control_f(p.inputs[1],j));}else scales=attr_fs(r,"scales");auto shape=requested_shape(r,"y");auto sizes=attr_is(r,"sizes");if(!sizes.empty()){check(sizes.size()==axes.size(),"Resize sizes length");shape=x.shape;for(size_t j=0;j<axes.size();++j)shape[size_t(axes[j])]=sizes[j];}
   if(!scales.empty()){check(scales.size()==axes.size(),"Resize scales length");auto inferred=x.shape;for(size_t j=0;j<axes.size();++j){check(std::isfinite(scales[j])&&scales[j]>0,"Resize scales positive");inferred[size_t(axes[j])]=int64_t(std::floor(double(x.shape[size_t(axes[j])])*scales[j]));}if(!shape.empty())check(shape==inferred,"Resize requested output differs from scales");shape=inferred;}
   if(shape.empty())shape=x.shape;check(shape.size()==d,"Resize output rank");for(size_t j=0;j<d;++j){check(shape[j]>=0,"Resize negative size");if(!seen.count(int64_t(j)))check(shape[j]==x.shape[j],"Resize changes an axis outside axes");}
   p.args[2]=selection(attr_s(r,"mode","nearest"),{"nearest","linear","cubic"});p.args[3]=selection(attr_s(r,"coordinate_transformation_mode",op==S_UPSAMPLE?"asymmetric":"half_pixel"),{"half_pixel","half_pixel_symmetric","pytorch_half_pixel","align_corners","asymmetric","tf_crop_and_resize"});p.args[5]=selection(attr_s(r,"nearest_mode",op==S_UPSAMPLE?"floor":"round_prefer_floor"),{"round_prefer_floor","round_prefer_ceil","floor","ceil"});p.args[14]=scales.empty()?0u:1u;p.args[11]=uint32_t(attr_i(r,"antialias",0));p.args[12]=uint32_t(attr_i(r,"exclude_outside",0));setf(p,224,attr_f(r,"cubic_coeff_a",-0.75));setf(p,225,attr_f(r,"extrapolation_value",0));check(attr_s(r,"keep_aspect_ratio_policy","stretch")=="stretch","Resize aspect ratio policy outside frozen contract");auto roi=attr_fs(r,"roi");check(roi.empty()||roi.size()==2*axes.size(),"Resize ROI length");
   for(size_t j=0;j<d;++j){p.args[16+j]=seen.count(int64_t(j))?1:0;double scale=x.shape[j]?double(shape[j])/double(x.shape[j]):1;auto pos=std::find(axes.begin(),axes.end(),int64_t(j));if(pos!=axes.end()&&!scales.empty())scale=scales[size_t(pos-axes.begin())];setf(p,128+j,scale);setf(p,176+j,scale-double(float(scale)));setf(p,144+j,0);setf(p,160+j,1);}if(!roi.empty())for(size_t j=0;j<axes.size();++j){setf(p,144+size_t(axes[j]),roi[j]);setf(p,160+size_t(axes[j]),roi[j+axes.size()]);}
   emit(p,r,"y",shape,x.dtype);
  }
  finish(p);return true;
 }
 if(op>=S_DFT&&op<=S_HANN){
  if(op>=S_BLACKMAN){bind_inputs(p,r,{"size"},{"size"});auto count=control_i(p.inputs[0]);check(count>=0&&uint64_t(count)<=UINT32_MAX,"Window size invalid");p.args[2]=uint32_t(attr_i(r,"periodic",1));emit(p,r,"y",{count},onnx_type(attr_i(r,"output_datatype",1)));}
  else if(op==S_DFT){bind_inputs(p,r,{"input","dft_length","axis"},{"input"});const auto&x=p.inputs[0];check(x.shape.size()>=2&&(x.shape.back()==1||x.shape.back()==2),"DFT input needs real/complex trailing dimension");int64_t ax=int64_t(x.shape.size())-2;if(p.inputs[2].dtype!=DType::Unspecified)ax=control_i(p.inputs[2]);ax=axis(ax,x.shape.size());check(size_t(ax)+1<x.shape.size(),"DFT axis cannot be complex component");int64_t length=x.shape[size_t(ax)];if(p.inputs[1].dtype!=DType::Unspecified)length=control_i(p.inputs[1]);check(length>=0&&uint64_t(length)<=UINT32_MAX,"DFT length nonnegative");bool inverse=attr_i(r,"inverse",0)!=0,onesided=attr_i(r,"onesided",0)!=0;check(!onesided||inverse||x.shape.back()==1,"One-sided forward DFT needs real input");if(inverse&&onesided&&p.inputs[1].dtype==DType::Unspecified)length=std::max<int64_t>(0,2*(x.shape[size_t(ax)]-1));auto shape=x.shape;shape[size_t(ax)]=!inverse&&onesided?length/2+1:length;shape.back()=inverse&&onesided?1:2;p.args[8]=uint32_t(ax);p.args[5]=uint32_t(length);p.args[2]=inverse?1:0;p.args[3]=onesided?1:0;emit(p,r,"output",shape,x.dtype);}
  else{bind_inputs(p,r,{"signal","window"},{"signal"});const auto&x=p.inputs[0];check(x.shape.size()==3&&(x.shape[2]==1||x.shape[2]==2),"STFT input shape");int64_t step=attr_i(r,"frameStep",attr_i(r,"frame_step",0));int64_t length=attr_i(r,"frameLength",attr_i(r,"frame_length",0));if(p.inputs[1].dtype!=DType::Unspecified){check(p.inputs[1].shape.size()==1,"STFT window rank");if(length)check(length==int64_t(numel(p.inputs[1])),"STFT frame/window length mismatch");length=int64_t(numel(p.inputs[1]));}check(step>0&&length>0&&length<=x.shape[1],"STFT frame controls");bool onesided=attr_i(r,"onesided",1)!=0;check(!onesided||x.shape[2]==1,"One-sided STFT needs real signal");p.args[5]=uint32_t(length);p.args[8]=uint32_t(step);p.args[3]=onesided?1:0;emit(p,r,"output",{x.shape[0],(x.shape[1]-length)/step+1,onesided?length/2+1:length,2},x.dtype);}
  finish(p);return true;
 }
 throw std::logic_error("Spatial registry has no paired planner");
}
} // namespace xrgpu::operators::extended::detail
