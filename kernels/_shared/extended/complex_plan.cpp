// SPDX-License-Identifier: Apache-2.0
#include "framework.h"
#include <sstream>
#include <set>
namespace xrgpu::operators::extended::detail {
namespace {
struct Record {const char* name;uint32_t code;const char* inputs;const char* outputs;};
const Record records[]={
 {"ai.onnx.Attention",1000,"q,k,v,attn_mask?,past_key?,past_value?","y,present_key?,present_value?"},
 {"com.microsoft.MultiHeadAttention",1001,"queryT,keyT,valueT,biasT?,attentionBiasT?","outputT"},
 {"com.microsoft.GroupQueryAttention",1002,"queryT,keyT,valueT,pastKeyT?,pastValueT?,seqlensKT,totalSequenceLengthT,cosCacheT?,sinCacheT?,attentionBiasT?,headSinkT?,kScaleT?,vScaleT?,qNormWeightT?,kNormWeightT?","outputT,presentKeyT,presentValueT"},
 {"com.microsoft.DecoderMaskedMultiHeadAttention",1003,"queryT,keyT,valueT,maskIndexT?,attentionBiasT?,pastKeyT?,pastValueT?,pastSequenceLengthT?","outputT,presentKeyT?,presentValueT?,qkT?"},
 {"com.microsoft.SparseAttention",1004,"queryT,keyT?,valueT?,pastKeyT,pastValueT,blockRowIndicesT,blockColIndicesT,totalSequenceLengthT,keyTotalSequenceLengthsT,cosCacheT?,sinCacheT?","outputT,pastKeyT,pastValueT"},
 {"com.microsoft.PackedAttention",1005,"inputT,weightsT,biasT,tokenOffsetT,cumulativeSequenceLengthT,attentionBiasT?","outputT"},
 {"com.microsoft.PackedMultiHeadAttention",1006,"queryT,keyT?,valueT?,biasT?,tokenOffsetT,cumulativeSequenceLengthT,attentionBiasT?","outputT"},
 {"com.microsoft.PagedAttention",1007,"queryT,keyT?,valueT?,keyCacheT,valueCacheT,cumulativeSequenceLengthT,pastSeqlensT,blockTableT,slotMappingT?","outputT,keyCacheT?,valueCacheT?"},
 {"com.microsoft.QAttention",1008,"input,weight,bias,input_scale,weight_scale,mask_index?,input_zero_point?,weight_zero_point?,past?","output,present?"},
 {"ai.onnx.RotaryEmbedding",1009,"x,cos,sin,positionIds?","y"},
 {"com.microsoft.RotaryEmbedding",1010,"x,positionIds,cos,sin","y"},
 {"com.microsoft.MRotaryEmbedding",1011,"x,positionIds,cos,sin","y"},
 {"com.microsoft.GemmaRotaryEmbedding",1012,"embT,qT,qRotT,kT,kRotT","output1T,output2T"},
 {"ai.onnx.LinearAttention",1013,"queryT,keyT,valueT,pastStateT?,decayT?,betaT?","outputT,presentStateT"},
 {"com.microsoft.LinearAttention",1014,"queryT,keyT,valueT,pastStateT?,decayT?,betaT?","outputT,presentStateT"},
 {"com.microsoft.LinearAttentionGate",1015,"aT,dtBiasT,decayScaleT,bT?","decayT,betaT?"},
 {"com.microsoft.GatedDeltaNet",1016,"queryT,keyT,valueT,cuSeqlensT?,decayT?,betaT?,initialStateT?,aLogT?,dtBiasT?","outputT,finalStateT"},
 {"ai.onnx.CausalConvWithState",1017,"inputT,weightT,biasT?,pastStateT?","outputT,presentStateT"},
 {"com.microsoft.CausalConvWithState",1018,"inputT,weightT,biasT?,pastStateT?","outputT,presentStateT"},
 {"com.microsoft.VarlenCausalConvWithState",1019,"inputT,weightT,cumulativeSequenceLengthT,biasT?,initialStateT,captureCountT?","outputT,finalStateT,stateUpdateT?"},
 {"ai.onnx.RNN",1020,"x,w,r,b,sequence_lens?,initial_h?","y,y_h"},
 {"ai.onnx.GRU",1021,"x,w,r,b,sequence_lens?,initial_h?","y,y_h"},
 {"ai.onnx.LSTM",1022,"x,w,r,b,sequence_lens?,initial_h?,initial_c?,p?","y,y_h,y_c"},
 {"ai.onnx.MatMulInteger",1023,"a,b,a_zero_point?,b_zero_point?","y"},
 {"ai.onnx.QLinearMatMul",1024,"a,a_scale,a_zero_point,b,b_scale,b_zero_point,y_scale,y_zero_point","y"},
 {"com.microsoft.MatMulNBits",1025,"aT,bT,scalesT,zeroPointsT?,biasT?","yT"},
 {"com.microsoft.MatMulBnb4",1026,"aT,bT,absmaxT","yT"},
 {"com.microsoft.MatMulNBitsMlp",1027,"aT,skipT?,normScaleT?,gateBT,gateScalesT,gateBiasT?,upBT,upScalesT,upBiasT?","yT,residualT?"},
 {"com.microsoft.MatMulNBitsQkv",1028,"aT,skipT?,normScaleT,qBT,qScalesT,kBT,kScalesT,vBT,vScalesT","qT,kT,vT,residualT?"},
 {"com.microsoft.MoE",1029,"inputT,routerT,fc1T,fc1BiasT?,fc2T,fc2BiasT?,fc3T?,fc3BiasT?","outputT"},
 {"com.microsoft.QMoE",1030,"inputT,routerT,fc1T,fc1ScalesT,fc2T,fc2ScalesT","outputT"},
 {"com.microsoft.GatherBlockQuantized",1031,"dataT,indicesT,scalesT,zeroPointsT?","outputT"},
 {"com.microsoft.EmbedLayerNormalization",1032,"inputIdsT,segmentIdsT?,wordEmbeddingT,positionEmbeddingT,segmentEmbeddingT?,gammaT,betaT,maskT?,positionIdsT?","outputT,maskIndexT?,embeddingSumT?"},
 {"com.microsoft.EngramGate",1033,"keyT,queryT,valueT,keyNormScaleT,queryNormScaleT","outputT"},
 {"com.microsoft.GatedRelativePositionBias",1034,"queryLayerT,queryBiasT,relPosT,weightT,biasT,ecoAT","outputT"},
 {"com.microsoft.NGramHashMapping",1035,"inputIdsT,multipliersT,vocabSizesT,pastIdsT?","hashIdsT,presentIdsT"},
};
std::vector<std::string> fields(const std::string& value){std::vector<std::string> out;std::istringstream s(value);std::string v;while(std::getline(s,v,','))out.push_back(v);return out;}
uint32_t D(const ExtendedPlan& p,size_t slot,size_t axis,uint32_t fallback=1){if(slot>=p.inputs.size()||p.inputs[slot].dtype==DType::Unspecified)return fallback;if(axis>=p.inputs[slot].shape.size())throw std::invalid_argument("Complex operator tensor rank mismatch");return uint32_t(p.inputs[slot].shape[axis]);}
bool H(const ExtendedPlan& p,size_t slot){return slot<p.inputs.size()&&p.inputs[slot].dtype!=DType::Unspecified;}
size_t R(const ExtendedPlan& p,size_t slot){return H(p,slot)?p.inputs[slot].shape.size():0;}
uint32_t last(const ExtendedPlan& p,size_t slot){return R(p,slot)?D(p,slot,R(p,slot)-1):1;}
uint32_t positive(int64_t value,const char* name){if(value<=0||uint64_t(value)>UINT32_MAX)throw std::invalid_argument(std::string("Positive complex operator attribute required: ")+name);return uint32_t(value);}
void check(bool ok,const char* reason){if(!ok)throw std::invalid_argument(reason);}
std::vector<int64_t> replace_last(const Tensor& t,int64_t n){auto s=t.shape;check(!s.empty(),"Rank-one or greater tensor required");s.back()=n;return s;}
uint32_t activation(const std::string& name){static const std::vector<std::string> names={"Sigmoid","Tanh","Relu","Affine","LeakyRelu","ThresholdedRelu","ScaledTanh","HardSigmoid","Elu","Softsign","Softplus"};auto i=std::find(names.begin(),names.end(),name);if(i==names.end())throw std::invalid_argument("Unsupported recurrent activation: "+name);return uint32_t(i-names.begin());}
uint32_t activation_lower(const std::string& name){if(name=="none"||name.empty())return 0;if(name=="relu")return 1;if(name=="silu"||name=="swish")return 2;if(name=="gelu")return 3;if(name=="tanh")return 4;if(name=="sigmoid")return 5;if(name=="identity")return 0;if(name=="swiglu")return 6;throw std::invalid_argument("Unknown fused activation: "+name);}
}

bool plan_complex(const Request& r,ExtendedPlan& p){
 auto rec=std::find_if(std::begin(records),std::end(records),[&](const Record& v){return r.operation==v.name;});if(rec==std::end(records))return false;
 p.operation=r.operation;p.shader="complex_"+std::to_string(rec->code)+".spv";p.work_items=1;p.args.assign(128,0);p.args[0]=rec->code;
 auto names=fields(rec->inputs);std::set<std::string> allowed;
 for(auto name:names){bool optional=name.back()=='?';if(optional)name.pop_back();allowed.insert(name);auto t=find_input(r,name);if(!t&&!optional)throw std::invalid_argument("Missing complex input: "+name);if(t)p.inputs.push_back(*t);else{Tensor missing;missing.name=name;missing.dtype=DType::Unspecified;p.inputs.push_back(missing);}}
 for(const auto& t:r.inputs)check(allowed.count(t.name),"Unknown complex operator input");
 auto I=[&](size_t at,const char* n,int64_t fallback=0){p.args.at(at)=uint32_t(attr_i(r,n,fallback));};
 auto F=[&](size_t at,const char* n,double fallback=0){p.args.at(at)=fword(float(attr_f(r,n,fallback)));};
 std::vector<Tensor> outputs;
 auto O=[&](size_t canonical,const std::string& name,std::vector<int64_t> shape,DType dtype,bool optional=false){
   bool requested=std::any_of(r.requested_outputs.begin(),r.requested_outputs.end(),[&](const Tensor& t){return t.name==name;});
   if((optional&&!requested)||(!r.requested_outputs.empty()&&!requested)){p.args[64+canonical]=UINT32_MAX;return;}
   p.args[64+canonical]=uint32_t(outputs.size());outputs.push_back(output(name,std::move(shape),dtype));
 };
 for(size_t i=64;i<80;++i)p.args[i]=UINT32_MAX;
 const auto code=rec->code;const auto type=p.inputs.empty()?DType::Float32:p.inputs[0].dtype;
 if(code<=1008){
   uint32_t heads=positive(attr_i(r,code==1000?"q_num_heads":"num_heads",code==1000&&R(p,0)==4?D(p,0,1):0),"num_heads");
   uint32_t kvheads=positive(attr_i(r,code==1000?"kv_num_heads":"kv_num_heads",code==1000&&R(p,1)==4?D(p,1,1):heads),"kv_num_heads");
   uint32_t batch=D(p,0,0),seq=R(p,0)==4?D(p,0,2):D(p,0,1),hd=last(p,0)/heads,vd=last(p,2)/kvheads,keys=R(p,1)==4?D(p,1,2):D(p,1,1),past=0;
   if(code==1000){if(R(p,0)==4)hd=last(p,0);if(R(p,2)==4)vd=last(p,2);past=H(p,4)?D(p,4,2):0;}
   if(code==1001){kvheads=heads;vd=last(p,2)/heads;}
   if(code==1002){past=H(p,3)?D(p,3,2):0;vd=last(p,2)/kvheads;}
   if(code==1003){past=H(p,7)?uint32_t(control_i(p.inputs[7])):0;keys=R(p,1)==4?D(p,1,2):D(p,1,1);vd=hd;}
   if(code==1004){past=D(p,3,2);if(!H(p,1)){hd=last(p,0)/(heads+2*kvheads);keys=seq;}vd=hd;I(20,"sparse_block_size",16);}
   if(code==1005||code==1006){uint32_t cu=code==1005?4:5;batch=D(p,cu,0)-1;seq=D(p,0,0);keys=seq;auto hidden=code==1005?D(p,1,1)/3:last(p,0);auto widths=attr_is(r,"qkv_hidden_sizes");if(!widths.empty())hidden=uint32_t(widths[0]);hd=hidden/heads;vd=hd;kvheads=heads;if(code==1005&&!widths.empty()){check(widths.size()==3&&widths[0]==widths[1]&&widths[2]>0&&widths[2]%heads==0,"Invalid packed QKV hidden sizes");vd=uint32_t(widths[2])/heads;}if(code==1006){if(R(p,0)==4){hd=last(p,0);vd=hd;}else vd=last(p,2)/heads;} }
   if(code==1007){batch=D(p,5,0)-1;seq=D(p,0,0);keys=D(p,3,0)*D(p,3,1);hd=D(p,3,3);vd=hd;kvheads=positive(attr_i(r,"kv_num_heads",D(p,3,2)),"kv_num_heads");}
   if(code==1008){seq=D(p,0,1);hd=D(p,1,1)/(3*heads);vd=hd;keys=seq;past=H(p,8)?D(p,8,3):0;}
   check(heads%kvheads==0&&hd>0&&vd>0,"Attention head geometry mismatch");
   p.args[1]=heads;p.args[2]=kvheads;F(3,"scale",1/std::sqrt(double(hd)));if(code!=1000&&unword(p.args[3])==0)p.args[3]=fword(float(1/std::sqrt(double(hd))));I(4,code==1000||code==1007?"is_causal":code==1002?"causal":"unidirectional",code==1002||code==1007?1:0);
   p.args[5]=past;p.args[6]=seq;p.args[7]=keys;p.args[8]=hd;p.args[9]=vd;p.args[10]=batch;F(16,"mask_filter_value",-10000);F(12,"softcap");F(13,"qk_norm_epsilon",1e-6);I(14,"do_rotary");I(15,"local_window_size",-1);if(code==1002){I(21,"kv_cache_bit_width",8);auto qm=[](const std::string& x){if(x=="NONE")return 0u;if(x=="PER_TENSOR")return 1u;if(x=="PER_CHANNEL")return 2u;throw std::invalid_argument("Unknown GQA cache quantization mode");};p.args[22]=qm(attr_s(r,"k_quant_type","NONE"));p.args[23]=qm(attr_s(r,"v_quant_type","NONE"));I(24,"sliding_window_cache");I(25,"smooth_softmax",-1);}I(26,"rotary_interleaved");
   p.scratch_words=std::max<uint32_t>(16,keys+past+16);
   if(code==1000){auto shape=p.inputs[0].shape;if(shape.size()==4)shape[3]=vd;else shape.back()=heads*vd;O(0,"y",shape,type);O(1,"present_key",{batch,kvheads,past+keys,hd},type,true);O(2,"present_value",{batch,kvheads,past+keys,vd},type,true);}
   else if(code==1005||code==1006||code==1007){O(0,"outputT",{D(p,0,0),heads*vd},type);p.work_items=D(p,0,0)*heads*vd;if(code==1007){O(1,"keyCacheT",p.inputs[3].shape,p.inputs[3].dtype,true);O(2,"valueCacheT",p.inputs[4].shape,p.inputs[4].dtype,true);}}
   else if(code==1008){O(0,"output",{batch,seq,heads*vd},p.inputs[2].dtype);O(1,"present",{2,batch,heads,past+keys,hd},p.inputs[2].dtype,true);}
   else {O(0,"outputT",{batch,seq,heads*vd},type);if(code==1002||code==1003||code==1004){size_t kslot=code==1002?3:code==1003?5:3,vslot=kslot+1;auto ks=H(p,kslot)?p.inputs[kslot].shape:std::vector<int64_t>{batch,kvheads,past+keys,hd};auto vs=H(p,vslot)?p.inputs[vslot].shape:std::vector<int64_t>{batch,kvheads,past+keys,vd};DType kt=H(p,kslot)?p.inputs[kslot].dtype:type,vt=H(p,vslot)?p.inputs[vslot].dtype:type;if(code==1002){if(!H(p,3)){kt=vt=p.args[22]?p.args[21]==4?DType::UInt8:DType::Int8:type;ks={batch,kvheads,keys,p.args[22]&&p.args[21]==4?(hd+1)/2:hd};vs={batch,kvheads,keys,p.args[23]&&p.args[21]==4?(vd+1)/2:vd};}for(const auto& requested:r.requested_outputs)if(requested.name=="presentKeyT"||requested.name=="presentValueT"){check(requested.shape.size()==4&&requested.shape[0]==batch&&requested.shape[1]==kvheads&&requested.shape[2]>=0,"Invalid GQA cache output geometry");bool key=requested.name=="presentKeyT";auto width=(key?hd:vd);bool quantized=(key?p.args[22]:p.args[23])!=0;check(requested.shape[3]==(quantized&&p.args[21]==4?(width+1)/2:width),"Invalid GQA packed cache width");if(key){ks=requested.shape;kt=requested.dtype;}else{vs=requested.shape;vt=requested.dtype;}}check(kt==vt,"GQA cache dtypes must agree");check(p.args[21]==4||p.args[21]==8,"Invalid GQA cache bit width");check(D(p,0,0)==D(p,1,0)&&D(p,1,0)==D(p,2,0)&&D(p,1,1)==D(p,2,1)&&last(p,0)==heads*hd&&last(p,1)==kvheads*hd&&last(p,2)==kvheads*vd,"Invalid GQA QKV geometry");for(uint32_t role=1;role<=2;++role){auto mode=p.args[21+role];check(mode?(kt==(p.args[21]==4?DType::UInt8:DType::Int8)):(kt==DType::Float32||kt==DType::Float16),"GQA quantization attributes differ from cache dtype");if(mode){size_t scale=10+role;check(H(p,scale)&&numel(p.inputs[scale])==(mode==1?1:uint64_t(kvheads)*(role==1?hd:vd)),"Invalid GQA cache scale shape");}}}O(1,code==1004?"pastKeyT":"presentKeyT",ks,kt,code==1003);O(2,code==1004?"pastValueT":"presentValueT",vs,vt,code==1003);if(code==1003){DType qkt=type;for(const auto& t:r.requested_outputs)if(t.name=="qkT")qkt=t.dtype;O(3,"qkT",{batch,heads,seq,past+keys},qkt,true);}}}
 }else if(code>=1009&&code<=1012){
   if(code==1012){p.work_items=uint32_t(numel(p.inputs[1]));O(0,"output1T",p.inputs[1].shape,p.inputs[1].dtype);O(1,"output2T",p.inputs[3].shape,p.inputs[3].dtype);}
   else {p.work_items=uint32_t(numel(p.inputs[0]));size_t cacheSlot=code==1009?1:2;check(R(p,0)==3||R(p,0)==4,"Rotary input must have rank3 or rank4");uint32_t headCount=R(p,0)==4?D(p,0,1):uint32_t(attr_i(r,"num_heads",0));if(R(p,0)==3&&headCount==0&&code==1010){uint32_t cacheHead=2*last(p,cacheSlot);check(cacheHead>0&&last(p,0)%cacheHead==0,"Invalid inferred rotary head size");headCount=last(p,0)/cacheHead;}check(headCount>0&&(R(p,0)==4||last(p,0)%headCount==0),"Rotary num_heads must divide hidden width");p.args[1]=headCount;I(2,"interleaved");I(3,"rotary_embedding_dim",0);uint32_t headSize=R(p,0)==4?last(p,0):last(p,0)/headCount,rotarySize=p.args[3]?p.args[3]:headSize;check(rotarySize>0&&rotarySize<=headSize&&rotarySize%2==0&&last(p,cacheSlot)==rotarySize/2,"Invalid rotary prefix/cache width");F(4,"scale",1);I(5,"is_packed_batching");I(6,"mrope_layout");auto sections=attr_is(r,"mrope_section");p.args[7]=uint32_t(sections.size());for(size_t i=0;i<sections.size()&&i<8;++i)p.args[8+i]=uint32_t(sections[i]);O(0,"y",p.inputs[0].shape,type);}
 }else if(code==1015){O(0,"decayT",p.inputs[0].shape,type);O(1,"betaT",p.inputs[0].shape,type,true);}
 else if(code==1013||code==1014||code==1016){
   bool delta=code==1016;uint32_t batch=R(p,0)==4?D(p,0,0):delta?(H(p,3)?D(p,3,0)-1:H(p,6)?D(p,6,0):1):D(p,0,0);
   uint32_t qheads=delta?D(p,0,R(p,0)-2):positive(attr_i(r,"q_num_heads"),"q_num_heads");uint32_t kh=delta?D(p,1,R(p,1)-2):positive(attr_i(r,"kv_num_heads"),"kv_num_heads");
   uint32_t vh=delta?D(p,2,R(p,2)-2):kh,dk=delta?last(p,0):last(p,0)/qheads,dv=delta?last(p,2):last(p,2)/kh;
   p.args[1]=qheads;p.args[2]=kh;p.args[3]=vh;p.args[4]=dk;p.args[5]=dv;p.args[6]=batch;p.args[7]=R(p,0)==4?D(p,0,1):delta?D(p,0,0)/batch:D(p,0,1);
   auto rule=attr_s(r,"update_rule","gated_delta");p.args[8]=rule=="linear"?0:(rule=="gated"||rule=="gated_linear")?1:rule=="delta"?2:rule=="gated_delta"?3:99;check(p.args[8]!=99,"Unknown linear attention update rule");F(9,"scale",0);I(10,"qk_l2_norm");I(11,"state_window");p.args[12]=attr_s(r,"gate_activation")=="qwen"?1:0;p.args[13]=attr_s(r,"beta_activation")=="sigmoid"?1:0;
   p.args[14]=delta?kh:last(p,1)/dk;check(qheads>0&&kh>0&&dk>0&&dv>0&&p.args[14]>0&&kh%p.args[14]==0,"Invalid linear attention head geometry");check(delta?vh%qheads==0:(code==1013?qheads%kh==0:(qheads%kh==0||kh%qheads==0)),"Invalid linear attention grouped heads");check(delta||last(p,1)%dk==0,"Invalid shared key head width");check(!(p.args[8]==1||p.args[8]==3)||H(p,4),"Gated linear attention requires decay");check(p.args[8]<2||H(p,5),"Delta attention requires beta");check(p.args[11]<=8,"Invalid linear state window");
   p.scratch_words=dk*dv+2*dk+dv+16;auto os=p.inputs[2].shape;if(!delta)os.back()=std::max(qheads,kh)*dv;O(0,"outputT",os,type);
   std::vector<int64_t> state=delta?std::vector<int64_t>{batch,vh,dv,dk}:std::vector<int64_t>{batch,kh,dk,dv};if(!delta&&p.args[11])state.insert(state.begin(),p.args[11]);size_t ps=delta?6:3;DType st=H(p,ps)?p.inputs[ps].dtype:delta?DType::Float32:type;O(1,delta?"finalStateT":"presentStateT",state,st);
 }else if(code>=1017&&code<=1019){
   bool varlen=code==1019;I(1,"channels_last");I(2,"dilation",1);I(3,"state_window");p.args[4]=activation_lower(attr_s(r,"activation","none"));I(5,"state_update_capacity");uint32_t batch=varlen?D(p,2,0)-1:D(p,0,0),channels=varlen?D(p,0,1):D(p,0,p.args[1]?2:1),kernel=D(p,1,2),length=(kernel-1)*p.args[2];
   p.args[6]=batch;p.args[7]=channels;p.args[8]=kernel;p.args[9]=length;O(0,"outputT",p.inputs[0].shape,type);std::vector<int64_t> state=(!varlen&&p.args[1])?std::vector<int64_t>{batch,length,channels}:std::vector<int64_t>{batch,channels,length};if(!varlen&&p.args[3])state.insert(state.begin(),p.args[3]);O(1,varlen?"finalStateT":"presentStateT",state,type);if(varlen)O(2,"stateUpdateT",{batch,p.args[5],channels},type,true);
 }else if(code>=1020&&code<=1022){
   uint32_t layout=uint32_t(attr_i(r,"layout")),seq=D(p,0,layout?1:0),batch=D(p,0,layout?0:1),hidden=D(p,2,2),dirs=D(p,1,0);check(type==DType::Float32,"Published recurrent tensors require float32");for(size_t slot=0;slot<p.inputs.size();++slot)if(H(p,slot))check(p.inputs[slot].dtype==(slot==4?DType::Int32:DType::Float32),"Recurrent tensor dtype differs from published contract");check(layout<=1&&dirs>=1&&dirs<=2,"Invalid recurrent layout/direction");
   p.args[1]=layout;auto dir=attr_s(r,"direction","forward");p.args[2]=dir=="reverse"?1:dir=="bidirectional"?2:0;I(3,code==1021?"linear_before_reset":"input_forget");F(4,"clip",0);p.args[5]=seq;p.args[6]=batch;p.args[7]=hidden;p.args[8]=dirs;p.args[9]=D(p,0,2);
   auto defaults=code==1020?std::vector<std::string>{"Tanh","Tanh"}:code==1021?std::vector<std::string>{"Sigmoid","Tanh","Sigmoid","Tanh"}:std::vector<std::string>{"Sigmoid","Tanh","Tanh","Sigmoid","Tanh","Tanh"};auto acts=attr_ss(r,"activations",defaults);auto aa=attr_fs(r,"activation_alpha"),bb=attr_fs(r,"activation_beta");size_t alpha_index=0,beta_index=0;for(size_t i=0;i<acts.size()&&i<6;++i){p.args[16+i]=activation(acts[i]);bool uses_alpha=acts[i]=="Affine"||acts[i]=="LeakyRelu"||acts[i]=="ThresholdedRelu"||acts[i]=="ScaledTanh"||acts[i]=="HardSigmoid"||acts[i]=="Elu";bool uses_beta=acts[i]=="Affine"||acts[i]=="ScaledTanh"||acts[i]=="HardSigmoid";double alpha=acts[i]=="HardSigmoid"?.2:acts[i]=="LeakyRelu"?.01:1,beta=acts[i]=="HardSigmoid"?.5:acts[i]=="ScaledTanh"?1:0;if(uses_alpha){if(alpha_index<aa.size())alpha=aa[alpha_index];++alpha_index;}if(uses_beta){if(beta_index<bb.size())beta=bb[beta_index];++beta_index;}p.args[24+i]=fword(float(alpha));p.args[32+i]=fword(float(beta));}
   O(0,"y",layout?std::vector<int64_t>{batch,seq,dirs,hidden}:std::vector<int64_t>{seq,dirs,batch,hidden},type);auto state=layout?std::vector<int64_t>{batch,dirs,hidden}:std::vector<int64_t>{dirs,batch,hidden};O(1,"y_h",state,type);if(code==1022)O(2,"y_c",state,type);p.scratch_words=hidden*16+16;
 }else if(code==1023||code==1024){
   size_t bs=code==1023?1:3;auto a=p.inputs[0].shape,b=p.inputs[bs].shape;check(!a.empty()&&!b.empty(),"Quantized MatMul rank must be positive");bool av=a.size()==1,bv=b.size()==1;if(av)a.insert(a.begin(),1);if(bv)b.push_back(1);check(a.back()==b[b.size()-2],"Quantized MatMul contraction mismatch");auto batch=broadcast(std::vector<int64_t>(a.begin(),a.end()-2),std::vector<int64_t>(b.begin(),b.end()-2));if(!av)batch.push_back(a[a.size()-2]);if(!bv)batch.push_back(b.back());p.args[1]=uint32_t(a.back());p.args[2]=uint32_t(a[a.size()-2]);p.args[3]=uint32_t(b.back());p.args[4]=av;p.args[5]=bv;O(0,"y",batch,code==1023?DType::Int32:p.inputs[7].dtype);
 }else if(code>=1025&&code<=1028){
   I(1,"K",last(p,0));I(2,"N");I(3,"block_size",32);I(4,"bits",4);I(5,"quant_type",1);I(6,"transB",1);F(7,"epsilon",code==1028?1e-6:1e-5);p.args[8]=activation_lower(attr_s(r,"activation","silu"));check(p.args[1]>0&&p.args[3]>0,"Invalid packed matmul geometry");
   if(code==1028){I(2,"Nq");I(9,"Nkv");O(0,"qT",replace_last(p.inputs[0],p.args[2]),type);O(1,"kT",replace_last(p.inputs[0],p.args[9]),type);O(2,"vT",replace_last(p.inputs[0],p.args[9]),type);O(3,"residualT",p.inputs[0].shape,type,true);}
   else{check(p.args[2]>0,"Packed matmul N required");O(0,code==1026?"yT":"yT",replace_last(p.inputs[0],p.args[2]),type);if(code==1027)O(1,"residualT",p.inputs[0].shape,type,true);}
 }else if(code==1029||code==1030){
   I(1,"k",1);I(2,"normalize_routing_weights");I(3,"swiglu_fusion");I(4,"use_sparse_mixer");I(5,"expert_weight_bits",4);I(6,"block_size",0);I(7,"weights_prepacked",-1);p.args[8]=activation_lower(attr_s(r,"activation_type","relu"));F(9,"activation_alpha",1);F(10,"activation_beta",0);F(11,"swiglu_limit",0);p.args[12]=r.attrs.numbers.count("swiglu_limit")||r.attrs.integers.count("swiglu_limit");
   uint32_t hidden=last(p,0),experts=D(p,1,1),fc1rows=D(p,2,1),fusion=p.args[8]==6&&p.args[3]?2:1,inter=fc1rows/fusion;
   check((R(p,0)==2||R(p,0)==3)&&hidden>0&&p.args[1]>=1&&p.args[1]<=experts&&p.args[4]==0,"Invalid MoE input or routing geometry");
   check(D(p,1,0)==numel(p.inputs[0])/hidden&&D(p,2,0)==experts&&D(p,4,0)==experts&&D(p,4,1)==hidden&&fc1rows%fusion==0,"Invalid MoE weight geometry");
   if(code==1029){check(D(p,2,2)==hidden&&D(p,4,2)==inter,"Invalid dense expert projection");check(p.args[3]<=2&&(p.args[8]==6||p.args[3]==0),"Invalid dense SwiGLU fusion");check(p.args[8]!=6||p.args[3]!=0||H(p,6),"Separate SwiGLU requires FC3");check(!H(p,6)||((p.args[8]==6||p.args[8]==2)&&D(p,6,0)==experts&&D(p,6,1)==inter&&D(p,6,2)==hidden),"Invalid FC3 projection");}
   else {check((p.args[5]==4||p.args[5]==8)&&(p.args[7]==UINT32_MAX||p.args[7]==0)&&attr_s(r,"quant_type","int")=="int","Unsupported QMoE integer layout");check((p.args[8]==1&&p.args[3]==0)||(p.args[8]==6&&p.args[3]==1),"Invalid quantized SwiGLU fusion");check(D(p,2,2)*(8/p.args[5])==hidden&&D(p,4,2)*(8/p.args[5])==inter,"Invalid packed expert projection");check(p.args[6]==0||(p.args[6]>=16&&(p.args[6]&(p.args[6]-1))==0&&hidden%p.args[6]==0&&inter%p.args[6]==0),"Invalid quantized expert block size");for(size_t slot:{size_t(3),size_t(5)})check(R(p,slot)==(p.args[6]?3:2)&&D(p,slot,0)==experts&&D(p,slot,1)==D(p,slot-1,1)&&(!p.args[6]||D(p,slot,2)==(slot==3?hidden:inter)/p.args[6]),"Invalid quantized expert scales");}
   O(0,"outputT",p.inputs[0].shape,type);p.work_items=uint32_t(numel(p.inputs[0]));p.scratch_words=16;
 }else if(code==1031){I(1,"bits",4);I(2,"block_size",128);I(3,"gather_axis");I(4,"quantize_axis",1);check((p.args[1]==4||p.args[1]==8)&&p.args[2]>=16&&(p.args[2]&(p.args[2]-1))==0&&p.args[3]==0&&p.args[4]==1,"Unsupported GatherBlockQuantized geometry");O(0,"outputT",{D(p,1,0),D(p,0,1)*(8/p.args[1])},p.inputs[2].dtype);}
 else if(code==1032){F(1,"epsilon",1e-12);I(2,"mask_index_type");uint32_t batch=D(p,0,0),seq=D(p,0,1),hidden=D(p,2,1);O(0,"outputT",{batch,seq,hidden},p.inputs[2].dtype);O(1,"maskIndexT",{batch},DType::Int32,true);O(2,"embeddingSumT",{batch,seq,hidden},p.inputs[2].dtype,true);}
 else if(code==1033){F(1,"epsilon",1e-5);O(0,"outputT",p.inputs[0].shape,type);}
 else if(code==1034){I(1,"num_heads");check(p.args[1]>0,"Gated relative position num_heads required");O(0,"outputT",{D(p,0,0),p.args[1],D(p,0,1),D(p,0,1)},type);}
 else if(code==1035){I(1,"max_ngram_size");I(2,"n_head_per_ngram");I(3,"pad_id");check(p.args[1]>=2&&p.args[2]>0,"Invalid ngram orders/heads");check(D(p,1,0)>=p.args[1]&&D(p,2,0)==(p.args[1]-1)*p.args[2],"Ngram multiplier/vocabulary geometry mismatch");O(0,"hashIdsT",{D(p,0,0),D(p,0,1),(p.args[1]-1)*p.args[2]},DType::Int32);O(1,"presentIdsT",{D(p,0,0),p.args[1]-1},DType::Int32);}
 else throw std::logic_error("Unregistered complex algorithm");
 // Respect explicit output ordering while retaining canonical shader slots.
 if(!r.requested_outputs.empty()){auto old=outputs;for(size_t i=0;i<r.requested_outputs.size();++i){auto it=std::find_if(old.begin(),old.end(),[&](const Tensor& t){return t.name==r.requested_outputs[i].name;});check(it!=old.end(),"Unknown requested complex output");outputs[i]=*it;for(size_t canonical=64;canonical<80;++canonical)if(p.args[canonical]==uint32_t(it-old.begin()))p.args[canonical]=uint32_t(i)|0x80000000u;}for(size_t canonical=64;canonical<80;++canonical)if(p.args[canonical]!=UINT32_MAX)p.args[canonical]&=0x7fffffffu;}
 if(code==1004){uint64_t qcount=uint64_t(p.args[10])*p.args[6]*p.args[1]*p.args[8],kcount=uint64_t(p.args[10])*p.args[6]*p.args[2]*p.args[8],capacity=D(p,3,2),rows=uint64_t(p.args[10])*p.args[6]*p.args[1],words=qcount+2*kcount+rows*capacity,transform=uint64_t(p.args[10])*p.args[6]*(p.args[1]+2ull*p.args[2]);check(words<=UINT32_MAX&&transform<=UINT32_MAX,"Sparse attention scratch exceeds uint32 storage");p.args[28]=0;p.args[29]=uint32_t(qcount);p.args[30]=uint32_t(qcount+kcount);p.args[31]=uint32_t(qcount+2*kcount);p.args[32]=uint32_t(capacity);p.scratch_words=uint32_t(words);for(uint32_t phase=0;phase<3;++phase){auto args=p.args;args[27]=phase;p.passes.push_back(Pass{phase==0?p.shader:phase==1?"complex_1004_attend.spv":"complex_1004_cache.spv",std::move(args),phase==0?uint32_t(transform):phase==1?uint32_t(rows):1u});}}
 if(code==1005){uint64_t qcount=uint64_t(D(p,0,0))*p.args[1]*p.args[8],vcount=uint64_t(D(p,0,0))*p.args[1]*p.args[9],words=2*qcount+vcount;check(words<=UINT32_MAX,"Packed projection scratch exceeds uint32 storage");p.args[28]=0;p.args[29]=uint32_t(qcount);p.args[30]=uint32_t(2*qcount);p.scratch_words=uint32_t(words);for(uint32_t phase=0;phase<2;++phase){auto args=p.args;args[27]=phase;p.passes.push_back(Pass{phase==0?p.shader:"complex_1005_attend.spv",std::move(args),phase==0?uint32_t(words):p.work_items});}}
 if(code==1007){uint64_t kcount=numel(p.inputs[3]),vcount=numel(p.inputs[4]);check(kcount+vcount<=UINT32_MAX,"Paged cache scratch exceeds uint32 storage");p.args[28]=0;p.args[29]=uint32_t(kcount);p.scratch_words=uint32_t(kcount+vcount);for(uint32_t phase=0;phase<2;++phase){auto args=p.args;args[27]=phase;p.passes.push_back(Pass{phase==0?p.shader:"complex_1007_attend.spv",std::move(args),phase==0?uint32_t(kcount+vcount):p.work_items});}}
 if(code==1002){uint64_t qcount=numel(p.inputs[0]),kcount=numel(p.inputs[1]),vcount=numel(p.inputs[2]),capacity=uint64_t(H(p,3)?D(p,3,2):0)+p.args[7],rows=uint64_t(p.args[10])*p.args[1]*p.args[6],words=qcount+kcount+vcount+rows*capacity,transform=uint64_t(p.args[10])*(uint64_t(p.args[1])*p.args[6]+2ull*p.args[2]*p.args[7]);check(words<=UINT32_MAX&&transform<=UINT32_MAX&&rows<=UINT32_MAX,"GQA scratch geometry exceeds uint32 storage");p.args[28]=0;p.args[29]=uint32_t(qcount);p.args[30]=uint32_t(qcount+kcount);p.args[31]=uint32_t(qcount+kcount+vcount);p.args[32]=uint32_t(capacity);p.scratch_words=uint32_t(words);for(uint32_t phase=0;phase<3;++phase){auto args=p.args;args[27]=phase;p.passes.push_back(Pass{phase==0?p.shader:phase==1?"complex_1002_attend.spv":"complex_1002_cache.spv",std::move(args),phase==0?uint32_t(transform):phase==1?uint32_t(rows):1u});}}
 p.outputs=std::move(outputs);return true;
}
}
