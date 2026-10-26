// SPDX-License-Identifier: Apache-2.0
#include "framework.h"
#include "half_host.h"
#include <set>
#include <sstream>
namespace xrgpu::operators::extended {
namespace detail {
std::string json_quote(const std::string& s){std::ostringstream o;o<<'"';const char hex[]="0123456789abcdef";for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<char(c);else if(c<32)o<<"\\u00"<<hex[c>>4]<<hex[c&15];else o<<char(c);}o<<'"';return o.str();}
static DType stored(const Tensor& t){return t.storage_dtype==DType::Unspecified?t.dtype:t.storage_dtype;}
static bool floating(DType t){return t==DType::Float32||t==DType::Float16;}
static bool signed_type(DType t){return t==DType::Int64||t==DType::Int32||t==DType::Int16||t==DType::Int8;}
static void validate_tensor(const Tensor& t,bool allow_missing){
 if(t.dtype==DType::Unspecified){if(allow_missing)return;throw std::invalid_argument("Missing logical dtype");}
 numel(t);dtype_bytes(t.dtype);dtype_bytes(stored(t));
 if(t.layout!="contiguous")throw std::invalid_argument("Extended kernels require contiguous row-major tensors");
 if(floating(t.dtype)!=floating(stored(t)))throw std::invalid_argument("Logical/storage float versus integer types differ");
}
static std::vector<uint32_t> contiguous_strides(const std::vector<int64_t>& shape){std::vector<uint32_t> s(shape.size());if(!numel(shape))return s;uint64_t n=1;for(size_t i=shape.size();i>0;--i){s[i-1]=uint32_t(n);n*=uint64_t(shape[i-1]);}return s;}
static void descriptor(std::vector<uint32_t>& params,uint32_t at,const Tensor& t,uint32_t offset){
 params[at]=offset;params[at+1]=uint32_t(t.dtype);
 if(t.dtype==DType::Unspecified){params[at+2]=UINT32_MAX;return;}
 params[at+2]=uint32_t(t.shape.size());params[at+3]=uint32_t(numel(t));auto strides=contiguous_strides(t.shape);
 for(size_t i=0;i<t.shape.size();++i){params[at+4+i]=uint32_t(t.shape[i]);params[at+20+i]=strides[i];}
}
static uint64_t extend_integer(const uint8_t* p,DType type){uint64_t v=0;auto n=dtype_bytes(type);std::memcpy(&v,p,n);if(signed_type(type)&&n<8&&(v&(uint64_t(1)<<(n*8-1))))v|=UINT64_MAX<<(n*8);return v;}
double control_f(const Tensor& t,size_t index){auto type=stored(t);if(floating(type)){auto bytes=dtype_bytes(type);if(index>=numel(t)||t.data.size()!=numel(t)*bytes)throw std::invalid_argument("Float shape control invalid");uint32_t bits=0;std::memcpy(&bits,t.data.data()+index*bytes,bytes);return type==DType::Float32?double(half_host::uintBitsToFloat(bits)):double(half_host::xrgpuHalfToFloat(bits));}return double(control_i(t,index));}
static uint64_t logical_bits(const Tensor& t,size_t index){
 auto physical=stored(t);const auto* p=t.data.data()+index*dtype_bytes(physical);
 if(floating(t.dtype)){uint32_t bits=0;std::memcpy(&bits,p,dtype_bytes(physical));if(t.dtype==physical)return bits;if(t.dtype==DType::Float16)return half_host::xrgpuFloatToHalf(half_host::uintBitsToFloat(bits));return half_host::floatBitsToUint(half_host::xrgpuHalfToFloat(bits));}
 uint64_t bits=extend_integer(p,physical);auto logical_bytes=dtype_bytes(t.dtype);if(logical_bytes<8)bits&=(uint64_t(1)<<(logical_bytes*8))-1;return bits;
}
PackedInvocation pack_plan(const ExtendedPlan& source,const Options& options){
 if(source.inputs.size()>kMaxInputs||source.outputs.size()>kMaxOutputs||source.outputs.empty())throw std::invalid_argument("Extended descriptor slot limit exceeded");
 PackedInvocation result;result.plan=source;result.base_params.resize(kArgsBase);result.output_words.resize(16,0);
 Tensor missing;missing.dtype=DType::Unspecified;for(uint32_t slot=0;slot<kMaxInputs;++slot)descriptor(result.base_params,kInputsBase+slot*36,missing,0);
 result.base_params[0]=0x58524732;result.base_params[1]=uint32_t(source.inputs.size());result.base_params[2]=uint32_t(source.outputs.size());
 auto check=[&](){uint64_t bytes=uint64_t(result.input_words.size()+result.output_words.size())*4+uint64_t(result.scratch_words.empty()?std::max<uint32_t>(source.scratch_words,1):result.scratch_words.size())*4;if(bytes>options.memory_limit_bytes)throw std::invalid_argument("Extended heap memory limit exceeded");};
 for(size_t i=0;i<source.inputs.size();++i){const auto& t=source.inputs[i];validate_tensor(t,true);descriptor(result.base_params,kInputsBase+uint32_t(i)*36,t,uint32_t(result.input_words.size()));if(t.dtype==DType::Unspecified)continue;
  if(t.data.size()!=numel(t)*dtype_bytes(stored(t)))throw std::invalid_argument("Extended input byte count mismatch: "+t.name);
  uint64_t words=numel(t)*(t.dtype==DType::Int64?2:1);if(words>options.memory_limit_bytes/4)throw std::invalid_argument("Extended input exceeds heap limit");result.input_words.reserve(result.input_words.size()+size_t(words));
  for(size_t j=0;j<numel(t);++j){uint64_t bits=logical_bits(t,j);result.input_words.push_back(uint32_t(bits));if(t.dtype==DType::Int64)result.input_words.push_back(uint32_t(bits>>32));}check();
 }
 for(size_t i=0;i<source.outputs.size();++i){const auto& t=source.outputs[i];validate_tensor(t,false);descriptor(result.base_params,kOutputsBase+uint32_t(i)*36,t,uint32_t(result.output_words.size()));uint64_t words=numel(t)*(t.dtype==DType::Int64?2:1);if(words>options.memory_limit_bytes/4)throw std::invalid_argument("Extended output exceeds heap limit");result.output_words.resize(result.output_words.size()+size_t(words));check();}
 result.scratch_words.resize(std::max<uint32_t>(source.scratch_words,1));for(auto d:source.dynamic_outputs){if(d.output_index>=source.outputs.size()||d.axis>=source.outputs[d.output_index].shape.size())throw std::invalid_argument("Invalid dynamic output axis");if(uint64_t(d.scratch_word)+1>options.memory_limit_bytes/4)throw std::invalid_argument("Dynamic count scratch exceeds memory limit");if(d.scratch_word>=result.scratch_words.size())result.scratch_words.resize(size_t(d.scratch_word)+1);}
 result.base_params[15]=uint32_t(result.scratch_words.size());result.scratch_words.push_back(0);check();
 result.passes=source.passes;if(result.passes.empty())result.passes.push_back({source.shader,source.args,source.work_items});
 if(result.passes.empty()||result.passes.size()>128)throw std::invalid_argument("Extended pass limit exceeded");
 for(auto& p:result.passes){if(p.shader.empty())throw std::invalid_argument("Extended plan requires a compiled shader");if(p.args.size()>262144)throw std::invalid_argument("Extended argument heap exceeds limit");if(!p.work_items)p.work_items=1;}
 if(result.input_words.empty())result.input_words.push_back(0);check();return result;
}
std::vector<Tensor> unpack_outputs(const PackedInvocation& p,const uint32_t* heap,const uint32_t* scratch,const std::vector<Tensor>& requested){
 auto outputs=p.plan.outputs;
 for(auto dynamic:p.plan.dynamic_outputs){auto& t=outputs[dynamic.output_index];auto actual=scratch[dynamic.scratch_word];if(actual>uint64_t(t.shape[dynamic.axis]))throw std::runtime_error("GPU dynamic output count exceeds capacity");t.shape[dynamic.axis]=actual;}
 for(size_t slot=0;slot<outputs.size();++slot){auto& t=outputs[slot];const auto& capacity=p.plan.outputs[slot];auto strides=contiguous_strides(capacity.shape);auto base=p.base_params[kOutputsBase+uint32_t(slot)*36];auto physical=stored(t);auto bytes=dtype_bytes(physical);uint64_t n=numel(t);t.data.resize(n*bytes);
  for(size_t j=0;j<n;++j){size_t remaining=j,source=0;for(size_t axis=t.shape.size();axis>0;--axis){size_t coordinate=remaining%size_t(t.shape[axis-1]);remaining/=size_t(t.shape[axis-1]);source+=coordinate*strides[axis-1];}auto address=base+source*(t.dtype==DType::Int64?2:1);uint64_t bits=heap[address];if(t.dtype==DType::Int64)bits|=uint64_t(heap[address+1])<<32;
   if(floating(t.dtype)&&physical!=t.dtype){if(t.dtype==DType::Float16)bits=half_host::floatBitsToUint(half_host::xrgpuHalfToFloat(uint32_t(bits)));else bits=half_host::xrgpuFloatToHalf(half_host::uintBitsToFloat(uint32_t(bits)));}
   else if(!floating(t.dtype)){auto logical_bytes=dtype_bytes(t.dtype);if(logical_bytes<8){auto mask=(uint64_t(1)<<(logical_bytes*8))-1;bits&=mask;if(signed_type(t.dtype)&&(bits&(uint64_t(1)<<(logical_bytes*8-1))))bits|=~mask;}
    if(bytes<logical_bytes){uint64_t masked=bits&((uint64_t(1)<<(bytes*8))-1);uint64_t restored=masked;if(signed_type(physical)&&(masked&(uint64_t(1)<<(bytes*8-1))))restored|=UINT64_MAX<<(bytes*8);if(restored!=bits)throw std::runtime_error("Output value does not fit requested physical storage");}
   }
   std::memcpy(t.data.data()+j*bytes,&bits,bytes);
  }
 }
 if(!requested.empty()){if(requested.size()!=outputs.size())throw std::runtime_error("Requested output count mismatch");for(size_t i=0;i<outputs.size();++i){const auto& wanted=requested[i];const auto& actual=outputs[i];if(wanted.name!=actual.name||wanted.dtype!=actual.dtype||wanted.shape.size()!=actual.shape.size())throw std::runtime_error("GPU output metadata differs from requested output");for(size_t axis=0;axis<actual.shape.size();++axis){bool dynamic=std::any_of(p.plan.dynamic_outputs.begin(),p.plan.dynamic_outputs.end(),[&](DynamicOutput d){return d.output_index==i&&d.axis==axis;});if(dynamic){if(actual.shape[axis]>wanted.shape[axis])throw std::runtime_error("GPU dynamic output exceeds requested capacity");}else if(actual.shape[axis]!=wanted.shape[axis])throw std::runtime_error("GPU output metadata differs from requested output");}}}
 return outputs;
}
}
ExtendedPlan plan(const Request& r){
 detail::validate_contract(r);
 std::set<std::string> names;for(const auto& t:r.inputs){if(t.name.empty()||!names.insert(t.name).second)throw std::invalid_argument("Duplicate or unnamed extended input");detail::validate_tensor(t,false);}
 Request effective=r;
 auto erase_attribute=[&](const std::string& key){effective.attrs.integers.erase(key);effective.attrs.numbers.erase(key);effective.attrs.strings.erase(key);effective.attrs.integer_arrays.erase(key);effective.attrs.number_arrays.erase(key);effective.attrs.string_arrays.erase(key);effective.attrs.tensors.erase(key);};
 for(const auto& item:r.args.integers)erase_attribute(item.first);
 for(const auto& item:r.args.numbers)erase_attribute(item.first);
 for(const auto& item:r.args.strings)erase_attribute(item.first);
 for(const auto& item:r.args.integer_arrays)erase_attribute(item.first);
 for(const auto& item:r.args.number_arrays)erase_attribute(item.first);
 for(const auto& item:r.args.string_arrays)erase_attribute(item.first);
 for(const auto& item:r.args.tensors)erase_attribute(item.first);
 for(auto item:r.args.integers)effective.attrs.integers[item.first]=item.second;
 for(auto item:r.args.numbers)effective.attrs.numbers[item.first]=item.second;
 for(auto item:r.args.strings)effective.attrs.strings[item.first]=item.second;
 for(auto item:r.args.integer_arrays)effective.attrs.integer_arrays[item.first]=item.second;
 for(auto item:r.args.number_arrays)effective.attrs.number_arrays[item.first]=item.second;
 for(auto item:r.args.string_arrays)effective.attrs.string_arrays[item.first]=item.second;
 for(auto item:r.args.tensors)effective.attrs.tensors[item.first]=item.second;
 ExtendedPlan p;p.operation=r.operation;
 if(!detail::plan_tensor(effective,p)&&!detail::plan_spatial(effective,p)&&!detail::plan_complex(effective,p))throw std::invalid_argument("Unsupported extended operator: "+r.operation);
 detail::validate_contract(r,&p.outputs);
 p.operation=r.operation;if(p.inputs.size()>detail::kMaxInputs||p.outputs.empty()||p.outputs.size()>detail::kMaxOutputs)throw std::invalid_argument("Extended input/output slot limits exceeded");
 names.clear();for(auto& t:p.outputs){detail::validate_tensor(t,false);if(t.name.empty()||!names.insert(t.name).second)throw std::invalid_argument("Duplicate or unnamed extended output");}
 if(!r.requested_outputs.empty()){
  if(r.requested_outputs.size()!=p.outputs.size())throw std::invalid_argument("Requested output count differs from plan");
  for(size_t i=0;i<p.outputs.size();++i){const auto& wanted=r.requested_outputs[i];detail::validate_tensor(wanted,false);auto& actual=p.outputs[i];if(wanted.name!=actual.name||wanted.dtype!=actual.dtype)throw std::invalid_argument("Requested output name/type differs from plan");
   if(wanted.shape.size()!=actual.shape.size())throw std::invalid_argument("Requested output rank differs from plan");for(size_t axis=0;axis<actual.shape.size();++axis){bool dynamic=std::any_of(p.dynamic_outputs.begin(),p.dynamic_outputs.end(),[&](DynamicOutput d){return d.output_index==i&&d.axis==axis;});if(dynamic){if(wanted.shape[axis]>actual.shape[axis])throw std::invalid_argument("Requested dynamic capacity exceeds allocation");}else if(wanted.shape[axis]!=actual.shape[axis])throw std::invalid_argument("Requested output shape differs from plan");}
   if(wanted.storage_dtype!=DType::Unspecified)actual.storage_dtype=wanted.storage_dtype;detail::validate_tensor(actual,false);
  }
 }
 for(auto d:p.dynamic_outputs){if(d.scratch_word==UINT32_MAX)throw std::invalid_argument("Dynamic scratch offset overflow");p.scratch_words=std::max(p.scratch_words,d.scratch_word+1);}return p;
}
}
