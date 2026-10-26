// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <xrgpu/extended_operators.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
namespace xrgpu::operators::extended::detail {
constexpr uint32_t kHeaderWords=16,kDescriptorWords=36,kMaxInputs=32,kMaxOutputs=16,kMaxRank=16;
constexpr uint32_t kInputsBase=16,kOutputsBase=16+32*36,kArgsBase=16+48*36;
inline uint32_t fword(float x){uint32_t v;std::memcpy(&v,&x,4);return v;}
inline float unword(uint32_t x){float v;std::memcpy(&v,&x,4);return v;}
inline uint64_t numel(const std::vector<int64_t>& shape){
 if(shape.size()>16)throw std::invalid_argument("Extended tensor rank exceeds 16");
 bool empty=false;for(auto d:shape){if(d<0||uint64_t(d)>UINT32_MAX)throw std::invalid_argument("Invalid extended tensor dimension");empty|=d==0;}
 if(empty)return 0;uint64_t n=1;for(auto d:shape){if(n>UINT32_MAX/uint64_t(d))throw std::invalid_argument("Extended tensor exceeds uint32 elements");n*=uint64_t(d);}return n;
}
inline uint64_t numel(const Tensor& t){return t.dtype==DType::Unspecified?0:numel(t.shape);}
inline const Tensor* find_input(const Request& r,const std::string& name){
 auto it=std::find_if(r.inputs.begin(),r.inputs.end(),[&](const Tensor& t){return t.name==name;});return it==r.inputs.end()?nullptr:&*it;
}
inline const Tensor& require_input(const Request& r,const std::string& name){auto p=find_input(r,name);if(!p)throw std::invalid_argument("Missing input: "+name);return *p;}
inline void bind_inputs(ExtendedPlan& p,const Request& r,std::initializer_list<const char*> names,std::initializer_list<const char*> required={}){
 p.inputs.clear();for(auto n:names){auto t=find_input(r,n);if(t)p.inputs.push_back(*t);else{Tensor missing;missing.name=n;missing.dtype=DType::Unspecified;missing.storage_dtype=DType::Unspecified;p.inputs.push_back(missing);}}
 for(auto n:required)require_input(r,n);
}
inline Tensor output(const std::string& name,std::vector<int64_t> shape,DType dtype){Tensor t;t.name=name;t.shape=std::move(shape);t.dtype=dtype;t.storage_dtype=dtype;numel(t);return t;}
inline void add_output(ExtendedPlan& p,const std::string& name,std::vector<int64_t> shape,DType dtype){p.outputs.push_back(output(name,std::move(shape),dtype));}
inline int64_t numeric_integer(const Attributes& a,const std::string& n,int64_t fallback){auto i=a.integers.find(n);if(i!=a.integers.end())return i->second;auto f=a.numbers.find(n);if(f==a.numbers.end())return fallback;double v=f->second;if(!std::isfinite(v)||v< -9223372036854775808.0||v>=9223372036854775808.0||std::trunc(v)!=v)throw std::invalid_argument("Integer attribute required: "+n);return int64_t(v);}
inline int64_t attr_i(const Request& r,const std::string& n,int64_t fallback=0){return numeric_integer(r.attrs,n,fallback);}
inline double numeric_attr(const Attributes& a,const std::string& n,double fallback){auto i=a.numbers.find(n);if(i!=a.numbers.end())return i->second;auto j=a.integers.find(n);if(j!=a.integers.end())return double(j->second);auto s=a.strings.find(n);if(s!=a.strings.end()){size_t used=0;double v=std::stod(s->second,&used);if(used!=s->second.size())throw std::invalid_argument("Invalid numeric attribute "+n);return v;}return fallback;}
inline double attr_f(const Request& r,const std::string& n,double fallback=0){return numeric_attr(r.attrs,n,fallback);}
inline std::string attr_s(const Request& r,const std::string& n,const std::string& fallback=""){auto i=r.attrs.strings.find(n);return i==r.attrs.strings.end()?fallback:i->second;}
inline std::vector<int64_t> attr_is(const Request& r,const std::string& n,std::vector<int64_t> fallback={}){auto i=r.attrs.integer_arrays.find(n);return i==r.attrs.integer_arrays.end()?fallback:i->second;}
inline std::vector<double> attr_fs(const Request& r,const std::string& n,std::vector<double> fallback={}){auto i=r.attrs.number_arrays.find(n);if(i!=r.attrs.number_arrays.end())return i->second;auto ints=r.attrs.integer_arrays.find(n);if(ints!=r.attrs.integer_arrays.end())return std::vector<double>(ints->second.begin(),ints->second.end());return fallback;}
inline std::vector<std::string> attr_ss(const Request& r,const std::string& n,std::vector<std::string> fallback={}){auto i=r.attrs.string_arrays.find(n);return i==r.attrs.string_arrays.end()?fallback:i->second;}
inline int64_t arg_i(const Request& r,const std::string& n,int64_t fallback=0){return numeric_integer(r.args,n,attr_i(r,n,fallback));}
inline double arg_f(const Request& r,const std::string& n,double fallback=0){return numeric_attr(r.args,n,attr_f(r,n,fallback));}
inline std::string arg_s(const Request& r,const std::string& n,const std::string& fallback=""){auto i=r.args.strings.find(n);return i==r.args.strings.end()?attr_s(r,n,fallback):i->second;}
inline std::vector<int64_t> arg_is(const Request& r,const std::string& n,std::vector<int64_t> fallback={}){auto i=r.args.integer_arrays.find(n);return i==r.args.integer_arrays.end()?attr_is(r,n,fallback):i->second;}
inline std::vector<double> arg_fs(const Request& r,const std::string& n,std::vector<double> fallback={}){auto i=r.args.number_arrays.find(n);if(i!=r.args.number_arrays.end())return i->second;auto ints=r.args.integer_arrays.find(n);if(ints!=r.args.integer_arrays.end())return std::vector<double>(ints->second.begin(),ints->second.end());return attr_fs(r,n,fallback);}
inline std::vector<std::string> arg_ss(const Request& r,const std::string& n,std::vector<std::string> fallback={}){auto i=r.args.string_arrays.find(n);return i==r.args.string_arrays.end()?attr_ss(r,n,fallback):i->second;}
inline const Tensor* attr_tensor(const Request& r,const std::string& n){auto a=r.attrs.tensors.find(n);return a==r.attrs.tensors.end()?nullptr:&a->second;}
inline int64_t axis(int64_t value,size_t rank){if(value<0)value+=int64_t(rank);if(value<0||uint64_t(value)>=rank)throw std::invalid_argument("Axis out of range");return value;}
inline std::vector<int64_t> broadcast(const std::vector<int64_t>& a,const std::vector<int64_t>& b){std::vector<int64_t> s(std::max(a.size(),b.size()),1);for(size_t j=0;j<s.size();++j){auto x=j<a.size()?a[a.size()-1-j]:1,y=j<b.size()?b[b.size()-1-j]:1;if(x!=y&&x!=1&&y!=1)throw std::invalid_argument("Shapes do not broadcast");s[s.size()-1-j]=x==1?y:x;}numel(s);return s;}
inline int64_t control_i(const Tensor& t,size_t index=0){
 if(index>=numel(t))throw std::invalid_argument("Shape/index control outside tensor");
 auto physical=t.storage_dtype==DType::Unspecified?t.dtype:t.storage_dtype;auto bytes=dtype_bytes(physical);if(t.data.size()!=numel(t)*bytes)throw std::invalid_argument("Control tensor byte count mismatch");
 const uint8_t* p=t.data.data()+index*bytes;
 switch(physical){case DType::Int64:{int64_t v;std::memcpy(&v,p,8);return v;}case DType::Int32:{int32_t v;std::memcpy(&v,p,4);return v;}case DType::UInt32:{uint32_t v;std::memcpy(&v,p,4);return v;}case DType::Int16:{int16_t v;std::memcpy(&v,p,2);return v;}case DType::Int8:return int8_t(*p);case DType::UInt8:case DType::Bool:return *p;default:throw std::invalid_argument("Integer shape/index control required");}
}
double control_f(const Tensor& t,size_t index=0);
inline void push_shape(std::vector<uint32_t>& args,const std::vector<int64_t>& shape){for(auto x:shape)args.push_back(uint32_t(x));}
struct PackedInvocation {
 ExtendedPlan plan;
 std::vector<uint32_t> input_words,output_words,scratch_words,base_params;
 std::vector<Pass> passes;
};
PackedInvocation pack_plan(const ExtendedPlan&,const Options&);
std::vector<Tensor> unpack_outputs(const PackedInvocation&,const uint32_t* output,const uint32_t* scratch,const std::vector<Tensor>& requested);
std::string json_quote(const std::string&);
void validate_contract(const Request&,const std::vector<Tensor>* outputs=nullptr);
bool plan_tensor(const Request&,ExtendedPlan&);
bool plan_spatial(const Request&,ExtendedPlan&);
bool plan_complex(const Request&,ExtendedPlan&);
}
