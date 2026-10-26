// SPDX-License-Identifier: Apache-2.0
#include "../framework.h"
#include "../half_host.h"
#include <cassert>
#include <iostream>
#include <functional>
using namespace xrgpu::operators;
using namespace xrgpu::operators::extended;
using namespace xrgpu::operators::extended::detail;
Tensor t(const char* name,DType type,std::vector<int64_t> shape){Tensor x;x.name=name;x.dtype=type;x.shape=shape;return x;}
void bad(const std::function<void()>& f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}assert(caught);}
int main(){
 assert(half_host::double_to_half(1.00048828125)==0x3c00);assert(half_host::double_to_half(std::nextafter(1.00048828125,2.0))==0x3c01);assert(half_host::double_to_half(std::nextafter(1.00048828125,0.0))==0x3c00);
 assert(numel(std::vector<int64_t>{})==1);assert(numel(std::vector<int64_t>{2,0,7})==0);
 bad([]{numel(std::vector<int64_t>{-1});});bad([]{numel(std::vector<int64_t>{65536,65536});});
 ExtendedPlan p;p.operation="unit";p.shader="tensor";
 auto a=t("half",DType::Float16,{2});uint16_t half[2]={0x3c00,0x8000};a.data.resize(4);std::memcpy(a.data.data(),half,4);
 auto b=t("indices",DType::Int64,{2});b.storage_dtype=DType::UInt32;uint32_t indices[2]={1,0xffffffff};b.data.resize(8);std::memcpy(b.data.data(),indices,8);
 Tensor missing;missing.dtype=DType::Unspecified;p.inputs={a,b,missing};auto output=t("out",DType::Int64,{2,5});output.storage_dtype=DType::UInt32;p.outputs={output};p.dynamic_outputs.push_back({0,1,5});p.scratch_words=2;
 auto packed=pack_plan(p,Options{});
 assert(packed.input_words==std::vector<uint32_t>({0x3c00,0x8000,1,0,0xffffffff,0}));
 assert(packed.base_params[kInputsBase+36]==2);assert(packed.base_params[kInputsBase+72+2]==UINT32_MAX);assert(packed.base_params[kInputsBase+31*36+2]==UINT32_MAX);
 assert(packed.base_params[15]==6&&packed.scratch_words.size()==7);
 assert(control_i(b,0)==1&&control_i(b,1)==4294967295LL);
 auto offset=packed.base_params[kOutputsBase];for(size_t i=0;i<10;++i){packed.output_words[offset+i*2]=uint32_t(10+i);packed.output_words[offset+i*2+1]=0;}packed.scratch_words[5]=3;
 auto actual=unpack_outputs(packed,packed.output_words.data(),packed.scratch_words.data(),{});
 auto capacity_bound=output;capacity_bound.shape={2,5};auto bounded=unpack_outputs(packed,packed.output_words.data(),packed.scratch_words.data(),{capacity_bound});assert(bounded[0].shape==std::vector<int64_t>({2,3}));capacity_bound.shape={2,4};bounded=unpack_outputs(packed,packed.output_words.data(),packed.scratch_words.data(),{capacity_bound});assert(bounded[0].data.size()==24);capacity_bound.shape={2,2};bad([&]{unpack_outputs(packed,packed.output_words.data(),packed.scratch_words.data(),{capacity_bound});});capacity_bound.shape={3,5};bad([&]{unpack_outputs(packed,packed.output_words.data(),packed.scratch_words.data(),{capacity_bound});});
 assert(actual[0].shape==std::vector<int64_t>({2,3}));assert(actual[0].data.size()==24);uint32_t values[6];std::memcpy(values,actual[0].data.data(),24);assert(values[0]==10&&values[2]==12&&values[3]==15&&values[5]==17);
 packed.output_words[offset+1]=UINT32_MAX;bad([&]{unpack_outputs(packed,packed.output_words.data(),packed.scratch_words.data(),{});});
 packed.output_words[offset+1]=0;packed.scratch_words[5]=6;bad([&]{unpack_outputs(packed,packed.output_words.data(),packed.scratch_words.data(),{});});
 ExtendedPlan signed_plan;signed_plan.shader="tensor";auto signed_out=t("signed",DType::Int8,{1});signed_out.storage_dtype=DType::Int32;signed_plan.outputs={signed_out};auto expanded=pack_plan(signed_plan,Options{});expanded.output_words[expanded.base_params[kOutputsBase]]=255;
 auto s=unpack_outputs(expanded,expanded.output_words.data(),expanded.scratch_words.data(),{});int32_t negative=0;std::memcpy(&negative,s[0].data.data(),4);assert(negative==-1);
 ExtendedPlan invalid=p;invalid.inputs[0].data.clear();bad([&]{pack_plan(invalid,Options{});});
 Options limit;limit.memory_limit_bytes=32;bad([&]{pack_plan(p,limit);});
 Request request;request.attrs.integers["axis"]=1;request.args.integers["axis"]=2;assert(arg_i(request,"axis")==2);request.attrs.strings["value"]="Infinity";assert(std::isinf(attr_f(request,"value")));
 Request numeric;numeric.attrs.integer_arrays["alpha"]={1,2};assert(attr_fs(numeric,"alpha")==std::vector<double>({1,2}));numeric.attrs.numbers["axis"]=2.0;assert(attr_i(numeric,"axis")==2);numeric.attrs.numbers["axis"]=2.5;bad([&]{attr_i(numeric,"axis");});numeric.attrs.numbers.erase("axis");numeric.attrs.integers["axis"]=0;numeric.args.numbers["axis"]=3.0;assert(arg_i(numeric,"axis")==3);
 assert(control_f(a,0)==1.0&&std::signbit(control_f(a,1)));
 Request valid;valid.operation="ai.onnx.Equal";valid.inputs={t("a",DType::Float32,{2}),t("b",DType::Float32,{2})};validate_contract(valid);std::vector<Tensor> boolean={t("c",DType::Bool,{2})};validate_contract(valid,&boolean);
 auto invalid_dtype=valid;invalid_dtype.inputs[0].dtype=static_cast<DType>(100);bad([&]{validate_contract(invalid_dtype);});
 auto mismatch=valid;mismatch.inputs[1].dtype=DType::Int32;bad([&]{validate_contract(mismatch);});
 mismatch=valid;mismatch.inputs[1].name="unknown";bad([&]{validate_contract(mismatch);});
 mismatch=valid;mismatch.inputs.pop_back();bad([&]{validate_contract(mismatch);});
 auto wrong_output=boolean;wrong_output[0].dtype=DType::Int32;bad([&]{validate_contract(valid,&wrong_output);});
 Request variadic;variadic.operation="ai.onnx.Sum";for(char name='a';name<='z';++name)variadic.inputs.push_back(t(std::string(1,name).c_str(),DType::Float32,{1}));validate_contract(variadic);variadic.inputs.back().name="other";bad([&]{validate_contract(variadic);});
 Request fixed;fixed.operation="com.microsoft.GroupQueryAttention";fixed.inputs={t("queryT",DType::Float32,{1,2})};bad([&]{validate_contract(fixed);});
 ExtendedPlan dynamic_limit;dynamic_limit.shader="tensor";dynamic_limit.outputs={t("out",DType::Float32,{1})};dynamic_limit.dynamic_outputs={{0,0,100}};Options small;small.memory_limit_bytes=256;bad([&]{pack_plan(dynamic_limit,small);});
 std::cout<<"Extended typed heaps, dynamic GPU-count layout, and storage integrity passed\n";
}
