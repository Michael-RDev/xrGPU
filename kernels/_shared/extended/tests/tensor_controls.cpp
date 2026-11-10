// SPDX-License-Identifier: Apache-2.0
#include "../framework.h"
#include "../half_host.h"
#include <iostream>

using namespace xrgpu::operators;
using namespace xrgpu::operators::extended;
using namespace xrgpu::operators::extended::detail;

namespace {
// Keep these checks active in Release builds as well as Debug builds.
void check(bool condition, const char* message) {
 if (!condition) throw std::runtime_error(message);
}

template<class F> void rejects(F action) {
 try { action(); } catch (const std::invalid_argument&) { return; }
 throw std::runtime_error("Expected invalid controls to be rejected");
}

Tensor tensor(const char* name, DType type, std::vector<int64_t> shape) {
 Tensor t; t.name=name; t.dtype=type; t.shape=std::move(shape); return t;
}

Tensor integer(const char* name, DType type, int64_t value) {
 auto t=tensor(name,type,{1});
 t.data.resize(dtype_bytes(type));
 std::memcpy(t.data.data(),&value,t.data.size());
 return t;
}

Tensor floating_scalar(const char* name, DType logical, DType storage, float value) {
 auto t=tensor(name,logical,{}); t.storage_dtype=storage;
 auto physical=storage==DType::Unspecified?logical:storage;
 uint32_t bits=physical==DType::Float16
  ? half_host::xrgpuFloatToHalf(value) : half_host::floatBitsToUint(value);
 t.data.resize(dtype_bytes(physical));
 std::memcpy(t.data.data(),&bits,t.data.size());
 return t;
}

void slice(DType storage, int64_t size, int64_t start, int64_t end, int64_t step,
           const std::vector<int64_t>& expected) {
 Request r; r.operation="ai.onnx.Slice";
 // The frozen contract uses logical int32 controls; physical storage may be wider.
 auto control=[&](const char* name,int64_t value){
  auto t=integer(name,storage,value);t.dtype=DType::Int32;t.storage_dtype=storage;return t;
 };
 r.inputs={tensor("data",DType::Float32,{size}),control("starts",start),
           control("ends",end),control("steps",step)};
 auto p=plan(r);
 check(p.outputs[0].shape==std::vector<int64_t>{int64_t(expected.size())},
       "Slice output extent differs from expected selection");
 for (size_t i=0;i<expected.size();++i) {
  // Interpret the encoded start/step as the shader does, checking actual selected indices.
  int64_t index=int32_t(p.args[32])+int64_t(i)*int32_t(p.args[48]);
  check(index==expected[i],"Slice encoded the wrong source index");
 }
}

void test_slice() {
 for (auto type : {DType::Int32,DType::Int64}) {
  int64_t maximum=INT32_MAX,minimum=INT32_MIN;
  slice(type,3,2,maximum,-1,{});
  slice(type,3,maximum,minimum,-1,{2,1,0});
  slice(type,3,2,minimum,-2,{2,0});
  slice(type,3,2,-1,-1,{});
  slice(type,3,-1,-3,-1,{2,1});
  slice(type,3,minimum,minimum,-1,{0});
  slice(type,3,0,maximum,1,{0,1,2});
  slice(type,3,0,maximum,2,{0,2});
  slice(type,3,0,maximum,maximum,{0});
  slice(type,3,2,minimum,minimum,{2});
  slice(type,0,maximum,minimum,-1,{});
  slice(type,0,0,maximum,1,{});
  rejects([&]{slice(type,3,0,3,0,{});});
 }
}

Request range(DType logical, DType storage, float start, float limit, float delta) {
 Request r; r.operation="ai.onnx.Range";
 r.inputs={floating_scalar("start",logical,storage,start),
           floating_scalar("limit",logical,storage,limit),
           floating_scalar("delta",logical,storage,delta)};
 return r;
}

void extent(const Request& r, int64_t expected) {
 check(plan(r).outputs[0].shape==std::vector<int64_t>{expected},
       "Range extent does not match logical scalar values");
}

void test_range() {
 for (auto logical : {DType::Float16,DType::Float32}) {
  for (auto storage : {DType::Unspecified,DType::Float16,DType::Float32}) {
   extent(range(logical,storage,0,3,1),3);
   extent(range(logical,storage,3,0,-1),3);
   extent(range(logical,storage,3,0,1),0);
   rejects([&]{plan(range(logical,storage,0,3,0));});
  }
 }
 // Non-representable values must be rounded exactly as they will be on upload.
 auto rounded=range(DType::Float16,DType::Float32,0,3.0001f,1.0001f);
 extent(rounded,3);
 auto packed=pack_plan(plan(rounded),Options{});
 check(packed.input_words==std::vector<uint32_t>({0x0000,0x4200,0x3c00}),
       "Range planning and packed half controls disagree");
 extent(range(DType::Float16,DType::Float32,0.9999f,3,1),2);
 extent(range(DType::Float32,DType::Float32,0,3.0001f,1),4);
 rejects([]{plan(range(DType::Float16,DType::Float32,0,3,1e-8f));});
 auto truncated=range(DType::Float16,DType::Float32,0,3,1);
 truncated.inputs[1].data.resize(2);
 rejects([&]{plan(truncated);});
 auto oversized=range(DType::Float16,DType::Float16,0,3,1);
 oversized.inputs[1].data.resize(4);
 rejects([&]{plan(oversized);});
 Request ints; ints.operation="ai.onnx.Range";
 ints.inputs={integer("start",DType::Int32,0),integer("limit",DType::Int32,3),
              integer("delta",DType::Int32,1)};
 for (auto& t:ints.inputs) t.shape.clear();
 extent(ints,3);
}

void compressed(const Request& r, std::vector<int64_t> shape, uint32_t axis, bool has_axis) {
 auto p=plan(r);
 check(p.outputs[0].shape==shape,"Compress output capacity/rank differs");
 check(p.args[1]==axis&&p.args[2]==uint32_t(has_axis),"Compress axis encoding differs");
 check(p.dynamic_outputs.size()==1&&p.dynamic_outputs[0].axis==axis,
       "Compress dynamic count targets the wrong output axis");
}

void test_compress() {
 Request r; r.operation="ai.onnx.Compress";
 r.inputs={tensor("input",DType::Float32,{2,3}),tensor("condition",DType::Bool,{3})};
 compressed(r,{3},0,false);
 r.attrs.integers["axis"]=1;
 compressed(r,{2,3},1,true);
 r.attrs.integers.clear(); r.attrs.numbers["axis"]=1.0;
 compressed(r,{2,3},1,true);
 r.attrs.numbers["axis"]=-1.0;
 compressed(r,{2,3},1,true);
 r.attrs.numbers["axis"]=0.0;
 compressed(r,{2,3},0,true);
 r.args.integers["axis"]=1;
 compressed(r,{2,3},1,true);
 r.args.integers.clear(); r.attrs.numbers.clear(); r.attrs.integers["axis"]=0;
 r.args.numbers["axis"]=1.0;
 compressed(r,{2,3},1,true);
 r.attrs.integers.clear();
 compressed(r,{2,3},1,true);
 r.args.numbers["axis"]=1.5;
 rejects([&]{plan(r);});
 r.args.numbers["axis"]=2.0;
 rejects([&]{plan(r);});
}
}

int main() {
 int failures=0;
 for (auto test : {std::pair<const char*,void(*)()>{"Slice",test_slice},
                   {"Range",test_range},{"Compress",test_compress}}) {
  try { test.second(); std::cout<<test.first<<" controls passed\n"; }
  catch (const std::exception& e) { ++failures; std::cerr<<test.first<<": "<<e.what()<<'\n'; }
 }
 return failures?1:0;
}
