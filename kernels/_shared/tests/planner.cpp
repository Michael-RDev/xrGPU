// SPDX-License-Identifier: Apache-2.0
#include <xrgpu/operators.h>
#include "../internal.h"
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>
using namespace xrgpu::operators;
Tensor tensor(const char* name,std::vector<int64_t> shape,DType type=DType::Float32) {Tensor t;t.name=name;t.shape=shape;t.dtype=type;return t;}
void fails(std::function<void()> fn) {bool caught=false;try{fn();}catch(const std::invalid_argument&){caught=true;}assert(caught);}
int main() {
 for(auto t:{DType::Float32,DType::Float16,DType::Int32,DType::UInt32,DType::Int16,DType::Int8,DType::UInt8,DType::Int64,DType::Bool})assert(parse_dtype(dtype_name(t))==t);
 auto out=plan("ai.onnx.Add",{tensor("b",{3}),tensor("a",{2,3})})[0];assert(out.name=="c"&&out.shape==std::vector<int64_t>({2,3}));
 assert(plan("ai.onnx.Add",{tensor("a",{}),tensor("b",{})})[0].shape.empty());
 assert(plan("ai.onnx.Add",{tensor("a",{0,3}),tensor("b",{1,3})})[0].shape==std::vector<int64_t>({0,3}));
 assert(plan("ai.onnx.Relu",{tensor("x",{0,3})})[0].shape==std::vector<int64_t>({0,3}));
 for(auto t:{DType::Float32,DType::Float16,DType::Int32,DType::UInt32,DType::Int8,DType::UInt8})assert(plan("ai.onnx.Add",{tensor("a",{7},t),tensor("b",{7},t)})[0].dtype==t);
 for(auto t:{DType::Float32,DType::Float16,DType::Int32,DType::Int16,DType::Int8})assert(plan("ai.onnx.Relu",{tensor("x",{7},t)})[0].dtype==t);
 assert(plan("ai.onnx.MatMul",{tensor("a",{7,5}),tensor("b",{5,9})})[0].shape==std::vector<int64_t>({7,9}));
 assert(plan("ai.onnx.MatMul",{tensor("a",{5}),tensor("b",{5})})[0].shape.empty());
 assert(plan("ai.onnx.MatMul",{tensor("a",{5}),tensor("b",{2,5,9})})[0].shape==std::vector<int64_t>({2,9}));
 assert(plan("ai.onnx.MatMul",{tensor("a",{2,7,5}),tensor("b",{5})})[0].shape==std::vector<int64_t>({2,7}));
 assert(plan("ai.onnx.MatMul",{tensor("a",{2,1,7,5}),tensor("b",{1,3,5,9})})[0].shape==std::vector<int64_t>({2,3,7,9}));
 auto zero=detail::make_plan("ai.onnx.MatMul",{tensor("a",{7,0}),tensor("b",{0,9})},{});assert(zero.parameters[0]==63&&zero.parameters[4]==0);
 auto broadcast=detail::make_plan("ai.onnx.Add",{tensor("a",{2,1,3}),tensor("b",{4,3})},{});
 assert(broadcast.parameters[24]==3&&broadcast.parameters[25]==0&&broadcast.parameters[26]==1);
 assert(broadcast.parameters[40]==0&&broadcast.parameters[41]==3&&broadcast.parameters[42]==1);
 auto batched=detail::make_plan("ai.onnx.MatMul",{tensor("a",{2,1,7,5}),tensor("b",{1,3,5,9})},{});
 assert(batched.parameters[24]==35&&batched.parameters[25]==0&&batched.parameters[40]==0&&batched.parameters[41]==45);
 fails([]{plan("ai.onnx.Add",{tensor("a",{2}),tensor("b",{3})});});
 fails([]{plan("ai.onnx.MatMul",{tensor("a",{}),tensor("b",{3})});});
 fails([]{plan("ai.onnx.MatMul",{tensor("a",{2,3}),tensor("b",{4,3})});});
 fails([]{plan("ai.onnx.Add",{tensor("a",{3}),tensor("a",{3})});});
 fails([]{plan("ai.onnx.Add",{tensor("a",{3}),tensor("b",{3},DType::Int8)});});
 fails([]{plan("ai.onnx.Relu",{tensor("x",{3},DType::UInt32)});});
 fails([]{plan("ai.onnx.Add",{tensor("a",{-1}),tensor("b",{1})});});
 fails([]{plan("ai.onnx.Add",{tensor("a",{65536,65536}),tensor("b",{1})});});
 fails([]{plan("ai.onnx.Add",{tensor("a",std::vector<int64_t>(17,1)),tensor("b",{1})});});
 fails([]{auto a=tensor("a",{3});a.layout="strided";plan("ai.onnx.Add",{a,tensor("b",{3})});});
 fails([]{auto a=tensor("a",{3});a.storage_dtype=DType::Int32;plan("ai.onnx.Add",{a,tensor("b",{3})});});
 fails([]{plan("ai.onnx.Add",{tensor("a",{3}),tensor("b",{3})},{tensor("wrong",{3})});});
 fails([]{plan("ai.onnx.Unknown",{});});
 std::cout<<"Native operator metadata planner cases passed\n";
}
