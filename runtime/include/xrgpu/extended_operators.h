// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <xrgpu/operators.h>
#include <map>
namespace xrgpu::operators::extended {
// Attributes and invocation arguments are host metadata. Tensor payloads are
// contiguous little-endian storage bytes; all numerical kernels execute on GPU.
struct Attributes {
 std::map<std::string,int64_t> integers;
 std::map<std::string,double> numbers;
 std::map<std::string,std::string> strings;
 std::map<std::string,std::vector<int64_t>> integer_arrays;
 std::map<std::string,std::vector<double>> number_arrays;
 std::map<std::string,std::vector<std::string>> string_arrays;
 std::map<std::string,Tensor> tensors;
};
// Names, logical dtypes and fixed ranks follow the frozen native contracts.
// requested_outputs selects optional results and physical storage; its bytes
// are ignored and can never supply expected numerical output values. Dynamic
// axis dimensions are capacity bounds; read_outputs() returns the GPU-determined
// actual shape, which must fit those bounds. Static dimensions remain exact.
struct Request {
 std::string operation;
 std::vector<Tensor> inputs;
 Attributes attrs;
 Attributes args;
 std::vector<Tensor> requested_outputs;
};
using Invocation=Request;
struct Pass {std::string shader;std::vector<uint32_t> args;uint32_t work_items=1;};
// The shader writes the actual axis length to scratch_heap[scratch_word].
// Readback copies the retained tensor region from capacity strides, never
// computes numerical results on the CPU. Scratch allocation includes the count.
// Requested dimensions on these axes bound capacity rather than requiring the
// caller to know a data-dependent result length before GPU execution.
struct DynamicOutput {uint32_t output_index=0,axis=0,scratch_word=0;};
struct ExtendedPlan {
 std::string operation;
 std::vector<Tensor> inputs,outputs;
 std::vector<uint32_t> args;
 std::string shader;
 uint32_t work_items=1,scratch_words=0;
 std::vector<Pass> passes;
 std::vector<DynamicOutput> dynamic_outputs;
};
// Shape planning may read explicit shape/index controls. It never evaluates
// output numerical values. prepare() additionally checks owned input bytes.
ExtendedPlan plan(const Request& request);
class Operator final {
public:
 static std::unique_ptr<Operator> prepare(const std::string& package_directory,const Request& request,Options options={});
 ~Operator();
 Operator(const Operator&)=delete;
 Operator& operator=(const Operator&)=delete;
 void run();
 std::vector<Tensor> read_outputs() const;
 std::string info() const;
 void close();
private:
 struct Impl;explicit Operator(std::unique_ptr<Impl> impl);std::unique_ptr<Impl> impl_;
};
}
