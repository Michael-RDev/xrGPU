#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

namespace vkcompute {
// Reuse the delegate's layout-aware unary dispatch and resize implementation.
void add_unary_op_node(ComputeGraph&, ValueRef, float, float, ValueRef, const std::string&);

void xrgpu_gelu(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_unary_op_node(graph, args[0], -1.0f, -1.0f, args[1], "xrgpu_gelu");
}

void xrgpu_attention(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const auto qkv=args[0], mask=args[1], out=args[3];
  const int32_t heads=graph.extract_scalar<int32_t>(args[2]);
  const int32_t length=graph.size_at<int32_t>(-2,qkv);
  const int32_t width=graph.size_at<int32_t>(-1,qkv)/3;
  VK_CHECK_COND(graph.sizes_of(qkv).size()==2 && graph.size_at<int32_t>(-1,qkv)==3*width);
  VK_CHECK_COND(heads>0 && width==heads*32 && length>0 && length<=256);
  VK_CHECK_COND(graph.numel_of(mask)==length && graph.sizes_of(out)==std::vector<int64_t>({length,width}));
  VK_CHECK_COND(graph.is_buffer_storage(qkv) && graph.is_buffer_storage(mask) && graph.is_buffer_storage(out));
  VK_CHECK_COND(graph.dtype_of(qkv)==vkapi::kFloat && graph.dtype_of(mask)==vkapi::kFloat && graph.dtype_of(out)==vkapi::kFloat);
  const int32_t params[4]={length,width,heads,0};
  graph.execute_nodes().emplace_back(new DispatchNode(
      graph, VK_KERNEL_FROM_STR("xrgpu_attention_float"),
      {64u,static_cast<uint32_t>(length),static_cast<uint32_t>(heads)}, {64u,1u,1u},
      {{out,vkapi::kWrite},{{qkv,mask},vkapi::kRead}}, {},
      {PushConstantDataInfo(params,sizeof(params))}));
}

void xrgpu_residual_norm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const auto in=args[0], residual=args[1], out=args[5];
  const int32_t rows=graph.size_at<int32_t>(-2,in), width=graph.size_at<int32_t>(-1,in);
  VK_CHECK_COND(graph.sizes_of(in).size()==2 && width>0 && width<=384 && rows>0);
  VK_CHECK_COND(graph.sizes_of(in)==graph.sizes_of(residual) && graph.sizes_of(in)==graph.sizes_of(out));
  VK_CHECK_COND(graph.is_buffer_storage(in) && graph.is_buffer_storage(residual) && graph.is_buffer_storage(out));
  VK_CHECK_COND(graph.dtype_of(in)==vkapi::kFloat && graph.dtype_of(residual)==vkapi::kFloat && graph.dtype_of(out)==vkapi::kFloat);
  const auto weight=prepack_standard_like(graph,args[2],in), bias=prepack_standard_like(graph,args[3],in);
  VK_CHECK_COND(graph.numel_of(weight)==width && graph.numel_of(bias)==width);
  const struct { int32_t rows,width; float eps; int32_t unused; } params={rows,width,graph.extract_scalar<float>(args[4]),0};
  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,VK_KERNEL_FROM_STR("xrgpu_residual_norm_float"),
      {64u,static_cast<uint32_t>(rows),1u},{64u,1u,1u},
      {{out,vkapi::kWrite},{{in,residual,weight,bias},vkapi::kRead}}, {},
      {PushConstantDataInfo(&params,sizeof(params))}));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(xrgpu.gelu.default, xrgpu_gelu);
  VK_REGISTER_OP(xrgpu.attention.default, xrgpu_attention);
  VK_REGISTER_OP(xrgpu.residual_norm.default, xrgpu_residual_norm);
}
} // namespace vkcompute
