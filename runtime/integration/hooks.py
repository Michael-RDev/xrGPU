"""Checked transforms of the pinned upstream sources, shared with xrBench."""


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Pinned xrGPU integration anchor changed: ' + old[:100])
    return source.replace(old, new, 1)


def gelu(source):
    source = '#include "xrgpu_kernel_package.h"\n' + source
    anchor = '  // args[1] is the `approximate` string'
    code = '''  const auto& mode = graph.extract_string(args[1]);
  auto request = xrgpu_package::descriptor("gelu", graph.sizes_of(args[0]),
      graph.sizes_of(args[2]), graph.is_buffer_storage(args[2]) ? "buffer" : "texture3d",
      {{"approximate", mode}});
  const bool supported = graph.dtype_of(args[0]) == vkapi::kFloat &&
      (graph.is_buffer_storage(args[2]) || graph.storage_type_of(args[2]) == utils::kTexture3D) &&
      (mode == "none" || mode == "tanh");
  if (!xrgpu_package::choose(request, supported, mode != "tanh",
      xrgpu_package::node_identity(&graph, args[2])).empty()) {
    return add_unary_op_node(graph, args[0], kDummyFloat, kDummyFloat, args[2],
        mode == "none" ? "xrgpu_gelu_erf" : "xrgpu_gelu_tanh");
  }
'''
    return once(source, anchor, code + anchor)


def linear(source):
    source = '#include "xrgpu_kernel_package.h"\n' + source
    helper = '''bool xrgpu_linear_pick(ComputeGraph* graph, ValueRef input, ValueRef out, bool bias) {
  const auto sizes = graph->sizes_of(input);
  const bool supported = (sizes.size() == 2 || sizes.size() == 3) &&
      (sizes.size() == 2 || sizes.at(0) == 1) &&
      graph->dtype_of(out) == vkapi::kFloat && graph->dtype_of(input) == vkapi::kFloat &&
      graph->storage_type_of(out) == utils::kTexture3D &&
      graph->storage_type_of(input) == utils::kTexture3D &&
      graph->packed_dim_of(input) == WHCN::kWidthDim && graph->packed_dim_of(out) == WHCN::kWidthDim;
  auto request = xrgpu_package::descriptor("linear", sizes, graph->sizes_of(out), "texture3d",
      {{"bias", bias}, {"weight_shape", {graph->size_at<int64_t>(-1, out), graph->size_at<int64_t>(-1, input)}}});
  return !xrgpu_package::choose(request, supported, false,
      xrgpu_package::node_identity(graph, out)).empty();
}

utils::uvec3 xrgpu_linear_local(ComputeGraph* graph, const vkapi::ShaderInfo& shader,
    const utils::uvec3& global, const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  if (xrgpu_linear_pick(graph, args.at(1).refs.at(0), args.at(0).refs.at(0), graph->get_bool(resize_args.at(1))))
    return {xrgpu_package::wg_x, xrgpu_package::wg_y, 1u};
  return pick_hw_square_wg_size(graph, shader, global, args, resize_args);
}

'''
    source = once(source, 'struct LinearIntParams final {', helper + 'struct LinearIntParams final {')
    # Both shader and global grid must select the same tile for the current node.
    if source.count('uint32_t tile_m = pick_matmul_tile_m(graph, out);') != 2:
        raise ValueError('Pinned linear tile selection anchors changed')
    source = source.replace('uint32_t tile_m = pick_matmul_tile_m(graph, out);',
        'uint32_t tile_m = xrgpu_linear_pick(graph, args.at(1).refs.at(0), out, graph->get_bool(resize_args.at(1))) ? xrgpu_package::tile_m : pick_matmul_tile_m(graph, out);')
    source = once(source, '  kernel_name.reserve(kShaderNameReserve);', '''  if (xrgpu_linear_pick(graph, input, out, has_bias)) {
    kernel_name = "xrgpu_linear_m" + std::to_string(tile_m) + (has_bias ? "_bias" : "");
  }
  kernel_name.reserve(kShaderNameReserve);''')
    source = once(source, '      pick_hw_square_wg_size,', '      xrgpu_linear_local,')
    source = once(source, 'bool use_coopmat = is_coopmat_eligible(graph, out, M, N, K);',
        'bool candidate = xrgpu_linear_pick(&graph, input, out, has_bias);\n  bool use_coopmat = !candidate && is_coopmat_eligible(graph, out, M, N, K);')
    return once(source, '/*force_buffer=*/use_coopmat);',
        '/*force_buffer=*/use_coopmat || (candidate && xrgpu_package::buffer_weights));')


def convolution(source):
    source = '#include "xrgpu_kernel_package.h"\n' + source
    anchor = '  // Use tiled path for all pointwise conv2d'
    code = '''  if (groups_val > 1) {
    const auto sizes = graph.sizes_of(in);
    const auto ws = graph.sizes_of(weight_data);
    const int32_t ci = utils::safe_downcast<int32_t>(sizes.at(1));
    const int32_t co = utils::safe_downcast<int32_t>(ws.at(0));
    auto request = xrgpu_package::descriptor("grouped_conv2d", sizes, graph.sizes_of(out), "texture3d",
        {{"weight_shape", ws}, {"groups", groups_val}, {"bias", graph.val_is_not_none(bias)},
         {"stride", *graph.get_int_list(stride)}, {"padding", *graph.get_int_list(padding)},
         {"dilation", *graph.get_int_list(dilation)}, {"transposed", transposed_val},
         {"clamp", clamp_out}, {"out_min", out_min_val}, {"out_max", out_max_val}});
    const bool supported = !transposed_val && sizes.at(0) == 1 && graph.dtype_of(in) == vkapi::kFloat &&
        graph.storage_type_of(in) == utils::kTexture3D && graph.storage_type_of(out) == utils::kTexture3D &&
        ci % groups_val == 0 && co % groups_val == 0 && ws.at(1) == ci / groups_val;
    const bool unsafe_stock = method != Conv2dMethod::Depthwise;
    if (!xrgpu_package::choose(request, supported, unsafe_stock,
        xrgpu_package::node_identity(&graph, out)).empty()) {
      check_conv_args(graph, in, out);
      const Kernel2dParams kp = create_kernel2d_params(graph, weight_data, false, stride, padding, dilation);
      check_conv2d_params(kp, false);
      auto pw = prepack_weights(graph, weight_data, Conv2dMethod::SlidingWindow);
      auto pb = prepack_biases(graph, bias, weight_data, false, utils::kTexture2D, utils::kWidthPacked);
      auto extra = create_conv2d_params(graph, weight_data, kp, false);
      OutputParams output{out_min_val, out_max_val};
      const utils::ivec2 channels = {ci, co};
      auto shader = VK_KERNEL_FROM_STR(clamp_out ? "xrgpu_grouped_conv2d_clamp_float" : "xrgpu_grouped_conv2d_float");
      graph.execute_nodes().emplace_back(new DynamicDispatchNode(graph, shader,
          conv2d_global_wg_size, conv2d_local_wg_size,
          {{out, vkapi::kWrite}, {{in, pw, pb}, vkapi::kRead}},
          {graph.logical_limits_ubo(out), graph.sizes_ubo(in), graph.create_params_buffer(kp),
           graph.create_params_buffer(extra), graph.create_params_buffer(output)},
          {PushConstantDataInfo(&channels, sizeof(channels))},
          {utils::safe_downcast<int32_t>(groups_val)},
          {weight_data, stride, padding, dilation, transposed, output_padding}, resize_conv2d_node));
      return;
    }
  }
'''
    return once(source, anchor, code + anchor)


TRANSFORMS = {'UnaryOp.cpp': gelu, 'Linear.cpp': linear, 'Convolution.cpp': convolution}
