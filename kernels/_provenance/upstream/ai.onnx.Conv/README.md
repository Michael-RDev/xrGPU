---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Conv

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 11

## Description

Applies an N-dimensional convolution to the input tensor `X` using filter weights `W` and an optional bias `B`. Supports grouped convolution, explicit per-axis padding, dilation, and stride along each spatial dimension.

See the [ONNX `Conv` spec](https://onnx.ai/onnx/operators/onnx__Conv.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input data tensor of shape `(N x C x D1 x ... x Dn)`, where `N` is the batch size and `C` is the number of channels. | required |
| `w` | `W` | `T` | — | — | Convolution filter weights of shape `(M x C/group x k1 x ... x kn)`, where `M` is the number of output feature maps. | required |
| `bias` | `B` | `T` | `1` | — | Optional 1D bias of length M added to each output channel. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | derived | Output tensor whose spatial dimensions are determined by the kernel size, strides, dilations, and padding. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `auto_pad` | `"NOTSET"` | Automatic padding mode. `NOTSET` uses `pads`; `SAME_UPPER` and `SAME_LOWER` choose padding so each output spatial size is `ceil(input / stride)`; `VALID` uses no padding. |
| `dilations` | — | Optional dilation factors, one positive integer per spatial axis. Omission means all ones. |
| `group` | `1` | Number of groups that input and output channels are split into; defaults to 1. |
| `kernel_shape` | — | Optional kernel shape, one positive integer per spatial axis. When present, it must match the spatial dimensions of the weight tensor; omission infers the shape from the weights. |
| `pads` | — | Optional explicit padding in ONNX order `[begin_axis_0, ..., begin_axis_n, end_axis_0, ..., end_axis_n]`. Omission means all zeros; it cannot be combined with an automatic padding mode. |
| `strides` | — | Optional stride factors, one positive integer per spatial axis. Omission means all ones. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `grouped_large_kernel_w4` — Shares each input row window across output channels and columns with f32 accumulation. Multi-channel groups loop over kernel rows to limit generated instructions; depthwise groups retain unrolling. Dilation-aware lanes reuse overlapping taps. Unrolled large kernels are demoted when the adapter does not report a fixed 32-wide subgroup range.
- `grouped_large_kernel_w4_bias` — Shares each input row window across output channels and columns with f32 accumulation. Multi-channel groups loop over kernel rows to limit generated instructions; depthwise groups retain unrolling. Dilation-aware lanes reuse overlapping taps. Unrolled large kernels are demoted when the adapter does not report a fixed 32-wide subgroup range.
- `grouped_large_kernel_w4_tail` — Shares each input row window across output channels and columns with f32 accumulation. Multi-channel groups loop over kernel rows to limit generated instructions; depthwise groups retain unrolling. Dilation-aware lanes reuse overlapping taps. Unrolled large kernels are demoted when the adapter does not report a fixed 32-wide subgroup range.
- `grouped_large_kernel_w4_tail_bias` — Shares each input row window across output channels and columns with f32 accumulation. Multi-channel groups loop over kernel rows to limit generated instructions; depthwise groups retain unrolling. Dilation-aware lanes reuse overlapping taps. Unrolled large kernels are demoted when the adapter does not report a fixed 32-wide subgroup range.
- `grouped_large_kernel_w4_dilated_lanes` — Shares each input row window across output channels and columns with f32 accumulation. Multi-channel groups loop over kernel rows to limit generated instructions; depthwise groups retain unrolling. Dilation-aware lanes reuse overlapping taps. Unrolled large kernels are demoted when the adapter does not report a fixed 32-wide subgroup range.
- `grouped_large_kernel_w4_dilated_lanes_bias` — Shares each input row window across output channels and columns with f32 accumulation. Multi-channel groups loop over kernel rows to limit generated instructions; depthwise groups retain unrolling. Dilation-aware lanes reuse overlapping taps. Unrolled large kernels are demoted when the adapter does not report a fixed 32-wide subgroup range.
- `pointwise_channel_reduce` — Handles pointwise unit-stride convolution with few output channels by assigning one invocation to each spatial position and accumulating every output channel in registers. It does not materialize im2col scratch.
- `pointwise_channel_reduce_bias` — Handles pointwise unit-stride convolution with few output channels by assigning one invocation to each spatial position and accumulating every output channel in registers. It does not materialize im2col scratch.
- `gemm_1x1_subgroup_matrix` — Multiplies 1x1 weights directly by NCHW input. Complete output-channel tiles use direct subgroup-matrix loads, eliminating input staging and reduction-loop barriers; partial tiles retain guarded staging. Tile geometry follows the device's workgroup limits.
- `gemm_1x1_subgroup_matrix_bias` — Multiplies 1x1 weights directly by NCHW input. Complete output-channel tiles use direct subgroup-matrix loads, eliminating input staging and reduction-loop barriers; partial tiles retain guarded staging. Tile geometry follows the device's workgroup limits.
- `im2col_gemm_subgroup_matrix` — Materializes convolution windows as a column matrix, then multiplies it by the weights with subgroup-matrix operations. It applies when reduction and spatial extents are tile-aligned and column scratch fits.
- `im2col_gemm_subgroup_matrix_bias` — Materializes convolution windows as a column matrix, then multiplies it by the weights with subgroup-matrix operations. It applies when reduction and spatial extents are tile-aligned and column scratch fits.
- `im2col_direct_f32_subgroup_matrix` — Materializes tile-aligned convolution columns and loads weights and columns directly into subgroup matrices, with no operand staging or K-loop barriers. Accumulates in f32 in the split-K reduction order of the other column-matrix routes; below the channel-tile reuse threshold, implicit gathering is used instead.
- `im2col_direct_f16_subgroup_matrix` — Materializes tile-aligned convolution columns and loads weights and columns directly into subgroup matrices, with no operand staging or K-loop barriers. Accumulates in f32 in the split-K reduction order of the other column-matrix routes; below the channel-tile reuse threshold, implicit gathering is used instead.
- `im2col_direct_f32_subgroup_matrix_bias` — Materializes tile-aligned convolution columns and loads weights and columns directly into subgroup matrices, with no operand staging or K-loop barriers. Accumulates in f32 in the split-K reduction order of the other column-matrix routes; below the channel-tile reuse threshold, implicit gathering is used instead.
- `im2col_direct_f16_subgroup_matrix_bias` — Materializes tile-aligned convolution columns and loads weights and columns directly into subgroup matrices, with no operand staging or K-loop barriers. Accumulates in f32 in the split-K reduction order of the other column-matrix routes; below the channel-tile reuse threshold, implicit gathering is used instead.
- `implicit_im2col_subgroup_matrix` — Gathers each logical 2-D or 3-D im2col tile directly from X instead of materializing a column matrix, trading extra address arithmetic for the scratch and bandwidth that materialization would cost. It applies when the output has few output-channel tiles, so the gather is not repeated across many tiles.
- `implicit_im2col_subgroup_matrix_bias` — Gathers each logical 2-D or 3-D im2col tile directly from X instead of materializing a column matrix, trading extra address arithmetic for the scratch and bandwidth that materialization would cost. It applies when the output has few output-channel tiles, so the gather is not repeated across many tiles.
- `implicit_im2col_subgroup_matrix_splitk` — Partitions the implicit-im2col reduction across workgroups when the unsplit output grid is too small, then combines raw partial sums from scratch. The combine reassociates floating-point addition relative to the unsplit route.
- `implicit_im2col_subgroup_matrix_bias_splitk` — Partitions the implicit-im2col reduction across workgroups when the unsplit output grid is too small, then combines raw partial sums from scratch. The combine reassociates floating-point addition relative to the unsplit route.
- `im2col_gemm_subgroup_matrix_padded` — Materializes a zero-padded column matrix before a subgroup-matrix multiply, allowing logical reduction and spatial extents that are not tile-aligned. Padded values contribute zero and are excluded from the logical output.
- `im2col_gemm_subgroup_matrix_padded_bias` — Materializes a zero-padded column matrix before a subgroup-matrix multiply, allowing logical reduction and spatial extents that are not tile-aligned. Padded values contribute zero and are excluded from the logical output.
- `im2col_gemm_subgroup_matrix_padded_splitk` — Materializes a zero-padded column matrix, partitions its subgroup-matrix reduction across workgroups, and combines raw partial sums from scratch. It preserves the padded route for low-parallelism outputs; the combine reassociates floating-point addition.
- `im2col_gemm_subgroup_matrix_padded_bias_splitk` — Materializes a zero-padded column matrix, partitions its subgroup-matrix reduction across workgroups, and combines raw partial sums from scratch. It preserves the padded route for low-parallelism outputs; the combine reassociates floating-point addition.
- `implicit_im2col_tiled_bias_reg_m32` — Gathers logical im2col elements directly from X while filling register-blocked GEMM tiles. It avoids column scratch and provides the portable implicit route without subgroup-matrix support.
- `implicit_im2col_tiled_reg` — Gathers logical im2col elements directly from X while filling register-blocked GEMM tiles. It avoids column scratch and provides the portable implicit route without subgroup-matrix support.
- `implicit_im2col_tiled_bias_reg` — Gathers logical im2col elements directly from X while filling register-blocked GEMM tiles. It avoids column scratch and provides the portable implicit route without subgroup-matrix support.
- `implicit_im2col_tiled_reg_splitk` — Partitions the register-blocked implicit-im2col reduction across workgroups when the unsplit output grid is too small, then combines raw partial sums. The combine reassociates floating-point addition relative to the unsplit route.
- `implicit_im2col_tiled_bias_reg_splitk` — Partitions the register-blocked implicit-im2col reduction across workgroups when the unsplit output grid is too small, then combines raw partial sums. The combine reassociates floating-point addition relative to the unsplit route.
- `im2col_gemm_tiled_reg` — Materialized f32 columns feed the configured tiled GEMM with f32 accumulation.
- `im2col_gemm_tiled` — Materialized f32 columns feed the configured tiled GEMM with f32 accumulation.
- `im2col_gemm_tiled_bias_reg` — Materialized f32 columns feed the configured tiled GEMM with f32 accumulation. Bias is applied before the final output cast.
- `im2col_gemm_tiled_bias` — Materialized f32 columns feed the configured tiled GEMM with f32 accumulation. Bias is applied before the final output cast.
- `im2col_gemm_tiled_reg_f16_columns` — Exact f16 columns feed the register-tiled GEMM with f32 accumulation. Total batched output workgroups determine eligibility. The row tile rounds output channels to 32 and caps at 64, reducing padding for narrow outputs. Actual tile storage and invocation geometry must fit the WebGPU limits.
- `im2col_gemm_tiled_f16_columns` — Exact f16 columns feed the smaller tiled GEMM with f32 accumulation. Short batched spatial workloads use the f32 column layout when it fits; this half-sized layout serves constrained buffers. The spatial-work crossover is independent of the gather workgroup size.
- `im2col_gemm_tiled_bias_reg_f16_columns` — Exact f16 columns feed register-tiled GEMM with f32 accumulation. Batched output work determines eligibility; channel tiles round to 32 and cap at 64. Actual storage and invocation geometry must fit WebGPU limits. One bias load per microtile row feeds its columns before the final output cast.
- `im2col_gemm_tiled_bias_f16_columns` — Exact f16 columns feed the smaller tiled GEMM with f32 accumulation. Short batched spatial workloads use the f32 column layout when it fits; this half-sized layout serves constrained buffers. The spatial-work crossover is independent of the gather workgroup size. Bias is applied before the final output cast.
- `gemm_1x1_tiled_reg_splitk` — Partitions the register-blocked 1x1 GEMM reduction across additional workgroups when the output grid is small, then combines raw partial sums. The combine reassociates floating-point addition relative to the unsplit route.
- `gemm_1x1_tiled_bias_reg_splitk` — Partitions the register-blocked 1x1 GEMM reduction across additional workgroups when the output grid is small, then combines raw partial sums. The combine reassociates floating-point addition relative to the unsplit route.

## Device requirements

Some implementation variants require `subgroup-matrix`, `shader-f16`, and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`conv-1x1-channel-reduce.wgsl.jinja`](build/webgpu/conv-1x1-channel-reduce.wgsl.jinja)
- [`conv-1x1-gemm-tiled-reg.wgsl.jinja`](build/webgpu/conv-1x1-gemm-tiled-reg.wgsl.jinja)
- [`conv-1x1-gemm-tiled.wgsl.jinja`](build/webgpu/conv-1x1-gemm-tiled.wgsl.jinja)
- [`conv-1x1-subgroup-matrix.wgsl.jinja`](build/webgpu/conv-1x1-subgroup-matrix.wgsl.jinja)
- [`conv-direct-nd.wgsl.jinja`](build/webgpu/conv-direct-nd.wgsl.jinja)
- [`conv-direct-unrolled.wgsl.jinja`](build/webgpu/conv-direct-unrolled.wgsl.jinja)
- [`conv-im2col-ncdhw.wgsl.jinja`](build/webgpu/conv-im2col-ncdhw.wgsl.jinja)
- [`conv-im2col-nchw.wgsl.jinja`](build/webgpu/conv-im2col-nchw.wgsl.jinja)
- [`conv-splitk-reduce.wgsl.jinja`](build/webgpu/conv-splitk-reduce.wgsl.jinja)
- [`conv1d-tiled-reg.wgsl.jinja`](build/webgpu/conv1d-tiled-reg.wgsl.jinja)
- [`conv2d-grouped-large-w4.wgsl.jinja`](build/webgpu/conv2d-grouped-large-w4.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Required output shapes and logical data types are inferred from the supplied inputs and attributes; result tensors are allocated automatically.

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Conv", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 1, 7] },
  w: { data: wData, shape: [1, 1, 1] },
});
```
