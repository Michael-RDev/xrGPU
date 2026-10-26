---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.FusedConv

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Applies an N-dimensional convolution with optional bias `B` and residual `Z`, followed by an optional fused activation. Omitting `activation` leaves the convolution result unchanged. Supported activations are `Relu`, `LeakyRelu`, `Sigmoid`, `Tanh`, `HardSigmoid`, `HardSwish`, and `Clip`; other schema-permitted activation strings are not implemented. The implementation supports one to three spatial dimensions and float16 or float32; higher spatial ranks and float64 are not implemented.

See the [ONNX Runtime `FusedConv` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedConv) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input data tensor of shape `(N, C, D1, ..., Dn)` for one to three spatial dimensions. | required |
| `w` | `W` | `T` | — | — | Convolution filter tensor of shape `(M, C/group, k1, ..., kn)`, with the same spatial rank as `X`. | required |
| `bias` | `B` | `T` | `1` | — | Optional 1-D bias tensor of length `out_channels`, broadcast-added to each output channel. | optional |
| `zResidual` | `Z` | `T` | same as `x` | — | Optional residual tensor with the same shape as the output `Y`, added before the activation. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | derived | Output feature map tensor after convolution, optional bias/residual addition, and the fused activation. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `activation` | — | Optional fused activation name: `Relu`, `LeakyRelu`, `Sigmoid`, `Tanh`, `HardSigmoid`, `HardSwish`, `Clip`, `QuickGelu` (x * sigmoid(alpha * x); alpha 1 is SiLU), `Elu`, `Gelu` (erf form, or the tanh approximation when `activation_params[0]` is nonzero), `FastGelu` (always the tanh approximation), `Softplus`, `ThresholdedRelu`, or `Erf`. Omission applies no activation. The last seven are the set onnxruntime's WebGPU provider fuses into Conv; its CPU provider rejects them. |
| `activation_params` | — | Positional parameters for the fused activation: exactly `[alpha]` is required for `LeakyRelu`, and exactly `[alpha, beta]` or `[min, max]` is required for `HardSigmoid` or `Clip`, respectively. `QuickGelu`, `Elu` and `ThresholdedRelu` take an optional `[alpha]` (defaults 1.702, 1.0 and 1.0); `Gelu` takes an optional `[approximate]` flag (0 = erf, nonzero = tanh). Parameter-free activations ignore this attribute. |
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

- `implicit_im2col_tiled_reg_splitk` — Partitions the register-blocked implicit-im2col reduction across workgroups when the unsplit output grid is too small, then combines raw partial sums and applies the bias and fused activation once. The combine reassociates floating-point addition relative to the unsplit route.
- `implicit_im2col_tiled_bias_reg_splitk` — Partitions the register-blocked implicit-im2col reduction across workgroups when the unsplit output grid is too small, then combines raw partial sums and applies the bias and fused activation once. The combine reassociates floating-point addition relative to the unsplit route.
- `implicit_im2col_tiled_reg_splitk_preferred` — Partitions the register-blocked implicit-im2col reduction across workgroups when the unsplit output grid is too small, then combines raw partial sums and applies the bias and fused activation once. The combine reassociates floating-point addition relative to the unsplit route.
- `implicit_im2col_tiled_bias_reg_splitk_preferred` — Partitions the register-blocked implicit-im2col reduction across workgroups when the unsplit output grid is too small, then combines raw partial sums and applies the bias and fused activation once. The combine reassociates floating-point addition relative to the unsplit route.
- `implicit_im2col_tiled_reg` — Gathers logical im2col elements directly from `X` into register-blocked GEMM tiles and applies the fused activation in the epilogue, avoiding a materialized column scratch buffer.
- `implicit_im2col_tiled_bias_reg` — Gathers logical im2col elements directly from `X` into register-blocked GEMM tiles and applies bias and the fused activation in the epilogue, avoiding a materialized column scratch buffer.
- `gemm_1x1_subgroup_matrix` — Multiplies 1x1 weights directly by NCHW input, then applies the selected bias, residual and activation epilogue. Complete output-channel tiles bypass input staging and reduction-loop barriers; partial tiles retain guarded staging. Tile geometry follows the device's workgroup limits.
- `gemm_1x1_subgroup_matrix_z` — Multiplies 1x1 weights directly by NCHW input, then applies the selected bias, residual and activation epilogue. Complete output-channel tiles bypass input staging and reduction-loop barriers; partial tiles retain guarded staging. Tile geometry follows the device's workgroup limits.
- `gemm_1x1_subgroup_matrix_bias` — Multiplies 1x1 weights directly by NCHW input, then applies the selected bias, residual and activation epilogue. Complete output-channel tiles bypass input staging and reduction-loop barriers; partial tiles retain guarded staging. Tile geometry follows the device's workgroup limits.
- `gemm_1x1_subgroup_matrix_bias_z` — Multiplies 1x1 weights directly by NCHW input, then applies the selected bias, residual and activation epilogue. Complete output-channel tiles bypass input staging and reduction-loop barriers; partial tiles retain guarded staging. Tile geometry follows the device's workgroup limits.
- `im2col_gemm_subgroup_matrix` — Multiply aligned materialized columns through workgroup-staged subgroup matrices with f32 accumulation and the fused epilogue.
- `im2col_direct_inputs_subgroup_matrix` — Multiply aligned materialized f32 columns through direct subgroup-matrix loads, retaining the f32 accumulation order and fused epilogue while omitting operand staging and K-loop barriers.
- `im2col_gemm_subgroup_matrix_z` — Multiply aligned materialized columns through workgroup-staged subgroup matrices with f32 accumulation and the fused epilogue.
- `im2col_direct_inputs_subgroup_matrix_z` — Multiply aligned materialized f32 columns through direct subgroup-matrix loads, retaining the f32 accumulation order and fused epilogue while omitting operand staging and K-loop barriers.
- `im2col_gemm_subgroup_matrix_bias` — Multiply aligned materialized columns through workgroup-staged subgroup matrices with f32 accumulation and the fused epilogue.
- `im2col_direct_inputs_subgroup_matrix_bias` — Multiply aligned materialized f32 columns through direct subgroup-matrix loads, retaining the f32 accumulation order and fused epilogue while omitting operand staging and K-loop barriers.
- `im2col_gemm_subgroup_matrix_bias_z` — Multiply aligned materialized columns through workgroup-staged subgroup matrices with f32 accumulation and the fused epilogue.
- `im2col_direct_inputs_subgroup_matrix_bias_z` — Multiply aligned materialized f32 columns through direct subgroup-matrix loads, retaining the f32 accumulation order and fused epilogue while omitting operand staging and K-loop barriers.
- `im2col_half_direct_subgroup_matrix` — Materialize aligned f16 convolution columns without widening their storage, then load weights and columns directly into subgroup matrices. Preserve f32 accumulation and the existing bias, residual, activation and f16 output rounding while removing operand staging and K-loop barriers.
- `im2col_half_direct_subgroup_matrix_z` — Materialize aligned f16 convolution columns without widening their storage, then load weights and columns directly into subgroup matrices. Preserve f32 accumulation and the existing bias, residual, activation and f16 output rounding while removing operand staging and K-loop barriers.
- `im2col_half_direct_subgroup_matrix_bias` — Materialize aligned f16 convolution columns without widening their storage, then load weights and columns directly into subgroup matrices. Preserve f32 accumulation and the existing bias, residual, activation and f16 output rounding while removing operand staging and K-loop barriers.
- `im2col_half_direct_subgroup_matrix_bias_z` — Materialize aligned f16 convolution columns without widening their storage, then load weights and columns directly into subgroup matrices. Preserve f32 accumulation and the existing bias, residual, activation and f16 output rounding while removing operand staging and K-loop barriers.
- `im2col_gemm_tiled_reg` — Materialize f32 columns and run a register-tiled GEMM with f32 accumulation.
- `im2col_gemm_tiled_bias_reg` — Materialize f32 columns and run a register-tiled GEMM with f32 accumulation. Add bias before activation.
- `im2col_gemm_tiled_reg_f16_columns` — Materialize f16 inputs directly into f16 columns, then run the register-tiled GEMM with f32 accumulation. This preserves the input values exactly and halves column-buffer traffic and storage; eligibility uses the actual half-precision allocation and WebGPU buffer limits.
- `im2col_gemm_tiled_bias_reg_f16_columns` — Materialize f16 inputs directly into f16 columns, then run the register-tiled GEMM with f32 accumulation. This preserves the input values exactly and halves column-buffer traffic and storage; eligibility uses the actual half-precision allocation and WebGPU buffer limits. Add bias before activation.
- `grouped_large_kernel_w4` — Shares input windows across channels and columns with f32 accumulation. Filter area, channel bytes and activation presence choose row looping or unrolling. Bias seeds the accumulators; activation follows the reduction. Small workloads and large unrolled kernels outside a fixed 32-wide subgroup range remain eligible as demoted fallbacks.
- `grouped_large_kernel_w4_bias` — Shares input windows across channels and columns with f32 accumulation. Filter area, channel bytes and activation presence choose row looping or unrolling. Bias seeds the accumulators; activation follows the reduction. Small workloads and large unrolled kernels outside a fixed 32-wide subgroup range remain eligible as demoted fallbacks.
- `grouped_large_kernel_w4_tail` — Shares input windows across channels and columns with f32 accumulation. Filter area, channel bytes and activation presence choose row looping or unrolling. Bias seeds the accumulators; activation follows the reduction. Small workloads and large unrolled kernels outside a fixed 32-wide subgroup range remain eligible as demoted fallbacks.
- `grouped_large_kernel_w4_tail_bias` — Shares input windows across channels and columns with f32 accumulation. Filter area, channel bytes and activation presence choose row looping or unrolling. Bias seeds the accumulators; activation follows the reduction. Small workloads and large unrolled kernels outside a fixed 32-wide subgroup range remain eligible as demoted fallbacks.
- `grouped_large_kernel_w4_dilated_lanes` — Shares input windows across channels and columns with f32 accumulation. Filter area, channel bytes and activation presence choose row looping or unrolling. Bias seeds the accumulators; activation follows the reduction. Small workloads and large unrolled kernels outside a fixed 32-wide subgroup range remain eligible as demoted fallbacks.
- `grouped_large_kernel_w4_dilated_lanes_bias` — Shares input windows across channels and columns with f32 accumulation. Filter area, channel bytes and activation presence choose row looping or unrolling. Bias seeds the accumulators; activation follows the reduction. Small workloads and large unrolled kernels outside a fixed 32-wide subgroup range remain eligible as demoted fallbacks.
- `implicit_im2col_subgroup_matrix` — Gathers convolution input tiles directly into subgroup-matrix operands and applies the existing bias, residual, and activation epilogue, avoiding a materialized column matrix when output-channel reuse permits or both column layouts exceed device allocation limits.
- `implicit_im2col_subgroup_matrix_z` — Gathers convolution input tiles directly into subgroup-matrix operands and applies the existing bias, residual, and activation epilogue, avoiding a materialized column matrix when output-channel reuse permits or both column layouts exceed device allocation limits.
- `implicit_im2col_subgroup_matrix_bias` — Gathers convolution input tiles directly into subgroup-matrix operands and applies the existing bias, residual, and activation epilogue, avoiding a materialized column matrix when output-channel reuse permits or both column layouts exceed device allocation limits.
- `implicit_im2col_subgroup_matrix_bias_z` — Gathers convolution input tiles directly into subgroup-matrix operands and applies the existing bias, residual, and activation epilogue, avoiding a materialized column matrix when output-channel reuse permits or both column layouts exceed device allocation limits.
- `implicit_im2col_subgroup_matrix_f16` — Gathers convolution input tiles directly into subgroup-matrix operands and applies the existing bias, residual, and activation epilogue, avoiding a materialized column matrix when output-channel reuse permits or both column layouts exceed device allocation limits.
- `implicit_im2col_subgroup_matrix_z_f16` — Gathers convolution input tiles directly into subgroup-matrix operands and applies the existing bias, residual, and activation epilogue, avoiding a materialized column matrix when output-channel reuse permits or both column layouts exceed device allocation limits.
- `implicit_im2col_subgroup_matrix_bias_f16` — Gathers convolution input tiles directly into subgroup-matrix operands and applies the existing bias, residual, and activation epilogue, avoiding a materialized column matrix when output-channel reuse permits or both column layouts exceed device allocation limits.
- `implicit_im2col_subgroup_matrix_bias_z_f16` — Gathers convolution input tiles directly into subgroup-matrix operands and applies the existing bias, residual, and activation epilogue, avoiding a materialized column matrix when output-channel reuse permits or both column layouts exceed device allocation limits.

## Device requirements

Some implementation variants require `subgroup-matrix`, `shader-f16`, and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`conv-1x1-gemm-tiled-reg.wgsl.jinja`](build/webgpu/conv-1x1-gemm-tiled-reg.wgsl.jinja)
- [`conv-1x1-gemm-tiled.wgsl.jinja`](build/webgpu/conv-1x1-gemm-tiled.wgsl.jinja)
- [`conv-1x1-subgroup-matrix.wgsl.jinja`](build/webgpu/conv-1x1-subgroup-matrix.wgsl.jinja)
- [`conv-direct-nd.wgsl.jinja`](build/webgpu/conv-direct-nd.wgsl.jinja)
- [`conv-direct-unrolled.wgsl.jinja`](build/webgpu/conv-direct-unrolled.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/com.microsoft.FusedConv", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 32, 8, 8] },
  w: { data: wData, shape: [32, 32, 1, 1] },
});
```
