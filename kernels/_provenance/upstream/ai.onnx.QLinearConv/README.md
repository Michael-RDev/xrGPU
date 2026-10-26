---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.QLinearConv

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 10

## Description

Performs convolution on a quantized input using quantized weights, producing a quantized output. Input and output quantization use per-tensor scales and zero points; weight quantization may be per-tensor or per-output-channel; an optional int32 bias must be pre-quantized with `scale = x_scale * w_scale` and zero point 0.

See the [ONNX `QLinearConv` spec](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `x` | `TX` | — | — | Quantized input data tensor shaped `(N x C x D1 x ... x Dn)`. | required |
| `x_scale` | `TF` | — | — | Per-tensor scale for input `x`. | required |
| `x_zero_point` | `TX` | — | — | Per-tensor zero point for input `x`. | required |
| `w` | `TW` | — | — | Quantized weight tensor shaped `(M x C/group x k1 x ... x kn)`. | required |
| `w_scale` | `TF` | — | — | Scale for weights `w`; scalar for per-tensor or 1-D of length M for per-output-channel quantization. | required |
| `w_zero_point` | `TW` | — | — | Zero point for weights `w`; scalar or 1-D of length M matching `w_scale`. | required |
| `y_scale` | `TF` | — | — | Per-tensor scale for output `y`. | required |
| `y_zero_point` | `TY` | — | — | Per-tensor zero point for output `y`. | required |
| `B` | `int32` | `1` | — | Optional 1-D bias of length `M`, pre-quantized with scale `x_scale * w_scale` and zero point 0. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `y` | `TY` | same as `x` | derived | Quantized output tensor; shape is determined by kernel size, strides, and padding. | required |

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
| `TX` | `uint8`, `int8` |
| `TW` | `uint8`, `int8` |
| `TY` | `uint8`, `int8` |
| `TF` | `float32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`conv-int-accumulate-spatial.wgsl.jinja`](build/webgpu/conv-int-accumulate-spatial.wgsl.jinja)
- [`conv-int-im2col-spatial.wgsl.jinja`](build/webgpu/conv-int-im2col-spatial.wgsl.jinja)
- [`qlinear-conv-dp4a-direct-3x3.wgsl.jinja`](build/webgpu/qlinear-conv-dp4a-direct-3x3.wgsl.jinja)
- [`qlinear-conv-nchw-x4.wgsl.jinja`](build/webgpu/qlinear-conv-nchw-x4.wgsl.jinja)
- [`qlinear-conv-requantize.wgsl.jinja`](build/webgpu/qlinear-conv-requantize.wgsl.jinja)
- [`quant-dp4a-matmul.wgsl.jinja`](build/webgpu/quant-dp4a-matmul.wgsl.jinja)
- [`quant-pad-k-matrices.wgsl.jinja`](build/webgpu/quant-pad-k-matrices.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.QLinearConv", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 1, 1] },
  x_scale: { data: x_scaleData, shape: [1] },
  x_zero_point: { data: x_zero_pointData, shape: [1] },
  w: { data: wData, shape: [1, 1, 1] },
  w_scale: { data: w_scaleData, shape: [1] },
  w_zero_point: { data: w_zero_pointData, shape: [1] },
  y_scale: { data: y_scaleData, shape: [1] },
  y_zero_point: { data: y_zero_pointData, shape: [1] },
});
```
