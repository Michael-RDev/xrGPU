---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.ConvInteger

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 10

## Description

Performs integer convolution on quantized inputs `x` and filter `w`, each with an optional zero point, producing an `int32` output. Zero-point subtraction is applied before accumulation; the result must not overflow 32 bits during accumulation.

See the [ONNX `ConvInteger` spec](https://onnx.ai/onnx/operators/onnx__ConvInteger.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `x` | `TX` | — | — | Input data tensor of shape `(N x C x D1 x ... x Dn)`. | required |
| `w` | `TW` | — | — | Convolution weight tensor of shape `(M x C/group x k1 x ... x kn)`. | required |
| `x_zero_point` | `TX` | — | — | Optional scalar zero point for `x`; defaults to 0. | optional |
| `w_zero_point` | `TW` | — | — | Optional scalar or per-output-channel zero point for `w`; defaults to 0. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `y` | `TY` | same as `x` | derived | Output tensor containing `int32` convolution results. | required |

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
| `TY` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `dp4a_pointwise_1x1_tail` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.
- `dp4a_pointwise_1x1_tail_x_zero_point_only` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.
- `dp4a_pointwise_1x1_tail_w_zero_point_only` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.
- `dp4a_pointwise_1x1_tail_no_zero_points` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.
- `dp4a_pointwise_1x1` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.
- `dp4a_pointwise_1x1_x_zero_point_only` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.
- `dp4a_pointwise_1x1_w_zero_point_only` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.
- `dp4a_pointwise_1x1_no_zero_points` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.
- `dp4a_pointwise_1x1_tail_per_channel` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.
- `dp4a_pointwise_1x1_tail_per_channel_w_zero_point_only` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.
- `dp4a_pointwise_1x1_per_channel` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.
- `dp4a_pointwise_1x1_per_channel_w_zero_point_only` — Exact tiled pointwise convolution for one or two spatial dimensions. Native packed dots or exact integer-vector dots share zero-point specialization; tile resources and dispatch eligibility follow device limits.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`conv-int-accumulate-spatial.wgsl.jinja`](build/webgpu/conv-int-accumulate-spatial.wgsl.jinja)
- [`conv-int-im2col-spatial.wgsl.jinja`](build/webgpu/conv-int-im2col-spatial.wgsl.jinja)
- [`quant-dp4a-matmul.wgsl.jinja`](build/webgpu/quant-dp4a-matmul.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.ConvInteger", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 1, 2, 1, 1] },
  w: { data: wData, shape: [1, 1, 1, 1, 1] },
});
```
