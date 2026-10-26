---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.DeformConv

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 22

## Description

Performs deformable convolution by sampling input at spatially offset locations specified per output position, enabling the kernel to adapt its receptive field shape. The `offset` tensor provides `(y, x)` offsets for each kernel point and output location; fractional offsets are resolved via bilinear interpolation and out-of-bounds locations contribute zero. An optional `mask` tensor modulates each sampling point's contribution.

See the [ONNX `DeformConv` spec](https://onnx.ai/onnx/operators/onnx__DeformConv.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | `4` | — | Input data tensor of shape `(N, C, H, W)` for 2D or `(N, C, D1, ..., Dn)` in general. | required |
| `w` | `W` | `T` | `4` | — | Convolution weight tensor of shape `(oC, C/group, kH, kW)`. | required |
| `offset` | — | `T` | `4` | — | Per-output-position sampling offsets of shape `(N, offset_group * kH * kW * 2, oH, oW)` for 2D data. | required |
| `bias` | `B` | `T` | `1` | — | Optional 1D bias of length oC added to the convolution output. | optional |
| `mask` | — | `T` | `4` | — | Optional modulation mask of shape `(N, offset_group * kH * kW, oH, oW)` scaling each sampled kernel point; defaults to ones. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | `4` | derived | Output tensor of shape `(N, oC, oH, oW)` containing the deformable convolution result. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `dilations` | — | Dilation factors for the spatial axes in height-width order; defaults to `[1, 1]`. |
| `group` | `1` | Number of groups the input channels C and output channels oC are divided into; both must be divisible by group. |
| `kernel_shape` | — | Kernel dimensions in height-width order. When omitted, they are inferred from the spatial dimensions of `W`. |
| `offset_group` | `1` | Number of offset groups; input channels `C` must be divisible by `offset_group`. |
| `pads` | — | Padding at the beginning and end of each spatial axis in `[top, left, bottom, right]` order; defaults to zeros. |
| `strides` | — | Stride factors for the spatial axes in height-width order; defaults to `[1, 1]`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`deform-conv2d-aligned-groups.wgsl.jinja`](build/webgpu/deform-conv2d-aligned-groups.wgsl.jinja)
- [`deform-conv2d-nchw.wgsl.jinja`](build/webgpu/deform-conv2d-nchw.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.DeformConv", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 1, 1, 1] },
  w: { data: wData, shape: [1, 1, 1, 1] },
  offset: { data: offsetData, shape: [1, 2, 1, 1] },
});
```
