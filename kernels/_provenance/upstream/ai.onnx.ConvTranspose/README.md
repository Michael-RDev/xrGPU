---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.ConvTranspose

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 11

## Description

Computes the transpose of a convolution, also known as a fractionally strided convolution or deconvolution, from input tensor `X`, filter weights `W`, and an optional bias `B`. Output spatial dimensions follow the stride, dilation, padding, and optional `output_padding` attributes. Supports grouped convolution through `group`.

See the [ONNX `ConvTranspose` spec](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input data tensor of shape `(N x C x D1 x ... x Dn)`, where `N` is batch size and `C` is the number of input channels. | required |
| `w` | `W` | `T` | — | — | Filter weight tensor of shape `(C x M/group x k1 x ... x kn)`, where `M` is the number of output feature maps. | required |
| `bias` | `B` | `T` | `1` | — | Optional 1-D bias of length M added to each output channel. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | derived | Output tensor whose spatial dimensions are computed from the input size, kernel shape, strides, dilations, and padding. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `auto_pad` | `"NOTSET"` | Padding mode: `NOTSET` uses explicit pads; `SAME_UPPER` and `SAME_LOWER` make output spatial size equal input size times stride, with any odd extra padding added at the end or beginning respectively; `VALID` applies no padding. |
| `dilations` | — | Dilation factors for each spatial axis; defaults to one on every axis. |
| `group` | `1` | Number of groups that input and output channels are divided into for grouped (depthwise) convolution. |
| `kernel_shape` | — | Kernel dimensions for each spatial axis. When omitted, they are inferred from the spatial dimensions of `W`. |
| `output_padding` | — | Additional size on the high-index end of each output spatial axis; each value must be smaller than the corresponding stride or dilation. |
| `output_shape` | — | Requested output spatial dimensions. When present, it must match the declared spatial shape of `Y`. |
| `pads` | — | Padding at the beginning of every spatial axis followed by padding at the end of every spatial axis; defaults to zeros. |
| `strides` | — | Stride factors for each spatial axis; defaults to one on every axis. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `ncdhw3d_implicit_sgmat` — Stride-1 transposed convolution as a forward convolution of the unchanged input with tap-reversed weights and begin pads of kernel-1-pad: the implicit-gather GEMM reads input windows directly and stores output elements directly, so no column matrix is materialized and no scatter pass runs, while the reduction spans the full in-channel-by-tap depth.
- `ncdhw3d_implicit_sgmat_bias` — Stride-1 transposed convolution as a forward convolution of the unchanged input with tap-reversed weights and begin pads of kernel-1-pad: the implicit-gather GEMM reads input windows directly and stores output elements directly, so no column matrix is materialized and no scatter pass runs, while the reduction spans the full in-channel-by-tap depth.
- `nchw2d_grouped_stride_phase_blocked` — Grouped stride-phase ConvTranspose where one invocation owns one output pixel and a block of that group's output channels: every gathered input sample feeds the whole block and the group's weight slab is staged once per workgroup. It serves grouped stride-two-or-more shapes whose per-group weights fit workgroup memory.
- `nchw2d_grouped_stride_phase_blocked_bias` — Grouped stride-phase ConvTranspose where one invocation owns one output pixel and a block of that group's output channels: every gathered input sample feeds the whole block and the group's weight slab is staged once per workgroup. It serves grouped stride-two-or-more shapes whose per-group weights fit workgroup memory.
- `nchw2d_exact_store_gemm_f32` — Exact-tiling f32 transposed convolution retains the register GEMM contraction and stores directly into output, removing the column intermediate and gather pass.
- `nchw2d_exact_store_gemm_f32_bias` — Exact-tiling f32 transposed convolution retains the register GEMM contraction and stores directly into output, removing the column intermediate and gather pass.

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
- [`conv-transpose-empty-input.wgsl.jinja`](build/webgpu/conv-transpose-empty-input.wgsl.jinja)
- [`conv-transpose1d-ncl.wgsl.jinja`](build/webgpu/conv-transpose1d-ncl.wgsl.jinja)
- [`conv-transpose1d-phase-tiled.wgsl.jinja`](build/webgpu/conv-transpose1d-phase-tiled.wgsl.jinja)
- [`conv-transpose2d-col2im.wgsl.jinja`](build/webgpu/conv-transpose2d-col2im.wgsl.jinja)
- [`conv-transpose2d-grouped-stride-phase-blocked.wgsl.jinja`](build/webgpu/conv-transpose2d-grouped-stride-phase-blocked.wgsl.jinja)
- [`conv-transpose2d-grouped-stride-phase.wgsl.jinja`](build/webgpu/conv-transpose2d-grouped-stride-phase.wgsl.jinja)
- [`conv-transpose2d-nchw.wgsl.jinja`](build/webgpu/conv-transpose2d-nchw.wgsl.jinja)
- [`conv-transpose2d-phase-gemm-tiled.wgsl.jinja`](build/webgpu/conv-transpose2d-phase-gemm-tiled.wgsl.jinja)
- [`conv-transpose2d-weight-reorder.wgsl.jinja`](build/webgpu/conv-transpose2d-weight-reorder.wgsl.jinja)
- [`conv-transpose3d-col2im.wgsl.jinja`](build/webgpu/conv-transpose3d-col2im.wgsl.jinja)
- [`conv-transpose3d-ncdhw.wgsl.jinja`](build/webgpu/conv-transpose3d-ncdhw.wgsl.jinja)
- [`conv-transpose3d-weight-reorder.wgsl.jinja`](build/webgpu/conv-transpose3d-weight-reorder.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.ConvTranspose", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 1, 3] },
  w: { data: wData, shape: [1, 2, 2] },
});
```
