---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.MaxPool

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 12

## Description

Applies max pooling over a sliding kernel window on input tensor `X`, computing the maximum value within each window and writing it to output `Y`. Output spatial dimensions are determined by kernel size, strides, padding, and dilations; `ceil_mode` controls whether output size is rounded up or down.

See the [ONNX `MaxPool` spec](https://onnx.ai/onnx/operators/onnx__MaxPool.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor of shape `(N x C x D1 x ... x Dn)`; batch size `N`, channels `C`, followed by spatial dimensions. | required |

## Outputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | same as `x` | derived | Pooled output tensor with the same batch and channel dimensions as X but reduced spatial dimensions. | required |
| `indices` | `Indices` | `I` | `uint32` | same as `x` | derived | Optional logical int64 flat indices of the maximum values selected during pooling, with the same shape as Y; indices do not account for padding and use uint32 WebGPU storage. | optional |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `auto_pad` | `"NOTSET"` | Deprecated auto-padding mode: `NOTSET` (use explicit pads), `SAME_UPPER`, `SAME_LOWER` (pad so output size is `ceil(input / stride)`), or `VALID` (no padding). It cannot be used together with `pads`. |
| `ceil_mode` | `0` | When non-zero, use ceiling instead of floor when computing output spatial dimensions. |
| `dilations` | — | Dilation along each spatial axis. When omitted, every dilation is 1. |
| `kernel_shape` | — | Required kernel shape, with one positive value per spatial axis. |
| `pads` | — | Padding at the beginning and end of each spatial axis, ordered as `[begin_0, ..., begin_n, end_0, ..., end_n]`. When omitted, every pad is 0. |
| `storage_order` | `0` | Storage order of the Indices output tensor: 0 for row-major, 1 for column-major. |
| `strides` | — | Stride along each spatial axis. When omitted, every stride is 1. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int8`, `uint8` |
| `I` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `window_parallel_ncl1d` — Workgroup-cooperative max pooling over a 1-D window: one workgroup reduces a single output element's kernel window, its invocations striding the flattened taps. Chosen when the serial one-invocation-per-output route cannot fill the device and the window is long. NaNs propagate through the fold; an all-out-of-bounds window writes 0.
- `window_parallel_nchw2d` — Workgroup-cooperative max pooling over a 2-D window: one workgroup reduces a single output element's kernel window, its invocations striding the flattened taps. Chosen when the serial one-invocation-per-output route cannot fill the device and the window is long. NaNs propagate through the fold; an all-out-of-bounds window writes 0.
- `window_parallel_ncdhw3d` — Workgroup-cooperative max pooling over a 3-D window: one workgroup reduces a single output element's kernel window, its invocations striding the flattened taps. Chosen when the serial one-invocation-per-output route cannot fill the device and the window is long. NaNs propagate through the fold; an all-out-of-bounds window writes 0.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`max-pool2d-nchw-k3s2p1-vec4.wgsl.jinja`](build/webgpu/max-pool2d-nchw-k3s2p1-vec4.wgsl.jinja)
- [`max-pool2d-nchw-u32.wgsl.jinja`](build/webgpu/max-pool2d-nchw-u32.wgsl.jinja)
- [`max-pool3d-ncdhw-k5-tiled.wgsl.jinja`](build/webgpu/max-pool3d-ncdhw-k5-tiled.wgsl.jinja)
- [`pool-global-reduction.wgsl.jinja`](build/webgpu/pool-global-reduction.wgsl.jinja)
- [`pool-ncl1d-x4.wgsl.jinja`](build/webgpu/pool-ncl1d-x4.wgsl.jinja)
- [`pool-window-nd.wgsl.jinja`](build/webgpu/pool-window-nd.wgsl.jinja)
- [`pool-window-reduction.wgsl.jinja`](build/webgpu/pool-window-reduction.wgsl.jinja)
- [`pool-window-unroll.wgsl.jinja`](build/webgpu/pool-window-unroll.wgsl.jinja)
- [`pool2d-nchw-k2s2-vec4.wgsl.jinja`](build/webgpu/pool2d-nchw-k2s2-vec4.wgsl.jinja)
- [`pool2d-nchw-separable.wgsl.jinja`](build/webgpu/pool2d-nchw-separable.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.MaxPool", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [1, 1, 3] } }, {
  attrs: { kernel_shape: [2] },
});
```
