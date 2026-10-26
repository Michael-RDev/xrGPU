---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.AveragePool

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 19

## Description

Applies average pooling over a sliding kernel window on input tensor `X`, computing the mean of values within each window position and writing results to `Y`. Output spatial dimensions are determined by `kernel_shape`, `strides`, `dilations`, `pads`, and `ceil_mode`; padded positions are excluded from the average by default unless `count_include_pad` is set.

See the [ONNX `AveragePool` spec](https://onnx.ai/onnx/operators/onnx__AveragePool.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input data tensor of shape `(N x C x D1 x D2 ... Dn)`, where `N` is the batch size and `C` is the number of channels. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | derived | Output tensor with pooled values; spatial dimensions vary based on kernel, stride, dilation, and pad settings. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `auto_pad` | `"NOTSET"` | Deprecated auto-padding mode: `NOTSET` (use explicit pads), `SAME_UPPER`, `SAME_LOWER` (pad so output size is `ceil(input / stride)`), or `VALID` (no padding). |
| `ceil_mode` | `0` | When non-zero, uses ceiling instead of floor when computing the output spatial shape; defaults to 0. |
| `count_include_pad` | `0` | When non-zero, pad pixels are counted in the divisor when computing the average; defaults to 0 (exclude pad). |
| `dilations` | — | Dilation along each spatial axis. When omitted, every dilation is 1. |
| `kernel_shape` | — | Required kernel shape, with one positive value per spatial axis. |
| `pads` | — | Padding at the beginning and end of each spatial axis, ordered as `[begin_0, ..., begin_n, end_0, ..., end_n]`. When omitted, every pad is 0. |
| `strides` | — | Stride along each spatial axis. When omitted, every stride is 1. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `window_parallel_ncl1d` — Workgroup-cooperative average pooling over a 1-D window: one workgroup reduces a single output element's kernel window, its invocations striding the flattened taps. Chosen when the serial one-invocation-per-output route cannot fill the device and the window is long. Padded taps leave the divisor unless `count_include_pad` is set.
- `window_parallel_nchw2d` — Workgroup-cooperative average pooling over a 2-D window: one workgroup reduces a single output element's kernel window, its invocations striding the flattened taps. Chosen when the serial one-invocation-per-output route cannot fill the device and the window is long. Padded taps leave the divisor unless `count_include_pad` is set.
- `window_parallel_ncdhw3d` — Workgroup-cooperative average pooling over a 3-D window: one workgroup reduces a single output element's kernel window, its invocations striding the flattened taps. Chosen when the serial one-invocation-per-output route cannot fill the device and the window is long. Padded taps leave the divisor unless `count_include_pad` is set.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`average-pool2d-nchw-horizontal-reuse.wgsl.jinja`](build/webgpu/average-pool2d-nchw-horizontal-reuse.wgsl.jinja)
- [`average-pool2d-nchw-w3s1-reuse.wgsl.jinja`](build/webgpu/average-pool2d-nchw-w3s1-reuse.wgsl.jinja)
- [`pool-global-reduction.wgsl.jinja`](build/webgpu/pool-global-reduction.wgsl.jinja)
- [`pool-window-nd.wgsl.jinja`](build/webgpu/pool-window-nd.wgsl.jinja)
- [`pool-window-reduction.wgsl.jinja`](build/webgpu/pool-window-reduction.wgsl.jinja)
- [`pool-window-unroll.wgsl.jinja`](build/webgpu/pool-window-unroll.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.AveragePool", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [1, 3, 32] } }, {
  attrs: { kernel_shape: [2] },
});
```
