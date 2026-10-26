---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.LpPool

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 18

## Description

Applies Lp pooling over a spatial input tensor by computing the Lp norm within each kernel window and writing the result to the output. Output spatial dimensions are determined by the kernel size, strides, padding, and `ceil_mode`; `p` controls which norm is used (e.g. `p=1` for sum-of-absolutes, `p=2` for Euclidean).

See the [ONNX `LpPool` spec](https://onnx.ai/onnx/operators/onnx__LpPool.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor of shape `(N x C x D1 x ... x Dn)`; for images the spatial axes are `H` and `W`. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | derived | Output tensor after Lp pooling; spatial dimensions vary with kernel, stride, and pad settings. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `auto_pad` | `"NOTSET"` | Deprecated auto-padding mode (`NOTSET`, `SAME_UPPER`, `SAME_LOWER`, or `VALID`). It cannot be used together with `pads`. |
| `ceil_mode` | `0` | When non-zero, uses `ceil` instead of `floor` to compute output spatial dimensions. |
| `dilations` | — | Dilation along each spatial axis. When omitted, every dilation is 1. |
| `kernel_shape` | — | Required kernel shape, with one positive value per spatial axis. |
| `p` | `2` | The exponent of the Lp norm used for pooling; default `2` gives Euclidean (L2) pooling. |
| `pads` | — | Padding at the beginning and end of each spatial axis, ordered as `[begin_0, ..., begin_n, end_0, ..., end_n]`. When omitted, every pad is 0. |
| `strides` | — | Stride along each spatial axis. When omitted, every stride is 1. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `window_parallel_ncl1d` — Workgroup-cooperative Lp pooling over a 1-D window: one workgroup reduces a single output element's kernel window, its invocations striding the flattened taps. Chosen when the serial one-invocation-per-output route cannot fill the device and the window is long. The `p`-root is applied once, by the invocation that writes the output.
- `window_parallel_nchw2d` — Workgroup-cooperative Lp pooling over a 2-D window: one workgroup reduces a single output element's kernel window, its invocations striding the flattened taps. Chosen when the serial one-invocation-per-output route cannot fill the device and the window is long. The `p`-root is applied once, by the invocation that writes the output.
- `window_parallel_ncdhw3d` — Workgroup-cooperative Lp pooling over a 3-D window: one workgroup reduces a single output element's kernel window, its invocations striding the flattened taps. Chosen when the serial one-invocation-per-output route cannot fill the device and the window is long. The `p`-root is applied once, by the invocation that writes the output.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.LpPool", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [1, 1, 4] } }, {
  attrs: { kernel_shape: [3] },
});
```
