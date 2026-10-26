---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.CumProd

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 26

## Description

Computes the cumulative product of input elements along a given `axis`. By default the product is inclusive (each output element is the product of all preceding elements including itself); setting `exclusive` to 1 shifts the window so the current element is excluded. Setting `reverse` to 1 performs the scan in the opposite direction along the axis.

See the [ONNX `CumProd` spec](https://onnx.ai/onnx/operators/onnx__CumProd.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `x` | `T` | — | — | Input tensor to be scanned along the specified axis. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `y` | `T` | same as `x` | same as `x` | Output tensor of the same shape and type as `x` containing the cumulative products. | required |

## Runtime arguments

| Name | Kind | Description | Presence |
| --- | --- | --- | --- |
| `axis` | `i32` | Axis along which to accumulate; negative values count from the back. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `exclusive` | `0` | If 1, returns the exclusive product: each output element is the product of all elements before it (not including itself); if 0, the current element is included. |
| `reverse` | `0` | If 1, performs the cumulative product in the reverse direction along the axis; if 0, proceeds in the forward direction. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`cumprod-tiled-short-rows.wgsl.jinja`](build/webgpu/cumprod-tiled-short-rows.wgsl.jinja)
- [`scan-row-chunked-vec4.wgsl.jinja`](build/webgpu/scan-row-chunked-vec4.wgsl.jinja)
- [`scan-row-chunked.wgsl.jinja`](build/webgpu/scan-row-chunked.wgsl.jinja)
- [`scan-thread-row.wgsl.jinja`](build/webgpu/scan-thread-row.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.CumProd", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [3] }, axis: 0 });
```
