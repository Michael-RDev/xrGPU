---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.CumSum

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 14

## Description

Computes the cumulative sum of elements along a specified axis. By default, the sum is inclusive (each output element includes itself); setting `exclusive` to 1 excludes the current element, and setting `reverse` to 1 scans in the opposite direction.

See the [ONNX `CumSum` spec](https://onnx.ai/onnx/operators/onnx__CumSum.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `x` | `T` | — | — | Input tensor to accumulate. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `y` | `T` | same as `x` | same as `x` | Output tensor of the same shape and type as `x`, containing cumulative sums along the specified axis. | required |

## Runtime arguments

| Name | Kind | Description | Presence |
| --- | --- | --- | --- |
| `axis` | `i32` | Axis along which to accumulate; negative values count from the back. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `exclusive` | `0` | If 1, the j-th output element is the sum of the first (j-1) input elements, excluding the j-th element itself; defaults to 0 (inclusive). |
| `reverse` | `0` | If 1, performs the cumulative sum scanning in the reverse direction along the axis; defaults to 0. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `single_segment_row_scan_vec4` — A single computed segment needs no cross-segment carry. Reuses the cooperative scan with an identity seed, omitting preparatory passes and scratch storage.
- `single_segment_row_scan` — A single computed segment needs no cross-segment carry. Reuses the cooperative scan with an identity seed, omitting preparatory passes and scratch storage.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`cumsum-splitk-carry.wgsl.jinja`](build/webgpu/cumsum-splitk-carry.wgsl.jinja)
- [`cumsum-splitk-scan.wgsl.jinja`](build/webgpu/cumsum-splitk-scan.wgsl.jinja)
- [`cumsum-splitk-totals.wgsl.jinja`](build/webgpu/cumsum-splitk-totals.wgsl.jinja)
- [`scan-row-chunked-vec4.wgsl.jinja`](build/webgpu/scan-row-chunked-vec4.wgsl.jinja)
- [`scan-row-chunked.wgsl.jinja`](build/webgpu/scan-row-chunked.wgsl.jinja)
- [`scan-strided-row.wgsl.jinja`](build/webgpu/scan-strided-row.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.CumSum", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [5] }, axis: 0 });
```
