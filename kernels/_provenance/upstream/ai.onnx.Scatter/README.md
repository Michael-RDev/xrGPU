---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Scatter

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 11

## Description

Copies `data` and overwrites positions identified by `indices` with values from `updates` along a single `axis`. For each element in `updates`, the index along `axis` comes from the corresponding `indices` value, while all other dimension indices come from the element's own position. Deprecated in favor of ScatterElements, which provides identical behavior.

See the [ONNX `Scatter` spec](https://onnx.ai/onnx/operators/onnx__Scatter.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `data` | `T` | — | — | The input tensor to copy; the output has the same shape. | required |
| `indices` | `I` | — | — | Integer tensor of the same rank as `data`, specifying the target index along `axis` for each update. | required |
| `updates` | `T` | — | — | Values to scatter into the output; must have the same rank and shape as `indices`. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `data` | same as `data` | Copy of `data` with entries overwritten at positions specified by `indices`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `0` | The axis of `data` along which scatter indices are applied. Negative values count from the end; defaults to `0` (the outer-most axis). |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `int16`, `int8`, `uint32`, `uint8`, `bool` |
| `I` | `int32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`scatter-flat-copy.wgsl.jinja`](build/webgpu/scatter-flat-copy.wgsl.jinja)
- [`scatter.wgsl.jinja`](build/webgpu/scatter.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Scatter", { version: 1 });
const { output } = await kernel({
  data: { data: dataData, shape: [1, 3] },
  indices: { data: indicesData, shape: [3, 3] },
  updates: { data: updatesData, shape: [3, 3] },
});
```
