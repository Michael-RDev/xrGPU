---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Shape

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 15

## Description

Returns a 1-D tensor containing the shape of the input tensor. Optional `start` and `end` attributes select a slice of the shape axes; negative values count from the back, and axes are clamped to `[0, rank]`.

See the [ONNX `Shape` spec](https://onnx.ai/onnx/operators/onnx__Shape.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `data` | `T` | — | — | The input tensor whose shape is computed. | required |

## Outputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `shape` | `S` | `uint32` | `1` | derived | Logical int64 1-D tensor of dimension sizes for the selected axes of the input; WebGPU emits bounded uint32 values. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `end` | — | Last axis (exclusive) of the shape slice; negative values count from the back; if omitted, all axes through the last are included. |
| `start` | `0` | First axis (inclusive) of the shape slice; negative values count from the back, default is 0. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `int16`, `uint32`, `int8`, `uint8`, `bool` |
| `S` | `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`shape.wgsl.jinja`](build/webgpu/shape.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Shape", { version: 1 });
const { shape } = await kernel({ data: { data: dataData, shape: [1, 2, 2] } });
```
