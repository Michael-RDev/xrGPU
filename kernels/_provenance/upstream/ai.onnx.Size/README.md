---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Size

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Returns the total number of elements in a tensor as a logical int64 scalar equal to the product of the input dimensions. The input may have any shape; supported element types are listed below.

See the [ONNX `Size` spec](https://onnx.ai/onnx/operators/onnx__Size.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `data` | `T` | — | — | Input tensor of arbitrary shape. | required |

## Outputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `size` | `S` | `uint32` | `0` | `[]` | Logical int64 scalar holding the total number of elements in the input tensor; WebGPU emits the bounded value as uint32. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32`, `int16`, `int8`, `uint8`, `bool` |
| `S` | `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`size.wgsl.jinja`](build/webgpu/size.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Size", { version: 1 });
const { size } = await kernel({ data: { data: dataData, shape: [2, 3] } });
```
