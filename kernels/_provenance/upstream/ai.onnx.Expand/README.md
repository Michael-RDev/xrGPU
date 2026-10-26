---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Expand

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Broadcasts `input` to a given `shape` following NumPy-style rules: dimensions are right-aligned, and corresponding dimensions must be equal or one of them must be 1. The output shape may differ from `shape` when a requested dimension is 1 or `shape` has fewer dimensions than `input`.

See the [ONNX `Expand` spec](https://onnx.ai/onnx/operators/onnx__Expand.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `input` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | Input tensor to broadcast. | required |
| `shape` | `S` | `uint32` | `1` | — | Logical int64 1-D tensor specifying the non-negative target shape; WebGPU stores it as uint32. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | derived | — | Output tensor with the broadcasted shape. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool`, `int64` |
| `S` | `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`expand-vec4.wgsl.jinja`](build/webgpu/expand-vec4.wgsl.jinja)
- [`expand.wgsl.jinja`](build/webgpu/expand.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `output`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Expand", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({
  input: { data: inputData, shape: [1, 4] },
  shape: { data: shapeData, shape: [2] },
}, {
  outputs: { output: { shape: [2, 4], dtype: "int16" } },
});
```
