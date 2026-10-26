---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Squeeze

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Removes single-dimensional entries from the shape of a tensor. If `axes` is provided, only those dimensions are removed; if omitted, all size-1 dimensions are removed. Selecting an axis whose size is not 1 raises an error.

See the [ONNX `Squeeze` spec](https://onnx.ai/onnx/operators/onnx__Squeeze.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `data` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | Input tensor with at least as many dimensions as the largest axis index specified. | required |
| `axes` | `S` | `int32` | `1` | — | Logical int64 1-D tensor of axis indices to squeeze; negative values count from the back, signed axes use int32 WebGPU storage, and omission squeezes all size-1 dimensions. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `squeezed` | `T` | — | — | Reshaped tensor with the same data as the input but with the specified size-1 dimensions removed. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32`, `int16`, `uint8`, `int8`, `bool`, `int64` |
| `S` | `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`datamove-elementwise-copy.wgsl.jinja`](build/webgpu/datamove-elementwise-copy.wgsl.jinja)
- [`datamove-flat-copy.wgsl.jinja`](build/webgpu/datamove-flat-copy.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `squeezed`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Squeeze", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { squeezed } = await kernel({ data: { data: dataData, shape: [1, 1, 1, 1] } }, {
  outputs: { squeezed: { shape: [], dtype: "float32" } },
});
```
