---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Tile

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Constructs a tensor by tiling a given tensor: each dimension `i` of the input is repeated `repeats[i]` times, so `output_dim[i] = input_dim[i] * repeats[i]`. Equivalent to NumPy `tile` but without broadcasting.

See the [ONNX `Tile` spec](https://onnx.ai/onnx/operators/onnx__Tile.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `input` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | Input tensor of any shape. | required |
| `repeats` | `S` | `uint32` | `1` | — | Logical int64 1-D tensor of length equal to the input rank, specifying non-negative repeat counts stored as uint32 by WebGPU. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `input` | — | Output tensor of the same type as the input, with each dimension scaled by the corresponding repeat count. | required |

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
- [`datamove-tile-vec4.wgsl.jinja`](build/webgpu/datamove-tile-vec4.wgsl.jinja)
- [`tile.wgsl.jinja`](build/webgpu/tile.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Tile", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({
  input: { data: inputData, shape: [3] },
  repeats: { data: repeatsData, shape: [1] },
}, {
  outputs: { output: { shape: [6], dtype: "int64" } },
});
```
