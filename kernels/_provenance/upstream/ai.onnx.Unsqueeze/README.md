---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Unsqueeze

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Inserts size-1 dimensions into a tensor at the specified axis positions. The `axes` input lists dimension indices in the output shape where new dimensions of size 1 are inserted; negative values count from the back of the output rank. The output `expanded` has the same data as `data` with rank increased by the number of axes.

See the [ONNX `Unsqueeze` spec](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `data` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | Original input tensor whose data is preserved in the output. | required |
| `axes` | `S` | `int32` | `1` | — | Logical int64 1-D tensor of axis indices in the output tensor at which to insert size-1 dimensions; negative values count from the back and use int32 WebGPU storage. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `expanded` | `T` | derived | — | Reshaped tensor with the same data as the input and size-1 dimensions inserted at the specified axes. | required |

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

- `expanded`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Unsqueeze", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { expanded } = await kernel({
  data: { data: dataData, shape: [] },
  axes: { data: axesData, shape: [1] },
}, {
  outputs: { expanded: { shape: [1], dtype: "float32" } },
});
```
