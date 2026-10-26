---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Reshape

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 14

## Description

Reshapes the input tensor while preserving its elements. The target shape—including `-1`, zero dimensions, and `allowzero`—must be resolved and validated before `reshaped` is allocated; input and output element counts must match.

See the [ONNX `Reshape` spec](https://onnx.ai/onnx/operators/onnx__Reshape.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `data` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | The input tensor to reshape. | required |
| `shape` | `S` | `int32` | `1` | — | Logical int64 1-D ONNX target-shape tensor, stored as int32 by WebGPU so `-1` remains representable. The values must be validated and `reshaped` allocated with the resolved shape before dispatch. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `reshaped` | `T` | derived | — | The output tensor with the resolved target shape and the same elements and data type as the input. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `allowzero` | `0` | Controls target-shape resolution: `0` copies the corresponding input dimension for a zero entry, while `1` preserves a literal zero dimension. The standalone kernel does not interpret this attribute or the `shape` tensor. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `int16`, `uint32`, `uint8`, `int8`, `bool`, `int64` |
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

- `reshaped`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Reshape", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { reshaped } = await kernel({
  data: { data: dataData, shape: [2] },
  shape: { data: shapeData, shape: [2] },
}, {
  outputs: { reshaped: { shape: [1, 2], dtype: "int8" } },
});
```
