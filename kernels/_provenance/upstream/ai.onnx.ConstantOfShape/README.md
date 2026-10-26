---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.ConstantOfShape

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 9

## Description

Generates a tensor filled with a constant value and an input-specified shape. The shape is provided as a 1-D integer tensor; if empty, the output is a scalar. The fill value and its dtype are taken from the `value` attribute, defaulting to `0` of type float32.

See the [ONNX `ConstantOfShape` spec](https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `input` | `S` | `uint32` | `1` | — | Logical int64 1-D tensor whose elements define the output shape; all values must be non-negative and use uint32 WebGPU storage. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | derived | — | Output tensor of the shape given by `input`, filled with the constant `value`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `value` | — | Optional single-element tensor specifying the fill value and output dtype; defaults to 0 of type float32 if omitted. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `S` | `int64` |
| `T` | `float32`, `float16`, `int32`, `int16`, `uint32`, `int8`, `uint8`, `bool` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `fill_float` — Fills with vec4 stores when the output has at least one full vector; a count not divisible by four adds a one-invocation scalar tail pass.
- `fill_integer` — Fills with vec4 stores when the output has at least one full vector; a count not divisible by four adds a one-invocation scalar tail pass.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`constant-of-shape.wgsl.jinja`](build/webgpu/constant-of-shape.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.ConstantOfShape", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({ input: { data: inputData, shape: [2] } }, {
  outputs: { output: { shape: [2, 3], dtype: "float32" } },
});
```
