---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Upsample

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 9

## Description

Upsamples the input by applying a per-dimension scale factor; each output dimension equals `floor(input_dimension * scale)`. Deprecated in favor of Resize; supports `nearest` and `linear` interpolation modes.

See the [ONNX `Upsample` spec](https://onnx.ai/onnx/operators/onnx__Upsample.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor to upsample. | required |
| `scales` | — | `S` | `1` | — | Per-dimension scale factors, one value per input dimension. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | — | Upsampled output tensor; each dimension is `floor(input_dimension * scale)`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `mode` | `"nearest"` | Interpolation algorithm to use when mapping output coordinates back to input values; either `"nearest"` or `"linear"`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `int8`, `uint8` |
| `S` | `float32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`resize-coord-transform.wgsl.jinja`](build/webgpu/resize-coord-transform.wgsl.jinja)
- [`resize-generic.wgsl.jinja`](build/webgpu/resize-generic.wgsl.jinja)
- [`resize-linear-2x-stencil.wgsl.jinja`](build/webgpu/resize-linear-2x-stencil.wgsl.jinja)
- [`resize-nearest-integer-scale.wgsl.jinja`](build/webgpu/resize-nearest-integer-scale.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `y`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Upsample", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({
  x: { data: xData, shape: [1, 1, 1, 2] },
  scales: { data: scalesData, shape: [4] },
}, {
  outputs: { y: { shape: [1, 1, 1, 4], dtype: "float32" } },
});
```
