---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.OneHot

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 11

## Description

Produces a one-hot tensor from an `indices` input: positions matching each index are filled with `on_value` and all other positions with `off_value`, where both are taken from the two-element `values` tensor `[off_value, on_value]`. The output rank is one greater than `indices`, with the new dimension of size `depth` inserted at the position given by `axis`; indices outside `[-depth, depth-1]` yield all-`off_value` rows.

See the [ONNX `OneHot` spec](https://onnx.ai/onnx/operators/onnx__OneHot.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `indices` | `I` | — | — | Integer or float index tensor; values outside `[-depth, depth-1]` produce all-`off_value` output rows. | required |
| `depth` | `D` | — | — | Scalar (or length-1 rank-1) tensor specifying the number of classes and the size of the one-hot dimension. | required |
| `values` | `T` | `1` | — | Rank-1 tensor of exactly two elements `[off_value, on_value]` giving the values written to inactive and active positions respectively. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | derived | — | One-hot tensor with rank equal to `rank(indices) + 1`, same element type as `values`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `-1` | Axis along which the one-hot dimension is inserted; default `-1` appends it as the last dimension. Negative values count from the back; accepted range is `[-r-1, r]` where `r = rank(indices)`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `I` | `float32`, `float16`, `int32`, `int16`, `int8`, `uint32`, `uint8` |
| `D` | `float32`, `float16`, `int32`, `int16`, `int8`, `uint32`, `uint8` |
| `T` | `float32`, `float16`, `int32`, `int16`, `int8`, `uint32`, `uint8`, `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`one-hot-fill.wgsl.jinja`](build/webgpu/one-hot-fill.wgsl.jinja)
- [`one-hot-last-axis-vec4.wgsl.jinja`](build/webgpu/one-hot-last-axis-vec4.wgsl.jinja)
- [`one-hot-scatter.wgsl.jinja`](build/webgpu/one-hot-scatter.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.OneHot", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({
  indices: { data: indicesData, shape: [1] },
  depth: { data: depthData, shape: [] },
  values: { data: valuesData, shape: [2] },
}, {
  outputs: { output: { shape: [1, 2], dtype: "float32" } },
});
```
