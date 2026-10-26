---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Compress

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 11

## Description

Selects slices from the input tensor along an axis where the corresponding `condition` element is true. If `axis` is omitted, the input is flattened and elements are selected by position. The condition may be shorter than the selected dimension; values beyond its length are discarded. The data-dependent output extent must equal the number of true entries in the inspected prefix.

See the [ONNX `Compress` spec](https://onnx.ai/onnx/operators/onnx__Compress.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `input` | `T` | — | — | Input tensor of rank `r >= 1` to select from. | required |
| `condition` | `C` | `1` | — | Rank-1 boolean mask indicating which slices or elements to select; may be shorter than the axis dimension, in which case trailing slices are discarded. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | derived | — | Selected slices with rank `r` when `axis` is specified, or rank 1 when the input is flattened. The selected dimension equals the number of true values in the inspected condition prefix. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | — | Axis along which to select slices; if omitted the input is flattened before selection. Negative values index from the end; accepted range is `[-r, r-1]`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool` |
| `C` | `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`compress-scatter.wgsl.jinja`](build/webgpu/compress-scatter.wgsl.jinja)
- [`compress.wgsl.jinja`](build/webgpu/compress.wgsl.jinja)
- [`scan-block-prefix-u32.wgsl.jinja`](build/webgpu/scan-block-prefix-u32.wgsl.jinja)
- [`scan-flags-block-exclusive.wgsl.jinja`](build/webgpu/scan-flags-block-exclusive.wgsl.jinja)

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

A `bool` input takes 0/1 values in a `Uint8Array` together with `dtype: "bool"`, since no typed-array class implies it.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Compress", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({
  input: { data: inputData, shape: [3, 2] },
  condition: { data: conditionData, shape: [5], dtype: "bool" },
}, {
  outputs: { output: { shape: [2], dtype: "float32" } },
});
```
