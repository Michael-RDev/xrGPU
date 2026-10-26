---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Slice

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Produces a slice of the input tensor along multiple axes, using `starts`, `ends`, `axes`, and `steps` to select a sub-tensor. Negative indices are resolved relative to the dimension size, and out-of-range values are clamped. Omitting `axes` defaults to all axes in order; omitting `steps` defaults to stride 1.

See the [ONNX `Slice` spec](https://onnx.ai/onnx/operators/onnx__Slice.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `data` | `T` | — | — | Tensor of data to extract slices from. | required |
| `starts` | `S` | `1` | — | 1-D tensor of starting indices for each axis in `axes`. | required |
| `ends` | `S` | `1` | — | 1-D tensor of ending indices (exclusive) for each axis in `axes`. | required |
| `axes` | `S` | `1` | — | Optional 1-D tensor of axes that `starts` and `ends` apply to; defaults to all axes if omitted. | optional |
| `steps` | `S` | `1` | — | Optional 1-D tensor of step sizes per axis; negative steps slice backward, defaults to 1. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `data` | — | Sliced data tensor. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool`, `int64` |
| `S` | `int32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`datamove-slice-block.wgsl.jinja`](build/webgpu/datamove-slice-block.wgsl.jinja)
- [`slice-rank2-single-axis-x4.wgsl.jinja`](build/webgpu/slice-rank2-single-axis-x4.wgsl.jinja)
- [`slice.wgsl.jinja`](build/webgpu/slice.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Slice", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({
  data: { data: dataData, shape: [2, 2] },
  starts: { data: startsData, shape: [1] },
  ends: { data: endsData, shape: [1] },
}, {
  outputs: { output: { shape: [1, 2], dtype: "float32" } },
});
```
