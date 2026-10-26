---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.NonZero

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 9

## Description

Returns the indices of all non-zero elements in `X`, in row-major order, as a 2-D tensor of shape `[rank(X), nnz]` where each column is an N-dimensional index. This follows ONNX `NonZero`, including scalar output shape `[0, N]` and logical int64 output type. Because every emitted coordinate is bounded by a WebGPU-addressable input dimension, the backend stores this logical int64 tensor losslessly as uint32. The exact data-dependent output shape must be supplied as output metadata. String, float64, 64-bit input, and other ONNX input types not listed below are unsupported.

See the [ONNX `NonZero` spec](https://onnx.ai/onnx/operators/onnx__NonZero.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor of any shape whose non-zero element indices are to be found. | required |

## Outputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `I` | `uint32` | `2` | — | Logical int64 tensor of shape `[rank(X), nnz]` containing the multi-dimensional indices of non-zero elements, one index per column. The WebGPU storage representation is uint32 because every coordinate is within an addressable input dimension. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool` |
| `I` | `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`nonzero-flag-block-scan-vec4.wgsl.jinja`](build/webgpu/nonzero-flag-block-scan-vec4.wgsl.jinja)
- [`nonzero-scatter-vec4.wgsl.jinja`](build/webgpu/nonzero-scatter-vec4.wgsl.jinja)
- [`nonzero.wgsl.jinja`](build/webgpu/nonzero.wgsl.jinja)
- [`scan-block-prefix-u32.wgsl.jinja`](build/webgpu/scan-block-prefix-u32.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.NonZero", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({ x: { data: xData, shape: [4] } }, {
  outputs: { y: { shape: [1, 2], dtype: "int64" } },
});
```
