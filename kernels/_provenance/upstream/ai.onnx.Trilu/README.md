---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Trilu

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 14

## Description

Returns the upper or lower triangular part of a 2-D matrix or batch of 2-D matrices of shape `[*, N, M]`, zeroing all other elements. The diagonal offset `k` shifts the boundary: positive values move it above the main diagonal, negative values below it.

See the [ONNX `Trilu` spec](https://onnx.ai/onnx/operators/onnx__Trilu.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `input` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | Input tensor of rank 2 or higher whose triangular part is extracted. | required |
| `k` | `I` | `int32` | `0` | `[]` | Optional logical int64 scalar specifying the signed diagonal offset; WebGPU stores it as int32 and defaults to 0 (main diagonal). | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `input` | same as `input` | Output tensor of the same type and shape as the input, with non-triangular elements set to zero. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `upper` | `1` | When 1 (true), retains the upper triangular part; when 0 (false), retains the lower triangular part. Default is 1. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `int16`, `int8`, `uint32`, `uint8`, `bool` |
| `I` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `mask_inner_packed_vec8_kinput` — Processes eight `float16` elements as four 32-bit storage words. Integer masks preserve retained bits and zero discarded halves; fully masked vectors skip the input load.
- `mask_inner_packed_vec8` — Processes eight `float16` elements as four 32-bit storage words. Integer masks preserve retained bits and zero discarded halves; fully masked vectors skip the input load.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`trilu-packed.wgsl.jinja`](build/webgpu/trilu-packed.wgsl.jinja)
- [`trilu.wgsl.jinja`](build/webgpu/trilu.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Required output shapes and logical data types are inferred from the supplied inputs and attributes; result tensors are allocated automatically.

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Trilu", { version: 1 });
const { output } = await kernel({ input: { data: inputData, shape: [2, 2] } });
```
