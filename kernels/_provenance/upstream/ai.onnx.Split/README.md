---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Split

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 18

## Description

Splits a tensor into a list of tensors along the specified `axis`. The sizes of each output slice are given by the optional `split` input; if omitted, the tensor is divided into equal parts (the last chunk may be smaller if the axis dimension is not evenly divisible).

See the [ONNX `Split` spec](https://onnx.ai/onnx/operators/onnx__Split.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `input` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | The tensor to split. | required |
| `split` | `S` | `uint32` | `1` | — | Optional logical int64 1-D tensor specifying the size of each output along the split axis; values must be non-negative, sum to the axis dimension, and use uint32 WebGPU storage. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y0` | `Y0` | `T` | same as `input` | — | First output slice after splitting. | required |
| `y1` | `Y1` | `T` | same as `input` | — | Second output slice after splitting (optional). | optional |
| `y2` | `Y2` | `T` | same as `input` | — | Third output slice after splitting. | optional |
| `y3` | `Y3` | `T` | same as `input` | — | Fourth output slice after splitting. | optional |
| `y4` | `Y4` | `T` | same as `input` | — | Fifth output slice after splitting. | optional |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `0` | The axis along which to split. Negative values count from the end; accepted range is `[-rank, rank-1]`. |
| `num_outputs` | — | Optional number of outputs when the `split` input is omitted. The final output may be smaller when the axis dimension is not evenly divisible. |

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
- [`datamove-flat-copy.wgsl.jinja`](build/webgpu/datamove-flat-copy.wgsl.jinja)
- [`datamove-split-block.wgsl.jinja`](build/webgpu/datamove-split-block.wgsl.jinja)
- [`split-n.wgsl.jinja`](build/webgpu/split-n.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `y0`
- `y1`
- `y2`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Split", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y0, y1, y2 } = await kernel({ input: { data: inputData, shape: [6] } }, {
  outputs: {
    y0: { shape: [2], dtype: "float32" },
    y1: { shape: [2], dtype: "float32" },
    y2: { shape: [2], dtype: "float32" },
  },
});
```
