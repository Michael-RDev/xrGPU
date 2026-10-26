---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.SplitToSequence

`ai.onnx`  ·  internal tensor lowering (non-standard)  ·  reviewed against ONNX opset 11

## Description

Internal fixed-output split lowering that returns two, three, four, or six tensors with invocation-supplied shapes. It does not construct an ONNX sequence value.

See the [standard ONNX `SplitToSequence` spec](https://onnx.ai/onnx/operators/onnx__SplitToSequence.html) for the contract this internal lowering does not implement.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `input` | `T` | — | — | The tensor to split. | required |
| `split` | `S` | — | — | Length of each output slice: a scalar for uniform chunks or a 1-D tensor of per-output lengths. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y0` | `Y0` | `T` | derived | — | First output tensor slice. | required |
| `y1` | `Y1` | `T` | derived | — | Second output tensor slice. | required |
| `y2` | `Y2` | `T` | derived | — | Third output tensor slice. | optional |
| `y3` | `Y3` | `T` | derived | — | Fourth output tensor slice. | optional |
| `y4` | `Y4` | `T` | derived | — | Fifth output tensor slice. | optional |
| `y5` | `Y5` | `T` | derived | — | Sixth output tensor slice. | optional |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `0` | Axis along which to split; negative values count from the back. Accepted range is `[-rank, rank-1]`. |
| `keepdims` | `1` | Whether to keep the split dimension in the output (default `1`). Ignored when `split` is provided. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `bool` |
| `S` | `uint32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`split-to-sequence.wgsl.jinja`](build/webgpu/split-to-sequence.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `y0`
- `y1`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.SplitToSequence", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y0, y1 } = await kernel({ input: { data: inputData, shape: [3, 2] } }, {
  outputs: {
    y0: { shape: [1, 2], dtype: "float32" },
    y1: { shape: [1, 2], dtype: "float32" },
  },
});
```
