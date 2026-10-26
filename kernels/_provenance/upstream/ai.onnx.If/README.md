---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.If

`ai.onnx`  ·  internal tensor lowering (non-standard)  ·  reviewed against ONNX opset 25

## Description

Support status: the standard ONNX `If` control-flow operator is not implemented because standalone kernel packages cannot carry or execute its `then_branch` and `else_branch` graph attributes. This internal lowering only selects elementwise between two pre-evaluated, equal-sized tensors from a scalar condition and must not be treated as ONNX `If`.

See the [standard ONNX `If` spec](https://onnx.ai/onnx/operators/onnx__If.html) for the contract this internal lowering does not implement.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `cond` | `B` | — | — | Scalar boolean condition that selects which value tensor to output. | required |
| `then_value` | `T` | — | — | Values to output when `cond` is true. | required |
| `else_value` | `T` | — | — | Values to output when `cond` is false. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `y` | `T` | — | — | Output tensor with the same number of elements as `then_value` and `else_value`. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32` |
| `B` | `uint32`, `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`if-select.wgsl.jinja`](build/webgpu/if-select.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.If", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({
  cond: { data: condData, shape: [1] },
  then_value: { data: then_valueData, shape: [1] },
  else_value: { data: else_valueData, shape: [1] },
}, {
  outputs: { y: { shape: [1], dtype: "float32" } },
});
```
