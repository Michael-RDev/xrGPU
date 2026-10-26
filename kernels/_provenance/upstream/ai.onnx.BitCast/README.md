---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.BitCast

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 26

## Description

Reinterprets the raw bit pattern of a tensor as a different data type without any value conversion. The target type must have the same bit-width as the input type, and the output tensor has the same shape as the input.

See the [ONNX `BitCast` spec](https://onnx.ai/onnx/operators/onnx__BitCast.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `input` | `T` | — | — | Input tensor to be bitwise reinterpreted. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `U` | same as `input` | same as `input` | Output tensor with the same shape as the input, reinterpreted as the target type. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `to` | — | Required TensorProto DataType enum integer naming the output dtype; the target type must have the same bit-width as the input type. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `int8`, `int32`, `uint8`, `uint32` |
| `U` | `float32`, `int8`, `int32`, `uint8`, `uint32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`bitcast.wgsl.jinja`](build/webgpu/bitcast.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.BitCast", { version: 1 });
const { output } = await kernel({ input: { data: inputData, shape: [3] } }, {
  attrs: { to: 1 },
});
```
