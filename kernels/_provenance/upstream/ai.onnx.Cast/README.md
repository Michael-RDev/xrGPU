---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Cast

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Casts every element of the input tensor to a supported target numeric dtype, producing an output of the same shape. A conversion may change values, for example when narrowing an integer or converting a float to boolean.

See the [ONNX `Cast` spec](https://onnx.ai/onnx/operators/onnx__Cast.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `input` | `T` | — | — | Input tensor to be cast. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `output` | `U` | same as `x` | same as `x` | Output tensor with the same shape as the input, with elements converted to the target type. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `to` | — | Required TensorProto DataType enum integer naming the output element type. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint32`, `int32`, `uint8`, `int8`, `bool` |
| `U` | `float32`, `float16`, `uint32`, `int32`, `uint8`, `int8`, `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`cast-scalar-x4.wgsl.jinja`](build/webgpu/cast-scalar-x4.wgsl.jinja)
- [`unary-scalar.wgsl.jinja`](build/webgpu/unary-scalar.wgsl.jinja)
- [`unary-vec4.wgsl.jinja`](build/webgpu/unary-vec4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Cast", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [5] } }, {
  attrs: { to: 6 },
});
```
