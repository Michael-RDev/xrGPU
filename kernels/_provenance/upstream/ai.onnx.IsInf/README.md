---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.IsInf

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 20

## Description

Maps each element of a floating-point tensor to `true` if it is infinite and `false` otherwise. The `detect_negative` and `detect_positive` attributes independently control whether negative and positive infinity are considered infinite for this mapping.

See the [ONNX `IsInf` spec](https://onnx.ai/onnx/operators/onnx__IsInf.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input floating-point tensor. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `B` | same as `x` | same as `x` | Boolean output tensor with the same shape as the input; `true` where the input is infinite. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `detect_negative` | `1` | When 1 (default), negative infinity maps to true; set to 0 to map negative infinity to false. |
| `detect_positive` | `1` | When 1 (default), positive infinity maps to true; set to 0 to map positive infinity to false. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `B` | `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`detect-neither.wgsl.jinja`](build/webgpu/detect-neither.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.IsInf", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [1, 1, 1, 1, 1, 1, 3] } });
```
