---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Asinh

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 9

## Description

Computes the hyperbolic arcsine (`asinh`) of each element in the input tensor. The output has the same shape as the input.

See the [ONNX `Asinh` spec](https://onnx.ai/onnx/operators/onnx__Asinh.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `input` | `T` | — | — | Values to map elementwise through inverse hyperbolic sine. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `output` | `T` | same as `x` | same as `x` | The hyperbolic arcsine of each input element, same shape as the input. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Asinh", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [3] } });
```
