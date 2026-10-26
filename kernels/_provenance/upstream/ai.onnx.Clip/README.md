---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Clip

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Limits each element of the input tensor to the interval `[min, max]`, equivalent to `Min(max, Max(input, min))`. Each bound is an optional scalar input tensor; an omitted bound defaults to the input dtype's lowest or highest numeric value. When `min` exceeds `max`, all elements are set to `max`.

See the [ONNX `Clip` spec](https://onnx.ai/onnx/operators/onnx__Clip.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `input` | `T` | — | — | Input tensor whose elements are to be clipped. | required |
| `min` | `T` | `0` | `[]` | Scalar lower bound; elements below this value are replaced by it. | optional |
| `max` | `T` | `0` | `[]` | Scalar upper bound; elements above this value are replaced by it. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `input` | same as `input` | Output tensor with each element clipped to the specified interval. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32`, `int8`, `uint8` |

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Clip", { version: 1 });
const { output } = await kernel({ input: { data: inputData, shape: [3] } });
```
