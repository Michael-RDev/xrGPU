---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.FastGelu

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Applies the GELU (Gaussian Error Linear Unit) activation using a tanh approximation: `Y = 0.5 * X * (1 + tanh(0.797885 * X + 0.035677 * X^3))`. An optional `bias` is added to `X` before the activation is computed. This WebGPU package implements float16 and float32; the schema-allowed double and bfloat16 types are not supported.

See the [ONNX Runtime `FastGelu` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FastGelu) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `X` | `T` | — | — | Values transformed by FastGelu after adding the optional `bias`. | required |
| `bias` | `T` | `1` | — | Optional 1-D bias added to `X` along the last dimension before the GELU activation. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `Y` | `T` | same as `X` | same as `X` | Output tensor after applying the GELU activation; same shape as `X`. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`elementwise-bias-gelu.wgsl.jinja`](build/webgpu/elementwise-bias-gelu.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.FastGelu", { version: 1 });
const { Y } = await kernel({ X: { data: XData, shape: [5] } });
```
