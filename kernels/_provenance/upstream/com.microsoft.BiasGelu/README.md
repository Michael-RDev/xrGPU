---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.BiasGelu

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Applies GELU to `A + B`, where the 1-D bias `B` is broadcast along the last dimension of `A`. This implementation supports float16 and float32; the schema's double and bfloat16 types are not implemented.

See the [ONNX Runtime `BiasGelu` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasGelu) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `A` | `T` | — | — | The main input tensor of any shape. | required |
| `B` | `T` | `1` | — | 1-D bias tensor whose length equals the last dimension of `A`. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `C` | `T` | same as `A` | same as `A` | Output tensor after applying GELU to `A + B`; same shape as `A`. | required |

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.BiasGelu", { version: 1 });
const { C } = await kernel({ A: { data: AData, shape: [3] }, B: { data: BData, shape: [3] } });
```
