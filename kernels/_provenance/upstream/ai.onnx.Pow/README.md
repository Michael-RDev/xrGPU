---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Pow

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 15

## Description

Computes elementwise exponentiation `Z = X ^ Y`, applying `f(x) = x^y` to each element. Supports multidirectional (NumPy-style) broadcasting between `X` and `Y`.

See the [ONNX `Pow` spec](https://onnx.ai/onnx/operators/onnx__Pow.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Base tensor; first operand of the exponentiation. | required |
| `y` | `Y` | `U` | — | — | Exponent tensor; power applied to each base element. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `z` | `Z` | `T` | derived | broadcast result of `x` and `y` | Output tensor containing `X` raised to the power `Y`, same type as `X`. | required |

## Runtime arguments

| Name | Kind | Description | Presence |
| --- | --- | --- | --- |
| `constantY` | `f32` | Optional host-known constant exponent. When the exponent tensor `y` holds one integer value in `[-16, 16]` that the caller also passes here, a specialized integer-power kernel is selected. | optional |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32` |
| `U` | `float32`, `float16`, `int32`, `int16`, `uint32`, `int8`, `uint8` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`pow-scalar-exponent-vec4.wgsl.jinja`](build/webgpu/pow-scalar-exponent-vec4.wgsl.jinja)
- [`pow-vec4.wgsl.jinja`](build/webgpu/pow-vec4.wgsl.jinja)
- [`pow.wgsl.jinja`](build/webgpu/pow.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Pow", { version: 1 });
const { z } = await kernel({ x: { data: xData, shape: [1] }, y: { data: yData, shape: [1] } });
```
