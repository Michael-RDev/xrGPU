---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Mod

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Performs elementwise binary modulo on tensors `A` and `B` with multidirectional broadcasting. When `fmod` is `0` (default), applies Python-style `%` with the sign of the divisor; when `fmod` is `1`, applies C-style `fmod` with the sign of the dividend.

See the [ONNX `Mod` spec](https://onnx.ai/onnx/operators/onnx__Mod.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `T` | — | — | Dividend tensor. | required |
| `b` | `B` | `T` | — | — | Divisor tensor. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `c` | `C` | `T` | derived | broadcast result of `a` and `b` | Remainder tensor; same shape as the broadcast result of A and B. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `fmod` | `0` | Controls the modulo mode: `0` (default) uses Python-style integer mod (sign of divisor); `1` uses C-style `fmod` (sign of dividend, floating-point types only). |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32`, `int16`, `int8`, `uint8` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`mod-vec4.wgsl.jinja`](build/webgpu/mod-vec4.wgsl.jinja)
- [`mod.wgsl.jinja`](build/webgpu/mod.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Mod", { version: 1 });
const { c } = await kernel({ a: { data: aData, shape: [3] }, b: { data: bData, shape: [3] } });
```
