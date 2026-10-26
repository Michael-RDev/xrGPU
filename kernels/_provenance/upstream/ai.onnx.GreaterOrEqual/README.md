---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.GreaterOrEqual

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 16

## Description

Compares `A >= B` elementwise with NumPy-style multidirectional broadcasting, producing a boolean tensor.

See the [ONNX `GreaterOrEqual` spec](https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `T` | — | — | First input operand for the comparison. | required |
| `b` | `B` | `T` | — | — | Second input operand for the comparison. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `c` | `C` | `B` | derived | broadcast result of `a` and `b` | Boolean result tensor; `true` where `A >= B`. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `int16`, `uint32`, `int8`, `uint8` |
| `B` | `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`compare-broadcast-vec4.wgsl.jinja`](build/webgpu/compare-broadcast-vec4.wgsl.jinja)
- [`compare-broadcast.wgsl.jinja`](build/webgpu/compare-broadcast.wgsl.jinja)
- [`compare-vec4.wgsl.jinja`](build/webgpu/compare-vec4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.GreaterOrEqual", { version: 1 });
const { c } = await kernel({ a: { data: aData, shape: [4] }, b: { data: bData, shape: [4] } });
```
