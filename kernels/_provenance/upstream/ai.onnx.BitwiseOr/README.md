---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.BitwiseOr

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 18

## Description

Computes the elementwise bitwise `or` of two integer tensors `A` and `B`, with multidirectional (NumPy-style) broadcasting. The output `C` has the broadcasted shape and the same dtype as the inputs.

See the [ONNX `BitwiseOr` spec](https://onnx.ai/onnx/operators/onnx__BitwiseOr.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `T` | — | — | First input operand for the bitwise OR. | required |
| `b` | `B` | `T` | — | — | Second input operand for the bitwise OR. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `c` | `C` | `T` | derived | broadcast result of `a` and `b` | Result tensor containing the elementwise bitwise OR of A and B. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `uint32`, `int32`, `int16`, `uint8`, `int8` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`bitwise-binary-broadcast.wgsl.jinja`](build/webgpu/bitwise-binary-broadcast.wgsl.jinja)
- [`bitwise-binary-vec4.wgsl.jinja`](build/webgpu/bitwise-binary-vec4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.BitwiseOr", { version: 1 });
const { c } = await kernel({ a: { data: aData, shape: [3] }, b: { data: bData, shape: [3] } });
```
