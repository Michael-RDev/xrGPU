---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Add

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 14

## Description

Performs elementwise binary addition of two tensors with multidirectional (NumPy-style) broadcasting support. The output has the same element type as the inputs.

See the [ONNX `Add` spec](https://onnx.ai/onnx/operators/onnx__Add.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `T` | — | — | First operand. | required |
| `b` | `B` | `T` | — | — | Second operand. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `c` | `C` | `T` | derived | broadcast result of `a` and `b` | Elementwise sum of `A` and `B`; has the same element type as the inputs. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32`, `int8`, `uint8` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`binary-broadcast-vec4.wgsl.jinja`](build/webgpu/binary-broadcast-vec4.wgsl.jinja)
- [`binary-broadcast.wgsl.jinja`](build/webgpu/binary-broadcast.wgsl.jinja)
- [`binary-vec4.wgsl.jinja`](build/webgpu/binary-vec4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Add", { version: 1 });
const { c } = await kernel({ a: { data: aData, shape: [] }, b: { data: bData, shape: [1] } });
```
