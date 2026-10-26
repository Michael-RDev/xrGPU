---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Not

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 1

## Description

Computes the elementwise logical NOT of a boolean tensor. Each output element is `true` where the corresponding input element is `false`, and vice versa.

See the [ONNX `Not` spec](https://onnx.ai/onnx/operators/onnx__Not.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `B` | — | — | Input boolean tensor. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `B` | same as `x` | same as `x` | Output boolean tensor with each element logically negated. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `B` | `bool` |

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

A `bool` input takes 0/1 values in a `Uint8Array` together with `dtype: "bool"`, since no typed-array class implies it.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Not", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [2], dtype: "bool" } });
```
