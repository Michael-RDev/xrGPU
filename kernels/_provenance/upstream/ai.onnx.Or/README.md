---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Or

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 7

## Description

Computes the elementwise logical `or` of two boolean input tensors `A` and `B`, producing a boolean result tensor `C` of the same type. Supports multidirectional (NumPy-style) broadcasting.

See the [ONNX `Or` spec](https://onnx.ai/onnx/operators/onnx__Or.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `B` | — | — | First input operand for the logical or. | required |
| `b` | `B` | `B` | — | — | Second input operand for the logical or. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `c` | `C` | `B` | derived | broadcast result of `a` and `b` | Boolean result tensor of the elementwise or. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `B` | `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`logical-broadcast.wgsl.jinja`](build/webgpu/logical-broadcast.wgsl.jinja)
- [`logical-vec4.wgsl.jinja`](build/webgpu/logical-vec4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Or", { version: 1 });
const { c } = await kernel({
  a: { data: aData, shape: [4], dtype: "bool" },
  b: { data: bData, shape: [4], dtype: "bool" },
});
```
