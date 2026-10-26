---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.BitShift

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 11

## Description

Performs an elementwise bitwise shift on unsigned integer tensors. `X` is shifted left or right by the amounts in `Y`, with the direction controlled by the `direction` attribute. Supports multidirectional (NumPy-style) broadcasting between `X` and `Y`.

See the [ONNX `BitShift` spec](https://onnx.ai/onnx/operators/onnx__BitShift.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor to be shifted. | required |
| `y` | `Y` | `T` | — | — | Tensor specifying the number of bit positions to shift each element of X. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `z` | `Z` | `T` | derived | broadcast result of `x` and `y` | Output tensor with the same shape as the broadcast result of X and Y. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `direction` | — | Direction of the bit shift: `LEFT` shifts bits toward higher significance (increasing value), while `RIGHT` shifts toward lower significance (decreasing value). |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `uint32`, `uint8` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`bitshift-vec4.wgsl.jinja`](build/webgpu/bitshift-vec4.wgsl.jinja)
- [`bitshift.wgsl.jinja`](build/webgpu/bitshift.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.BitShift", { version: 1 });
const { z } = await kernel({ x: { data: xData, shape: [1] }, y: { data: yData, shape: [3] } }, {
  attrs: { direction: "LEFT" },
});
```
