---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.BiasSplitGelu

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Adds `bias` to `X`, splits the last dimension in half, then multiplies the left half elementwise by the GELU activation of the right half, producing an output with half the hidden dimension.

See the [ONNX Runtime `BiasSplitGelu` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasSplitGelu) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `X` | `T` | `3` | — | Input tensor of shape `(N, S, D_in)`, where `N` is the batch size, `S` is the spatial size, and `D_in` is an even hidden dimension. | required |
| `bias` | `T` | `1` | — | 1-D bias tensor of length `D_in`, matching the input hidden dimension. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `Y` | `T` | `3` | derived | Output tensor of shape `(N, S, D_out)`, where `D_out = D_in / 2`. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`bias-split-gelu.wgsl.jinja`](build/webgpu/bias-split-gelu.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.BiasSplitGelu", { version: 1 });
const { Y } = await kernel({
  X: { data: XData, shape: [1, 2, 4] },
  bias: { data: biasData, shape: [4] },
});
```
