---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.BiasAdd

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Adds a 1-D `bias` (broadcast over the channel dimension) to input `X`, then adds the residual tensor `skip` elementwise. All three tensors share the same channel count `C`; `X` and `skip` have shape `(N, S, C)`.

See the [ONNX Runtime `BiasAdd` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasAdd) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `X` | `T` | `3` | — | Input tensor of shape `(N, S, C)`: batch size `N`, spatial size `S`, and `C` channels. | required |
| `bias` | `T` | `1` | — | 1-D bias vector of length C, broadcast-added along the channel dimension. | required |
| `skip` | `T` | `3` | — | Residual tensor with the same `(N, S, C)` shape as `X`, added after the bias. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `Y` | `T` | `3` | same as `X` | Output tensor of shape `(N, S, C)`: the elementwise sum `X + bias + skip`. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`bias-add.wgsl.jinja`](build/webgpu/bias-add.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.BiasAdd", { version: 1 });
const { Y } = await kernel({
  X: { data: XData, shape: [1, 2, 4] },
  bias: { data: biasData, shape: [4] },
  skip: { data: skipData, shape: [1, 2, 4] },
});
```
