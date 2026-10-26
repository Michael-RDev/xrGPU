---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.GlobalAveragePool

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 1

## Description

Applies average pooling across all spatial dimensions of `X`, reducing each channel to a single scalar. Equivalent to `AveragePool` with kernel size equal to the full spatial extent of the input. The output shape is `(N x C x 1 x ... x 1)`, preserving batch and channel dimensions.

See the [ONNX `GlobalAveragePool` spec](https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor of shape `(N x C x D1 x ... x Dn)` where `N` is batch size, `C` is channels, and `D1...Dn` are spatial dimensions. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | — | Output tensor of the same rank as the input, with shape `(N x C x 1 x ... x 1)` — all spatial dimensions collapsed to 1. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`pool-global-reduction.wgsl.jinja`](build/webgpu/pool-global-reduction.wgsl.jinja)
- [`pool-global-serial.wgsl.jinja`](build/webgpu/pool-global-serial.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `y`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.GlobalAveragePool", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({ x: { data: xData, shape: [2, 3, 1, 1] } }, {
  outputs: { y: { shape: [2, 3, 1, 1], dtype: "float32" } },
});
```
