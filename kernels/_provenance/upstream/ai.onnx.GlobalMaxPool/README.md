---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.GlobalMaxPool

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 1

## Description

Applies max pooling across all spatial dimensions of `X`, producing one value per channel. Equivalent to MaxPool with kernel size equal to the full spatial extent of the input; output shape is `(N x C x 1 x ... x 1)`.

See the [ONNX `GlobalMaxPool` spec](https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor of shape `(N x C x D1 x ... x Dn)`, where `N` is the batch size and `C` is the number of channels. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | — | Output tensor of shape `(N x C x 1 x ... x 1)`; the maximum value over each spatial region per channel. | required |

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.GlobalMaxPool", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({ x: { data: xData, shape: [1, 2, 3] } }, {
  outputs: { y: { shape: [1, 2, 1], dtype: "float32" } },
});
```
