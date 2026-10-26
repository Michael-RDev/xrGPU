---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.GlobalLpPool

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 2

## Description

Applies Lp pooling globally across all spatial dimensions of the input, equivalent to `LpPool` with the kernel spanning the full spatial extent. Each output value is the Lp norm (`p`-th root of sum of absolute values raised to the power `p`) over all spatial positions in a single channel.

See the [ONNX `GlobalLpPool` spec](https://onnx.ai/onnx/operators/onnx__GlobalLpPool.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor of shape `(N x C x D1 x ... x Dn)`; the spatial dimensions `D1...Dn` are reduced. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | — | Output tensor of shape `(N x C x 1 x ... x 1)`; same rank as the input with all spatial dimensions set to 1. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `p` | `2` | The `p` value of the Lp norm for pooling over the spatial dimensions; defaults to `2` (Euclidean norm). |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Device requirements

Some implementation variants require `shader-f16`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`global-lp-pool.wgsl.jinja`](build/webgpu/global-lp-pool.wgsl.jinja)
- [`pool-global-reduction.wgsl.jinja`](build/webgpu/pool-global-reduction.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.GlobalLpPool", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({ x: { data: xData, shape: [1, 2, 4] } }, {
  outputs: { y: { shape: [1, 2, 1], dtype: "float32" } },
});
```
