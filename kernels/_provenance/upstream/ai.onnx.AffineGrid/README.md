---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.AffineGrid

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 20

## Description

Generates a 2-D or 3-D flow field (sampling grid) from a batch of affine matrices `theta`. Each affine matrix is applied to the target output's normalized coordinate grid in `[-1, 1]`, producing sample coordinates at every output position.

See the [ONNX `AffineGrid` spec](https://onnx.ai/onnx/operators/onnx__AffineGrid.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `theta` | `T` | `3` | — | Batch of affine matrices with shape `(N, 2, 3)` for 2-D or `(N, 3, 4)` for 3-D. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `grid` | `T` | derived | — | Sampling grid of 2-D coordinates with shape `(N, H, W, 2)`, or 3-D coordinates with shape `(N, D, H, W, 3)`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `align_corners` | `0` | When 1, the values -1 and 1 refer to the centers of the corner pixels; when 0 (default), they refer to the outer edges of the corner pixels. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`affine-grid.wgsl.jinja`](build/webgpu/affine-grid.wgsl.jinja)
- [`affine-grid3d-w4.wgsl.jinja`](build/webgpu/affine-grid3d-w4.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `grid`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.AffineGrid", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { grid } = await kernel({ theta: { data: thetaData, shape: [1, 2, 3] } }, {
  outputs: { grid: { shape: [1, 3, 2, 2], dtype: "float32" } },
});
```
