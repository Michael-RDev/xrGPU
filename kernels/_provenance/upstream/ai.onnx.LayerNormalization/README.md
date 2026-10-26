---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.LayerNormalization

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 17

## Description

Normalizes a tensor along a suffix of axes starting at `axis` by subtracting the mean and dividing by the square root of the variance plus `epsilon`, then scales and optionally shifts the result with learnable `Scale` and `B` tensors. The output `Y` has the same shape as `X`; optional outputs `Mean` and `InvStdDev` expose the per-normalization-group statistics computed during normalization.

See the [ONNX `LayerNormalization` spec](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Tensor to be normalized. | required |
| `scale` | `Scale` | `T` | — | — | Scale tensor applied after normalization. | required |
| `b` | `B` | `T` | — | — | Optional bias tensor added after scaling. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | same as `x` | Normalized and scaled output tensor; same shape as X. | required |
| `mean` | `Mean` | `float32` | same as `x` | — | Per-normalization-group mean in the ONNX broadcastable keepdims shape: dimensions before `axis` are preserved and dimensions from `axis` onward are 1. | optional |
| `invStdDev` | `InvStdDev` | `float32` | same as `x` | — | Per-normalization-group reciprocal standard deviation `1 / sqrt(variance + epsilon)`, returned in the same ONNX broadcastable keepdims shape as `Mean`. | optional |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `-1` | The first axis of the normalization range; all axes from `axis` to the last are normalized together. Negative values count from the end; the default `-1` normalizes only the last dimension. |
| `epsilon` | `0.00001` | Small constant added to the variance before taking the square root to avoid division by zero. |
| `stash_type` | `1` | TensorProto element type used for the normalization stage and optional statistics; the implemented ONNX route supports the standard float32 value (`1`). |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `last_axis_row_vec2` — Uses paired float32 loads for even last-axis rows that cannot use four-wide storage.
- `last_axis_bias_row_vec2` — Uses paired float32 loads for even last-axis rows that cannot use four-wide storage.
- `last_axis_broadcast_row_vec4` — Normalizes packed rows with affine tensors broadcast across leading dimensions; short rows share a workgroup within device limits.
- `last_axis_broadcast_bias_row_vec4` — Normalizes packed rows with affine tensors broadcast across leading dimensions; short rows share a workgroup within device limits.
- `last_axis_broadcast_rows_vec4` — Normalizes packed rows with affine tensors broadcast across leading dimensions; short rows share a workgroup within device limits.
- `last_axis_broadcast_bias_rows_vec4` — Normalizes packed rows with affine tensors broadcast across leading dimensions; short rows share a workgroup within device limits.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`layer-normalization.wgsl.jinja`](build/webgpu/layer-normalization.wgsl.jinja)
- [`norm-row-stats.wgsl.jinja`](build/webgpu/norm-row-stats.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.LayerNormalization", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 4] },
  scale: { data: scaleData, shape: [4] },
});
```
