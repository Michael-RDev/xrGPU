---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.SimplifiedLayerNormalization

`ai.onnx`  ·  ONNX Runtime compatibility operator  ·  default-domain since_version 1

## Description

Implements ONNX Runtime's legacy RMS normalization for models that serialize `SimplifiedLayerNormalization` in the default ONNX domain: `Y = (X / sqrt(mean(X^2) + epsilon)) * scale`. Its arithmetic corresponds to [ONNX `RMSNormalization`](https://onnx.ai/onnx/operators/onnx__RMSNormalization.html), while retaining scale-before-output-cast ordering and optional `inv_std_var`. See the [compatibility discussion](https://github.com/onnx/onnx/issues/6582#issuecomment-3591862327). Supports float16/float32, float32 statistics, and a nonempty normalization suffix.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor; the mean square is taken over the suffix dimensions starting at `axis`. | required |
| `scale` | — | `V` | — | — | Scale tensor, unidirectionally broadcastable to `X`; output `Y` has this tensor's dtype. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `V` | same as `x` | same as `x` | Normalized and scaled output; same shape as `X` and same dtype as `scale`. | required |
| `invStdVar` | `inv_std_var` | `U` | same as `x` | derived | Optional inverse RMS statistic `1 / sqrt(mean(X^2) + epsilon)`; same rank as `X`, with the dimensions from `axis` onward set to one. | optional |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `-1` | The first dimension of the normalization suffix; negative values count from the end, so the default `-1` normalizes only the last dimension. |
| `epsilon` | `0.00001` | Small constant added to the mean square before taking the square root to avoid division by zero. |
| `keep_dims` | `1` | Legacy compatibility attribute. Only `1` is supported: optional statistics retain the input rank with normalized dimensions set to one. |
| `stash_type` | `1` | TensorProto element type of `inv_std_var`; this package implements the legacy float32 value `1`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `V` | `float32`, `float16` |
| `U` | `float32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`norm-row-stats.wgsl.jinja`](build/webgpu/norm-row-stats.wgsl.jinja)
- [`rms-normalization-splitk-normalize.wgsl.jinja`](build/webgpu/rms-normalization-splitk-normalize.wgsl.jinja)
- [`rms-normalization-splitk-partials.wgsl.jinja`](build/webgpu/rms-normalization-splitk-partials.wgsl.jinja)
- [`rms-normalization.wgsl.jinja`](build/webgpu/rms-normalization.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.SimplifiedLayerNormalization", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [8] }, scale: { data: scaleData, shape: [8] } });
```
