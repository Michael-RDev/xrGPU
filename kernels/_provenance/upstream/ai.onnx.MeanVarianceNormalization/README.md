---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.MeanVarianceNormalization

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Normalizes each group as `(X - mean) / sqrt(variance)`, reducing over `axes` (default `[0, 2, 3]`).

See the [ONNX `MeanVarianceNormalization` spec](https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor to normalize. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | same as `x` | Normalized tensor with the same shape as `X`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axes` | `[0,2,3]` | Axes that share a mean and variance; negative values count from the back. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `packed_short_groups` — Caches contiguous short groups in vector words, uses scaled shifted moments, and guards scalar access for partial words.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`mean-variance-normalization-packed-rows.wgsl.jinja`](build/webgpu/mean-variance-normalization-packed-rows.wgsl.jinja)
- [`mean-variance-normalization-serial-rows.wgsl.jinja`](build/webgpu/mean-variance-normalization-serial-rows.wgsl.jinja)
- [`mean-variance-normalization-subgroup.wgsl.jinja`](build/webgpu/mean-variance-normalization-subgroup.wgsl.jinja)
- [`noop.wgsl.jinja`](build/webgpu/noop.wgsl.jinja)
- [`norm-flat-apply.wgsl.jinja`](build/webgpu/norm-flat-apply.wgsl.jinja)
- [`norm-flat-splitk-combine.wgsl.jinja`](build/webgpu/norm-flat-splitk-combine.wgsl.jinja)
- [`norm-flat-splitk-partials.wgsl.jinja`](build/webgpu/norm-flat-splitk-partials.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.MeanVarianceNormalization", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [1, 1, 2, 2] } });
```
