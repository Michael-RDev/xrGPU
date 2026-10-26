---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.GroupNormalization

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 21

## Description

Applies group normalization to the input: `y = scale * (x - mean) / sqrt(variance + epsilon) + bias`, where mean and variance are computed per instance per group of channels. The number of groups `num_groups` must divide the channel count `C` evenly; when `num_groups == C` this is equivalent to InstanceNormalization, and when `num_groups == 1` it is equivalent to LayerNormalization. The normalization stage supports TensorProto `stash_type` values `1` (float32) and `10` (float16).

See the [ONNX `GroupNormalization` spec](https://onnx.ai/onnx/operators/onnx__GroupNormalization.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input data tensor of shape `(N x C x D1 x ... x Dn)` where `N` is batch size and `C` is the number of channels. | required |
| `scale` | — | `T` | `1` | — | Scale tensor of shape `(C)`, one value per channel. | required |
| `bias` | — | `T` | `1` | — | Bias tensor of shape `(C)`, one value per channel. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | same as `x` | Normalized output tensor of the same shape as `X`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `epsilon` | `0.00001` | Small value added to the variance denominator to avoid division by zero. |
| `num_groups` | — | Required number of groups to divide the channels into; must be a divisor of `C`. |
| `stash_type` | `1` | TensorProto element type used for the normalization stage: `1` computes in float32, while `10` computes in float16. Normalized values are cast back to the input type before scale and bias are applied. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`group-normalization-splitk-apply.wgsl.jinja`](build/webgpu/group-normalization-splitk-apply.wgsl.jinja)
- [`group-normalization-splitk-partials.wgsl.jinja`](build/webgpu/group-normalization-splitk-partials.wgsl.jinja)
- [`group-normalization-stash-f16-serial.wgsl.jinja`](build/webgpu/group-normalization-stash-f16-serial.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.GroupNormalization", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 2, 3] },
  scale: { data: scaleData, shape: [2] },
  bias: { data: biasData, shape: [2] },
}, {
  attrs: { num_groups: 1 },
});
```
