---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.InstanceNormalization

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 6

## Description

Applies instance normalization to the input: `y = scale * (x - mean) / sqrt(variance + epsilon) + B`, where `mean` and `variance` are computed per instance per channel over the spatial dimensions. Equivalent to batch normalization with a batch size of one per channel.

See the [ONNX `InstanceNormalization` spec](https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `input` | — | `T` | — | — | Input tensor of shape `(N x C x D1 x ... x Dn)`; at least 3-D. | required |
| `scale` | — | `T` | `1` | — | 1-D scale tensor of size C, one scale factor per channel. | required |
| `b` | `B` | `T` | `1` | — | 1-D bias tensor of size C, one bias value per channel. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `input` | same as `input` | Normalized output tensor; same shape as the input. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `epsilon` | `0.00001` | Small constant added to the variance before taking the square root to avoid division by zero. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Device requirements

Some implementation variants require `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`instance-normalization-apply.wgsl.jinja`](build/webgpu/instance-normalization-apply.wgsl.jinja)
- [`instance-normalization-batched-planes-vec4.wgsl.jinja`](build/webgpu/instance-normalization-batched-planes-vec4.wgsl.jinja)
- [`instance-normalization-splitk-combine.wgsl.jinja`](build/webgpu/instance-normalization-splitk-combine.wgsl.jinja)
- [`instance-normalization-splitk-partials.wgsl.jinja`](build/webgpu/instance-normalization-splitk-partials.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.InstanceNormalization", { version: 1 });
const { output } = await kernel({
  input: { data: inputData, shape: [1, 2, 1, 3] },
  scale: { data: scaleData, shape: [2] },
  b: { data: bData, shape: [2] },
});
```
