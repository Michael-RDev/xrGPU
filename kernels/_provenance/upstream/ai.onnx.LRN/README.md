---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.LRN

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Applies Local Response Normalization across the channel dimension of an input tensor of shape `(N x C x D1 x ... x Dk)`: each element is divided by `(bias + alpha / size * square_sum) ^ beta`, where `square_sum` is the sum of squares over a local window of `size` channels centered on that channel.

See the [ONNX `LRN` spec](https://onnx.ai/onnx/operators/onnx__LRN.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | `4` | — | Input data tensor of shape `(N x C x H x W)` or `(N x C x D1 x ... x Dk)`. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | same as `x` | Output tensor with the same shape and type as the input. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `alpha` | `0.0001` | Scaling parameter applied to the sum of squares. |
| `beta` | `0.75` | Exponent applied to the normalization term. |
| `bias` | `1` | Additive constant in the normalization denominator. |
| `size` | — | Required positive number of channels to sum over in the local normalization window. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`lrn-nchw.wgsl.jinja`](build/webgpu/lrn-nchw.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.LRN", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [1, 3, 1, 1] } }, {
  attrs: { size: 3 },
});
```
