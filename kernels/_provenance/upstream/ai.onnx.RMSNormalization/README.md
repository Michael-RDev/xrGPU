---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.RMSNormalization

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 23

## Description

Computes RMS normalization over the suffix dimensions of `X` starting at `axis`: `Y = X / sqrt(mean(X^2) + epsilon) * scale`. The normalization stage supports TensorProto `stash_type` values `1` (float32) and `10` (float16), and is cast back to the dtype of `X` before `scale` is applied. The input type `T` and scale/output type `V` may independently be float16 or float32; ONNX bfloat16 and double cases are unsupported.

See the [ONNX `RMSNormalization` spec](https://onnx.ai/onnx/operators/onnx__RMSNormalization.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor to be normalized; the RMS is taken over the last dimensions starting at `axis`. | required |
| `scale` | — | `V` | — | — | Scale tensor, unidirectionally broadcastable to `X`; its dtype `V` may differ from the input dtype `T`. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `V` | same as `x` | same as `x` | Normalized and scaled output tensor; same shape as `X` and same dtype `V` as `scale`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `-1` | The first dimension of the normalization suffix; negative values count from the end, so the default `-1` normalizes over only the last dimension. |
| `epsilon` | `0.00001` | Small constant added to the mean square before taking the square root to avoid division by zero. |
| `stash_type` | `1` | TensorProto element type used for normalization: `1` computes in float32, while `10` computes in float16. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `V` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`norm-row-stats.wgsl.jinja`](build/webgpu/norm-row-stats.wgsl.jinja)
- [`rms-normalization-splitk-normalize.wgsl.jinja`](build/webgpu/rms-normalization-splitk-normalize.wgsl.jinja)
- [`rms-normalization-splitk-partials.wgsl.jinja`](build/webgpu/rms-normalization-splitk-partials.wgsl.jinja)
- [`rms-normalization-stash-f16-serial.wgsl.jinja`](build/webgpu/rms-normalization-stash-f16-serial.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.RMSNormalization", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 2, 3] },
  scale: { data: scaleData, shape: [3] },
});
```
