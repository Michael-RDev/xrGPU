---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.BatchNormalization

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 15

## Description

Applies inference-mode batch normalization: `Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B`. This package supports `training_mode=0`, rank-2-or-higher inputs, and a common float16 or float32 dtype for every tensor. ONNX training mode is intentionally not implemented because this inference-only release does not expose its required running-mean and running-variance outputs.

See the [ONNX `BatchNormalization` spec](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input data tensor with shape `(N, C, D1, ..., Dn)`, normalized independently per channel using the supplied estimated statistics. | required |
| `scale` | — | `T` | `1` | — | Per-channel scale tensor with shape `(C)`. | required |
| `b` | `B` | `T` | `1` | — | Per-channel bias tensor with shape `(C)`. | required |
| `inputMean` | `input_mean` | `T` | `1` | — | Precomputed estimated mean tensor with shape `(C)` used for inference. | required |
| `inputVar` | `input_var` | `T` | `1` | — | Precomputed estimated variance tensor with shape `(C)` used for inference. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | same as `x` | Batch-normalized output tensor with the same shape as `X`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `epsilon` | `0.00001` | Small value added to the variance before taking the square root to avoid division by zero. |
| `momentum` | `0.9` | Standard ONNX running-statistics momentum. This inference-only package accepts the default `0.9`; non-default values are reserved for the unsupported training-state update. |
| `training_mode` | `0` | Execution mode. This inference-only package supports the default value 0; value 1 is rejected because the ONNX training outputs are not exposed. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`batch-normalization-nc-vec4.wgsl.jinja`](build/webgpu/batch-normalization-nc-vec4.wgsl.jinja)
- [`batch-normalization-nchw-flat-vec4.wgsl.jinja`](build/webgpu/batch-normalization-nchw-flat-vec4.wgsl.jinja)
- [`batch-normalization-nchw-vec4.wgsl.jinja`](build/webgpu/batch-normalization-nchw-vec4.wgsl.jinja)
- [`batch-normalization-nchw.wgsl.jinja`](build/webgpu/batch-normalization-nchw.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.BatchNormalization", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [2, 3] },
  scale: { data: scaleData, shape: [3] },
  b: { data: bData, shape: [3] },
  inputMean: { data: inputMeanData, shape: [3] },
  inputVar: { data: inputVarData, shape: [3] },
});
```
