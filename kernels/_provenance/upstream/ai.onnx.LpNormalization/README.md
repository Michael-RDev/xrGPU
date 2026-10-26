---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.LpNormalization

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 1

## Description

Applies Lp-normalization to the input tensor along the specified axis: `output = input / Lp_norm(input, axis)`. Supports L1 (`p=1`) and L2 (`p=2`) norms. Where the Lp norm is zero, the output is defined as zero to avoid division by zero.

See the [ONNX `LpNormalization` spec](https://onnx.ai/onnx/operators/onnx__LpNormalization.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `input` | `T` | — | — | Input tensor to normalize. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `input` | same as `input` | Tensor after Lp-normalization; same shape as the input. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `-1` | The axis along which normalization is applied; `-1` means the last axis. |
| `p` | `2` | The order of the Lp norm to use; only `1` (L1) or `2` (L2) are supported. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`lp-norm-divide.wgsl.jinja`](build/webgpu/lp-norm-divide.wgsl.jinja)
- [`lp-norm-reduce.wgsl.jinja`](build/webgpu/lp-norm-reduce.wgsl.jinja)
- [`lp-norm-split-combine.wgsl.jinja`](build/webgpu/lp-norm-split-combine.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.LpNormalization", { version: 1 });
const { output } = await kernel({ input: { data: inputData, shape: [2, 2] } });
```
