---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Hardmax

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Computes the hardmax of the input along a single axis: sets the position of the first maximum value along `axis` to 1 and all other positions to 0. The output has the same shape as the input.

See the [ONNX `Hardmax` spec](https://onnx.ai/onnx/operators/onnx__Hardmax.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `input` | `T` | — | — | Input tensor with rank at least 1. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `output` | `T` | same as `x` | same as `x` | The output tensor with the same shape as the input, containing hardmax values. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `-1` | The dimension along which hardmax is computed. Negative values count from the back; accepted range is `[-r, r-1]` where `r` is the rank of the input. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`hardmax-axis-tree.wgsl.jinja`](build/webgpu/hardmax-axis-tree.wgsl.jinja)
- [`hardmax-last-axis-subgroup.wgsl.jinja`](build/webgpu/hardmax-last-axis-subgroup.wgsl.jinja)
- [`hardmax-last-axis-vec4.wgsl.jinja`](build/webgpu/hardmax-last-axis-vec4.wgsl.jinja)
- [`hardmax.wgsl.jinja`](build/webgpu/hardmax.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Hardmax", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [1, 3] } });
```
