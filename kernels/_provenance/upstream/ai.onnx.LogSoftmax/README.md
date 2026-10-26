---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.LogSoftmax

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Computes `log(softmax(input, axis))` along a single axis using a numerically stable shifted reduction. The output has the same shape as the input.

See the [ONNX `LogSoftmax` spec](https://onnx.ai/onnx/operators/onnx__LogSoftmax.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `input` | `T` | — | — | The input tensor of rank >= 1. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `output` | `T` | same as `x` | same as `x` | The log-softmax values; same shape as the input. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `-1` | The axis along which log-softmax is computed. Negative values count from the end; the default `-1` operates over the last dimension. Accepted range is `[-r, r-1]` where `r` is the input rank. |

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
- [`softmax-longrow-normalize.wgsl.jinja`](build/webgpu/softmax-longrow-normalize.wgsl.jinja)
- [`softmax-longrow-stats.wgsl.jinja`](build/webgpu/softmax-longrow-stats.wgsl.jinja)
- [`softmax-normalize.wgsl.jinja`](build/webgpu/softmax-normalize.wgsl.jinja)
- [`softmax-online-packed-rows.wgsl.jinja`](build/webgpu/softmax-online-packed-rows.wgsl.jinja)
- [`softmax-online.wgsl.jinja`](build/webgpu/softmax-online.wgsl.jinja)
- [`softmax-row-stage-strided-vec4.wgsl.jinja`](build/webgpu/softmax-row-stage-strided-vec4.wgsl.jinja)
- [`softmax-row-stage.wgsl.jinja`](build/webgpu/softmax-row-stage.wgsl.jinja)
- [`softmax-strided-online-coop.wgsl.jinja`](build/webgpu/softmax-strided-online-coop.wgsl.jinja)
- [`softmax-strided-packed4-tail.wgsl.jinja`](build/webgpu/softmax-strided-packed4-tail.wgsl.jinja)
- [`softmax-subgroup-rows.wgsl.jinja`](build/webgpu/softmax-subgroup-rows.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.LogSoftmax", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [1, 3] } });
```
