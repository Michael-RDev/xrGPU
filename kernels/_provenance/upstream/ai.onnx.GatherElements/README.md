---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.GatherElements

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Gathers values from the `data` tensor using index positions specified by `indices`, both of the same rank. For each element in `indices`, the corresponding output value is read from `data` by substituting that index along the chosen `axis` while keeping all other coordinates the same. The output shape equals the shape of `indices`.

See the [ONNX `GatherElements` spec](https://onnx.ai/onnx/operators/onnx__GatherElements.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `data` | `T` | — | — | Source tensor of rank `r >= 1` from which values are gathered. | required |
| `indices` | `I` | — | — | Integer index tensor of the same rank as `data`; each value selects a position along `axis` within bounds `[-s, s-1]` for axis size `s`. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `indices` | same as `indices` | Output tensor with the same shape as `indices`, containing the gathered values from `data`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `0` | The axis of `data` along which indexing is performed. Negative values count from the last dimension; accepted range is `[-r, r-1]` where `r` is the rank of `data`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool`, `int64` |
| `I` | `int32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`gather-elements-row-vec4.wgsl.jinja`](build/webgpu/gather-elements-row-vec4.wgsl.jinja)
- [`gather-elements.wgsl.jinja`](build/webgpu/gather-elements.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.GatherElements", { version: 1 });
const { output } = await kernel({
  data: { data: dataData, shape: [2, 3] },
  indices: { data: indicesData, shape: [1, 2] },
});
```
