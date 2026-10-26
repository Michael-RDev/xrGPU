---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Gather

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Gathers entries from `data` along the specified `axis` using an `indices` tensor, producing an output of rank `q + (r - 1)` where `r` is the rank of `data` and `q` is the rank of `indices`. Each index value selects a `(r-1)`-dimensional slice from `data` along `axis`, and the resulting slices are arranged into the output tensor with the `q` index dimensions replacing the gathered axis.

See the [ONNX `Gather` spec](https://onnx.ai/onnx/operators/onnx__Gather.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `data` | `T` | — | — | Tensor of rank `r >= 1` to gather entries from. | required |
| `indices` | `I` | — | — | Integer index tensor of any rank q; each value must be in `[-s, s-1]` where `s` is the size of `data` along `axis`. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | derived | derived | Output tensor of rank q + (r - 1) containing the gathered slices. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `0` | Axis of `data` along which to gather; negative values count from the back. Accepted range is `[-r, r-1]` where `r = rank(data)`. |

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
- [`datamove-gather-row.wgsl.jinja`](build/webgpu/datamove-gather-row.wgsl.jinja)
- [`gather.wgsl.jinja`](build/webgpu/gather.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Gather", { version: 1 });
const { output } = await kernel({
  data: { data: dataData, shape: [4] },
  indices: { data: indicesData, shape: [4] },
});
```
