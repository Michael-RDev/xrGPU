---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.GatherND

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Gathers slices from `data` using an N-dimensional index tensor, producing an output of rank `q + r - indices_shape[-1] - 1 - b` where `r` is the data rank, `q` the indices rank, and `b` the number of batch dimensions. Each index-tuple in `indices` selects either a scalar or a sub-tensor slice from `data`; this is the inverse of `ScatterND`.

See the [ONNX `GatherND` spec](https://onnx.ai/onnx/operators/onnx__GatherND.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `data` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | Tensor of rank `r >= 1` to gather slices from. | required |
| `indices` | `I` | `int32` | — | — | Logical int64 index tensor of rank `q >= 1`. Its last dimension specifies the indexing depth into `data`; for an indexed axis of size `s`, values must be in `[-s, s - 1]`. Signed indices use int32 WebGPU storage. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | derived | derived | Gathered output tensor of rank `q + r - indices_shape[-1] - 1 - batch_dims`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `batch_dims` | `0` | Number of leading batch dimensions shared by `data` and `indices`; gathering begins at dimension `batch_dims` of `data`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool`, `int64` |
| `I` | `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`datamove-gather-row.wgsl.jinja`](build/webgpu/datamove-gather-row.wgsl.jinja)
- [`gather-nd.wgsl.jinja`](build/webgpu/gather-nd.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Required output shapes and logical data types are inferred from the supplied inputs and attributes; result tensors are allocated automatically.

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

A `bool` input takes 0/1 values in a `Uint8Array` together with `dtype: "bool"`, since no typed-array class implies it.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.GatherND", { version: 1 });
const { output } = await kernel({
  data: { data: dataData, shape: [2, 2], dtype: "bool" },
  indices: { data: indicesData, shape: [2, 1, 2] },
});
```
