---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.TensorScatter

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 24

## Description

Functionally updates a KV cache tensor by scattering an `update` tensor into the `past_cache` along a sequence axis, producing `present_cache` with the same shape. Each batch sample's update is written at the offset given by `write_indices` (zero if omitted), either linearly or in wrap-around `circular` fashion.

See the [ONNX `TensorScatter` spec](https://onnx.ai/onnx/operators/onnx__TensorScatter.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `past` | `past_cache` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | Existing cache tensor with shape `(batch_size, ..., max_sequence_length, ...)`. | required |
| `update` | — | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | New values to scatter in, with the same shape as `past_cache` except the sequence dimension equals `sequence_length`. | required |
| `writeIndices` | `write_indices` | `I` | `uint32` | `1` | — | Logical int64 per-sample write offset into the cache sequence dimension; shape `(batch_size,)`, stored as uint32 by WebGPU, and assumed all zeros if absent. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `present` | `present_cache` | `T` | same as `past` | same as `past` | Updated cache; same shape as `past_cache`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `-2` | Sequence dimension of `past_cache` and `update`; cannot be 0 (the batch dimension). Default is `-2`. |
| `mode` | `"linear"` | Write mode: `linear` requires `write_indices + sequence_length <= max_sequence_length`; `circular` wraps the write index modulo `max_sequence_length`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `int16`, `int8`, `uint32`, `uint8`, `bool` |
| `I` | `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`scatter-flat-copy.wgsl.jinja`](build/webgpu/scatter-flat-copy.wgsl.jinja)
- [`tensor-scatter.wgsl.jinja`](build/webgpu/tensor-scatter.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.TensorScatter", { version: 1 });
const { present } = await kernel({
  past: { data: pastData, shape: [1, 4, 2] },
  update: { data: updateData, shape: [1, 3, 2] },
});
```
