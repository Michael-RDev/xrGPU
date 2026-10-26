---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.RotaryEmbedding

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Rotary positional embedding (RoPE): each head's embedding vector is rotated with the `cos_cache` and `sin_cache` rows selected by `position_ids`, which is either a single base offset (token `s` reads row `position_ids[0] + s`) or a `(batch_size, sequence_length)` table. `input` is rank 3 `(batch_size, sequence_length, hidden_size)` or rank 4 `(batch_size, num_heads, sequence_length, head_size)`. `rotary_embedding_dim` rotates a prefix and copies the tail unchanged. Rotation arithmetic is float32 with one narrowing store. Bfloat16 and non-default `scale` are not implemented.

See the [ONNX Runtime `RotaryEmbedding` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.RotaryEmbedding) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `x` | `input` | `T` | same as logical dtype | — | — | Input token embeddings, shaped `(batch_size, sequence_length, hidden_size)` at rank 3 or `(batch_size, num_heads, sequence_length, head_size)` at rank 4. At rank 3 the head size comes from `num_heads` when that attribute is positive and from `2 * cos_cache.shape[1]` otherwise; at rank 4 both the head count and head size are read from the shape. | required |
| `positionIds` | `position_ids` | `M` | `uint32` | — | — | Logical int64 cache-row selector in either upstream format: a scalar or one-element vector holding a base offset, so token `s` reads row `position_ids[0] + s`; or a `(batch_size, sequence_length)` table read per token. Valid positions are non-negative rows of the caches and use uint32 WebGPU storage, so a negative value is rejected at the host boundary. | required |
| `cos` | `cos_cache` | `T` | same as logical dtype | `2` | — | Precomputed cosine values of shape `(max_sequence_length, rotary_dim / 2)`, where `rotary_dim` is `rotary_embedding_dim` when that is positive and the head size otherwise. | required |
| `sin` | `sin_cache` | `T` | same as logical dtype | `2` | — | Precomputed sine values with the same shape and type as `cos_cache`. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `output` | `T` | same as `x` | same as `x` | Rotary-position-encoded tensor with the same shape and type as `input`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `interleaved` | `0` | Set to 1 to pair adjacent even/odd elements, or 0 to pair each element of the first half of the rotary window with the matching element of the second half. Default is 0. |
| `is_packed_batching` | `0` | Ragged (packed) batch inputs. Its only upstream effect is to lift the `sequence_length <= max_sequence_length` bound, and this implementation never imposes that bound: every gathered row is required to index the caches whatever the sequence length. Default is 0. |
| `num_heads` | `0` | Number of attention heads. Default is 0, which asks the rank-3 path to take the head size from the cache width instead; a positive value is required whenever `rotary_embedding_dim` is nonzero. At rank 4 the head count comes from the input shape and this attribute is not consulted. |
| `rotary_embedding_dim` | `0` | Positive even count of leading head-dimension elements to rotate; the remaining tail is copied unchanged. Default is 0, meaning the whole head dimension, which must then be even. An odd value is rejected. |
| `scale` | `1` | Declared scale for the gathered rotation. No ONNX Runtime provider applies it, so the default and only accepted value is 1.0 and a caller's other value is rejected rather than silently discarded. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `M` | `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`rotary-embedding-slices.wgsl.jinja`](build/webgpu/rotary-embedding-slices.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.RotaryEmbedding", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 2, 18] },
  positionIds: { data: positionIdsData, shape: [1, 2] },
  cos: { data: cosData, shape: [4, 3] },
  sin: { data: sinData, shape: [4, 3] },
});
```
