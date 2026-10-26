---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.RotaryEmbedding

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 23

## Description

Implements ONNX opset-23 RotaryEmbedding for float16 and float32 tensors. Applies rotary positional embeddings (RoPE) by rotating each head's embedding vector using precomputed `cos_cache` and `sin_cache` values. A partial rotation can be applied by setting `rotary_embedding_dim` to rotate only a prefix of the head dimension. `position_ids` keeps its standard logical int64 type; valid positions are non-negative and bounded by the WebGPU-addressable cache, so the backend stores them losslessly as uint32. Other ONNX floating-point input types are unsupported.

See the [ONNX `RotaryEmbedding` spec](https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | same as logical dtype | — | — | Input token embeddings. Shape is `(batch_size, sequence_length, hidden_size)` for rank 3 or `(batch_size, num_heads, sequence_length, head_size)` for rank 4. `head_size` must be even, and the `num_heads` attribute is required for rank-3 input. | required |
| `cos` | `cos_cache` | `T` | same as logical dtype | — | — | Precomputed cosine values. Without `position_ids`, shape is `(batch_size, sequence_length, rotary_dim/2)`; with `position_ids`, shape is `(max_sequence_length, rotary_dim/2)`. | required |
| `sin` | `sin_cache` | `T` | same as logical dtype | — | — | Precomputed sine values with the same shape and type as `cos_cache`. | required |
| `positionIds` | `position_ids` | `M` | `uint32` | `2` | — | Optional logical int64 per-token position indices of shape `(batch_size, sequence_length)`. Valid positions are non-negative cache-row indices and use uint32 WebGPU storage. When supplied, the 2D cache tables are gathered at these positions. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | same as `x` | Rotary-position-encoded tensor with the same shape and type as `X`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `interleaved` | `0` | Set to 1 to rotate using an interleaved pattern (even/odd elements), or 0 to split the head dimension into two contiguous halves. Default is 0. |
| `num_heads` | — | Optional number of attention heads. ONNX requires this attribute when `X` is rank 3; it is unnecessary for rank-4 input because the head count is explicit in the shape. |
| `rotary_embedding_dim` | `0` | Number of head-dimension elements to rotate; `0` means rotate the full head dimension. When set, only the leading `rotary_embedding_dim` elements are rotated and the rest are passed through unchanged. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `M` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `rank3_cache2_pos_quad` — Processes four pairs with vector loads and stores when head and cache alignment keep each group within a head and rotation region.
- `rank3_cache2_pos` — Processes one rotated pair per invocation with scalar storage, or copies an unchanged pair from the tail.
- `rank3_cache2_pos_output_vec4` — Writes four contiguous outputs per invocation with gathered pair values. Non-rotated components retain their original storage bits.
- `rank4_cache2_pos_quad` — Processes four pairs with vector loads and stores when head and cache alignment keep each group within a head and rotation region.
- `rank4_cache2_pos` — Processes one rotated pair per invocation with scalar storage, or copies an unchanged pair from the tail.
- `rank4_cache2_pos_output_vec4` — Writes four contiguous outputs per invocation with gathered pair values. Non-rotated components retain their original storage bits.
- `rank3_cache3_quad` — Processes four pairs with vector loads and stores when head and cache alignment keep each group within a head and rotation region.
- `rank3_cache3` — Processes one rotated pair per invocation with scalar storage, or copies an unchanged pair from the tail.
- `rank3_cache3_output_vec4` — Writes four contiguous outputs per invocation with gathered pair values. Non-rotated components retain their original storage bits.
- `rank4_cache3_quad` — Processes four pairs with vector loads and stores when head and cache alignment keep each group within a head and rotation region.
- `rank4_cache3` — Processes one rotated pair per invocation with scalar storage, or copies an unchanged pair from the tail.
- `rank4_cache3_output_vec4` — Writes four contiguous outputs per invocation with gathered pair values. Non-rotated components retain their original storage bits.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`rotary-embedding-output-vec4.wgsl.jinja`](build/webgpu/rotary-embedding-output-vec4.wgsl.jinja)
- [`rotary-embedding-quad.wgsl.jinja`](build/webgpu/rotary-embedding-quad.wgsl.jinja)
- [`rotary-embedding.wgsl.jinja`](build/webgpu/rotary-embedding.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.RotaryEmbedding", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 2, 1, 4] },
  cos: { data: cosData, shape: [16, 2] },
  sin: { data: sinData, shape: [16, 2] },
  positionIds: { data: positionIdsData, shape: [1, 1] },
});
```
