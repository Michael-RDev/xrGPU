---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.MRotaryEmbedding

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Multimodal rotary position embedding (M-RoPE) for Qwen models. Each token has temporal, height, and width position streams; `mrope_section` partitions the half-rotary axis and `mrope_layout` assigns them. Text-only tokens set all streams equal, reducing the op to `RotaryEmbedding`. The effective rotary dimension must be positive and even; an odd head size is supported with a smaller even `rotary_embedding_dim`. This package supports float16/float32 and non-packed mode; bfloat16 and packed batching are not implemented. An out-of-range position id copies its rotation pair through unchanged.

See the [ONNX Runtime `MRotaryEmbedding` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MRotaryEmbedding) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `x` | `input` | `T` | same as logical dtype | — | — | Input token embeddings. Shape is `(batch_size, sequence_length, hidden_size)` for rank 3 or `(batch_size, num_heads, sequence_length, head_size)` for rank 4. The effective rotary dimension must be even, and `num_heads` is required for rank-3 input. | required |
| `positionIds` | `position_ids` | `M` | `int32` | `3` | — | Logical int64 position indices of shape `(3, batch_size, sequence_length)` for temporal, height and width streams. Signed int32 storage saturates larger values; negative or out-of-cache indices copy the corresponding pair unchanged. | required |
| `cos` | `cos_cache` | `T` | same as logical dtype | `2` | — | Precomputed cosine values of shape `(max_sequence_length, rotary_dim/2)`, shared by all three position streams. | required |
| `sin` | `sin_cache` | `T` | same as logical dtype | `2` | — | Precomputed sine values with the same shape and type as `cos_cache`. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `output` | `T` | same as `x` | same as `x` | Rotary-position-encoded tensor with the same shape and type as `input`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `interleaved` | `0` | Set to 1 to rotate using an interleaved pattern (even/odd elements), or 0 to split the head dimension into two contiguous halves. Default is 0. This is the rotation pairing and is independent of `mrope_layout`. |
| `is_packed_batching` | `0` | Whether `position_ids` uses packed-batch metadata. The default and only supported value is 0; packed batching (1) is not implemented. |
| `mrope_layout` | `0` | How the three sections are combined into one per-token cos/sin vector: `0` for the sectioned/chunked layout (Qwen2-VL, Qwen2.5-VL) or `1` for the interleaved layout (Qwen3-VL, Qwen3.5). Default is 0. |
| `mrope_section` | — | Three non-negative integers `[section_t, section_h, section_w]` dividing the half-rotary axis among the temporal, height and width streams. They must sum to `rotary_embedding_dim / 2`, or to `head_size / 2` when `rotary_embedding_dim` is 0. Required. |
| `num_heads` | `0` | Number of attention heads. The schema default is 0. A positive value is required for rank-3 `input` and whenever `rotary_embedding_dim` is nonzero; rank-4 execution otherwise infers the head count from `input`. |
| `rotary_embedding_dim` | `0` | Positive even number of head-dimension elements to rotate; `0` means the full head dimension, which must then be even. A smaller even value permits an odd head size and copies the remaining tail unchanged. |
| `scale` | `1` | Scale applied to the gathered cosine and sine values before the rotation. Default is 1.0. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `M` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `pairs4` — Four contiguous split-half pairs share vector activation loads and stores. Each component gathers its own position stream and preserves scaled-coefficient rounding to the storage type.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`mrotary-embedding-vector.wgsl.jinja`](build/webgpu/mrotary-embedding-vector.wgsl.jinja)
- [`mrotary-embedding.wgsl.jinja`](build/webgpu/mrotary-embedding.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.MRotaryEmbedding", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [2, 2, 16] },
  positionIds: { data: positionIdsData, shape: [3, 2, 2] },
  cos: { data: cosData, shape: [2, 4] },
  sin: { data: sinData, shape: [2, 4] },
}, {
  attrs: { num_heads: 2, mrope_section: [2, 1, 1] },
});
```
