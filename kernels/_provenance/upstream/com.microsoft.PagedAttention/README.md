---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.PagedAttention

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Attention over a block-based (paged) KV cache: `cumulative_sequence_length` marks the sequence boundaries and `block_table` maps a sequence's history onto scattered blocks. This step's K/V are scattered into the cache, then attended with that history. Grouped-query heads, `scale`, packed `[Q|K|V]`, `slot_mapping`, and float16 cache storage are supported; the cache outputs alias the input caches and are updated in place. Rotary embeddings, softcap, local windows, LATENT layout, narrower value heads, quantized KV, head sinks, q/k normalization, scales, and attention metadata are not implemented.

See the [ONNX Runtime `PagedAttention` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.PagedAttention) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `queryT` | `query` | `T` | same as logical dtype | `2` | — | Packed queries of shape `(num_tokens, num_heads * head_size)`, or `(num_tokens, (num_heads + 2 * kv_num_heads) * head_size)` when `key` and `value` are absent and Q, K and V share one row. | required |
| `keyT` | `key` | `T` | same as logical dtype | `2` | — | Keys of shape `(num_tokens, kv_num_heads * head_size)`. Absent means `query` carries packed `[Q\|K\|V]`. | optional |
| `valueT` | `value` | `T` | same as logical dtype | `2` | — | Values of shape `(num_tokens, kv_num_heads * head_size)`. Present exactly when `key` is. | optional |
| `keyCacheT` | `key_cache` | `T` | same as logical dtype | `4` | — | Block-based key cache of shape `(num_blocks, block_size, kv_num_heads, head_size)`, updated in place. | required |
| `valueCacheT` | `value_cache` | `T` | same as logical dtype | `4` | — | Block-based value cache with the same shape as `key_cache`, updated in place. | required |
| `cumulativeSequenceLengthT` | `cumulative_sequence_length` | `S` | `int32` | `1` | — | Exclusive prefix sums of the per-sequence token counts, shape `(batch_size + 1)`; sequence `b` owns packed tokens `[cum[b], cum[b+1])`. | required |
| `pastSeqlensT` | `past_seqlens` | `S` | `int32` | `1` | — | Cached history length per sequence, shape `(batch_size)`. | required |
| `blockTableT` | `block_table` | `S` | `int32` | `2` | — | Physical block index per sequence and logical block, shape `(batch_size, max_blocks_per_sequence)`. | required |
| `slotMappingT` | `slot_mapping` | `S` | `int32` | `1` | — | Flat destination slot, `block_id * block_size + offset`, for each token; `-1` suppresses that token's cache write. When omitted, the slot is derived from `past_seqlens`. `block_table` remains required because it defines the read path. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `2` | derived | Attention output of shape `(num_tokens, num_heads * head_size)`. | required |
| `keyCacheT` | `key_cache` | `T` | `4` | same as `keyCacheT` | Optional return alias for the updated in-place key cache. Both caches are updated even when only this result is requested. | optional |
| `valueCacheT` | `value_cache` | `T` | `4` | same as `valueCacheT` | Optional return alias for the updated in-place value cache. Both caches are updated even when only this result is requested. | optional |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `is_causal` | `1` | Whether to apply causal masking. This package supports only value 1. |
| `kv_num_heads` | — | Number of key/value heads. |
| `num_heads` | — | Number of query heads. |
| `scale` | — | Scale applied to query-key products; zero or omission selects `1 / sqrt(head_size)`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float16` |
| `S` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `separate_derived_splitk` — Splits each token's key history into contiguous ranges, processes one range per workgroup, and merges the resulting online-softmax states. This adds parallel key ranges for grouped-query decode shapes with few `(token, KV head)` tuples.
- `separate_slot_splitk` — Splits each token's key history into contiguous ranges, processes one range per workgroup, and merges the resulting online-softmax states. This adds parallel key ranges for grouped-query decode shapes with few `(token, KV head)` tuples.
- `packed_derived_splitk` — Splits each token's key history into contiguous ranges, processes one range per workgroup, and merges the resulting online-softmax states. This adds parallel key ranges for grouped-query decode shapes with few `(token, KV head)` tuples.
- `packed_slot_splitk` — Splits each token's key history into contiguous ranges, processes one range per workgroup, and merges the resulting online-softmax states. This adds parallel key ranges for grouped-query decode shapes with few `(token, KV head)` tuples.

## Device requirements

Every implementation variant requires `shader-f16`; the package has no variant-level fallback without that capability.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`attn-flash-decode-splitk-merge.wgsl.jinja`](build/webgpu/attn-flash-decode-splitk-merge.wgsl.jinja)
- [`paged-attention.wgsl.jinja`](build/webgpu/paged-attention.wgsl.jinja)
- [`paged-scatter-kv.wgsl.jinja`](build/webgpu/paged-scatter-kv.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.PagedAttention", { version: 1 });
const { keyCacheT, valueCacheT, outputT } = await kernel({
  queryT: { data: queryTData, shape: [2, 4] },
  keyT: { data: keyTData, shape: [2, 2] },
  valueT: { data: valueTData, shape: [2, 2] },
  keyCacheT: { data: keyCacheTData, shape: [3, 2, 1, 2] },
  valueCacheT: { data: valueCacheTData, shape: [3, 2, 1, 2] },
  cumulativeSequenceLengthT: { data: cumulativeSequenceLengthTData, shape: [2] },
  pastSeqlensT: { data: pastSeqlensTData, shape: [1] },
  blockTableT: { data: blockTableTData, shape: [1, 3] },
}, {
  attrs: { num_heads: 2, kv_num_heads: 1 },
});
```
