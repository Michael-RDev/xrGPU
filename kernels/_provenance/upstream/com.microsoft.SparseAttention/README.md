---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.SparseAttention

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Block-sparse causal attention used by Phi-3-small. `block_row_indices` and `block_col_indices` encode one or more CSR block masks, and layouts cycle over query heads. Grouped-query heads, separate or packed `[Q|K|V]`, explicit scaling, partial or full rotary embedding in NeoX or interleaved layout, and float16 are supported. The past/present key and value tensors share allocations and are updated in place. Head sizes must be non-zero multiples of 8; bfloat16 is not implemented.

See the [ONNX Runtime `SparseAttention` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SparseAttention) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `queryT` | `query` | `T` | same as logical dtype | `3` | — | Query `(batch_size, sequence_length, num_heads * head_size)`, or packed `[Q\|K\|V]` `(batch_size, sequence_length, (num_heads + 2 * kv_num_heads) * head_size)` when `key` and `value` are omitted. | required |
| `keyT` | `key` | `T` | same as logical dtype | `3` | — | Key `(batch_size, sequence_length, kv_num_heads * head_size)`. Omitted for packed QKV. | optional |
| `valueT` | `value` | `T` | same as logical dtype | `3` | — | Value `(batch_size, sequence_length, kv_num_heads * head_size)`. Omitted for packed QKV. | optional |
| `pastKeyT` | `past_key` | `T` | same as logical dtype | `4` | — | Key cache `(batch_size, kv_num_heads, max_cache_sequence_length, head_size)`, updated in place. | required |
| `pastValueT` | `past_value` | `T` | same as logical dtype | `4` | — | Value cache with the same shape as `past_key`, updated in place. | required |
| `blockRowIndicesT` | `block_row_indices` | `M` | `int32` | `2` | — | CSR row pointers `(num_layout, max_blocks + 1)`. Each layout starts at zero, is monotonically non-decreasing, and ends no later than that layout's `block_col_indices` width. | required |
| `blockColIndicesT` | `block_col_indices` | `M` | `int32` | `2` | — | CSR column indices `(num_layout, max_nnz_blocks)`, right-padded past each layout's non-zero count. Every active entry is in `[0, max_blocks)`. | required |
| `totalSequenceLengthT` | `total_sequence_length` | `M` | `int32` | — | — | Scalar or one-element vector holding the maximum total key length. Equal to `sequence_length` exactly in the prompt case, which is how the past length is decided. The value fits the cache, the sparse layout's `max_blocks * sparse_block_size` capacity, and the rotary-cache row count when rotary is enabled. | required |
| `keyTotalSequenceLengthsT` | `key_total_sequence_lengths` | `M` | `int32` | `1` | — | Per-batch total key length excluding padding, shape `(batch_size)`. Each value is at most `total_sequence_length` and is at least 1 for a prompt or at least `sequence_length` otherwise. | required |
| `cosCacheT` | `cos_cache` | `T` | same as logical dtype | `2` | — | Rotary cosine cache `(max_rotary_sequence_length, rotary_dimension / 2)`, where the width is a multiple of 8 no larger than `head_size / 2`. Required with `sin_cache` when `do_rotary` is 1. | optional |
| `sinCacheT` | `sin_cache` | `T` | same as logical dtype | `2` | — | Rotary sine cache with the same shape as `cos_cache`; required with it when `do_rotary` is 1. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `3` | derived | Attention output `(batch_size, sequence_length, num_heads * head_size)`. | required |
| `pastKeyT` | `past_key` | `T` | `4` | same as `pastKeyT` | The key cache tensor itself after the in-place append; ONNX names this output `present_key`. | required |
| `pastValueT` | `past_value` | `T` | `4` | same as `pastValueT` | The value cache tensor itself after the in-place append; ONNX names this output `present_value`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `do_rotary` | `0` | Set to 1 to apply rotary embedding to Q and to K before it enters the cache; every other value disables rotary embedding. |
| `kv_num_heads` | — | Number of key/value heads; must divide `num_heads`. |
| `num_heads` | — | Number of query heads. |
| `rotary_interleaved` | `0` | Set to 1 to rotate adjacent pairs instead of using the NeoX half-split; every other value selects the NeoX layout. |
| `scale` | — | Scale applied to query-key products; omitted or zero uses `1 / sqrt(head_size)`. |
| `sparse_block_size` | — | Tokens per sparse block; one of 16, 32, 64, 128. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `M` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `separate` — Portable online attention shares each key/value load across query rows. Spare lanes partition value accumulation, with the partition count bounded by the head width, workgroup size, and available workgroup storage. Single-partition staging requires one output column per lane and includes the staging array in its storage budget.
- `separate_sgmat` — Float32 subgroup-matrix online attention computes each score tile once, rescales output fragments through existing score scratch, and directly loads complete query tiles. Only a partial query tile is staged; key-tile width and fragment-bank capacity follow the reported workgroup-storage limit.
- `separate_sgmat_tail` — For a prompt, a thin query tail uses the shared portable template after complete matrix tiles. History calls retain the full matrix computation. Tail size and value partitions follow workgroup capacity. Heads narrower than its query tile use the full matrix path, since their shorter value contraction does not repay a separate portable dispatch.
- `separate_rotary` — Portable online attention shares each key/value load across query rows. Spare lanes partition value accumulation, with the partition count bounded by the head width, workgroup size, and available workgroup storage. Single-partition staging requires one output column per lane and includes the staging array in its storage budget.
- `separate_rotary_sgmat` — Float32 subgroup-matrix online attention computes each score tile once, rescales output fragments through existing score scratch, and directly loads complete query tiles. Only a partial query tile is staged; key-tile width and fragment-bank capacity follow the reported workgroup-storage limit.
- `separate_rotary_sgmat_tail` — For a prompt, a thin query tail uses the shared portable template after complete matrix tiles. History calls retain the full matrix computation. Tail size and value partitions follow workgroup capacity. Heads narrower than its query tile use the full matrix path, since their shorter value contraction does not repay a separate portable dispatch.
- `packed` — Portable online attention shares each key/value load across query rows. Spare lanes partition value accumulation, with the partition count bounded by the head width, workgroup size, and available workgroup storage. Single-partition staging requires one output column per lane and includes the staging array in its storage budget.
- `packed_sgmat` — Float32 subgroup-matrix online attention computes each score tile once, rescales output fragments through existing score scratch, and directly loads complete query tiles. Only a partial query tile is staged; key-tile width and fragment-bank capacity follow the reported workgroup-storage limit.
- `packed_sgmat_tail` — For a prompt, a thin query tail uses the shared portable template after complete matrix tiles. History calls retain the full matrix computation. Tail size and value partitions follow workgroup capacity. Heads narrower than its query tile use the full matrix path, since their shorter value contraction does not repay a separate portable dispatch.
- `packed_rotary` — Portable online attention shares each key/value load across query rows. Spare lanes partition value accumulation, with the partition count bounded by the head width, workgroup size, and available workgroup storage. Single-partition staging requires one output column per lane and includes the staging array in its storage budget.
- `packed_rotary_sgmat` — Float32 subgroup-matrix online attention computes each score tile once, rescales output fragments through existing score scratch, and directly loads complete query tiles. Only a partial query tile is staged; key-tile width and fragment-bank capacity follow the reported workgroup-storage limit.
- `packed_rotary_sgmat_tail` — For a prompt, a thin query tail uses the shared portable template after complete matrix tiles. History calls retain the full matrix computation. Tail size and value partitions follow workgroup capacity. Heads narrower than its query tile use the full matrix path, since their shorter value contraction does not repay a separate portable dispatch.

## Device requirements

Some implementation variants require `subgroup-matrix` and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`sparse-attention-sgmat.wgsl.jinja`](build/webgpu/sparse-attention-sgmat.wgsl.jinja)
- [`sparse-attention.wgsl.jinja`](build/webgpu/sparse-attention.wgsl.jinja)
- [`sparse-kv-append.wgsl.jinja`](build/webgpu/sparse-kv-append.wgsl.jinja)
- [`sparse-q-rotary.wgsl.jinja`](build/webgpu/sparse-q-rotary.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.SparseAttention", { version: 1 });
const { pastKeyT, pastValueT, outputT } = await kernel({
  queryT: { data: queryTData, shape: [1, 32, 8] },
  keyT: { data: keyTData, shape: [1, 32, 8] },
  valueT: { data: valueTData, shape: [1, 32, 8] },
  pastKeyT: { data: pastKeyTData, shape: [1, 1, 32, 8] },
  pastValueT: { data: pastValueTData, shape: [1, 1, 32, 8] },
  blockRowIndicesT: { data: blockRowIndicesTData, shape: [1, 3] },
  blockColIndicesT: { data: blockColIndicesTData, shape: [1, 3] },
  totalSequenceLengthT: { data: totalSequenceLengthTData, shape: [1] },
  keyTotalSequenceLengthsT: { data: keyTotalSequenceLengthsTData, shape: [1] },
}, {
  attrs: { num_heads: 1, kv_num_heads: 1, sparse_block_size: 16 },
});
```
