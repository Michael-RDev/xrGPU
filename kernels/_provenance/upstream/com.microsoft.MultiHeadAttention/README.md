---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.MultiHeadAttention

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Computes multi-head self- or cross-attention over explicit query, key, and value tensors, with an optional fused QKV projection bias and additive attention bias. Supports causal masking through `unidirectional` and a configurable score scale that defaults to `1 / sqrt(head_size)`. Query, key, and value must have the same hidden width; distinct value-head widths, KV-cache, key-padding-mask, and diagnostic-QK modes are unsupported.

See the [ONNX Runtime `MultiHeadAttention` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MultiHeadAttention) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `queryT` | `query` | `T` | `3` | — | Query tensor of shape `(batch_size, sequence_length, hidden_size)`. | required |
| `keyT` | `key` | `T` | `3` | — | Key tensor of shape `(batch_size, kv_sequence_length, hidden_size)`; its hidden width must equal the query hidden width. | required |
| `valueT` | `value` | `T` | `3` | — | Value tensor of shape `(batch_size, kv_sequence_length, v_hidden_size)`; `v_hidden_size` must equal the query hidden width. | required |
| `biasT` | `bias` | `T` | `1` | — | Optional fused bias from input projection with shape `(hidden_size + hidden_size + v_hidden_size)`. | optional |
| `attentionBiasT` | `attention_bias` | `T` | `4` | — | Optional additive bias applied to the attention scores before softmax, of shape `(batch_size or 1, num_heads or 1, sequence_length, kv_sequence_length)`; the first two dimensions broadcast. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `3` | `[queryT[0], queryT[1], valueT[2]]` | Attention output of shape `(batch_size, sequence_length, v_hidden_size)`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `num_heads` | — | Number of attention heads. |
| `scale` | — | Optional score scale; zero or omission selects `1 / sqrt(head_size)`. |
| `unidirectional` | `0` | When 1, applies an upper-left causal mask: query position `i` attends only to key positions `0..min(i, kv_sequence_length - 1)`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `qkv_no_bias_small_head_value_subgroups` — Cooperatively reduces values across key lanes and reuses keys/values for two queries when device workgroup storage fits. Uses portable-width subgroup collectives with f32 scores, probabilities and accumulation. Short key sequences use the serial value path.
- `qkv_no_bias_small_head_value_tree` — Cooperatively reduces values across key lanes and reuses keys/values for two queries when device workgroup storage fits. Uses shared-memory tree reductions with f32 scores, probabilities and accumulation. Short key sequences use the serial value path.
- `qkv_bias_small_seq_blocked` — Blocked whole-head attention for short key sequences with bias and optional causal masking. A workgroup stages one head's K and V once, then assigns each query in a small block to a group of key lanes whose register-resident online-softmax partials are merged in one pass.
- `qkv_no_bias_small_seq_blocked` — Blocked whole-head attention for short key sequences without bias and with optional causal masking. A workgroup stages one head's K and V once, then assigns each query in a small block to a group of key lanes whose register-resident online-softmax partials are merged in one pass.
- `qkv_no_bias_small_seq` — Whole-head attention for short bidirectional float32 requests without bias. One workgroup stages a head's complete key and value planes, and each participating invocation owns one query row through scoring, normalization, and context accumulation.
- `qkv_no_bias_tiled_nosg` — Portable tiled prefill route for devices without an admissible subgroup path. It computes attention online without materializing the full score matrix.
- `qkv_no_bias_tiled_attn_bias_nosg` — Portable tiled prefill route for additive attention bias when no subgroup path is admissible. It computes attention online without materializing the full score matrix.
- `qkv_no_bias_materialized_sgmat_fused_f32` — Materializes FP32 scores for subgroup-matrix score and value products, then folds softmax normalization into the apply pass using row statistics emitted by the score pass.
- `qkv_bias_materialized_sgmat_fused_f32` — Materializes FP32 scores for subgroup-matrix score and value products, then folds softmax normalization into the apply pass using row statistics emitted by the score pass.
- `qkv_no_bias_materialized_sgmat_fused_f16` — Materializes FP32 scores for subgroup-matrix score and value products, then folds softmax normalization into the apply pass using row statistics emitted by the score pass.
- `qkv_no_bias_flash_cluster_lpq4_nosg` — Keeps query/key tiles and online-softmax state inside one workgroup instead of materializing all scores. The family supplies subgroup and portable forms for compatible prefill shapes.
- `qkv_no_bias_flash_cluster_nosg` — Keeps query/key tiles and online-softmax state inside one workgroup instead of materializing all scores. The family supplies subgroup and portable forms for compatible prefill shapes.
- `qkv_bias_flash_cluster_nosg` — Keeps query/key tiles and online-softmax state inside one workgroup instead of materializing all scores. The family supplies subgroup and portable forms for compatible prefill shapes.
- `qkv_no_bias_flash_cluster_lpq4` — Keeps query/key tiles and online-softmax state inside one workgroup instead of materializing all scores. The family supplies subgroup and portable forms for compatible prefill shapes.
- `qkv_no_bias_flash_cluster` — Keeps query/key tiles and online-softmax state inside one workgroup instead of materializing all scores. The family supplies subgroup and portable forms for compatible prefill shapes.
- `qkv_bias_flash_cluster` — Keeps query/key tiles and online-softmax state inside one workgroup instead of materializing all scores. The family supplies subgroup and portable forms for compatible prefill shapes.
- `qkv_no_bias_flash_cluster_attn_bias` — Keeps query/key tiles and online-softmax state inside one workgroup instead of materializing all scores. The family supplies subgroup and portable forms for compatible prefill shapes.
- `qkv_bias_flash_cluster_attn_bias` — Keeps query/key tiles and online-softmax state inside one workgroup instead of materializing all scores. The family supplies subgroup and portable forms for compatible prefill shapes.
- `qkv_no_bias_flash_splitk_nosg` — Partitions the key reduction across workgroups and merges partial softmax statistics and values. Used when decode or another short query has too few independent query/head workgroups; the merge reassociates the key reduction.
- `qkv_bias_flash_splitk_nosg` — Partitions the key reduction across workgroups and merges partial softmax statistics and values. Used when decode or another short query has too few independent query/head workgroups; the merge reassociates the key reduction.
- `qkv_no_bias_flash_splitk` — Partitions the key reduction across workgroups and merges partial softmax statistics and values. Used when decode or another short query has too few independent query/head workgroups; the merge reassociates the key reduction.
- `qkv_bias_flash_splitk` — Partitions the key reduction across workgroups and merges partial softmax statistics and values. Used when decode or another short query has too few independent query/head workgroups; the merge reassociates the key reduction.
- `qkv_no_bias_materialized_gemm_f32` — Materializes float32 scores and applies tiled score, softmax, and value passes without requiring subgroup-matrix support.
- `qkv_bias_materialized_gemm_f32` — Materializes float32 scores and applies tiled score, softmax, and value passes without requiring subgroup-matrix support.
- `qkv_no_bias_materialized_gemm_fused_f32` — Materializes FP32 scores while publishing each row's maximum, combines those row statistics, then normalizes and applies the scores in one pass. This route is eligible for score matrices above the configured size threshold.
- `qkv_bias_materialized_gemm_fused_f32` — Materializes FP32 scores while publishing each row's maximum, combines those row statistics, then normalizes and applies the scores in one pass. This route is eligible for score matrices above the configured size threshold.
- `qkv_no_bias_attn_bias` — Portable scalar online attention with bias and additive-mask support. It applies online softmax while accumulating context, without materializing a score tensor.
- `qkv_bias_attn_bias` — Portable scalar online attention with bias and additive-mask support. It applies online softmax while accumulating context, without materializing a score tensor.
- `qkv_no_bias` — Portable scalar online attention with bias and additive-mask support. It applies online softmax while accumulating context, without materializing a score tensor.
- `qkv_bias` — Portable scalar online attention with bias and additive-mask support. It applies online softmax while accumulating context, without materializing a score tensor.
- `qkv_no_bias_flash` — Computes one-pass online-softmax attention without score scratch. Family suffixes select subgroup, portable, and additive-attention-bias forms without changing the algorithm.
- `qkv_bias_flash` — Computes one-pass online-softmax attention with fused projection bias and no score scratch. Family suffixes select subgroup, portable, and additive-attention-bias forms without changing the algorithm.
- `qkv_no_bias_flash_nosg` — Computes one-pass online-softmax attention without score scratch. Family suffixes select subgroup, portable, and additive-attention-bias forms without changing the algorithm.
- `qkv_bias_flash_nosg` — Computes one-pass online-softmax attention with fused projection bias and no score scratch. Family suffixes select subgroup, portable, and additive-attention-bias forms without changing the algorithm.
- `qkv_no_bias_flash_attn_bias` — Computes one-pass online-softmax attention without score scratch. Family suffixes select subgroup, portable, and additive-attention-bias forms without changing the algorithm.
- `qkv_bias_flash_attn_bias` — Computes one-pass online-softmax attention with fused projection bias and no score scratch. Family suffixes select subgroup, portable, and additive-attention-bias forms without changing the algorithm.
- `qkv_no_bias_flash_attn_bias_nosg` — Computes one-pass online-softmax attention without score scratch. Family suffixes select subgroup, portable, and additive-attention-bias forms without changing the algorithm.
- `qkv_bias_flash_attn_bias_nosg` — Computes one-pass online-softmax attention with fused projection bias and no score scratch. Family suffixes select subgroup, portable, and additive-attention-bias forms without changing the algorithm.
- `qkv_no_bias_materialized_sgmat_f32` — Materializes FP32 scores for subgroup-matrix score and value products, with normalization performed by a separate softmax pass.
- `qkv_bias_materialized_sgmat_f32` — Materializes FP32 scores for subgroup-matrix score and value products, with normalization performed by a separate softmax pass.

## Device requirements

Some implementation variants require `subgroup-matrix`, `shader-f16`, and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`attention-rank4-tiled.wgsl.jinja`](build/webgpu/attention-rank4-tiled.wgsl.jinja)
- [`attn-flash-decode-splitk-merge.wgsl.jinja`](build/webgpu/attn-flash-decode-splitk-merge.wgsl.jinja)
- [`attn-flash-decode-splitk.wgsl.jinja`](build/webgpu/attn-flash-decode-splitk.wgsl.jinja)
- [`attn-flash-online.wgsl.jinja`](build/webgpu/attn-flash-online.wgsl.jinja)
- [`attn-flash-prefill-cluster.wgsl.jinja`](build/webgpu/attn-flash-prefill-cluster.wgsl.jinja)
- [`attn-flash-q32-broadcast.wgsl.jinja`](build/webgpu/attn-flash-q32-broadcast.wgsl.jinja)
- [`attn-materialized-apply-f32.wgsl.jinja`](build/webgpu/attn-materialized-apply-f32.wgsl.jinja)
- [`attn-materialized-rowstats-combine-f32.wgsl.jinja`](build/webgpu/attn-materialized-rowstats-combine-f32.wgsl.jinja)
- [`attn-materialized-score-f32.wgsl.jinja`](build/webgpu/attn-materialized-score-f32.wgsl.jinja)
- [`attn-materialized-sgmat-f32.wgsl.jinja`](build/webgpu/attn-materialized-sgmat-f32.wgsl.jinja)
- [`attn-materialized-softmax-f32.wgsl.jinja`](build/webgpu/attn-materialized-softmax-f32.wgsl.jinja)
- [`attn-online-scalar.wgsl.jinja`](build/webgpu/attn-online-scalar.wgsl.jinja)
- [`attn-small-head-parallel.wgsl.jinja`](build/webgpu/attn-small-head-parallel.wgsl.jinja)
- [`attn-small-head-value.wgsl.jinja`](build/webgpu/attn-small-head-value.wgsl.jinja)
- [`mha-small-seq-blocked.wgsl.jinja`](build/webgpu/mha-small-seq-blocked.wgsl.jinja)
- [`mha-small-seq.wgsl.jinja`](build/webgpu/mha-small-seq.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.MultiHeadAttention", { version: 1 });
const { outputT } = await kernel({
  queryT: { data: queryTData, shape: [2, 2, 4] },
  keyT: { data: keyTData, shape: [2, 3, 4] },
  valueT: { data: valueTData, shape: [2, 3, 4] },
}, {
  attrs: { num_heads: 2 },
});
```
