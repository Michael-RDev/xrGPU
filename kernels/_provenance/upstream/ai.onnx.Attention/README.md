---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Attention

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 24

## Description

Computes scaled dot-product attention as `softmax(Q * K^T * scale) * V`, with optional masks and softcap. Supports MHA, GQA, and MQA via `q_num_heads` and `kv_num_heads`, with 4D `(batch, heads, sequence, head_size)` or 3D `(batch, sequence, hidden)` inputs. Rank-4 routes support past/present KV caches and masks over the joined key sequence. This package implements Attention-24, including causal alignment for its internal `past_key` cache; it does not expose `nonpad_kv_seqlen` or Attention-25's `left_window_size` and `right_window_size` attributes.

See the [ONNX `Attention` spec](https://onnx.ai/onnx/operators/onnx__Attention.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `q` | `Q` | `T` | — | — | Query tensor; 4D `(batch, q_num_heads, q_seq, head_size)` or 3D `(batch, q_seq, q_hidden_size)`. | required |
| `k` | `K` | `T` | — | — | Key tensor; 4D `(batch, kv_num_heads, kv_seq, head_size)` or 3D `(batch, kv_seq, k_hidden_size)`. | required |
| `v` | `V` | `T` | — | — | Value tensor; 4D `(batch, kv_num_heads, kv_seq, v_head_size)` or 3D `(batch, kv_seq, v_hidden_size)`. | required |
| `attn_mask` | — | `M` | — | — | Optional attention mask broadcastable to `(batch, q_num_heads, q_seq, kv_seq)`; a true boolean permits attention, while a mask of the same floating-point type as `Q`, `K`, and `V` is added to the scores. | optional |
| `past_key` | — | `T` | `4` | — | Optional cached keys of shape `(batch, kv_num_heads, past_sequence_length, head_size)`; the cached tokens precede `K` on the sequence axis. | optional |
| `past_value` | — | `T` | `4` | — | Optional cached values of shape `(batch, kv_num_heads, past_sequence_length, v_head_size)`; supplied together with `past_key`. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `q` | derived | Attention output; 4D `(batch, q_num_heads, q_seq, v_head_size)` or 3D `(batch, q_seq, hidden_size)`. | required |
| `present_key` | — | `T` | `4` | derived | Optional joined keys, `past_key` followed by `K`, of shape `(batch, kv_num_heads, past_sequence_length + kv_sequence_length, head_size)`. | optional |
| `present_value` | — | `T` | `4` | derived | Optional joined values, `past_value` followed by `V`, of shape `(batch, kv_num_heads, past_sequence_length + kv_sequence_length, v_head_size)`. | optional |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `is_causal` | `0` | When set to `1`, applies causal masking. Without a cache, query index `i` attends keys through `i`; with `past_key`, it attends keys through `past_sequence_length + i`, aligning the mask to the query's absolute position in the joined cache. |
| `kv_num_heads` | — | Number of key/value attention heads; required when Q, K, V are 3D tensors (set equal to `q_num_heads` for MHA, or 1 for MQA). |
| `q_num_heads` | — | Number of query attention heads; required when Q, K, V are 3D tensors. |
| `qk_matmul_output_mode` | `0` | Selects the stage exposed by the optional QK-matmul output. This package does not expose that output yet, so only the default mode `0` is accepted. |
| `scale` | — | Scalar multiplier applied to `Q * K^T`; defaults to `1 / sqrt(head_size)`. |
| `softcap` | `0` | If non-zero, attention logits are capped via `tanh(logits / softcap) * softcap` before the softmax. |
| `softmax_precision` | — | Optional TensorProto element-type code for the softmax computation. This package supports FLOAT (`1`); FLOAT16 (`10`) is unsupported. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `M` | `float32`, `float16`, `bool` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `materialized_sgmat_head_chunk_f32` — Reuses float32 materialized scores for disjoint query-head groups; score arithmetic and accumulation match the unchunked subgroup-matrix route.
- `materialized_sgmat_f32` — Computes BSH/BHSD GQA with f32 matrix operands, scores, probabilities and accumulation, including upper-left causality. Half inputs widen on load and output narrows once. The flash route is used instead for half inputs whose head fits one key tile, and for large noncausal rank-4 query grids.
- `materialized_gemm_f32` — Computes unmasked noncausal BSH/BHSD GQA in three passes with f32 tiles, scores and accumulation for f16/f32 storage. Half output narrows once. Row reductions use power-of-two sizes within device limits. Flash and subgroup routes are used instead for some head widths and for query grids large enough to fill the device.
- `rank3_flash_tiled` — Reuses shared K/V tiles for token-major MHA and GQA with independent hidden strides and float32 accumulation. Per-lane private state has a tunable budget; device limits bound staging, invocations and dispatches. Portable clusters batch reductions for small effective query grids or long slices. Empty K/V use scalar bindings.
- `rank4_kv_cache_online` — Joins paired `past_key`/`past_value` inputs with `K`/`V`, retaining an omitted present cache in private intermediate storage when necessary, then runs the online rank-4 attention kernel over the joined tensors. Requested `present_key` and `present_value` outputs remain independently optional.
- `rank4_kv_cache_online_key_only` — Joins paired `past_key`/`past_value` inputs with `K`/`V`, retaining an omitted present cache in private intermediate storage when necessary, then runs the online rank-4 attention kernel over the joined tensors. Requested `present_key` and `present_value` outputs remain independently optional.
- `rank4_kv_cache_online_value_only` — Joins paired `past_key`/`past_value` inputs with `K`/`V`, retaining an omitted present cache in private intermediate storage when necessary, then runs the online rank-4 attention kernel over the joined tensors. Requested `present_key` and `present_value` outputs remain independently optional.
- `rank4_kv_cache_online_y_only` — Joins paired `past_key`/`past_value` inputs with `K`/`V`, retaining an omitted present cache in private intermediate storage when necessary, then runs the online rank-4 attention kernel over the joined tensors. Requested `present_key` and `present_value` outputs remain independently optional.
- `rank4_kv_cache_mask_online` — Joins paired `past_key`/`past_value` inputs with `K`/`V`, retaining an omitted present cache in private intermediate storage when necessary, then runs the online rank-4 attention kernel over the joined tensors. Requested `present_key` and `present_value` outputs remain independently optional.
- `rank4_kv_cache_mask_online_key_only` — Joins paired `past_key`/`past_value` inputs with `K`/`V`, retaining an omitted present cache in private intermediate storage when necessary, then runs the online rank-4 attention kernel over the joined tensors. Requested `present_key` and `present_value` outputs remain independently optional.
- `rank4_kv_cache_mask_online_value_only` — Joins paired `past_key`/`past_value` inputs with `K`/`V`, retaining an omitted present cache in private intermediate storage when necessary, then runs the online rank-4 attention kernel over the joined tensors. Requested `present_key` and `present_value` outputs remain independently optional.
- `rank4_kv_cache_mask_online_y_only` — Joins paired `past_key`/`past_value` inputs with `K`/`V`, retaining an omitted present cache in private intermediate storage when necessary, then runs the online rank-4 attention kernel over the joined tensors. Requested `present_key` and `present_value` outputs remain independently optional.
- `rank4_present_outputs_online` — Copies current K and V to the independently requested present cache outputs when no past cache is supplied, while the online rank-4 attention pass reads K and V directly.
- `rank4_present_outputs_mask_online` — Copies current K and V to the independently requested present cache outputs when no past cache is supplied, while the online rank-4 attention pass reads K and V directly.
- `rank4_present_key_online` — Copies the independently requested current key or value tensor to its present cache output when no past cache is supplied, while the online rank-4 attention pass reads K and V directly.
- `rank4_present_value_online` — Copies the independently requested current key or value tensor to its present cache output when no past cache is supplied, while the online rank-4 attention pass reads K and V directly.
- `rank4_present_key_mask_online` — Copies the independently requested current key or value tensor to its present cache output when no past cache is supplied, while the online rank-4 attention pass reads K and V directly.
- `rank4_present_value_mask_online` — Copies the independently requested current key or value tensor to its present cache output when no past cache is supplied, while the online rank-4 attention pass reads K and V directly.
- `flash_online_tiled_cluster_nosg` — Processes a tile of query rows against successive key tiles with online softmax, sharing K/V staging across the query tile without materializing the score matrix.
- `flash_online_tiled_mask_cluster_nosg` — Processes a tile of query rows against successive key tiles with online softmax, sharing K/V staging across the query tile without materializing the score matrix.
- `flash_online_tiled_cluster` — Processes a tile of query rows against successive key tiles with online softmax, sharing K/V staging across the query tile without materializing the score matrix.
- `flash_online_tiled_mask_cluster` — Processes a tile of query rows against successive key tiles with online softmax, sharing K/V staging across the query tile without materializing the score matrix.
- `rank4_no_mask_2pass` — Stores unnormalized softmax weights and row denominators, then applies them to V in a tiled second pass. This admits value-head widths beyond the generic online route's workgroup-storage bound.
- `rank4_mask_2pass` — Stores unnormalized softmax weights and row denominators, then applies them to V in a tiled second pass. This admits value-head widths beyond the generic online route's workgroup-storage bound.
- `flash_decode_splitk_nosg` — Partitions the key sequence across workgroups to create parallelism when a short-query grid is too small, then merges online-softmax state from scratch. The merge reassociates floating-point reductions relative to a single key sweep.
- `flash_online_tiled_decode_splitk` — Partitions the key sequence across workgroups to create parallelism when a short-query grid is too small, then merges online-softmax state from scratch. The merge reassociates floating-point reductions relative to a single key sweep.
- `flash_short_query_splitk_nosg` — Partitions the key sequence across workgroups to create parallelism when a short-query grid is too small, then merges online-softmax state from scratch. The merge reassociates floating-point reductions relative to a single key sweep.
- `flash_short_query_splitk` — Partitions the key sequence across workgroups to create parallelism when a short-query grid is too small, then merges online-softmax state from scratch. The merge reassociates floating-point reductions relative to a single key sweep.
- `flash_masked_decode_splitk` — Partitions the key sequence across workgroups to create parallelism when a short-query grid is too small, then merges online-softmax state from scratch. The merge reassociates floating-point reductions relative to a single key sweep.
- `flash_masked_short_query_splitk` — Partitions the key sequence across workgroups to create parallelism when a short-query grid is too small, then merges online-softmax state from scratch. The merge reassociates floating-point reductions relative to a single key sweep.
- `flash_cluster_splitkv` — Tiles query rows while partitioning a long key/value sequence across workgroups, increasing parallelism without materializing scores. It merges per-partition online-softmax state from scratch, which reassociates floating-point reductions relative to a single key sweep.
- `flash_cluster_splitkv_nosg` — Tiles query rows while partitioning a long key/value sequence across workgroups, increasing parallelism without materializing scores. It merges per-partition online-softmax state from scratch, which reassociates floating-point reductions relative to a single key sweep.
- `flash_cluster_splitkv_mask` — Tiles query rows while partitioning a long key/value sequence across workgroups, increasing parallelism without materializing scores. It merges per-partition online-softmax state from scratch, which reassociates floating-point reductions relative to a single key sweep.
- `flash_cluster_splitkv_mask_nosg` — Tiles query rows while partitioning a long key/value sequence across workgroups, increasing parallelism without materializing scores. It merges per-partition online-softmax state from scratch, which reassociates floating-point reductions relative to a single key sweep.

## Device requirements

Some implementation variants require `subgroup-matrix`, `shader-f16`, and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`attention-rank4-apply-tiled.wgsl.jinja`](build/webgpu/attention-rank4-apply-tiled.wgsl.jinja)
- [`attention-rank4-online.wgsl.jinja`](build/webgpu/attention-rank4-online.wgsl.jinja)
- [`attention-rank4-softmax.wgsl.jinja`](build/webgpu/attention-rank4-softmax.wgsl.jinja)
- [`attention-rank4-tiled.wgsl.jinja`](build/webgpu/attention-rank4-tiled.wgsl.jinja)
- [`attention-rank4.wgsl.jinja`](build/webgpu/attention-rank4.wgsl.jinja)
- [`attn-flash-decode-splitk-merge.wgsl.jinja`](build/webgpu/attn-flash-decode-splitk-merge.wgsl.jinja)
- [`attn-flash-decode-splitk.wgsl.jinja`](build/webgpu/attn-flash-decode-splitk.wgsl.jinja)
- [`attn-flash-online.wgsl.jinja`](build/webgpu/attn-flash-online.wgsl.jinja)
- [`attn-flash-prefill-cluster.wgsl.jinja`](build/webgpu/attn-flash-prefill-cluster.wgsl.jinja)
- [`attn-flash-q32-broadcast.wgsl.jinja`](build/webgpu/attn-flash-q32-broadcast.wgsl.jinja)
- [`attn-kv-cache-concat.wgsl.jinja`](build/webgpu/attn-kv-cache-concat.wgsl.jinja)
- [`attn-materialized-apply-f32.wgsl.jinja`](build/webgpu/attn-materialized-apply-f32.wgsl.jinja)
- [`attn-materialized-rowstats-combine-f32.wgsl.jinja`](build/webgpu/attn-materialized-rowstats-combine-f32.wgsl.jinja)
- [`attn-materialized-score-f32.wgsl.jinja`](build/webgpu/attn-materialized-score-f32.wgsl.jinja)
- [`attn-materialized-sgmat-f32.wgsl.jinja`](build/webgpu/attn-materialized-sgmat-f32.wgsl.jinja)
- [`attn-online-scalar.wgsl.jinja`](build/webgpu/attn-online-scalar.wgsl.jinja)
- [`datamove-flat-copy.wgsl.jinja`](build/webgpu/datamove-flat-copy.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Attention", { version: 1 });
const { y } = await kernel({
  q: { data: qData, shape: [1, 1, 1, 2] },
  k: { data: kData, shape: [1, 1, 2, 2] },
  v: { data: vData, shape: [1, 1, 2, 1] },
});
```
