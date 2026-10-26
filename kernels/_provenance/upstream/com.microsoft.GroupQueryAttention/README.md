---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.GroupQueryAttention

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Grouped-query attention for explicit BSH Q/K/V and BNSH caches. Direct Q/K/V supports bidirectional attention or causal local windows and may store its generated float cache independently as float16 or float32; existing unquantized cache inputs match the Q/K/V dtype. Causal cache paths support rotary embeddings, sliding windows, bias, head sinks, softcap, smooth softmax, and paired Q/K RMS normalization. Int8/int4 caches require float32 Q/K/V and output; int4 is prompt-only. Packed QKV, position IDs, interleaved rotary, bfloat16/float8, and diagnostic QK output are not implemented.

See the [ONNX Runtime `GroupQueryAttention` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `queryT` | `query` | `T` | `3` | — | Query tensor of shape `(batch_size, sequence_length, num_heads * head_size)`. | required |
| `keyT` | `key` | `T` | `3` | — | Key tensor of shape `(batch_size, kv_sequence_length, kv_num_heads * head_size)`. | required |
| `valueT` | `value` | `T` | `3` | — | Value tensor of shape `(batch_size, kv_sequence_length, kv_num_heads * head_size)`. | required |
| `pastKeyT` | `past_key` | `T_CACHE` | `4` | — | Optional cached key state in BNSH format. Its sequence axis is `max_sequence_length` when the past and present buffers are shared, otherwise `past_sequence_length`; int4 stores each signed value as a +8-biased nibble, with the even head coordinate low, packing two values per logical uint8 element and widening each byte to one u32 WebGPU buffer word. | optional |
| `pastValueT` | `past_value` | `T_CACHE` | `4` | — | Optional cached value state in BNSH format with the same length and packing semantics as `past_key`. | optional |
| `seqlensKT` | `seqlens_k` | `M` | `1` | — | Int32 tensor of shape `(batch_size)` containing each sample's total sequence length minus one. | required |
| `totalSequenceLengthT` | `total_sequence_length` | `M` | `1` | — | Length-one int32 tensor containing the maximum total sequence length (past plus new) in the batch. | required |
| `cosCacheT` | `cos_cache` | `T` | `2` | — | Optional cosine cache for rotary embeddings with shape `(max_sequence_length, head_size / 2)`. | optional |
| `sinCacheT` | `sin_cache` | `T` | `2` | — | Optional sine cache for rotary embeddings with shape `(max_sequence_length, head_size / 2)`. | optional |
| `attentionBiasT` | `attention_bias` | `T` | `4` | — | Optional additive term for QK scores with shape `(batch_size or 1, num_heads or 1, sequence_length, total_sequence_length)`; the first two dimensions broadcast. | optional |
| `headSinkT` | `head_sink` | `T` | `1` | — | Optional per-head smooth factor of shape `(num_heads)` added to the softmax denominator. | optional |
| `kScaleT` | `k_scale` | `T_KV_SCALE` | `1` | — | Optional float32 key-cache scale: one value for `PER_TENSOR`, or `kv_num_heads * head_size` values for `PER_CHANNEL`. | optional |
| `vScaleT` | `v_scale` | `T_KV_SCALE` | `1` | — | Optional float32 value-cache scale with the same shape convention as `k_scale`. | optional |
| `qNormWeightT` | `q_norm_weight` | `T` | `1` | — | Optional per-head RMS-normalization weight of shape `(head_size)` applied to queries before rotary embedding. It must be provided together with `k_norm_weight`. | optional |
| `kNormWeightT` | `k_norm_weight` | `T` | `1` | — | Optional per-head RMS-normalization weight of shape `(head_size)` applied to keys before rotary embedding. It must be provided together with `q_norm_weight`. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `3` | same as `queryT` | Attention output of shape `(batch_size, sequence_length, hidden_size)`. | required |
| `presentKeyT` | `present_key` | `T_CACHE` | `4` | — | Updated key cache in BNSH format. Its sequence axis is `max_sequence_length` for a shared buffer, otherwise `past_sequence_length + kv_sequence_length`; int4 stores each signed value as a +8-biased nibble, with the even head coordinate low, packing two values per logical uint8 element and widening each byte to one u32 WebGPU buffer word. | required |
| `presentValueT` | `present_value` | `T_CACHE` | `4` | — | Updated value cache in BNSH format with the same length and packing semantics as `present_key`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `causal` | `1` | Whether to apply a causal mask. Set to 0 for bidirectional attention; `local_window_size` must then be -1. |
| `do_rotary` | `0` | Set to 1 to apply rotary position embeddings. The default 0 disables them. |
| `k_quant_type` | `"NONE"` | Key-cache quantization mode: `NONE`, `PER_TENSOR`, or `PER_CHANNEL`. |
| `kv_cache_bit_width` | — | Quantized cache bit width, either 8 or 4. Four-bit values are packed two per uint8 element. |
| `kv_num_heads` | — | Number of key/value attention heads. |
| `local_window_size` | `-1` | Left window size for causal local attention. The default -1 disables local attention, and the value must be -1 when `causal` is 0. |
| `num_heads` | — | Number of query attention heads. |
| `qk_norm_epsilon` | `0.000001` | Epsilon for the per-head Q/K RMS normalization applied when both normalization weights are provided. |
| `scale` | — | Optional QK score scale; zero or omission selects `1 / sqrt(head_size)`. |
| `sliding_window_cache` | `0` | Set to 1 when past/present caches are fixed-size window buffers that evict old tokens from the front. Requires `local_window_size > 0` and enough cache capacity. |
| `smooth_softmax` | `-1` | Set to 1 to enable the smooth-softmax denominator term. |
| `softcap` | `0` | Positive softcap applied to attention scores. The default 0 disables soft-capping. |
| `v_quant_type` | `"NONE"` | Value-cache quantization mode: `NONE`, `PER_TENSOR`, or `PER_CHANNEL`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `T_CACHE` | `float32`, `float16`, `uint8`, `int8` |
| `T_KV_SCALE` | `float32` |
| `M` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `qkv_present_materialized_sgmat_f32` — Materialized float32 subgroup-matrix prefill for bidirectional QKV without past state. `seqlens_k` is metadata-only; the score pass emits per-row softmax statistics, the apply pass normalizes and applies the scores, and the present copy preserves the supplied key and value tensors.
- `past_kv_materialized_sgmat_f32` — Materializes the causal score matrix with float32 subgroup-matrix tiles over the shared float cache, then applies the softmax-normalized weights with subgroup-matrix tiles. Both passes skip key tiles beyond each query tile's causal bound.
- `past_kv_bias_materialized_sgmat_f32` — Materializes causal scores with float32 subgroup-matrix tiles, broadcasts and adds the attention bias across batch and head dimensions, folds the biased scores into row statistics, and applies the normalized weights with subgroup-matrix tiles.
- `past_kv_materialized_sgmat_f16` — Materializes the causal score matrix with `f16` operand tiles feeding `f32`-accumulating subgroup matrices over the shared cache and applies the softmax-normalized weights the same way; scores and row statistics stay `f32`. The score pass skips key tiles past each query tile's causal bound and the apply pass stops its reduction there.
- `past_kv_rotary_materialized_sgmat_f32` — Rotates each query block at its absolute positions, then materializes causal scores against rotary-transformed cached keys and applies the normalized weights. The score and apply passes skip tiles beyond each query tile's causal bound.
- `past_kv_rotary_materialized_sgmat_f16` — Materializes the causal score matrix with `f16` operand tiles feeding `f32`-accumulating subgroup matrices over the shared cache and applies the softmax-normalized weights the same way; scores and row statistics stay `f32`. The score pass skips key tiles past each query tile's causal bound and the apply pass stops its reduction there.
- `new_kv_past_materialized_sgmat_f32` — For chunked prefill, merges the past cache with new key/value rows, materializes causal scores over the merged cache, and applies the normalized weights using the same right-aligned causal bound.
- `window_shift_materialized_sgmat_f32` — Materializes causal chunked-prefill attention with a windowed cache. The shift pass compacts surviving rows and appends the chunk; score and apply passes enforce both the sliding-window floor and causal bound.
- `share_append_materialized_sgmat_f32` — Materializes causal chunked-prefill attention while updating a shared-capacity cache in place. Score and apply passes bound every causal tile by the live length from `seqlens_k`, not the buffer capacity, so right-padded batches remain left-aligned.
- `new_kv_share_append_split` — Retains the existing cache and appends new key/value rows in separate passes before portable attention. It avoids rebuilding unchanged cache positions when the past and present allocations share capacity.
- `new_kv_share_append_bidirectional_split` — Bidirectional (`causal = 0`) attention over a shared-capacity cache: the new key/value rows are appended exactly as in the causal sibling, then every query attends to the whole live range `[0, seqlens_k[b] + 1)` instead of stopping at its own position. ONNX Runtime forbids `causal = 0` with a local window, so the window floor never applies here.
- `qkv_present_tiled_nosg` — Portable tiled prefill route that computes attention online and writes the present cache separately. It is used when the flash shape is valid but no suitable subgroup route is admissible.
- `qkv_present_flash_causal` — Causal prefill without a past cache and without a sliding window (`causal=1`, `local_window_size=-1`): the spelling onnxruntime's GroupQueryAttention tests use for a first prompt. The twin of `qkv_present_flash` with the causal ceiling enabled; the key sequence equals the query sequence, so the upper-left ceiling is the right one.
- `qkv_present_causal` — Causal prefill without a past cache and without a sliding window (`causal=1`, `local_window_size=-1`): the spelling onnxruntime's GroupQueryAttention tests use for a first prompt. The twin of `qkv_present` with the causal ceiling enabled; the key sequence equals the query sequence, so the upper-left ceiling is the right one.
- `quant_int8_decode_splitk` — Appends int8-quantized key/value rows, partitions cached decode across the key axis, and merges partial online-softmax results. Used when one workgroup per head exposes too little independent work.
- `qkv_present_flash_splitk` — Partitions direct Q/K/V attention across the key axis, merges partial online-softmax results, and writes the present cache separately. It serves short-query shapes needing more key-axis parallelism.
- `qkv_present_flash_cluster` — Computes clustered online-softmax prefill directly from Q/K/V and writes the present cache separately. The family provides subgroup and portable reductions for the same tiled algorithm.
- `past_kv_decode_splitk` — Updates the float cache in copy, merge, or window-shift mode, then partitions cached decode across the key axis and combines partial online-softmax results. The family includes subgroup and portable forms.
- `new_kv_past_decode_splitk` — Updates the float cache in copy, merge, or window-shift mode, then partitions cached decode across the key axis and combines partial online-softmax results. The family includes subgroup and portable forms.
- `window_shift_decode_splitk` — Updates the float cache in copy, merge, or window-shift mode, then partitions cached decode across the key axis and combines partial online-softmax results. The family includes subgroup and portable forms.
- `past_kv_flash_prefill` — Updates the float cache in the selected mode and applies clustered online-softmax causal prefill with the configured cache, mask, rotary, softcap, bias, or head-sink features.
- `new_kv_past_flash_prefill` — Updates the float cache in the selected mode and applies clustered online-softmax causal prefill with the configured cache, mask, rotary, softcap, bias, or head-sink features.
- `past_kv_rotary_flash_prefill` — Updates the float cache in the selected mode and applies clustered online-softmax causal prefill with the configured cache, mask, rotary, softcap, bias, or head-sink features.
- `past_kv_softcap_flash_prefill` — Updates the float cache in the selected mode and applies clustered online-softmax causal prefill with the configured cache, mask, rotary, softcap, bias, or head-sink features.
- `past_kv_headsink_flash_prefill` — Updates the float cache in the selected mode and applies clustered online-softmax causal prefill with the configured cache, mask, rotary, softcap, bias, or head-sink features.
- `past_kv_bias_headsink_flash_prefill` — Updates the float cache in the selected mode and applies clustered online-softmax causal prefill with the configured cache, mask, rotary, softcap, bias, or head-sink features.
- `window_shift_flash_prefill` — Updates the float cache in the selected mode and applies clustered online-softmax causal prefill with the configured cache, mask, rotary, softcap, bias, or head-sink features.
- `window_shift_rotary_append` — Windowed cache with rotary embeddings. The shift pass slides surviving rows down by the eviction count and clears what left the window; the append pass writes the new key/value rows and rotates K at its ABSOLUTE position (window origin plus row, not the cache row); attention rotates Q at the same absolute position and masks cache-relative.
- `window_shift_rotary_headsink_append` — Windowed rotary cache with a per-head softmax sink (the gpt-oss attention layer). The sink contributes `exp(sink - m)` to the denominator once per query row and nothing to the numerator.
- `window_shift_rotary_bias_append` — Windowed rotary cache with an additive attention bias. The bias row is indexed by ABSOLUTE key position with the tensor's own last dimension as the stride (ONNX spells it `total_sequence_length`, which keeps counting past the cache capacity), so resident cache row j reads bias column `origin + j`.
- `quant_int8_flash_prefill` — Builds an int8 or int4 cache and applies clustered online-softmax prefill directly from the packed values. The family supplies subgroup and portable reductions.
- `quant_int4_flash_prefill` — Builds an int8 or int4 cache and applies clustered online-softmax prefill directly from the packed values. The family supplies subgroup and portable reductions.
- `share_append_split_decode_splitk` — Retains shared-capacity cache rows, appends new rows separately, then partitions decode across the key axis. It avoids rebuilding unchanged cache positions while exposing split-key parallelism.
- `share_append_split_flash_prefill` — Retains shared-capacity cache rows, appends new rows separately, then applies clustered causal prefill. It avoids rebuilding unchanged cache positions and includes subgroup and portable forms.
- `share_append_bidirectional_flash_prefill` — Retains and appends a shared-capacity cache, then applies clustered bidirectional prefill to the full live key range.

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
- [`attn-materialized-rowstats-combine-f32.wgsl.jinja`](build/webgpu/attn-materialized-rowstats-combine-f32.wgsl.jinja)
- [`attn-materialized-sgmat-f32.wgsl.jinja`](build/webgpu/attn-materialized-sgmat-f32.wgsl.jinja)
- [`attn-online-scalar.wgsl.jinja`](build/webgpu/attn-online-scalar.wgsl.jinja)
- [`gqa-attention.wgsl.jinja`](build/webgpu/gqa-attention.wgsl.jinja)
- [`gqa-present.wgsl.jinja`](build/webgpu/gqa-present.wgsl.jinja)
- [`gqa-qprep.wgsl.jinja`](build/webgpu/gqa-qprep.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `presentKeyT`
- `presentValueT`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/com.microsoft.GroupQueryAttention", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { outputT, presentKeyT, presentValueT } = await kernel({
  queryT: { data: queryTData, shape: [1, 2, 8] },
  keyT: { data: keyTData, shape: [1, 2, 8] },
  valueT: { data: valueTData, shape: [1, 2, 8] },
  seqlensKT: { data: seqlensKTData, shape: [1] },
  totalSequenceLengthT: { data: totalSequenceLengthTData, shape: [1] },
}, {
  attrs: { num_heads: 1, kv_num_heads: 1 },
  outputs: {
    presentKeyT: { shape: [1, 1, 2, 8], dtype: "float32" },
    presentValueT: { shape: [1, 1, 2, 8], dtype: "float32" },
  },
});
```
