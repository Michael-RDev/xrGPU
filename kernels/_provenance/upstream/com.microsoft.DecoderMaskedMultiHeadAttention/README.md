---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.DecoderMaskedMultiHeadAttention

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Single-token decode attention, self and cross, excluding the QKV projection. Cross attention reads whole encoder key and value tensors; self attention appends this token to a capacity-sized cache whose live length arrives in `past_sequence_length`. The mask is additive: a zero entry adds `mask_filter_value` rather than removing the key. The optional `qk` output is the scaled score row before the softmax, and is produced for cross attention only.

See the [ONNX Runtime `DecoderMaskedMultiHeadAttention` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DecoderMaskedMultiHeadAttention) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `queryT` | `query` | `T` | `3` | — | Query row for the single decoded token, with shape `(batch_size, 1, num_heads * head_size)`. | required |
| `keyT` | `key` | `T` | — | — | Cross attention: the full encoder keys with shape `(batch_size, num_heads, kv_sequence_length, head_size)`. Self attention: this token's key row with shape `(batch_size, 1, num_heads * head_size)`. | required |
| `valueT` | `value` | `T` | — | — | Cross attention: the full encoder values, shaped like `key`. Self attention: this token's value row, shaped like `key`. | required |
| `maskIndexT` | `mask_index` | `M` | `2` | — | Key-padding mask with shape `(batch_size, total_sequence_length)`. A zero entry ADDS `mask_filter_value` to that key's score; it does not remove the key. | optional |
| `attentionBiasT` | `attention_bias` | `T` | `4` | — | Additive score bias with shape `(batch_size or 1, num_heads or 1, 1, total_sequence_length)`. Self attention only. | optional |
| `pastKeyT` | `past_key` | `T` | `4` | — | Key cache at capacity, with shape `(batch_size, num_heads, max_sequence_length, head_size)`. Its presence selects self attention. | optional |
| `pastValueT` | `past_value` | `T` | `4` | — | Value cache at capacity, shaped like `past_key`. | optional |
| `pastSequenceLengthT` | `past_sequence_length` | `M` | `1` | — | Live cache length before this token, as a single value. It selects the cache row this token is written to and bounds the key loop, so it is read from tensor data rather than baked. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `3` | `[queryT[0], 1, queryT[2]]` | Attention result with shape `(batch_size, 1, num_heads * head_size)`. | required |
| `presentKeyT` | `present_key` | `T` | `4` | same as `pastKeyT` | Key cache after appending this token, at capacity. Every row other than the appended one keeps its past value. | optional |
| `presentValueT` | `present_value` | `T` | `4` | same as `pastValueT` | Value cache after appending this token, shaped like `present_key`. | optional |
| `qkT` | `qk` | `QK` | `4` | `[queryT[0], num_heads, 1, keyT[2]]` | Scaled `Q * K^T` with shape `(batch_size, num_heads, 1, kv_sequence_length)`, taken after the mask and BEFORE the softmax despite the schema calling it normalized. Cross attention only. | optional |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `mask_filter_value` | `-10000` | Value added to the score of a key whose mask entry is zero. |
| `num_heads` | — | Number of attention heads. The head size is the query hidden size divided by this count. |
| `output_qk` | — | Set to 1 to emit the pre-softmax `qk` output. It must agree with whether the caller supplies that output. |
| `past_present_share_buffer` | — | Set to 1 when the past and present caches are one capacity-sized buffer. Self attention requires it; cross attention has no cache and leaves it at 0. |
| `scale` | — | Multiplier applied to `Q * K^T`. Omitted or zero selects the default `1 / sqrt(head_size)`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `QK` | `float32`, `float16` |
| `M` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `cross_direct` — A single key partition writes normalized attention directly, removing partial buffers and the merge dispatch while preserving masks and diagnostic-score outputs.
- `cross_qk_direct` — A single key partition writes normalized attention directly, removing partial buffers and the merge dispatch while preserving masks and diagnostic-score outputs.
- `cross_mask_direct` — A single key partition writes normalized attention directly, removing partial buffers and the merge dispatch while preserving masks and diagnostic-score outputs.
- `cross_mask_qk_direct` — A single key partition writes normalized attention directly, removing partial buffers and the merge dispatch while preserving masks and diagnostic-score outputs.
- `cross_nosg_direct` — A single key partition writes normalized attention directly, removing partial buffers and the merge dispatch while preserving masks and diagnostic-score outputs.
- `cross_qk_nosg_direct` — A single key partition writes normalized attention directly, removing partial buffers and the merge dispatch while preserving masks and diagnostic-score outputs.
- `cross_mask_nosg_direct` — A single key partition writes normalized attention directly, removing partial buffers and the merge dispatch while preserving masks and diagnostic-score outputs.
- `cross_mask_qk_nosg_direct` — A single key partition writes normalized attention directly, removing partial buffers and the merge dispatch while preserving masks and diagnostic-score outputs.

## Device requirements

Some implementation variants require `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`attn-flash-decode-splitk-merge.wgsl.jinja`](build/webgpu/attn-flash-decode-splitk-merge.wgsl.jinja)
- [`attn-flash-decode-splitk.wgsl.jinja`](build/webgpu/attn-flash-decode-splitk.wgsl.jinja)
- [`dmmha-present.wgsl.jinja`](build/webgpu/dmmha-present.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.DecoderMaskedMultiHeadAttention", { version: 1 });
const { outputT } = await kernel({
  queryT: { data: queryTData, shape: [1, 1, 32] },
  keyT: { data: keyTData, shape: [1, 2, 1, 16] },
  valueT: { data: valueTData, shape: [1, 2, 1, 16] },
}, {
  attrs: { num_heads: 2 },
});
```
