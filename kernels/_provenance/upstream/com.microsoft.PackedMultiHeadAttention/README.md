---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.PackedMultiHeadAttention

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Multi-head self-attention over a padding-removed token stream. The `token_offset` and `cumulative_sequence_length` schedule maps every packed token to its sequence, and a token attends only the keys of that sequence. Query is either packed `[T, N, 3, H]` with key and value absent, or a separate `[T, hidden]` triple. The optional attention bias is indexed in PADDED coordinates. Float32 and float16 only; bfloat16 is not implemented.

See the [ONNX Runtime `PackedMultiHeadAttention` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.PackedMultiHeadAttention) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `queryT` | `query` | `T` | same as logical dtype | — | — | Packed queries: `[token_count, num_heads, 3, head_size]` with `key` and `value` absent, or `[token_count, hidden_size]` alongside both. | required |
| `keyT` | `key` | `T` | same as logical dtype | `2` | — | Keys with shape `[token_count, hidden_size]`, matching `query` exactly. Present only in the separate form. | optional |
| `valueT` | `value` | `T` | same as logical dtype | `2` | — | Values with shape `[token_count, v_hidden_size]`. Present only in the separate form. | optional |
| `biasT` | `bias` | `T` | same as logical dtype | `1` | — | Optional packed input-projection bias with shape `[hidden_size + hidden_size + v_hidden_size]`, added to Q, K and V. | optional |
| `tokenOffsetT` | `token_offset` | `M` | `int32` | `2` | — | Shape `[batch_size, sequence_length]`. The first `token_count` entries hold each packed token's flat index in the padded grid; the rest hold the padding positions. This tensor carries the padded `sequence_length`. | required |
| `cumulativeSequenceLengthT` | `cumulative_sequence_length` | `M` | `int32` | `1` | — | Exclusive prefix sums with shape `[batch_size + 1]`, starting at 0 and ending at `token_count`. Sequence `i` owns packed tokens `[cum[i], cum[i + 1])`. Outputs are unspecified for a malformed schedule. | required |
| `attentionBiasT` | `attention_bias` | `T` | same as logical dtype | `4` | — | Optional additive score bias with shape `[batch_size or 1, num_heads or 1, sequence_length, sequence_length]`. Its trailing axes are PADDED positions inside a sequence, not packed indices. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `2` | derived | Attention output with shape `[token_count, v_hidden_size]`, in packed token order. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `num_heads` | — | Number of attention heads. Required; the hidden size must be divisible by it. |
| `scale` | — | Optional score scale. Omission, and the value `0`, both select `1 / sqrt(head_size)`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `M` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `packed_flash` — Tiled attention with independent Q/K and V slices and subgroup dot reductions. Equal wide heads target four `vec4` fragments per lane, bounded by subgroup/workgroup limits and native staging capacity. Unequal tiles cap work by token count and yield to serial or wider portable clusters when reuse is limited.
- `packed_flash_nosg` — Portable tiled attention with independent query/key and value slices. Lanes cover the larger width, with guarded zero Q/K fragments and shared reductions. Device limits bound lane count and staging. Unequal-width tiles yield to serial attention below the query-reuse target.
- `packed_serial` — Portable serial-key attention: one workgroup owns one (packed token, head), walks that token's own sequence and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.
- `packed_attn_bias_flash` — Tiled attention with independent Q/K and V slices and subgroup dot reductions. Equal wide heads target four `vec4` fragments per lane, bounded by subgroup/workgroup limits and native staging capacity. Unequal tiles cap work by token count and yield to serial or wider portable clusters when reuse is limited.
- `packed_attn_bias_flash_nosg` — Portable tiled attention with independent query/key and value slices. Lanes cover the larger width, with guarded zero Q/K fragments and shared reductions. Device limits bound lane count and staging. Unequal-width tiles yield to serial attention below the query-reuse target.
- `packed_attn_bias_serial` — Portable serial-key attention: one workgroup owns one (packed token, head), walks that token's own sequence and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.
- `packed_bias_flash` — Tiled attention with independent Q/K and V slices and subgroup dot reductions. Equal wide heads target four `vec4` fragments per lane, bounded by subgroup/workgroup limits and native staging capacity. Unequal tiles cap work by token count and yield to serial or wider portable clusters when reuse is limited.
- `packed_bias_flash_nosg` — Portable tiled attention with independent query/key and value slices. Lanes cover the larger width, with guarded zero Q/K fragments and shared reductions. Device limits bound lane count and staging. Unequal-width tiles yield to serial attention below the query-reuse target.
- `packed_bias_serial` — Portable serial-key attention: one workgroup owns one (packed token, head), walks that token's own sequence and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.
- `packed_bias_attn_bias_flash` — Tiled attention with independent Q/K and V slices and subgroup dot reductions. Equal wide heads target four `vec4` fragments per lane, bounded by subgroup/workgroup limits and native staging capacity. Unequal tiles cap work by token count and yield to serial or wider portable clusters when reuse is limited.
- `packed_bias_attn_bias_flash_nosg` — Portable tiled attention with independent query/key and value slices. Lanes cover the larger width, with guarded zero Q/K fragments and shared reductions. Device limits bound lane count and staging. Unequal-width tiles yield to serial attention below the query-reuse target.
- `packed_bias_attn_bias_serial` — Portable serial-key attention: one workgroup owns one (packed token, head), walks that token's own sequence and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.
- `separate_flash` — Tiled attention with independent Q/K and V slices and subgroup dot reductions. Equal wide heads target four `vec4` fragments per lane, bounded by subgroup/workgroup limits and native staging capacity. Unequal tiles cap work by token count and yield to serial or wider portable clusters when reuse is limited.
- `separate_flash_nosg` — Portable tiled attention with independent query/key and value slices. Lanes cover the larger width, with guarded zero Q/K fragments and shared reductions. Device limits bound lane count and staging. Unequal-width tiles yield to serial attention below the query-reuse target.
- `separate_serial` — Portable serial-key attention: one workgroup owns one (packed token, head), walks that token's own sequence and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.
- `separate_attn_bias_flash` — Tiled attention with independent Q/K and V slices and subgroup dot reductions. Equal wide heads target four `vec4` fragments per lane, bounded by subgroup/workgroup limits and native staging capacity. Unequal tiles cap work by token count and yield to serial or wider portable clusters when reuse is limited.
- `separate_attn_bias_flash_nosg` — Portable tiled attention with independent query/key and value slices. Lanes cover the larger width, with guarded zero Q/K fragments and shared reductions. Device limits bound lane count and staging. Unequal-width tiles yield to serial attention below the query-reuse target.
- `separate_attn_bias_serial` — Portable serial-key attention: one workgroup owns one (packed token, head), walks that token's own sequence and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.
- `separate_bias_flash` — Tiled attention with independent Q/K and V slices and subgroup dot reductions. Equal wide heads target four `vec4` fragments per lane, bounded by subgroup/workgroup limits and native staging capacity. Unequal tiles cap work by token count and yield to serial or wider portable clusters when reuse is limited.
- `separate_bias_flash_nosg` — Portable tiled attention with independent query/key and value slices. Lanes cover the larger width, with guarded zero Q/K fragments and shared reductions. Device limits bound lane count and staging. Unequal-width tiles yield to serial attention below the query-reuse target.
- `separate_bias_serial` — Portable serial-key attention: one workgroup owns one (packed token, head), walks that token's own sequence and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.
- `separate_bias_attn_bias_flash` — Tiled attention with independent Q/K and V slices and subgroup dot reductions. Equal wide heads target four `vec4` fragments per lane, bounded by subgroup/workgroup limits and native staging capacity. Unequal tiles cap work by token count and yield to serial or wider portable clusters when reuse is limited.
- `separate_bias_attn_bias_flash_nosg` — Portable tiled attention with independent query/key and value slices. Lanes cover the larger width, with guarded zero Q/K fragments and shared reductions. Device limits bound lane count and staging. Unequal-width tiles yield to serial attention below the query-reuse target.
- `separate_bias_attn_bias_serial` — Portable serial-key attention: one workgroup owns one (packed token, head), walks that token's own sequence and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.

## Device requirements

Some implementation variants require `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`attn-packed-varlen-cluster.wgsl.jinja`](build/webgpu/attn-packed-varlen-cluster.wgsl.jinja)
- [`attn-packed-varlen-scalar.wgsl.jinja`](build/webgpu/attn-packed-varlen-scalar.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.PackedMultiHeadAttention", { version: 1 });
const { outputT } = await kernel({
  queryT: { data: queryTData, shape: [1, 16] },
  keyT: { data: keyTData, shape: [1, 16] },
  valueT: { data: valueTData, shape: [1, 20] },
  tokenOffsetT: { data: tokenOffsetTData, shape: [1, 1] },
  cumulativeSequenceLengthT: { data: cumulativeSequenceLengthTData, shape: [2] },
}, {
  attrs: { num_heads: 1 },
});
```
