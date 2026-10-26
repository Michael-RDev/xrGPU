---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.PackedAttention

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Self-attention with a fused Q/K/V input projection over a padding-removed token stream. The `token_offset` and `cumulative_sequence_length` schedule maps every packed token to its sequence, and a token attends only the keys of that sequence; there is no mask input. The optional attention bias is indexed in PADDED coordinates. The output is 2-D in packed token order. Float32 and float16 only; bfloat16 is not implemented.

See the [ONNX Runtime `PackedAttention` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.PackedAttention) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `inputT` | `input` | `T` | same as logical dtype | `2` | — | Packed token embeddings with shape `[token_count, input_hidden_size]`. | required |
| `weightsT` | `weights` | `T` | same as logical dtype | `2` | — | Merged projection weights with shape `[input_hidden_size, q_hidden + k_hidden + v_hidden]`, column-ordered `[Q all heads \| K all heads \| V all heads]`. | required |
| `biasT` | `bias` | `T` | same as logical dtype | `1` | — | Required projection bias with shape `[q_hidden + k_hidden + v_hidden]`. | required |
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
| `num_heads` | — | Number of attention heads. Required; every Q/K/V hidden size must be divisible by it. |
| `qkv_hidden_sizes` | — | Optional `[q_hidden, k_hidden, v_hidden]`. Omission splits the bias length in three; `q_hidden` must equal `k_hidden`. |
| `scale` | — | Optional score scale. Omission, and the value `0`, both select `1 / sqrt(head_size)`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `M` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `matrix_direct_f32_flash` — Projects with dtype-preserving matrix tiles and direct input reads; a gated staged pass covers partial rows. Attention uses subgroup or portable tiled reductions according to its own resource and lane budget.
- `matrix_staged_f32_flash` — Projects with staged 32-row matrix tiles and float32 accumulation, then runs subgroup or portable tiled attention. Small projection grids avoid a separate tail dispatch; both passes retain independent resource guards.
- `f32_flash` — Fuses the Q/K/V projection into a register-tiled GEMM, then runs tiled flash attention over the packed token stream: one workgroup covers `CLUSTER_TILE_Q` consecutive tokens for one head and stages `CLUSTER_TILE_K` keys, masking each token to its own sequence. Cluster dot fragments combine through subgroup shuffles.
- `matrix_direct_f32_flash_nosg` — Projects with dtype-preserving matrix tiles and direct input reads; a gated staged pass covers partial rows. Attention uses subgroup or portable tiled reductions according to its own resource and lane budget.
- `matrix_staged_f32_flash_nosg` — Projects with staged 32-row matrix tiles and float32 accumulation, then runs subgroup or portable tiled attention. Small projection grids avoid a separate tail dispatch; both passes retain independent resource guards.
- `f32_flash_nosg` — Projects Q/K/V with the register-tiled GEMM, then performs packed flash attention. Portable clusters distribute dot fragments and softmax work across lanes. Wider heads use more lanes and fewer queries per workgroup; device limits bound geometry and key staging. Four-lane clusters retain the existing reduction.
- `matrix_direct_f32_serial` — Projects with dtype-preserving subgroup matrices, direct input reads and double-buffered weights. Float32 uses four accumulation streams. A gated staged pass handles partial rows; sparse grids use smaller staged tiles.
- `matrix_staged_f32_serial` — Projects with staged 32-row subgroup-matrix tiles and float32 accumulation before packed attention. Small and partial-row grids avoid an extra projection dispatch; reported matrix configurations, subgroup width and resource limits retain portable fallbacks.
- `f32_serial` — Fuses the Q/K/V projection into a register-tiled GEMM, then runs portable serial-key attention: one workgroup owns one (packed token, head) and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.
- `matrix_direct_f32_attn_bias_flash` — Projects with dtype-preserving matrix tiles and direct input reads; a gated staged pass covers partial rows. Attention uses subgroup or portable tiled reductions according to its own resource and lane budget.
- `matrix_staged_f32_attn_bias_flash` — Projects with staged 32-row matrix tiles and float32 accumulation, then runs subgroup or portable tiled attention. Small projection grids avoid a separate tail dispatch; both passes retain independent resource guards.
- `f32_attn_bias_flash` — Fuses the Q/K/V projection into a register-tiled GEMM, then runs tiled flash attention over the packed token stream: one workgroup covers `CLUSTER_TILE_Q` consecutive tokens for one head and stages `CLUSTER_TILE_K` keys, masking each token to its own sequence. Cluster dot fragments combine through subgroup shuffles.
- `matrix_direct_f32_attn_bias_flash_nosg` — Projects with dtype-preserving matrix tiles and direct input reads; a gated staged pass covers partial rows. Attention uses subgroup or portable tiled reductions according to its own resource and lane budget.
- `matrix_staged_f32_attn_bias_flash_nosg` — Projects with staged 32-row matrix tiles and float32 accumulation, then runs subgroup or portable tiled attention. Small projection grids avoid a separate tail dispatch; both passes retain independent resource guards.
- `f32_attn_bias_flash_nosg` — Projects Q/K/V with the register-tiled GEMM, then performs packed flash attention. Portable clusters distribute dot fragments and softmax work across lanes. Wider heads use more lanes and fewer queries per workgroup; device limits bound geometry and key staging. Four-lane clusters retain the existing reduction.
- `matrix_direct_f32_attn_bias_serial` — Projects with dtype-preserving subgroup matrices, direct input reads and double-buffered weights. Float32 uses four accumulation streams. A gated staged pass handles partial rows; sparse grids use smaller staged tiles.
- `matrix_staged_f32_attn_bias_serial` — Projects with staged 32-row subgroup-matrix tiles and float32 accumulation before packed attention. Small and partial-row grids avoid an extra projection dispatch; reported matrix configurations, subgroup width and resource limits retain portable fallbacks.
- `f32_attn_bias_serial` — Fuses the Q/K/V projection into a register-tiled GEMM, then runs portable serial-key attention: one workgroup owns one (packed token, head) and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.
- `matrix_direct_f16_flash` — Projects with dtype-preserving matrix tiles and direct input reads; a gated staged pass covers partial rows. Attention uses subgroup or portable tiled reductions according to its own resource and lane budget.
- `matrix_staged_f16_flash` — Projects with staged 32-row matrix tiles and float32 accumulation, then runs subgroup or portable tiled attention. Small projection grids avoid a separate tail dispatch; both passes retain independent resource guards.
- `f16_flash` — Fuses the Q/K/V projection into a register-tiled GEMM, then runs tiled flash attention over the packed token stream: one workgroup covers `CLUSTER_TILE_Q` consecutive tokens for one head and stages `CLUSTER_TILE_K` keys, masking each token to its own sequence. Cluster dot fragments combine through subgroup shuffles.
- `matrix_direct_f16_flash_nosg` — Projects with dtype-preserving matrix tiles and direct input reads; a gated staged pass covers partial rows. Attention uses subgroup or portable tiled reductions according to its own resource and lane budget.
- `matrix_staged_f16_flash_nosg` — Projects with staged 32-row matrix tiles and float32 accumulation, then runs subgroup or portable tiled attention. Small projection grids avoid a separate tail dispatch; both passes retain independent resource guards.
- `f16_flash_nosg` — Projects Q/K/V with the register-tiled GEMM, then performs packed flash attention. Portable clusters distribute dot fragments and softmax work across lanes. Wider heads use more lanes and fewer queries per workgroup; device limits bound geometry and key staging. Four-lane clusters retain the existing reduction.
- `matrix_direct_f16_serial` — Projects with dtype-preserving subgroup matrices, direct input reads and double-buffered weights. Float32 uses four accumulation streams. A gated staged pass handles partial rows; sparse grids use smaller staged tiles.
- `matrix_staged_f16_serial` — Projects with staged 32-row subgroup-matrix tiles and float32 accumulation before packed attention. Small and partial-row grids avoid an extra projection dispatch; reported matrix configurations, subgroup width and resource limits retain portable fallbacks.
- `f16_serial` — Fuses the Q/K/V projection into a register-tiled GEMM, then runs portable serial-key attention: one workgroup owns one (packed token, head) and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.
- `matrix_direct_f16_attn_bias_flash` — Projects with dtype-preserving matrix tiles and direct input reads; a gated staged pass covers partial rows. Attention uses subgroup or portable tiled reductions according to its own resource and lane budget.
- `matrix_staged_f16_attn_bias_flash` — Projects with staged 32-row matrix tiles and float32 accumulation, then runs subgroup or portable tiled attention. Small projection grids avoid a separate tail dispatch; both passes retain independent resource guards.
- `f16_attn_bias_flash` — Fuses the Q/K/V projection into a register-tiled GEMM, then runs tiled flash attention over the packed token stream: one workgroup covers `CLUSTER_TILE_Q` consecutive tokens for one head and stages `CLUSTER_TILE_K` keys, masking each token to its own sequence. Cluster dot fragments combine through subgroup shuffles.
- `matrix_direct_f16_attn_bias_flash_nosg` — Projects with dtype-preserving matrix tiles and direct input reads; a gated staged pass covers partial rows. Attention uses subgroup or portable tiled reductions according to its own resource and lane budget.
- `matrix_staged_f16_attn_bias_flash_nosg` — Projects with staged 32-row matrix tiles and float32 accumulation, then runs subgroup or portable tiled attention. Small projection grids avoid a separate tail dispatch; both passes retain independent resource guards.
- `f16_attn_bias_flash_nosg` — Projects Q/K/V with the register-tiled GEMM, then performs packed flash attention. Portable clusters distribute dot fragments and softmax work across lanes. Wider heads use more lanes and fewer queries per workgroup; device limits bound geometry and key staging. Four-lane clusters retain the existing reduction.
- `matrix_direct_f16_attn_bias_serial` — Projects with dtype-preserving subgroup matrices, direct input reads and double-buffered weights. Float32 uses four accumulation streams. A gated staged pass handles partial rows; sparse grids use smaller staged tiles.
- `matrix_staged_f16_attn_bias_serial` — Projects with staged 32-row subgroup-matrix tiles and float32 accumulation before packed attention. Small and partial-row grids avoid an extra projection dispatch; reported matrix configurations, subgroup width and resource limits retain portable fallbacks.
- `f16_attn_bias_serial` — Fuses the Q/K/V projection into a register-tiled GEMM, then runs portable serial-key attention: one workgroup owns one (packed token, head) and reduces each dot through workgroup memory. Head dimensions need no alignment and the value head dimension may differ.

## Device requirements

Some implementation variants require `subgroup-matrix` and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`attn-packed-varlen-cluster.wgsl.jinja`](build/webgpu/attn-packed-varlen-cluster.wgsl.jinja)
- [`attn-packed-varlen-scalar.wgsl.jinja`](build/webgpu/attn-packed-varlen-scalar.wgsl.jinja)
- [`gemm-epilogue-tiled-reg.wgsl.jinja`](build/webgpu/gemm-epilogue-tiled-reg.wgsl.jinja)
- [`gemm-subgroup-matrix.wgsl.jinja`](build/webgpu/gemm-subgroup-matrix.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.PackedAttention", { version: 1 });
const { outputT } = await kernel({
  inputT: { data: inputTData, shape: [2, 4] },
  weightsT: { data: weightsTData, shape: [4, 12] },
  biasT: { data: biasTData, shape: [12] },
  tokenOffsetT: { data: tokenOffsetTData, shape: [1, 2] },
  cumulativeSequenceLengthT: { data: cumulativeSequenceLengthTData, shape: [2] },
}, {
  attrs: { num_heads: 2 },
});
```
