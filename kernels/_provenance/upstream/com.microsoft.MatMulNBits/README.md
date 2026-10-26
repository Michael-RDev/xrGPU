---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.MatMulNBits

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Matrix multiplication with `B` block-quantized along K and dequantized as `(code - zero_point) * scale`. Each power-of-two `block_size` group has a scale and optional zero point; optional bias is added afterward. Two-, four-, and eight-bit codes are packed low-first, and `A` may have rank 2 or 3. This package supports standard unpacked zero points with the same dtype as `A`. Deprecated `g_idx`, prepacked weights, and bfloat16 tensors are not implemented.

See the [ONNX Runtime `MatMulNBits` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `aT` | `A` | `T1` | — | — | Float input matrix, not quantized. Rank 2 has shape `(M, K)` and rank 3 has shape `(batch, sequence, K)`; only the last axis is the reduction axis and the leading axes fold into the row count, so the ordinary activation needs no surrounding Reshape. | required |
| `bT` | `B` | `uint8` | `3` | — | Bit-packed uint8 weight matrix of shape `(N, k_blocks, blob_size)`, where `k_blocks = ceil(K / block_size)` and `blob_size = block_size * bits / 8`. Codes are packed low-first along K. Bound in the packed storage layout: four blob bytes per u32 word, so the kernels stream the blob's own bytes rather than one widened word per byte. | required |
| `scalesT` | `scales` | `T1` | `2` | — | Per-block dequantization scale factors of shape `(N, k_blocks)`, with the same dtype as `A`. | required |
| `zeroPointsT` | `zero_points` | `T3` | `2` | — | Standard unpacked per-block zero points with shape `(N, k_blocks)` and the same dtype as `A`. Omission uses `2^(bits - 1)`. | optional |
| `biasT` | `bias` | `T1` | `1` | — | Optional bias vector of shape `[N]` added to the output. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `yT` | `Y` | `T1` | same as `aT` | derived | Result of A multiplied by the dequantized weight matrix, with optional bias, same dtype and rank as A: the leading axes of A with a trailing N. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `K` | — | Input feature dimension of the weight matrix. |
| `N` | — | Output feature dimension of the weight matrix. |
| `accuracy_level` | `0` | Minimum internal accuracy level: 0 (unset), 1 (float32), 2 (float16), 3 (bfloat16), or 4 (int8). |
| `bits` | `4` | Bit width used to quantize B; this package supports 2, 4, and 8. |
| `block_size` | — | Power-of-two quantization block size along K; it must be at least 16. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T1` | `float32`, `float16` |
| `T3` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `prefill_tiled_reg_vec4_splitk_default_zero` — Splits the vec4 register-tiled K reduction across dispatch.z, then combines f32 partials and bias. Large 2-bit f16 tiles pin the narrowest subgroup width of a variable range under subgroup-size control; fixed 32-lane paths reuse each packed word, scale, and zero point across four vectors.
- `prefill_tiled_reg_vec4_default_zero` — Register-blocked prefill with vec4 activation loads. Taller inputs use 128-row tiles. Large 2-bit f16 grids pin the narrowest width of a variable subgroup range under size control, with BK=16 on 16–32 lanes; other tiers use BK=32. Deep 2-bit reductions on fixed 32-lane subgroups unpack four vectors per packed word and reuse its scale and zero point.
- `prefill_tiled_reg_vec4_splitk_zero_bias` — Splits the vec4 register-tiled K reduction across dispatch.z, then combines f32 partials and bias. Large 2-bit f16 tiles pin the narrowest subgroup width of a variable range under subgroup-size control; fixed 32-lane paths reuse each packed word, scale, and zero point across four vectors.
- `prefill_tiled_reg_vec4_zero_bias` — Register-blocked prefill with vec4 activation loads. Taller inputs use 128-row tiles. Large 2-bit f16 grids pin the narrowest width of a variable subgroup range under size control, with BK=16 on 16–32 lanes; other tiers use BK=32. Deep 2-bit reductions on fixed 32-lane subgroups unpack four vectors per packed word and reuse its scale and zero point.
- `prefill_tiled_reg_vec4_splitk_zero_only` — Splits the vec4 register-tiled K reduction across dispatch.z, then combines f32 partials and bias. Large 2-bit f16 tiles pin the narrowest subgroup width of a variable range under subgroup-size control; fixed 32-lane paths reuse each packed word, scale, and zero point across four vectors.
- `prefill_tiled_reg_vec4_zero_only` — Register-blocked prefill with vec4 activation loads. Taller inputs use 128-row tiles. Large 2-bit f16 grids pin the narrowest width of a variable subgroup range under size control, with BK=16 on 16–32 lanes; other tiers use BK=32. Deep 2-bit reductions on fixed 32-lane subgroups unpack four vectors per packed word and reuse its scale and zero point.
- `prefill_tiled_reg_vec4_splitk_bias_only` — Splits the vec4 register-tiled K reduction across dispatch.z, then combines f32 partials and bias. Large 2-bit f16 tiles pin the narrowest subgroup width of a variable range under subgroup-size control; fixed 32-lane paths reuse each packed word, scale, and zero point across four vectors.
- `prefill_tiled_reg_vec4_bias_only` — Register-blocked prefill with vec4 activation loads. Taller inputs use 128-row tiles. Large 2-bit f16 grids pin the narrowest width of a variable subgroup range under size control, with BK=16 on 16–32 lanes; other tiers use BK=32. Deep 2-bit reductions on fixed 32-lane subgroups unpack four vectors per packed word and reuse its scale and zero point.

## Device requirements

Some implementation variants require `subgroup-matrix` and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`matmul-nbits-dp4a-quantize.wgsl.jinja`](build/webgpu/matmul-nbits-dp4a-quantize.wgsl.jinja)
- [`matmul-nbits-gemv-q4.wgsl.jinja`](build/webgpu/matmul-nbits-gemv-q4.wgsl.jinja)
- [`matmul-nbits-q4-dp4a-prefill.wgsl.jinja`](build/webgpu/matmul-nbits-q4-dp4a-prefill.wgsl.jinja)
- [`matmul-nbits-q4-prefill-tile4x4.wgsl.jinja`](build/webgpu/matmul-nbits-q4-prefill-tile4x4.wgsl.jinja)
- [`matmul-nbits-q4-prefill-tiled-reg.wgsl.jinja`](build/webgpu/matmul-nbits-q4-prefill-tiled-reg.wgsl.jinja)
- [`matmul-nbits-q4-prefill-tiled.wgsl.jinja`](build/webgpu/matmul-nbits-q4-prefill-tiled.wgsl.jinja)
- [`matmul-nbits-q4-sgmat.wgsl.jinja`](build/webgpu/matmul-nbits-q4-sgmat.wgsl.jinja)
- [`matmul-nbits.wgsl.jinja`](build/webgpu/matmul-nbits.wgsl.jinja)
- [`reduce-axis0-splitk-combine.wgsl.jinja`](build/webgpu/reduce-axis0-splitk-combine.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.MatMulNBits", { version: 1 });
const { yT } = await kernel({
  aT: { data: aTData, shape: [2, 17] },
  bT: { data: bTData, shape: [2, 2, 8] },
  scalesT: { data: scalesTData, shape: [2, 2] },
}, {
  attrs: { K: 17, N: 2, block_size: 16 },
});
```
