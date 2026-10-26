---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.TransposeMatMul

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Matrix product of two N-dimensional tensors `A` and `B`, following NumPy-style matrix-multiplication broadcasting, with optional transposition of either operand's last two dimensions and a scalar `alpha` multiplier. This is the strict subset of `FusedMatMul` that omits batch-dimension transposition, and it uses the same kernels. Float32 and float16 are supported; double and bfloat16 are not implemented.

See the [ONNX Runtime `TransposeMatMul` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.TransposeMatMul) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `A` | `T` | — | — | N-dimensional matrix A. | required |
| `B` | `T` | — | — | N-dimensional matrix B. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `Y` | `T` | derived | derived | Matrix-multiplication result whose shape follows NumPy-style rules after applying the requested batch and matrix transpositions. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `alpha` | `1` | Scalar multiplier applied to the product of the input tensors. |
| `transA` | `0` | When non-zero, transposes `A` on its last two dimensions before multiplication. |
| `transB` | `0` | When non-zero, transposes `B` on its last two dimensions before multiplication. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `broadcast_transb_tiled_reg` — Register-blocked broadcast product with physically transposed B. Reuses the shared batch-addressing tile, keeps f32 accumulation, and preserves scalar K order for f16. Low tile count and excessive padding demote this otherwise correct path.
- `broadcast_transb_subgroup_matrix_f16` — Broadcast transposed-B product using a supported 8x8x8 subgroup-matrix configuration with f32 accumulation. Logical shapes and physical B strides share the existing matrix engine; insufficient output tiles or excessive padding retain the generic tile.
- `broadcast_transb_subgroup_matrix_f32` — Broadcast transposed-B product using a supported 8x8x8 subgroup-matrix configuration with f32 accumulation. Logical shapes and physical B strides share the existing matrix engine; insufficient output tiles or excessive padding retain the generic tile.
- `m1_gemv_vec4` — Vector-by-matrix specialization for a single output row: each workgroup owns 32 consecutive vec4 column groups and partitions the reduction across the workgroup's second dimension. The accumulator stays float32 for both tensor types.
- `rank2_band_vec4_splitk` — Splits the vec4 band's K axis across up to sixteen workgroups. Each range writes an f32 partial band with alpha applied, and a combine pass sums the partials.
- `rank2_band_vec4` — Few-row band for a rank-2 product without transposes. Each lane owns one vec4 column group and one accumulator per row, so every B word feeds all 2 to 16 rows; alpha is applied at the store.
- `rank2_band_vec4_f32_preferred` — Few-row band for a rank-2 product without transposes. Each lane owns one vec4 column group and one accumulator per row, so every B word feeds all 2 to 16 rows; alpha is applied at the store.
- `subgroup_matrix_splitk` — Partitions the K reduction of small-M rank-2 products across subgroup-matrix workgroups, then combines float32 partials that already include alpha.
- `plain_rank2_tiled_reg` — Register-blocked rank-2 `Y = alpha * A @ B` specialization for non-transposed inputs on tiers without subgroup-matrix support.

## Device requirements

Some implementation variants require `subgroup-matrix` and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`fused-matmul-subgroup-matrix.wgsl.jinja`](build/webgpu/fused-matmul-subgroup-matrix.wgsl.jinja)
- [`matmul-band-vec4.wgsl.jinja`](build/webgpu/matmul-band-vec4.wgsl.jinja)
- [`matmul-subgroup-matrix-ext.wgsl.jinja`](build/webgpu/matmul-subgroup-matrix-ext.wgsl.jinja)
- [`matmul-tiled-general-reg.wgsl.jinja`](build/webgpu/matmul-tiled-general-reg.wgsl.jinja)
- [`matmul-tiled-general.wgsl.jinja`](build/webgpu/matmul-tiled-general.wgsl.jinja)
- [`matmul-vector-matrix-vec4.wgsl.jinja`](build/webgpu/matmul-vector-matrix-vec4.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/com.microsoft.TransposeMatMul", { version: 1 });
const { Y } = await kernel({ A: { data: AData, shape: [3] }, B: { data: BData, shape: [3] } });
```
