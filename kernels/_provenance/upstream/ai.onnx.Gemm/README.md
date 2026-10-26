---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Gemm

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Computes `Y = alpha * A' * B' + beta * C`, where `A'` and `B'` are optionally transposed views of the 2-D input matrices `A` and `B`, and `C` is an optional bias broadcastable to shape `(M, N)`. Output `Y` has shape `(M, N)`.

See the [ONNX `Gemm` spec](https://onnx.ai/onnx/operators/onnx__Gemm.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `T` | `2` | — | Input matrix `A` with shape `(M, K)` when `transA` is 0, or `(K, M)` when `transA` is non-zero. | required |
| `b` | `B` | `T` | `2` | — | Input matrix `B` with shape `(K, N)` when `transB` is 0, or `(N, K)` when `transB` is non-zero. | required |
| `bias` | `C` | `T` | — | — | Optional bias tensor broadcastable to `(M, N)`; treated as scalar 0 if absent. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | `2` | derived | Output matrix of shape `(M, N)`. | required |

## Runtime arguments

| Name | Kind | Upstream attribute | Description | Presence |
| --- | --- | --- | --- | --- |
| `M` | `u32` | — | Optional output-row count for the packed model-layout route; inferred from `A` for the standard matrix route. | optional |
| `inFeatures` | `u32` | `K` | Optional reduction dimension for the packed model-layout route; inferred from `A` and `B` for the standard matrix route. | optional |
| `outFeatures` | `u32` | `N` | Optional output-column count for the packed model-layout route; inferred from `B` for the standard matrix route. | optional |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `alpha` | `1` | Scalar multiplier applied to the matrix product A' * B'. |
| `beta` | `1` | Scalar multiplier applied to the optional bias tensor C. |
| `transA` | `0` | When non-zero, A is transposed before the multiplication. |
| `transB` | `0` | When non-zero, B is transposed before the multiplication. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `reg_generic_transA_f32_vec4` — Transposed-A f32 Gemm on the four-wide 128x64 register tile: A is stored `[K, M]`, so a lane stages a 4x4 block from four M-contiguous words and transposes it in registers into the packed K words the tile consumes.
- `reg_generic_transB_f16_vec4_wg8` — Uses an 8x8-lane f16 transposed-B register tile over 32x32 blocks for the configured 128x64 block-count band; other block counts use the wider tile.
- `reg_generic_notrans_vec4_splitk` — Cuts the K reduction of the no-transpose 128x64 f32 register tile into power-of-two slices across dispatch.z, with at least four K tiles per slice, then sums the f32 partials in a second pass.
- `model_subgroup_matrix_splitk` — Partitions the model-layout K reduction across subgroup-matrix workgroups, then combines float32 partials. The combine pass reassociates the reduction.
- `model_subgroup_matrix_bias_splitk` — Partitions the model-layout K reduction across subgroup-matrix workgroups, then combines float32 partials and adds the bias. The combine pass reassociates the reduction.
- `model_subgroup_matrix_m_tail_split` — Runs complete 64-row bands with direct A loads and stores, then computes the partial final band with guarded staging.
- `model_subgroup_matrix_n_tail_split` — Runs aligned output columns through a direct subgroup-matrix bulk pass and handles the remaining columns with a tail-safe pass. This keeps the bulk path free of edge guards when N is not tile-aligned.
- `model_subgroup_matrix` — Runs model-layout GEMM with subgroup matrices. Deep float32 reductions can use a 64-row tile even when the final row band is partial, provided it fits workgroup storage.
- `model_subgroup_matrix_bias` — Uses narrower subgroup-matrix column tiles for short reductions when the resulting workgroup count stays within the small-grid budget. The chosen tile must fit dispatch and workgroup-storage limits; larger grids retain the wider tile and bias is added once in the output stage.
- `model_band_transb` — Model-layout band for 2 to 32 rows with transposed weights. Each lane owns four columns and a group of rows, and each vec4 word from a K-contiguous weight row feeds the whole row band.
- `model_band_transb_splitk` — Splits the model-layout band's K axis into up to sixteen workgroup ranges. Each range writes f32 partials that a combine pass sums and adds to the bias.
- `model_band_transb_bias` — Model-layout band for 2 to 32 rows with transposed weights. Each lane owns four columns and a group of rows, and each vec4 word from a K-contiguous weight row feeds the whole row band.
- `model_band_transb_splitk_bias` — Splits the model-layout band's K axis into up to sixteen workgroup ranges. Each range writes f32 partials that a combine pass sums and adds to the bias.
- `subgroup_matrix_generic_transA_split_tails` — Computes the aligned interior with direct subgroup-matrix stores, then adds the K remainder and writes row and column fringes in correction passes. It keeps the subgroup-matrix route available when all three matrix extents have tails.
- `scalar_bias` — Portable tile for every broadcastable bias: scalar, full `(M, N)`, `(M, 1)` column and `(N)` row biases are shader branches, tested in that order.

## Device requirements

Some implementation variants require `subgroup-matrix`, `shader-f16`, and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`dense-tiled-matmul.wgsl.jinja`](build/webgpu/dense-tiled-matmul.wgsl.jinja)
- [`gemm-dense-f32-tiled-reg.wgsl.jinja`](build/webgpu/gemm-dense-f32-tiled-reg.wgsl.jinja)
- [`gemm-epilogue-tiled-reg.wgsl.jinja`](build/webgpu/gemm-epilogue-tiled-reg.wgsl.jinja)
- [`gemm-model-band-transb.wgsl.jinja`](build/webgpu/gemm-model-band-transb.wgsl.jinja)
- [`gemm-subgroup-matrix.wgsl.jinja`](build/webgpu/gemm-subgroup-matrix.wgsl.jinja)
- [`gemm-transa-tail-correction.wgsl.jinja`](build/webgpu/gemm-transa-tail-correction.wgsl.jinja)
- [`gemm.model.wgsl.jinja`](build/webgpu/gemm.model.wgsl.jinja)
- [`gemm.wgsl.jinja`](build/webgpu/gemm.wgsl.jinja)
- [`gemv-contiguous.wgsl.jinja`](build/webgpu/gemv-contiguous.wgsl.jinja)
- [`matmul-notrans-vec4-tiled-reg.wgsl.jinja`](build/webgpu/matmul-notrans-vec4-tiled-reg.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Gemm", { version: 1 });
const { y } = await kernel({ a: { data: aData, shape: [2, 4] }, b: { data: bData, shape: [4, 3] } });
```
