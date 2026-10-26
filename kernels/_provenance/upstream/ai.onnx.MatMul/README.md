---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.MatMul

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 9

## Description

Computes the matrix product `A * B` with ONNX `MatMul` semantics, including vector promotion and NumPy-style broadcasting over batch dimensions. This package supports float16, float32, int32, and uint32 tensors. All legal rank-1 through rank-4 combinations are implemented; rank-5 is supported only when both inputs have rank 5. Mixed-rank combinations involving rank 5, higher-rank inputs, and other ONNX-supported types are unsupported.

See the [ONNX `MatMul` spec](https://onnx.ai/onnx/operators/onnx__MatMul.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `T` | — | — | N-dimensional left-hand matrix. | required |
| `b` | `B` | `T` | — | — | N-dimensional right-hand matrix. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | derived | ONNX MatMul result of `a` and `b` | Matrix multiply result of `A * B`, including vector promotion and broadcast batch dimensions. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `subgroup_matrix` — Computes padded tiles with subgroup matrices and float32 accumulation. An `N` that is not a multiple of the 64-wide column tile clamps the trailing tile's `B` columns to `N - 1` and drops every store at or past `N`. Not selected when row padding and per-split `A` reuse meet the split-band thresholds, or when the padded column ratio is too high.
- `subgroup_matrix_splitk` — Partitions K across subgroup-matrix workgroups, then combines float32 partials, reassociating the reduction. Selected for heavily padded row tiles when the split band's scaled dispatch population and per-split `A` footprint fit.
- `folded_rows_subgroup_matrix_splitk` — Treats rank-3 A with a unit row axis as a contiguous `[M, K]` view, partitions K across subgroup-matrix workgroups, and combines float32 partials.
- `rank2_notrans_f32_vec4_tiled_reg_splitk` — Cuts the K reduction of the 128x64 f32 register tile into power-of-two slices across dispatch.z, with at least four K tiles per slice, then sums the f32 partials in a second pass.
- `rank2_m1_vector_matrix_tail1_f16` — Computes one f16 output per invocation at the true row stride, prefetching bounded groups of K operands before ordered f32 accumulation and one final f16 conversion.
- `rank2_band_vec4_splitk` — Splits the vec4 band's K axis into up to sixteen workgroup ranges. Each range writes an f32 partial band that a combine pass sums.
- `batched_band_vec4` — Vec4 band route for equal-batch rank-3/rank-4 operands with 2..BAND_VEC4_MAX_ROWS rows per matrix. One workgroup row handles each matrix, and K slices are derived from the batch-times-column-group count.
- `vector_matrix_vec4` — One output row times a rank-2 `B`: a rank-1 `A`, a one-row rank-2 `A`, or a batched `A` whose rows fold to one. Each lane owns four output columns.
- `matrix_vector_tail_subgroup` — Assigns one output row to each workgroup and uses scalar loads for a reduction width that is not vec4-aligned, then folds the row with a subgroup reduction, plus an in-order fold of subgroup sums when a subgroup is narrower than the row. It is the parallel large-matrix-vector route on subgroup-capable devices.
- `matrix_vector_tail_nosg` — Uses the same tail-safe, one-workgroup-per-row traversal but folds partials through workgroup memory. It applies when subgroups are unavailable.
- `dot_splitk` — Partitions a long vector dot product across workgroups and combines float32 partials to expose parallelism beyond one workgroup. The combine reassociates the sum, so this route is not bit-identical to the single-workgroup dot route.
- `dot` — Folds a vector dot product inside one workgroup: every lane strides the vectors, then a shared-memory tree fold sums the lane partials. Integer inputs accumulate in their own type; half inputs widen to float32.

## Device requirements

Some implementation variants require `subgroup-matrix`, `shader-f16`, and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`dot-splitk-partial.wgsl.jinja`](build/webgpu/dot-splitk-partial.wgsl.jinja)
- [`matmul-band-vec4.wgsl.jinja`](build/webgpu/matmul-band-vec4.wgsl.jinja)
- [`matmul-batched-vector.wgsl.jinja`](build/webgpu/matmul-batched-vector.wgsl.jinja)
- [`matmul-batched.wgsl.jinja`](build/webgpu/matmul-batched.wgsl.jinja)
- [`matmul-dot.wgsl.jinja`](build/webgpu/matmul-dot.wgsl.jinja)
- [`matmul-matrix-vector-subgroup.wgsl.jinja`](build/webgpu/matmul-matrix-vector-subgroup.wgsl.jinja)
- [`matmul-notrans-vec4-tiled-reg.wgsl.jinja`](build/webgpu/matmul-notrans-vec4-tiled-reg.wgsl.jinja)
- [`matmul-subgroup-matrix-ext.wgsl.jinja`](build/webgpu/matmul-subgroup-matrix-ext.wgsl.jinja)
- [`matmul-tiled-general-reg.wgsl.jinja`](build/webgpu/matmul-tiled-general-reg.wgsl.jinja)
- [`matmul-tiled-general.wgsl.jinja`](build/webgpu/matmul-tiled-general.wgsl.jinja)
- [`matmul-vector-matrix-tail.wgsl.jinja`](build/webgpu/matmul-vector-matrix-tail.wgsl.jinja)
- [`matmul-vector-matrix-vec4.wgsl.jinja`](build/webgpu/matmul-vector-matrix-vec4.wgsl.jinja)
- [`matmul-vector-product.wgsl.jinja`](build/webgpu/matmul-vector-product.wgsl.jinja)
- [`matmul.wgsl.jinja`](build/webgpu/matmul.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.MatMul", { version: 1 });
const { y } = await kernel({ a: { data: aData, shape: [3, 5] }, b: { data: bData, shape: [5, 4] } });
```
