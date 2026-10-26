---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Einsum

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 12

## Description

Evaluates algebraic tensor operations using the Einstein summation convention, specified by an `equation` string of the form `term1,term2,...->output-term`. Output elements are computed as the product of the input operands summed over any indices that appear in the input terms but not in the output term; if no `->` is given, output indices are the alphabetically sorted set of indices that appear exactly once across all terms.

See the [ONNX `Einsum` spec](https://onnx.ai/onnx/operators/onnx__Einsum.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `T` | — | — | First operand tensor. | required |
| `b` | `B` | `T` | — | — | Second operand tensor (optional). | optional |
| `c` | `C` | `T` | — | — | Third operand tensor (optional). | optional |
| `d` | `D` | `T` | — | — | Fourth operand tensor (optional); ONNX Einsum takes a variadic operand list. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Output` | `T` | — | — | Output tensor produced by the einsum contraction. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `equation` | — | Required einsum expression string defining the subscript labels for each input operand and the output, in the form `inputs->output` (e.g. `ij,jk->ik` for matrix multiplication). |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `abc_bcd_ad_subgroup_matrix_splitk` — Partitions the contraction reduction across workgroups while retaining subgroup-matrix tiles, then combines float32 partials. It serves output grids with too little unsplit parallelism, and the combine reassociates the reduction.
- `abc_bcd_ad_tiled_reg_splitk` — Partitions the contraction reduction across workgroups in the portable register-tiled path, then combines float32 partials. It serves output grids with too little unsplit parallelism, and the combine reassociates the reduction.
- `bij_bjk_bik_band_vec4` — Batched bij,bjk->bik with 2..16 rows per matrix on MatMul's vec4 band kernel: one workgroup row per matrix and K slices sized from the batch times column-group count. The generic 32x32 tile leaves most of its threads idle at these row counts.
- `i_i_scalar_splitk` — Splits large scalar or narrow-output reductions across workgroups and combines their float32 partials, avoiding one underfilled reducer. The combine changes float32 association relative to a single-workgroup reduction, so results need not be bit-identical.
- `ij_scalar_splitk` — Splits large scalar or narrow-output reductions across workgroups and combines their float32 partials, avoiding one underfilled reducer. The combine changes float32 association relative to a single-workgroup reduction, so results need not be bit-identical.
- `ij_j_splitk_narrow_cooperative` — Splits large scalar or narrow-output reductions across workgroups and combines their float32 partials, avoiding one underfilled reducer. The combine changes float32 association relative to a single-workgroup reduction, so results need not be bit-identical.
- `ij_j_splitk` — Splits large scalar or narrow-output reductions across workgroups and combines their float32 partials, avoiding one underfilled reducer. The combine changes float32 association relative to a single-workgroup reduction, so results need not be bit-identical.

## Device requirements

Some implementation variants require `subgroup-matrix` and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`datamove-elementwise-copy.wgsl.jinja`](build/webgpu/datamove-elementwise-copy.wgsl.jinja)
- [`dot-splitk-partial.wgsl.jinja`](build/webgpu/dot-splitk-partial.wgsl.jinja)
- [`einsum-4d-contract.wgsl.jinja`](build/webgpu/einsum-4d-contract.wgsl.jinja)
- [`einsum-ab-ba-ab.wgsl.jinja`](build/webgpu/einsum-ab-ba-ab.wgsl.jinja)
- [`einsum-abc-contract.wgsl.jinja`](build/webgpu/einsum-abc-contract.wgsl.jinja)
- [`einsum-batch-diagonal-iij-j.wgsl.jinja`](build/webgpu/einsum-batch-diagonal-iij-j.wgsl.jinja)
- [`einsum-batch-diagonal.wgsl.jinja`](build/webgpu/einsum-batch-diagonal.wgsl.jinja)
- [`einsum-batch-matmul.wgsl.jinja`](build/webgpu/einsum-batch-matmul.wgsl.jinja)
- [`einsum-batch-reduce-all-split.wgsl.jinja`](build/webgpu/einsum-batch-reduce-all-split.wgsl.jinja)
- [`einsum-batch-reduce-all.wgsl.jinja`](build/webgpu/einsum-batch-reduce-all.wgsl.jinja)
- [`einsum-batch-reduce-last.wgsl.jinja`](build/webgpu/einsum-batch-reduce-last.wgsl.jinja)
- [`einsum-bilinear-split.wgsl.jinja`](build/webgpu/einsum-bilinear-split.wgsl.jinja)
- [`einsum-diag-matmul-iij-jk.wgsl.jinja`](build/webgpu/einsum-diag-matmul-iij-jk.wgsl.jinja)
- [`einsum-diagonal-extract.wgsl.jinja`](build/webgpu/einsum-diagonal-extract.wgsl.jinja)
- [`einsum-diagonal-iji-reduce.wgsl.jinja`](build/webgpu/einsum-diagonal-iji-reduce.wgsl.jinja)
- [`einsum-indexed-2d-output.wgsl.jinja`](build/webgpu/einsum-indexed-2d-output.wgsl.jinja)
- [`einsum-matmul.wgsl.jinja`](build/webgpu/einsum-matmul.wgsl.jinja)
- [`einsum-mul.wgsl.jinja`](build/webgpu/einsum-mul.wgsl.jinja)
- [`einsum-nd-generic.wgsl.jinja`](build/webgpu/einsum-nd-generic.wgsl.jinja)
- [`einsum-reduce-middle-transpose.wgsl.jinja`](build/webgpu/einsum-reduce-middle-transpose.wgsl.jinja)
- [`einsum-reduce-rows.wgsl.jinja`](build/webgpu/einsum-reduce-rows.wgsl.jinja)
- [`einsum-serial-scalar.wgsl.jinja`](build/webgpu/einsum-serial-scalar.wgsl.jinja)
- [`einsum-transpose.wgsl.jinja`](build/webgpu/einsum-transpose.wgsl.jinja)
- [`matmul-band-vec4.wgsl.jinja`](build/webgpu/matmul-band-vec4.wgsl.jinja)
- [`matmul-subgroup-matrix-ext.wgsl.jinja`](build/webgpu/matmul-subgroup-matrix-ext.wgsl.jinja)
- [`matmul-tiled-general-reg.wgsl.jinja`](build/webgpu/matmul-tiled-general-reg.wgsl.jinja)
- [`matmul-tiled-general.wgsl.jinja`](build/webgpu/matmul-tiled-general.wgsl.jinja)
- [`matmul-vector-matrix-vec4.wgsl.jinja`](build/webgpu/matmul-vector-matrix-vec4.wgsl.jinja)
- [`reduce-axis0-splitk-combine.wgsl.jinja`](build/webgpu/reduce-axis0-splitk-combine.wgsl.jinja)
- [`reduce-axis0-splitk-reduce.wgsl.jinja`](build/webgpu/reduce-axis0-splitk-reduce.wgsl.jinja)
- [`reduce-axis0-tilecols.wgsl.jinja`](build/webgpu/reduce-axis0-tilecols.wgsl.jinja)
- [`reduce-flat-partial.wgsl.jinja`](build/webgpu/reduce-flat-partial.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `y`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Einsum", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({ a: { data: aData, shape: [2, 3] }, b: { data: bData, shape: [3, 2] } }, {
  attrs: { equation: "ij,jk->ik" },
  outputs: { y: { shape: [2, 2], dtype: "float32" } },
});
```
