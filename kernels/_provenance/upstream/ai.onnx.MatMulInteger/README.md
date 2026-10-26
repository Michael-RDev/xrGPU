---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.MatMulInteger

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 10

## Description

Computes an integer matrix product with 8-bit inputs, `int32` accumulation, and independently optional zero points that default to 0. The package implements rank-1 dot products, rank-2 products, rank-2/rank-3 broadcasting, rank-3 products, and rank-4-by-rank-4 products. Scalar zero points are supported throughout; `b_zero_point` additionally supports `[N]` for rank-2 B and `[batch, 1, N]` for non-broadcast rank-3 B. Other standard ONNX matmul rank combinations and N-D per-row/per-column zero-point layouts are unsupported.

See the [ONNX `MatMulInteger` spec](https://onnx.ai/onnx/operators/onnx__MatMulInteger.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `TA` | — | — | N-dimensional integer matrix A (int8 or uint8). | required |
| `b` | `B` | `TB` | — | — | N-dimensional integer matrix B (int8 or uint8). | required |
| `a_zero_point` | — | `TA` | — | — | Optional scalar zero point for A; defaults to 0. Standard N-D per-row layouts are unsupported. | optional |
| `b_zero_point` | — | `TB` | — | — | Optional zero point for B; defaults to 0. Supports a scalar, `[N]` for rank-2 B, or `[batch, 1, N]` for non-broadcast rank-3 B; other standard N-D per-column layouts are unsupported. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `TY` | derived | ONNX MatMul result of `a` and `b` | int32 matrix product result of A * B. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `TA` | `uint8`, `int8` |
| `TB` | `uint8`, `int8` |
| `TY` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `sgmat_precast_a0_bnone_f16` — Prepare exact f16 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a0_bnone_f16` — Prepare lossless f16 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_a0_bnone` — Prepare exact f32 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a0_bnone` — Prepare lossless f32 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_a0_bscalar_f16` — Prepare exact f16 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a0_bscalar_f16` — Prepare lossless f16 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_a0_bscalar` — Prepare exact f32 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a0_bscalar` — Prepare lossless f32 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_a0_bcolumn_f16` — Prepare exact f16 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a0_bcolumn_f16` — Prepare lossless f16 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_a0_bcolumn` — Prepare exact f32 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a0_bcolumn` — Prepare lossless f32 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_a1_bnone_f16` — Prepare exact f16 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a1_bnone_f16` — Prepare lossless f16 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_a1_bnone` — Prepare exact f32 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a1_bnone` — Prepare lossless f32 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_a1_bscalar_f16` — Prepare exact f16 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a1_bscalar_f16` — Prepare lossless f16 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_a1_bscalar` — Prepare exact f32 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a1_bscalar` — Prepare lossless f32 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_a1_bcolumn_f16` — Prepare exact f16 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a1_bcolumn_f16` — Prepare lossless f16 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_a1_bcolumn` — Prepare exact f32 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_a1_bcolumn` — Prepare lossless f32 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_exact_a0_bnone` — Exact integer products through bounded f32 subgroup-matrix partials accumulated with modular int32 additions. Tile height follows workgroup memory and invocation limits.
- `sgmat_exact_a0_bscalar` — Exact integer products through bounded f32 subgroup-matrix partials accumulated with modular int32 additions. Tile height follows workgroup memory and invocation limits.
- `sgmat_exact_a0_bcolumn` — Exact integer products through bounded f32 subgroup-matrix partials accumulated with modular int32 additions. Tile height follows workgroup memory and invocation limits.
- `sgmat_exact_a1_bnone` — Exact integer products through bounded f32 subgroup-matrix partials accumulated with modular int32 additions. Tile height follows workgroup memory and invocation limits.
- `sgmat_exact_a1_bscalar` — Exact integer products through bounded f32 subgroup-matrix partials accumulated with modular int32 additions. Tile height follows workgroup memory and invocation limits.
- `sgmat_exact_a1_bcolumn` — Exact integer products through bounded f32 subgroup-matrix partials accumulated with modular int32 additions. Tile height follows workgroup memory and invocation limits.
- `dp4a_rank2_b_zero_point_per_column` — Packed-dot rank-2 matmul with per-column B zero points. Large prefill tiles use twice as many output-column lanes on variable 16–32-lane devices with sufficient workgroup capacity.
- `dp4a_rank2_a_zero_point_b_per_column` — Packed-dot rank-2 matmul with per-column B zero points. Large prefill tiles use twice as many output-column lanes on variable 16–32-lane devices with sufficient workgroup capacity.

## Device requirements

Some implementation variants require `subgroup-matrix`, `shader-f16`, and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`matmul-integer-batched.wgsl.jinja`](build/webgpu/matmul-integer-batched.wgsl.jinja)
- [`quant-dp4a-matmul.wgsl.jinja`](build/webgpu/quant-dp4a-matmul.wgsl.jinja)
- [`quant-exact-matrix.wgsl.jinja`](build/webgpu/quant-exact-matrix.wgsl.jinja)
- [`quant-exact-portable.wgsl.jinja`](build/webgpu/quant-exact-portable.wgsl.jinja)
- [`quant-exact-prepare.wgsl.jinja`](build/webgpu/quant-exact-prepare.wgsl.jinja)
- [`quant-matmul-accumulate-rank2.wgsl.jinja`](build/webgpu/quant-matmul-accumulate-rank2.wgsl.jinja)
- [`quant-matmul-accumulate-rank4.wgsl.jinja`](build/webgpu/quant-matmul-accumulate-rank4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.MatMulInteger", { version: 1 });
const { y } = await kernel({ a: { data: aData, shape: [1, 1] }, b: { data: bData, shape: [1, 1] } });
```
