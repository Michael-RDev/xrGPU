---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.QLinearMatMul

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 21

## Description

Quantized matrix multiplication of tensors `a` and `b` with supported ONNX `MatMul` vector and batch-broadcasting shapes. Each operand and the output carry their own scale and zero point; the result is rounded to nearest even and saturated to the output type. This package requires single-element rank-1 quantization tensors for `a` and `y`. For `b`, it supports single-element rank-1 tensors and selected per-column layouts; per-row quantization of `a` is unsupported.

See the [ONNX `QLinearMatMul` spec](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `a` | `TA` | — | — | N-dimensional quantized matrix a. | required |
| `a_scale` | `TF` | `1` | — | Per-tensor scale of quantized input `a`, supplied as a single-element rank-1 tensor. | required |
| `a_zero_point` | `TA` | `1` | — | Zero point of quantized input `a`; must have the same shape as `a_scale`. | required |
| `b` | `TB` | — | — | N-dimensional quantized matrix b. | required |
| `b_scale` | `TF` | — | — | Scale of quantized input `b`: a single-element rank-1 tensor for per-tensor quantization, a length-N vector for a rank-2 matrix, or `[batch, 1, N]` for a rank-3 matrix with per-column quantization. | required |
| `b_zero_point` | `TB` | — | — | Zero point of quantized input `b`: a single-element rank-1 tensor, a length-N vector or `[1, N]` for a rank-2 matrix, or `[batch, 1, N]` for a rank-3 matrix. Rank-2 per-column scales and zero points may independently use their per-tensor forms. | required |
| `y_scale` | `TF` | `1` | — | Per-tensor scale of quantized output `y`, supplied as a single-element rank-1 tensor. | required |
| `y_zero_point` | `TY` | `1` | — | Per-tensor zero point of quantized output `y`, supplied as a single-element rank-1 tensor. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `y` | `TY` | derived | ONNX MatMul result of `a` and `b` | Quantized matrix multiply result of a * b. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `TA` | `uint8`, `int8` |
| `TB` | `uint8`, `int8` |
| `TY` | `uint8`, `int8` |
| `TF` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `sgmat_precast_scalar_f16` — Prepare exact f16 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_scalar_f16` — Prepare lossless f16 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_scalar` — Prepare exact f32 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_scalar` — Prepare lossless f32 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_column_f16` — Prepare exact f16 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_column_f16` — Prepare lossless f16 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_column` — Prepare exact f32 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_column` — Prepare lossless f32 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_column_scale_f16` — Prepare exact f16 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_column_scale_f16` — Prepare lossless f16 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_column_scale` — Prepare exact f32 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_column_scale` — Prepare lossless f32 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_column_zero_f16` — Prepare exact f16 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_column_zero_f16` — Prepare lossless f16 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_precast_column_zero` — Prepare exact f32 representations of adjusted byte integers once, then load them directly into f32-accumulating subgroup matrices. Row groups follow matrix geometry, output rows, invocation limits and workgroup storage; bounded partial sums retain exact modular int32 arithmetic.
- `portable_precast_column_zero` — Prepare lossless f32 representations of adjusted byte integers, then use vector workgroup GEMM with exact bounded f32 partials and modular int32 totals. Geometry follows output rows, invocation and storage limits; f32 storage requires enough row reuse to amortize preparation.
- `sgmat_exact_scalar` — Exact integer products through bounded f32 matrix partials, fused with scalar or per-column requantization. Tile height follows workgroup memory and invocation limits.
- `sgmat_exact_column` — Exact integer products through bounded f32 matrix partials, fused with scalar or per-column requantization. Tile height follows workgroup memory and invocation limits.
- `sgmat_exact_column_scale` — Exact integer products through bounded f32 matrix partials, fused with scalar or per-column requantization. Tile height follows workgroup memory and invocation limits.
- `sgmat_exact_column_zero` — Exact integer products through bounded f32 matrix partials, fused with scalar or per-column requantization. Tile height follows workgroup memory and invocation limits.
- `portable_accurate_dp4a_rank2_scalar_quantization` — Uses tiled integer accumulation and accurate requantization when a fused path is unavailable, including devices whose storage-binding limit cannot accommodate it. Native packed dot products are used when available.
- `portable_accurate_dp4a_rank2_per_column_quantization` — Uses tiled integer accumulation and accurate requantization when a fused path is unavailable, including devices whose storage-binding limit cannot accommodate it. Native packed dot products are used when available.
- `portable_accurate_dp4a_rank2_per_column_scale_scalar_zp` — Uses tiled integer accumulation and accurate requantization when a fused path is unavailable, including devices whose storage-binding limit cannot accommodate it. Native packed dot products are used when available.
- `portable_accurate_dp4a_rank2_scalar_quantization_tail` — Uses tiled integer accumulation and accurate requantization when a fused path is unavailable, including devices whose storage-binding limit cannot accommodate it. Native packed dot products are used when available.
- `portable_accurate_dp4a_rank2_per_column_scale_scalar_zp_tail` — Uses tiled integer accumulation and accurate requantization when a fused path is unavailable, including devices whose storage-binding limit cannot accommodate it. Native packed dot products are used when available.
- `portable_accurate_dp4a_rank3_scalar_quantization` — Uses tiled integer accumulation and accurate requantization when a fused path is unavailable, including devices whose storage-binding limit cannot accommodate it. Native packed dot products are used when available.
- `portable_accurate_dp4a_rank4_scalar_quantization` — Uses tiled integer accumulation and accurate requantization when a fused path is unavailable, including devices whose storage-binding limit cannot accommodate it. Native packed dot products are used when available.
- `portable_accurate_dp4a_rank2_scalar_scale_per_column_zp` — Uses tiled integer accumulation and accurate requantization when a fused path is unavailable, including devices whose storage-binding limit cannot accommodate it. Native packed dot products are used when available.
- `portable_accurate_dp4a_rank2_scalar_scale_per_column_zp_tail` — Uses tiled integer accumulation and accurate requantization when a fused path is unavailable, including devices whose storage-binding limit cannot accommodate it. Native packed dot products are used when available.

## Device requirements

Some implementation variants require `subgroup-matrix`, `shader-f16`, and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`qlinear-matmul-accumulate.wgsl.jinja`](build/webgpu/qlinear-matmul-accumulate.wgsl.jinja)
- [`qlinear-matmul-requantize.wgsl.jinja`](build/webgpu/qlinear-matmul-requantize.wgsl.jinja)
- [`quant-dp4a-matmul.wgsl.jinja`](build/webgpu/quant-dp4a-matmul.wgsl.jinja)
- [`quant-exact-matrix.wgsl.jinja`](build/webgpu/quant-exact-matrix.wgsl.jinja)
- [`quant-exact-portable.wgsl.jinja`](build/webgpu/quant-exact-portable.wgsl.jinja)
- [`quant-exact-prepare.wgsl.jinja`](build/webgpu/quant-exact-prepare.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.QLinearMatMul", { version: 1 });
const { y } = await kernel({
  a: { data: aData, shape: [1, 1] },
  a_scale: { data: a_scaleData, shape: [1] },
  a_zero_point: { data: a_zero_pointData, shape: [1] },
  b: { data: bData, shape: [1, 2] },
  b_scale: { data: b_scaleData, shape: [1] },
  b_zero_point: { data: b_zero_pointData, shape: [1] },
  y_scale: { data: y_scaleData, shape: [1] },
  y_zero_point: { data: y_zero_pointData, shape: [1] },
});
```
