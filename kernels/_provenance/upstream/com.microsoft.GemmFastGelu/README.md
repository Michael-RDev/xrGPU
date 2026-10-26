---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.GemmFastGelu

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Fuses MatMul, an optional bias, and FastGelu: `Y = FastGelu(X @ W + bias)`. `X` has rank at least 2 with shape `(..., K)`, `W` has shape `(K, N)`, and `bias` has shape `(N)`. The activation runs in the float32 accumulator before the output is narrowed, avoiding an intermediate `(..., N)` tensor. Bfloat16 is not implemented.

See the [ONNX Runtime `GemmFastGelu` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GemmFastGelu) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `X` | `T` | — | — | Left operand of rank 2 or greater with shape `(..., K)`; every leading-axis coordinate identifies a row of the product. | required |
| `W` | `T` | `2` | — | Right operand with shape `(K, N)`. | required |
| `bias` | `T` | `1` | — | Optional bias with shape `(N)`, added before the activation. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `Y` | `T` | same as `X` | ONNX MatMul result of `X` and `W` | `FastGelu(X @ W + bias)`, with the same rank and leading dimensions as `X` and a trailing `N`. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `vector_matrix_bias` — Partitions a single-row vector-matrix product across device-bounded workgroup slices. Optional bias and FastGelu are fused into the f32 reduction before the single output cast; no subgroup feature is required.
- `vector_matrix` — Partitions a single-row vector-matrix product across device-bounded workgroup slices. Optional bias and FastGelu are fused into the f32 reduction before the single output cast; no subgroup feature is required.
- `sgmat_medium_bias` — Uses supported subgroup matrices with 32x32 direct activation tiles and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_medium` — Uses supported subgroup matrices with 32x32 direct activation tiles and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_medium_bias_f16` — Uses supported subgroup matrices with 32x32 direct activation tiles and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_medium_f16` — Uses supported subgroup matrices with 32x32 direct activation tiles and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_small_bias` — Uses supported subgroup matrices with 32x16 direct activation tiles and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_small` — Uses supported subgroup matrices with 32x16 direct activation tiles and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_small_bias_f16` — Uses supported subgroup matrices with 32x16 direct activation tiles and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_small_f16` — Uses supported subgroup matrices with 32x16 direct activation tiles and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_direct_bias` — Uses supported subgroup matrices with direct activation loads and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_direct` — Uses supported subgroup matrices with direct activation loads and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_direct_bias_f16` — Uses supported subgroup matrices with direct activation loads and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_direct_f16` — Uses supported subgroup matrices with direct activation loads and double-buffered weights. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_bias` — Uses supported subgroup matrices with staged operands. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat` — Uses supported subgroup matrices with staged operands. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_bias_f16` — Uses supported subgroup matrices with staged operands. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `sgmat_f16` — Uses supported subgroup matrices with staged operands. Bias and FastGelu run on the f32 accumulator before the single output cast. Resource, alignment and matrix-type guards retain the portable path on other devices.
- `tiled_bias` — Portable register tiles use the output-grid budget and a device-specific f32 tile choice to improve occupancy, with device-bounded shared padding and compensated f32 tile sums before the fused activation.
- `tiled` — Portable register tiles use the output-grid budget and a device-specific f32 tile choice to improve occupancy, with device-bounded shared padding and compensated f32 tile sums before the fused activation.

## Device requirements

Some implementation variants require `subgroup-matrix` and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`gemm-epilogue-tiled-reg.wgsl.jinja`](build/webgpu/gemm-epilogue-tiled-reg.wgsl.jinja)
- [`gemm-subgroup-matrix.wgsl.jinja`](build/webgpu/gemm-subgroup-matrix.wgsl.jinja)
- [`matmul-vector-matrix-vec4.wgsl.jinja`](build/webgpu/matmul-vector-matrix-vec4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.GemmFastGelu", { version: 1 });
const { Y } = await kernel({ X: { data: XData, shape: [1, 1] }, W: { data: WData, shape: [1, 4] } });
```
