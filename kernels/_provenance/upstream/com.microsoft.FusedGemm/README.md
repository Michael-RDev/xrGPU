---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.FusedGemm

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Gemm with a fused activation: `Y = act(alpha * A' * B' + beta * C)`, where `A'` and `B'` are optionally transposed and `C` is broadcastable to `(M, N)`. The activation runs in the f32 accumulator before the single output narrowing. This package supports `Relu`, `LeakyRelu`, `Sigmoid`, `Tanh` and `HardSigmoid`; the other activation strings and numeric types admitted by the open schema are not implemented. Omitting `activation` gives plain Gemm.

See the [ONNX Runtime `FusedGemm` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedGemm) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `A` | `T` | `2` | — | Left operand, `(M, K)` when `transA` is 0 and `(K, M)` otherwise. | required |
| `B` | `T` | `2` | — | Right operand, `(K, N)` when `transB` is 0 and `(N, K)` otherwise. | required |
| `C` | `T` | — | — | Optional additive term, unidirectionally broadcastable to `(M, N)`: a scalar, a row `(N)`, a column `(M, 1)`, or the full matrix. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `Y` | `T` | `2` | derived | `act(alpha * A' * B' + beta * C)` with shape `(M, N)`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `activation` | — | Optional fused activation name. Supported modes are `Relu`, `LeakyRelu`, `Sigmoid`, `Tanh` and `HardSigmoid`; omission applies none. |
| `activation_alpha` | — | First activation parameter: the slope for `LeakyRelu` or `alpha` for `HardSigmoid`. |
| `activation_beta` | — | Second activation parameter: `beta` for `HardSigmoid`. |
| `alpha` | `1` | Scalar multiplier for the product `A' * B'`; the standard default is 1. |
| `beta` | `1` | Scalar multiplier for `C`; the standard default is 1. |
| `transA` | `0` | Whether `A` is stored transposed. The standard default is 0. |
| `transB` | `0` | Whether `B` is stored transposed. The standard default is 0. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `gemv_plain` — Reduces contiguous weight rows for a single output row in f32, applies alpha and activation, and casts once. Four aligned columns share the input vector; other widths use one column. The reduction width is bounded by K and device workgroup limits. Either storage orientation of A is supported.
- `gemv_bias` — Reduces contiguous weight rows for a single output row in f32, applies alpha, broadcast beta-scaled bias and activation, and casts once. Four aligned columns share the input vector; other widths use one column. The reduction width is bounded by K and device workgroup limits. Either storage orientation of A is supported.
- `sgmat_bias` — Uses supported subgroup matrices with f32 accumulation and a fused epilogue. Complete non-transposed f32 A tiles load directly from storage; long reductions on that path merge two-tile sums with f32 compensation. Staged A clamps row tails. Workgroup memory is checked against device limits.
- `sgmat` — Uses supported subgroup matrices with f32 accumulation and a fused epilogue. Complete non-transposed f32 A tiles load directly from storage; long reductions on that path merge two-tile sums with f32 compensation. Staged A clamps row tails. Workgroup memory is checked against device limits.

## Device requirements

Some implementation variants require `subgroup-matrix` and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`gemm-epilogue-tiled-reg.wgsl.jinja`](build/webgpu/gemm-epilogue-tiled-reg.wgsl.jinja)
- [`gemm-subgroup-matrix.wgsl.jinja`](build/webgpu/gemm-subgroup-matrix.wgsl.jinja)
- [`gemv-contiguous.wgsl.jinja`](build/webgpu/gemv-contiguous.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.FusedGemm", { version: 1 });
const { Y } = await kernel({
  A: { data: AData, shape: [32, 224] },
  B: { data: BData, shape: [224, 64] },
});
```
