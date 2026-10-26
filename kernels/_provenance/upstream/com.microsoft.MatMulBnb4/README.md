---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.MatMulBnb4

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Computes `A @ dequant(B)^T` where `B` uses bitsandbytes 4-bit quantization: `quant_type = 0` selects FP4 and `quant_type = 1` selects NF4. Supports rank-2 float16/float32 `A`, `transB = 1`, and `training_mode = 0`; rank-1 and rank-3-or-higher `A`, bfloat16, `transB = 0`, and training are not implemented. `B` is the flattened `[N, K]` weight, two codes per byte with the even flat index in the high nibble. Each code indexes a fixed 16-entry codebook, and the value is `codebook[code] * absmax[flat_index / block_size]`.

See the [ONNX Runtime `MatMulBnb4` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulBnb4) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `aT` | `A` | `T1` | `2` | — | Float input matrix of shape `(M, K)`, not quantized. | required |
| `bT` | `B` | `T2` | `1` | — | The `[N, K]` weight, flattened and quantized to 4 bits, stored as `(N * K + 1) / 2` bytes; the ONNX type is uint8 (this WebGPU implementation reads one widened u32 per stored byte). | required |
| `absmaxT` | `absmax` | `T1` | `1` | — | Per-block absolute-maximum dequantization scales of shape `((N * K + block_size - 1) / block_size)`, same dtype as A. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `yT` | `Y` | `T1` | `2` | `[aT[0], N]` | Result of `A` multiplied by the dequantized, transposed weight matrix, with shape `(M, N)` and the same dtype as `A`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `K` | — | Input feature count (the shared dimension). |
| `N` | — | Output feature count. |
| `block_size` | — | Number of weights sharing one absmax scale; a power of two, at least 16. |
| `quant_type` | — | Codebook selector: 0 = FP4, 1 = NF4. |
| `training_mode` | `0` | Whether training outputs are requested. This inference-only implementation supports the standard default value 0. |
| `transB` | `1` | Whether the quantized weight is stored transposed. This implementation supports the standard default value 1. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T1` | `float32`, `float16` |
| `T2` | `uint8` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `sgmat_hybrid_rows` — Combines full and partial row tiles in one matrix dispatch. Complete tiles read activations directly; partial tiles stage bounded rows in f32 and reuse weight staging for output. Float16 requests widen before the multiply and narrow once afterward.
- `sgmat_hybrid_rows_widened` — Combines full and partial row tiles in one matrix dispatch. Complete tiles read activations directly; partial tiles stage bounded rows in f32 and reuse weight staging for output. Float16 requests widen before the multiply and narrow once afterward.
- `sgmat_staged_rows` — Stages partial activation rows in f32, dequantizes the shared weight tile, and reuses dead weight staging for masked output stores.
- `sgmat_widened` — Runs the subgroup-matrix tier for a float16 request by widening the activations to float32 either side of the multiply. The matrix units this operator uses accumulate in float32 from float32 operands; the device's float16 configuration returns a float16 result, so feeding them float16 directly would drop the accumulator's precision.

## Device requirements

Some implementation variants require `subgroup-matrix` and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`cast-scalar-x4.wgsl.jinja`](build/webgpu/cast-scalar-x4.wgsl.jinja)
- [`matmul-bnb4-gemv.wgsl.jinja`](build/webgpu/matmul-bnb4-gemv.wgsl.jinja)
- [`matmul-bnb4-sgmat.wgsl.jinja`](build/webgpu/matmul-bnb4-sgmat.wgsl.jinja)
- [`matmul-bnb4-tiled.wgsl.jinja`](build/webgpu/matmul-bnb4-tiled.wgsl.jinja)
- [`matmul-bnb4.wgsl.jinja`](build/webgpu/matmul-bnb4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.MatMulBnb4", { version: 1 });
const { yT } = await kernel({
  aT: { data: aTData, shape: [2, 24] },
  bT: { data: bTData, shape: [36] },
  absmaxT: { data: absmaxTData, shape: [5] },
}, {
  attrs: {
    K: 24,
    N: 3,
    block_size: 16,
    quant_type: 1,
  },
});
```
