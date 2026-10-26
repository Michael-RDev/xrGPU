---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.MatMulNBitsQkv

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Fuses RMS normalization with three block-quantized attention projections: `A_norm = RMSNorm(A + skip, norm_scale)` (or without `skip`), followed by Q, K, and V projections. The optional fourth output returns `A + skip`. Only 4-bit weights with `block_size = 32` are supported; projection biases, bfloat16, and non-default `accuracy_level` values are not implemented.

See the [ONNX Runtime `MatMulNBitsQkv` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBitsQkv) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `aT` | `A` | `T1` | — | — | Shared activation of rank 2 `(M, K)` or rank 3 `(batch, sequence, K)`; only the last axis is the reduction axis. | required |
| `skipT` | `skip` | `T1` | — | — | Residual added to A before the normalization, with A's shape. | optional |
| `normScaleT` | `norm_scale` | `T1` | `1` | — | Simplified-layer-normalization (RMS) gain of shape `[K]`. | required |
| `qBT` | `q_B` | `T2` | `3` | — | Bit-packed uint8 Q weights of shape `(Nq, k_blocks, blob_size)`. Bound in the packed storage layout: four blob bytes per u32 word. | required |
| `qScalesT` | `q_scales` | `T1` | `2` | — | Per-block Q scales of shape `(Nq, k_blocks)`. Quantization is symmetric: there is no zero-point input, so codes are offset by the midpoint `2^(bits - 1)`. | required |
| `kBT` | `k_B` | `T2` | `3` | — | Bit-packed K weights of shape `(Nkv, k_blocks, blob_size)`. Bound in the packed storage layout: four blob bytes per u32 word. | required |
| `kScalesT` | `k_scales` | `T1` | `2` | — | Per-block K scales of shape `(Nkv, k_blocks)`. | required |
| `vBT` | `v_B` | `T2` | `3` | — | Bit-packed V weights of shape `(Nkv, k_blocks, blob_size)`. Bound in the packed storage layout: four blob bytes per u32 word. | required |
| `vScalesT` | `v_scales` | `T1` | `2` | — | Per-block V scales of shape `(Nkv, k_blocks)`. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `qT` | `Q` | `T1` | same as `aT` | derived | Query projection: A's leading axes with a trailing Nq. | required |
| `kT` | `K` | `T1` | same as `aT` | derived | Key projection: A's leading axes with a trailing Nkv. | required |
| `vT` | `V` | `T1` | same as `aT` | derived | Value projection: A's leading axes with a trailing Nkv. | required |
| `residualT` | `input_skip_bias_sum` | `T1` | same as `aT` | same as `aT` | The residual sum A + skip, with A's shape. Requires the skip input. | optional |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `K` | — | Input feature dimension shared by the normalized input and all projection weights. |
| `Nkv` | — | Output feature dimension shared by the K and V projections. |
| `Nq` | — | Output feature dimension of the Q projection. |
| `accuracy_level` | `0` | Minimum internal accuracy level, following MatMulNBits semantics; this implementation supports the standard default 0. |
| `bits` | `4` | Bit width used to quantize all three weight matrices; only 4 is supported. |
| `block_size` | — | Size of each quantization block along K; only 32 is supported. |
| `epsilon` | `9.999999974752427e-7` | Epsilon used by the simplified layer-normalization reduction. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T1` | `float32`, `float16` |
| `T2` | `uint8` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`matmul-nbits-fused-rms-norm.wgsl.jinja`](build/webgpu/matmul-nbits-fused-rms-norm.wgsl.jinja)
- [`qkv-projection.wgsl.jinja`](build/webgpu/qkv-projection.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.MatMulNBitsQkv", { version: 1 });
const { qT, kT, vT } = await kernel({
  aT: { data: aTData, shape: [1, 21] },
  normScaleT: { data: normScaleTData, shape: [21] },
  qBT: { data: qBTData, shape: [5, 1, 16] },
  qScalesT: { data: qScalesTData, shape: [5, 1] },
  kBT: { data: kBTData, shape: [3, 1, 16] },
  kScalesT: { data: kScalesTData, shape: [3, 1] },
  vBT: { data: vBTData, shape: [3, 1, 16] },
  vScalesT: { data: vScalesTData, shape: [3, 1] },
}, {
  attrs: {
    K: 21,
    Nq: 5,
    Nkv: 3,
    block_size: 32,
  },
});
```
