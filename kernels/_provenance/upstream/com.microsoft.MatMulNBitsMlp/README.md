---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.MatMulNBitsMlp

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Fuses a gated MLP over two block-quantized projections that share one activation: `Y = silu(A_norm @ gate + gate_bias) * (A_norm @ up + up_bias)`, using the `MatMulNBits` weight packing with no zero-point input. `A_norm` is `A`, `SimplifiedLayerNormalization(A, norm_scale)`, or `SkipSimplifiedLayerNormalization(A, skip, norm_scale)`, whose residual sum may be returned as a second output. Only `silu` and the default `accuracy_level = 0` are implemented; bfloat16 is not implemented.

See the [ONNX Runtime `MatMulNBitsMlp` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBitsMlp) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `aT` | `A` | `T1` | — | — | Shared activation of rank 2 `(M, K)` or rank 3 `(batch, sequence, K)`; only the last axis is the reduction axis. | required |
| `skipT` | `skip` | `T1` | — | — | Residual added to `A` before normalization, with `A`'s shape. Requires `norm_scale`. | optional |
| `normScaleT` | `norm_scale` | `T1` | `1` | — | Simplified-layer-normalization (RMS) gain of shape `[K]`. Absent means the projections read `A` unnormalized. | optional |
| `gateBT` | `gate_B` | `uint8` | `3` | — | Bit-packed uint8 gate weights of shape `(N, k_blocks, blob_size)`. Bound in the packed storage layout: four blob bytes per u32 word. | required |
| `gateScalesT` | `gate_scales` | `T1` | `2` | — | Per-block gate scales of shape `(N, k_blocks)`, with the same dtype as `A`. Quantization is symmetric: this operator has no zero-point input, so codes are offset by the midpoint `2^(bits - 1)`. | required |
| `gateBiasT` | `gate_bias` | `T1` | `1` | — | Optional gate bias of shape `[N]`, added before the activation. | optional |
| `upBT` | `up_B` | `uint8` | `3` | — | Bit-packed up weights, same shape and packing as gate_B. Bound in the packed storage layout: four blob bytes per u32 word. | required |
| `upScalesT` | `up_scales` | `T1` | `2` | — | Per-block up scales of shape `(N, k_blocks)`. | required |
| `upBiasT` | `up_bias` | `T1` | `1` | — | Optional up bias of shape `[N]`, added before the product. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `yT` | `Y` | `T1` | same as `aT` | derived | Gated MLP output: A's leading axes with a trailing N. | required |
| `residualT` | `input_skip_bias_sum` | `T1` | same as `aT` | same as `aT` | The residual sum A + skip, with A's shape. Requires the skip input. | optional |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `K` | — | Input feature dimension shared by both quantized weight matrices. |
| `N` | — | Output feature dimension shared by both quantized weight matrices. |
| `accuracy_level` | `0` | Minimum internal accuracy level: 0 (unset), 1 (float32), 2 (float16), 3 (bfloat16), or 4 (int8). |
| `activation` | — | Activation applied to the gate projection; this implementation supports `silu`. |
| `bits` | `4` | Bit width used to quantize both weight matrices; this implementation supports 2, 4, and 8. |
| `block_size` | — | Size of each quantization block along K. |
| `epsilon` | `0.00001` | Epsilon used by the optional fused RMS normalization. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T1` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`matmul-nbits-fused-rms-norm.wgsl.jinja`](build/webgpu/matmul-nbits-fused-rms-norm.wgsl.jinja)
- [`mlp-gate-up.wgsl.jinja`](build/webgpu/mlp-gate-up.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.MatMulNBitsMlp", { version: 1 });
const { yT } = await kernel({
  aT: { data: aTData, shape: [2, 16] },
  gateBT: { data: gateBTData, shape: [4, 2, 4] },
  gateScalesT: { data: gateScalesTData, shape: [4, 2] },
  upBT: { data: upBTData, shape: [4, 2, 4] },
  upScalesT: { data: upScalesTData, shape: [4, 2] },
}, {
  attrs: {
    K: 16,
    N: 4,
    block_size: 8,
    activation: "silu",
  },
});
```
