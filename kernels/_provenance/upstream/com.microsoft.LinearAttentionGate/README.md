---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.LinearAttentionGate

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Fuses the gate projections used by `com.microsoft.LinearAttention`'s gated-delta recurrence: `decay = decay_scale * softplus(a + dt_bias)` and, when requested, `beta = sigmoid(b)`. The last input axis is the head axis; `dt_bias` and `decay_scale` are float32 per-head vectors. Gate arithmetic is performed in float32 and narrowed only on store. Requesting `beta` requires `b`; an unconsumed `b` is permitted when `beta` is omitted.

See the [ONNX Runtime `LinearAttentionGate` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.LinearAttentionGate) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `aT` | `a` | `T` | — | — | Decay gate projection with shape `(B, T, H)`. Any rank of at least 1 is accepted; the last axis is the head count and the leading axes are folded. | required |
| `dtBiasT` | `dt_bias` | `TF` | `1` | — | Per-head float32 bias added to `a`, with shape (H). | required |
| `decayScaleT` | `decay_scale` | `TF` | `1` | — | Per-head float32 multiplier applied to `softplus(a + dt_bias)`, with shape `(H)`. For gated DeltaNet this is `-exp(A_log)`. | required |
| `bT` | `b` | `T` | — | — | Update-rate projection with the same shape as `a` when `beta` is requested. It is accepted but unused when `beta` is omitted. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `decayT` | `decay` | `T` | same as `aT` | same as `aT` | `decay_scale * softplus(a + dt_bias)`, with the same shape as `a`. | required |
| `betaT` | `beta` | `T` | same as `aT` | same as `aT` | sigmoid(b), with the same shape as `a`. Requires the `b` input. | optional |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `TF` | `float32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`linear-attention-gate.wgsl.jinja`](build/webgpu/linear-attention-gate.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.LinearAttentionGate", { version: 1 });
const { decayT } = await kernel({
  aT: { data: aTData, shape: [5] },
  dtBiasT: { data: dtBiasTData, shape: [5] },
  decayScaleT: { data: decayScaleTData, shape: [5] },
});
```
