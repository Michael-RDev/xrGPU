---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.LinearAttention

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Recurrent linear attention for packed `[B, T, H*D]` decode and prefill. It supports all four update rules, standard and inverse GQA, shared-key heads, and rollback states through `state_window`. Activations and state may independently use float16 or float32; bfloat16 is not implemented. `past_state` is optional for every update rule and defaults to zeros.

See the [ONNX Runtime `LinearAttention` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.LinearAttention) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `queryT` | `query` | `T` | `3` | — | Query vectors with 3D packed shape `(B, T, H_q * d_k)`; heads are packed into the last dimension. | required |
| `keyT` | `key` | `T` | `3` | — | Key vectors with 3D packed shape `(B, T, H_k * d_k)`, where positive `H_k` divides `H_kv`; `H_k < H_kv` shares each key head across multiple KV-state heads. Keys should be L2-normalized for `delta`/`gated_delta` modes. | required |
| `valueT` | `value` | `T` | `3` | — | Value vectors with 3D packed shape `(B, T, H_kv * d_v)`. | required |
| `pastStateT` | `past_state` | `S` | derived | derived | Recurrent state from the previous step with shape `(B, H_kv, d_k, d_v)`, or `(W, B, H_kv, d_k, d_v)` when `state_window = W > 0`; defaults to zeros if absent. | optional |
| `decayT` | `decay` | `T` | `3` | — | Exponential decay gate in log-space with shape `(B, T, H_kv * d_k)` or `(B, T, H_kv)`; required for `gated` and `gated_delta` modes. | optional |
| `betaT` | `beta` | `T` | `3` | — | Update rate (sigmoid output) with shape `(B, T, H_kv)` or `(B, T, 1)`; required for `delta` and `gated_delta` modes. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `3` | derived | Attention output with 3D packed shape `(B, T, max(H_q, H_kv) * d_v)`. | required |
| `presentStateT` | `present_state` | `S` | derived | derived | Updated recurrent state with shape `(B, H_kv, d_k, d_v)`, or `(W, B, H_kv, d_k, d_v)` when `state_window = W > 0`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `chunk_size` | `64` | Accepted for schema compatibility; does not affect the result. |
| `kv_num_heads` | — | Number of key/value heads. |
| `q_num_heads` | — | Number of query heads. |
| `scale` | `0` | Scale applied to query-key products. Zero selects `1 / sqrt(d_k)`. |
| `state_window` | `0` | Number of recent recurrent states retained in `present_state`, in the supported range 0 to 8; zero returns only the current state. |
| `update_rule` | `"gated_delta"` | Recurrent update rule: `linear`, `gated`, `delta`, or `gated_delta`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `S` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `linear_zero_serial_small_dk` — Small key dimensions keep each state column private across the sequence. Linear updates specialize shape-known indexing and loop bounds, while gated-delta updates retain their uniform-driven recurrence.
- `linear_state_serial_small_dk` — Small key dimensions keep each state column private across the sequence. Linear updates specialize shape-known indexing and loop bounds, while gated-delta updates retain their uniform-driven recurrence.
- `gated_delta_zero_serial_small_dk` — Small key dimensions keep each state column private across the sequence. Linear updates specialize shape-known indexing and loop bounds, while gated-delta updates retain their uniform-driven recurrence.
- `gated_delta_state_serial_small_dk` — Small key dimensions keep each state column private across the sequence. Linear updates specialize shape-known indexing and loop bounds, while gated-delta updates retain their uniform-driven recurrence.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`chunk-out.wgsl.jinja`](build/webgpu/chunk-out.wgsl.jinja)
- [`chunk-prep.wgsl.jinja`](build/webgpu/chunk-prep.wgsl.jinja)
- [`chunk-scan.wgsl.jinja`](build/webgpu/chunk-scan.wgsl.jinja)
- [`chunk-ut.wgsl.jinja`](build/webgpu/chunk-ut.wgsl.jinja)
- [`linear-attention.scalar.wgsl.jinja`](build/webgpu/linear-attention.scalar.wgsl.jinja)
- [`linear-attention.serial.wgsl.jinja`](build/webgpu/linear-attention.serial.wgsl.jinja)
- [`linear-attention.vec4.wgsl.jinja`](build/webgpu/linear-attention.vec4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.LinearAttention", { version: 1 });
const { outputT, presentStateT } = await kernel({
  queryT: { data: queryTData, shape: [1, 3, 8] },
  keyT: { data: keyTData, shape: [1, 3, 4] },
  valueT: { data: valueTData, shape: [1, 3, 4] },
  pastStateT: { data: pastStateTData, shape: [1, 1, 4, 4] },
  decayT: { data: decayTData, shape: [1, 3, 1] },
  betaT: { data: betaTData, shape: [1, 3, 1] },
}, {
  attrs: { q_num_heads: 2, kv_num_heads: 1 },
});
```
