---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.LinearAttention

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 27

## Description

Computes ONNX recurrent linear attention for packed `[B, T, H*D]` tensors. Standard GQA requires positive `H_q` divisible by `H_kv`; each KV head owns one state, and `d_v` must be positive. Supports all four update rules, optional rank-4 state, and float16 or float32 activations/state; bfloat16 is not implemented. Every positive `d_k` through 256 is supported; larger four-aligned widths are accepted when their required resources fit the WebGPU limits. Other head widths are unsupported.

See the [ONNX `LinearAttention` spec](https://onnx.ai/onnx/operators/onnx__LinearAttention.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `queryT` | `query` | `T` | `3` | — | Query vectors with packed shape `(B, T, H_q * d_k)`; `H_q` must be a positive multiple of `H_kv`. | required |
| `keyT` | `key` | `T` | `3` | — | Key vectors with packed shape `(B, T, H_kv * d_k)`; keys should be L2-normalized for the `delta` update rules. | required |
| `valueT` | `value` | `T` | `3` | — | Value vectors with packed shape `(B, T, H_kv * d_v)`. | required |
| `pastStateT` | `past_state` | `S` | `4` | derived | Optional recurrent state with shape `(B, H_kv, d_k, d_v)`. When absent, the recurrence starts from zero. | optional |
| `decayT` | `decay` | `T` | `3` | — | Log-space decay with shape `(B, T, H_kv * d_k)` or `(B, T, H_kv)`. Required only by `gated` and `gated_delta`. | optional |
| `betaT` | `beta` | `T` | `3` | — | Update rate with shape `(B, T, H_kv)` or `(B, T, 1)`. Required only by `delta` and `gated_delta`. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `3` | derived | Attention output with packed shape `(B, T, H_q * d_v)`. | required |
| `presentStateT` | `present_state` | `S` | `4` | derived | Recurrent state after the final token, with shape `(B, H_kv, d_k, d_v)`. Its dtype matches `past_state` when supplied, otherwise it matches `query`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `chunk_size` | `64` | Correctness-neutral chunk-size hint for parallel prefill implementations. Every integer value has identical semantics. |
| `kv_num_heads` | — | Required positive number of key/value heads. |
| `q_num_heads` | — | Required positive number of query heads; must be a multiple of `kv_num_heads`. |
| `scale` | `0` | Output scaling factor. Zero selects `1 / sqrt(d_k)`. |
| `update_rule` | `"gated_delta"` | Recurrent update rule: `linear`, `gated`, `delta`, or `gated_delta`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `S` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`linear-attention.scalar.wgsl.jinja`](build/webgpu/linear-attention.scalar.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.LinearAttention", { version: 1 });
const { outputT, presentStateT } = await kernel({
  queryT: { data: queryTData, shape: [1, 2, 8] },
  keyT: { data: keyTData, shape: [1, 2, 4] },
  valueT: { data: valueTData, shape: [1, 2, 3] },
  decayT: { data: decayTData, shape: [1, 2, 4] },
  betaT: { data: betaTData, shape: [1, 2, 1] },
}, {
  attrs: { q_num_heads: 2, kv_num_heads: 1 },
});
```
