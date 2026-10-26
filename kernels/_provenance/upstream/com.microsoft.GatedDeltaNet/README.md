---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.GatedDeltaNet

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Token-major gated delta network with an explicit recurrent state: `S_t = exp(g_t) * S_{t-1} + k_t * (beta_t * (v_t - exp(g_t) * S_{t-1}^T k_t))^T` and `o_t = scale * S_t^T q_t` per value head; `update_rule` selects which terms survive. Query, key and value are `(total_tokens, heads, head_size)` or `(batch, sequence, heads, head_size)`. `num_heads_v` is a positive multiple of `num_heads_q`, and value head `h` reads query head `h * num_heads_q / num_heads_v`. The V-major state and the gate fusions are float32. Bfloat16, per-key-dimension decay and compact capture are not implemented.

See the [ONNX Runtime `GatedDeltaNet` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GatedDeltaNet) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `queryT` | `query` | `T` | — | — | Query vectors, `(total_tokens, num_heads_q, head_size_qk)` or `(batch_size, sequence_length, num_heads_q, head_size_qk)`. The two spellings share one memory layout; the rank-4 form lets an exporter keep a static reshape target. | required |
| `keyT` | `key` | `T` | — | — | Key vectors with exactly the shape of `query`. The delta rules need L2-normalized keys, either from upstream or from `qk_l2_norm=1`; without them the per-chunk system is ill-conditioned and the recurrence diverges. | required |
| `valueT` | `value` | `T` | — | — | Value vectors, `(total_tokens, num_heads_v, head_size_v)`, sharing the leading token axes of `query`. | required |
| `cuSeqlensT` | `cu_seqlens` | `TI` | `1` | — | Exclusive prefix sums of the per-request token counts, `(batch_size + 1)`. Requires the rank-3 spelling. Offsets are clamped on device to `[0, total_tokens]`, so a decreasing or out-of-range entry yields an empty request rather than an out-of-bounds read. Absent means uniform packing. | optional |
| `decayT` | `decay` | `TS` | — | — | Log-space decay, `(..., num_heads_v)` over the same leading token axes as `query`, always float32. Present exactly for `update_rule` `gated` and `gated_delta`. The per-key-dimension spelling `(..., num_heads_v, head_size_qk)` is not implemented. | optional |
| `betaT` | `beta` | `TS` | — | — | Update rate, `(..., num_heads_v)` over the same leading token axes as `query`, always float32. Present exactly for `update_rule` `delta` and `gated_delta`. | optional |
| `initialStateT` | `initial_state` | `TS` | `4` | — | Incoming recurrent state, V-major `(batch_size, num_heads_v, head_size_v, head_size_qk)`, always float32. Absent means a zero state. It is a separate allocation from `final_state`: WebGPU usage-scope validation tracks whole buffers, so binding one buffer as both read-only and read-write invalidates the command buffer. | optional |
| `aLogT` | `a_log` | `TS` | `1` | — | Per-head `A_log`, `(num_heads_v)`, always float32. Present exactly when `gate_activation` is `qwen`. | optional |
| `dtBiasT` | `dt_bias` | `TS` | `1` | — | Per-head gate bias, `(num_heads_v)`, always float32. Present exactly when `gate_activation` is `qwen`. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | same as `queryT` | derived | Attention output, `(total_tokens, num_heads_v, head_size_v)` or `(batch_size, sequence_length, num_heads_v, head_size_v)`. `num_heads_v` is the larger head count because it is a positive multiple of `num_heads_q`. | required |
| `finalStateT` | `final_state` | `TS` | `4` | derived | State after the last token of each request, V-major `(batch_size, num_heads_v, head_size_v, head_size_qk)`, always float32. A request with no tokens copies its incoming state through. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `beta_activation` | `"none"` | `none` treats `beta` as the effective update rate; `sigmoid` applies a logistic sigmoid to it in float32. Default is `none`. |
| `chunk_size` | `64` | Accepted scheduling hint for a chunk-parallel prefill algorithm. Every value is accepted and none changes the result. Default is `64`. |
| `gate_activation` | `"none"` | `none` treats `decay` as the effective log-space decay. `qwen` computes `-exp(a_log) * Softplus(decay + dt_bias)` in float32 and requires both `a_log` and `dt_bias`. Default is `none`. |
| `qk_l2_norm` | `0` | Set to `1` to L2-normalize each query and key head vector, with epsilon `1e-12`, before the recurrence. Default is `0`. |
| `scale` | `0` | Output scaling factor applied to `S_t^T q_t`. The default `0.0` selects `1 / sqrt(head_size_qk)`. |
| `state_update_capacity` | `0` | Capacity for compact transition capture. The default and only supported value is `0`; compact `state_update` capture is not implemented. |
| `update_rule` | `"gated_delta"` | Which terms of the recurrence survive: `linear` drops the decay and the delta retrieval, `gated` keeps only the decay, `delta` keeps only the retrieval, and `gated_delta` keeps both. Default is `gated_delta`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `TS` | `float32` |
| `TI` | `int32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`gated-delta-net-gate.wgsl.jinja`](build/webgpu/gated-delta-net-gate.wgsl.jinja)
- [`gated-delta-net.wgsl.jinja`](build/webgpu/gated-delta-net.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.GatedDeltaNet", { version: 1 });
const { outputT, finalStateT } = await kernel({
  queryT: { data: queryTData, shape: [4, 1, 12] },
  keyT: { data: keyTData, shape: [4, 1, 12] },
  valueT: { data: valueTData, shape: [4, 2, 9] },
  decayT: { data: decayTData, shape: [4, 2] },
  betaT: { data: betaTData, shape: [4, 2] },
  initialStateT: { data: initialStateTData, shape: [1, 2, 9, 12] },
});
```
