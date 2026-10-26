---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.VarlenCausalConvWithState

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Stateful causal depthwise convolution over packed token-major variable-length sequences, without reads across sequence boundaries. `initial_state` carries preceding raw samples and `final_state` is fully written. At positive `state_update_capacity`, `capture_count` selects a clamped prefix of raw input tokens for compact `state_update`; inactive slots are zero. SiLU and Swish are aliases. This implementation supports float16 and float32 with float32 accumulation; bfloat16 is not implemented.

See the [ONNX Runtime `VarlenCausalConvWithState` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.VarlenCausalConvWithState) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `inputT` | `input` | `T` | same as logical dtype | `2` | — | Token-major packed input with shape `(total_tokens, channels)`. | required |
| `weightT` | `weight` | `T` | same as logical dtype | `3` | — | Depthwise kernel with shape `(channels, 1, kernel_size)`. | required |
| `cumulativeSequenceLengthT` | `cumulative_sequence_length` | `M` | `int32` | `1` | — | Exclusive prefix sums with shape `(batch_size + 1)`, starting at 0, ending at `total_tokens`, and strictly increasing so every sequence is non-empty. Sequence `i` owns tokens `[cum[i], cum[i + 1])`. Outputs are unspecified for a malformed schedule. | required |
| `biasT` | `bias` | `T` | same as logical dtype | `1` | — | Optional per-channel bias with shape `(channels,)`. In an ONNX graph an omitted bias must still occupy input index 3 as an empty name so `initial_state` stays at index 4. | optional |
| `initialStateT` | `initial_state` | `T` | same as logical dtype | `3` | — | Required committed carry state with shape `(batch_size, channels, (kernel_size - 1) * dilation)`, holding the raw samples immediately preceding this call. | required |
| `captureCountT` | `capture_count` | `M` | `int32` | `1` | — | Optional int32 vector with shape `(batch_size)`. Required exactly when `state_update_capacity` is positive; each value is clamped to `[0, min(state_update_capacity, sequence_length)]`. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | same as `inputT` | same as `inputT` | Convolution output with the same shape as `input`. | required |
| `finalStateT` | `final_state` | `T` | `3` | derived | State after each sequence's final token, shape `(batch_size, channels, (kernel_size - 1) * dilation)`. Always fully written. | required |
| `stateUpdateT` | `state_update` | `T` | `3` | derived | Optional compact transition values with shape `(batch_size, state_update_capacity, channels)`. Active slots contain the original local input tokens and all other slots are zero. | optional |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `activation` | `"none"` | Fused activation applied after convolution and bias. One of `none`, `silu`, or `swish`; the standard default is `none`. |
| `dilation` | `1` | Positive integer spacing between taps; the initial and final states hold (`kernel_size` - 1) * dilation raw samples. |
| `state_update_capacity` | `0` | Static number of compact per-request prefix transition values to expose, in `[0, 8]`. The standard default is 0. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `M` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `unit_plain` — Direct single-token sequence convolution with contiguous vector state traffic and fused capture writes. Strictly increasing schedules with `total_tokens == batch_size` imply one token per sequence. Channel divisibility selects the vector width; the workgroup respects device limits. Dilated and f16 requests retain their existing paths.
- `unit_bias` — Direct single-token sequence convolution with contiguous vector state traffic and fused capture writes. Strictly increasing schedules with `total_tokens == batch_size` imply one token per sequence. Channel divisibility selects the vector width; the workgroup respects device limits. Dilated and f16 requests retain their existing paths.
- `unit_state_update` — Direct single-token sequence convolution with contiguous vector state traffic and fused capture writes. Strictly increasing schedules with `total_tokens == batch_size` imply one token per sequence. Channel divisibility selects the vector width; the workgroup respects device limits. Dilated and f16 requests retain their existing paths.
- `unit_bias_state_update` — Direct single-token sequence convolution with contiguous vector state traffic and fused capture writes. Strictly increasing schedules with `total_tokens == batch_size` imply one token per sequence. Channel divisibility selects the vector width; the workgroup respects device limits. Dilated and f16 requests retain their existing paths.
- `capture_copy_state_update_stream` — Copy the capture prefix in contiguous channel vectors while preserving the convolution path and exact zero-fill semantics. The existing copy template operates on vector groups; no extra storage bindings or optional GPU features are needed.
- `capture_copy_state_update` — Copy the capture prefix in contiguous channel vectors while preserving the convolution path and exact zero-fill semantics. The existing copy template operates on vector groups; no extra storage bindings or optional GPU features are needed.
- `capture_copy_bias_state_update_stream` — Copy the capture prefix in contiguous channel vectors while preserving the convolution path and exact zero-fill semantics. The existing copy template operates on vector groups; no extra storage bindings or optional GPU features are needed.
- `capture_copy_bias_state_update` — Copy the capture prefix in contiguous channel vectors while preserving the convolution path and exact zero-fill semantics. The existing copy template operates on vector groups; no extra storage bindings or optional GPU features are needed.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`varlen-causal-conv-stream.wgsl.jinja`](build/webgpu/varlen-causal-conv-stream.wgsl.jinja)
- [`varlen-causal-conv.wgsl.jinja`](build/webgpu/varlen-causal-conv.wgsl.jinja)
- [`varlen-state-update.wgsl.jinja`](build/webgpu/varlen-state-update.wgsl.jinja)
- [`varlen-unit-sequence.wgsl.jinja`](build/webgpu/varlen-unit-sequence.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.VarlenCausalConvWithState", { version: 1 });
const { outputT, finalStateT } = await kernel({
  inputT: { data: inputTData, shape: [1, 4] },
  weightT: { data: weightTData, shape: [4, 1, 4] },
  cumulativeSequenceLengthT: { data: cumulativeSequenceLengthTData, shape: [2] },
  initialStateT: { data: initialStateTData, shape: [1, 4, 3] },
});
```
