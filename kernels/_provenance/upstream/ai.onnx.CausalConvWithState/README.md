---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.CausalConvWithState

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 27

## Description

Stateful causal 1-D depthwise convolution. It left-pads each channel with `past_state` (or zeros), applies a `(channels, 1, kernel_size)` depthwise convolution and optional SiLU/Swish activation, and returns both the output and the last `kernel_size - 1` values as `present_state`. This package supports float16 and float32 tensors; bfloat16 is unsupported.

See the [ONNX `CausalConvWithState` spec](https://onnx.ai/onnx/operators/onnx__CausalConvWithState.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `inputT` | `input` | `T` | `3` | — | Input tensor with shape `(batch_size, channels, length)` in channels-first layout. | required |
| `weightT` | `weight` | `T` | `3` | — | Depthwise convolution kernel with shape `(channels, 1, kernel_size)`, matching the ONNX Conv weight layout for `group = channels`. | required |
| `biasT` | `bias` | `T` | `1` | — | Optional per-channel bias with shape `(channels)`. | optional |
| `pastStateT` | `past_state` | `T` | `3` | — | Carry state from the previous step with shape `(batch_size, channels, kernel_size - 1)`. If omitted, the left padding is zero. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `3` | same as `inputT` | Convolution output with the same shape as `input`. | required |
| `presentStateT` | `present_state` | `T` | `3` | derived | Updated carry state with shape `(batch_size, channels, kernel_size - 1)`, including prior state or zero-padding when the current input is shorter than the state. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `activation` | `"none"` | Optional fused activation. `silu` and `swish` are aliases; `none` leaves the convolution result unchanged. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`causal-conv-with-state-tiled.wgsl.jinja`](build/webgpu/causal-conv-with-state-tiled.wgsl.jinja)
- [`causal-conv-with-state-vec4.wgsl.jinja`](build/webgpu/causal-conv-with-state-vec4.wgsl.jinja)
- [`causal-conv-with-state.wgsl.jinja`](build/webgpu/causal-conv-with-state.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.CausalConvWithState", { version: 1 });
const { outputT, presentStateT } = await kernel({
  inputT: { data: inputTData, shape: [1, 1, 2] },
  weightT: { data: weightTData, shape: [1, 1, 5] },
});
```
