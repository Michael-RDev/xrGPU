---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.CausalConvWithState

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Microsoft contrib stateful 1-D causal depthwise convolution. Each channel uses its own `(channels, 1, kernel_size)` weight over current and past positions, with optional activation and `past_state`/`present_state` tensors for incremental decoding. The `state_window` attribute may retain several rollback states. This package supports `ndim = 1`, float16 or float32 tensors, and float32 accumulation; spatial ranks 2 and 3 and bfloat16 are unsupported.

See the [ONNX Runtime `CausalConvWithState` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.CausalConvWithState) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `inputT` | `input` | `T` | `3` | — | Input with shape `(batch, channels, length)`, or `(batch, length, channels)` when `channels_last` is 1. | required |
| `weightT` | `weight` | `T` | `3` | — | Depthwise convolution kernel with shape `(channels, 1, kernel_size)` for the supported 1-D mode. | required |
| `biasT` | `bias` | `T` | `1` | — | Optional per-channel bias with shape `(channels,)`. | optional |
| `pastStateT` | `past_state` | `T` | derived | — | Carry state from the previous step; shape `(batch_size, channels, (kernel_size - 1) * dilation)`, or `(W, batch_size, channels, (kernel_size - 1) * dilation)` when `state_window = W > 0`, in which case only slot `W - 1` is read. If absent, the left-side padding is zero. With `channels_last`=1, the length and channel axes are swapped. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `3` | same as `inputT` | Convolution output with the same shape as `input`. | required |
| `presentStateT` | `present_state` | `T` | derived | derived | Updated carry state; shape `(batch_size, channels, (kernel_size - 1) * dilation)`, or `(W, batch_size, channels, (kernel_size - 1) * dilation)` when `state_window = W > 0`. Slot `W - 1` holds the last `(kernel_size - 1) * dilation` values along the causal axis; slot `j` holds the same values for the prefix ending at position `sequence_length - W + j`. With `channels_last`=1, the length and channel axes are swapped. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `activation` | `"none"` | Activation applied after convolution and bias. Defaults to `none`; `swish` is an alias of SiLU. |
| `channels_last` | `0` | Activation and state layout: 0 selects `(batch, channels, length)`; 1 selects `(batch, length, channels)`. |
| `dilation` | `1` | Positive integer spacing between taps; the carry state holds (`kernel_size` - 1) * dilation raw samples. |
| `ndim` | `1` | Number of spatial dimensions. This implementation supports the contrib 1D mode (`ndim = 1`). |
| `state_window` | `0` | Contrib extension selecting the number of rollback state slots to retain, in the range 0 through 8. Defaults to 0. |

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.CausalConvWithState", { version: 1 });
const { outputT, presentStateT } = await kernel({
  inputT: { data: inputTData, shape: [1, 1, 5] },
  weightT: { data: weightTData, shape: [1, 1, 4] },
});
```
