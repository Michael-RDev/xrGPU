---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.LSTM

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 22

## Description

Computes a single-layer LSTM over an input sequence using input, output, forget, and cell gates at each time step. Supports `forward`, `reverse`, and `bidirectional` directions, optional peephole connections, and optional per-sequence length masking.

See the [ONNX `LSTM` spec](https://onnx.ai/onnx/operators/onnx__LSTM.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | `3` | — | Input sequences with shape `[seq_length, batch_size, input_size]` when `layout=0`, or `[batch_size, seq_length, input_size]` when `layout=1`. | required |
| `w` | `W` | `T` | `3` | — | Gate weight tensor of shape `[num_directions, 4*hidden_size, input_size]`, concatenating weights for input, output, forget, and cell gates. | required |
| `r` | `R` | `T` | `3` | — | Recurrence weight tensor of shape `[num_directions, 4*hidden_size, hidden_size]`, concatenating recurrence weights for all gates. | required |
| `b` | `B` | `T` | `2` | — | Bias tensor of shape `[num_directions, 8*hidden_size]`, concatenating input and recurrence biases for all gates. ONNX defines an omitted bias as zero; this package requires an explicit tensor, which may be zero-filled to represent omission. | required |
| `sequence_lens` | — | `int32` | `1` | — | Per-batch sequence lengths of shape `[batch_size]`; assumed `seq_length` for all sequences if absent. | optional |
| `initial_h` | — | `T` | `3` | — | Initial hidden state with shape `[num_directions, batch_size, hidden_size]` when `layout=0`, or `[batch_size, num_directions, hidden_size]` when `layout=1`; assumed zero if absent. | optional |
| `initial_c` | — | `T` | `3` | — | Initial cell state with shape `[num_directions, batch_size, hidden_size]` when `layout=0`, or `[batch_size, num_directions, hidden_size]` when `layout=1`; assumed zero if absent. | optional |
| `p` | `P` | `T` | `2` | — | Peephole weight tensor of shape `[num_directions, 3*hidden_size]` for input, output, and forget gates; assumed zero if absent. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | `4` | derived | Intermediate hidden states for all time steps, with shape `[seq_length, num_directions, batch_size, hidden_size]` when `layout=0`, or `[batch_size, seq_length, num_directions, hidden_size]` when `layout=1`. | required |
| `y_h` | `Y_h` | `T` | `3` | derived | Final hidden state with shape `[num_directions, batch_size, hidden_size]` when `layout=0`, or `[batch_size, num_directions, hidden_size]` when `layout=1`. | required |
| `y_c` | `Y_c` | `T` | `3` | derived | Final cell state with shape `[num_directions, batch_size, hidden_size]` when `layout=0`, or `[batch_size, num_directions, hidden_size]` when `layout=1`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `activation_alpha` | — | Optional alpha parameters for activation functions that use alpha, consumed in activation-list order; omitted entries use the ONNX defaults for their activation. |
| `activation_beta` | — | Optional beta parameters for activation functions that use beta, consumed in activation-list order; omitted entries use the ONNX defaults for their activation. |
| `activations` | — | Activation functions for the input/output/forget gates, cell candidate, and cell output. Defaults to `["Sigmoid", "Tanh", "Tanh"]` per direction. |
| `clip` | — | Optional non-negative threshold applied to activation inputs as `[-clip, +clip]`; omission disables clipping, while an explicit `0` clamps them to zero. |
| `direction` | `"forward"` | Computation direction: `"forward"`, `"reverse"`, or `"bidirectional"`. |
| `hidden_size` | — | Optional number of neurons in the hidden layer; when omitted, it is inferred from the W and R tensor shapes. |
| `input_forget` | `0` | When set to `1`, couples the input gate and forget gate (i.e. `ft = 1 - it`). |
| `layout` | `0` | Tensor layout convention: `0` uses `X=[seq_length, batch_size, input_size]`, `Y=[seq_length, num_directions, batch_size, hidden_size]`, and states `[num_directions, batch_size, hidden_size]`; `1` uses `X=[batch_size, seq_length, input_size]`, `Y=[batch_size, seq_length, num_directions, hidden_size]`, and states `[batch_size, num_directions, hidden_size]`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`datamove-transpose-2d-tiled.wgsl.jinja`](build/webgpu/datamove-transpose-2d-tiled.wgsl.jinja)
- [`lstm-general.wgsl.jinja`](build/webgpu/lstm-general.wgsl.jinja)
- [`lstm-parallel-global.wgsl.jinja`](build/webgpu/lstm-parallel-global.wgsl.jinja)
- [`lstm-recur-step.wgsl.jinja`](build/webgpu/lstm-recur-step.wgsl.jinja)
- [`lstm-recur-subgroup-batched-step.wgsl.jinja`](build/webgpu/lstm-recur-subgroup-batched-step.wgsl.jinja)
- [`lstm-recur-subgroup-step.wgsl.jinja`](build/webgpu/lstm-recur-subgroup-step.wgsl.jinja)
- [`lstm-seed-state.wgsl.jinja`](build/webgpu/lstm-seed-state.wgsl.jinja)
- [`lstm-timestep-body.wgsl.jinja`](build/webgpu/lstm-timestep-body.wgsl.jinja)
- [`matmul-notrans-vec4-tiled-reg.wgsl.jinja`](build/webgpu/matmul-notrans-vec4-tiled-reg.wgsl.jinja)
- [`rnn-input-proj.wgsl.jinja`](build/webgpu/rnn-input-proj.wgsl.jinja)
- [`rnn-sequence-mask.wgsl.jinja`](build/webgpu/rnn-sequence-mask.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.LSTM", { version: 1 });
const { y, y_h, y_c } = await kernel({
  x: { data: xData, shape: [1, 1, 1] },
  w: { data: wData, shape: [1, 4, 1] },
  r: { data: rData, shape: [1, 4, 1] },
  b: { data: bData, shape: [1, 8] },
});
```
