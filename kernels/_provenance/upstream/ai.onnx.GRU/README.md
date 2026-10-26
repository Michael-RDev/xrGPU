---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.GRU

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 22

## Description

Computes a single-layer Gated Recurrent Unit (GRU) over an input sequence, applying update (`z`), reset (`r`), and hidden (`h`) gates at each time step. Supports `forward`, `reverse`, and `bidirectional` traversal; activation functions default to Sigmoid for `f` and Tanh for `g`.

See the [ONNX `GRU` spec](https://onnx.ai/onnx/operators/onnx__GRU.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | `3` | — | Input sequences with shape `[seq_length, batch_size, input_size]` when `layout=0`, or `[batch_size, seq_length, input_size]` when `layout=1`. | required |
| `w` | `W` | `T` | `3` | — | Gate weight tensor with shape `[num_directions, 3*hidden_size, input_size]`, ordered as `W[zrh]` within each direction. | required |
| `r` | `R` | `T` | `3` | — | Recurrence weight tensor with shape `[num_directions, 3*hidden_size, hidden_size]`, ordered as `R[zrh]` within each direction. | required |
| `b` | `B` | `T` | `2` | — | Bias tensor for the gates, shape `[num_directions, 6*hidden_size]`, concatenating input and recurrence biases. ONNX defines an omitted bias as zero; this package requires an explicit tensor, which may be zero-filled to represent omission. | required |
| `sequence_lens` | — | `int32` | `1` | — | Per-batch sequence lengths of shape `[batch_size]`; assumed all equal to `seq_length` if absent. | optional |
| `initial_h` | — | `T` | `3` | — | Initial hidden state with shape `[num_directions, batch_size, hidden_size]` when `layout=0`, or `[batch_size, num_directions, hidden_size]` when `layout=1`; assumed zero if absent. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | `4` | derived | Intermediate hidden outputs for all time steps, with shape `[seq_length, num_directions, batch_size, hidden_size]` when `layout=0`, or `[batch_size, seq_length, num_directions, hidden_size]` when `layout=1`. | required |
| `y_h` | `Y_h` | `T` | `3` | derived | Final hidden state with shape `[num_directions, batch_size, hidden_size]` when `layout=0`, or `[batch_size, num_directions, hidden_size]` when `layout=1`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `activation_alpha` | — | Optional alpha parameters for activation functions that use alpha, consumed in activation-list order; omitted entries use the ONNX defaults for their activation. |
| `activation_beta` | — | Optional beta parameters for activation functions that use beta, consumed in activation-list order; omitted entries use the ONNX defaults for their activation. |
| `activations` | — | Activation functions for the update/reset gates and candidate state. Defaults to `["Sigmoid", "Tanh"]` per direction. |
| `clip` | — | Optional non-negative threshold applied to activation inputs as `[-clip, +clip]`; omission disables clipping, while an explicit `0` clamps them to zero. |
| `direction` | `"forward"` | Specifies traversal direction: `"forward"` (default), `"reverse"`, or `"bidirectional"`. |
| `hidden_size` | — | Optional number of neurons in the hidden layer; when omitted, it is inferred from the W and R tensor shapes. |
| `layout` | `0` | Controls the shape format of `X`, `initial_h`, `Y`, and `Y_h`: `0` uses `[seq_length, batch_size, ...]` ordering (default), `1` uses `[batch_size, seq_length, ...]`. |
| `linear_before_reset` | `0` | When non-zero, applies the recurrence linear transform to the hidden state before multiplying by the reset gate output. |

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
- [`dense-tiled-matmul.wgsl.jinja`](build/webgpu/dense-tiled-matmul.wgsl.jinja)
- [`gru-general.wgsl.jinja`](build/webgpu/gru-general.wgsl.jinja)
- [`gru-parallel-global.wgsl.jinja`](build/webgpu/gru-parallel-global.wgsl.jinja)
- [`gru-recur-candidate-step.wgsl.jinja`](build/webgpu/gru-recur-candidate-step.wgsl.jinja)
- [`gru-recur-candidate-subgroup-step.wgsl.jinja`](build/webgpu/gru-recur-candidate-subgroup-step.wgsl.jinja)
- [`gru-recur-gates-step.wgsl.jinja`](build/webgpu/gru-recur-gates-step.wgsl.jinja)
- [`gru-recur-gates-subgroup-step.wgsl.jinja`](build/webgpu/gru-recur-gates-subgroup-step.wgsl.jinja)
- [`gru-recur-init-step.wgsl.jinja`](build/webgpu/gru-recur-init-step.wgsl.jinja)
- [`gru-timestep-body.wgsl.jinja`](build/webgpu/gru-timestep-body.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.GRU", { version: 1 });
const { y, y_h } = await kernel({
  x: { data: xData, shape: [1, 1, 1] },
  w: { data: wData, shape: [1, 3, 1] },
  r: { data: rData, shape: [1, 3, 1] },
  b: { data: bData, shape: [1, 6] },
});
```
