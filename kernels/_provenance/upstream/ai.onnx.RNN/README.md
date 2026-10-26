---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.RNN

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 22

## Description

Computes a one-layer simple RNN, updating the hidden state each time step as `Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)` where `f` defaults to `Tanh`. Supports `forward`, `reverse`, and `bidirectional` traversal; optional `clip` bounds pre-activation inputs to `[-clip, +clip]`.

See the [ONNX `RNN` spec](https://onnx.ai/onnx/operators/onnx__RNN.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | `3` | — | Input sequences with shape `[seq_length, batch_size, input_size]` when `layout=0`, or `[batch_size, seq_length, input_size]` when `layout=1`. | required |
| `w` | `W` | `T` | `3` | — | Weight tensor for the input gate, shape `[num_directions, hidden_size, input_size]`. | required |
| `r` | `R` | `T` | `3` | — | Recurrence weight tensor, shape `[num_directions, hidden_size, hidden_size]`. | required |
| `b` | `B` | `T` | `2` | — | Bias tensor concatenating input and recurrence biases, shape `[num_directions, 2*hidden_size]`. ONNX defines an omitted bias as zero; this package requires an explicit tensor, which may be zero-filled to represent omission. | required |
| `sequence_lens` | — | `int32` | `1` | — | Per-batch sequence lengths of shape `[batch_size]`; all sequences are treated as full length if absent. | optional |
| `initial_h` | — | `T` | `3` | — | Initial hidden state with shape `[num_directions, batch_size, hidden_size]` when `layout=0`, or `[batch_size, num_directions, hidden_size]` when `layout=1`; assumed zero if absent. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | `4` | derived | All intermediate hidden outputs, with shape `[seq_length, num_directions, batch_size, hidden_size]` when `layout=0`, or `[batch_size, seq_length, num_directions, hidden_size]` when `layout=1`. | required |
| `y_h` | `Y_h` | `T` | `3` | derived | Final hidden state with shape `[num_directions, batch_size, hidden_size]` when `layout=0`, or `[batch_size, num_directions, hidden_size]` when `layout=1`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `activation_alpha` | — | Optional alpha parameters for activation functions that use them, consumed in activation order; omitted entries use each activation's standard default. |
| `activation_beta` | — | Optional beta parameters for activation functions that use them, consumed in activation order; omitted entries use each activation's standard default. |
| `activations` | `["Tanh","Tanh"]` | Activation function for each direction. ONNX defaults both directions to `Tanh`; `forward` and `reverse` use the first entry, while `bidirectional` execution uses both entries. |
| `clip` | — | Optional non-negative threshold applied to pre-activation values as `[-clip, +clip]`; omission disables clipping, while an explicit `0` clamps them to zero. |
| `direction` | `"forward"` | Traversal direction: `"forward"`, `"reverse"`, or `"bidirectional"`. |
| `hidden_size` | — | Optional number of neurons in the hidden layer; when omitted, it is inferred from the W and R tensor shapes. |
| `layout` | `0` | Shape format for `X`, `initial_h`, `Y`, and `Y_h`: `0` puts `seq_length` first; `1` puts `batch_size` first. |

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
- [`rnn-general.wgsl.jinja`](build/webgpu/rnn-general.wgsl.jinja)
- [`rnn-input-proj.wgsl.jinja`](build/webgpu/rnn-input-proj.wgsl.jinja)
- [`rnn-parallel-global.wgsl.jinja`](build/webgpu/rnn-parallel-global.wgsl.jinja)
- [`rnn-parallel.wgsl.jinja`](build/webgpu/rnn-parallel.wgsl.jinja)
- [`rnn-recur-step.wgsl.jinja`](build/webgpu/rnn-recur-step.wgsl.jinja)
- [`rnn-recur-subgroup-step.wgsl.jinja`](build/webgpu/rnn-recur-subgroup-step.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.RNN", { version: 1 });
const { y, y_h } = await kernel({
  x: { data: xData, shape: [1, 1, 1] },
  w: { data: wData, shape: [1, 1, 1] },
  r: { data: rData, shape: [1, 1, 1] },
  b: { data: bData, shape: [1, 2] },
});
```
