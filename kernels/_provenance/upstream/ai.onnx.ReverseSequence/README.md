---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.ReverseSequence

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 10

## Description

Reverses variable-length sequences in a batch along a time axis. For each batch slice `i`, the first `sequence_lens[i]` elements along `time_axis` are reversed in place; elements beyond that length are copied unchanged. The output has the same shape as the input.

See the [ONNX `ReverseSequence` spec](https://onnx.ai/onnx/operators/onnx__ReverseSequence.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `input` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | Input tensor of rank 2 or greater containing the batch of sequences to reverse. | required |
| `sequence_lens` | `I` | `uint32` | `1` | — | Logical int64 1-D tensor of shape `[batch_size]` specifying each non-negative sequence length; WebGPU stores it as uint32. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `input` | same as `input` | Output tensor with the same shape as the input, with each sequence reversed up to its specified length. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `batch_axis` | `1` | Axis that represents the batch dimension; must be 0 or 1 (default 1). |
| `time_axis` | `0` | Axis that represents the time/sequence dimension to be reversed; must be 0 or 1 (default 0). |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool` |
| `I` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `rankn_output_packed8` — Copies eight half-precision values per invocation using integer storage words, preserving each source payload across reversal and vector boundaries.
- `rankn_output_vec4` — Writes four contiguous output values per invocation while independently gathering their reversed source positions.
- `rankn_vec4` — Copies aligned inner vectors whose four components share a reversed time slot.
- `rankn` — Copies individual elements for shapes that cannot form complete output vectors.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`reverse-sequence-output.wgsl.jinja`](build/webgpu/reverse-sequence-output.wgsl.jinja)
- [`reverse-sequence-vec4.wgsl.jinja`](build/webgpu/reverse-sequence-vec4.wgsl.jinja)
- [`reverse-sequence.wgsl.jinja`](build/webgpu/reverse-sequence.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.ReverseSequence", { version: 1 });
const { y } = await kernel({
  input: { data: inputData, shape: [4, 2, 1] },
  sequence_lens: { data: sequence_lensData, shape: [2] },
});
```
