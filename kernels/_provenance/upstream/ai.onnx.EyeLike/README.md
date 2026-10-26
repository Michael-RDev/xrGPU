---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.EyeLike

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 22

## Description

Generates a 2D identity-like matrix with ones on (or offset from) the main diagonal and zeros everywhere else. The output has the same shape as the 2D input tensor; the output dtype defaults to the input dtype but can be overridden. Attribute `k` shifts the populated diagonal: `k=0` is the main diagonal, `k>0` is upper, `k<0` is lower.

See the [ONNX `EyeLike` spec](https://onnx.ai/onnx/operators/onnx__EyeLike.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `input` | `T1` | `2` | — | 2D input tensor whose shape (and optionally type) is copied. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T2` | same as `input` | same as `input` | Output tensor of the same shape as the input, with ones on the selected diagonal and zeros elsewhere. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `dtype` | — | Optional TensorProto DataType enum for the output. When omitted, the output dtype is the same as the input dtype. |
| `k` | `0` | Index of the diagonal to populate with ones: `0` is the main diagonal, positive values select upper diagonals, negative values select lower diagonals. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T1` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool` |
| `T2` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`eyelike-clear-vec4.wgsl.jinja`](build/webgpu/eyelike-clear-vec4.wgsl.jinja)
- [`eyelike-diagonal.wgsl.jinja`](build/webgpu/eyelike-diagonal.wgsl.jinja)
- [`eyelike.wgsl.jinja`](build/webgpu/eyelike.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.EyeLike", { version: 1 });
const { output } = await kernel({ input: { data: inputData, shape: [2, 2] } });
```
