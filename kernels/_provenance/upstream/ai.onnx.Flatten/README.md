---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Flatten

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Flattens the input tensor into a 2-D matrix by splitting its dimensions at `axis`: dimensions before `axis` are collapsed into the first output dimension, and dimensions from `axis` onward into the second. For input shape `(d_0, ..., d_n)`, the output shape is `(d_0 * ... * d_{axis-1}, d_axis * ... * d_n)`.

See the [ONNX `Flatten` spec](https://onnx.ai/onnx/operators/onnx__Flatten.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `input` | `T` | — | — | Input tensor to flatten into a matrix. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | `2` | derived | A 2-D tensor whose outer dimension combines axes before `axis` and whose inner dimension combines the remaining axes. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `1` | The axis at which to split the input dimensions (exclusive upper bound for the outer dimension). Must be in `[-r, r]` where `r` is the input rank; negative values count from the back. When `axis = 0` the output shape is `(1, total_elements)`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32`, `int16`, `int8`, `uint8`, `bool`, `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`datamove-elementwise-copy.wgsl.jinja`](build/webgpu/datamove-elementwise-copy.wgsl.jinja)
- [`datamove-flat-copy.wgsl.jinja`](build/webgpu/datamove-flat-copy.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Flatten", { version: 1 });
const { output } = await kernel({ input: { data: inputData, shape: [2, 1] } });
```
