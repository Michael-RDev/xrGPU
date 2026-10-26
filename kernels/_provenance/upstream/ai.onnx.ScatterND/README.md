---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.ScatterND

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 18

## Description

Copies `data` and then scatters values from `updates` into the copy at positions specified by `indices`, producing an output with the same shape as `data`. Each entry in `indices` is a k-tuple that addresses either a single element (when k equals the rank of `data`) or a slice; an optional `reduction` attribute replaces the default overwrite with an elementwise `add`, `mul`, `max`, or `min` accumulation.

See the [ONNX `ScatterND` spec](https://onnx.ai/onnx/operators/onnx__ScatterND.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `data` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | Source tensor of rank r >= 1 that is copied to form the base of the output. | required |
| `indices` | `I` | `int32` | — | — | Logical int64 index tensor of rank q >= 1 whose last dimension gives signed k-tuples that address into `data`; WebGPU stores it as int32. | required |
| `updates` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | — | — | Values to scatter into the output, of rank q + r - k - 1, matching the leading (q-1) index dimensions and the trailing (r-k) data slice dimensions. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `data` | same as `data` | Result tensor of rank r with the same shape as `data`, with selected positions overwritten or reduced by `updates`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `reduction` | `"none"` | Reduction to apply when writing each update value: `none` (overwrite, no duplicate index tuples), `add`, `mul`, `max`, or `min`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32`, `int8`, `uint8`, `int16`, `bool`, `int64` |
| `I` | `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`datamove-scatter-row.wgsl.jinja`](build/webgpu/datamove-scatter-row.wgsl.jinja)
- [`scatter-f16-f32-convert.wgsl.jinja`](build/webgpu/scatter-f16-f32-convert.wgsl.jinja)
- [`scatter-flat-copy.wgsl.jinja`](build/webgpu/scatter-flat-copy.wgsl.jinja)
- [`scatter-narrow-wrap.wgsl.jinja`](build/webgpu/scatter-narrow-wrap.wgsl.jinja)
- [`scatter-nd-reduction-atomic.wgsl.jinja`](build/webgpu/scatter-nd-reduction-atomic.wgsl.jinja)
- [`scatter-nd-reduction-f16-packed-atomic.wgsl.jinja`](build/webgpu/scatter-nd-reduction-f16-packed-atomic.wgsl.jinja)
- [`scatter-nd-reduction-slice.wgsl.jinja`](build/webgpu/scatter-nd-reduction-slice.wgsl.jinja)
- [`scatter-nd.wgsl.jinja`](build/webgpu/scatter-nd.wgsl.jinja)
- [`scatter-scalar-add.wgsl.jinja`](build/webgpu/scatter-scalar-add.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.ScatterND", { version: 1 });
const { output } = await kernel({
  data: { data: dataData, shape: [2, 2] },
  indices: { data: indicesData, shape: [2, 1] },
  updates: { data: updatesData, shape: [2, 2] },
});
```
