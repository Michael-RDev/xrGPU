---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Pad

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 19

## Description

Pads a tensor along each axis using one of four modes: `constant` (fill with a scalar value), `reflect` (mirror edge values), `edge` (replicate boundary values), or `wrap` (torus-like wrap-around). Supply the required ONNX `pads` tensor, and optional `axes` tensor when present, through the `pads` request attribute as a full-rank list. Negative counts crop the corresponding edge.

See the [ONNX `Pad` spec](https://onnx.ai/onnx/operators/onnx__Pad.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `data` | `T` | — | — | Input tensor to be padded. | required |
| `constant_value` | `T` | `0` | — | Optional scalar value used as the fill constant when `mode` is `constant`; defaults to 0. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `data` | — | Tensor after padding, with each axis enlarged by the corresponding begin and end pad counts. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `mode` | `"constant"` | Padding mode: `constant` (default), `reflect`, `edge`, or `wrap`. |
| `pads` | — | Values of the required `pads` tensor, with an optional `axes` tensor expanded to full rank, supplied as `[x1_begin, ..., xN_begin, x1_end, ..., xN_end]`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint32`, `int32`, `uint8`, `int8`, `bool`, `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`pad-quad-vec4.wgsl.jinja`](build/webgpu/pad-quad-vec4.wgsl.jinja)
- [`pad.wgsl.jinja`](build/webgpu/pad.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `output`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Pad", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({ data: { data: dataData, shape: [2] } }, {
  attrs: { pads: [0, 0] },
  outputs: { output: { shape: [2], dtype: "float32" } },
});
```
