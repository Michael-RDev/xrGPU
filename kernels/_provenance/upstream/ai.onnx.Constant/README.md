---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Constant

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 24

## Description

Produces a numeric constant from ONNX's tensor-valued `value` attribute or its float scalar/list shorthands. Sparse, string, and int64-only shorthand attributes are unsupported.

See the [ONNX `Constant` spec](https://onnx.ai/onnx/operators/onnx__Constant.html) for the reference semantics.

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | — | — | Output tensor containing the constant value(s) specified by the attribute. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `value` | — | Tensor-valued constant represented as `{ dtype, shape, values }`; `dtype` and `shape` must match the output contract and `values` supplies the tensor elements in row-major order. |
| `value_float` | — | A single float32 value producing a scalar tensor. |
| `value_floats` | — | Float32 values producing a one-dimensional tensor. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `int16`, `uint32`, `int8`, `uint8`, `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`constant.wgsl.jinja`](build/webgpu/constant.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `output`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Constant", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({}, {
  attrs: { value_float: -2.5 },
  outputs: { output: { shape: [], dtype: "float32" } },
});
```
