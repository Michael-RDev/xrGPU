---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.QuantizeLinear

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 25

## Description

Linearly quantizes a high-precision tensor to a lower-precision integer type using the formula `y = saturate((x / y_scale) + y_zero_point)`, with rounding to nearest even. Supports per-tensor, per-axis, and blocked quantization granularities determined by the shape of `y_scale`.

See the [ONNX `QuantizeLinear` spec](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `x` | `TX` | — | — | N-D full-precision input tensor to be quantized. | required |
| `y_scale` | `TS` | — | — | Scale factor; scalar for per-tensor, 1-D for per-axis, or same rank as `x` (with one axis blocked) for blocked quantization. | required |
| `y_zero_point` | `TQ` | — | — | Zero point for quantization; must have the same shape as `y_scale`. Defaults to zero if omitted. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `y` | `TQ` | same as `x` | same as `x` | N-D quantized output tensor with the same shape as `x`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `1` | Axis of the quantization dimension in `x`, used for per-axis and blocked quantization; negative values count from the end. |
| `block_size` | `0` | Number of elements along `axis` that share a single scale value for blocked quantization; 0 means blocked quantization is not used. |
| `output_dtype` | `0` | ONNX TensorProto element-type code for `y`; 0 infers the type from `y_zero_point`, or uint8 when the zero point is omitted. |
| `precision` | `0` | ONNX TensorProto element-type code used for `x / y_scale`; `0` uses the dtype of `y_scale`, `1` selects FLOAT, and `10` selects FLOAT16. |
| `saturate` | `1` | Controls out-of-range conversion for float8 outputs. The implemented int8/uint8 subset accepts the ONNX default `1`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `TX` | `float32`, `float16` |
| `TS` | `float32`, `float16` |
| `TQ` | `uint8`, `int8` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`quant-linear-blocked-axis.wgsl.jinja`](build/webgpu/quant-linear-blocked-axis.wgsl.jinja)
- [`quant-linear-scalar.wgsl.jinja`](build/webgpu/quant-linear-scalar.wgsl.jinja)
- [`quant-linear-vec4.wgsl.jinja`](build/webgpu/quant-linear-vec4.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `y`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.QuantizeLinear", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({
  x: { data: xData, shape: [1] },
  y_scale: { data: y_scaleData, shape: [1] },
}, {
  outputs: { y: { shape: [1], dtype: "uint8" } },
});
```
