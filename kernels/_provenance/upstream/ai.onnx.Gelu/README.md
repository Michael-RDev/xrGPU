---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Gelu

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 20

## Description

Applies the Gaussian Error Linear Unit activation elementwise: `y = 0.5 * x * (1 + erf(x / sqrt(2)))`. When `approximate` is `"tanh"`, uses the tanh-based approximation `y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))` instead.

See the [ONNX `Gelu` spec](https://onnx.ai/onnx/operators/onnx__Gelu.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Values transformed elementwise by the selected GELU formulation. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | same as `x` | Output tensor; same shape as the input, with GELU applied elementwise. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `approximate` | `"none"` | Selects the GELU approximation algorithm: `"none"` (default) uses the `erf`-based formula; `"tanh"` uses the tanh-based polynomial approximation. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `same_layout_vec4_tail` — Processes the vec4-aligned prefix in a bulk pass and any remaining elements in a scalar tail pass.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`unary-scalar.wgsl.jinja`](build/webgpu/unary-scalar.wgsl.jinja)
- [`unary-vec4.wgsl.jinja`](build/webgpu/unary-vec4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Gelu", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [3] } });
```
