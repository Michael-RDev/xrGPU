---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.PRelu

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 16

## Description

Applies the parametric ReLU function elementwise: `f(x) = x` for `x >= 0` and `f(x) = slope * x` for `x < 0`. The `slope` tensor may be smaller than `X` and is broadcast unidirectionally to match `X`.

See the [ONNX `PRelu` spec](https://onnx.ai/onnx/operators/onnx__PRelu.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Values transformed by parametric ReLU using the broadcast `slope`. | required |
| `slope` | — | `T` | — | — | Per-element slope values; must be unidirectionally broadcastable to X. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | same as `x` | Output tensor of the same shape as X. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`prelu-vec4.wgsl.jinja`](build/webgpu/prelu-vec4.wgsl.jinja)
- [`prelu.wgsl.jinja`](build/webgpu/prelu.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.PRelu", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [3, 1, 1, 1] },
  slope: { data: slopeData, shape: [1, 1, 1] },
});
```
