---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.CastLike

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 25

## Description

Casts every element of `input` to the same dtype as `target_type`, producing an output with the same shape as `input`. The `target_type` tensor itself is used only for its dtype and is not read elementwise.

See the [ONNX `CastLike` spec](https://onnx.ai/onnx/operators/onnx__CastLike.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `input` | `T1` | — | — | Input tensor whose elements are to be cast. | required |
| `target` | `target_type` | `T2` | — | — | Tensor whose element type defines the destination dtype; its values are not used. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `output` | `T2` | same as `x` | same as `x` | Output tensor with the same shape as `input` and the element type of `target_type`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `round_mode` | `"up"` | Rounding direction used only when casting to float8e8m0. The implemented non-float8 subset accepts the ONNX default `"up"`. |
| `saturate` | `1` | Whether casts to float8 saturate at the finite range. The implemented non-float8 subset accepts the ONNX default `1`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T1` | `float32`, `float16`, `uint32`, `int32`, `uint8`, `int8`, `bool` |
| `T2` | `float32`, `float16`, `uint32`, `int32`, `uint8`, `int8`, `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`cast-scalar-x4.wgsl.jinja`](build/webgpu/cast-scalar-x4.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.CastLike", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [] }, target: { data: targetData, shape: [3] } });
```
