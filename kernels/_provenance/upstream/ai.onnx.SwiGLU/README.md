---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.SwiGLU

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 28

## Description

Applies the Swish activation to gate input `A` and multiplies it elementwise by linear input `B`: `Y = (A * sigmoid(alpha * A)) * B`. Inputs must have identical shapes; broadcasting is not part of this operator. This package supports float16 and float32; standard bfloat16 and double tensors are unsupported.

See the [ONNX `SwiGLU` spec](https://onnx.ai/onnx/operators/onnx__SwiGLU.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `T` | — | — | Gate input transformed by the Swish activation. | required |
| `b` | `B` | `T` | — | — | Linear value input multiplied by the activated gate; its shape must exactly match `A`. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `a` | same as `a` | Elementwise gated product with the same shape and dtype as both inputs. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `alpha` | `1` | Coefficient applied to `A` inside the sigmoid; defaults to `1.0`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`swiglu-elementwise.wgsl.jinja`](build/webgpu/swiglu-elementwise.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.SwiGLU", { version: 1 });
const { y } = await kernel({ a: { data: aData, shape: [2, 4] }, b: { data: bData, shape: [2, 4] } });
```
