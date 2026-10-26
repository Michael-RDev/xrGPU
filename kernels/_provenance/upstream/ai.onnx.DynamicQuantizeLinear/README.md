---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.DynamicQuantizeLinear

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 11

## Description

Computes a per-tensor scale and zero point from the range of floating-point input `x`, extending the range to include zero, then quantizes each value to `uint8` as `saturate(round(x / y_scale) + y_zero_point)`. Uses round-to-nearest-even and clamps results to `[0, 255]`.

See the [ONNX `DynamicQuantizeLinear` spec](https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `x` | `T` | — | — | Float32 input tensor to quantize. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `y` | `TQ` | same as `x` | same as `x` | Quantized output tensor; same shape as the input. | required |
| `y_scale` | `T` | `0` | `[]` | Per-tensor scale factor derived from the input min/max range; scalar. | required |
| `y_zero_point` | `TQ` | `0` | `[]` | Per-tensor zero point for the quantization; scalar. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32` |
| `TQ` | `uint8` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `single_invocation` — Uses one invocation to find the range and quantize the tensor, avoiding partial buffers for small inputs. It also provides the fallback when the parallel reduction cannot satisfy device limits.
- `parallel_subgroup_reduce_vec4` — Reduces independent input blocks to min/max partials, combines them, and quantizes in a separate pass. The family uses packed reads when the input length is vec4-aligned.
- `parallel_subgroup_reduce` — Reduces independent input blocks to min/max partials, combines them, and quantizes in a separate pass. The family uses packed reads when the input length is vec4-aligned.
- `grid_stride_reduce_vec4` — Caps the number of min/max partials and grid-strides each workgroup across the input. This bounds scratch size and finalization work for large tensors.
- `grid_stride_reduce` — Caps the number of min/max partials and grid-strides each workgroup across the input. This bounds scratch size and finalization work for large tensors.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`dynamic-quantize-linear-quantize.wgsl.jinja`](build/webgpu/dynamic-quantize-linear-quantize.wgsl.jinja)
- [`dynamic-quantize-linear-reduce.wgsl.jinja`](build/webgpu/dynamic-quantize-linear-reduce.wgsl.jinja)
- [`dynamic-quantize-linear.wgsl.jinja`](build/webgpu/dynamic-quantize-linear.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.DynamicQuantizeLinear", { version: 1 });
const { y, y_scale, y_zero_point } = await kernel({ x: { data: xData, shape: [1] } });
```
