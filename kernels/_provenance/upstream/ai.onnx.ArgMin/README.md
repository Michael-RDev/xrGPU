---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.ArgMin

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Returns the index of the minimum value along an axis, choosing the first equal value unless `select_last_index` is enabled.

See the [ONNX `ArgMin` spec](https://onnx.ai/onnx/operators/onnx__ArgMin.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `data` | `T` | — | — | Values whose minimum index is selected along `axis`. | required |

## Outputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `y` | `reduced` | `I` | `uint32` | derived | derived | Logical int64 indices of the minimum values along the reduced axis; WebGPU stores these bounded indices as uint32. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `0` | Axis to reduce; negative values count from the back. |
| `keepdims` | `1` | Retain the reduced dimension with length one when non-zero. |
| `select_last_index` | `0` | Choose the last equal minimum instead of the first when non-zero. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32`, `int16`, `int8`, `uint8` |
| `I` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `axis_split_packed` — Scans four adjacent float32 inner-axis columns per invocation while preserving each column’s axis order and the scalar partial/combine layout. Small output grids retain the scalar route.

## Device requirements

Some implementation variants require `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`reduce-arg-axis-split-combine.wgsl.jinja`](build/webgpu/reduce-arg-axis-split-combine.wgsl.jinja)
- [`reduce-arg-axis-split-reduce.wgsl.jinja`](build/webgpu/reduce-arg-axis-split-reduce.wgsl.jinja)
- [`reduce-arg-axis-split-tiled.wgsl.jinja`](build/webgpu/reduce-arg-axis-split-tiled.wgsl.jinja)
- [`reduce-arg-axis-tiled.wgsl.jinja`](build/webgpu/reduce-arg-axis-tiled.wgsl.jinja)
- [`reduce-arg-axis.wgsl.jinja`](build/webgpu/reduce-arg-axis.wgsl.jinja)
- [`reduce-arg-row-split.wgsl.jinja`](build/webgpu/reduce-arg-row-split.wgsl.jinja)
- [`reduce-arg-row-subgroup.wgsl.jinja`](build/webgpu/reduce-arg-row-subgroup.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.ArgMin", { version: 1 });
const { y } = await kernel({ x: { data: xData, shape: [2, 2] } });
```
