---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.ReduceSumSquare

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 18

## Description

Computes the sum of squared elements of the input tensor along the specified axes. The output rank matches the input if `keepdims` is 1; otherwise the reduced dimensions are pruned. Reduction over an empty set of values yields 0.

See the [ONNX `ReduceSumSquare` spec](https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `data` | `T` | — | — | The input tensor to reduce. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `reduced` | `T` | derived | — | The reduced output tensor containing the sum of squares. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axes` | `[]` | Values of the optional ONNX `axes` tensor input, supplied through this request attribute; an empty list follows `noop_with_empty_axes`. |
| `keepdims` | `1` | If 1 (default in spec), retain the reduced dimensions with size 1; if 0, remove them. |
| `noop_with_empty_axes` | `0` | If 1 and axes is empty, acts as a no-op that squares each element without reducing; if 0 (default), reduces over all axes when axes is empty. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `axis0_splitk_i32` — Partitions a long rank-two axis-zero integer reduction across workgroups and combines exact int32 partials. It applies when the reduced row dimension is too large for one pass to expose enough parallelism.
- `subgroup_last_axis_vec4` — Reduces each contiguous last-axis row with subgroup collectives and vec4-packed reads.
- `subgroup_last_axis` — Reduces each contiguous last-axis row with subgroup collectives and scalar reads for an unaligned row width.
- `strided_axis_serial` — Flatten a single non-last reduction axis into outer/axis/inner geometry. Compile its strides and loop bound, keep one output per lane and float32 accumulation, and cap the workgroup by the device limits.

## Device requirements

Some implementation variants require `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`reduce-axis-split-reduce.wgsl.jinja`](build/webgpu/reduce-axis-split-reduce.wgsl.jinja)
- [`reduce-axis0-splitk-combine.wgsl.jinja`](build/webgpu/reduce-axis0-splitk-combine.wgsl.jinja)
- [`reduce-axis0-splitk-reduce.wgsl.jinja`](build/webgpu/reduce-axis0-splitk-reduce.wgsl.jinja)
- [`reduce-axis0-tilecols.wgsl.jinja`](build/webgpu/reduce-axis0-tilecols.wgsl.jinja)
- [`reduce-flat-partial.wgsl.jinja`](build/webgpu/reduce-flat-partial.wgsl.jinja)
- [`reduce-multi-axis-coop.wgsl.jinja`](build/webgpu/reduce-multi-axis-coop.wgsl.jinja)
- [`reduce-noop-empty-axes.wgsl.jinja`](build/webgpu/reduce-noop-empty-axes.wgsl.jinja)
- [`reduce-row-subgroup-rows.wgsl.jinja`](build/webgpu/reduce-row-subgroup-rows.wgsl.jinja)
- [`reduce-row-subgroup.wgsl.jinja`](build/webgpu/reduce-row-subgroup.wgsl.jinja)
- [`reduce-row-tree.wgsl.jinja`](build/webgpu/reduce-row-tree.wgsl.jinja)
- [`reduce-serial-axis.wgsl.jinja`](build/webgpu/reduce-serial-axis.wgsl.jinja)
- [`reduce-strided-axis.wgsl.jinja`](build/webgpu/reduce-strided-axis.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.ReduceSumSquare", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({ x: { data: xData, shape: [3, 2, 2] } }, {
  outputs: { y: { shape: [1, 1, 1], dtype: "float32" } },
});
```
