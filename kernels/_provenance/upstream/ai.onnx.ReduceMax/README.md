---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.ReduceMax

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 20

## Description

Computes the maximum of input tensor elements along the specified axes. The output rank matches the input when `keepdims` is 1; reduced dimensions are pruned when `keepdims` is 0. Reduction over an empty set yields negative infinity when the dtype supports it, or the dtype's minimum value otherwise. For Boolean inputs, `false` is less than `true`.

See the [ONNX `ReduceMax` spec](https://onnx.ai/onnx/operators/onnx__ReduceMax.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `data` | `T` | — | — | The input tensor to reduce. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `reduced` | `T` | derived | — | The reduced output tensor containing the maximum values. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axes` | `[]` | Values of the optional ONNX `axes` tensor input, supplied through this request attribute; an empty list follows `noop_with_empty_axes`. |
| `keepdims` | `1` | Whether to retain reduced dimensions in the output with size 1 (1) or prune them (0). |
| `noop_with_empty_axes` | `0` | When 1 and `axes` is empty, the op acts as an identity (no-op); when 0 (default), reduction happens over all axes. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32`, `int8`, `uint8`, `bool` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `subgroup_last_axis_vec4` — Reduces each contiguous last-axis row with subgroup collectives and vec4-packed reads.
- `subgroup_last_axis` — Reduces each contiguous last-axis row with subgroup collectives and scalar reads for an unaligned row width.

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
- [`reduce-narrow-empty-identity.wgsl.jinja`](build/webgpu/reduce-narrow-empty-identity.wgsl.jinja)
- [`reduce-noop-empty-axes.wgsl.jinja`](build/webgpu/reduce-noop-empty-axes.wgsl.jinja)
- [`reduce-row-subgroup-rows.wgsl.jinja`](build/webgpu/reduce-row-subgroup-rows.wgsl.jinja)
- [`reduce-row-subgroup.wgsl.jinja`](build/webgpu/reduce-row-subgroup.wgsl.jinja)
- [`reduce-row-tree.wgsl.jinja`](build/webgpu/reduce-row-tree.wgsl.jinja)
- [`reduce-serial-axis.wgsl.jinja`](build/webgpu/reduce-serial-axis.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.ReduceMax", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({ x: { data: xData, shape: [3, 2, 2] } }, {
  outputs: { y: { shape: [1, 1, 1], dtype: "float32" } },
});
```
