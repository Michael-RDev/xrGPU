---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.ScatterElements

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 18

## Description

Produces a copy of `data` with values updated at positions given by `indices` along the specified `axis`. For each entry in `updates`, the axis coordinate comes from `indices` while all other coordinates come from the entry's own position in `updates`. An optional `reduction` (`add`, `mul`, `max`, `min`) combines updates with existing values instead of overwriting; with `none`, duplicate indices are not allowed.

See the [ONNX `ScatterElements` spec](https://onnx.ai/onnx/operators/onnx__ScatterElements.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `data` | `T` | — | — | Input tensor of rank r >= 1 that is copied to form the output base. | required |
| `indices` | `I` | — | — | Integer index tensor of the same rank as `data`; each value selects a position along `axis`. | required |
| `updates` | `T` | — | — | Values to scatter, same rank and shape as `indices`. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `data` | same as `data` | Copy of `data` with scattered updates applied; same shape as `data`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `0` | Which axis to scatter on; negative values count from the back. Accepted range is `[-r, r-1]` where `r = rank(data)`. |
| `reduction` | `"none"` | Reduction to apply when writing updates: `none` (overwrite, no duplicate indices), `add`, `mul`, `max`, or `min`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32`, `int8`, `uint8`, `int16`, `bool`, `int64` |
| `I` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `reduction_f32_add_axis0_histogram` — Fold the updates axis into workgroup bins per 32-column tile, then add each touched bin into the copied output with one device atomic. Striping the updates axis widens the grid a column-only dispatch caps at ceil(columns / 32) workgroups.
- `reduction_f16_slab` — Stages f16 slabs in workgroup memory as f32 bits and reduces repeated indices there with a compare-exchange loop. Requires a fixed 32-wide subgroup: on adapters reporting any other subgroup range the duplicate-index merge can lose updates, so the variant is not used there.

## Device requirements

Some implementation variants require `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`scatter-elements-f32-add-axis0-histogram.wgsl.jinja`](build/webgpu/scatter-elements-f32-add-axis0-histogram.wgsl.jinja)
- [`scatter-elements-reduction-atomic.wgsl.jinja`](build/webgpu/scatter-elements-reduction-atomic.wgsl.jinja)
- [`scatter-elements-reduction-slab.wgsl.jinja`](build/webgpu/scatter-elements-reduction-slab.wgsl.jinja)
- [`scatter-elements-reduction.wgsl.jinja`](build/webgpu/scatter-elements-reduction.wgsl.jinja)
- [`scatter-elements.wgsl.jinja`](build/webgpu/scatter-elements.wgsl.jinja)
- [`scatter-f16-f32-convert.wgsl.jinja`](build/webgpu/scatter-f16-f32-convert.wgsl.jinja)
- [`scatter-flat-copy.wgsl.jinja`](build/webgpu/scatter-flat-copy.wgsl.jinja)
- [`scatter-narrow-wrap.wgsl.jinja`](build/webgpu/scatter-narrow-wrap.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.ScatterElements", { version: 1 });
const { output } = await kernel({
  data: { data: dataData, shape: [3] },
  indices: { data: indicesData, shape: [2] },
  updates: { data: updatesData, shape: [2] },
});
```
