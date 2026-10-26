---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.TopK

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 11

## Description

Retrieves the top-`k` largest or smallest elements along the selected axis, returning values and stable lower-index tie-breaking indices.

See the [ONNX `TopK` spec](https://onnx.ai/onnx/operators/onnx__TopK.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Values from which the top `k` entries are selected along `axis`. | required |

## Outputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `values` | `Values` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | same as `x` | derived | Selected values; the reduced axis has length `k`. | required |
| `indices` | `Indices` | `I` | `uint32` | same as `x` | derived | Logical int64 indices of the selected values along the reduced axis; WebGPU stores these bounded indices as uint32. | required |

## Runtime arguments

| Name | Kind | Description | Presence |
| --- | --- | --- | --- |
| `k` | `u32` | Number of values to select along the configured axis: ONNX's one-element `K` input, passed as a scalar. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `-1` | Axis to reduce; negative values count from the back. |
| `largest` | `1` | Select largest values when 1, smallest values when 0. |
| `sorted` | `1` | Sort selected values when 1. A sorted result is also valid when output order is unspecified (`sorted=0`). |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int8`, `int16`, `int32`, `uint8`, `uint32` |
| `I` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `axis_smallk_tournament` — Scans a strided non-last axis with one workgroup per output position and retains only a short candidate list instead of sorting the whole axis. It is selected when many independent outputs amortize the strided scan, and it is the bounded-storage route when the axis does not fit shared memory.
- `last_axis_large_top1` — Finds one winner on rows too wide for the direct shared-memory route by reducing blocks to scratch candidates. A second pass selects the final candidate.
- `axis_bitonic` — Shared bitonic selection for arbitrary axes; floating inputs encode order keys once and gather original values after sorting, preserving stable ties and value bits.
- `subgroup_rows_smallk` — Retains only K candidates per row and selects a row-parallel reduction layout according to row width and available subgroup geometry. It avoids shared storage proportional to the full axis.
- `subgroup_min_rows_smallk` — Retains only K candidates per row and selects a row-parallel reduction layout according to row width and available subgroup geometry. It avoids shared storage proportional to the full axis.
- `portable_rows_smallk` — Retains only K candidates per row and selects a row-parallel reduction layout according to row width and available subgroup geometry. It avoids shared storage proportional to the full axis.
- `small_rows_batched` — Retains only K candidates per row and selects a row-parallel reduction layout according to row width and available subgroup geometry. It avoids shared storage proportional to the full axis.
- `last_axis_large_one_merge` — Staged stable TopK with bounded candidate buffers. Floating inputs encode order keys on load and gather selected source values after the final merge; integer inputs retain exact integer keys.
- `last_axis_large_two_merge` — Staged stable TopK with bounded candidate buffers. Floating inputs encode order keys on load and gather selected source values after the final merge; integer inputs retain exact integer keys.
- `last_axis_large_one_merge_int` — Staged stable TopK with bounded candidate buffers. Floating inputs encode order keys on load and gather selected source values after the final merge; integer inputs retain exact integer keys.
- `last_axis_large_two_merge_int` — Staged stable TopK with bounded candidate buffers. Floating inputs encode order keys on load and gather selected source values after the final merge; integer inputs retain exact integer keys.
- `axis_large_one_merge` — Staged stable TopK with bounded candidate buffers. Floating inputs encode order keys on load and gather selected source values after the final merge; integer inputs retain exact integer keys.
- `axis_large_two_merge` — Staged stable TopK with bounded candidate buffers. Floating inputs encode order keys on load and gather selected source values after the final merge; integer inputs retain exact integer keys.
- `last_axis_large_three_merge` — Staged stable TopK with bounded candidate buffers. Floating inputs encode order keys on load and gather selected source values after the final merge; integer inputs retain exact integer keys.
- `last_axis_large_three_merge_int` — Staged stable TopK with bounded candidate buffers. Floating inputs encode order keys on load and gather selected source values after the final merge; integer inputs retain exact integer keys.

## Device requirements

Some implementation variants require `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`topk-axis.wgsl.jinja`](build/webgpu/topk-axis.wgsl.jinja)
- [`topk-large-block.wgsl.jinja`](build/webgpu/topk-large-block.wgsl.jinja)
- [`topk-noop.wgsl.jinja`](build/webgpu/topk-noop.wgsl.jinja)
- [`topk-portable-rows-smallk.wgsl.jinja`](build/webgpu/topk-portable-rows-smallk.wgsl.jinja)
- [`topk-small-rows-batched.wgsl.jinja`](build/webgpu/topk-small-rows-batched.wgsl.jinja)
- [`topk-strided-smallk.wgsl.jinja`](build/webgpu/topk-strided-smallk.wgsl.jinja)
- [`topk-subgroup-rows.wgsl.jinja`](build/webgpu/topk-subgroup-rows.wgsl.jinja)
- [`topk-top1-last-axis.wgsl.jinja`](build/webgpu/topk-top1-last-axis.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.TopK", { version: 1 });
const { values, indices } = await kernel({ x: { data: xData, shape: [1, 3] }, k: 1 });
```
