---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.GridSample

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 20

## Description

Samples values from input tensor `X` at positions defined by a flow-field `grid`, producing output `Y` with spatial dimensions taken from `grid`. Grid coordinates are normalized to `[-1, 1]` over the input spatial extent; positions outside this range are handled according to `padding_mode`. Supports spatial `(rank-4, NCHW)` and volumetric `(rank-5, NCDHW)` inputs with `linear`, `nearest`, or `cubic` interpolation.

See the [ONNX `GridSample` spec](https://onnx.ai/onnx/operators/onnx__GridSample.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | Input tensor of shape `(N, C, D1, ..., Dr)` whose values are sampled. | required |
| `grid` | — | `T` | — | — | Flow-field of shape `(N, D1_out, ..., Dr_out, r)` with normalized sampling coordinates in `[-1, 1]`. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `grid` | derived | Output tensor of shape `(N, C, D1_out, ..., Dr_out)` containing the interpolated samples. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `align_corners` | `0` | When 1, extrema values `-1` and `1` map to the center of the corner pixels; when 0 (default) they map to the outer edge of corner pixels, making sampling resolution-agnostic. |
| `mode` | `"linear"` | Interpolation method: `linear` (bilinear or trilinear, depending on rank), `nearest`, or `cubic` (bicubic for rank-4 inputs and tricubic for rank-5 inputs). |
| `padding_mode` | `"zeros"` | How out-of-bound grid positions are handled: `zeros` pads with 0, `border` clamps to the border value, or `reflection` reflects coordinates back into the valid range. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `nchw_rank4_channel_vector` — Resolves the sampling coordinate once per pixel and reuses it across four channels, so one thread gathers from four channel planes instead of reading one coalesced. This amortizes the coordinate math across channels, trading a coalesced read for a scattered gather; adapters where scattered gathers are slow use the coalesced variant instead.
- `nchw_rank4` — Portable scalar spatial sampling: one thread per output element, so reads stay coalesced within one channel plane. The route for every adapter whose gather throughput the channel vector would spend.
- `ncdhw_rank5_channel_vector` — Shares volumetric coordinates, padding and interpolation across a vector of channels; uses two lanes for two channels and four lanes otherwise, with masked tails and device-capped workgroups.
- `ncdhw_rank5` — Portable scalar volumetric sampling for all interpolation and padding modes.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`grid-sample.wgsl.jinja`](build/webgpu/grid-sample.wgsl.jinja)
- [`grid-sample3d.wgsl.jinja`](build/webgpu/grid-sample3d.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.GridSample", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 1, 2, 2] },
  grid: { data: gridData, shape: [1, 1, 3, 2] },
});
```
