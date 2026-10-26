---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Resize

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 19

## Description

Resizes the input tensor by sampling neighboring input values. Output dimensions are determined by per-axis scale factors or explicit target sizes. Supports `nearest`, `linear`, and `cubic` interpolation with configurable coordinate transformation modes.

See the [ONNX `Resize` spec](https://onnx.ai/onnx/operators/onnx__Resize.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | N-D input tensor to be resized. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | same as `x` | — | N-D output tensor after resizing. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `antialias` | `0` | When set to 1, stretches the resampling filter during downscaling so that more input pixels contribute to each output pixel, reducing aliasing. |
| `axes` | — | Optional axes to resize. When omitted, the scale values or requested output size apply to every input axis. |
| `coordinate_transformation_mode` | `"half_pixel"` | How to map a coordinate in the output tensor back to the input tensor; supported modes include `"half_pixel"`, `"pytorch_half_pixel"`, `"align_corners"`, `"asymmetric"`, and `"tf_crop_and_resize"`. |
| `cubic_coeff_a` | `-0.75` | Coefficient `a` in the cubic interpolation filter; the default is -0.75. Valid only when `mode` is `"cubic"`. |
| `exclude_outside` | `0` | When set to 1, assigns zero weight to sampling locations outside the input tensor and renormalizes the remaining weights to sum to 1. |
| `extrapolation_value` | `0` | Fill value used for out-of-bounds samples when `coordinate_transformation_mode` is `"tf_crop_and_resize"`. |
| `keep_aspect_ratio_policy` | `"stretch"` | How explicit sizes preserve aspect ratio. This package supports only the default `"stretch"` policy; `"not_larger"` and `"not_smaller"` are unsupported. |
| `mode` | `"nearest"` | Interpolation mode: `"nearest"` (default), `"linear"` (bilinear/N-linear), or `"cubic"` (bicubic/N-cubic). |
| `nearest_mode` | `"round_prefer_floor"` | Rounding strategy used when `mode` is `"nearest"`: `"round_prefer_floor"` (default), `"round_prefer_ceil"`, `"floor"`, or `"ceil"`. |
| `roi` | `[]` | Values of the optional `roi` tensor, supplied as `[starts..., ends...]` with one pair per input axis or per `axes` entry; used only by `"tf_crop_and_resize"`. |
| `scales` | `[]` | Values of the optional `scales` tensor. Supply one positive value per input axis, or one per `axes` entry when that attribute is present; omit it when the output shape represents the exact `sizes` input. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint8`, `int8` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `generic_rank_n_suffix_vec4` — Reuses interpolation coordinates and corner weights for four contiguous floating-point values on an aligned, unchanged final axis. Explicit scale and ROI must preserve that axis; linear and cubic modes retain f32 per-lane accumulation in the scalar tap order.
- `generic_rank_n_cubic_suffix_vec4` — Reuses interpolation coordinates and corner weights for four contiguous floating-point values on an aligned, unchanged final axis. Explicit scale and ROI must preserve that axis; linear and cubic modes retain f32 per-lane accumulation in the scalar tap order.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`resize-antialias.wgsl.jinja`](build/webgpu/resize-antialias.wgsl.jinja)
- [`resize-coord-transform-5d.wgsl.jinja`](build/webgpu/resize-coord-transform-5d.wgsl.jinja)
- [`resize-coord-transform.wgsl.jinja`](build/webgpu/resize-coord-transform.wgsl.jinja)
- [`resize-cubic.wgsl.jinja`](build/webgpu/resize-cubic.wgsl.jinja)
- [`resize-generic.wgsl.jinja`](build/webgpu/resize-generic.wgsl.jinja)
- [`resize-linear-2x-stencil-x8.wgsl.jinja`](build/webgpu/resize-linear-2x-stencil-x8.wgsl.jinja)
- [`resize-linear-2x-stencil.wgsl.jinja`](build/webgpu/resize-linear-2x-stencil.wgsl.jinja)
- [`resize-nearest-integer-scale.wgsl.jinja`](build/webgpu/resize-nearest-integer-scale.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Resize", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({ x: { data: xData, shape: [1, 1, 2, 4] } }, {
  outputs: { y: { shape: [1, 1, 1, 2], dtype: "float32" } },
});
```
