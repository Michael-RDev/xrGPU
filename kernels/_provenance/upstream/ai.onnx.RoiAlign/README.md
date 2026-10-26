---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.RoiAlign

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 22

## Description

Region of Interest (RoI) align pooling as described in Mask R-CNN: applies `avg` or `max` pooling over each RoI in the input feature map, producing a fixed-size `(num_rois, C, output_height, output_width)` output. Uses bilinear interpolation to avoid quantization misalignment between image space and feature-map space.

See the [ONNX `RoiAlign` spec](https://onnx.ai/onnx/operators/onnx__RoiAlign.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | same as logical dtype | `4` | — | 4-D input feature map of shape `(N, C, H, W)`. | required |
| `rois` | — | `T` | same as logical dtype | `2` | — | Regions of interest; 2-D tensor of shape `(num_rois, 4)` with `[x1, y1, x2, y2]` coordinates in input-image space. | required |
| `batch_indices` | — | `I` | `uint32` | `1` | — | Logical int64 1-D tensor of shape `(num_rois,)` mapping each RoI to its non-negative image index in the batch; WebGPU stores it as uint32. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | `4` | `[rois[0], x[1], output_height, output_width]` | Pooled output of shape `(num_rois, C, output_height, output_width)`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `coordinate_transformation_mode` | `"half_pixel"` | `"half_pixel"` shifts input coordinates by −0.5 for better alignment (recommended); `"output_half_pixel"` omits the shift for backward compatibility. |
| `mode` | `"avg"` | Pooling method applied within each RoI bin: `"avg"` (default) or `"max"`. |
| `output_height` | `1` | Pooled output `Y`'s height; default `1`. |
| `output_width` | `1` | Pooled output `Y`'s width; default `1`. |
| `sampling_ratio` | `0` | Number of sampling points per bin side for bilinear interpolation; `0` uses an adaptive count (`ceil(roi_size / output_size)`). |
| `spatial_scale` | `1` | Multiplicative factor to translate ROI coordinates from input-image scale to feature-map scale; default `1.0`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `I` | `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`roi-align-cooperative.wgsl.jinja`](build/webgpu/roi-align-cooperative.wgsl.jinja)
- [`roi-align-split-combine.wgsl.jinja`](build/webgpu/roi-align-split-combine.wgsl.jinja)
- [`roi-align.wgsl.jinja`](build/webgpu/roi-align.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.RoiAlign", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 1, 4, 4] },
  rois: { data: roisData, shape: [1, 4] },
  batch_indices: { data: batch_indicesData, shape: [1] },
});
```
