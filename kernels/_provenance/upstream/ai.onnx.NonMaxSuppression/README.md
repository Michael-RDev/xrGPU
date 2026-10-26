---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.NonMaxSuppression

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 11

## Description

Filters boxes whose intersection over union (IoU) with a higher-scoring selected box exceeds `iou_threshold`. Processing is independent per batch and class, and returns logical int64 `[batch_index, class_index, box_index]` triples with lossless uint32 WebGPU storage. The int64 selection limit uses a semantics-preserving saturating projection to uint32. Because the result length is data-dependent, its exact shape must be supplied. This package supports float32 boxes, scores, and thresholds.

See the [ONNX `NonMaxSuppression` spec](https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `boxes` | `T` | same as logical dtype | `3` | — | Bounding box coordinates with shape `[num_batches, spatial_dimension, 4]`; box format is controlled by `center_point_box`. | required |
| `scores` | `T` | same as logical dtype | `3` | — | Per-class confidence scores with shape `[num_batches, num_classes, spatial_dimension]`. | required |
| `max_output_boxes_per_class` | `M` | `uint32` | `0` | — | Optional logical int64 scalar limiting boxes selected per batch and class. When omitted, the ONNX default is zero and the output is empty. WebGPU maps non-positive values to zero and values above uint32 range to `UINT32_MAX`; this preserves results because no group can select more boxes than its finite input. | optional |
| `iou_threshold` | `T` | same as logical dtype | `0` | — | Optional scalar IoU threshold in `[0, 1]`; boxes whose IoU is strictly greater are suppressed. Defaults to zero. | optional |
| `score_threshold` | `T` | same as logical dtype | `0` | — | Optional scalar score threshold. When present, only boxes whose score is strictly greater than the threshold are considered; when omitted, scores are not filtered. The ordered comparison excludes NaN scores; a NaN threshold admits no scores. | optional |

## Outputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `selected_indices` | `I` | `uint32` | `2` | — | Logical int64 selected-box indices with shape `[num_selected_indices, 3]`, each row containing `[batch_index, class_index, box_index]`. WebGPU stores the bounded indices as uint32. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `center_point_box` | `0` | Box coordinate format: 0 for corner format `[y1, x1, y2, x2]`, 1 for center format `[x_center, y_center, width, height]`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32` |
| `M` | `int64` |
| `I` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `rank_window` — Sort scores once, then construct and greedily resolve bounded workgroup rank windows until the runtime cap or candidate exhaustion. Demote when the average selected count is below the sort depth or the greedy candidate visits would not cover one workgroup pair window.
- `bitmask_low_occupancy` — Low group counts use score sorting and all-pairs IoU bitmasks. Windowed replay preserves greedy order, folds masks only into unvisited words, and stops at the selection limit. Signed-zero keys preserve lower-index ties; compaction writes batch/class-major `selected_indices`.
- `groups_parallel` — Workgroup-parallel greedy selection keeps sticky suppression flags in global scratch, followed by stable batch/class compaction.
- `groups_private` — Each invocation owns a bounded bitset for its strided candidates. Scores below the threshold are discarded once before greedy selection, avoiding their repeated IoU work and global suppression traffic.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`nms-bitmask.wgsl.jinja`](build/webgpu/nms-bitmask.wgsl.jinja)
- [`nms-groups-compact.wgsl.jinja`](build/webgpu/nms-groups-compact.wgsl.jinja)
- [`nms-groups-select.wgsl.jinja`](build/webgpu/nms-groups-select.wgsl.jinja)
- [`nms-shared-sort.wgsl.jinja`](build/webgpu/nms-shared-sort.wgsl.jinja)
- [`nms-window.wgsl.jinja`](build/webgpu/nms-window.wgsl.jinja)
- [`non-max-suppression.wgsl.jinja`](build/webgpu/non-max-suppression.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `selected_indices`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.NonMaxSuppression", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { selected_indices } = await kernel({
  boxes: { data: boxesData, shape: [1, 1, 4] },
  scores: { data: scoresData, shape: [1, 1, 1] },
  max_output_boxes_per_class: { data: max_output_boxes_per_classData, shape: [] },
  iou_threshold: { data: iou_thresholdData, shape: [] },
  score_threshold: { data: score_thresholdData, shape: [] },
}, {
  outputs: { selected_indices: { shape: [1, 3], dtype: "int64" } },
});
```
