---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Col2Im

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 18

## Description

Rearranges column blocks back into a batched multidimensional image. Takes a 3-D input of shape `[N, C * product(block_shape), L]` (where `L` is the number of blocks) and accumulates overlapping block contributions into the output image using the specified `block_shape`, `strides`, `pads`, and `dilations`.

See the [ONNX `Col2Im` spec](https://onnx.ai/onnx/operators/onnx__Col2Im.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `input` | `T` | same as logical dtype | `3` | — | Column-block data tensor of shape `[N, C * product(block_shape), L]` to be folded back into an image. | required |
| `image_shape` | `I` | `uint32` | `1` | — | Logical int64 metadata tensor specifying the output spatial dimensions (for example, `[H, W]` for 2-D); non-negative values use uint32 WebGPU storage. | required |
| `block_shape` | `I` | `uint32` | `1` | — | Logical int64 metadata tensor specifying the positive block size on each spatial axis (for example, `[H_block, W_block]` for 2-D); values use uint32 WebGPU storage. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | derived | — | Output image tensor produced by accumulating rearranged column blocks. Its rank is two greater than the length of `image_shape`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `dilations` | — | Dilation factors for each spatial axis; defaults to one on every axis. |
| `pads` | — | Padding at the beginning of every spatial axis followed by padding at the end of every spatial axis; defaults to zeros. |
| `strides` | — | Stride factors for each spatial axis; defaults to one on every axis. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32` |
| `I` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `identity_copy_vec4` — A 1x1 block with unit strides and dilations and no padding folds every column straight onto its pixel, so the column tensor and the image are the same bytes: one flat vec4 copy, elided outright when the output is a view of the input. The block product is read off the shapes (column channels equal image channels only when every block extent is one).
- `identity_copy` — Copies 1x1 blocks with scalar loads and stores when the element count is not a multiple of four.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`col2im-nd.wgsl.jinja`](build/webgpu/col2im-nd.wgsl.jinja)
- [`datamove-elementwise-copy.wgsl.jinja`](build/webgpu/datamove-elementwise-copy.wgsl.jinja)
- [`datamove-flat-copy.wgsl.jinja`](build/webgpu/datamove-flat-copy.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `output`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Col2Im", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({
  input: { data: inputData, shape: [1, 9, 1] },
  image_shape: { data: image_shapeData, shape: [2] },
  block_shape: { data: block_shapeData, shape: [2] },
}, {
  outputs: { output: { shape: [1, 1, 3, 3], dtype: "float32" } },
});
```
