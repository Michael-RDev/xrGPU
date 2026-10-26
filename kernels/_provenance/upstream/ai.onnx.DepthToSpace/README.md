---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.DepthToSpace

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Rearranges data from the depth dimension into spatial blocks, expanding height and width by `blocksize` while reducing channels by `blocksize * blocksize`. The inverse of SpaceToDepth; supports `DCR` (depth-column-row) and `CRD` (column-row-depth) element orderings.

See the [ONNX `DepthToSpace` spec](https://onnx.ai/onnx/operators/onnx__DepthToSpace.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `input` | `T` | `4` | — | 4-D input tensor of shape `[N, C, H, W]`. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | `4` | derived | 4-D output tensor of shape `[N, C/(blocksize*blocksize), H*blocksize, W*blocksize]`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `blocksize` | — | Side length of the spatial blocks; input channels are divided by `blocksize * blocksize`. |
| `mode` | `"DCR"` | Element ordering within each block: `DCR`, the default depth-column-row order, or `CRD`, the column-row-depth order. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `int16`, `int8`, `uint32`, `uint8`, `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`depth-to-space-nchw-coarsened4.wgsl.jinja`](build/webgpu/depth-to-space-nchw-coarsened4.wgsl.jinja)
- [`space-depth-permute.wgsl.jinja`](build/webgpu/space-depth-permute.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.DepthToSpace", { version: 1 });
const { output } = await kernel({ input: { data: inputData, shape: [1, 8, 1, 1] } }, {
  attrs: { blocksize: 2 },
});
```
