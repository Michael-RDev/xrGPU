---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.SpaceToDepth

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Rearranges blocks of spatial data into depth by moving values from the height and width dimensions into the channel dimension. An NCHW input of shape `[N, C, H, W]` produces an output of shape `[N, C * blocksize * blocksize, H / blocksize, W / blocksize]`.

See the [ONNX `SpaceToDepth` spec](https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `input` | `T` | `4` | — | 4-D input tensor of shape `[N, C, H, W]`. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | `4` | derived | 4-D output tensor of shape `[N, C * blocksize * blocksize, H / blocksize, W / blocksize]`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `blocksize` | — | Size of the spatial block to collapse into depth; each `blocksize x blocksize` patch of pixels becomes additional channels. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `int16`, `int8`, `uint32`, `uint8`, `bool` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `flat_vector` — Groups four flat outputs per invocation, including groups that cross spatial rows or channels. Reuses the shared permutation coordinates and writes one contiguous vector.
- `collapsed_input_vector` — When one spatial block spans the input plane, reads contiguous vec4 words and writes channel streams. Spatial tiles balance the available channel workgroups; paired f16 channels give each invocation whole output words.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`space-depth-collapsed-input.wgsl.jinja`](build/webgpu/space-depth-collapsed-input.wgsl.jinja)
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

const kernel = await getKernel("webgpu-kernels/ai.onnx.SpaceToDepth", { version: 1 });
const { output } = await kernel({ input: { data: inputData, shape: [1, 1, 2, 4] } }, {
  attrs: { blocksize: 2 },
});
```
