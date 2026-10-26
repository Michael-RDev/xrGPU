---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.MaxUnpool

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 22

## Description

Computes the partial inverse of MaxPool: each pooled value in `X` is scattered back to the position given by its index in `I`, with all other positions set to zero. The optional `output_shape` input disambiguates the output size when multiple input sizes would produce the same pooled result.

See the [ONNX `MaxUnpool` spec](https://onnx.ai/onnx/operators/onnx__MaxUnpool.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | same as logical dtype | — | — | Pooled input tensor to be unpooled, typically the first output of a MaxPool op, with shape `(N x C x D1 x ... x Dn)`. | required |
| `indices` | `I` | `I` | `uint32` | — | — | Logical int64 flat linear indices of the maximal elements corresponding to X, typically the second output of a MaxPool op; same shape as X and stored as uint32 by WebGPU. | required |
| `output_shape` | — | `I` | `uint32` | `1` | — | Optional logical int64 1-D tensor specifying the full non-negative output shape, for example `(N, C, H, W)`. It uses uint32 WebGPU storage and, when provided, causes `pads` to be ignored. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `x` | — | Unpooled output tensor with pooled values scattered to their original positions and zeros elsewhere. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `kernel_shape` | — | The size of the pooling kernel along each spatial axis; required and must match the kernel used in the corresponding MaxPool. |
| `pads` | — | Optional padding at the beginning and end of each spatial axis in `[x1_begin, x2_begin, ..., x1_end, x2_end]` format. When omitted, output-shape inference uses zero padding; values are ignored when `output_shape` is provided, and an explicit list must contain two values per spatial axis. |
| `strides` | — | Optional stride along each spatial axis. When omitted, output-shape inference uses a stride of 1 along every spatial axis; an explicit list must contain one value per spatial axis. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `I` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `empty` — With no pooled values, only clear the output; no winner buffer or election is needed.
- `generic` — Elects the greatest input index per destination, then gathers each elected payload or writes zero. The publication pass replaces output initialization and scatter, preserving deterministic duplicate handling and contiguous output writes.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`maxunpool-elect.wgsl.jinja`](build/webgpu/maxunpool-elect.wgsl.jinja)
- [`maxunpool-gather.wgsl.jinja`](build/webgpu/maxunpool-gather.wgsl.jinja)
- [`maxunpool-zerofill.wgsl.jinja`](build/webgpu/maxunpool-zerofill.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.MaxUnpool", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({
  x: { data: xData, shape: [1, 1, 4] },
  indices: { data: indicesData, shape: [1, 1, 4] },
}, {
  attrs: { kernel_shape: [2] },
  outputs: { output: { shape: [1, 1, 5], dtype: "float32" } },
});
```
