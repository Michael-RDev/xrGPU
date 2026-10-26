---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.HannWindow

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 17

## Description

Generates a Hann window of a given length using the formula `0.5 - 0.5 * cos(2π * n / N)`, where `N` is `size` for periodic mode or `size - 1` for symmetric mode. The output is a 1-D tensor of length `size`.

See the [ONNX `HannWindow` spec](https://onnx.ai/onnx/operators/onnx__HannWindow.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `size` | `T1` | `0` | — | Scalar indicating the number of elements in the output window. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `output` | `T2` | `1` | — | 1-D Hann window tensor of shape `[size]`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `output_datatype` | `1` | Data type of the output tensor, encoded as a DataType enum value from TensorProto (default `1` = FLOAT). |
| `periodic` | `1` | If `1` (default), returns a periodic window of length `size`; if `0`, returns a symmetric window of length `size`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T1` | `int32` |
| `T2` | `float32`, `float16`, `uint32`, `int32`, `uint8`, `int8`, `int16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`window-cosine-sum.wgsl.jinja`](build/webgpu/window-cosine-sum.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.HannWindow", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({ size: { data: sizeData, shape: [] } }, {
  outputs: { y: { shape: [1], dtype: "float32" } },
});
```
