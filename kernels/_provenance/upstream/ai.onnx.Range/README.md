---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Range

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 27

## Description

Generates a 1-D tensor of numbers starting at `start`, incrementing by `delta`, up to but not including `limit`. The output length is `max(ceil((limit - start) / delta), 0)`, and element `i` equals `start + i * delta`. Float16 inputs use float32 intermediate arithmetic when `stash_type` is `1`, then cast each result back to float16.

See the [ONNX `Range` spec](https://onnx.ai/onnx/operators/onnx__Range.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `start` | `T` | `0` | — | Scalar first value in the output sequence. | required |
| `limit` | `T` | `0` | — | Scalar exclusive upper bound of the sequence. | required |
| `delta` | `T` | `0` | — | Scalar step size between consecutive output values. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | `1` | — | 1-D tensor containing the generated range of values, with the same dtype as the inputs. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `stash_type` | `1` | TensorProto element type used for float16 intermediate arithmetic; this implementation supports the standard float32 value (`1`). It has no effect for float32 and int32 inputs. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`range.wgsl.jinja`](build/webgpu/range.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Range", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({
  start: { data: startData, shape: [] },
  limit: { data: limitData, shape: [] },
  delta: { data: deltaData, shape: [] },
}, {
  outputs: { output: { shape: [1], dtype: "float32" } },
});
```
