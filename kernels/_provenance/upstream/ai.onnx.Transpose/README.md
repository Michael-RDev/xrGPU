---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Transpose

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Transposes the input tensor by permuting its axes according to the `perm` attribute. Axis `i` of the output corresponds to axis `perm[i]` of the input; if `perm` is omitted, the axes are reversed (`n-1, ..., 0`).

See the [ONNX `Transpose` spec](https://onnx.ai/onnx/operators/onnx__Transpose.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `data` | `T` | — | — | The input tensor to transpose. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `transposed` | `T` | same as `x` | — | The transposed output tensor with permuted axes. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `perm` | — | Optional permutation of the input axes. It must contain every axis from 0 through rank - 1 exactly once. When omitted, the axes are reversed. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `int16`, `uint32`, `uint8`, `int8`, `bool`, `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `strided_planes_vec4` — Tiles independent physical strided planes while preserving the intervening axis. Both global IO directions are contiguous inside the tile. Demoted when padding occupies more than half the tile slots.
- `tiled_scalar` — Stages ragged two-dimensional transpose planes through a padded workgroup tile so reads and writes remain coalesced when vec4 alignment is unavailable.
- `inner_vec4` — Vectorizes the contiguous innermost dimension for permutations that leave that axis in place. It assigns multiple vectors per invocation only when enough invocations remain to keep the dispatch populated.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`datamove-flat-copy.wgsl.jinja`](build/webgpu/datamove-flat-copy.wgsl.jinja)
- [`datamove-transpose-2d-tiled-scalar.wgsl.jinja`](build/webgpu/datamove-transpose-2d-tiled-scalar.wgsl.jinja)
- [`datamove-transpose-2d-tiled.wgsl.jinja`](build/webgpu/datamove-transpose-2d-tiled.wgsl.jinja)
- [`datamove-transpose-vec4.wgsl.jinja`](build/webgpu/datamove-transpose-vec4.wgsl.jinja)
- [`transpose.wgsl.jinja`](build/webgpu/transpose.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Transpose", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({ x: { data: xData, shape: [2, 3] } }, {
  outputs: { y: { shape: [3, 2], dtype: "float32" } },
});
```
