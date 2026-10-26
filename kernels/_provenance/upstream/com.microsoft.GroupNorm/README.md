---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.GroupNorm

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Group normalization over a 4-D image tensor with an optional fused SiLU: `Y = gamma * (X - mean) / sqrt(variance + epsilon) + beta`, with mean and variance taken per batch item over each group of `C / groups` channels and every spatial position. Both layouts are implemented: `channels_last = 1` reads `(N, H, W, C)` and `channels_last = 0` reads `(N, C, H, W)`. `X`/`Y` and `gamma`/`beta` carry independent element types, so float16 activations with float32 weights is a first-class case. Statistics and the affine are float32 whatever the tensor type; only the store narrows.

See the [ONNX Runtime `GroupNorm` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupNorm) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `X` | `T` | `4` | — | Input image tensor, `(N, H, W, C)` when `channels_last` is 1 and `(N, C, H, W)` otherwise. | required |
| `gamma` | `M` | `1` | — | Per-channel affine scale of shape `(C)`. | required |
| `beta` | `M` | `1` | — | Per-channel affine offset of shape `(C)`. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `Y` | `T` | `4` | same as `X` | Normalized tensor with the same shape and element type as `X`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `activation` | — | Activation applied after the affine transform: `0` for none, `1` for SiLU. The request must supply it; the operator declares no default. |
| `channels_last` | `1` | `1` when `X` and `Y` are `(N, H, W, C)`, `0` when they are `(N, C, H, W)`. Defaults to `1`. Any non-zero value selects the channels-last layout. |
| `epsilon` | `0.00001` | Value added to the group variance before the reciprocal square root. Defaults to `1e-5`. |
| `groups` | — | Number of channel groups the statistics are taken over. It must divide `C`. The request must supply it; the operator declares no default. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `M` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `split` — Three dispatches: partial group statistics, combine mean and reciprocal standard deviation, then normalize and apply affine weights. Channels-last statistics use four, two, or one aligned values per lane; group blocks and spatial rows follow device limits.
- `fused` — One workgroup per batch item and group, walking the group twice inside a single dispatch: once for the statistics and once to write the affine result. It avoids the split route's two extra launches on small tensors and is the route that needs no scratch buffer.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`group-norm-combine.wgsl.jinja`](build/webgpu/group-norm-combine.wgsl.jinja)
- [`group-norm-fused.wgsl.jinja`](build/webgpu/group-norm-fused.wgsl.jinja)
- [`group-norm-slab.wgsl.jinja`](build/webgpu/group-norm-slab.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.GroupNorm", { version: 1 });
const { Y } = await kernel({
  X: { data: XData, shape: [1, 2, 2, 16] },
  gamma: { data: gammaData, shape: [16] },
  beta: { data: betaData, shape: [16] },
}, {
  attrs: { groups: 4, activation: 0 },
});
```
