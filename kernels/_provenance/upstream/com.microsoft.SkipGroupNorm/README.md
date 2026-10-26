---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.SkipGroupNorm

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Adds a residual and an optional bias to the input, then group-normalizes the sum with an optional fused SiLU: `S = X + skip + bias`, `Y = gamma * (S - mean) / sqrt(variance + epsilon) + beta`. Statistics are taken per batch item over each group of `C / groups` channels and every spatial position, under either layout. `skip` is either exactly `X`'s shape or a per-channel `(N, C)` / `(N, 1, 1, C)` residual. The sum is rounded to the tensor type before the statistics see it, so `Y` does not depend on whether `S` was requested. `X` and `gamma`/`beta` carry independent element types.

See the [ONNX Runtime `SkipGroupNorm` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipGroupNorm) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `X` | `T` | `4` | — | Input image tensor, `(N, H, W, C)` when `channels_last` is 1 and `(N, C, H, W)` otherwise. | required |
| `gamma` | `M` | `1` | — | Per-channel affine scale of shape `(C)`. | required |
| `beta` | `M` | `1` | — | Per-channel affine offset of shape `(C)`. | required |
| `skip` | `T` | — | — | Residual added to `X`. Either exactly `X`'s shape, or `(N, C)` / `(N, 1, 1, C)` to broadcast one value per batch item and channel. A `(N, 1, 1, C)` tensor at `H = W = 1` is an exact match and takes the non-broadcast route, which is what upstream does. | required |
| `bias` | `T` | `1` | — | Optional per-channel bias of shape `(C)` added to `X + skip` before normalization. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `Y` | `T` | `4` | same as `X` | Normalized tensor with the same shape and element type as `X`. | required |
| `S` | `T` | `4` | same as `X` | Optional second output holding `X + skip + bias` at the tensor's element type, the values the statistics are taken over. | optional |

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

- `split_bias_sum` — Three dispatches: partial group statistics, combine mean and reciprocal standard deviation, then normalize and apply affine weights. Channels-last statistics use four, two, or one aligned values per lane; group blocks and spatial rows follow device limits. Fuses skip, optional bias, and optional residual output.
- `fused_bias_sum` — One workgroup per batch item and group for a request with a bias and the residual sum output, walking the group twice inside a single dispatch: once to form the residual sum and its moments, once to write the affine result.
- `split_bias_plain` — Three dispatches: partial group statistics, combine mean and reciprocal standard deviation, then normalize and apply affine weights. Channels-last statistics use four, two, or one aligned values per lane; group blocks and spatial rows follow device limits. Fuses skip, optional bias, and optional residual output.
- `fused_bias_plain` — One workgroup per batch item and group for a request with a bias and no residual sum output, walking the group twice inside a single dispatch: once to form the residual sum and its moments, once to write the affine result.
- `split_sum` — Three dispatches: partial group statistics, combine mean and reciprocal standard deviation, then normalize and apply affine weights. Channels-last statistics use four, two, or one aligned values per lane; group blocks and spatial rows follow device limits. Fuses skip, optional bias, and optional residual output.
- `fused_sum` — One workgroup per batch item and group for a request with no bias and the residual sum output, walking the group twice inside a single dispatch: once to form the residual sum and its moments, once to write the affine result.
- `split_plain` — Three dispatches: partial group statistics, combine mean and reciprocal standard deviation, then normalize and apply affine weights. Channels-last statistics use four, two, or one aligned values per lane; group blocks and spatial rows follow device limits. Fuses skip, optional bias, and optional residual output.
- `fused_plain` — One workgroup per batch item and group for a request with neither a bias nor the residual sum output, walking the group twice inside a single dispatch: once to form the residual sum and its moments, once to write the affine result.

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.SkipGroupNorm", { version: 1 });
const { Y } = await kernel({
  X: { data: XData, shape: [2, 3, 2, 8] },
  gamma: { data: gammaData, shape: [8] },
  beta: { data: betaData, shape: [8] },
  skip: { data: skipData, shape: [2, 8] },
}, {
  attrs: { groups: 2, activation: 0 },
});
```
