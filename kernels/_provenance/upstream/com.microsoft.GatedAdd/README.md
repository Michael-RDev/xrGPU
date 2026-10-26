---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.GatedAdd

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Adds `Y`, scaled by a per-row `gate`, to `X`: `output = X + round_to_T(Y * gate)`. `X` and `Y` have shape `(..., C)`; `gate` has the same rank with a trailing dimension of 1, so one value covers each row of `C` channels. Rounding the product to `T` before the addition preserves the semantics of a separate `Mul` followed by `Add`. Bfloat16 is not implemented.

See the [ONNX Runtime `GatedAdd` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GatedAdd) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `X` | `T` | — | — | Unscaled input with shape `(..., C)`. Any rank of at least 1 is accepted; only the trailing channel axis is distinguished. | required |
| `Y` | `T` | — | — | Input scaled by the gate, with the same shape as `X`. | required |
| `gate` | `T` | — | — | Per-row gate with shape `(..., 1)`: the same rank and leading dimensions as `X`, with a trailing dimension of 1 that broadcasts over the `C` channels. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `X` | same as `X` | Gated sum `X + round_to_T(Y * gate)`, with the same shape as `X`. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`gated-add.wgsl.jinja`](build/webgpu/gated-add.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.GatedAdd", { version: 1 });
const { output } = await kernel({
  X: { data: XData, shape: [2, 3] },
  Y: { data: YData, shape: [2, 3] },
  gate: { data: gateData, shape: [2, 1] },
});
```
