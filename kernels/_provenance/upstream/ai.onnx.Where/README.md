---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Where

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 16

## Description

Selects elements from `X` or `Y` according to a boolean `condition`, following NumPy-style multidirectional broadcasting. Where `condition` is `true`, the output takes the value from `X`; otherwise it takes the value from `Y`.

See the [ONNX `Where` spec](https://onnx.ai/onnx/operators/onnx__Where.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `condition` | — | `C` | — | — | Boolean mask; nonzero entries select from X, zero entries select from Y. | required |
| `x` | `X` | `T` | — | — | Values selected at indices where condition is true. | required |
| `y` | `Y` | `T` | — | — | Values selected at indices where condition is false. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | derived | broadcast result of `condition`, `x`, and `y` | Tensor of shape equal to the broadcasted shape of condition, X, and Y. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `C` | `bool` |
| `T` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool`, `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`where-broadcast.wgsl.jinja`](build/webgpu/where-broadcast.wgsl.jinja)
- [`where-vec4.wgsl.jinja`](build/webgpu/where-vec4.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Required output shapes and logical data types are inferred from the supplied inputs and attributes; result tensors are allocated automatically.

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

A `bool` input takes 0/1 values in a `Uint8Array` together with `dtype: "bool"`, since no typed-array class implies it.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Where", { version: 1 });
const { output } = await kernel({
  condition: { data: conditionData, shape: [3], dtype: "bool" },
  x: { data: xData, shape: [1, 3] },
  y: { data: yData, shape: [] },
});
```
