---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Loop

`ai.onnx`  ·  internal tensor lowering (non-standard)  ·  reviewed against ONNX opset 25

## Description

Support status: the standard ONNX `Loop` control-flow operator is not implemented because standalone kernel packages cannot carry or execute its body graph. This internal lowering performs one fixed recurrence—adding `step` to a rank-1 loop-carried tensor for up to `M` iterations while writing a dense scan tensor; `step` is not an ONNX `Loop` input, and this lowering must not be treated as ONNX `Loop`.

See the [standard ONNX `Loop` spec](https://onnx.ai/onnx/operators/onnx__Loop.html) for the contract this internal lowering does not implement.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `m` | `M` | `I` | — | — | Required uint32 trip-count limit encoded as either a scalar tensor or a one-element rank-1 tensor. The lowering executes at most `min(M, scan_output.shape[0])` iterations. | required |
| `cond` | — | `B` | — | — | Required initial condition encoded as either a scalar tensor or a one-element rank-1 tensor. A false value executes zero iterations; a true value permits all iterations selected by `M`. This fixed lowering does not update the condition inside the loop. | required |
| `v_initial` | — | `T` | `1` | — | Initial rank-1 loop-carried state of shape `[dim]`. | required |
| `step` | — | `T` | `1` | — | Implementation-specific rank-1 increment of shape `[dim]`, added elementwise to the state on every executed iteration. This is not a standard ONNX `Loop` input. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `v_final` | `T` | `1` | same as `v_initial` | Final rank-1 state of shape `[dim]` after the executed additions. | required |
| `scan_output` | `T` | `2` | — | Dense tensor of shape `[scan_steps, dim]`. Each executed row contains the updated state for that iteration; rows beyond the executed trip count are zero-filled. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32` |
| `I` | `uint32` |
| `B` | `uint32`, `bool` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`loop-add-step.wgsl.jinja`](build/webgpu/loop-add-step.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `scan_output`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Loop", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { v_final, scan_output } = await kernel({
  m: { data: mData, shape: [1] },
  cond: { data: condData, shape: [1] },
  v_initial: { data: v_initialData, shape: [1] },
  step: { data: stepData, shape: [1] },
}, {
  outputs: { scan_output: { shape: [1, 1], dtype: "float32" } },
});
```
