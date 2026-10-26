---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Scan

`ai.onnx`  ·  internal tensor lowering (non-standard)  ·  reviewed against ONNX opset 25

## Description

Internal additive prefix-scan lowering over one rank-1 state and one rank-2 input, with forward and reverse traversal. This package does not implement ONNX `Scan` body-graph semantics.

See the [standard ONNX `Scan` spec](https://onnx.ai/onnx/operators/onnx__Scan.html) for the contract this internal lowering does not implement.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `initial_state` | `T` | `1` | — | Initial rank-1 additive state of shape `[dim]`. | required |
| `scan_input` | `T` | `2` | — | Rank-2 input of shape `[steps, dim]`. Each row is added to the running state in forward or reverse traversal order. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `final_state` | `T` | `1` | same as `initial_state` | Final rank-1 state of shape `[dim]` after all input rows have been accumulated. | required |
| `scan_output` | `T` | `2` | same as `scan_input` | Inclusive additive prefix results with the same `[steps, dim]` shape as `scan_input`. Reverse traversal still writes each result at its corresponding input row. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `reverse` | `0` | When non-zero, the scan input sequence is traversed in reverse order (equivalent to `scan_input_directions=1`); default `0` scans forward. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `vector_channel_prefix_sum` — Scan four adjacent state channels together with float32 accumulation; bound the power-of-two workgroup by the scan length, tuning and device resources, and use subgroups only when it contains complete groups.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`scan-coop-channel.wgsl.jinja`](build/webgpu/scan-coop-channel.wgsl.jinja)
- [`scan-multichunk-apply.wgsl.jinja`](build/webgpu/scan-multichunk-apply.wgsl.jinja)
- [`scan-multichunk-carries.wgsl.jinja`](build/webgpu/scan-multichunk-carries.wgsl.jinja)
- [`scan-multichunk-local.wgsl.jinja`](build/webgpu/scan-multichunk-local.wgsl.jinja)
- [`scan-prefix-sum.wgsl.jinja`](build/webgpu/scan-prefix-sum.wgsl.jinja)
- [`scan-strided-row.wgsl.jinja`](build/webgpu/scan-strided-row.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Scan", { version: 1 });
const { final_state, scan_output } = await kernel({
  initial_state: { data: initial_stateData, shape: [1] },
  scan_input: { data: scan_inputData, shape: [3, 1] },
});
```
