---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.SkipLayerNormalization

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Fuses skip addition with layer normalization over a non-empty final hidden axis of rank-2 or rank-3 input. Exact-shape skip supports float32 and float16. Optional float32 mean and inverse-standard-deviation outputs expose row statistics. Broadcast skip supports rank-3 float32 input with beta, no bias or residual output, and a hidden size divisible by four. Output-only and residual-only paths have additional beta, bias and alignment requirements stated by their variants. Bfloat16 is not implemented.

See the [ONNX Runtime `SkipLayerNormalization` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipLayerNormalization) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `inputT` | `input` | `T` | — | — | Primary input normalized over the final hidden-size axis. Rank 3 is the standard shape; rank 2 is a supported extension. | required |
| `skipT` | `skip` | `T` | — | — | Residual tensor. For rank-3 input it is exact shape, `(1, sequence_length, hidden_size)`, or `(sequence_length, hidden_size)`; rank-2 input requires exact shape. | required |
| `gammaT` | `gamma` | `T` | `1` | — | Layer-norm scale weights of shape `(hidden_size)`. | required |
| `betaT` | `beta` | `T` | `1` | — | Layer-norm bias weights of shape `(hidden_size)`. | optional |
| `biasT` | `bias` | `T` | `1` | — | Optional additive bias of shape `(hidden_size)` added to `input + skip` before normalization. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | same as `inputT` | same as `inputT` | Normalized output tensor with the same shape as `input`. | required |
| `meanT` | `mean` | `U` | same as `inputT` | derived | Per-row mean; zero for simplified RMS normalization. Shape matches the input with its final axis replaced by one. | optional |
| `invStdT` | `inv_std_var` | `U` | same as `inputT` | derived | Per-row inverse standard deviation, or inverse RMS for simplified normalization. Shape matches the input with its final axis replaced by one. | optional |
| `residualT` | `input_skip_bias_sum` | `T` | same as `inputT` | same as `inputT` | Sum of `input`, `skip`, and `bias` (when present) before normalization, with the same shape as `input`. | optional |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `epsilon` | `9.999999960041972e-13` | Non-negative epsilon added to the variance before taking the square root. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `U` | `float32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `hidden1_f32_no_beta` — Closed-form one-element rows bind only gamma, optional beta, the output, and row parameters. Each invocation writes one row without reading the unused residual inputs.
- `hidden1_f32_beta` — Closed-form one-element rows bind only gamma, optional beta, the output, and row parameters. Each invocation writes one row without reading the unused residual inputs.
- `hidden1_f16_no_beta` — Closed-form one-element rows bind only gamma, optional beta, the output, and row parameters. Each invocation writes one row without reading the unused residual inputs.
- `hidden1_f16_beta` — Closed-form one-element rows bind only gamma, optional beta, the output, and row parameters. Each invocation writes one row without reading the unused residual inputs.
- `stats_mean_plain` — Row normalization returning mean statistics with `plain` optional inputs and outputs.
- `stats_mean_residual` — Row normalization returning mean statistics with `residual` optional inputs and outputs.
- `stats_mean_beta` — Row normalization returning mean statistics with `beta` optional inputs and outputs.
- `stats_mean_beta_residual` — Row normalization returning mean statistics with `beta_residual` optional inputs and outputs.
- `stats_mean_bias` — Row normalization returning mean statistics with `bias` optional inputs and outputs.
- `stats_mean_bias_residual` — Row normalization returning mean statistics with `bias_residual` optional inputs and outputs.
- `stats_mean_bias_beta` — Row normalization returning mean statistics with `bias_beta` optional inputs and outputs.
- `stats_mean_bias_beta_residual` — Row normalization returning mean statistics with `bias_beta_residual` optional inputs and outputs.
- `stats_inv_plain` — Row normalization returning inv statistics with `plain` optional inputs and outputs.
- `stats_inv_residual` — Row normalization returning inv statistics with `residual` optional inputs and outputs.
- `stats_inv_beta` — Row normalization returning inv statistics with `beta` optional inputs and outputs.
- `stats_inv_beta_residual` — Row normalization returning inv statistics with `beta_residual` optional inputs and outputs.
- `stats_inv_bias` — Row normalization returning inv statistics with `bias` optional inputs and outputs.
- `stats_inv_bias_residual` — Row normalization returning inv statistics with `bias_residual` optional inputs and outputs.
- `stats_inv_bias_beta` — Row normalization returning inv statistics with `bias_beta` optional inputs and outputs.
- `stats_inv_bias_beta_residual` — Row normalization returning inv statistics with `bias_beta_residual` optional inputs and outputs.
- `stats_both_plain` — Row normalization returning both statistics with `plain` optional inputs and outputs.
- `stats_both_residual` — Row normalization returning both statistics with `residual` optional inputs and outputs.
- `stats_both_beta` — Row normalization returning both statistics with `beta` optional inputs and outputs.
- `stats_both_beta_residual` — Row normalization returning both statistics with `beta_residual` optional inputs and outputs.
- `stats_both_bias` — Row normalization returning both statistics with `bias` optional inputs and outputs.
- `stats_both_bias_residual` — Row normalization returning both statistics with `bias_residual` optional inputs and outputs.
- `stats_both_bias_beta` — Row normalization returning both statistics with `bias_beta` optional inputs and outputs.
- `stats_both_bias_beta_residual` — Row normalization returning both statistics with `bias_beta_residual` optional inputs and outputs.

## Device requirements

Some implementation variants require `shader-f16`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`norm-skip-row-vec4.wgsl.jinja`](build/webgpu/norm-skip-row-vec4.wgsl.jinja)
- [`norm-skip-row.wgsl.jinja`](build/webgpu/norm-skip-row.wgsl.jinja)
- [`norm-stats-copy.wgsl.jinja`](build/webgpu/norm-stats-copy.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.SkipLayerNormalization", { version: 1 });
const { outputT } = await kernel({
  inputT: { data: inputTData, shape: [2, 4] },
  skipT: { data: skipTData, shape: [2, 4] },
  gammaT: { data: gammaTData, shape: [4] },
});
```
