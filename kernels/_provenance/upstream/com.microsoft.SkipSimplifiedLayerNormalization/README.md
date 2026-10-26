---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.SkipSimplifiedLayerNormalization

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Adds `input` and `skip` (plus optional `bias`), then applies RMS normalization scaled by `gamma`. The optional second output exposes the pre-normalization sum. The schema's training-only mean and inverse-standard-deviation outputs are not implemented.

See the [ONNX Runtime `SkipSimplifiedLayerNormalization` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipSimplifiedLayerNormalization) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `inputT` | `input` | `T` | — | — | Input tensor of shape `(token_count, hidden_size)` or `(batch, sequence, hidden_size)`, normalized over the last axis. | required |
| `skipT` | `skip` | `T` | — | — | Residual tensor of the same shape as `input`, added before normalization. | required |
| `gammaT` | `gamma` | `T` | `1` | — | 1-D scale tensor with shape `(hidden_size)` applied after normalization. | required |
| `biasT` | `bias` | `T` | `1` | — | Optional 1-D bias tensor with shape `(hidden_size)` added to the `input + skip` sum. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | same as `inputT` | same as `inputT` | Normalized output tensor with the same shape as `input`. | required |
| `meanT` | `mean` | `U` | same as `inputT` | derived | Per-row mean; zero for simplified RMS normalization. Shape matches the input with its final axis replaced by one. | optional |
| `invStdT` | `inv_std_var` | `U` | same as `inputT` | derived | Per-row inverse standard deviation, or inverse RMS for simplified normalization. Shape matches the input with its final axis replaced by one. | optional |
| `residualT` | `input_skip_bias_sum` | `T` | same as `inputT` | same as `inputT` | Sum of `input`, `skip`, and optional `bias` before normalization, with the same shape as `input`. | optional |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `epsilon` | `9.999999960041972e-13` | Non-negative epsilon added to the mean square before taking the square root. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `U` | `float32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `stats_mean_plain` — Row normalization returning mean statistics with `plain` optional inputs and outputs.
- `stats_mean_residual` — Row normalization returning mean statistics with `residual` optional inputs and outputs.
- `stats_mean_bias` — Row normalization returning mean statistics with `bias` optional inputs and outputs.
- `stats_mean_bias_residual` — Row normalization returning mean statistics with `bias_residual` optional inputs and outputs.
- `stats_inv_plain` — Row normalization returning inv statistics with `plain` optional inputs and outputs.
- `stats_inv_residual` — Row normalization returning inv statistics with `residual` optional inputs and outputs.
- `stats_inv_bias` — Row normalization returning inv statistics with `bias` optional inputs and outputs.
- `stats_inv_bias_residual` — Row normalization returning inv statistics with `bias_residual` optional inputs and outputs.
- `stats_both_plain` — Row normalization returning both statistics with `plain` optional inputs and outputs.
- `stats_both_residual` — Row normalization returning both statistics with `residual` optional inputs and outputs.
- `stats_both_bias` — Row normalization returning both statistics with `bias` optional inputs and outputs.
- `stats_both_bias_residual` — Row normalization returning both statistics with `bias_residual` optional inputs and outputs.

## Device requirements

Some implementation variants require `shader-f16`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`norm-skip-row-vec4.wgsl.jinja`](build/webgpu/norm-skip-row-vec4.wgsl.jinja)
- [`norm-skip-row.wgsl.jinja`](build/webgpu/norm-skip-row.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.SkipSimplifiedLayerNormalization", { version: 1 });
const { outputT } = await kernel({
  inputT: { data: inputTData, shape: [2, 4] },
  skipT: { data: skipTData, shape: [2, 4] },
  gammaT: { data: gammaTData, shape: [4] },
});
```
