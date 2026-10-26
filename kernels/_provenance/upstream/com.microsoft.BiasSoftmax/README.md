---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.BiasSoftmax

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Computes `softmax(data + bias)` over the flattened suffix beginning at `axis`. The required `is_inner_broadcast` attribute selects how bias rows are reused: consecutive groups for inner broadcast or cyclic groups for outer broadcast. This specializes the `softmax(scores + additive_mask)` pattern used by transformer attention. Float16 and float32 are supported; the schema's double type is not.

See the [ONNX Runtime `BiasSoftmax` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasSoftmax) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `data` | `T` | — | — | The input data tensor. | required |
| `bias` | `T` | — | — | The bias (or additive mask) tensor. Its element count must be an integral number of flattened softmax rows and that row count must divide the data row count. | required |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `data` | same as `data` | The output tensor; same shape as data. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | `1` | The axis from which softmax is applied; dimensions from `axis` onward are included in the softmax reduction. |
| `is_inner_broadcast` | — | Let `data_row` and `bias_row` index flattened rows of length `product(data.shape[axis:])`. When 1, consecutive data-row groups reuse each bias row: `bias_row = floor(data_row / (data_row_count / bias_row_count))`. When 0, bias rows repeat cyclically: `bias_row = data_row % bias_row_count`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `online_subgroup_rows_vec4` — Subgroup-per-row fused bias and softmax with register-resident vec4 values, f32 addition and reduction, and the existing zero-output convention for fully masked rows.
- `online_workgroup_vec4` — Vectorized workgroup online bias-softmax with f32 addition and reduction; portable fallback below the register-resident subgroup-row path.
- `online_workgroup_scalar_vec4` — Logical vec4 online reduction with scalar storage for unaligned row strides. Invalid tail components contribute zero denominator and are never stored.

## Device requirements

Some implementation variants require `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`bias-softmax-longrow-normalize.wgsl.jinja`](build/webgpu/bias-softmax-longrow-normalize.wgsl.jinja)
- [`bias-softmax.wgsl.jinja`](build/webgpu/bias-softmax.wgsl.jinja)
- [`softmax-longrow-stats.wgsl.jinja`](build/webgpu/softmax-longrow-stats.wgsl.jinja)
- [`softmax-online.wgsl.jinja`](build/webgpu/softmax-online.wgsl.jinja)
- [`softmax-subgroup-rows.wgsl.jinja`](build/webgpu/softmax-subgroup-rows.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.BiasSoftmax", { version: 1 });
const { output } = await kernel({
  data: { data: dataData, shape: [1, 2, 2] },
  bias: { data: biasData, shape: [1, 2, 2] },
}, {
  attrs: { is_inner_broadcast: 1 },
});
```
