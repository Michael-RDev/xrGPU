---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.GatedRMSNorm

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Gated RMS normalization as used by Mamba2 and gated DeltaNet attention outputs: `Y = X * rsqrt(mean(X^2) + epsilon) * scale * SiLU(gate)`. The mean of squares is taken over each contiguous group of `C` elements, where `C` is the element count of `scale`, so a per-head norm runs on a packed `(..., H * C)` tensor with no surrounding reshape. Rows are counted flat as the element count of `X` divided by `C` and any rank of at least 1 is accepted. Every intermediate including SiLU is float32 whatever the tensor type; only the store narrows. Bfloat16 is not implemented.

See the [ONNX Runtime `GatedRMSNorm` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GatedRMSNorm) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `xT` | `X` | `T` | — | — | Input activations with shape `(..., H * C)`. Any rank of at least 1 is accepted and the leading axes are folded into the row count. | required |
| `scaleT` | `scale` | `T` | — | — | Normalization weight. Its element count is the normalization span `C`; the standard spelling is `(C)`, and the reshaped `(1, C)` and `(C, 1)` spellings of the same vector are accepted because upstream measures the span as an element count. | required |
| `gateT` | `gate` | `T` | — | — | Gate activations with exactly the same shape as `X`; SiLU is applied to them internally. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `yT` | `Y` | `T` | same as `xT` | same as `xT` | Normalized, scaled and gated activations with the same shape and dtype as `X`. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `epsilon` | `0.00001` | Constant added to the mean of squares before the reciprocal square root. The standard default is `1e-5`; note that the Hugging Face gated-RMS archetype defaults to `1e-6` instead, so a value must be carried explicitly when porting between the two. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `subgroup_rows_vec4` — One aligned lane group per normalization row inside a subgroup, with the row held in registers. The fold is a segmented butterfly over the low lane bits, so several rows share a subgroup with no workgroup memory, no barrier and no idle lanes beyond the row's own power-of-two padding.
- `packed_rows_vec4` — Portable vec4 route: the workgroup is split into a power-of-two lane group per row and each group folds its own contiguous slice of the shared array, so a short row does not idle the workgroup and no subgroup support is required. It also carries rows too wide to stage in registers, which it walks twice.
- `subgroup_rows_scalar` — The barrier-free lane-group schedule for a normalization span that is not a multiple of four: the same segmented butterfly over scalar loads.
- `packed_rows_scalar` — Scalar fallback for a span that is not a multiple of four, and the route every device can select. It keeps the lane-group-per-row schedule so a narrow span still fills the workgroup.
- `subgroup_wide_rows_vec4` — Subgroup partials for a row spanning several whole subgroups; every lane reads the short row sum after one barrier.
- `subgroup_wide_rows_scalar` — Subgroup partials for a row spanning several whole subgroups; every lane reads the short row sum after one barrier.

## Device requirements

Some implementation variants require `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`gated-rms-norm-rows.wgsl.jinja`](build/webgpu/gated-rms-norm-rows.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.GatedRMSNorm", { version: 1 });
const { yT } = await kernel({
  xT: { data: xTData, shape: [3, 4] },
  scaleT: { data: scaleTData, shape: [1] },
  gateT: { data: gateTData, shape: [3, 4] },
});
```
