---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.EngramGate

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Fuses the Engram gate. For each `(batch, sequence, hc_mult)` row it RMS-normalizes `key` and `query` with their per-stream scales, forms `dot = sum(RMSNorm(key) * RMSNorm(query)) / sqrt(hidden_size)`, and writes `sigmoid(sign(dot) * sqrt(max(abs(dot), 1e-6))) * value`, broadcasting the value row shared by every hyper-connection. That `1e-6` floor is fixed and is not `epsilon`; a zero dot product gives a gate of exactly `0.5`. Both sums of squares and the scaled cross term accumulate in one float32 pass, and only the store narrows. Bfloat16 is not implemented.

See the [ONNX Runtime `EngramGate` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.EngramGate) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `keyT` | `key` | `T` | `4` | — | Projected Engram keys with shape `(batch_size, sequence_length, hc_mult, hidden_size)`. | required |
| `queryT` | `query` | `T` | `4` | — | Hidden-state queries, exactly the same shape as `key`. The upstream kernels require equality rather than broadcasting, and so does this one. | required |
| `valueT` | `value` | `T` | `3` | — | Projected Engram value shared by every hyper-connection, with shape `(batch_size, sequence_length, hidden_size)`. Each row of it is gated `hc_mult` times, once per stream. | required |
| `keyNormScaleT` | `key_norm_scale` | `T` | `2` | — | RMSNorm scale for keys with shape `(hc_mult, hidden_size)`: one weight row per hyper-connection stream, selected by the row's stream index. | required |
| `queryNormScaleT` | `query_norm_scale` | `T` | `2` | — | RMSNorm scale for queries with shape `(hc_mult, hidden_size)`, indexed by the same stream index as the key scale. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `4` | same as `keyT` | The gated value tensor, with the same shape and dtype as `key`. Every element of a row carries the same gate, so the row's stream index is only observable through the scale lookup. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `epsilon` | `0.00001` | Constant added to both mean-of-squares denominators before the reciprocal square root. It does not reach the gate: the `1e-6` floor under the square root is a separate fixed constant. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `subgroup_rows_vec4` — One aligned lane group per gated row inside a subgroup, with the row's value words held in registers across the fold. The three accumulators travel as one vector through a segmented butterfly over the low lane bits, so several rows share a subgroup with no workgroup memory and no barrier.
- `packed_rows_vec4` — Portable vec4 route: the workgroup is split into a power-of-two lane group per row and each group folds its own contiguous slice of one shared vector array, so a short row does not idle the workgroup and no subgroup support is required. It also carries rows too wide to stage in registers, which it walks twice.
- `subgroup_rows_scalar` — The barrier-free lane-group schedule for a hidden size that is not a multiple of four: the same segmented butterfly over scalar loads.
- `packed_rows_scalar` — Scalar fallback for a hidden size that is not a multiple of four, and the route every device can select. It keeps the lane-group-per-row schedule so a narrow row still fills the workgroup.

## Device requirements

Some implementation variants require `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`engram-gate-rows.wgsl.jinja`](build/webgpu/engram-gate-rows.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.EngramGate", { version: 1 });
const { outputT } = await kernel({
  keyT: { data: keyTData, shape: [1, 1, 1, 2] },
  queryT: { data: queryTData, shape: [1, 1, 1, 2] },
  valueT: { data: valueTData, shape: [1, 1, 2] },
  keyNormScaleT: { data: keyNormScaleTData, shape: [1, 2] },
  queryNormScaleT: { data: queryNormScaleTData, shape: [1, 2] },
});
```
