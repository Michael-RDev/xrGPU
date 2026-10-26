---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.GatedRelativePositionBias

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Scales a precomputed relative-position bias by a per-query gate. The biased query row for each head is projected through `weight` and `bias` to `D` values; the first `D/2` sum to one gate and the last `D/2` to another, both through a logistic, and `output = (gate_u * (gate_r * eco_a - 1) + 2) * rel_pos`. `D` must be even and at most 32. All gate arithmetic is float32 whatever the tensor type, and only the store narrows.

See the [ONNX Runtime `GatedRelativePositionBias` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GatedRelativePositionBias) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `queryLayerT` | `query_layer` | `T` | `3` | — | Query activations with shape `(batch_size, seq_len, num_heads * head_size)`. | required |
| `queryBiasT` | `query_bias` | `T` | `1` | — | Bias added to `query_layer` before the projection, with shape `(num_heads * head_size)`. | required |
| `relPosT` | `rel_pos` | `T` | `4` | — | Relative position bias with shape `(1, num_heads, seq_len, seq_len)`. Its single batch is broadcast over the output's batches. | required |
| `weightT` | `weight` | `T` | `2` | — | Projection weight for the gate, with shape `(head_size, D)` where `D` is even. | required |
| `biasT` | `bias` | `T` | `1` | — | Projection bias for the gate, with shape `(D)`. | required |
| `ecoAT` | `eco_a` | `T` | `4` | — | Per-head coefficient with shape `(1, num_heads, 1, 1)`. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | `4` | `[queryLayerT[0], num_heads, queryLayerT[1], queryLayerT[1]]` | Gated relative position bias with shape `(batch_size, num_heads, seq_len, seq_len)`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `num_heads` | — | Number of attention heads. The head size is the query hidden size divided by this count. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `row` — One workgroup per output row, streaming the bias row as `vec4` words when the sequence length is a multiple of four and as scalars otherwise; the route every device can select. The gate is computed once into workgroup memory, so its cost is amortized over the whole row rather than repeated per element.
- `row_head_tree` — Projects head elements in parallel with a tree reduction of both float32 gate sums. Used when the head dimension fills at least four head lanes per projection lane or, on adapters with a variable or unreported subgroup range, for long vec4-aligned rows with a head-to-projection ratio of at least eight. Subgroup leaders combine sums in workgroup memory.
- `row_head_subgroups` — Projects head elements in parallel with a subgroups reduction of both float32 gate sums. Used when the head dimension fills at least four head lanes per projection lane or, on adapters with a variable or unreported subgroup range, for long vec4-aligned rows with a head-to-projection ratio of at least eight. Subgroup leaders combine sums in workgroup memory.

## Device requirements

Some implementation variants require `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`gated-relative-position-bias.wgsl.jinja`](build/webgpu/gated-relative-position-bias.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.GatedRelativePositionBias", { version: 1 });
const { outputT } = await kernel({
  queryLayerT: { data: queryLayerTData, shape: [2, 1, 3] },
  queryBiasT: { data: queryBiasTData, shape: [3] },
  relPosT: { data: relPosTData, shape: [1, 3, 1, 1] },
  weightT: { data: weightTData, shape: [1, 2] },
  biasT: { data: biasTData, shape: [2] },
  ecoAT: { data: ecoATData, shape: [1, 3, 1, 1] },
}, {
  attrs: { num_heads: 3 },
});
```
