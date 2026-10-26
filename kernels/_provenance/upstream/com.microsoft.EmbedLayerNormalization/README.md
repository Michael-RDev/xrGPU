---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.EmbedLayerNormalization

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

BERT embedding fusion: looks up word and position tables, optionally adds a segment table, then applies layer normalization. A segment table without IDs uses row 0. `embedding_sum` is the pre-normalization sum. `mask_index` is the first zero or the sequence length; without `mask`, it is zero. Batch and sequence dimensions must be non-empty.

See the [ONNX Runtime `EmbedLayerNormalization` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.EmbedLayerNormalization) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `inputIdsT` | `input_ids` | `T1` | `int32` | `2` | — | Word ids of shape `(batch_size, sequence_length)`. | required |
| `segmentIdsT` | `segment_ids` | `T1` | `int32` | `2` | — | Segment ids `(batch_size, sequence_length)`. Requires `segment_embedding`; when omitted with that table present, every token uses row 0. Values must be valid non-negative table-row indices. | optional |
| `wordEmbeddingT` | `word_embedding` | `T` | same as logical dtype | `2` | — | Non-empty word embedding table `(vocab, hidden_size)`. Every `input_ids` value must be a valid non-negative row index. | required |
| `positionEmbeddingT` | `position_embedding` | `T` | same as logical dtype | `2` | — | Non-empty position embedding table `(max_positions, hidden_size)`. Without `position_ids`, it must contain at least `sequence_length` rows. | required |
| `segmentEmbeddingT` | `segment_embedding` | `T` | same as logical dtype | `2` | — | Non-empty segment embedding table `(segments, hidden_size)`. If `segment_ids` is absent, row 0 is used for every token. | optional |
| `gammaT` | `gamma` | `T` | same as logical dtype | `1` | — | Layer-normalization scale of shape `(hidden_size)`. | required |
| `betaT` | `beta` | `T` | same as logical dtype | `1` | — | Layer-normalization bias of shape `(hidden_size)`. | required |
| `maskT` | `mask` | `T1` | `int32` | `2` | — | Attention mask of shape `(batch_size, sequence_length)`. Only used to produce `mask_index`. | optional |
| `positionIdsT` | `position_ids` | `T1` | `int32` | `2` | — | Position ids `(batch_size, sequence_length)`, or `(1, sequence_length)` to share one row across the batch. Values must be valid non-negative table-row indices; absent uses the position within the sequence. | optional |

## Outputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | same as logical dtype | `3` | derived | Normalized embeddings of shape `(batch_size, sequence_length, hidden_size)`. | required |
| `maskIndexT` | `mask_index` | `T1` | `int32` | `1` | `[inputIdsT[0]]` | Position of the first zero in each mask row, or `sequence_length` when no zero exists; shape `(batch_size)`. It is zero when the optional mask input is absent. | optional |
| `embeddingSumT` | `embedding_sum` | `T` | same as logical dtype | `3` | derived | The summed embeddings before normalization, including the segment term when present. Float16 uses staged `(word + segment) + position`; float32 uses `(word + position) + segment`. | optional |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `epsilon` | `9.999999960041972e-13` | Non-negative epsilon added to the layer-normalization variance before taking the square root. |
| `mask_index_type` | — | Optional shape-inference hint for the `mask_index` output type. The schema's `T1` constraint fixes the output tensor data type to int32. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `T1` | `int32` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`embed-mask-index.wgsl.jinja`](build/webgpu/embed-mask-index.wgsl.jinja)
- [`embed-normalize.wgsl.jinja`](build/webgpu/embed-normalize.wgsl.jinja)
- [`embed-sum.wgsl.jinja`](build/webgpu/embed-sum.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.EmbedLayerNormalization", { version: 1 });
const { outputT } = await kernel({
  inputIdsT: { data: inputIdsTData, shape: [1, 2] },
  wordEmbeddingT: { data: wordEmbeddingTData, shape: [2, 2] },
  positionEmbeddingT: { data: positionEmbeddingTData, shape: [2, 2] },
  gammaT: { data: gammaTData, shape: [2] },
  betaT: { data: betaTData, shape: [2] },
});
```
