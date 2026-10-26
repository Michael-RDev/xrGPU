---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.NGramHashMapping

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Engram n-gram hash ids from compressed tokenizer ids. For every order `n` in `[2, max_ngram_size]` it mixes the `n` causal shifts of `input_ids` as `mix = ids[t] * multipliers[0] xor ... xor ids[t-n+1] * multipliers[n-1]`, the products wrapping in two's complement, then emits `mix` modulo each of that order's head vocabulary sizes. `past_ids`/`present_ids` carry the `max_ngram_size - 1` preceding ids, so chunked prefill and decode agree with one full-sequence run. Only `int32` ids are implemented: `int64` is a different answer, not a wider one, since its products wrap at another width.

See the [ONNX Runtime `NGramHashMapping` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.NGramHashMapping) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `inputIdsT` | `input_ids` | `M` | `2` | — | Compressed tokenizer ids with shape `(batch_size, sequence_length)`. A zero-length sequence is accepted: it emits no hash ids and passes the carry state through unchanged. | required |
| `multipliersT` | `multipliers` | `M` | `1` | — | Per-shift hash multipliers with shape `(max_ngram_size)`. Conventionally odd, but any value is accepted; the product wraps in two's complement rather than saturating. | required |
| `vocabSizesT` | `vocab_sizes` | `M` | `1` | — | Per-output-head vocabulary sizes with shape `((max_ngram_size - 1) * n_head_per_ngram)`, conventionally prime and strictly positive. A non-positive entry has no meaningful modulo; this kernel emits a hash id of `0` for that head, which is what every GPU implementation of the operator does, while the reference CPU implementation rejects it instead. | required |
| `pastIdsT` | `past_ids` | `M` | `2` | — | Optional ids for the `max_ngram_size - 1` positions preceding this call, with shape `(batch_size, max_ngram_size - 1)`. Right-aligned, so the last slot is the most recent id. When it is absent the missing history is `pad_id`, which is what a fresh sequence means. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `hashIdsT` | `hash_ids` | `M` | `3` | derived | Hash ids with shape `(batch_size, sequence_length, (max_ngram_size - 1) * n_head_per_ngram)`. The heads of order `n = 2` come first, then `n = 3`, and so on. | required |
| `presentIdsT` | `present_ids` | `M` | `2` | derived | The trailing `max_ngram_size - 1` ids of `past_ids` followed by `input_ids`, with shape `(batch_size, max_ngram_size - 1)`. Feed it back as `past_ids` on the next call. It is always written, into a buffer distinct from `past_ids`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `max_ngram_size` | — | Highest n-gram order, at least 2. It is baked into the rendered kernel, so the shift walk and the head arithmetic are compile-time; this package renders orders up to 16. |
| `n_head_per_ngram` | — | Number of hash heads emitted for each n-gram order, at least 1. It is baked into the rendered kernel alongside `max_ngram_size`, and their product with `max_ngram_size - 1` is capped at 64 heads. |
| `pad_id` | — | Compressed tokenizer id used for causal-shift positions before the beginning of the whole sequence. It must be representable as `int32`. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `M` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `past_shared_prefix` — Stage each causal prefix once per token in workgroup memory, then distribute contiguous head outputs across lanes. The tile obeys device storage and invocation limits. Small combined coefficient tables retain the compact token-owned path to avoid staging and barrier overhead.
- `fresh_shared_prefix` — Stage each causal prefix once per token in workgroup memory, then distribute contiguous head outputs across lanes. The tile obeys device storage and invocation limits. Small combined coefficient tables retain the compact token-owned path to avoid staging and barrier overhead.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`ngram-hash-mapping-shared-prefix.wgsl.jinja`](build/webgpu/ngram-hash-mapping-shared-prefix.wgsl.jinja)
- [`ngram-hash-mapping.wgsl.jinja`](build/webgpu/ngram-hash-mapping.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.NGramHashMapping", { version: 1 });
const { hashIdsT, presentIdsT } = await kernel({
  inputIdsT: { data: inputIdsTData, shape: [1, 1] },
  multipliersT: { data: multipliersTData, shape: [3] },
  vocabSizesT: { data: vocabSizesTData, shape: [4] },
}, {
  attrs: { max_ngram_size: 3, n_head_per_ngram: 2, pad_id: 9 },
});
```
