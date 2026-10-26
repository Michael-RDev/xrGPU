---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.GemmaRotaryEmbedding

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Fuses the Gemma rotary-embedding tail: computes `sin` and `cos` from float32 `emb`, casts them to float16, then evaluates `q * cos + q_rot * sin` and the corresponding expression for `k`. `emb` has shape `(batch, seq, dim)` and is broadcast over the head axis of the `(batch, heads, seq, dim)` operands. Each product is rounded to float16 before the addition.

See the [ONNX Runtime `GemmaRotaryEmbedding` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GemmaRotaryEmbedding) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `embT` | `emb` | `U` | `float32` | `3` | — | Rotary angles with shape `(batch_size, seq_len, dim)`, shared by every head. | required |
| `qT` | `q` | `T` | same as logical dtype | `4` | — | Query state with shape `(batch_size, num_heads, seq_len, dim)`. | required |
| `qRotT` | `q_rot` | `T` | same as logical dtype | `4` | — | Half-rotated query state, same shape as `q`. | required |
| `kT` | `k` | `T` | same as logical dtype | `4` | — | Key state, same shape as `q`. | required |
| `kRotT` | `k_rot` | `T` | same as logical dtype | `4` | — | Half-rotated key state, same shape as `q`. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `output1T` | `output1` | `T` | same as `qT` | same as `qT` | Rotary-embedded query, same shape as `q`. | required |
| `output2T` | `output2` | `T` | same as `qT` | same as `qT` | Rotary-embedded key, same shape as `q`. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float16` |
| `U` | `float32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `elementwise` — Elementwise rotation. Streams `vec4` words when each head's `(seq, dim)` plane holds a multiple of four elements, so no word straddles two heads; scalar elements otherwise.

## Device requirements

Every implementation variant requires `shader-f16`; the package has no variant-level fallback without that capability.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`gemma-rotary-embedding.wgsl.jinja`](build/webgpu/gemma-rotary-embedding.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.GemmaRotaryEmbedding", { version: 1 });
const { output1T, output2T } = await kernel({
  embT: { data: embTData, shape: [1, 2, 4] },
  qT: { data: qTData, shape: [1, 1, 2, 4] },
  qRotT: { data: qRotTData, shape: [1, 1, 2, 4] },
  kT: { data: kTData, shape: [1, 1, 2, 4] },
  kRotT: { data: kRotTData, shape: [1, 1, 2, 4] },
});
```
