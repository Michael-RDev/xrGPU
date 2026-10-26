---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.QAttention

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Fused 8-bit self-attention: an exact int32 QKV projection of a uint8 activation by an int8 or uint8 weight, dequantized by `input_scale * weight_scale` and biased, followed by masked multi-head attention over the projected heads. Weight columns are laid out `[Q | K | V]`, and per-column `weight_scale` / `weight_zero_point` are indexed in that column space. `mask_index` is additive with `mask_filter_value`; a unidirectional exclusion overwrites it. Only the uint8-activation, float32 form the reference CPU kernel supports is implemented; `do_rotary` and a shared past/present buffer are not.

See the [ONNX Runtime `QAttention` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QAttention) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `input` | `T1` | — | — | Quantized activations with shape `(batch, sequence, input_hidden_size)`. The input hidden size may exceed `hidden_size` when the projection was pruned. | required |
| `weight` | `T2` | — | — | Quantized projection weight with shape `(input_hidden_size, 3 * hidden_size)`; the column space is `[Q all heads \| K all heads \| V all heads]`. | required |
| `bias` | `T3` | — | — | Float bias with shape `(3 * hidden_size)`, added after dequantization in the same column space as `weight`. | required |
| `input_scale` | `T3` | — | — | Per-tensor activation scale; a scalar or a one-element tensor. | required |
| `weight_scale` | `T3` | — | — | Weight scale: a scalar for per-tensor quantization, or `(3 * hidden_size)` values indexed in the weight column space for per-column quantization. | required |
| `mask_index` | `T4` | — | — | Optional attention mask. Accepted spellings are `(batch)` end positions, `(2 * batch)` end positions followed by start positions, a raw `(batch, total_sequence_length)` key mask, and a raw `(batch, sequence, total_sequence_length)` mask. Every other spelling is rejected. | optional |
| `input_zero_point` | `T1` | — | — | Optional per-tensor activation zero point; a scalar or a one-element tensor. Absent means zero. | optional |
| `weight_zero_point` | `T2` | — | — | Optional weight zero point: a scalar, or `(3 * hidden_size)` values indexed in the weight column space. Absent means zero. | optional |
| `past` | `T3` | — | — | Optional cached keys and values with shape `(2, batch, num_heads, past_sequence_length, head_size)`; the cached tokens precede the projected ones. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T3` | `3` | derived | Attention output with shape `(batch, sequence, hidden_size)`. | required |
| `present` | `T3` | `5` | derived | Optional joined keys and values with shape `(2, batch, num_heads, past_sequence_length + sequence, head_size)`. It is required whenever `past` is supplied. | optional |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `do_rotary` | `0` | Rotary position embedding switch. Defaults to `0`, and only `0` is accepted: the reference CPU kernel ignores the attribute, so no reference definition of the rotation exists for this operator. |
| `mask_filter_value` | `-10000` | Value added to the score of a masked-out key. Defaults to `-10000.0`. |
| `num_heads` | — | Number of attention heads; `hidden_size` must be divisible by it. The request must supply it. |
| `past_present_share_buffer` | — | Declared for schema fidelity. Only `0` is accepted: `past` and `present` are distinct tensors with distinct bindings in every variant of this package, so there is no shared buffer for a non-zero value to name. |
| `scale` | — | Scale applied to the query-key product. When absent or `0`, `1 / sqrt(head_size)` is used. |
| `unidirectional` | `0` | When `1`, a token attends only to itself and earlier tokens. Defaults to `0`. A single-token query is never restricted, matching the reference kernel. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T1` | `uint8` |
| `T2` | `uint8`, `int8` |
| `T3` | `float32` |
| `T4` | `int32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `sgmat_plain_mask_izp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_plain_mask_izp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_plain_mask_izp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_plain_mask_izp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_plain_mask_noizp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_plain_mask_noizp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_plain_mask_noizp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_plain_mask_noizp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_plain_nomask_izp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_plain_nomask_izp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_plain_nomask_izp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_plain_nomask_izp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_plain_nomask_noizp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_plain_nomask_noizp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_plain_nomask_noizp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_plain_nomask_noizp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_present_mask_izp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_present_mask_izp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_present_mask_izp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_present_mask_izp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_present_mask_noizp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_present_mask_noizp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_present_mask_noizp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_present_mask_noizp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_present_nomask_izp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_present_nomask_izp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_present_nomask_izp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_present_nomask_izp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_present_nomask_noizp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_present_nomask_noizp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_present_nomask_noizp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_present_nomask_noizp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_past_mask_izp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_past_mask_izp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_past_mask_izp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_past_mask_izp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_past_mask_noizp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_past_mask_noizp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_past_mask_noizp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_past_mask_noizp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_past_nomask_izp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_past_nomask_izp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_past_nomask_izp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_past_nomask_izp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_past_nomask_noizp_wzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_past_nomask_noizp_wzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `sgmat_past_nomask_noizp_nowzp` — Projects QKV with exact integer products through bounded float32 matrix partials, then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.
- `packed_past_nomask_noizp_nowzp` — Projects QKV with the packed 8-bit integer dot product (portable helpers where the language feature is absent), then dequantizes and scatters into the head-padded working buffers the flash scoring pass reads.

## Device requirements

Some implementation variants require `subgroup-matrix` and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`qattention-attend.wgsl.jinja`](build/webgpu/qattention-attend.wgsl.jinja)
- [`qattention-kv-permute.wgsl.jinja`](build/webgpu/qattention-kv-permute.wgsl.jinja)
- [`qattention-mask.wgsl.jinja`](build/webgpu/qattention-mask.wgsl.jinja)
- [`qattention-qkv-scatter.wgsl.jinja`](build/webgpu/qattention-qkv-scatter.wgsl.jinja)
- [`quant-dp4a-matmul.wgsl.jinja`](build/webgpu/quant-dp4a-matmul.wgsl.jinja)
- [`quant-exact-matrix.wgsl.jinja`](build/webgpu/quant-exact-matrix.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.QAttention", { version: 1 });
const { output } = await kernel({
  input: { data: inputData, shape: [1, 2, 4] },
  weight: { data: weightData, shape: [4, 12] },
  bias: { data: biasData, shape: [12] },
  input_scale: { data: input_scaleData, shape: [1] },
  weight_scale: { data: weight_scaleData, shape: [1] },
}, {
  attrs: { num_heads: 2 },
});
```
