---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.QMoE

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Quantized mixture of experts over float32 activations and raw uint8-packed integer expert weights. The package implements 4- and 8-bit symmetric dequantization, ReLU, and interleaved SwiGLU for rank-2 or rank-3 input. Routing uses the softmax over the selected top-k logits with `router_weights` absent. Biases, explicit zero points, FC3, sparse mixing, provider-prepacked layouts, and FP4/FP8 modes are not implemented.

See the [ONNX Runtime `QMoE` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QMoE) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `inputT` | `input` | `T` | — | — | Token activations of shape `(num_tokens, hidden_size)` or `(batch_size, sequence_length, hidden_size)`. | required |
| `routerT` | `router_probs` | `T` | `2` | — | Routing logits of shape `(num_tokens, num_experts)`, where `num_tokens` is the product of the leading input dimensions. | required |
| `fc1T` | `fc1_experts_weights` | `T1` | `3` | — | Raw packed FC1 weights of shape `(num_experts, fusion_size * inter_size, hidden_size / pack_size)`, where `fusion_size` is 2 only for interleaved SwiGLU and `pack_size` is 2 for 4-bit or 1 for 8-bit weights. | required |
| `fc1ScalesT` | `fc1_scales` | `T2` | — | — | Required FC1 dequantization scales: rank 2 `(num_experts, fusion_size * inter_size)` for column-wise quantization, or rank 3 with a trailing `hidden_size / block_size` axis. | required |
| `fc2T` | `fc2_experts_weights` | `T1` | `3` | — | Raw packed FC2 weights of shape `(num_experts, hidden_size, inter_size / pack_size)`. | required |
| `fc2ScalesT` | `fc2_scales` | `T2` | — | — | Required FC2 dequantization scales: rank 2 `(num_experts, hidden_size)` for column-wise quantization, or rank 3 with a trailing `inter_size / block_size` axis. | required |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | same as `inputT` | same as `inputT` | Routed expert output with the same shape as `input`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `activation_alpha` | `1` | Alpha used by SwiGLU; the exact standard default is 1. |
| `activation_beta` | `0` | Beta added to the SwiGLU linear branch; the exact standard default is 0. |
| `activation_type` | `"relu"` | Activation applied after FC1. This package supports `relu` and `swiglu`; the exact standard default is `relu`. |
| `block_size` | — | Optional quantization block size along the reduction dimension. Omission selects column-wise scaling; a supplied value must be a power of two of at least 16 and divide both `hidden_size` and `inter_size`. |
| `expert_weight_bits` | `4` | Integer expert-weight bit width. This package supports 4 and 8; the exact standard default is 4. |
| `k` | `1` | Number of experts selected per token; the exact standard default is 1. |
| `normalize_routing_weights` | `0` | Accepted values are 0 and 1. With the separate `router_weights` input omitted, routing applies a softmax over the selected top-k logits for either value; the attribute only distinguishes the unsupported separate-weight path. |
| `quant_type` | `"int"` | Quantization family. This package supports only the exact standard default `int`. |
| `swiglu_fusion` | `0` | SwiGLU packing mode. ReLU uses the exact standard default 0; supported SwiGLU interleaves gate/up FC1 rows with value 1. |
| `swiglu_limit` | — | Optional SwiGLU clamp limit. Omission means no finite clamp. |
| `use_sparse_mixer` | `0` | Whether to use sparse-mixer routing. The exact standard default and only supported value is 0. |
| `weights_prepacked` | `-1` | Provider weight-layout selector. Values -1 and 0 both consume the public raw packed tensor layout; provider-specific prepacked value 1 is not portable and is rejected. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32` |
| `T1` | `uint8` |
| `T2` | `float32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `quant_grouped_sgmat_routed` — Expert-grouped f32 matrix multiplication with per-invocation routing slots and reuse of the input tile for output publication. The larger operand/output footprint determines workgroup storage admission from device limits.

## Device requirements

Some implementation variants require `subgroup-matrix` and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`expert-group-slots.wgsl.jinja`](build/webgpu/expert-group-slots.wgsl.jinja)
- [`expert-slot-mix.wgsl.jinja`](build/webgpu/expert-slot-mix.wgsl.jinja)
- [`qmoe-fc1-activation-gemv.wgsl.jinja`](build/webgpu/qmoe-fc1-activation-gemv.wgsl.jinja)
- [`qmoe-fc1-activation-grouped-sgmat.wgsl.jinja`](build/webgpu/qmoe-fc1-activation-grouped-sgmat.wgsl.jinja)
- [`qmoe-fc1-activation-grouped.wgsl.jinja`](build/webgpu/qmoe-fc1-activation-grouped.wgsl.jinja)
- [`qmoe-fc1-activation-stage.wgsl.jinja`](build/webgpu/qmoe-fc1-activation-stage.wgsl.jinja)
- [`qmoe-output-gemv.wgsl.jinja`](build/webgpu/qmoe-output-gemv.wgsl.jinja)
- [`qmoe-output-grouped-sgmat.wgsl.jinja`](build/webgpu/qmoe-output-grouped-sgmat.wgsl.jinja)
- [`qmoe-output-grouped.wgsl.jinja`](build/webgpu/qmoe-output-grouped.wgsl.jinja)
- [`qmoe-output-stage.wgsl.jinja`](build/webgpu/qmoe-output-stage.wgsl.jinja)
- [`qmoe-output-zero-inter.wgsl.jinja`](build/webgpu/qmoe-output-zero-inter.wgsl.jinja)
- [`qmoe-route-stage.wgsl.jinja`](build/webgpu/qmoe-route-stage.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.QMoE", { version: 1 });
const { outputT } = await kernel({
  inputT: { data: inputTData, shape: [2, 2, 4] },
  routerT: { data: routerTData, shape: [4, 2] },
  fc1T: { data: fc1TData, shape: [2, 4, 2] },
  fc1ScalesT: { data: fc1ScalesTData, shape: [2, 4] },
  fc2T: { data: fc2TData, shape: [2, 4, 2] },
  fc2ScalesT: { data: fc2ScalesTData, shape: [2, 4] },
});
```
