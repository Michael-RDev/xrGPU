---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.MoE

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Mixture of Experts: applies softmax to `router_probs`, routes each token to the top-`k` experts, applies FC1 and `activation_type`, projects through FC2, then sums the selected outputs using their routing probabilities. SwiGLU takes its operands from a separate FC3 (`swiglu_fusion` 0) or a fused FC1 in interleaved (1) or concatenated (2) order; SiLU may also use FC3 as its multiplicative linear projection. This inference package supports float32 and dense routing (`use_sparse_mixer = 0`); float16, bfloat16, and sparse mixing are not implemented. Quantized weights use `com.microsoft.QMoE`.

See the [ONNX Runtime `MoE` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MoE) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `inputT` | `input` | `T` | — | — | Token activations, either 2D `(num_tokens, hidden_size)` or 3D `(batch_size, sequence_length, hidden_size)`. | required |
| `routerT` | `router_probs` | `T` | `2` | — | 2D router logits of shape `(num_tokens, num_experts)`, where `num_tokens` is the product of every leading dimension of `input`. A full softmax is applied before top-k selection. | required |
| `fc1T` | `fc1_experts_weights` | `T` | `3` | — | 3D first-layer expert weights of shape `(num_experts, fusion_size * inter_size, hidden_size)`, where `fusion_size` is 2 for fused SwiGLU (`swiglu_fusion` 1 or 2) and 1 otherwise. | required |
| `fc1BiasT` | `fc1_experts_bias` | `T` | `2` | — | Optional 2D FC1 bias of shape `(num_experts, fusion_size * inter_size)`. | optional |
| `fc2T` | `fc2_experts_weights` | `T` | `3` | — | 3D second-layer expert weights of shape `(num_experts, hidden_size, inter_size)`. | required |
| `fc2BiasT` | `fc2_experts_bias` | `T` | `2` | — | Optional 2D FC2 bias of shape `(num_experts, hidden_size)`, added per expert before that expert's routing weight is applied. | optional |
| `fc3T` | `fc3_experts_weights` | `T` | `3` | — | Optional 3D third-layer expert weights of shape `(num_experts, inter_size, hidden_size)`. It supplies the separate linear operand for SwiGLU when `swiglu_fusion` is 0, or the multiplicative linear projection for SiLU gating. Other activations do not consume FC3. | optional |
| `fc3BiasT` | `fc3_experts_bias` | `T` | `2` | — | Optional 2D FC3 bias of shape `(num_experts, inter_size)`. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T` | same as `inputT` | same as `inputT` | Routed expert output with the same shape as `input`. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `activation_alpha` | `1` | Alpha parameter used by the activation; the schema default is 1. |
| `activation_beta` | `0` | Beta parameter used by the activation; the schema default is 0. |
| `activation_type` | `"relu"` | Activation applied to the FC1 projection: `relu`, `gelu`, `silu`, `swiglu`, or `identity`. The schema default is `relu`. |
| `k` | `1` | Number of experts selected per token; the schema default is 1. |
| `normalize_routing_weights` | `0` | Whether to normalize the selected routing weights; the schema default is 0. |
| `swiglu_fusion` | `0` | 0 keeps the SwiGLU operands in separate FC1/FC3 GEMMs, 1 interleaves them in one FC1 row, and 2 concatenates them. The schema default is 0. |
| `swiglu_limit` | — | Optional SwiGLU clamp limit; omission means no clamp. |
| `use_sparse_mixer` | `0` | Whether to use sparse-mixer routing. The standard default and only supported value is 0. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `sgmat_grouped_routed_fc1plain_fc3none_fc2plain` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.
- `sgmat_grouped_routed_fc1plain_fc3none_fc2bias` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.
- `sgmat_grouped_routed_fc1plain_fc3plain_fc2plain` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.
- `sgmat_grouped_routed_fc1plain_fc3plain_fc2bias` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.
- `sgmat_grouped_routed_fc1plain_fc3biased_fc2plain` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.
- `sgmat_grouped_routed_fc1plain_fc3biased_fc2bias` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.
- `sgmat_grouped_routed_fc1bias_fc3none_fc2plain` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.
- `sgmat_grouped_routed_fc1bias_fc3none_fc2bias` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.
- `sgmat_grouped_routed_fc1bias_fc3plain_fc2plain` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.
- `sgmat_grouped_routed_fc1bias_fc3plain_fc2bias` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.
- `sgmat_grouped_routed_fc1bias_fc3biased_fc2plain` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.
- `sgmat_grouped_routed_fc1bias_fc3biased_fc2bias` — Expert-grouped f32 matrix projections load public weight layouts directly. Two statically interleaved accumulation chains limit rounding growth; the input tile is reused for result publication. Requires compatible subgroups, f32 fragments, workgroup limits, and complete weight tiles.

## Device requirements

Some implementation variants require `subgroup-matrix` and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`expert-group-slots.wgsl.jinja`](build/webgpu/expert-group-slots.wgsl.jinja)
- [`expert-slot-mix.wgsl.jinja`](build/webgpu/expert-slot-mix.wgsl.jinja)
- [`moe-ffn-gemv.wgsl.jinja`](build/webgpu/moe-ffn-gemv.wgsl.jinja)
- [`moe-ffn-grouped.wgsl.jinja`](build/webgpu/moe-ffn-grouped.wgsl.jinja)
- [`moe-ffn-stage.wgsl.jinja`](build/webgpu/moe-ffn-stage.wgsl.jinja)
- [`moe-grouped-sgmat.wgsl.jinja`](build/webgpu/moe-grouped-sgmat.wgsl.jinja)
- [`moe-output-gemv.wgsl.jinja`](build/webgpu/moe-output-gemv.wgsl.jinja)
- [`moe-output-grouped.wgsl.jinja`](build/webgpu/moe-output-grouped.wgsl.jinja)
- [`moe-output-stage.wgsl.jinja`](build/webgpu/moe-output-stage.wgsl.jinja)
- [`moe-route-stage.wgsl.jinja`](build/webgpu/moe-route-stage.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.MoE", { version: 1 });
const { outputT } = await kernel({
  inputT: { data: inputTData, shape: [1, 1] },
  routerT: { data: routerTData, shape: [1, 2] },
  fc1T: { data: fc1TData, shape: [2, 1, 1] },
  fc2T: { data: fc2TData, shape: [2, 1, 1] },
});
```
