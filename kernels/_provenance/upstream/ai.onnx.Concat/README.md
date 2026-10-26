---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Concat

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Concatenates a list of tensors along a single axis into one output tensor. All inputs must have the same shape except along the concatenation `axis`, which may differ between inputs. Negative `axis` values count from the end of the rank.

See the [ONNX `Concat` spec](https://onnx.ai/onnx/operators/onnx__Concat.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `T` | — | — | First input tensor to concatenate. | required |
| `b` | `B` | `T` | — | — | Second input tensor to concatenate (optional). | optional |
| `d` | `D` | `T` | — | — | Third input tensor to concatenate (optional). | optional |
| `e` | `E` | `T` | — | — | Fourth input tensor to concatenate (optional). | optional |
| `f` | `F` | `T` | — | — | Fifth input tensor to concatenate (optional). | optional |
| `g` | `G` | `T` | — | — | Sixth input tensor to concatenate (optional). | optional |
| `h` | `H` | `T` | — | — | Seventh input tensor to concatenate (optional). | optional |
| `i` | `I` | `T` | — | — | Eighth input tensor to concatenate (optional). | optional |
| `j` | `J` | `T` | — | — | Ninth input tensor to concatenate (optional). | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `c` | `concat_result` | `T` | same as `a` | derived | Concatenated output tensor. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | — | Required axis along which to concatenate; negative values count from the back, and the accepted range is `[-r, r-1]` where `r` is the rank of the inputs. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool`, `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`concat-last-axis-vec4-tail.wgsl.jinja`](build/webgpu/concat-last-axis-vec4-tail.wgsl.jinja)
- [`concat-n.wgsl.jinja`](build/webgpu/concat-n.wgsl.jinja)
- [`datamove-concat-block.wgsl.jinja`](build/webgpu/datamove-concat-block.wgsl.jinja)
- [`datamove-concat-last-axis-cooperative.wgsl.jinja`](build/webgpu/datamove-concat-last-axis-cooperative.wgsl.jinja)
- [`datamove-flat-copy.wgsl.jinja`](build/webgpu/datamove-flat-copy.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Concat", { version: 1 });
const { c } = await kernel({ a: { data: aData, shape: [1] } }, {
  attrs: { axis: 0 },
});
```
