---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Min

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 13

## Description

Computes the elementwise minimum across one or more input tensors with NumPy-style multidirectional broadcasting. All inputs and the output share the same data type.

See the [ONNX `Min` spec](https://onnx.ai/onnx/operators/onnx__Min.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `a` | `A` | `T` | — | — | First input tensor. | required |
| `b` | `B` | `T` | — | — | Second input tensor; broadcast-compatible with the other inputs. | optional |
| `c` | `C` | `T` | — | — | Third input tensor; broadcast-compatible with the other inputs. | optional |
| `d` | `D` | `T` | — | — | Fourth input tensor; broadcast-compatible with the other inputs. | optional |
| `e` | `E` | `T` | — | — | Fifth input tensor, broadcast-compatible with all other inputs. | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `y` | `min` | `T` | derived | derived | Elementwise minimum of all input tensors, broadcast to the output shape. | required |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `int32`, `uint32`, `int16`, `int8`, `uint8` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`datamove-elementwise-copy.wgsl.jinja`](build/webgpu/datamove-elementwise-copy.wgsl.jinja)
- [`minmax-broadcast.wgsl.jinja`](build/webgpu/minmax-broadcast.wgsl.jinja)
- [`minmax-vec4.wgsl.jinja`](build/webgpu/minmax-vec4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.Min", { version: 1 });
const { y } = await kernel({ a: { data: aData, shape: [1, 3] } });
```
