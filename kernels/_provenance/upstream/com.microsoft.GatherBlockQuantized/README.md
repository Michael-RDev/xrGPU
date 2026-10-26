---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.GatherBlockQuantized

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Gathers rows from a block-wise quantized weight matrix and dequantizes them. This inference implementation supports the standard `gather_axis = 0`, `quantize_axis = 1` matrix subset with uint8 `data`, 4-bit packed or 8-bit values, rank-1 non-negative in-bounds int64 `indices` projected to uint32 WebGPU storage, and float32 scales/output. Higher-rank gathers, negative indices, int32 indices, int4/uint4 data, 2-bit data, float16/bfloat16 output, and non-default axes are not implemented.

See the [ONNX Runtime `GatherBlockQuantized` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GatherBlockQuantized) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dataT` | `data` | `T1` | runtime-selected; narrow integers and bool use 32-bit slots | `2` | — | Constant uint8 weight matrix. With `bits = 4`, each byte stores two values low-nibble first; with `bits = 8`, each byte stores one value. | required |
| `indicesT` | `indices` | `Tind` | `uint32` | `1` | — | Non-negative logical int64 indices selecting rows from axis 0 of `data`. Every index must be less than the row count; values use checked uint32 WebGPU storage. | required |
| `scalesT` | `scales` | `T2` | same as logical dtype | `2` | — | Per-block dequantization scale factors of shape `(rows, ceil(output_columns / block_size))`. | required |
| `zeroPointsT` | `zero_points` | `T1` | runtime-selected; narrow integers and bool use 32-bit slots | `2` | — | Optional uint8 zero points. At 4 bits two zero points are packed per byte along the quantized axis, low-nibble first; at 8 bits the shape matches `scales`. If absent, uint8 data uses 2^(bits-1). | optional |

## Outputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `outputT` | `output` | `T2` | `2` | derived | Dequantized floating-point output rows corresponding to the gathered indices. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `bits` | `4` | Bits per quantized value. The schema default is 4; this implementation supports 4 or 8. |
| `block_size` | `128` | Number of values sharing a scale. Defaults to 128 and must be a power of two at least 16. |
| `gather_axis` | `0` | Axis from which values are gathered. This matrix implementation supports the standard default, axis 0. |
| `quantize_axis` | `1` | Axis split into quantization blocks. This matrix implementation supports the standard default, axis 1. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T1` | `uint8` |
| `T2` | `float32` |
| `Tind` | `int64` |

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`gather-block-quantized-q4-pair.wgsl.jinja`](build/webgpu/gather-block-quantized-q4-pair.wgsl.jinja)
- [`gather-block-quantized-q8-vec4.wgsl.jinja`](build/webgpu/gather-block-quantized-q8-vec4.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.GatherBlockQuantized", { version: 1 });
const { outputT } = await kernel({
  dataT: { data: dataTData, shape: [4, 8] },
  indicesT: { data: indicesTData, shape: [2] },
  scalesT: { data: scalesTData, shape: [4, 1] },
});
```
