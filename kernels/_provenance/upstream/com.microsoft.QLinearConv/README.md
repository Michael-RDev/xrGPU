---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# com.microsoft.QLinearConv

`com.microsoft`  ·  ONNX Runtime contrib operator  ·  contrib since_version 1

## Description

Convolution on a quantized input with quantized weights, producing a quantized output, in either tensor layout. `channels_last = 0` reads `X` as `(N, C, D1..Dn)` and writes `Y` as `(N, M, D1..Dn)`; `channels_last = 1` reads `(N, D1..Dn, C)` and writes `(N, D1..Dn, M)`. The weight stays `(M, C/group, k1..kn)` in both layouts. Input and output quantization are per-tensor; weight quantization is per-tensor or per-output-channel. The optional int32 bias is pre-quantized with `scale = x_scale * w_scale` and zero point 0. Spatial ranks 1 to 3 are implemented.

See the [ONNX Runtime `QLinearConv` contrib-operator spec](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearConv) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `x` | `TX` | — | — | Quantized input tensor, `(N, C, D1 x ... x Dn)` when `channels_last` is 0 and `(N, D1 x ... x Dn, C)` when it is 1. | required |
| `x_scale` | `TF` | — | — | Per-tensor scale for input `x`. | required |
| `x_zero_point` | `TX` | — | — | Per-tensor zero point for input `x`. | required |
| `w` | `TW` | — | — | Quantized weight tensor shaped `(M, C/group, k1 x ... x kn)` in both layouts; `channels_last` never transposes the weight. | required |
| `w_scale` | `TF` | — | — | Scale for weights `w`; scalar for per-tensor or 1-D of length M for per-output-channel quantization. | required |
| `w_zero_point` | `TW` | — | — | Zero point for weights `w`; scalar or 1-D of length M matching `w_scale`. | required |
| `y_scale` | `TF` | — | — | Per-tensor scale for output `y`. | required |
| `y_zero_point` | `TY` | — | — | Per-tensor zero point for output `y`. | required |
| `B` | `int32` | `1` | — | Optional 1-D bias of length `M`, pre-quantized with scale `x_scale * w_scale` and zero point 0. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `y` | `TY` | same as `x` | derived | Quantized output tensor; the channel axis follows `channels_last` and the spatial sizes follow the kernel, strides, dilations and padding. | required |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `auto_pad` | `"NOTSET"` | Automatic padding mode. `NOTSET` uses `pads`; `SAME_UPPER` and `SAME_LOWER` choose padding so each output spatial size is `ceil(input / stride)`; `VALID` uses no padding. |
| `channels_last` | `0` | Tensor layout convention for `x` and `y`: `0` places the channel axis at index 1, `1` places it last. The weight layout is unaffected. Defaults to 0. |
| `dilations` | — | Optional dilation factors, one positive integer per spatial axis. Omission means all ones. |
| `group` | `1` | Number of groups that input and output channels are split into; defaults to 1. |
| `kernel_shape` | — | Optional kernel shape, one positive integer per spatial axis. When present, it must match the spatial dimensions of the weight tensor; omission infers the shape from the weights. |
| `pads` | — | Optional explicit padding in ONNX order `[begin_axis_0, ..., begin_axis_n, end_axis_0, ..., end_axis_n]`. Omission means all zeros; it cannot be combined with an automatic padding mode. |
| `strides` | — | Optional stride factors, one positive integer per spatial axis. Omission means all ones. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `TX` | `uint8`, `int8` |
| `TW` | `uint8`, `int8` |
| `TY` | `uint8`, `int8` |
| `TF` | `float32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `nchw2d_columns_x4` — Channel-first 2-D route without integer dot-product support: one invocation folds four adjacent output columns into a vec4 accumulator for up to four output channels, so each weight load serves sixteen products.
- `dp4a_pointwise_nchw` — Channel-first 1x1 route: the convolution is exactly one integer GEMM whose A operand is the weight and whose B operand is the activation image, so the shared packed integer dot-product core runs with no materialized column matrix.
- `matrix_pointwise_nchw` — Exact subgroup-matrix convolution using the same operand preparation. Bounded f32 partial dots preserve byte-integer products; int32 partial accumulation, optional bias and compensated requantization preserve output quantization. Small grids and short contractions use the packed integer kernels.
- `dp4a_im2col_nchw` — Channel-first route that materializes the column matrix in `(ic, kh, kw)` order and hands it to the shared packed integer dot-product core as the B operand, with the weight as A.
- `matrix_im2col_nchw` — Exact subgroup-matrix convolution using the same operand preparation. Bounded f32 partial dots preserve byte-integer products; int32 partial accumulation, optional bias and compensated requantization preserve output quantization. Small grids and short contractions use the packed integer kernels.
- `dp4a_im2col_nchw_bias` — Channel-first route that materializes the column matrix in `(ic, kh, kw)` order and hands it to the shared packed integer dot-product core as the B operand, with the weight as A. A pre-quantized int32 bias is added per output channel before requantization.
- `matrix_im2col_nchw_bias` — Exact subgroup-matrix convolution using the same operand preparation. Bounded f32 partial dots preserve byte-integer products; int32 partial accumulation, optional bias and compensated requantization preserve output quantization. Small grids and short contractions use the packed integer kernels.
- `dp4a_pointwise_nhwc` — Channels-last 1x1 route: the activation is already the GEMM's row-major A operand, so only the weight is reordered into K-major `(kh, kw, ic)` columns for the shared packed integer dot-product core, whose output rows land as the channels-last image.
- `matrix_pointwise_nhwc` — Exact subgroup-matrix convolution using the same operand preparation. Bounded f32 partial dots preserve byte-integer products; int32 partial accumulation, optional bias and compensated requantization preserve output quantization. Small grids and short contractions use the packed integer kernels.
- `dp4a_pointwise_nhwc_bias` — Channels-last 1x1 route: the activation is already the GEMM's row-major A operand, so only the weight is reordered into K-major `(kh, kw, ic)` columns for the shared packed integer dot-product core, whose output rows land as the channels-last image. A pre-quantized int32 bias is added per output channel before requantization.
- `matrix_pointwise_nhwc_bias` — Exact subgroup-matrix convolution using the same operand preparation. Bounded f32 partial dots preserve byte-integer products; int32 partial accumulation, optional bias and compensated requantization preserve output quantization. Small grids and short contractions use the packed integer kernels.
- `dp4a_im2col_nhwc` — Channels-last general route: the activation is already the GEMM's row-major A operand, so only the weight is reordered into K-major `(kh, kw, ic)` columns for the shared packed integer dot-product core, whose output rows land as the channels-last image.
- `matrix_im2col_nhwc` — Exact subgroup-matrix convolution using the same operand preparation. Bounded f32 partial dots preserve byte-integer products; int32 partial accumulation, optional bias and compensated requantization preserve output quantization. Small grids and short contractions use the packed integer kernels.
- `dp4a_im2col_nhwc_bias` — Channels-last general route: the activation is already the GEMM's row-major A operand, so only the weight is reordered into K-major `(kh, kw, ic)` columns for the shared packed integer dot-product core, whose output rows land as the channels-last image. A pre-quantized int32 bias is added per output channel before requantization.
- `matrix_im2col_nhwc_bias` — Exact subgroup-matrix convolution using the same operand preparation. Bounded f32 partial dots preserve byte-integer products; int32 partial accumulation, optional bias and compensated requantization preserve output quantization. Small grids and short contractions use the packed integer kernels.
- `precast_f16_pointwise_nchw_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_pointwise_nchw_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_pointwise_nchw_row` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_pointwise_nchw_row` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_im2col_nchw_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_im2col_nchw_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_im2col_nchw_row` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_im2col_nchw_row` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_im2col_nchw_bias_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_im2col_nchw_bias_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_im2col_nchw_bias_row` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_im2col_nchw_bias_row` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_pointwise_nhwc_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_pointwise_nhwc_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_pointwise_nhwc_column` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_pointwise_nhwc_column` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_pointwise_nhwc_bias_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_pointwise_nhwc_bias_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_pointwise_nhwc_bias_column` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_pointwise_nhwc_bias_column` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_im2col_nhwc_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_im2col_nhwc_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_im2col_nhwc_column` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_im2col_nhwc_column` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_im2col_nhwc_bias_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_im2col_nhwc_bias_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f16_im2col_nhwc_bias_column` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f16_im2col_nhwc_bias_column` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_pointwise_nchw_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_pointwise_nchw_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_pointwise_nchw_row` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_pointwise_nchw_row` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_im2col_nchw_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_im2col_nchw_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_im2col_nchw_row` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_im2col_nchw_row` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_im2col_nchw_bias_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_im2col_nchw_bias_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_im2col_nchw_bias_row` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_im2col_nchw_bias_row` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_pointwise_nhwc_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_pointwise_nhwc_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_pointwise_nhwc_column` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_pointwise_nhwc_column` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_pointwise_nhwc_bias_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_pointwise_nhwc_bias_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_pointwise_nhwc_bias_column` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_pointwise_nhwc_bias_column` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_im2col_nhwc_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_im2col_nhwc_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_im2col_nhwc_column` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_im2col_nhwc_column` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_im2col_nhwc_bias_tensor` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_im2col_nhwc_bias_tensor` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.
- `precast_f32_im2col_nhwc_bias_column` — Fuse spatial sampling, weight layout and exact byte centering into padded matrix operands. Both component tiers use bounded `f32` partial dots and `int32` totals before compensated requantization; `f16` exactly represents `[-255, 255]`. Tiles follow device limits, and small workloads use the other routes.
- `portable_f32_im2col_nhwc_bias_column` — Reuse fused exact byte preparation with a portable vector GEMM. Each workgroup computes a device-bounded tile with bounded exact f32 partial dots, int32 totals, optional per-channel bias and compensated requantization. Exact f16 operand storage is used when supported; f32 requires no optional arithmetic features.

## Device requirements

Some implementation variants require `subgroup-matrix`, `shader-f16`, and `subgroups`. These are route-specific capabilities, not package-wide requirements; availability also depends on the request shape and dtype.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`conv-int-accumulate-spatial.wgsl.jinja`](build/webgpu/conv-int-accumulate-spatial.wgsl.jinja)
- [`conv-int-im2col-spatial.wgsl.jinja`](build/webgpu/conv-int-im2col-spatial.wgsl.jinja)
- [`qlinear-conv-im2col-nhwc.wgsl.jinja`](build/webgpu/qlinear-conv-im2col-nhwc.wgsl.jinja)
- [`qlinear-conv-nchw-x4.wgsl.jinja`](build/webgpu/qlinear-conv-nchw-x4.wgsl.jinja)
- [`qlinear-conv-nhwc-oc4.wgsl.jinja`](build/webgpu/qlinear-conv-nhwc-oc4.wgsl.jinja)
- [`qlinear-conv-requantize.wgsl.jinja`](build/webgpu/qlinear-conv-requantize.wgsl.jinja)
- [`quant-dp4a-matmul.wgsl.jinja`](build/webgpu/quant-dp4a-matmul.wgsl.jinja)
- [`quant-exact-matrix.wgsl.jinja`](build/webgpu/quant-exact-matrix.wgsl.jinja)
- [`quant-exact-portable.wgsl.jinja`](build/webgpu/quant-exact-portable.wgsl.jinja)
- [`quant-exact-prepare.wgsl.jinja`](build/webgpu/quant-exact-prepare.wgsl.jinja)
- [`quant-weight-k-major.wgsl.jinja`](build/webgpu/quant-weight-k-major.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/com.microsoft.QLinearConv", { version: 1 });
const { y } = await kernel({
  x: { data: xData, shape: [1, 1, 1, 1] },
  x_scale: { data: x_scaleData, shape: [1] },
  x_zero_point: { data: x_zero_pointData, shape: [1] },
  w: { data: wData, shape: [2, 1, 1, 1] },
  w_scale: { data: w_scaleData, shape: [1] },
  w_zero_point: { data: w_zero_pointData, shape: [1] },
  y_scale: { data: y_scaleData, shape: [1] },
  y_zero_point: { data: y_zero_pointData, shape: [1] },
});
```
