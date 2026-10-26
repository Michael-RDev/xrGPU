---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.DFT

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 20

## Description

Computes the discrete Fourier Transform (DFT) of the input along a specified axis. For a signal of length N, output bin k is `sum_{n=0}^{N-1} exp(-2*pi*j*k*n/N) * x[n]`; the inverse divides by N and negates the exponent sign. Supports forward/inverse, real-to-complex (RFFT), and complex-to-real (IRFFT) modes via the `onesided` and `inverse` attributes.

See the [ONNX `DFT` spec](https://onnx.ai/onnx/operators/onnx__DFT.html) for the reference semantics.

## Inputs

| Name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `input` | `T` | same as logical dtype | — | — | Input signal tensor; the last dimension is 1 for real values or 2 for complex `(real, imaginary)` pairs. | required |
| `dft_length` | `L` | same as logical dtype | `0` | — | Optional int32 scalar controlling the signal length used for the transform; input is zero-padded or truncated to this length. | optional |
| `axis` | `I` | `int32` | `0` | — | Optional logical int64 scalar specifying the dimension over which to compute the DFT (the last axis is reserved for the real/imaginary component and is excluded); signed axes use int32 WebGPU storage and default to `-2` when omitted. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | same as `input` | — | DFT result tensor; last dimension is 2 (complex) for forward DFT and RFFT, or 1 (real) for IRFFT. | required |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `inverse` | `0` | When set to 1, computes the inverse DFT (IDFT/IRFFT) instead of the forward transform; default is 0 (forward). |
| `onesided` | `0` | When set to 1, exploits conjugate symmetry to return only the non-redundant half of the spectrum (RFFT for forward, IRFFT for inverse); default is 0 (full spectrum). |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16` |
| `L` | `int32` |
| `I` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `batch_axis1_fft_real_residual_tiles` — Real FFT over residual tiles, followed by a paired-root gather. Tiles use half the reported shared memory, and full capacity when needed to keep gather work logarithmic.
- `batch_axis1_fft_complex_residual_tiles` — Full complex residual FFT over device-sized residue-class tiles and a paired-root gather. A transform-stage work budget limits extra splitting when batches already supply parallel tiles.
- `batch_axis1_fft_real_residual_storage` — Real forward FFT with paired roots and residual components retained in global scratch. One workgroup owns each half-length packed transform, so storage barriers preserve stage order beyond the shared-memory capacity band.
- `batch_axis1_fft_real_packed` — Real forward FFT packed into a half-length complex residual transform. A shared paired-root table preserves cancellation accuracy across stages; full and one-sided output avoid a global transform scratch buffer.
- `batch_axis1_fft_complex_residual_storage` — Complex forward FFT retains paired roots and high/low components in disjoint per-transform storage rows beyond the shared-memory limit.
- `batch_axis1_mixed_fft` — Compensated mixed-factor FFT for real forward rows that fit shared memory. Smooth tails use staged small-radix butterflies; remaining tails use direct roots. Preserves full and one-sided spectra.
- `batch_axis1_fft_storage_inplace` — Inverse transforms beyond shared-memory capacity use an in-place full-spectrum output buffer. Forward transforms retain residual components in their precision-preserving paths.
- `contiguous_naive` — Direct DFT over the default axis of a contiguous rank-2 signal or a batch of rank-3 signals, for lengths no FFT route admits.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`dft-contiguous-naive.wgsl.jinja`](build/webgpu/dft-contiguous-naive.wgsl.jinja)
- [`dft-general.wgsl.jinja`](build/webgpu/dft-general.wgsl.jinja)
- [`dft-rank4.wgsl.jinja`](build/webgpu/dft-rank4.wgsl.jinja)
- [`dft-residual-gather.wgsl.jinja`](build/webgpu/dft-residual-gather.wgsl.jinja)
- [`dft-residual-twiddles.wgsl.jinja`](build/webgpu/dft-residual-twiddles.wgsl.jinja)
- [`dft-runtime-axis-fft-shared.wgsl.jinja`](build/webgpu/dft-runtime-axis-fft-shared.wgsl.jinja)
- [`dft-tiled-real.wgsl.jinja`](build/webgpu/dft-tiled-real.wgsl.jinja)
- [`fft-radix2-dit-storage.wgsl.jinja`](build/webgpu/fft-radix2-dit-storage.wgsl.jinja)
- [`fft-residual-shared.wgsl.jinja`](build/webgpu/fft-residual-shared.wgsl.jinja)
- [`fft-stockham-shared.wgsl.jinja`](build/webgpu/fft-stockham-shared.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `output`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.DFT", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { output } = await kernel({ input: { data: inputData, shape: [1, 1, 1, 1, 1, 1, 2, 1] } }, {
  outputs: { output: { shape: [1, 1, 1, 1, 1, 1, 2, 2], dtype: "float32" } },
});
```
