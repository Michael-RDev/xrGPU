---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.STFT

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 17

## Description

Computes the short-time Fourier transform (STFT) of a rank-3 real or complex signal by sliding a complete frame over the input and applying the DFT. The `frameStep` scalar represents the required positive `frame_step` input. At least one of `window` or `frameLength` is required; when both are supplied, their lengths must match. The output contains the real and imaginary components of every frequency bin for each complete frame.

See the [ONNX `STFT` spec](https://onnx.ai/onnx/operators/onnx__STFT.html) for the reference semantics.

## Inputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `signal` | `T` | `3` | — | Real input signal with shape `(batch_size, signal_length, 1)`, or complex input with shape `(batch_size, signal_length, 2)`. | required |
| `window` | `T` | `1` | — | Optional 1-D window tensor with shape `(window_length)` to multiply by each frame before the DFT. | optional |

## Outputs

| Name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- |
| `output` | `T` | `4` | derived | STFT result with shape `(batch_size, frames, dft_unique_bins, 2)` in one-sided mode or `(batch_size, frames, frame_length, 2)` in two-sided mode, with real and imaginary parts in the last dimension. `frames = floor((signal_length - frame_length) / frame_step) + 1`; partial trailing frames are not emitted. | required |

## Runtime arguments

| Name | Kind | Upstream attribute | Description | Presence |
| --- | --- | --- | --- | --- |
| `frameStep` | `i32` | `frame_step` | Positive number of signal positions between the starts of consecutive frames. | required |
| `frameLength` | `i32` | `frame_length` | Optional number of signal values in each frame; must equal the window length when both are supplied. | optional |

## Attributes

Default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `onesided` | `1` | When 1 (default), returns `floor(n_fft / 2) + 1` non-redundant frequency bins with indices `[0, ..., floor(n_fft / 2)]` for a real-valued input (RFFT); set to 0 to return the full spectrum. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `rank3_window_fft` — Compensated residual FFT for windows that fit shared memory: radix-2 for power-of-two lengths, mixed factors with a small-radix tail otherwise.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`dft-tiled-real.wgsl.jinja`](build/webgpu/dft-tiled-real.wgsl.jinja)
- [`fft-radix2-dit-shared-inplace.wgsl.jinja`](build/webgpu/fft-radix2-dit-shared-inplace.wgsl.jinja)
- [`fft-radix2-dit-storage-pass.wgsl.jinja`](build/webgpu/fft-radix2-dit-storage-pass.wgsl.jinja)
- [`fft-radix2-dit-storage.wgsl.jinja`](build/webgpu/fft-radix2-dit-storage.wgsl.jinja)
- [`fft-real-packed-stockham-shared.wgsl.jinja`](build/webgpu/fft-real-packed-stockham-shared.wgsl.jinja)
- [`fft-residual-shared.wgsl.jinja`](build/webgpu/fft-residual-shared.wgsl.jinja)
- [`fft-stockham-shared.wgsl.jinja`](build/webgpu/fft-stockham-shared.wgsl.jinja)
- [`fft-stockham-twiddles.wgsl.jinja`](build/webgpu/fft-stockham-twiddles.wgsl.jinja)
- [`stft-general.wgsl.jinja`](build/webgpu/stft-general.wgsl.jinja)

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

const kernel = await getKernel("webgpu-kernels/ai.onnx.STFT", { version: 1 });
const { output } = await kernel({
  signal: { data: signalData, shape: [1, 4, 1] },
  window: { data: windowData, shape: [4] },
  frameStep: 4,
});
```
