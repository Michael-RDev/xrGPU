# xrGPU

Faster model layers for standalone XR. xrGPU is a native Quest 3/3S experiment
toolkit: profile a model, test kernel variants, integrate the winners, and measure
the complete model while a VR scene is rendering.

The first workload is `sentence-transformers/all-MiniLM-L6-v2`: local text search
over a small collection of object descriptions and commands. The app includes
the model and tokenizer and has **no Internet permission**.

## What is implemented

- Offline Android diagnostics and search UI, plus a native OpenXR stereo scene.
- Pinned MiniLM export at sequence lengths 32, 128, and 256, including reference
  embeddings, tokenizer fixtures, and real feed-forward layer weights.
- XNNPACK CPU and stock Vulkan baselines; three experimental FP32 linear
  variants integrated inside ExecuTorch's Vulkan delegate.
- On-device correctness, paired AB/BA benchmarks, sustained comparisons, GPU
  timestamp profiles, thermal status, process memory, and optional XR runtime
  frame counters.
- Local variant selection that requires matching, accepted evidence. Unvalidated
  workloads use stock kernels. Changing model, driver, runtime, kernel, or workload
  invalidates a selection.
- Host analysis and regression tests for exported CPU execution, tokenization,
  and acceptance logic.

**Device performance is unverified.** A built app or a faster microbenchmark is
not proof of a whole-model speedup. Quest 3 and Quest 3S must each pass the real
device protocol before they are described as validated. See [validation](docs/validation.md).

## Quick start

On macOS with Python/uv, Java 17+ and Git:

```sh
UV_CACHE_DIR=artifacts/cache/uv uv sync --extra export --python 3.12
.venv/bin/python -m runtime.prepare
.venv/bin/python -m runtime.bootstrap_android
source artifacts/toolchain/env.sh
.venv/bin/python -m models.export
.venv/bin/python -m runtime.build
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n org.xrgpu/.MainActivity
```

All dependencies, weights, and build products stay in the project environment or
the ignored `artifacts/` directory. Toolchain installation downloads several GB;
it does not change global packages or shell profiles. See [setup](docs/setup.md)
for an existing Android toolchain and result export.

## Repository

| Directory | Responsibility |
|---|---|
| `app/` | Android controls, tokenizer, correctness gates, local search |
| `kernels/` | Linear kernel contract, shader template, candidate variants |
| `benchmarks/` | Workload and acceptance settings |
| `runtime/` | Pinned dependency preparation, delegate patch, native code, builds |
| `models/` | Model export, offline corpus, reference fixtures |
| `analysis/` | Compare raw device evidence and produce local selections |
| `tests/` | Host numerical, tokenizer, and evidence-gate tests |
| `docs/` | Setup, architecture, measurement protocol |
| `artifacts/` | Ignored local weights, native builds, APK inputs, logs |

The model runtime is [ExecuTorch](https://github.com/pytorch/executorch).
Kernel packaging is inspired by [WebGPU Kernels and Fleet](https://huggingface.co/blog/webgpu-kernels).
Third-party licenses and provenance are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
