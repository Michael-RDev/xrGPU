# Architecture

```text
Python export -> pinned .pte programs + tokenizer + references
                         |
                    offline APK
                         |
                 Android benchmark worker
                         |
                    ExecuTorch graph
                         |
             stock / candidate Vulkan dispatch
                         |
                    Adreno GPU

OpenXR render thread ----+---- shared GPU resources
                         |
           local evidence -> acceptance -> local selection
```

## Kernels and integration

`kernels/linear/manifest.json` specifies `Y = X @ W.T + bias`. The GLSL template
derives from ExecuTorch's BSD-licensed linear implementation and reuses its
packing, boundary checks, storage conventions, and FP32 accumulation. Candidates
compute eight adjacent output channels per invocation, reusing an input tile.
The variants change row tile size, workgroup shape, and packed-weight storage.

`runtime/patches/0001-xrgpu-linear.patch` routes only the two MiniLM feed-forward
projection shapes to these candidates. Other shapes and all other dtypes use
upstream dispatch. Kernels remain inside the original delegate; no layer result
is copied back to the CPU to invoke an external shader.

Candidate selection is explicit for diagnostics. The search demo uses a candidate
only if a local accepted report matches the hardware/driver, OS, runtime/kernel
revision, model hash, precision, sequence length, and execution mode. Stock is
the default. Candidate selection is thread-local and set before loading each
program; a loaded model is not shared across worker threads.

FP16, quantization, and bias/GELU fusion are not enabled in this first experiment.
Fusion is intentionally conditional on measured benefit; the current candidates
must be measured on device before adding another experimental factor.

## Model semantics

MiniLM uses its pinned BERT tokenizer, attention masking, masked mean pooling,
and L2 normalization. Pooling retains dimensions during reduction so export can
keep it in the Vulkan subgraph. The app returns a normalized 384-element vector.
Input lengths include special tokens; longer text is truncated to the selected
32/128/256-token program. There is no speech recognition component.

Golden tokenizer fixtures include empty input, whitespace, accents, Chinese,
special tokens, and truncation. The validation corpus is fixed before tuning.
Export saves the delegated graph for checking partition breaks and provenance.

## Rendering and telemetry

The native OpenXR activity renders 64 fixed cubes into two 1440×1440 swapchains
using GLES 3 while inference executes on Vulkan. Rendering and inference share
the physical GPU, but have separate contexts and threads. No real-world camera
or scene permission is needed. The app requests 72 Hz and checks the observed rate.

GPU operation timings come from ExecuTorch's Vulkan timestamp queries. Host
events use their own clock. These timelines are not subtracted from one another.
GPU totals sum dispatched operations and do not claim to equal critical-path
latency. Model and text-to-result wall times are reported separately. The runtime
is instrumented for both arms of an experiment.

Input/output staging copies and host submission-plus-wait have separately named
timings. Submission alone, exact device memory, and power are explicitly null:
the current backend does not expose those measurements independently. The kernel
revision also fingerprints native integration, app source, and build settings so
a change to the measurement implementation invalidates saved selections.

Frame time is the maximum of runtime-reported application CPU and GPU frame
time. Missed-frame rate uses the compositor's cumulative dropped-frame counter
delta divided by submitted-plus-dropped frames. The report names that metric;
it is not an estimate derived from how long `xrWaitFrame` blocks. Availability
depends on `XR_META_performance_metrics`. Missing counters cannot pass VR gates.

## Extending

Add operations with a contract, references, edge cases, and device benchmark
cases. Add headsets as independently validated device profiles. NPU and WebGPU
backends, remote fleets, and learned selectors are future work, not placeholder
implementations or advertised capabilities.
