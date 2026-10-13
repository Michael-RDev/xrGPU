# Validation protocol

No headset performance numbers are bundled with the repository. Synthetic data
used in unit tests is only a test of rejection and acceptance logic.

## Build-host validation

The initial implementation was checked on Apple Silicon macOS:

- Android arm64 native libraries and all candidate shader variants compile.
- The signed offline debug APK builds; all 103 bundled files match their manifest
  hashes, the packaged native build matches the compiled library, and the manifest
  has no Internet permission. The local verification record is
  `artifacts/build-validation.json`.
- All 20 Python tests pass, including execution of the actual exported XNNPACK
  CPU programs against the PyTorch embeddings and normalization checks.
- The Java tokenizer matches all 69 Hugging Face fixtures; the Android Java
  sources compile and the on-device acceptance logic passes its host checks.
- The three exported model lengths each contain one Vulkan delegate. Twelve
  layer cases include six cases that must use stock fallback dispatch.

These checks validate the build and CPU path. No Quest was connected for GPU,
OpenXR, thermal, or whole-model speedup measurements.

## Correctness

- CPU exported programs are executed on the build host against PyTorch references.
- The Android tokenizer must match every bundled token ID and attention mask.
- All model outputs must have cosine similarity >= 0.999 to the FP32 reference.
- The demo's labeled nearest-neighbor results must remain unchanged.
- Layer tests compare real pretrained projection weights and both ordinary and
  odd-row and odd-channel inputs with `atol=1e-4`, `rtol=1e-4`. Non-finite output is a failure.
- Unsupported shapes must use stock dispatch. Shader compilation is necessary,
  but is not proof of on-device numerical correctness.

## Timing

The primary workload is batch 1, 128 tokens, FP32. Short and long contexts are
separate workloads and receive separate selection entries. XNNPACK CPU execution
is a secondary reference; the speedup gate compares stock Vulkan with candidate
Vulkan at the same precision and with identical instrumentation.

Each trial validates outputs, records cold load separately, warms up 20 times,
then collects at least 100 samples. Inference starts at most once every 100 ms;
overruns do not trigger catch-up bursts. Four paired trials alternate AB and BA.
VR comparisons keep scene, resolution, refresh rate, and inference cadence fixed.

Sustained validation alternates ten blocks of at least 60 seconds, five paired
comparisons in total. It must use the same device and software/workload identity.
Both short and sustained results must pass. A thermal transition invalidates an
affected trial rather than quietly attributing the transition to the kernel.
Warm the headset to stable conditions and rerun if necessary; do not tune with
the final held-out correctness corpus.

## Acceptance

A candidate is eligible only when every comparison is complete and correct,
custom dispatch is observed, the median end-to-end latency reduction is at least
10%, p95 model latency increases by at most 5%, and the speedup repeats in every
pair. VR additionally requires p95 frame time within 5% and missed-frame rate
within one percentage point of baseline. Missing telemetry, incomplete runs,
changed software, and mixed candidates cannot establish a pass.

The range of per-trial speedups is reported alongside the aggregate. It is not
a statistical confidence interval. Collect more independent sessions before
making general claims. Record headset model, OS/driver versions, charging state,
room conditions, and other active applications when sharing results manually.

The first milestone remains open until a real Quest completes this protocol.
Quest 3 results do not automatically validate Quest 3S.
