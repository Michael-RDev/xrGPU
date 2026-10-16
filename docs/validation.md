# Validation protocol

The [2026-10-15 Quest 3 feasibility result](quest3-feasibility-2026-10-15.md)
records real 128-token measurements. Synthetic data used in unit tests is only
a test of rejection and acceptance logic.

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

### Windows host validation, 2026-10-15

- Native Android arm64 libraries and the candidate shader catalog compile on Windows.
- The signed debug APK builds; all 103 bundled asset hashes, packaged native
  fingerprints, debug command activity, and absence of Internet permission pass
  `python -m runtime.verify_apk`. The record is `artifacts/build-validation.json`.
- All 54 Python tests pass, including actual exported CPU inference, native
  device/dispatch checks, CRLF shader preprocessing, and diagnostic failure handling.
- Android Java sources compile, including the debug-only activity; all 69 tokenizer
  fixtures and host acceptance checks pass.
- The authorized Quest 3 completed baseline, layer, and paired model checks for
  all three candidates. Candidate 2 repeated a 2.83% and 2.60% median end-to-end
  reduction in two sessions. See the linked result for raw-evidence locations,
  p95 latency, dispatch checks, exclusions, and profiling. This is a smaller
  repeatable gain, not full acceptance.

## Correctness

### Short-run feasibility diagnostics

The Windows/ADB workflow is `python -m runtime.device_benchmark --length 128`.
The debug-only command activity reuses the normal benchmark runner, preserving
precision, model semantics, cadence, warmups, and paired trial counts. It does not
run sustained or VR comparisons. The host checks both model and end-to-end p95
separately, rejects incomplete/confounded evidence, and repeats the best valid
candidate in a fresh app process. Thermal/charging-confounded comparisons receive
at most two retries after stable conditions return.

`python -m analysis.report RUN.json --feasibility` analyzes one short report.
It cannot be combined with `--selection`; full acceptance still requires the
matching sustained report. A strong preliminary result requires at least 10%
median end-to-end reduction in both sessions, improvement in every pair, and
neither model nor end-to-end p95 worsening by more than 5%. Smaller repeatable
gains, no demonstrated gain, and inconclusive runs are reported distinctly.

Individual commands are available with `--command inspect|correctness|baseline|layers|paired`,
`--variant ID`, and optional `--serial SERIAL`. Each request has a unique run ID;
the host retrieves that run's files rather than whichever result happens to be
newest. Completed trial data and periodic current-trial snapshots survive failures.
Keeping the headset awake and the diagnostic activity focused is required.

Use `--keep-awake` to apply Meta's temporary developer proximity override when
testing with the headset on a desk. The host returns control to the physical wear
sensor in its exit handler and records restoration in `power-control.json`.
After a host crash or USB disconnect, restore it manually with
`adb -s SERIAL shell am broadcast -a com.oculus.vrpowermanager.automation_disable`.
This follows [Meta's device test instructions](https://developers.meta.com/vr/documentation/unity/meta-xr-operator/quest/).

These measurements establish feasibility for the tested workload only. Summed GPU
operation times are not critical-path or end-to-end latency measurements.

### Numerical checks

- CPU exported programs are executed on the build host against PyTorch references.
- The Android tokenizer must match every bundled token ID and attention mask.
- All model outputs must have cosine similarity >= 0.999 to the FP32 reference.
- The demo's labeled nearest-neighbor results must remain unchanged.
- Layer tests compare real pretrained projection weights and both ordinary and
  odd-row and odd-channel inputs with `atol=1e-4`, `rtol=1e-4`. Non-finite output is a failure.
- Unsupported shapes must use stock dispatch. Shader compilation is necessary,
  but is not proof of on-device numerical correctness.
- Host dispatch checks cover profile matching, unknown-device fallback, shape
  limits, and a synthetic unregistered profile with different tuning and variant
  IDs. This fixture tests extensibility; it is not another supported device.

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

Profile ID, kernel-set ID, available variant IDs, GPU vendor/device/API identity,
and driver are part of the evidence identity. Mixed profiles or kernel sets and
legacy reports without those fields cannot enable a selection, including when
the hardware happens to share a GPU vendor. Each new device starts with its own
correctness and performance measurements.

The range of per-trial speedups is reported alongside the aggregate. It is not
a statistical confidence interval. Collect more independent sessions before
making general claims. Record headset model, OS/driver versions, charging state,
room conditions, and other active applications when sharing results manually.

The first milestone remains open until a real Quest 3 completes this protocol.
Current candidate kernels are restricted to Quest 3. Adding Quest 3S or another
chip requires a separate device profile and the complete validation protocol;
Quest 3 results cannot be reused as evidence for that device.
