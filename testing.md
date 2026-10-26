# Quest testing checklist

Updated: 2026-10-26. Target: **Meta Quest 3 / Adreno 740**.

This checklist covers the tests that still require the physical Quest: on-device
inference, xrBench qualification, automatic xrGPU kernel selection, and sustained
XR behavior. All items are pending until actual device results are recorded.
Commands below run from the connected computer to drive tests on the headset.

Section 11 separately records the native catalog software baseline and the
remaining conformance/API/xrBench integration checks. These local checks can
continue while hardware is unavailable; software passes do not check off any
physical-device gate. This documentation update did not rerun tests.

Use the [xrBench qualification runbook](../xrBench/docs/model-qualification.md)
for campaign preparation and the [integration workflow](docs/automatic-integration.md)
for building the final qualified library. The current frozen release is
**unqualified**; use the development core for candidate measurements and the final
production core for automatic-selection acceptance.

## 1. Connect, install, and identify the Quest

- [ ] Authorize ADB and select the intended headset by serial. Confirm that an
  unauthorized, disconnected, or wrong device produces a clear error.
- [ ] Install and launch the matching xrBench benchmark app, qualification APK,
  xrGPU demo, and an Android consumer of the xrGPU AAR.
- [ ] Confirm native ARM64 libraries load without missing JNI symbols or shared
  libraries. Test both debug and release/shrunk consumer applications.
- [ ] Probe the actual device model, GPU, vendor/device IDs, encoded driver, and
  Vulkan API version. Save the Quest OS and OpenXR runtime versions too.
- [ ] Record the installed APK/native-library hashes, kernel package digest,
  model bundle identities, and qualification digest when present.
- [ ] Verify sufficient storage and app-readable bundle paths. Test app-private
  bundle staging through `run-as`, exact staged bytes, and SELinux/permission errors.
- [ ] Confirm actual battery, charging, thermal, focus, refresh-rate, frame-time,
  and missed-frame telemetry is available. Use focused **72 Hz** for XR qualification.

From xrBench:

```sh
adb devices -l
adb -s DEVICE_SERIAL install -r \
  engine/.build/qualification/xrbench-qualification-debug.apk
.venv/bin/python -m engine.qualification campaign probe \
  --serial DEVICE_SERIAL --output engine/.state/quest3-target.json
```

Use the exact package and bundle inventories in the campaign. Changing the
package, runtime, target driver, or relevant source requires matching new evidence.

## 2. Run all four model configurations on the Quest

Test each configuration through both the public **C++ Session** and
**Java/JNI Session** APIs. Run its CPU path and its candidate/qualified Vulkan
path, checking the backend actually used in `Session.info()`.

| Configuration | Inputs | Expected output |
| --- | --- | --- |
| MiniLM, length 32 | Three INT64 tensors `[1,32]` | FP32 embedding `[1,384]` |
| MiniLM, length 128 | Three INT64 tensors `[1,128]` | FP32 embedding `[1,384]` |
| MiniLM, length 256 | Three INT64 tensors `[1,256]` | FP32 embedding `[1,384]` |
| MobileNetV2 | FP32 RGB image `[1,3,224,224]` | FP32 logits `[1,1000]` |

- [ ] Run all six reference fixtures for each configuration: **24 fixtures per
  tested execution path**. Compare device outputs to the hashed references.
- [ ] Check finite FP32 outputs within `1e-4 + 1e-4*abs(reference)` and exact
  INT64 outputs, using any stricter declared workload comparison policy.
- [ ] Check MiniLM input ordering (`input_ids`, `attention_mask`, `token_type_ids`),
  tokenization, padding/truncation, masked pooling, and normalized embeddings.
- [ ] Check MobileNet RGB ordering, resize/crop, channel normalization, and logits.
  Include real text/image application inputs as a preprocessing smoke test.
- [ ] Confirm every reported Vulkan run actually executes on the GPU. CPU
  fallback results must be visible and cannot count as GPU qualification evidence.
- [ ] Compare shell runner and qualification APK results for the same fixture;
  verify both use the intended shared inference core and package.

Use discovery seeds `20261025`, `20261026`, `20261027`; reserve held-out seeds
`20261125`, `20261126`, `20261127` for confirmation after freezing the shortlist.

## 3. Exercise the Android/C++/JNI Session lifecycle

Use device acceptance fixtures in addition to the pretrained models where needed.

- [ ] FP32 and INT64 tensors, multiple inputs/outputs, exact shape/type/byte-count
  validation, and copying of noncontiguous outputs.
- [ ] Invalid/missing/extra inputs and wrong types/shapes return useful errors.
- [ ] Outputs remain correct after another `forward`, input mutation, and Session
  close; Java/C++ outputs own their storage.
- [ ] Repeated load/forward/close, repeated close, and calls after close behave
  safely and release resources.
- [ ] Multiple models and independent Sessions do not share selection/context state.
- [ ] Simultaneous first loads and concurrent forwards work; calls on the same
  Session serialize safely.
- [ ] A failed Vulkan load can load and execute the valid CPU program, with
  correct outputs and a visible fallback reason.
- [ ] Corrupt bundles, changed file hashes, incompatible runtimes, missing programs,
  and invalid tensor contracts fail clearly without crashing the application.
- [ ] `info()` reports actual backend/device, package/qualification identity,
  active decisions, attempted Vulkan decisions, traces, and fallback reasons.
- [ ] Repeated runs and model switches show no growing native/GPU memory usage,
  leaked handles, deadlocks, or stale outputs.

## 4. Validate GPU operators and screen candidates

- [ ] Run the entire regression inventory on the Quest GPU. The current campaign
  requires **93 regression cases**: the original 80 plus 13 semantic/packing/tail
  cases. Verify coverage against the campaign inventory.
- [ ] Cover GELU semantics, grouped/depthwise convolution packing, linear tails,
  odd dimensions, bias/no-bias, and all declared shapes/layouts/types/attributes.
- [ ] Screen all **18 linear tile/workgroup/storage variants** for every declared
  exact request using all three discovery fixtures. Retain failures too.
- [ ] Freeze at most **three finalists per exact request** before examining
  held-out results. Preserve the immutable shortlist and discovery session IDs.
- [ ] Confirm finalists with all three held-out seeds in at least **three fresh
  independent hardware sessions**. Do not reuse discovery sessions.
- [ ] Apply the same necessary correctness repairs to baseline and comparison.
  Check outputs before and after every timing arm.

Example discovery run, from xrBench:

```sh
.venv/bin/python -m engine.qualification campaign run \
  --campaign engine/.state/quest3-campaign.json \
  --runner engine/.build/qualification/xrgpu-qualification-runner \
  --serial DEVICE_SERIAL --stage screen \
  --workload operator_qualification_linear_tails_bias1 \
  --variant m1_wg8x4_texture2d --seed 20261025 --session discovery-1
```

Repeat for the actual workload/variant/seed inventory. All non-screen runs need
`--shortlist`; `correctness`, `model`, and `sustained` additionally need `--rules`
for the final frozen candidate set.

## 5. Prove that measurements and dispatch selection are correct

Run `confirm`, `noise`, `incorrect`, `slower`, and separate `trace` stages for
the finalists on held-out fixtures and fresh sessions.

- [ ] Ordinary latency runs contain at least **10 balanced randomized pairs**,
  **20 warmups**, and **100 samples per arm**. Record stable thermal and charging
  conditions and correctness checks before/after each arm.
- [ ] Noise control: identical repaired-stock programs show absolute apparent
  improvement below **5%**.
- [ ] Incorrect-output control: the real perturbed candidate output is rejected
  by numerical validation.
- [ ] Slower control: the deliberately slower candidate has a bootstrap CI upper
  bound below **−5%** improvement.
- [ ] Dispatch trace identifies the intended replacement name, exact shader SHA,
  dispatch count, global dimensions, and local workgroup. Shader hash alone cannot
  distinguish workgroup variants that share the same SPIR-V.
- [ ] Repeated equivalent graph nodes select correctly; stock dispatches remain
  visible. Trace collection is separate from latency measurement.
- [ ] Confirmed invocation improvement is at least **5%**, with a **95% paired
  bootstrap CI lower bound above zero** and **p95 regression no greater than 5%**.
- [ ] Freeze the confirmed rules using `campaign candidates`, then run the full
  regression inventory at `--stage correctness --rules FINAL_RULES`. Every report
  must bind to that exact final rule digest.

## 6. Measure complete models on the Quest

- [ ] Run `--stage model --rules FINAL_RULES` for every configuration in section 2,
  covering all held-out seeds and at least three fresh sessions.
- [ ] All model outputs pass reference checks with the final exact rules.
- [ ] Verify actual GPU execution and intended kernel decisions; repaired-stock
  runs or CPU fallback cannot substitute for candidate model results.
- [ ] Every configuration stays within **5% median and p95 latency regression**.
- [ ] At least one configuration has a repeatable **10% or greater median model
  improvement** in confirmation sessions.
- [ ] Record invocation and application costs with their measurement scope,
  including staging/layout/conversion costs when judging application benefit.

## 7. Run sustained inference alongside real XR rendering

Use `--stage sustained --rules FINAL_RULES`; this selects the qualification
OpenXR activity automatically. Supply the campaign, runner, serial, shortlist,
model workload, held-out seed, and fresh session as for other stages.

- [ ] Test **all four model configurations**, covering all held-out seeds and
  at least three fresh sessions with the final candidate rule digest.
- [ ] Each report has one baseline and one comparison arm in randomized order,
  each lasting at least **600 seconds**. These are separate from the ordinary
  10-pair latency measurements.
- [ ] Keep the headset focused at **72 Hz** and confirm actual measured arm
  duration, raw frame times, and compositor missed-frame counters.
- [ ] Frame p95 regression is **no greater than 5%**.
- [ ] Missed-frame-rate increase is **no greater than one percentage point**.
- [ ] Record battery/charging and thermal behavior throughout both arms.
- [ ] Focus loss, unsupported refresh/telemetry, severe thermal state, charging
  changes, or CPU fallback stop/invalidate qualification rather than producing
  passing evidence.
- [ ] Watch for stutter, rendering disruption, crashes, resource growth, and
  degradation during repeated sustained runs. Keep diagnostics separate from
  formal timing samples.

## 8. Test interruptions, errors, and recovery on the Quest

- [ ] Disconnect/reconnect USB during execution and after device completion.
- [ ] Interrupt the host controller; test timeout, failed launch, app/process
  termination, and missing/unreadable fixture files.
- [ ] Remove headset focus and change refresh or power conditions; verify clear
  diagnostic status and rejection of invalid qualification evidence.
- [ ] Recover a completed report without launching the workload again. Its
  original terminal bytes must be unchanged.
- [ ] Partial/failed runs remain diagnostic; duplicate, mixed-device, stale-package,
  wrong-run, or mismatched-rule reports cannot qualify a replacement.
- [ ] Reconnect/relaunch after failure and complete a fresh run with correct
  outputs and no stale Session, job, or transport state.

From xrBench, using the transport file saved by the interrupted run:

```sh
.venv/bin/python -m engine.qualification campaign recover \
  TRANSPORT_JSON --output RECOVERED_DIRECTORY
```

## 9. Verify final automatic integration in a production Quest app

After xrBench assesses the original device evidence and exports an eligible
qualification record, rebuild the production library with that record and install
its consumer app. Follow the linked integration runbook for the build commands.

- [ ] Run all four models through public C++ and Java APIs with the final
  production AAR/native library and **no experiment overrides**.
- [ ] Exact matching device/GPU/driver/runtime/package and operation descriptors
  select only their qualified kernels automatically.
- [ ] Confirm actual dispatched shader/geometry, numerical outputs, and `info()`
  decisions match the exported eligibility record.
- [ ] Nonmatching operation descriptors use eligible stock behavior. A missing
  required correctness repair causes whole-model CPU fallback with a reason.
- [ ] Unqualified/pending records, mismatched target/driver/package/runtime, and
  missing Vulkan never grant replacement eligibility. Use controlled test builds
  or a different actual target where needed; record how each path was exercised.
- [ ] A package-less production library loads and executes its safe CPU path.
- [ ] Production rejects experiment/rule overrides, and bundle metadata cannot
  inject qualification or authorize a kernel.
- [ ] The reusable library still loads when Vulkan is unavailable and completes
  CPU inference. Corrupt/incompatible model bundles fail validation.
- [ ] Repeat lifecycle/concurrency tests with the final production app after a
  cold launch, process restart, and model switch.
- [ ] Run with network disabled; inference and selection remain local and offline.
- [ ] The existing xrGPU demo's text/image inputs, model selection, results,
  errors, and report export behave correctly on the headset.

Correctness repairs require the hardware, provenance, correctness, controls,
and holdout gates. Performance replacements also require invocation, whole-model,
and sustained XR gates. Only rules satisfying their required gates may be embedded
as eligible; pending evidence grants no selection.

## 10. Additional Quest GPU features, when included in the device test build

These are separate validation tracks from the model-inference package above.
Mark an unavailable test as blocked with its missing integration/runner identified.

- [ ] **Standalone native operators (detailed gates in section 11):** execute
  every compiled operation/type variant and applicable frozen upstream fixture
  on the Quest's native Vulkan
  backend. Record actual coverage, failures, unsupported cases, and oracle conflicts.
- [ ] Exercise broadcasting, scalar/empty/rank-one/batched tensors, zero-length
  contraction, dtype boundaries, FP16 rounding/subnormals/NaN/Inf/signed zero,
  integer overflow, and transcendental edge cases supported by each contract.
- [ ] Check prepare/upload/run/read/close, synchronization, owned output bytes,
  device/allocation limits, error cleanup, and memory growth on the Quest.
- [ ] Unsupported native-operator contracts/devices produce explicit errors;
  this standalone API must not silently fall back to CPU computation.
- [ ] **Fused-attention prototype:** run all three workgroups on short, masked,
  odd-shaped, and boundary fixtures; compare actual GPU outputs to references.
- [ ] Once graph integration exists, measure attention including conversion costs
  and repeat independent confirmation/model/XR validation before promotion.
- [ ] Keep physical correctness, performance, and sustained-XR results distinct
  for each package; a pass in one track does not qualify another.

## 11. Standalone native catalog: remaining test gates

See [native operators](kernels/_shared/README.md) and the
[extended typed API](kernels/_shared/extended/README.md). This library executes C++ and
Vulkan/SPIR-V without a browser runtime. Unsupported standalone operator requests
must fail explicitly without numerical CPU fallback; the model Session fallback
policy above is a different validation track.

The active native source authority is [`kernels/catalog.json`](kernels/catalog.json).
The 221 direct operator folders contain native descriptors and GLSL entrypoints;
`kernels/_shared/` contains shared C++/shader code, while `kernels/_provenance/`
retains unchanged upstream reference bytes. Run `.venv/bin/python -m kernels.catalog`
to validate descriptor digests, provenance, all shader dependencies and generated
registry consistency. Native entries are not model Session replacements; the
three existing model override contracts remain a separate kind and fingerprint.

The saved all-v3 baseline below predates this source-layout migration and remains
immutable. The fresh all-v4 package and direct checks are recorded separately.
New snapshots use `kernels._shared.*` and include an explicit `operators.*`
compatibility adapter. Select the tools actually delivered in the package when
setting `PYTHONPATH`, and use `python -B` to keep its inventory unchanged.

### Native layout migration validation — 2026-10-26

The migrated package is
[`native-kernels-all-v4`](artifacts/packages/native-kernels-all-v4/manifest.json),
SHA-256 `82291e879d9aa5db9c8fb2ae129e7aaebd79deae270815e237e61daec127b605`.
It has 221 operators, 315 freshly compiled and SPIR-V-validated shaders, both
Android ARM64 libraries, and 2,646 inventoried files. All 315 shader binaries,
21 C++ runtime/planner/header files, both public API headers, and 1,108 frozen
provenance files match all-v3 exactly. Previous packages still verify unchanged.

| Direct final-package software check | Passed | Failed |
| --- | ---: | ---: |
| Legacy operator/type representatives | 117 | 0 |
| Tensor representatives | 90 | 0 |
| Spatial representatives | 57 | 0 |
| Attention/recurrent/quantized representatives | 36 | 0 |
| Tensor edge cases | 54 | 0 |
| API/storage/dynamic-capacity checks | 12 | 0 |

The first four rows cover all **221 identities in 300 executions**; the total is
**366 passing executions**. Native software Vulkan runners were built from the
exact package without recompiling its shaders. The
[migration ledger](artifacts/conformance/native-kernels-all-v4/migration.json)
binds reports, package/source/runner/shader identities, prior releases and scope.
Package inventories verified unchanged after execution. No browser process or
JavaScript runtime participates; the optional software test loader/ICD is not
shipped in Android artifacts.

Catalog/provenance, discovery, model fingerprint isolation and optional Hub
export regression checks pass. A full run recorded 392 passes and four failures;
all four received passing focused follow-ups after stale migration tests were
updated. The additional destination/generated-header packaging suite passes
12/12 with external compilation tools mocked; this is separate from numerical
evidence. It reproduced the release-path bug before the fix. The Hub protocol
suite passes 33/33 and covers 221 operator exports plus the aggregate.

**Remaining gates:** complete 13,919-case conformance, recorded Unique/provider
and unsupported-route gaps, physical Quest 3 correctness/capabilities, performance
and sustained XR behavior. No device was connected for this validation. The
existing all-v3 failure/unsupported/reference-conflict evidence below remains
applicable; the migration does not close those gaps or enroll all operators in
model Session or xrBench's current three-operator fixture lane.

### Recorded native baseline — 2026-10-26

The immutable local package is
[`native-operators-all-v3`](artifacts/packages/native-operators-all-v3/manifest.json),
SHA-256 `f929c8ddf3aa440ed6610db96a79b1d1a0edce8c892793ae98153d08e4502bbb`.
It contains 38 legacy and 183 extended operator identities, 315 validated SPIR-V
binaries, both Android arm64 operator libraries, and 1,716 inventoried files.
Its qualification status remains **unqualified**.

| Direct v3 check | Saved result | Coverage boundary |
| --- | --- | --- |
| Legacy native API | 117/117 operator/type cases passed | 38 operator identities |
| Extended tensor operators | 90/90 representatives passed | One representative per identity |
| Extended spatial operators | 57/57 representatives passed | One representative per identity |
| Extended attention/recurrent/quantized operators | 36/36 representatives passed | One representative per identity |
| Tensor edge-case suite | 54/54 passed | Selected integer, precision, shape, control-flow and sorting cases |
| Core/API suite | 12/12 passed | Includes optional Unique output capacities and undersized-bound rejection |
| Empty Pad regression | Passed | Shape `[0,1048594,3]`, zero output bytes |
| Frozen NMS/QuantizeLinear subset | 156 passed, 1 failed, 18 unsupported | 175 cases; the failed Int16 quantization route is outside the frozen declared `uint8`/`int8` output mask and remains a non-pass |
| Compilation and package integrity | Passed | 315 SPIR-V validations, host build checks, Android libraries and file digests |

The representative runs cover **all 221 identities in 300 legacy-type/extended
smoke cases**. They do not establish all dtype/attribute/shape combinations or
completion of the **13,919 frozen correctness cases**. Android compilation is not
Android execution. No physical XR-device or performance qualification exists.
These are saved results; this checklist update did not rerun them.

Evidence:

- [Final evidence ledger](artifacts/conformance/native-operators-all-v3/v2-to-v3-transfer.json)
- [Legacy](artifacts/conformance/native-operators-all-v3/legacy-variants.json),
  [tensor](artifacts/conformance/native-operators-all-v3/tensor-representatives.json),
  [spatial](artifacts/conformance/native-operators-all-v3/spatial-representatives.json),
  [complex](artifacts/conformance/native-operators-all-v3/complex-representatives.json)
- [Tensor edge cases](artifacts/conformance/native-operators-all-v3/tensor-features.json),
  [core/dynamic capacities](artifacts/conformance/native-operators-all-v3/core-dynamic-capacity.json),
  [empty Pad](artifacts/conformance/native-operators-all-v3/empty-pad/report.json)
- [Frozen NMS/quantization results](artifacts/conformance/native-operators-all-v3/nms-quantize-frozen.json)
- [Unique gap analysis](artifacts/conformance/native-operators-all-v2/unique-gap-analysis.json)

The ledger separately transfers 104 spatial, four complex quantization/recurrent,
six GQA and four SparseAttention feature checks from v2 by verified component
identity. Those are **v2 executions**, not 118 additional direct v3 runs. The
104-case spatial suite includes its 57 representatives. Of 315 v2/v3 shader
binaries, 225 are byte-identical and 90 tensor binaries changed and were rerun.
The older 29 rotary development goldens are not transferable to the final
shader inventory. Preserve original report/package bindings and v1/v2 packages.

### Remaining contract conformance — priority 1

- [ ] Build a per-contract/per-case ledger for all 13,919 frozen cases, preserving
  source revisions, fixture hashes and original `skipGpu` annotations. Reassess
  each skip; unexecuted, unsupported, timed-out and reference-blocked cases do
  not count as parity passes.
- [ ] Implement the exact published fixture generators still unsupported by the
  harness, including `fillFloat32`; add independent references where expected
  outputs are absent. Use ONNX Runtime CPU where compatible and independent
  NumPy/PyTorch references otherwise. Do not guess generator semantics.
- [ ] Cover declared types, attributes/defaults and argument precedence,
  broadcasting, optional inputs/outputs, scalar versus empty shapes, irregular
  dimensions, zero-length contractions, overflow, signed zero, NaN/Inf and
  subnormal behavior. Check typed FP16 intermediate stages and quantization
  precision, not only final storage conversion.
- [ ] Resolve Unique's frozen NaN/provider behavior and rerun its full suite.
  The earlier 307-case audit recorded 242 passes, 60 failures, four reference
  conflicts and one resource gap. Its detailed analysis separates **35 observed
  native NaN/provider mismatches**, **23 oracle shape failures before GPU
  execution**, **two logical-shape versus padded-capacity differences**, and
  **four conflicts where native Y matches the published result but NumPy axis
  semantics differ**. These are different failure classes, not 60 proven shader
  errors or 60 bad goldens; they are not a fresh full v3 Unique run.
- [ ] Resolve the Int16 QuantizeLinear route explicitly: implement and test a
  documented native extension if required, or retain its unsupported status.
  Preserve the original failed fixture status; do not silently cast to an
  implemented output type or edit the frozen type mask.
- [ ] Resolve missing tolerance/accumulation-policy questions, including older
  strict-exact Gemm/MatMul comparisons, using independent evidence. Investigate
  published-reference conflicts separately. Compare integer/bool outputs
  exactly and floats with the original contract-specific policy; never loosen
  global tolerances to turn failures into passes.
- [ ] Add a dedicated negative-invocation suite. The current frozen harness
  leaves `expectError`/`expect_error` fixtures unsupported. Exercise invalid
  types/ranks/attributes, non-broadcastable shapes, bad controls, allocation and
  dispatch limits, and documented index behavior: Gather zero-fill differs from
  Scatter's explicit error contract.
- [ ] Expand resource-boundary cases beyond the configured fixture element cap;
  retain bounded preparation/execution timeouts and separate timeout from a
  numerical mismatch. Record every remaining limit and untested combination.

### Remaining API, state and failure checks — priority 2

- [ ] Expand optional-output combinations and dynamic-count tests: zero actual
  count, exact capacity, spare capacity, insufficient capacity, static-dimension
  rejection, and large counts. Read back actual GPU shapes and bytes without a
  CPU-computed result length. The general capacity-bound fix already passes.
- [ ] Cover logical Int64/physical UInt32 controls and outputs, full-width signed
  values, narrowing overflow rejection, raw copy/bitcast fidelity, all half
  encodings and concurrent byte/half stores. No silent type conversion.
- [ ] Verify prepare/run/read_outputs/info/close ownership, repeated execution,
  synchronization, cancellation, timeout/device-loss cleanup, use after close,
  multiple independent operators and concurrent preparation. Check resources
  and output validity after failures; no stale successful result may escape.
- [ ] Test multi-step attention, recurrent and causal-convolution state feedback:
  request changed outputs, read them, and supply them to a **new prepare**.
  Repeated `run()` uses the original prepared inputs and is not state feedback.
  Compare ragged lengths, growth/eviction, paging/CSR layouts, quantized caches,
  rotary/normalization options and retained padding against independent oracles.
- [ ] Verify the documented native cache ownership policy. Returned cache/state
  tensors are owned copies; caller host tensors are not mutated, and no aliased
  GPU-buffer ABI is exposed. Preserve this difference from published in-place
  cache contracts in compatibility reports.
- [ ] Recheck all pass dependencies and the actual loaded shader hashes after
  source/compiler/runtime changes. Rebuild the runner from the same package's
  source and generated hash headers; a live-checkout runner is not equivalent.

### Android/Quest 3 execution and performance — priority 3

- [ ] Connect an authorized Quest 3, record its actual Android/GPU/driver/Vulkan
  identities, and execute a production Android runner built against this exact
  native package. Include both legacy and extended APIs and all required passes.
- [ ] Run the full typed correctness matrix on the device against preserved
  references, starting small before larger/resource-boundary cases. Software
  Vulkan results do not establish Adreno behavior, including precision and
  subnormal handling.
- [ ] Probe device limits/capabilities before dispatch. Verify unsupported GPU
  features, type routes, memory/descriptor/dispatch limits and CPU Vulkan devices
  fail explicitly; Android production must reject software-test overrides.
- [ ] Measure preparation/pipeline compilation, upload, execution and readback
  separately. Use warmups, randomized paired comparisons and repeated sessions;
  measure cold preparation separately from resident repeated execution. No speedup
  follows from compilation, an operator count or a software-driver timing.
- [ ] Test sustained OpenXR coexistence, cancellation/focus changes, telemetry,
  memory/resource growth, thermal behavior and recovery. Apply the applicable
  device/performance/XR gates above before making those claims.

### xrBench extended-operator integration — separate gate

The existing `xrgpu_vulkan` fixture lane covers legacy Add/Relu/MatMul. It does
not prove all 221 operators are selectable or executable in xrBench.

- [ ] Add/validate the extended native adapter and separate typed fixture suite:
  attributes, logical/storage types, optional inputs, multiple outputs, dynamic
  shapes and cache/state outputs must survive the host/Android interface.
- [ ] Verify backend selection and explicit capability skips in the existing
  controls, cancellation and OpenXR measurement flow. Preserve current suites,
  model qualification and their independent acceptance policies.
- [ ] Embed/link the verified libraries and every required shader pass. Rebuild
  affected APKs, verify embedded hashes/provenance, and reject stale APKs.
- [ ] Check reports carry operator/contract identity, package revision, tensor
  descriptors, correctness, actual shader hashes, skip reasons and separately
  measured preparation, transfer/readback and execution costs. Run offline.

### Rerun and evidence requirements

- [ ] Validate the native catalog and check its generated registries; reject bad
  paths, symlinks, corrupt descriptors and provenance, incomplete pass inventories,
  and registry drift. Native catalog/provenance edits must not change the separate
  model override discovery set or its Session eligibility fingerprint.
- [ ] For the next revision, run working-tree host regressions covering native
  operators, half/unary routines, package provenance, Int64/storage conversion,
  quantized caches, spatial/complex references and frozen fixture validation.
  Run the planner/framework tests with `python -m kernels._shared.build --plan-only`.
- [ ] Build and validate every mapped SPIR-V dependency and both Android arm64
  libraries from an immutable source inventory. A runner built from the live
  checkout cannot qualify a different packaged snapshot.
- [ ] Rerun `kernels._shared.verify_local` and the extended tensor/spatial/complex
  representatives, core/dynamic-capacity, GQA, SparseAttention and Unique suites
  against the delivered bytes. Broaden frozen coverage with `verify_frozen`,
  `verify_spatial_published` and `kernels._shared.verify_upstream`; inspect unsupported
  and reference-conflict counts as well as the process exit code.
- [ ] Execute packaged Python references from outside the checkout with
  `PYTHONPATH=PACKAGE/source`, `PYTHONDONTWRITEBYTECODE=1` and `python -B`. Use a
  native software Vulkan loader/ICD only in the explicitly marked test build.
  Build host executables outside the package; keep Android production free of
  the software-test override. See the linked API instructions for build flags.
- [ ] Choose fresh evidence/output directories, preserve raw tensors and original
  reports, and record package/source/runner/reference/shader hashes and versions.
  Saved job JSON may point at old output paths; do not replay it unchanged.
  Verify the full package inventory before and after execution.
- [ ] Keep pass, failure, unsupported, oracle-conflict, timeout and unexecuted
  statuses distinct. Never add transferred v2 checks to direct v3 counts or count
  the 29 older rotary goldens as final-shader proof. Document every remaining gap.

## Record results and finish

For each checked item, save test ID/configuration, pass/fail/blocked status,
expected versus actual result, device/driver identity, binary/package/bundle/rule
hashes, seed/session, logs, raw report path, and relevant `Session.info()` output.

- [ ] All four model configurations pass references through the final device APIs.
- [ ] All required operator, control, hold-out, model, and XR evidence is complete
  for every promoted rule; failed or skipped cases remain explicit.
- [ ] Final production automatic selection, fallback, lifecycle, and recovery
  tests pass on the intended Quest.
- [ ] A standalone full-parity claim has per-contract/case conformance evidence;
  operator-identity smoke coverage is insufficient. Keep that status separate
  from model-rule eligibility and physical/performance/XR qualification.
- [ ] Preserve original reports, campaign, shortlist, final rules, qualification
  record, and exact installed artifact inventory. Mark remaining gaps explicitly.
