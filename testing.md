# Quest testing checklist

Updated: 2026-10-26. Target: **Meta Quest 3 / Adreno 740**.

This checklist covers the tests that still require the physical Quest: on-device
inference, xrBench qualification, automatic xrGPU kernel selection, and sustained
XR behavior. All items are pending until actual device results are recorded.
Commands below run from the connected computer to drive tests on the headset.

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

- [ ] **Standalone native operators:** execute every compiled operation/type
  variant and applicable frozen upstream fixture on the Quest's native Vulkan
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

## Record results and finish

For each checked item, save test ID/configuration, pass/fail/blocked status,
expected versus actual result, device/driver identity, binary/package/bundle/rule
hashes, seed/session, logs, raw report path, and relevant `Session.info()` output.

- [ ] All four model configurations pass references through the final device APIs.
- [ ] All required operator, control, hold-out, model, and XR evidence is complete
  for every promoted rule; failed or skipped cases remain explicit.
- [ ] Final production automatic selection, fallback, lifecycle, and recovery
  tests pass on the intended Quest.
- [ ] Preserve original reports, campaign, shortlist, final rules, qualification
  record, and exact installed artifact inventory. Mark remaining gaps explicitly.
