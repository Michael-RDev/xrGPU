# Adding a chip or headset

Quest 3 is the first target. New hardware joins through an explicit device
profile, with independent evidence; it does not replace or broaden the Quest 3
match. Keep the current profile usable while developing another one.

1. Inspect the new hardware using stock execution. Record Android model, GPU
   vendor/device IDs, Vulkan capabilities, OS/driver, memory, and available VR
   telemetry. Confirm that the runtime and model execute before tuning kernels.
2. Add `devices/<device>/profile.h`. Define an exact, non-overlapping capability
   match using `DeviceCapabilities`, then bind the profile to an implementation.
   Extend capability fields when the new chip requires additional feature checks;
   do not infer compatibility from a vendor ID alone.
3. Add an implementation under `kernels/<operation>/<backend>/<implementation>/`
   when needed. Preserve the shared contract/reference. For the existing Vulkan
   linear adapter, supply a `LinearKernelSet` with a distinct ID, workload
   predicate, variant table, and shader prefix. Use unique shader filenames.
   The current shader ABI uses row-tile, bias, storage, weight-storage, and dtype
   suffixes; another shader ABI or operation needs an adapter and build checks.
4. Register the profile in `devices/registry.h`. The app discovers variant IDs
   and labels from the matching profile, and native loading rejects variants
   outside that profile. Merely adding files does not enable dispatch.
5. Extend shared layer fixtures for new shapes and precisions as necessary.
   Exported fixtures describe inputs/outputs rather than asserting universal
   candidate eligibility. Add host checks for the new profile and stock fallback,
   then run the complete [device validation protocol](validation.md).
6. Enable a variant only through accepted local results. Reports include the
   profile, kernel set, GPU identity, driver, workload, and source revisions.
   Selections live under `selections/<profile>/`; evidence from another profile
   never authorizes the new one.

An existing implementation can be referenced by a second profile after its
requirements are confirmed. Correctness, performance, sustained thermals, and VR
responsiveness must still be measured separately. Keep raw reports local under
`artifacts/device-results/<profile>/` when exporting them for analysis.

The current application/build targets Android arm64 and ExecuTorch Vulkan.
Supporting a different OS or execution backend requires its own integration;
the folder structure does not imply that those paths already work.
