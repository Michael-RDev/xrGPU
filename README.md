# xrGPU

Native Android Vulkan kernels for XR devices, alongside reusable model inference
with device-specific replacements qualified through xrBench.

- [Automatic integration and qualification workflow](docs/automatic-integration.md)
- [Pinned model export examples](models/README.md)

## Native operator catalog

The authoritative [native catalog](kernels/catalog.json) contains 221 operator
identities under `kernels/<full-operator-ID>/`. Each entry has a native
`contract.json`, fixture/benchmark descriptors, and Vulkan GLSL entrypoints.
Shared C++ runtime and shader implementations live in `kernels/_shared/`.
Checksummed upstream reference metadata lives in `kernels/_provenance/`; it is
immutable audit data and does not supply an executable browser backend.

```sh
.venv/bin/python -m kernels.catalog
.venv/bin/python -m kernels._shared.build --plan-only
.venv/bin/python -m kernels._shared.build --verify /absolute/native-package
```

The catalog is the source for generated native registries. After an intentional
native descriptor edit, validate/reindex with `python -m kernels.catalog --refresh`
and regenerate with `python -m kernels.catalog --generate`; ordinary validation
rejects descriptor corruption and registry drift. See the [native build/API
instructions](kernels/_shared/README.md) and [remaining tests](testing.md#11-standalone-native-catalog-remaining-test-gates).

This catalog targets Android arm64/Vulkan 1.1. Implemented identities and software
Vulkan evidence do not establish complete contract parity, physical XR support,
or performance qualification. Existing immutable packages and evidence retain
their original hashes and layouts; build migrated sources into a fresh output.

`kernels/linear`, `kernels/gelu`, and `kernels/grouped_conv2d` remain explicitly
marked `xrgpu.model-override` contracts for the separate model integration track.
Standalone `xrgpu.native-operator` entries do not enroll themselves in Session,
model replacement discovery, or qualified automatic selection.

## Discover additional kernel targets

Scan the complete measured operation suite, including operations without an
xrGPU replacement:

```sh
python -m analysis.discovery --archive artifacts/benchmarks/archives \
  --output artifacts/discovery/quest3-kernel-opportunities.json
```

The report ranks compute and layout work within each run, retains raw-report
hashes, and separates staging costs, incorrect cases and partial runs. These
trace observations identify experiments; speedups require fresh paired latency
measurements and full-model validation.

The first additional prototype is fused attention, motivated by the masked
attention case's 17 dispatches. Compile its three workgroup variants and export
fresh discovery fixtures with:

```sh
python -m kernels.experimental.attention.build \
  --glslc artifacts/toolchain/shaderc/2026.4/bin/glslc \
  --output artifacts/experiments/attention-discovery-v1
```

This standalone buffer prototype supports short, masked and odd-shaped attention.
It still needs GPU correctness checks and graph integration, including conversion
costs, before comparing against stock inference. Its sources under
`kernels/experimental/` are separate from the qualified library package. Generate
`--stage confirmation` fixtures only after freezing the discovery shortlist.

