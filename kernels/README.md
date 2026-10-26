# Native XR kernels

The native catalog is `catalog.json`: 221 qualified operator identities with
315 Vulkan/SPIR-V variants. Each `ai.onnx.*` or `com.microsoft.*` folder contains
`contract.json`, `test.json`, `bench.json`, native shader entrypoints and a card.
The target is Android ARM64, API 29+, Vulkan 1.1. Quest 3 device validation is pending.

```sh
python -m kernels.catalog
python -m kernels.catalog --generate
python -m kernels._shared.build --output artifacts/packages/native-kernels-all-v4
python -m kernels._shared.build --verify artifacts/packages/native-kernels-all-v4
```

Edit native descriptors deliberately; `python -m kernels.catalog --refresh --generate`
reindexes them after validation and regenerates derived runtime registries. Frozen
contracts cannot be changed by reindexing. Builds check generated registries for drift.
The historical `python -m operators.build` command forwards to the same implementation.

Shared C++ planners, Vulkan execution, common GLSL bodies and build tools live in
`_shared/`. Public headers remain in `runtime/include/xrgpu/`. The legacy API covers
38 operators; the typed extended API covers 183. Both expose explicit preparation,
execution, output readback, metadata and close. Unsupported contracts/capabilities
raise errors; reference CPUs are test tools and are never a numerical runtime fallback.

The unchanged `_provenance/` archive freezes the original 221 upstream `v1`
revisions, 13,919 correctness cases, 5,839 benchmark cases, notices and SHA-256 hashes.
Its historical WebGPU paths and metadata record provenance; native dispatch and
compilation use the native catalog exclusively. `provenance.py` resolves checked
reference bytes offline. Native fixture descriptors preserve case identities and
require individual reassessment of upstream GPU skips.

Implementation does not imply complete conformance or physical support. Unique
NaN/provider differences and unsupported fixture types remain recorded in the
original conformance evidence. Dynamic outputs use explicit capacity bounds;
cache outputs are owned and do not alias caller inputs. See the shared native
contract differences and repository `testing.md` for remaining tests.

`linear`, `gelu` and `grouped_conv2d` retain separate model override identities and
qualification. `experimental/` contains isolated research. Standalone catalog
membership does not add operators to the model Session/ExecuTorch integration.

New native code is Apache-2.0. Preserve the original reference licenses/notices.
