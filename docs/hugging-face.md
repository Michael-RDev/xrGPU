# Native XR kernel catalog and optional Hugging Face export

The local `kernels/catalog.json` is the authoritative catalog of 221 native
Android ARM64 Vulkan operators. Each `kernels/<operator-id>/` directory owns its
portable contract, fixture descriptors and native shader entrypoints. Shared C++
runtime, shader helpers and build tools live in `kernels/_shared/`.
No browser, WebGPU, JavaScript or downloaded Python runtime participates in
native execution.

All 221 names have native implementations. This does not establish complete
contract parity or physical XR device qualification: the native APIs reject
unsupported routes, conformance is incomplete, and hardware, performance and
sustained XR status remain unverified. See the native package and its bound
conformance reports for measured results and remaining gaps.

## Local validation and original provenance

```sh
python -m kernels.catalog
python -m hub.snapshot verify
```

Original upstream contracts, test and benchmark data, metadata and cards remain
unchanged under `kernels/_provenance/`. Its `catalog.json` retains the original
relative paths and canonical digest. Each source revision is an immutable commit.
The shared `kernels.provenance` resolver verifies checksums and rejects paths
outside this archive and symlink aliases. Reference expressions and test generator
descriptions are data, never executable native kernels. Original WGSL digests are
audit provenance; WGSL implementation files are not shipped or executed.

The native catalog keeps its own identity and points to portable semantics and
actual native C++/SPIR-V metadata. `freeze_contracts.py` derives JSON-free runtime
validation tables from those native definitions, with original provenance checked
alongside them. Existing immutable package snapshots keep their original layouts
and remain independently verifiable; moving repository sources does not rewrite
them.

A later upstream review can be frozen explicitly into a new directory with
`python -m hub.snapshot freeze --output artifacts/upstream-review`. Normal catalog
validation, builds and release preparation use local data. This command is a
separate network operation and does not replace the retained archive implicitly.

## Prepare a local immutable export

```sh
python -m kernels._shared.build --output artifacts/packages/native-kernels
python -m hub.registry prepare \
  --native-package artifacts/packages/native-kernels \
  --output artifacts/hub-releases/native-kernels
python -m hub.registry verify artifacts/hub-releases/native-kernels
```

Preparation verifies the supplied compiled package and the native catalog. It
copies only inventoried native artifacts and allowlisted catalog sources, then
creates `aggregate/`, one `operators/<id>/` directory for each operator actually
present in the supplied package, and `release.json` binding all prepared files.
A complete native package yields 221 operator exports plus the aggregate.
A subset package never invents missing binary implementations.

The aggregate retains the complete original native package at `native/` and the
self-contained native catalog at `kernels/`, including frozen provenance.
Per-operator exports carry a smaller native package containing headers, Android
runtime libraries and the operator's shader dependencies. Each subset has its own
manifest digest and records the original package digest; its native metadata is
not confused with upstream WebGPU execution metadata.

The deleted benchmark source directory is optional and is never restored by
preparation. To include an existing verified package, provide it explicitly:

```sh
python -m hub.registry prepare \
  --native-package artifacts/packages/native-kernels \
  --benchmark-package /absolute/path/to/verified-benchmark-package \
  --output artifacts/hub-releases/native-kernels-with-benchmark
```

That package must retain identity `xrgpu.benchmark-kernels.v1`, an inventoried
license and matching file checksums. Its manifest identity, bytes and original
license are preserved. Without this argument, no benchmark package is included.
The native package remains `xrgpu.native-operators.v1`; compiled model-inference
packages such as `xrgpu.quest3.fp32.v1` remain distinct.

## Optional publication and pinned consumption

Preparation is local and does not publish. Public kernel repositories under the
requested `xrML` namespace require Hugging Face kernel-publishing entitlement.
When publication is explicitly requested and that entitlement is active:

```sh
python -m hub.registry publish artifacts/hub-releases/native-kernels \
  --env-file ../xrBench/.env
```

The separately locked SDK transport recognizes `HF_TOKEN` and `HF_token` without
sourcing shell code, saving a global login or putting credentials in artifact
URLs. It verifies prepared inventories, creates true public `kernel` repositories,
refuses to change existing visibility, and returns actual immutable commit IDs.
Private project files and credentials are not export inputs. Do not substitute
model repositories or invent published commits when kernel creation is unavailable.

Pin an actual aggregate commit, package path `native`, its package SHA-256 and
`--digest-field package_sha256` for direct native-operator downloads. Fetch verifies
every inventoried byte before local promotion; execution remains offline.
For an explicitly included benchmark source package, use xrBench's existing
`engine.kernel_package` CLI with that package's original manifest digest. A
publication does not itself update xrBench's pin or establish device eligibility.

Run `python -m pytest tests/test_hub_registry.py tests/test_upstream_operator_fixtures.py`
for catalog provenance, export boundaries, optional package and fixture integrity
checks. Native numerical and physical qualification results remain separate.
