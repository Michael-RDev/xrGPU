# Generic native Vulkan operator API

`xrgpu::operators::extended` adds typed attributes and multiple named tensors to
`xrgpu::operators`. Link `xrgpu_extended_operators` and include
`<xrgpu/extended_operators.h>`. The original 38-operator API remains available.
The tensor, spatial and complex families add 183 operator identities. Each has a
native GLSL implementation; identity coverage does not establish complete
contract conformance. Read the measured reports for supported cases and gaps.

## Prepare, execute and read back

```cpp
#include <xrgpu/extended_operators.h>
#include <cstring>
using namespace xrgpu::operators;
namespace ex = xrgpu::operators::extended;

Tensor data; data.name = "data"; data.dtype = DType::Float32;
data.shape = {3};
float values[] = {10.f, 20.f, 30.f};
data.data.resize(sizeof(values));
std::memcpy(data.data.data(), values, sizeof(values));
Tensor indices; indices.name = "indices"; indices.dtype = DType::Int32;
indices.shape = {2};
int32_t selected[] = {2, 0};
indices.data.resize(sizeof(selected));
std::memcpy(indices.data.data(), selected, sizeof(selected));
ex::Request request;
request.operation = "ai.onnx.Gather";
request.inputs = {data, indices};
request.attrs.integers["axis"] = 0;
auto metadata = ex::plan(request);
auto op = ex::Operator::prepare("/data/local/tmp/native-package", request);
op->run();
auto outputs = op->read_outputs(); // output = [30.f, 10.f]
auto statistics = op->info();
op->close();
```

The Android library dynamically loads the system Vulkan loader and requires a
compute-capable GPU. It does not execute through a browser or Python. Host shape
planning, upload, execution, and output readback are separate. `run()` waits for
completion; `info()` reports submit/wait wall time and optional device timestamps
separately. Repeated `run()` calls reset control/error scratch and reuse resident
buffers. `close()` releases resources after any submitted work completes.

Native tensors own their bytes. `prepare()` copies the inputs, and every `run()`
uses that prepared input snapshot. Repeating `run()` benchmarks the same request;
it does not automatically feed a returned state or cache into the next run.
For the next decoding step, request every changed present/cache/state output,
call `read_outputs()`, use those returned bytes under the operation's appropriate
past/initial input names, and prepare a new invocation. Readback returns all
changed cache bytes, including retained history and unchanged padding.

This ownership model deliberately differs from the published PagedAttention and
SparseAttention alias semantics: the caller's original host tensors are not
mutated, and the API exposes no aliased GPU-memory ABI. GPU kernels still compute
the full cache updates. Numerical cache parity and this ownership difference
must be reported separately; operator identity coverage is not full contract
parity.

`Attributes` supports integers, numbers, strings, arrays of those types, and typed
tensor literals. `Request.args` has priority over same-named attributes, including
when a number changes between integer and floating representation. A Constant
literal is uploaded then copied by the GPU. Input names may be omitted only when
the frozen contract marks them optional.

Tensors distinguish logical type from physical storage. An unspecified storage
type uses the logical type. For example a logical Int64 index output can use
UInt32 storage when every result fits; checked narrowing rejects overflow.
Expanded internal heaps use one uint32 word per element, or two for Int64,
avoiding byte/half output races. Original IEEE Float16 conversion and two-word
Int64 conversion require neither shaderFloat16 nor shaderInt64 support.

Scalars use an empty shape. Empty tensors retain their zero-sized dimensions.
Dynamic results allocate capacity on the GPU, write GPU counts, then return the
actual shape and retained bytes. Requested dynamic-axis dimensions are capacity
bounds: callers can request optional Unique indices/counts with input-size
capacity without computing the unique count on the CPU. The returned axes use
the actual GPU count; static dimensions remain exact. Results exceeding a
requested capacity are rejected. GPU validation flags out-of-range indices and
invalid runtime lengths explicitly. No numerical CPU fallback is present.

`contracts.json` and its compiled `contract_data.h` freeze the checksummed local
catalog's names, logical dtype constraints, optional tensors and fixed ranks.
`native_extensions.json` separately declares the native a..z (up to26 input)
Sum/Mean/Max/Min extension; it preserves shared logical typing and uses one GPU
input heap. Immutable upstream metadata remains unchanged.
Production has no JSON dependency. Symbolic shape equations and each operator's
attributes are handled by its family planner and shader; the table is not a proof
that every published shape/attribute combination has passed conformance.

## Local development and tests

From the xrGPU root with its existing environment:

The canonical shared tool namespace is `kernels._shared`. Immutable packages
created before the catalog migration retain their original `operators.*` tools
and source layout; invoke the namespace shipped inside that exact snapshot.
Do not edit old packages or relabel their evidence after a source migration.

```sh
python -m kernels._shared.build --plan-only
python -m kernels._shared.extended.develop --output /private/tmp/xrgpu-check \
  --operator ai.onnx.Gather --operator ai.onnx.TopK --jobs 4
```

Every development build snapshots source into a new `run-*` directory, compiles
only selected shaders, and bakes their hashes into a test-only native library.
`--family tensor`, `--family spatial`, and `--family complex` select whole
families. The emitted JSON identifies the package directory and JSON test runner.
The runner accepts one local job JSON file, with `operation`, `inputs`, `attrs`,
`args`, optional requested `outputs`, `package` and `output_directory`. Tensor
payloads can be raw `file` paths or literal `values`. A `plan_only:true` job skips
Vulkan. Set `repeat` to exercise reuse; output readback follows the final run.
The test adapter alone depends on nlohmann JSON.

To execute software Vulkan tests, explicitly set `XRGPU_TEST_VULKAN_LIBRARY` to an
existing native Vulkan loader and `VK_ICD_FILENAMES` to its software ICD JSON.
The test runtime reports `software_vulkan`, `test_only:true` and
`physical_qualification:false`; production Android builds reject this override.
Use the family's independent verifier and the frozen-fixture verifier without
changing original comparison tolerances. A software pass is numerical evidence,
not an XR device support or performance claim.

Build an immutable Android package with:

```sh
python -m kernels._shared.build --output artifacts/packages/native-kernels-all-v4 --jobs 4
python -m kernels._shared.build --verify artifacts/packages/native-kernels-all-v4
```

The package contains both native APIs, arm64 libraries, original source, all
selected validated SPIR-V passes and licenses. `source/kernels/_provenance/` also contains the
checksummed catalog and exactly its five public metadata/docs/fixture files per
operator, with individual immutable revisions. It contains no browser kernels or
network credentials. The frozen-fixture verifiers can run from the source
snapshot without consulting the live checkout. Its manifest binds all delivered bytes. Existing
outputs are never overwritten. Preserve failure, unsupported-case and oracle
conflict statuses separately from passing cases.

## Limits

Contiguous row-major tensors, rank at most 16, 32 input slots, 16 outputs, uint32
address/count limits and a configurable default 256 MiB allocation budget apply.
GQA and SparseAttention use separate GPU transform/attention/cache passes.
Large Unique and TopK use GPU index mergesort; Unique restores first-seen order
for unsorted results on the GPU. Compile-time pass wrappers keep unrelated
control flow out of driver compilation. No speedup is inferred from these
algorithm choices.

The reference kernels prioritize contract coverage and correctness. Some
algorithms use serial GPU work and do not claim production-level throughput.
Published dynamic/graph/state variants and numerical edge cases require measured
fixture evidence. All packages remain physically unqualified until native
correctness and performance tests run on the intended XR hardware.
