# Native Vulkan operator implementation

This directory contains shared C++17 planners, Vulkan execution, GLSL bodies,
contract validation, reference tests and package tooling for the authoritative
`../catalog.json`. Operator-specific native entrypoints live in the adjacent
fully qualified operator folders. Generated registries are derived from that
catalog and must pass `python -m kernels.catalog` before a build.

The public APIs remain `runtime/include/xrgpu/operators.h` (38 legacy operators)
and `extended_operators.h` (183 typed operators). Production artifacts target
Android ARM64, API 29 and Vulkan 1.1. Named inputs, optional ports, multiple
outputs, explicit logical/storage types and bounded dynamic outputs follow
native C++ planners. There is no browser runtime or numerical CPU fallback.

Build with `python -m kernels._shared.build --output <fresh-package-directory>`.
The compatibility command `python -m operators.build` runs the same implementation.
Packages contain exact source inputs, both Android shared libraries, validated
SPIR-V, public headers, immutable reference bytes and required licenses. Existing
packages are immutable; source migrations require a new package directory/digest.

Software Vulkan runners and CPU references are explicit test tools. Complete
contract conformance, hardware correctness, performance and sustained XR behavior
remain separate evidence gates; Quest 3 validation is pending. Keep original
failure/unsupported records. See `extended/complex_contract_differences.json`
for cache ownership and other differences, and repository `testing.md` for tests.
