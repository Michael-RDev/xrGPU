# Benchmark kernel source package

xrGPU owns these sources; xrBench fetches a pinned Git commit and compiles them
into its Android runner. This package is separate from `xrgpu.quest3.fp32.v1`.
The initial migration preserves the CPU scalar/NEON matmul, ExecuTorch matmul
templates, resident Vulkan matmul, calibration shaders, and tile configuration
from xrBench byte for byte. Existing names intentionally remain unchanged.
The GLSL templates retain their upstream Meta copyright and BSD license (LICENSE).

The manifest pins the compatible ExecuTorch revision and SHA-256 of each source.
Its content digest is SHA-256 of the manifest object serialized as sorted-key,
compact UTF-8 JSON (without a trailing newline). Documentation is not executable
package content. When changing a source, update its checksum and xrBench's
dependency commit/content digest together; consumers never track a moving branch.

Compile with xrBench's existing integration and shader generator. The
`INCORRECT` template option remains a deliberate correctness control; it is not
a production variant. CPU and Vulkan resident variants and all calibration
probes retain their existing ABI. No performance or hardware qualification is
granted by this source migration.
