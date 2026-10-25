# Third-party notices

The linear shader template and its YAML variant definitions derive from
`pytorch/executorch` at commit `e4d02f41f7909e8ed5bf4a14ffc520d733453d9f`.
The original copyright headers are retained. The xrGPU changes expose experimental
tile, workgroup, and storage variants; no upstream speed claim is implied.

The native app links ExecuTorch (BSD-3-Clause), its Vulkan dependencies and
XNNPACK CPU backend, and Khronos OpenXR SDK (Apache-2.0 and component licenses).
XNNPACK's KleidiAI dependency is pinned by the upstream CMake download URL and
SHA-256; its Apache-2.0 and BSD-3-Clause texts are included as well.
Copies of the corresponding license texts are in `licenses/` and are bundled
with the app's offline assets. Native source revisions are pinned in
`dependencies.json` and the upstream submodule commits.

The `sentence-transformers/all-MiniLM-L6-v2` model is Apache-2.0 licensed. Its
upstream snapshot is recorded in the generated `artifacts/bundle/bundle.json`; exported weights and tokenizer
assets are generated locally and are not committed. The model's license text is
included in `licenses/model-Apache-2.0.txt` and the generated bundle.

The Android NDK's `libc++_shared.so` runtime is included in the APK under its
upstream license terms. Android tooling, Gradle, CMake, shaderc, Python packages,
and the JSON-java JVM test dependency retain their own licenses and are installed
as local build/test dependencies, not xrGPU source.
The NDK's complete notices are included to preserve the bundled C++ runtime's
attribution and LLVM license exceptions.

The Android demo has no network permission or automatic upload mechanism.
Benchmark collection is maintained in xrBench; xrGPU's host-side report client
only uploads when explicitly invoked.
