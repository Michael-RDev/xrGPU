# Build and run

The original reference build host is an Apple Silicon Mac with JDK 17 or newer. The
application target is Android arm64-v8a, API 29 minimum. The desktop does model
conversion and compilation; it is not needed while diagnostics run on the headset.

## Dependencies

### Windows

Open a Visual Studio 2022 Developer PowerShell configured for x64 host tools.
Run these commands from the repository root. Bootstrap downloads verified Android
archives and a portable JDK 17, and builds a pinned shaderc release and ExecuTorch's
pinned host flatc. Everything is kept under `artifacts/`; SDK registry changes and
global PATH changes are not needed. The shaderc source pin and JDK checksum live
in `runtime/windows-toolchain.json`.

```powershell
uv sync --locked --extra export --python 3.12 --cache-dir artifacts/cache/uv
$env:PATH = "$PWD\.venv\Scripts;" + $env:PATH
$env:PYTHONUTF8 = '1'
.venv/Scripts/python.exe -m runtime.prepare
.venv/Scripts/python.exe -m runtime.bootstrap_android
. artifacts/toolchain/env.ps1
.venv/Scripts/python.exe -m models.export
.venv/Scripts/python.exe -m runtime.build --jobs 4
```

`runtime.bootstrap_android --adb-only` installs just the verified Windows platform
tools for checking the connection while the rest of the build is being prepared.
USB debugging must be enabled and the headset must authorize this PC. A USB/PC VR
connection without an authorized ADB device cannot run the diagnostics.

### macOS

`dependencies.json` pins native source commits and Android build versions.
`uv.lock` pins Python dependencies. Use the same ExecuTorch version for model
export and native execution. `runtime.prepare` checks the checkout commit and
applies an idempotent patch without resetting source changes.
The known earlier xrGPU dispatch patch is upgraded through a checked migration;
unrecognized conflicting edits are left for review instead of resetting the checkout.

`runtime.bootstrap_android` installs a macOS toolchain inside `artifacts/toolchain`.
It verifies downloaded Android archives against repository checksums. Its shaderc
bottle is pinned by SHA-256. For an existing toolchain, set `ANDROID_HOME` and
`ANDROID_NDK`, put Gradle 8.11.1 and a recent Vulkan SDK `glslc` on `PATH`, and
skip bootstrap. The old NDK shader compiler may not compile all upstream shaders.

```sh
UV_CACHE_DIR=artifacts/cache/uv uv sync --extra export --python 3.12
.venv/bin/python -m runtime.prepare
.venv/bin/python -m runtime.bootstrap_android
source artifacts/toolchain/env.sh
.venv/bin/python -m models.export
.venv/bin/python -m runtime.build --jobs 4
```

After the first model download, `models.export --offline` uses the cached pinned
snapshot. The full export generates approximately 550 MB of uncompressed assets:
the first implementation deliberately uses separate static-shape programs. Native
libraries, shaders, and weights are packaged into one offline debug APK. Asset
checksums are verified during packaging and before use on the headset.

For native compilation only, pass `--native-only` to `runtime.build`. Source files
remain editable and intermediate products are under `artifacts/build`.

## Headset

Use a **Quest 3** for the current custom kernel experiments. The native runtime
rejects candidate loading on other models, including Quest 3S. Check that Inspect
headset reports `device_model: Quest 3`, `device_profile: quest3`, and
`candidate_capable: true` before testing
candidates. An unknown model stays on stock execution until its identity and a
separate device profile have been reviewed.

Kernel implementation discovery uses `kernels/<operation>/vulkan/<implementation>/`.
Shader filenames must be unique across implementations because ExecuTorch's shader
registry is flat. A device must also be explicitly registered in
`devices/registry.h`; staging shader files never enables another headset.

Enable developer mode and USB debugging, connect the headset, and authorize this
computer in the headset. Confirm it is listed as `device`, not `unauthorized`:

```sh
adb devices -l
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n org.xrgpu/.MainActivity
```

Launch xrGPU from the headset's installed/unknown-source applications. It runs
without a USB connection after installation. First use extracts and verifies the
bundled assets. Keep the headset awake while tests run.

1. Inspect capabilities and run model correctness.
2. Benchmark layer candidates individually.
3. Run a paired model comparison for a candidate and sequence length.
4. Repeat alongside the fixed VR scene. Stay in the scene and keep the headset on.
5. Leave the sustained checkbox enabled for a result eligible for local selection.

The VR scene returns to the controls when complete. Removing the headset,
switching away, cancellation, a severe thermal state, or an unsupported 72 Hz
session stops the run and records a failure. The app never roots the headset or
changes system clock limits. CPU/GPU power and exact memory counters not exposed
by the APIs remain unavailable rather than estimated.

## Results

Results stay in app-private storage. View their summary in the app, or explicitly
copy them to this workspace:

```sh
mkdir -p artifacts/device-results
adb exec-out run-as org.xrgpu tar -C files -cf - results > artifacts/device-results/results.tar
tar -xf artifacts/device-results/results.tar -C artifacts/device-results
.venv/bin/python -m analysis.report artifacts/device-results/results/RUN.json
```

`analysis.report --selection artifacts/selection.json` writes a host-side profile
only if the evidence passes. This is an analysis artifact; the app independently
creates and validates its own local profiles and does not import arbitrary host
profiles. No results are uploaded or published.

## Host checks

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check analysis models runtime tests
.venv/bin/python -m tests.host_java
```

CPU export integration tests require `models.export` first. They explicitly skip
when the exported assets or optional export dependencies are absent. They do not
substitute for running GPU correctness tests on Quest.
