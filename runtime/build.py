import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_shader_registry(registry, executorch=ROOT / "artifacts/deps/executorch"):
    """Derive every registered name from the active YAML, not a deleted manifest."""
    import runpy

    from runtime.prepare import vulkan_shader_sources

    generator = runpy.run_path(str(executorch / "backends/vulkan/runtime/gen_vulkan_spv.py"))
    env = {**generator["DEFAULT_ENV"], **generator["TYPE_MAPPINGS"], **generator["UTILITY_FNS"]}
    folders = sorted({str(p.parent) for p in vulkan_shader_sources()})
    mapping = generator["SPVGenerator"](folders, env, None).output_file_map
    names = {name for name, (source, _) in mapping.items() if Path(source).suffix == ".glsl"}
    if not names:
        raise RuntimeError("No xrGPU shader variants discovered")
    missing = sorted(name for name in names if f'"{name}"' not in registry)
    if missing:
        raise RuntimeError("Compiled shaders missing from registry: " + ", ".join(missing))
    return names


def build_library(args):
    """Build the reusable CPU host library or Android AAR without the demo."""
    import sys

    pins = json.loads((ROOT / "dependencies.json").read_text())
    env = dict(os.environ)
    toolchain = ROOT / "artifacts/toolchain"
    env["PATH"] = os.pathsep.join([str(ROOT / ".venv/bin"),
        str(toolchain / "gradle/bin"),
        *(str(p) for p in sorted((toolchain / "shaderc").glob("*/bin"))),
        env.get("PATH", "")])
    env.setdefault("ANDROID_HOME", str(toolchain / "android-sdk"))
    env.setdefault("ANDROID_NDK", str(Path(env["ANDROID_HOME"]) / "ndk" / pins["android"]["ndk"]))
    env["GRADLE_USER_HOME"] = str(ROOT / "artifacts/cache/gradle")
    env["ANDROID_USER_HOME"] = str(ROOT / "artifacts/cache/android")
    build_name = "host-library" if args.host else "android-library"
    output = getattr(args, "output", None)
    output = Path(output).resolve() if output else None
    build = output / "native" if output else ROOT / "artifacts/build" / build_name
    install_name = "artifacts/host-install" if args.host else "artifacts/library"
    if args.experiments:
        install_name += "-development"
    install = output / "install" if output else ROOT / install_name
    cmake = shutil.which("cmake", path=env["PATH"])
    if not cmake:
        raise RuntimeError("CMake required; synchronize the project's development environment")
    command = [cmake, "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
        "-DXRGPU_BUILD_DEMO=OFF", f"-DXRGPU_ENABLE_VULKAN={'OFF' if args.host else 'ON'}",
        f"-DXRGPU_ENABLE_EXPERIMENTS={'ON' if args.experiments else 'OFF'}",
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={install}",
        f"-DXRGPU_BUILD_TESTS={'ON' if args.host else 'OFF'}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        f"-DXRGPU_KERNEL_PACKAGE_ROOT={args.package.resolve() if args.package else ''}",
        f"-DXRGPU_QUALIFICATION_FILE={args.qualification.resolve() if args.qualification else ''}"]
    kleidiai = ROOT / "artifacts/build/android/kleidiai-source"
    if kleidiai.is_dir():
        command.append(f"-DKLEIDIAI_SOURCE_DIR={kleidiai}")
    if not args.host:
        from runtime.prepare import apply_patch

        apply_patch()
        ndk = Path(env["ANDROID_NDK"])
        if not (ndk / "build/cmake/android.toolchain.cmake").is_file():
            raise RuntimeError("Android NDK missing; run python -m runtime.bootstrap_android")
        if not shutil.which("glslc", path=env["PATH"]):
            raise RuntimeError("Shaderc missing from the local toolchain")
        command += [f"-DCMAKE_TOOLCHAIN_FILE={ndk}/build/cmake/android.toolchain.cmake",
                    "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-29",
                    "-DANDROID_STL=c++_shared"]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    targets = ["xrgpu_runtime", "xrgpu_run"]
    if args.host:
        targets.append("xrgpu_session_test")
    subprocess.run([cmake, "--build", str(build), "--target", *targets,
                    "-j", str(args.jobs)], env=env, check=True)
    notices = install / "share/xrgpu"
    if notices.exists():
        shutil.rmtree(notices)
    subprocess.run([cmake, "--install", str(build), "--component", "xrgpu"], env=env, check=True)
    if args.host:
        write_build_manifest(args, build, install, output or install)
        print(f"Host library and runner: {build}")
        return
    destination = install / "lib/arm64-v8a"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build / "libxrgpu_runtime.so", destination)
    legacy_library = install / "lib/libxrgpu_runtime.so"
    if legacy_library.exists():
        legacy_library.unlink()
    libcxx = next(ndk.glob("toolchains/llvm/prebuilt/*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"))
    shutil.copy2(libcxx, destination)
    aar = None
    if not args.native_only:
        gradle = shutil.which("gradle", path=env["PATH"])
        variant = "Debug" if args.experiments else "Release"
        subprocess.run([gradle, "--no-daemon", f"-PxrgpuNativeRoot={install / 'lib'}",
                        f":library:assemble{variant}", f":library:lint{variant}"],
                       cwd=ROOT, env=env, check=True)
        aar = ROOT / ("library/build/outputs/aar/library-" + variant.lower() + ".aar")
        print(f"Android AAR: {aar}")
    write_build_manifest(args, build, install, output or install, aar)


def write_build_manifest(args, build, install, output, aar=None):
    """Inventory distributable native bytes and the policy embedded by CMake."""
    from runtime.kernel_package import verify
    from runtime.integration.qualification import validate_record

    output = Path(output).resolve()
    binaries = output / "bin"
    binaries.mkdir(parents=True, exist_ok=True)
    runner = binaries / "xrgpu_run"
    shutil.copy2(build / "xrgpu_run", runner)
    libraries = []
    sources = sorted(build.glob("libxrgpu_runtime.*")) if args.host else sorted((install / "lib/arm64-v8a").glob("*.so"))
    for source in sources:
        if source.suffix not in (".so", ".dylib", ".dll"):
            continue
        destination = binaries / source.name
        shutil.copy2(source, destination)
        libraries.append(destination.relative_to(output).as_posix())
    if not libraries:
        raise RuntimeError("Native build produced no runtime library")
    paths = [runner, *(output / name for name in libraries)]
    if aar:
        destination = output / "xrgpu.aar"
        shutil.copy2(aar, destination)
        paths.append(destination)
    package = verify(args.package) if args.package else {}
    qualification = json.loads(args.qualification.read_text()) if args.qualification else {}
    if qualification:
        validate_record(qualification, package, allow_pending=False)
    value = dict(schema_version=1, kind="xrgpu.native.build",
                 build_type="development" if args.experiments else "production",
                 platform="host" if args.host else "android-arm64",
                 runtime_revision=json.loads((ROOT / "dependencies.json").read_text())["executorch"]["commit"],
                 package_sha256=package.get("package_sha256"),
                 qualification_sha256=qualification.get("qualification_sha256"),
                 runner=runner.relative_to(output).as_posix(), libraries=libraries,
                 files={p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (output / "build.json").write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--library", action="store_true", help="Build the reusable Android AAR")
    mode.add_argument("--host", action="store_true", help="Build the CPU host library and runner")
    parser.add_argument("--package", "--kernel-package", type=Path, help="Immutable optimized kernel package")
    parser.add_argument("--qualification", type=Path, help="Qualified hardware record to embed")
    parser.add_argument("--experiments", action="store_true", help="Development library with explicit candidate overrides")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Isolated native build, install and deployment inventory directory")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.qualification and not args.package:
        parser.error("--qualification requires --package")
    if args.library or args.host:
        build_library(args)
        return
    if args.package or args.qualification or args.experiments or args.output:
        parser.error("Library options require --library or --host")
    from runtime.prepare import apply_patch

    apply_patch()
    ndk = os.environ.get("ANDROID_NDK")
    if not ndk or not shutil.which("glslc"):
        raise SystemExit(
            "Source artifacts/toolchain/env.sh or set ANDROID_NDK and put glslc on PATH"
        )
    build = ROOT / "artifacts/build/android"
    install = ROOT / "artifacts/install"
    cmake = str(ROOT / ".venv/bin/cmake") if (ROOT / ".venv/bin/cmake").exists() else "cmake"
    subprocess.run(
        [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(build),
            f"-DCMAKE_TOOLCHAIN_FILE={ndk}/build/cmake/android.toolchain.cmake",
            "-DANDROID_ABI=arm64-v8a",
            "-DANDROID_PLATFORM=android-29",
            "-DANDROID_STL=c++_shared",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install}",
        ],
        check=True,
    )
    subprocess.run(
        [cmake, "--build", str(build), "--target", "xrgpu", "-j", str(args.jobs)], check=True
    )
    # Shader generation order determines registry names. Verify the actual
    # compiled registry matches every name our C++ dispatch can request.
    registry = (build / "vulkan_compute_shaders/spv.cpp").read_text()
    names = verify_shader_registry(registry)
    print(f"Verified {len(names)} xrGPU shaders in the native registry", flush=True)
    # Only install our targets, not the SDK's development archives.
    dest = install / "lib/arm64-v8a"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("libxrgpu.so", "libopenxr_loader.so"):
        shutil.copy2(next(build.rglob(name)), dest / name)
    libcxx = (
        Path(ndk)
        / "toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
    )
    shutil.copy2(libcxx, dest / libcxx.name)
    if args.native_only:
        return
    source = ROOT / "artifacts/bundle"
    manifest = json.loads((source / "bundle.json").read_text())
    if manifest["references_only"]:
        raise SystemExit("Run full model export before packaging")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Bundle checksum mismatch: {name}")
    destination = ROOT / "artifacts/bundle-parent/bundle"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    env = dict(os.environ)
    env["GRADLE_USER_HOME"] = str(ROOT / "artifacts/cache/gradle")
    env["ANDROID_USER_HOME"] = str(ROOT / "artifacts/cache/android")
    Path(env["ANDROID_USER_HOME"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(["gradle", "--no-daemon", ":app:assembleDebug"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
