"""Build native code, verify assets, and package the offline debug APK."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    from runtime.prepare import apply_patch
    apply_patch()
    ndk = os.environ.get("ANDROID_NDK")
    if not ndk or not shutil.which("glslc"):
        raise SystemExit("Source artifacts/toolchain/env.sh or set ANDROID_NDK and put glslc on PATH")
    build = ROOT / "artifacts/build/android"
    install = ROOT / "artifacts/install"
    cmake = str(ROOT / ".venv/bin/cmake") if (ROOT / ".venv/bin/cmake").exists() else "cmake"
    subprocess.run([cmake, "-S", str(ROOT), "-B", str(build),
                    f"-DCMAKE_TOOLCHAIN_FILE={ndk}/build/cmake/android.toolchain.cmake",
                    "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-29",
                    "-DANDROID_STL=c++_shared", "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_INSTALL_PREFIX={install}"], check=True)
    subprocess.run([cmake, "--build", str(build), "--target", "xrgpu", "-j", str(args.jobs)], check=True)
    # Shader generation order determines registry names. Verify the actual
    # compiled registry matches every name our C++ dispatch can request.
    registry = (build / "vulkan_compute_shaders/spv.cpp").read_text()
    for rows in (2, 4):
        for bias in ("", "_bias"):
            for storage in ("texture3d", "buffer"):
                for weight in ("texture2d", "buffer"):
                    name = f"xrgpu_linear_m{rows}{bias}_{storage}_{weight}_float"
                    if f'"{name}"' not in registry:
                        raise RuntimeError(f"Compiled shader missing from registry: {name}")
    # Only install our targets, not the SDK's development archives.
    dest = install / "lib/arm64-v8a"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("libxrgpu.so", "libopenxr_loader.so"):
        shutil.copy2(next(build.rglob(name)), dest / name)
    libcxx = Path(ndk) / "toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
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
