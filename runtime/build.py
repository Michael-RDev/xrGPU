import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def selected_manifest(manifest, profiles=None):
    """Package selected experiments without deleting any exported program."""
    if profiles is None:
        return manifest
    import copy
    result = copy.deepcopy(manifest)
    catalog = manifest.get("optimization_profiles", {})
    if not set(profiles) <= catalog.keys():
        raise ValueError("Requested optimization profile was not exported")
    result["optimization_profiles"] = {name: catalog[name] for name in profiles}
    allowed = set()
    for spec in result["optimization_profiles"].values():
        for entry in spec["lengths"].values():
            for key in ("model", "audit_model"):
                allowed.add(entry[key])
                allowed.add(str(Path(entry[key]).with_suffix(".graph.txt")))
    result["files"] = {name: digest for name, digest in manifest["files"].items()
                       if not name.startswith("minilm_opt_") or name in allowed}
    return result


def ndk_libcxx(ndk):
    host = {"win32": "windows-x86_64", "darwin": "darwin-x86_64", "linux": "linux-x86_64"}
    return (
        Path(ndk)
        / "toolchains/llvm/prebuilt"
        / host[sys.platform]
        / "sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--profiles", nargs="+", help="Optimization profiles to include; exports remain intact")
    args = parser.parse_args()
    from runtime.prepare import apply_patch

    apply_patch()
    ndk = os.environ.get("ANDROID_NDK")
    if not ndk or not shutil.which("glslc"):
        raise SystemExit(
            "Source artifacts/toolchain/env.sh or set ANDROID_NDK and put glslc on PATH"
        )
    build = ROOT / "artifacts/build/android"
    install = ROOT / "artifacts/install"
    cmake = shutil.which("cmake")
    if not cmake or not shutil.which("ninja"):
        raise SystemExit("Put the environment's cmake and ninja on PATH")
    subprocess.run(
        [
            cmake,
            "-G",
            "Ninja",
            "-S",
            str(ROOT),
            "-B",
            str(build),
            f"-DCMAKE_TOOLCHAIN_FILE={ndk}/build/cmake/android.toolchain.cmake",
            "-DANDROID_ABI=arm64-v8a",
            "-DANDROID_PLATFORM=android-29",
            "-DANDROID_STL=c++_shared",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPYTHON_EXECUTABLE={Path(sys.executable).as_posix()}",
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
    if '"xrgpu_attention_float"' not in registry:
        raise RuntimeError("Packed attention shader missing from compiled registry")
    if '"xrgpu_residual_norm_float"' not in registry:
        raise RuntimeError("Residual normalization shader missing from compiled registry")
    for dtype in ("float", "half"):
        for storage in ("buffer", "texture3d"):
            name = f"xrgpu_gelu_{dtype}_{storage}"
            if f'"{name}"' not in registry:
                raise RuntimeError(f"Compiled shader missing from registry: {name}")
    for spec_path in (ROOT / "kernels/linear/vulkan").glob("*/manifest.json"):
        spec = json.loads(spec_path.read_text())
        for rows in {v["tile_m"] for v in spec["variants"]}:
            for bias in ("", "_bias"):
                for storage in ("texture3d", "buffer"):
                    for weight in ("texture2d", "buffer"):
                        name = f"{spec['shader_prefix']}_m{rows}{bias}_{storage}_{weight}_float"
                        if f'"{name}"' not in registry:
                            raise RuntimeError(f"Compiled shader missing from registry: {name}")
    # Only install our targets, not the SDK's development archives.
    dest = install / "lib/arm64-v8a"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("libxrgpu.so", "libopenxr_loader.so"):
        shutil.copy2(next(build.rglob(name)), dest / name)
    libcxx = ndk_libcxx(ndk)
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
        if not destination.resolve().is_relative_to((ROOT / "artifacts").resolve()):
            raise RuntimeError("Bundle destination escaped artifacts directory")
        shutil.rmtree(destination)
    manifest = selected_manifest(manifest, args.profiles)
    destination.mkdir(parents=True, exist_ok=True)
    for name in manifest["files"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    (destination / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ)
    env["GRADLE_USER_HOME"] = str(ROOT / "artifacts/cache/gradle")
    env["ANDROID_USER_HOME"] = str(ROOT / "artifacts/cache/android")
    Path(env["ANDROID_USER_HOME"]).mkdir(parents=True, exist_ok=True)
    gradle = shutil.which("gradle")
    if not gradle:
        raise SystemExit("Gradle is not on PATH")
    subprocess.run([gradle, "--no-daemon", ":app:assembleDebug"], cwd=ROOT, env=env, check=True)
    from runtime.verify_apk import verify

    verify()


if __name__ == "__main__":
    main()
