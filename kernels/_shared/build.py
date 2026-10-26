"""Build an inventoried local native Vulkan operator package (no publication).

The output contains Android arm64 code, validated SPIR-V, exact source inputs,
and licenses. Compilation and host metadata tests do not qualify physical GPUs.
"""

# SPDX-License-Identifier: Apache-2.0
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "kernels"
SOURCE = KERNELS / "_shared"
TOOLCHAIN = ROOT.parent / "xrBench/artifacts/toolchain"
DTYPES = ["float32", "float16", "int32", "uint32", "int16", "int8", "uint8", "int64", "bool"]
COMPILER_TYPES = {
    "float32": ("f32", 0),
    "float16": ("f16", 1),
    "int32": ("i32", 2),
    "uint32": ("u32", 3),
    "int8": ("i8", 4),
    "uint8": ("u8", 5),
    "int16": ("i16", 6),
}
PACKAGE_ID = "xrgpu.native-operators.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog_root(source=SOURCE):
    """Accept the shared source path or its native catalog root."""
    path = Path(source)
    return path.parent if path.name == "_shared" else path


def registry(source=SOURCE):
    """Compatibility accessor; native catalog entries are authoritative."""
    from kernels.catalog import legacy_records

    return legacy_records(catalog_root(source))


def extended_registry(source=SOURCE):
    """Compatibility accessor for the typed native API catalog entries."""
    from kernels.catalog import extended_records

    return extended_records(catalog_root(source))


def shader_variants(source=SOURCE):
    from kernels.catalog import shader_variants as catalog_variants

    return catalog_variants(catalog_root(source))


def command(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def find_cmake():
    local = ROOT.parent / "xrBench/.venv/bin/cmake"
    return local if local.is_file() else Path(shutil.which("cmake") or "cmake")


def host_tests(build_dir=None, source=SOURCE, cmake=None, jobs=4):
    build_dir = Path(build_dir or ROOT / "artifacts/build/native-kernel-plan-tests")
    command(
        cmake or find_cmake(),
        "-S",
        source,
        "-B",
        build_dir,
        "-DXRGPU_OPERATORS_PLAN_ONLY=ON",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DBUILD_TESTING=ON",
    )
    command(cmake or find_cmake(), "--build", build_dir, "-j", jobs)
    command(build_dir / "xrgpu_operator_plan_tests")
    if (build_dir / "xrgpu_extended_framework_tests").is_file():
        command(build_dir / "xrgpu_extended_framework_tests")
    return build_dir


def verify(directory):
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("Native package root must not be a link")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("package_id") != PACKAGE_ID:
        raise ValueError("Unsupported native operator package")
    content = {k: v for k, v in manifest.items() if k != "package_sha256"}
    if hashlib.sha256(canonical(content)).hexdigest() != manifest.get("package_sha256"):
        raise ValueError("Native operator package identity mismatch")
    inventory = manifest.get("files")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Native operator package inventory is empty")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Native package links are forbidden")
        if path.is_file() and path.relative_to(root).as_posix() != "manifest.json":
            actual.add(path.relative_to(root).as_posix())
    if actual != set(inventory):
        raise ValueError("Native package file inventory differs")
    for name, digest in inventory.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or any(p in ("", ".", "..") for p in name.split("/"))
            or "\\" in name
        ):
            raise ValueError("Invalid native package path")
        if sha(root / relative) != digest:
            raise ValueError("Native package checksum mismatch: " + name)
    return manifest


def snapshot_contracts(destination):
    """Copy only checked, public frozen catalog metadata; never Hub credentials."""
    hub = KERNELS / "_provenance"
    if hub.is_symlink():
        raise ValueError("Frozen Hub root links are forbidden")

    def regular(relative):
        if (
            not isinstance(relative, str)
            or "\\" in relative
            or relative.startswith("/")
            or any(part in ("", ".", "..") for part in relative.split("/"))
        ):
            raise ValueError("Invalid frozen catalog source path")
        path = hub
        for part in relative.split("/"):
            path = path / part
            if path.is_symlink():
                raise ValueError("Frozen catalog source links are forbidden")
        if not path.is_file():
            raise ValueError("Frozen catalog source file missing")
        return path

    raw = regular("catalog.json").read_bytes()
    catalog = json.loads(raw)
    content = {key: value for key, value in catalog.items() if key != "catalog_sha256"}
    if (
        catalog.get("kind") != "xrgpu.upstream-catalog"
        or catalog.get("schema_version") != 1
        or hashlib.sha256(canonical(content)).hexdigest() != catalog.get("catalog_sha256")
    ):
        raise ValueError("Frozen upstream catalog identity mismatch")
    entries = catalog.get("entries", [])
    if len(entries) != catalog.get("operator_count") or len(
        {entry["id"] for entry in entries}
    ) != len(entries):
        raise ValueError("Frozen operator inventory differs from catalog count")
    destination = Path(destination) / "kernels/_provenance"
    destination.mkdir(parents=True)
    (destination / "catalog.json").write_bytes(raw)
    for entry in entries:
        identity = entry["id"]
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", identity) or not re.fullmatch(
            r"[0-9a-f]{40}", entry["revision"]
        ):
            raise ValueError("Invalid frozen operator identity/revision")
        prefix = "upstream/" + identity + "/"
        allowed = {prefix + "README.md"} | {
            prefix + "build/webgpu/" + name + ".json"
            for name in ("manifest", "metadata", "test", "bench")
        }
        inventory = entry.get("files", {})
        if set(inventory) != allowed:
            raise ValueError("Frozen catalog must inventory exactly its five public metadata files")
        for relative, digest in inventory.items():
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid frozen source digest")
            source = regular(relative)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha(target) != digest:
                raise ValueError("Frozen catalog file checksum mismatch: " + relative)
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(regular(name), destination / name)
    return catalog


def snapshot(destination, *, include_contracts=False):
    """Snapshot only checked catalog kernels, shared native code and provenance.

    Old immutable package verification does not depend on this source layout.
    Provenance is always included for offline validation; include_contracts keeps
    its compatibility meaning of returning the checked upstream catalog.
    """
    from kernels.catalog import generate_registries, load

    catalog = load(KERNELS)
    generate_registries(KERNELS, check=True)
    destination = Path(destination)
    allowed_suffixes = {
        ".py",
        ".cpp",
        ".h",
        ".def",
        ".glsl",
        ".glslh",
        ".comp",
        ".md",
        ".json",
        ".yaml",
        ".txt",
        ".cmake",
    }
    roots = ["_shared"] + [entry["path"] for entry in catalog["entries"]]
    if len(roots) != len(set(roots)):
        raise ValueError("Duplicate native source directory")
    for name in roots:
        if name != "_shared" and not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", name):
            raise ValueError("Invalid native source directory")
        original = KERNELS / name
        if original.is_symlink() or not original.is_dir():
            raise ValueError("Native source directory must be regular: " + name)
        for file in sorted(original.rglob("*")):
            relative = file.relative_to(KERNELS)
            if file.is_symlink():
                raise ValueError("Native kernel sources must not be links")
            if not file.is_file() or any(
                part.startswith(".") or part == "__pycache__" for part in relative.parts
            ):
                continue
            if file.suffix not in allowed_suffixes and file.name not in {
                "CMakeLists.txt",
                "LICENSE",
                "NOTICE",
            }:
                continue
            out = destination / "kernels" / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, out)
    for name in ("catalog.py", "catalog.json", "provenance.py", "__init__.py"):
        file = KERNELS / name
        if file.is_symlink() or not file.is_file():
            raise ValueError("Native catalog source missing: " + name)
        out = destination / "kernels" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, out)
    adapter = ROOT / "operators/__init__.py"
    if adapter.exists():
        if adapter.is_symlink() or not adapter.is_file():
            raise ValueError("Native compatibility adapter must be a regular file")
        out = destination / "operators/__init__.py"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(adapter, out)
    for header in ("operators.h", "extended_operators.h"):
        file = ROOT / "runtime/include/xrgpu" / header
        if file.exists():
            if file.is_symlink():
                raise ValueError("Native API headers must not be links")
            out = destination / "runtime/include/xrgpu" / header
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, out)
    upstream = snapshot_contracts(destination)
    load(destination / "kernels")
    generate_registries(destination / "kernels", check=True)
    return upstream if include_contracts else None


def build(output=None, *, ndk=None, glslc=None, spirv_val=None, cmake=None, jobs=4):
    output = Path(output or ROOT / "artifacts/packages/native-kernels-all-v4").absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Native package output exists; choose a new --output path")
    ndk = Path(ndk or os.environ.get("ANDROID_NDK", TOOLCHAIN / "android-sdk/ndk/28.2.13676358"))
    glslc = Path(glslc or os.environ.get("GLSLC_PATH", TOOLCHAIN / "shaderc/2026.4/bin/glslc"))
    validators = sorted(ndk.glob("shader-tools/*/spirv-val"))
    spirv_val = Path(spirv_val or (validators[0] if validators else "spirv-val"))
    if not (ndk / "build/cmake/android.toolchain.cmake").is_file():
        raise FileNotFoundError("Android NDK is missing")
    if not glslc.is_file() or not spirv_val.is_file():
        raise FileNotFoundError("Pinned glslc and SPIR-V validator are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".native-operators-", dir=output.parent))
    try:
        catalog = snapshot(stage / "source", include_contracts=True)
        source = stage / "source/kernels/_shared"
        records = registry(source)
        shaders = stage / "shaders"
        shaders.mkdir()
        from kernels.catalog import load

        native_catalog = load(source.parent)
        extended = extended_registry(source)
        if set(row["id"] for row in extended) & set(row["id"] for row in records):
            raise ValueError("Legacy and extended native membership overlap")
        native_ids = {entry["id"] for entry in native_catalog["entries"]}
        if {row["id"] for row in records + extended} != native_ids or native_ids != {
            entry["id"] for entry in catalog["entries"]
        }:
            raise ValueError(
                "Native APIs must cover exactly the checked native and upstream catalogs"
            )
        extended_shaders = shaders / "extended"
        extended_shaders.mkdir()
        required_extended_shaders = {
            name for row in extended for name in [row["shader"], *row["shader_dependencies"]]
        }
        variants = shader_variants(source)
        if len(variants) != native_catalog["shader_count"]:
            raise ValueError("Native catalog shader count differs from its unique variants")
        compile_jobs = []
        outputs = set()
        for variant in variants:
            shader_output = variant["output"]
            if (
                not re.fullmatch(r"(?:extended/)?[a-z0-9_]+\.spv", shader_output)
                or shader_output in outputs
            ):
                raise ValueError("Invalid or duplicate native shader output")
            outputs.add(shader_output)
            shader = source.parent / variant["source"]
            if (
                shader.is_symlink()
                or not shader.is_file()
                or not shader.resolve().is_relative_to(source.parent.resolve())
                or shader.suffix != ".comp"
            ):
                raise ValueError("Native catalog shader source is not a regular compute shader")
            definitions = []
            for definition in variant["defines"]:
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:=[A-Za-z0-9_.+-]+)?", definition):
                    raise ValueError("Invalid native shader definition")
                definitions.append("-D" + definition)
            includes = []
            for relative in variant["include_dirs"]:
                include = source.parent / relative
                if (
                    include.is_symlink()
                    or not include.is_dir()
                    or not include.resolve().is_relative_to(source.parent.resolve())
                ):
                    raise ValueError("Native shader include directory escapes catalog")
                includes.append("-I" + str(include))
            dest = shaders / shader_output
            dest.parent.mkdir(parents=True, exist_ok=True)
            compile_jobs.append(
                (
                    dest,
                    [
                        glslc,
                        "--target-env=vulkan1.1",
                        "-O",
                        *definitions,
                        *includes,
                        shader,
                        "-o",
                        dest,
                    ],
                )
            )
        if {
            name.removeprefix("extended/") for name in outputs if name.startswith("extended/")
        } != required_extended_shaders:
            raise ValueError("Native shader variants differ from extended entrypoint dependencies")

        def compile_validate(job):
            dest, arguments = job
            command(*arguments)
            command(spirv_val, "--target-env", "vulkan1.1", dest)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(compile_validate, compile_jobs))
        if {p.name for p in extended_shaders.glob("*.spv")} != required_extended_shaders:
            raise ValueError("Extended shader map references missing native sources")
        generated = source / "generated"
        generated.mkdir(exist_ok=True)
        hashes = [(p.name, sha(p)) for p in sorted(shaders.glob("*.spv"))]
        (generated / "shader_hashes.h").write_text(
            "// Generated from the compiled immutable SPIR-V inventory.\n#pragma once\n"
            "namespace xrgpu::operators::detail {\n"
            "struct ShaderHash { const char* name; const char* sha256; };\n"
            "inline constexpr ShaderHash kShaderHashes[] = {\n"
            + "".join('{"' + name + '","' + digest + '"},\n' for name, digest in hashes)
            + "};\n}\n"
        )
        extended_hashes = [(p.name, sha(p)) for p in sorted(extended_shaders.glob("*.spv"))]
        if not extended_hashes:
            raise ValueError("Extended native shader families are missing")
        (generated / "extended_shader_hashes.h").write_text(
            "// Generated immutable extended SPIR-V inventory.\n#pragma once\n"
            "namespace xrgpu::operators::extended::detail {\n"
            "struct ExtendedShaderHash {const char* name;const char* sha256;};\n"
            "inline constexpr ExtendedShaderHash kExtendedShaderHashes[] = {\n"
            + "".join('{"' + name + '","' + digest + '"},\n' for name, digest in extended_hashes)
            + "};\n}\n"
        )
        with tempfile.TemporaryDirectory(prefix="xrgpu-native-build-") as temporary:
            host_tests(Path(temporary) / "host", source, cmake, jobs)
            android = Path(temporary) / "android"
            command(
                cmake or find_cmake(),
                "-S",
                source,
                "-B",
                android,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_TESTING=OFF",
                "-DANDROID_ABI=arm64-v8a",
                "-DANDROID_PLATFORM=android-29",
                "-DANDROID_STL=c++_shared",
                "-DCMAKE_TOOLCHAIN_FILE=" + str(ndk / "build/cmake/android.toolchain.cmake"),
            )
            command(cmake or find_cmake(), "--build", android, "-j", jobs)
            lib = stage / "lib/android-arm64"
            lib.mkdir(parents=True)
            for name in ("libxrgpu_operators.so", "libxrgpu_extended_operators.so"):
                shutil.copyfile(android / name, lib / name)
        runtimes = sorted(
            ndk.glob(
                "toolchains/llvm/prebuilt/*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
            )
        )
        if len(runtimes) != 1:
            raise RuntimeError("Ambiguous Android shared C++ runtime")
        shutil.copyfile(runtimes[0], lib / "libc++_shared.so")
        include = stage / "include/xrgpu"
        include.mkdir(parents=True)
        for header in ("operators.h", "extended_operators.h"):
            shutil.copyfile(stage / "source/runtime/include/xrgpu" / header, include / header)
        for name in ("README.md", "LICENSE", "NOTICE"):
            shutil.copyfile(source / name, stage / name)
        notices = sorted(ndk.glob("toolchains/llvm/prebuilt/*/NOTICE"))
        if len(notices) != 1:
            raise RuntimeError("Android C++ runtime notice is missing")
        (stage / "THIRD_PARTY_NOTICES.md").write_text(
            "# Android shared C++ runtime notices\n\nThe bundled libc++_shared.so is from the pinned Android NDK.\n\n"
            + notices[0].read_text()
        )
        compiler = subprocess.check_output([str(glslc), "--version"], text=True).strip()
        supported = {
            row["id"]: dict(
                dtypes=row["dtypes"],
                input_names=row["input_names"],
                output_names=[row["output_name"]],
                layouts=["contiguous"],
                attributes=[],
                limits=dict(
                    max_rank=16, max_elements=4294967295, default_memory_limit_bytes=268435456
                ),
                qualification="unqualified",
                native_fp16=False,
                shader_float16_required=False,
            )
            for row in records
        }
        extensions = json.loads((source / "extended/native_extensions.json").read_text())
        for row in extended:
            supported[row["id"]] = dict(
                native_api="extended",
                family=row["family"],
                shader="extended/" + row["shader"],
                shader_dependencies=["extended/" + name for name in row["shader_dependencies"]],
                qualification="unqualified",
                attributes="typed invocation metadata",
                layouts=["contiguous"],
                native_extensions=extensions.get(row["id"]),
                limits=dict(
                    max_rank=16, max_inputs=32, max_outputs=16, default_memory_limit_bytes=268435456
                ),
            )
        content = dict(
            schema_version=1,
            package_id=PACKAGE_ID,
            version="0.1.0-alpha",
            backend="vulkan",
            target="android-arm64",
            vulkan_min_version="1.1",
            license="Apache-2.0",
            operations=[row["id"] for row in records] + [row["id"] for row in extended],
            supported_contracts=supported,
            native_apis=["legacy", "extended"],
            extended_descriptor_version=1,
            shader_count=len(hashes) + len(extended_hashes),
            operation_counts=dict(legacy=len(records), extended=len(extended)),
            qualification="unqualified",
            physical_device_evidence=None,
            source=dict(
                repository="xrGPU",
                upstream_kernel_source_revision="6fdf6301e2bbcc2f03bf1eaf493b7ad55ef33afc",
                upstream_kernel_source_revision_kind="origin source Git revision",
                upstream_catalog_sha256=catalog["catalog_sha256"],
                upstream_catalog_path="source/kernels/_provenance/catalog.json",
                native_catalog_sha256=native_catalog["catalog_sha256"],
                native_catalog_path="source/kernels/catalog.json",
                upstream_operator_revisions={
                    entry["id"]: entry["revision"] for entry in catalog["entries"]
                },
            ),
            compiler=compiler,
            compiler_flags=["--target-env=vulkan1.1", "-O"],
            files={
                p.relative_to(stage).as_posix(): sha(p)
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        )
        manifest = dict(content, package_sha256=hashlib.sha256(canonical(content)).hexdigest())
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify(stage)
        stage.rename(output)
        return output
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ndk", type=Path)
    parser.add_argument("--glslc", type=Path)
    parser.add_argument("--spirv-val", type=Path)
    parser.add_argument("--cmake", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 64:
        parser.error("--jobs must be between 1 and 64")
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
    elif args.plan_only:
        print(host_tests(cmake=args.cmake, jobs=args.jobs))
    else:
        print(
            build(
                args.output,
                ndk=args.ndk,
                glslc=args.glslc,
                spirv_val=args.spirv_val,
                cmake=args.cmake,
                jobs=args.jobs,
            )
        )


if __name__ == "__main__":
    main()
