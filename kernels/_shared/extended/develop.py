"""Build an isolated native software-Vulkan runner without mutating another run.

Normal development compiles only catalog-selected native entrypoints. --package
copies verified immutable package sources and reuses their inventoried SPIR-V;
no package source or shader is modified or recompiled in that mode.
"""

# SPDX-License-Identifier: Apache-2.0
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from kernels._shared.build import (
    ROOT,
    KERNELS,
    TOOLCHAIN,
    extended_registry,
    find_cmake,
    sha,
    shader_variants,
    snapshot,
    verify,
)


def develop(
    output,
    *,
    jobs=4,
    cmake=None,
    glslc=None,
    vulkan_headers=None,
    json_headers=None,
    families=None,
    operators=None,
    package=None,
):
    output = Path(output).absolute()
    if output.resolve().is_relative_to(KERNELS.resolve()):
        raise ValueError("Development output cannot be inside native kernel sources")
    supplied_package = Path(package).absolute() if package else None
    manifest = verify(supplied_package) if supplied_package else None
    if supplied_package and output.resolve().is_relative_to(supplied_package.resolve()):
        raise ValueError("Development output cannot modify an immutable package")
    if supplied_package and (families or operators):
        raise ValueError(
            "Package runners use its complete immutable shader inventory; omit selectors"
        )
    output.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    if supplied_package:
        shutil.copytree(supplied_package / "source", stage / "source")
    else:
        snapshot(stage / "source")
    source = stage / "source/kernels/_shared"
    if not source.is_dir():
        raise ValueError("This runner builder requires a native kernels-layout source snapshot")
    generated = source / "generated"
    generated.mkdir(exist_ok=True)
    shaders = stage / "shaders/extended"
    if not supplied_package:
        shaders.mkdir(parents=True)
    glslc = Path(glslc or TOOLCHAIN / "shaderc/2026.4/bin/glslc")
    vulkan_headers = Path(
        vulkan_headers
        or ROOT.parent
        / "xrBench/artifacts/deps/executorch/backends/vulkan/third-party/Vulkan-Headers/include"
    )
    json_headers = Path(
        json_headers
        or ROOT.parent / "xrBench/artifacts/deps/executorch/third-party/json/single_include"
    )
    log_path = stage / "build.log"
    try:
        with log_path.open("w") as log:
            records = extended_registry(source)
            if families:
                records = [row for row in records if row["family"] in families]
            if operators:
                known = {row["id"] for row in records}
                if set(operators) - known:
                    raise ValueError(
                        "Unknown native operator selection: "
                        + ",".join(sorted(set(operators) - known))
                    )
                records = [row for row in records if row["id"] in operators]
            needed = {
                "extended/" + name
                for row in records
                for name in [row["shader"], *row["shader_dependencies"]]
            }
            variants = [row for row in shader_variants(source) if row["output"] in needed]
            if {row["output"] for row in variants} != needed:
                raise ValueError("Mapped native catalog shader entrypoint is missing")
            if supplied_package:
                shader_records = [
                    (name.removeprefix("shaders/extended/"), digest)
                    for name, digest in sorted(manifest["files"].items())
                    if name.startswith("shaders/extended/") and name.endswith(".spv")
                ]
                if {"extended/" + name for name, _ in shader_records} != needed:
                    raise ValueError(
                        "Package and native catalog extended shader inventories differ"
                    )
            else:

                def compile_shader(variant):
                    destination = stage / "shaders" / variant["output"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        str(glslc),
                        "--target-env=vulkan1.1",
                        "-O",
                        *["-D" + value for value in variant["defines"]],
                        *["-I" + str(source.parent / path) for path in variant["include_dirs"]],
                        str(source.parent / variant["source"]),
                        "-o",
                        str(destination),
                    ]
                    subprocess.run(command, check=True, stdout=log, stderr=log)

                with ThreadPoolExecutor(max_workers=jobs) as pool:
                    list(pool.map(compile_shader, variants))
                shader_records = [(path.name, sha(path)) for path in sorted(shaders.glob("*.spv"))]
                if {"extended/" + name for name, _ in shader_records} != needed:
                    raise ValueError("Compiled native development shader inventory differs")
                (generated / "extended_shader_hashes.h").write_text(
                    "#pragma once\nnamespace xrgpu::operators::extended::detail {\n"
                    "struct ExtendedShaderHash {const char* name;const char* sha256;};\n"
                    "inline constexpr ExtendedShaderHash kExtendedShaderHashes[]={"
                    + ",".join(
                        '{"' + name + '","' + digest + '"}' for name, digest in shader_records
                    )
                    + "};}\n"
                )
            command = [
                str(cmake or find_cmake()),
                "-S",
                str(source),
                "-B",
                str(stage / "build"),
                "-DCMAKE_BUILD_TYPE=Debug",
                "-DBUILD_TESTING=OFF",
                "-DXRGPU_OPERATORS_TEST_SOFTWARE_VULKAN=ON",
                "-DXRGPU_OPERATORS_BUILD_EXTENDED_TEST_RUNNER=ON",
                "-DXRGPU_OPERATORS_BUILD_LEGACY_RUNTIME=" + ("ON" if supplied_package else "OFF"),
                "-DXRGPU_VULKAN_INCLUDE_DIR=" + str(vulkan_headers),
                "-DXRGPU_JSON_INCLUDE_DIR=" + str(json_headers),
            ]
            subprocess.run(command, check=True, stdout=log, stderr=log)
            targets = ["xrgpu_extended_execute"]
            if supplied_package:
                targets.append("xrgpu_operator_execute")
            subprocess.run(
                [
                    str(cmake or find_cmake()),
                    "--build",
                    str(stage / "build"),
                    "--target",
                    *targets,
                    "-j",
                    str(jobs),
                ],
                check=True,
                stdout=log,
                stderr=log,
            )
        source_files = {
            path.relative_to(stage / "source").as_posix(): sha(path)
            for path in sorted((stage / "source").rglob("*"))
            if path.is_file()
        }
        source_digest = hashlib.sha256(
            json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        result = dict(
            package=str(supplied_package or stage),
            runner=str(stage / "build/xrgpu_extended_execute"),
            source=str(stage / "source"),
            source_sha256=source_digest,
            runner_sha256=sha(stage / "build/xrgpu_extended_execute"),
            shader_sha256=dict(shader_records),
            development=True,
            physical_qualification=False,
            build_log=str(log_path),
            families=families or ["tensor", "spatial", "complex"],
            operators=operators,
            shaders_recompiled=supplied_package is None,
        )
        if supplied_package:
            verify(supplied_package)
            result.update(
                package_sha256=manifest["package_sha256"],
                legacy_runner=str(stage / "build/xrgpu_operator_execute"),
                legacy_runner_sha256=sha(stage / "build/xrgpu_operator_execute"),
            )
        (stage / "development.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception:
        if log_path.exists():
            print("\n".join(log_path.read_text(errors="replace").splitlines()[-30:]))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--package",
        type=Path,
        help="Build exact immutable package runtimes without recompiling SPIR-V",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cmake", type=Path)
    parser.add_argument("--glslc", type=Path)
    parser.add_argument("--vulkan-headers", type=Path)
    parser.add_argument("--json-headers", type=Path)
    parser.add_argument("--family", choices=["tensor", "spatial", "complex"], action="append")
    parser.add_argument("--operator", action="append")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 64:
        parser.error("--jobs must be 1..64")
    print(
        json.dumps(
            develop(
                args.output,
                jobs=args.jobs,
                cmake=args.cmake,
                glslc=args.glslc,
                vulkan_headers=args.vulkan_headers,
                json_headers=args.json_headers,
                families=args.family,
                operators=args.operator,
                package=args.package,
            )
        )
    )


if __name__ == "__main__":
    main()
