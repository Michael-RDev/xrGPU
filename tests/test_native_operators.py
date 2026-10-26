"""Integrity and local registry checks for the independent native operator package."""

import hashlib
import json
from pathlib import Path

import pytest

from kernels._shared import build as native_build
from kernels._shared.build import PACKAGE_ID, canonical, registry, sha, verify


def package(root):
    (root / "shader.spv").write_bytes(b"immutable example")
    content = dict(
        schema_version=1, package_id=PACKAGE_ID, files={"shader.spv": sha(root / "shader.spv")}
    )
    manifest = dict(content, package_sha256=hashlib.sha256(canonical(content)).hexdigest())
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_native_inventory_detects_tampering(tmp_path):
    root = package(tmp_path)
    verify(root)
    (root / "shader.spv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify(root)


def test_native_inventory_detects_manifest_tampering(tmp_path):
    root = package(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["backend"] = "another backend"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        verify(root)


def test_native_inventory_rejects_uninventoried_nested_manifest(tmp_path):
    root = package(tmp_path)
    (root / "nested").mkdir()
    (root / "nested/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        verify(root)


def test_native_inventory_rejects_symlinks(tmp_path):
    root = package(tmp_path)
    (root / "linked").symlink_to(root / "shader.spv")
    with pytest.raises(ValueError, match="links"):
        verify(root)


def test_native_inventory_rejects_missing_file(tmp_path):
    root = package(tmp_path)
    (root / "shader.spv").unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify(root)


def test_registry_covers_core_contracts_with_valid_types_and_names():
    records = registry()
    by_id = {row["id"]: row for row in records}
    assert {"ai.onnx.Add", "ai.onnx.Relu", "ai.onnx.MatMul"}.issubset(by_id)
    assert len(by_id) == len(records)
    assert set(by_id["ai.onnx.MatMul"]["dtypes"]) == {"float32", "float16", "int32", "uint32"}
    assert by_id["ai.onnx.Add"]["input_names"] == ["a", "b"]
    assert by_id["ai.onnx.Relu"]["output_name"] == "y"


@pytest.mark.parametrize("packaged_source", [False, True], ids=["source-tree", "packaged-source"])
def test_build_installs_all_catalog_variants_at_requested_output(
    tmp_path, monkeypatch, packaged_source
):
    """Exercise snapshot, compile-job assembly and installation with inert tools."""
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    requested = tmp_path / "releases/final-release"
    ndk = tmp_path / "ndk"
    glslc, validator, cmake = [
        tmp_path / "tools" / name for name in ("glslc", "spirv-val", "cmake")
    ]
    prebuilt = ndk / "toolchains/llvm/prebuilt/test-host"
    prerequisites = {
        ndk / "build/cmake/android.toolchain.cmake": b"# inert toolchain fixture\n",
        prebuilt
        / "sysroot/usr/lib/aarch64-linux-android/libc++_shared.so": b"mock Android runtime",
        prebuilt / "NOTICE": b"mock Android runtime notice\n",
        glslc: b"mock shader compiler",
        validator: b"mock SPIR-V validator",
        cmake: b"mock CMake",
    }
    for path, data in prerequisites.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    variants = native_build.shader_variants()
    expected_outputs = {row["output"] for row in variants}
    compiled, validated, host_sources = set(), set(), []
    real_snapshot = native_build.snapshot

    def snapshot_with_existing_headers(destination, *, include_contracts):
        frozen = real_snapshot(destination, include_contracts=include_contracts)
        if packaged_source:
            # An immutable package source tree already contains these outputs.
            generated = Path(destination) / "kernels/_shared/generated"
            generated.mkdir(exist_ok=True)
            for name in ("shader_hashes.h", "extended_shader_hashes.h"):
                (generated / name).write_text("stale packaged-source hash header\n")
        return frozen

    def fake_command(*arguments, **kwargs):
        assert not kwargs
        executable, *args = arguments
        if Path(executable) == glslc:
            destination = Path(args[args.index("-o") + 1])
            assert Path(args[args.index("-o") - 1]).is_file()
            relative = destination.relative_to(
                destination.parents[1]
                if destination.parent.name == "extended"
                else destination.parent
            )
            compiled.add(relative.as_posix())
            destination.write_bytes(b"mock SPIR-V: " + relative.as_posix().encode())
        elif Path(executable) == validator:
            assert args[:2] == ["--target-env", "vulkan1.1"]
            destination = Path(args[2])
            assert destination.is_file()
            validated.add(destination)
        elif Path(executable) == cmake:
            if args[0] == "-S":
                assert (Path(args[1]) / "CMakeLists.txt").is_file()
                assert "-DANDROID_ABI=arm64-v8a" in args
                Path(args[args.index("-B") + 1]).mkdir(parents=True)
            else:
                assert args[0] == "--build"
                android = Path(args[1])
                for name in ("libxrgpu_operators.so", "libxrgpu_extended_operators.so"):
                    (android / name).write_bytes(b"mock Android shared library: " + name.encode())
        else:
            pytest.fail(f"Unexpected external command: {executable}")

    def fake_host_tests(build_dir, source, cmake_path, jobs):
        assert cmake_path == cmake
        assert jobs == 1
        for name in ("shader_hashes.h", "extended_shader_hashes.h"):
            header = (source / "generated" / name).read_text()
            assert "stale packaged-source" not in header
            assert "Generated" in header
        host_sources.append(source)
        return build_dir

    def fake_check_output(arguments, *, text):
        assert arguments == [str(glslc), "--version"]
        assert text
        return "mock glslc for packaging regression\n"

    monkeypatch.setattr(native_build, "snapshot", snapshot_with_existing_headers)
    monkeypatch.setattr(native_build, "command", fake_command)
    monkeypatch.setattr(native_build, "host_tests", fake_host_tests)
    monkeypatch.setattr(native_build.subprocess, "check_output", fake_check_output)
    result = native_build.build(
        output=requested, ndk=ndk, glslc=glslc, spirv_val=validator, cmake=cmake, jobs=1
    )

    assert result == requested
    assert (requested / "manifest.json").is_file()
    assert not (caller / variants[-1]["output"]).exists()
    assert list(caller.iterdir()) == []
    assert list(requested.parent.iterdir()) == [requested]
    manifest = verify(requested)
    assert manifest["shader_count"] == len(expected_outputs) == 315
    assert len(manifest["operations"]) == 221
    assert manifest["operation_counts"] == {"legacy": 38, "extended": 183}
    assert compiled == expected_outputs
    assert len(validated) == len(expected_outputs)
    assert len(host_sources) == 1
    assert {
        path.removeprefix("shaders/") for path in manifest["files"] if path.startswith("shaders/")
    } == expected_outputs
    assert {
        f"lib/android-arm64/{name}"
        for name in ("libxrgpu_operators.so", "libxrgpu_extended_operators.so", "libc++_shared.so")
    } <= manifest["files"].keys()
    assert "source/kernels/catalog.json" in manifest["files"]
    assert "source/kernels/_provenance/catalog.json" in manifest["files"]
    assert {name for name in manifest["files"] if name.startswith("source/operators/")} == {
        "source/operators/__init__.py"
    }
