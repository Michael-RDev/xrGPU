"""Integrity and publication boundaries for the native xrML alpha distribution."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from hub import registry, snapshot
from kernels import catalog as native_catalog, provenance

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(snapshot.canonical(value) + b"\n")


@pytest.fixture
def native(tmp_path):
    """Protocol fixture only, never a claimed executable or hardware result."""
    root = tmp_path / "native"
    root.mkdir()
    files = {"include/xrgpu/operators.h": b"// test header\n",
             "lib/android-arm64/libxrgpu_operators.so": b"test-native-ELF-protocol-fixture",
             "shaders/add_f32.spv": b"test-spirv-protocol-fixture"}
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = dict(schema_version=1, package_id=registry.NATIVE_ID,
                    operations=["ai.onnx.Add"],
                    supported_contracts={"ai.onnx.Add": dict(dtypes=["float32"], attributes=[],
                                                             limits=dict(max_rank=16))},
                    files={name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
                    qualification="unqualified")
    manifest["package_sha256"] = snapshot.digest(manifest)
    write_json(root / "manifest.json", manifest)
    return root


def test_complete_frozen_catalog_has_immutable_sources_and_intact_cases():
    catalog = snapshot.verify(provenance.PROVENANCE)
    assert catalog["operator_count"] == 221
    assert catalog["test_count"] == 13919
    assert len({e["id"] for e in catalog["entries"]}) == 221
    assert all(e["repo_id"].startswith("webgpu-kernels/") for e in catalog["entries"])
    assert all(e["license"] == "Apache-2.0" for e in catalog["entries"])
    assert catalog["benchmark_count"] > 221
    for entry in catalog["entries"]:
        metadata = json.loads(provenance.regular(entry["metadata"]).read_text())
        for name, encoded in metadata["digest"]["files"].items():
            import base64
            assert base64.b64decode(encoded).hex() == entry["upstream_files"]["build/webgpu/" + name]
        assert not any(name.endswith((".wgsl", ".wgsl.jinja")) for name in entry["files"])


def test_snapshot_rejects_mutated_case_bytes(tmp_path):
    destination = tmp_path / "snapshot"
    shutil.copytree(provenance.PROVENANCE / "upstream", destination / "upstream")
    shutil.copyfile(provenance.PROVENANCE / "catalog.json", destination / "catalog.json")
    catalog = json.loads((destination / "catalog.json").read_text())
    path = destination / catalog["entries"][0]["tests"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        snapshot.verify(destination)


def test_prepared_inventory_rejects_injected_credentials_and_mutation(tmp_path):
    (tmp_path / "README.md").write_text("Native release")
    inventory = registry.inventory(tmp_path)
    assert registry.verify_prepared(tmp_path) == inventory
    (tmp_path / ".env").write_text("FAKE_TEST_TOKEN=fixture")
    with pytest.raises(ValueError, match="unlisted"):
        registry.verify_prepared(tmp_path)
    (tmp_path / ".env").unlink()
    (tmp_path / "README.md").write_text("Changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        registry.verify_prepared(tmp_path)


def test_prepared_inventory_never_accepts_a_symlink(tmp_path):
    (tmp_path / "README.md").write_text("Native release")
    (tmp_path / "linked").symlink_to(tmp_path / "README.md")
    with pytest.raises(ValueError, match="symlinks"):
        registry.inventory(tmp_path)


@pytest.mark.parametrize("path", ["../.env", "/tmp/secret", "a//b", "a/./b", "a\\b", "a/../b"])
def test_package_paths_cannot_escape_root(path):
    with pytest.raises(ValueError, match="Unsafe"):
        snapshot.safe_path(path)


def test_native_package_rejects_wrong_identity_and_tampering(native):
    manifest = registry.verify_native(native)
    assert manifest["package_id"] == registry.NATIVE_ID
    (native / "shaders/add_f32.spv").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="file digest mismatch"):
        registry.verify_native(native)
    manifest["package_id"] = "xrgpu.quest3.fp32.v1"
    manifest["package_sha256"] = snapshot.digest({k: v for k, v in manifest.items() if k != "package_sha256"})
    write_json(native / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Unsupported native"):
        registry.verify_native(native)


def test_source_only_native_packages_cannot_be_published(native):
    manifest = json.loads((native / "manifest.json").read_text())
    manifest["files"] = {name: value for name, value in manifest["files"].items() if not name.endswith(".so")}
    manifest["package_sha256"] = snapshot.digest({k: v for k, v in manifest.items() if k != "package_sha256"})
    write_json(native / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Native Android library required"):
        registry.verify_native(native)


def test_prepare_uses_native_catalog_without_restoring_optional_benchmark(native, tmp_path):
    output = tmp_path / "release"
    release = registry.prepare(native, output)
    assert registry.verify_release(output) == release
    assert [r["repo_id"] for r in release["repositories"]] == ["xrML/xrGPU-kernels", "xrML/ai.onnx.Add"]
    index = json.loads((output / "aggregate/registry.json").read_text())
    assert index["operator_count"] == 221 and index["implemented_count"] == 1
    assert "benchmark" not in index["packages"]
    assert not (output / "aggregate/kernels/benchmark").exists()
    assert native_catalog.load(output / "aggregate/kernels")["operator_count"] == 221
    subset = registry.verify_native(output / "operators/ai.onnx.Add/native")
    assert subset["origin_package_sha256"] == registry.verify_native(native)["package_sha256"]
    missing = [e for e in index["entries"] if e["status"]["implementation"] == "not_in_supplied_package"]
    assert len(missing) == 220 and all(e["repo_id"] is None for e in missing)
    assert all(e["status"]["hardware"] == "unverified" for e in index["entries"])
    assert not any(p.suffix in {".wgsl", ".js", ".mjs", ".html"} for p in output.rglob("*"))
    with pytest.raises(FileExistsError, match="immutable"):
        registry.prepare(native, output)
    (output / "aggregate/secret.env").write_text("unrelated private input")
    with pytest.raises(ValueError, match="unlisted"):
        registry.verify_release(output)


def test_registry_will_not_invent_implemented_operators():
    with pytest.raises(ValueError, match="implemented native"):
        registry.native_operations(dict(operations=["unknown.Unsupported"]))
    with pytest.raises(ValueError, match="Missing implemented"):
        registry.native_operations(dict(operations=["ai.onnx.Attention"]))


def test_explicit_existing_benchmark_package_preserves_identity_and_license(native, tmp_path):
    benchmark = tmp_path / "benchmark"
    files = {"LICENSE": b"BSD benchmark fixture license", "README.md": b"Protocol fixture", "kernel.comp": b"// Protocol fixture only"}
    for name, payload in files.items():
        path = benchmark / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = dict(schema_version=1, package_id=registry.BENCHMARK_ID,
                    files={name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()})
    write_json(benchmark / "manifest.json", manifest)
    output = tmp_path / "explicit-release"
    registry.prepare(native, output, benchmark_package=benchmark)
    index = json.loads((output / "aggregate/registry.json").read_text())
    assert index["packages"]["benchmark"]["package_id"] == registry.BENCHMARK_ID
    assert index["packages"]["benchmark"]["manifest_sha256"] == snapshot.digest(manifest)
    assert (output / "aggregate/kernels/benchmark/LICENSE").read_bytes() == files["LICENSE"]
    (benchmark / "kernel.comp").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="Benchmark source digest"):
        registry.verify_benchmark(benchmark)


def test_all_221_native_implementations_can_be_exported_without_placeholders(tmp_path):
    # Protocol bytes exercise packaging only and are never execution evidence.
    source = tmp_path / "native-all-protocol"
    catalog = native_catalog.load()
    operations = {}
    payloads = {"include/xrgpu/operators.h": b"// protocol only", "include/xrgpu/extended_operators.h": b"// protocol only",
                "lib/android-arm64/libxrgpu_operators.so": b"protocol only", "lib/android-arm64/libxrgpu_extended_operators.so": b"protocol only"}
    for entry in catalog["entries"]:
        native = entry["native"]
        operations[entry["id"]] = (dict(dtypes=native["registry"]["dtypes"]) if native["api"] == "legacy" else
                                    dict(native_api="extended", shader="extended/" + native["shader"]))
        for variant in entry["variants"]:
            payloads["shaders/" + variant["output"]] = b"protocol SPIR-V only"
    for name, payload in payloads.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = dict(schema_version=1, package_id=registry.NATIVE_ID, operations=list(operations),
                    supported_contracts=operations, files={name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()})
    manifest["package_sha256"] = snapshot.digest(manifest)
    write_json(source / "manifest.json", manifest)
    output = tmp_path / "all-native-release"
    release = registry.prepare(source, output)
    assert len(release["repositories"]) == 222
    assert registry.verify_release(output) == release
    index = json.loads((output / "aggregate/registry.json").read_text())
    assert index["implemented_count"] == index["operator_count"] == 221
    assert all(e["status"]["implementation"] == "implemented" for e in index["entries"])
    assert all(e["status"]["contract_conformance"] == "incomplete" for e in index["entries"])
    assert registry.verify_native(output / "operators/ai.onnx.Attention/native")["operations"] == ["ai.onnx.Attention"]


def test_provenance_resolver_rejects_symlink_roots_and_file_aliases(tmp_path):
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "alias.json").symlink_to(tmp_path / "data.json")
    with pytest.raises(ValueError, match="symlinks"):
        provenance.regular("alias.json", root=tmp_path)
    (tmp_path / "root-link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        provenance.regular("data.json", root=tmp_path / "root-link")
