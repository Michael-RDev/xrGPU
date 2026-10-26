"""Regression checks for the authoritative native catalog and its consumers."""

# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from kernels import catalog, provenance

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kernels"


@pytest.fixture(scope="module")
def isolated_catalog(tmp_path_factory):
    destination = tmp_path_factory.mktemp("native-catalog") / "kernels"
    shutil.copytree(KERNELS, destination, ignore=shutil.ignore_patterns("__pycache__"))
    catalog.generate_registries(destination, check=True)
    return destination


@contextmanager
def temporary_bytes(path, data):
    original = path.read_bytes()
    try:
        path.write_bytes(data)
        yield
    finally:
        if path.is_symlink():
            path.unlink()
        path.write_bytes(original)


@contextmanager
def reindexed_contract(tree, identity, mutate):
    index = tree / "catalog.json"
    contract = tree / identity / "contract.json"
    value = json.loads(index.read_text())
    entry = next(e for e in value["entries"] if e["id"] == identity)
    document = json.loads(contract.read_text())
    mutate(document)
    encoded = json.dumps(document, sort_keys=True, indent=2).encode() + b"\n"
    for key in ("native", "variants", "provenance", "status"):
        entry[key] = document[key]
    entry["descriptor_digests"]["contract"] = hashlib.sha256(encoded).hexdigest()
    value["catalog_sha256"] = catalog.digest(
        {k: v for k, v in value.items() if k != "catalog_sha256"}
    )
    with temporary_bytes(contract, encoded), temporary_bytes(index, json.dumps(value).encode()):
        yield


def test_native_catalog_inventory_and_api_partition():
    value = catalog.load()
    legacy, extended = catalog.legacy_records(), catalog.extended_records()
    assert value["kind"] == "xrgpu.native-kernel-catalog"
    assert value["operator_count"] == 221
    assert value["test_count"] == 13919 and value["benchmark_count"] == 5839
    assert (value["backend"], value["target"], value["vulkan_min_version"]) == (
        "vulkan",
        "android-arm64",
        "1.1",
    )
    assert len(legacy) == 38 and len(extended) == 183
    assert {r["id"] for r in legacy}.isdisjoint({r["id"] for r in extended})
    assert {e["id"] for e in value["entries"]} == {e["id"] for e in provenance.catalog()["entries"]}
    assert {
        family: sum(r["family"] == family for r in extended)
        for family in ["tensor", "spatial", "complex"]
    } == {"tensor": 90, "spatial": 57, "complex": 36}


def test_catalog_native_entrypoints_cover_all_pass_dependencies():
    value = catalog.load()
    variants = catalog.shader_variants()
    assert len(variants) == len({v["output"] for v in variants}) == 315
    for variant in variants:
        assert catalog.regular(KERNELS, variant["source"]).suffix == ".comp"
        assert not variant["source"].startswith(("_shared/", "_provenance/"))
        assert all(path.startswith("_shared/") for path in variant["include_dirs"])
    for entry in value["entries"]:
        if entry["native"]["api"] == "extended":
            native = entry["native"]
            assert {v["output"] for v in entry["variants"]} == {
                "extended/" + s for s in [native["shader"], *native["shader_dependencies"]]
            }


def test_portable_contracts_and_fixture_bytes_preserve_provenance():
    for entry in catalog.load()["entries"]:
        contract = json.loads((KERNELS / entry["contract"]).read_text())
        original = provenance.document(entry["id"], "contract")
        assert contract["kind"] == "xrgpu.native-operator"
        assert not {"bindings", "tunables", "derive", "when", "templates"} & set(contract)
        assert contract["capabilities"]["cpu_fallback"] is False
        for native, upstream in [
            ("inputs", "inputs"),
            ("outputs", "outputs"),
            ("type_constraints", "typeConstraints"),
            ("attributes", "attributes"),
            ("attribute_constraints", "attributeConstraints"),
        ]:
            assert contract[native] == original.get(upstream, {})
        for key in ["tests", "benchmarks"]:
            descriptor = json.loads((KERNELS / entry[key]).read_text())
            source = catalog.regular(KERNELS, descriptor["fixture_source"])
            assert hashlib.sha256(source.read_bytes()).hexdigest() == descriptor["fixture_sha256"]
        assert contract["status"]["hardware"] == "unverified"


def test_generated_registries_are_consistent_and_drift_is_rejected(isolated_catalog):
    assert catalog.generate_registries(isolated_catalog, check=True) == 10
    path = isolated_catalog / "_shared/registry.def"
    with temporary_bytes(path, path.read_bytes() + b"// unexpected registry drift\n"):
        with pytest.raises(ValueError, match="registry drift"):
            catalog.generate_registries(isolated_catalog, check=True)


def test_catalog_identity_rejects_unchecksummed_mutation(isolated_catalog):
    path = isolated_catalog / "catalog.json"
    value = json.loads(path.read_text())
    value["backend"] = "webgpu"
    with temporary_bytes(path, json.dumps(value).encode()):
        with pytest.raises(ValueError, match="identity"):
            catalog.load(isolated_catalog)


def test_descriptor_digest_rejects_corruption(isolated_catalog):
    path = isolated_catalog / "ai.onnx.Add/test.json"
    with temporary_bytes(path, path.read_bytes() + b" "):
        with pytest.raises(ValueError, match="descriptor checksum"):
            catalog.load(isolated_catalog)


def test_rechecks_portable_semantics_even_after_catalog_is_rehashed(isolated_catalog):
    def mutate(document):
        document["type_constraints"]["T"] = ["float32"]

    with reindexed_contract(isolated_catalog, "ai.onnx.Add", mutate):
        with pytest.raises(ValueError, match="portable contract"):
            catalog.load(isolated_catalog)


@pytest.mark.parametrize(
    "path", ["../outside.comp", "/private/tmp/outside.comp", "a//b", "a/./b", "a/../b", "a\\b"]
)
def test_native_path_resolver_rejects_escape_and_ambiguous_names(tmp_path, path):
    with pytest.raises(ValueError, match="Unsafe"):
        catalog.regular(tmp_path, path)


def test_rehashed_shader_source_path_cannot_escape_catalog(isolated_catalog):
    def mutate(document):
        document["variants"][0]["source"] = "../outside.comp"

    with reindexed_contract(isolated_catalog, "ai.onnx.Add", mutate):
        with pytest.raises(ValueError, match="Unsafe|variant mapping"):
            catalog.load(isolated_catalog)


def test_shader_dependency_path_cannot_escape_native_inventory(isolated_catalog):
    def mutate(document):
        document["native"]["shader_dependencies"].append("../outside.spv")
        extra = copy.deepcopy(document["variants"][0])
        extra["output"] = "extended/../outside.spv"
        document["variants"].append(extra)

    with reindexed_contract(isolated_catalog, "com.microsoft.GroupQueryAttention", mutate):
        with pytest.raises(ValueError, match="Unsafe native shader output"):
            catalog.load(isolated_catalog)


def test_native_entrypoint_symlink_is_rejected(isolated_catalog, tmp_path):
    path = isolated_catalog / "ai.onnx.Add/shaders/kernel.comp"
    original = path.read_bytes()
    outside = tmp_path / "outside.comp"
    outside.write_bytes(original)
    try:
        path.unlink()
        path.symlink_to(outside)
        with pytest.raises(ValueError, match="symlinks"):
            catalog.load(isolated_catalog)
    finally:
        path.unlink()
        path.write_bytes(original)


def test_rehashed_variant_defines_cannot_drift_from_native_registry(isolated_catalog):
    def mutate(document):
        document["variants"][0]["defines"][0] = "DTYPE=99"

    with reindexed_contract(isolated_catalog, "ai.onnx.Add", mutate):
        with pytest.raises(ValueError, match="variant mapping"):
            catalog.load(isolated_catalog)
