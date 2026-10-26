"""Native catalog completeness and frozen contract provenance checks."""

# SPDX-License-Identifier: Apache-2.0
import hashlib
import json


from kernels import catalog, provenance
from kernels._shared.build import ROOT, extended_registry, registry, snapshot_contracts


def test_every_frozen_operator_has_exactly_one_native_implementation():
    frozen = provenance.catalog()
    expected = {entry["id"] for entry in frozen["entries"]}
    legacy = {entry["id"] for entry in registry()}
    extended = {entry["id"] for entry in extended_registry()}
    assert len(expected) == 221
    assert len(legacy) == 38 and len(extended) == 183
    assert not legacy & extended
    assert legacy | extended == expected


def test_all_mapped_native_passes_have_original_glsl_sources():
    variants = {v["output"]: v for v in catalog.shader_variants()}
    for row in extended_registry():
        for name in [row["shader"], *row["shader_dependencies"]]:
            source = catalog.regular(ROOT / "kernels", variants["extended/" + name]["source"])
            assert source.suffix == ".comp", (row["id"], name)
            assert "SPDX-License-Identifier: Apache-2.0" in source.read_text()


def test_frozen_native_contract_table_retains_exact_local_provenance():
    native = json.loads((ROOT / "kernels/_shared/extended/contracts.json").read_text())
    frozen = provenance.catalog()
    assert native["upstream_catalog_sha256"] == frozen["catalog_sha256"]
    by_id = {entry["id"]: entry for entry in frozen["entries"]}
    assert {row["id"] for row in native["operations"]} == set(by_id)
    for row in native["operations"]:
        upstream = by_id[row["id"]]
        contract = provenance.regular(upstream["contract"])
        assert row["contract_sha256"] == hashlib.sha256(contract.read_bytes()).hexdigest()
        assert row["revision"] == upstream["revision"]
        original = json.loads(contract.read_text())
        assert row["inputs"] == original.get("inputs", {})
        assert row["outputs"] == original.get("outputs", {})
        assert row["type_constraints"] == original.get("typeConstraints", {})


# Native dependency/path rejection is exercised against a complete catalog in
# test_native_catalog.py, including checksummed adversarial descriptor changes.


def test_standalone_frozen_snapshot_is_complete_and_allowlisted(tmp_path):
    catalog = snapshot_contracts(tmp_path)
    root = tmp_path / "kernels/_provenance"
    files = {file.relative_to(root).as_posix() for file in root.rglob("*") if file.is_file()}
    expected = {"catalog.json", "LICENSE", "NOTICE"}
    for entry in catalog["entries"]:
        expected.update(entry["files"])
    assert files == expected
    assert len(files) == 1108
    assert not any(name.endswith((".wgsl", ".jinja", ".js", ".ts", ".env")) for name in files)
