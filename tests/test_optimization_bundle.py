import pytest

from runtime.build import selected_manifest


def test_selected_bundle_preserves_baseline_and_hashes_without_mutating_exports():
    manifest = {
        "files": {"minilm_vulkan_128.pte": "original", "references.json": "reference",
                  "minilm_opt_a_128.pte": "a", "minilm_opt_a_128_audit.pte": "audit",
                  "minilm_opt_a_128.graph.txt": "graph", "minilm_opt_b_128.pte": "b"},
        "optimization_profiles": {
            "a": {"lengths": {"128": {"model": "minilm_opt_a_128.pte",
                                     "audit_model": "minilm_opt_a_128_audit.pte"}}},
            "b": {"lengths": {"128": {"model": "minilm_opt_b_128.pte",
                                     "audit_model": "minilm_opt_b_128_audit.pte"}}}},
    }
    selected = selected_manifest(manifest, ["a"])
    assert selected["files"]["minilm_vulkan_128.pte"] == "original"
    assert selected["files"]["references.json"] == "reference"
    assert selected["files"]["minilm_opt_a_128_audit.pte"] == "audit"
    assert selected["files"]["minilm_opt_a_128.graph.txt"] == "graph"
    assert "minilm_opt_b_128.pte" not in selected["files"]
    assert "b" in manifest["optimization_profiles"]
    assert "minilm_opt_b_128.pte" in manifest["files"]
    with pytest.raises(ValueError):
        selected_manifest(manifest, ["missing"])
