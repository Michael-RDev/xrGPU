import hashlib
import json
from pathlib import Path
import shutil

import pytest

from runtime.model_bundle import bind_package, canonical, export_bundle, runtime_revision, verify_bundle, verify_cpu_reference


def write_manifest(path, **changes):
    manifest = json.loads((path / "bundle.json").read_text())
    manifest.update(changes)
    manifest.pop("bundle_sha256", None)
    manifest["bundle_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    (path / "bundle.json").write_bytes(canonical(manifest))


@pytest.fixture(scope="module")
def prepared_bundle(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("executorch")
    torch.set_num_threads(1)

    class Model(torch.nn.Module):
        def forward(self, values, ids):
            return values * 2 + 1, ids + 3

    target = tmp_path_factory.mktemp("typed-model") / "model"
    export_bundle(Model(), (torch.tensor([[0.25, -0.5]]), torch.tensor([[1, 2]])), target,
                  model_id="test/static", model_revision="fixture-v1", seed=11,
                  input_names=["values", "ids"], output_names=["result", "indices"],
                  additional_fixtures=[(12, (torch.tensor([[3.0, -2.0]]), torch.tensor([[3, 4]])))])
    return target


@pytest.fixture
def bundle(prepared_bundle, tmp_path):
    target = tmp_path / "model"
    shutil.copytree(prepared_bundle, target)
    return target


def test_export_reload_preserves_typed_inputs_outputs_and_fixed_weight_fixtures(bundle):
    manifest = verify_bundle(bundle, expected_runtime=runtime_revision())
    assert [item["dtype"] for item in manifest["inputs"]] == ["float32", "int64"]
    assert [item["dtype"] for item in manifest["outputs"]] == ["float32", "int64"]
    assert [item["seed"] for item in manifest["fixtures"]] == [11, 12]
    assert "cpu" in manifest["programs"]
    assert verify_cpu_reference(bundle)["passed"] is True
    for fixture in manifest["fixtures"]:
        for entry in fixture["inputs"] + fixture["outputs"]:
            assert entry["file"] in manifest["files"]


def test_corrupt_program_is_rejected_before_load(bundle):
    (bundle / "cpu.pte").write_bytes(b"altered")
    with pytest.raises(ValueError, match="file mismatch"):
        verify_cpu_reference(bundle)


@pytest.mark.parametrize("changes", [
    {"runtime_revision": "different-runtime"},
    {"inputs": [{"name": "values", "shape": [-1, 2], "dtype": "float32"}]},
    {"outputs": [{"name": "output", "shape": [1], "dtype": "float16"}]},
    {"qualification": {"status": "qualified"}},
    {"programs": {"vulkan": {"file": "cpu.pte"}}},
])
def test_invalid_contract_or_runtime_is_rejected(bundle, changes):
    write_manifest(bundle, **changes)
    with pytest.raises(ValueError):
        verify_bundle(bundle, expected_runtime=runtime_revision())


def test_file_manifest_rejects_escape_and_symlink(bundle, tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"external")
    write_manifest(bundle, files={"../outside": hashlib.sha256(b"external").hexdigest()})
    with pytest.raises(ValueError, match="Invalid bundle path"):
        verify_bundle(bundle)
    (bundle / "alias").symlink_to(outside)
    write_manifest(bundle, files={"alias": hashlib.sha256(b"external").hexdigest()})
    with pytest.raises(ValueError, match="Symlinks"):
        verify_bundle(bundle)


def test_failed_replacement_preserves_previous_bundle(bundle, monkeypatch):
    import torch
    from runtime import model_bundle
    original = (bundle / "bundle.json").read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(model_bundle, "_lower", fail)
    with pytest.raises(RuntimeError, match="export failed"):
        export_bundle(torch.nn.Identity(), (torch.ones(1),), bundle, model_id="new",
                      model_revision="v1", overwrite=True)
    assert (bundle / "bundle.json").read_bytes() == original


def test_vulkan_export_failure_preserves_checked_cpu_fallback(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("executorch")
    from runtime import model_bundle
    lower = model_bundle._lower

    def partial(module, inputs, destination, vulkan):
        if vulkan:
            raise RuntimeError("unsupported device graph")
        return lower(module, inputs, destination, vulkan)

    monkeypatch.setattr(model_bundle, "_lower", partial)
    manifest = export_bundle(torch.nn.ReLU(), (torch.tensor([-1.0, 2.0]),), tmp_path / "fallback",
                             model_id="fallback", model_revision="v1")
    assert set(manifest["programs"]) == {"cpu"}
    assert "unsupported device graph" in manifest["export_errors"]["vulkan"]


def test_manifest_tampering_is_rejected(bundle):
    manifest = json.loads((bundle / "bundle.json").read_text())
    manifest["model_revision"] = "tampered"
    (bundle / "bundle.json").write_bytes(canonical(manifest))
    with pytest.raises(ValueError, match="manifest digest"):
        verify_bundle(bundle)


def test_reference_wrong_answer_is_rejected_even_with_valid_file_hashes(bundle):
    manifest = verify_bundle(bundle)
    target = bundle / manifest["fixtures"][1]["outputs"][0]["file"]
    target.write_bytes(b"\0" * target.stat().st_size)
    files = dict(manifest["files"])
    files[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_manifest(bundle, files=files)
    with pytest.raises(ValueError, match="CPU reference mismatch"):
        verify_cpu_reference(bundle)


def test_reused_fixture_seeds_are_rejected(bundle):
    manifest = verify_bundle(bundle)
    fixtures = manifest["fixtures"]
    fixtures[1]["seed"] = fixtures[0]["seed"]
    write_manifest(bundle, fixtures=fixtures)
    with pytest.raises(ValueError, match="unique integers"):
        verify_bundle(bundle)


def write_kernel_package(directory, revision=None):
    directory.mkdir()
    payload = b"unqualified test package payload"
    (directory / "payload").write_bytes(payload)
    manifest = dict(schema_version=1, package_id="xrgpu.quest3.fp32.v1", selections=[],
                    runtime_revision=revision or runtime_revision(),
                    files={"payload": hashlib.sha256(payload).hexdigest()})
    manifest["package_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    (directory / "manifest.json").write_bytes(canonical(manifest))
    return manifest


def test_package_binding_preserves_model_programs_weights_and_fixtures(bundle, tmp_path):
    original = verify_bundle(bundle)
    original_bytes = (bundle / "bundle.json").read_bytes()
    package = tmp_path / "package"
    kernel = write_kernel_package(package)
    target = tmp_path / "bound"
    bound = bind_package(bundle, package, target)
    assert bound["files"] == original["files"]
    assert bound["programs"] == original["programs"]
    assert bound["fixtures"] == original["fixtures"]
    assert bound["kernel_package"] == {key: kernel[key] for key in ("package_id", "package_sha256")}
    assert bound["bundle_sha256"] != original["bundle_sha256"]
    assert (bundle / "bundle.json").read_bytes() == original_bytes
    assert verify_cpu_reference(target)["passed"] is True


def test_package_binding_rejects_runtime_mismatch(bundle, tmp_path):
    package = tmp_path / "package"
    write_kernel_package(package, revision="different-runtime")
    target = tmp_path / "bound"
    with pytest.raises(ValueError, match="runtime revisions differ"):
        bind_package(bundle, package, target)
    assert not target.exists()


def test_package_binding_rejects_corrupt_package(bundle, tmp_path):
    package = tmp_path / "package"
    write_kernel_package(package)
    (package / "payload").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Package file mismatch"):
        bind_package(bundle, package, tmp_path / "bound")


def test_package_binding_requires_separate_destination(bundle):
    with pytest.raises(ValueError, match="separate destination"):
        bind_package(bundle, bundle, bundle)


@pytest.mark.parametrize("name", ["minilm-32", "minilm-128", "minilm-256", "mobilenetv2"])
def test_available_pinned_model_bundles_execute_cpu_references(name):
    target = Path(__file__).resolve().parents[1] / "artifacts/model-bundles" / name
    if not target.exists():
        pytest.skip("Run python -m models.export --offline to prepare pinned examples")
    pytest.importorskip("torch")
    pytest.importorskip("executorch")
    manifest = verify_bundle(target)
    assert len(manifest["fixtures"]) >= 3
    assert verify_cpu_reference(target)["passed"] is True


def test_cached_mobilenet_weights_and_model_contract():
    torch = pytest.importorskip("torch")
    from models import mobilenetv2
    weights = mobilenetv2.ROOT / "artifacts/cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth"
    if not weights.is_file():
        pytest.skip("Pinned MobileNetV2 weights not cached")
    torch.set_num_threads(1)
    model = mobilenetv2.load(offline=True)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3504872
    with torch.inference_mode():
        output = model(*mobilenetv2.fixture(17))
    assert output.shape == (1, 1000)
    assert torch.isfinite(output).all()


def test_cached_minilm_matches_preserved_pretrained_reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from models import minilm
    from runtime.model_bundle import ROOT
    reference = ROOT / "artifacts/bundle/references.json"
    weights = (ROOT / "artifacts/cache/huggingface/hub"
               / "models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
               / minilm.REVISION / "model.safetensors")
    if not weights.exists() or not reference.exists():
        pytest.skip("Pinned MiniLM weights and preserved independent references not cached")
    torch.set_num_threads(1)
    model, tokenizer = minilm.load(offline=True)
    entries = json.loads(reference.read_text())
    for length in minilm.SEQUENCE_LENGTHS:
        case = next(entry for entry in entries if entry["sequence_length"] == length)
        arguments = minilm.inputs(tokenizer, case["text"], length)
        assert arguments[0][0].tolist() == case["input_ids"]
        assert arguments[1][0].tolist() == case["attention_mask"]
        with torch.inference_mode():
            actual = model(*arguments)[0]
        assert torch.allclose(actual, torch.tensor(case["embedding"]), atol=1e-5, rtol=1e-4)
        assert abs(torch.linalg.vector_norm(actual).item() - 1) < 1e-5
