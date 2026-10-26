import copy
import hashlib
import json

import pytest

from analysis.discovery import analyze, existing_operations, summarize_case


def case(operation="new_operation"):
    dispatches = [dict(kernel_name="nchw_to_image_texture3d_float", ms=.1),
                  dict(kernel_name="future_compute", ms=.3),
                  dict(kernel_name="view_texture_float", ms=.1)]
    sample = dict(model_ms=1., gpu_ms=.5, dispatch_count=3, dispatches=dispatches,
                  candidate_selected=False, kernel_variant="stock", thermal_status=0,
                  charging=False, profiling="trace")
    return dict(case="new_case", operation=operation, phase="isolated", trial=0,
                inputs=[dict(shape=[1, 7, 13], dtype="float32")], params={},
                status="passed", thermal_start=0, charging_start=False,
                correctness=dict(passed=True), correctness_before=dict(passed=True),
                correctness_after=dict(passed=True), samples=[copy.deepcopy(sample) for _ in range(3)])


def write_report(path, rows=None, **kwargs):
    report = dict(schema_version=2, run_id="one", backend="executorch_vulkan",
                  execution_location="headset", status="complete",
                  configuration=dict(mode="isolated"), results=rows or [case()])
    report.update(kwargs)
    path.write_text(json.dumps(report))
    return path


def test_discovers_unregistered_operations_and_preserves_scopes(tmp_path):
    path = write_report(tmp_path / "report.json")
    original = path.read_bytes()
    result = analyze([path])
    row = result["reports"][0]["eligible_cases"][0]
    assert row["operation"] == "new_operation"
    assert row["existing_replacement"] is False
    assert row["timings"]["target_ms"]["median"] == pytest.approx(.4)
    assert row["timings"]["staging_ms"]["median"] == pytest.approx(.1)
    assert row["timings"]["model_ms"]["median"] == 1.
    assert result["reports"][0]["raw_sha256"] == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original


def test_ranks_gpu_targets_instead_of_host_overhead(tmp_path):
    slow_host, slow_gpu = case(), case()
    slow_host["case"] = "host"
    slow_gpu["case"] = "gpu"
    for sample in slow_host["samples"]:
        sample["model_ms"] = 100.
    for sample in slow_gpu["samples"]:
        sample["dispatches"][1]["ms"] = .6
        sample["gpu_ms"] = .8
    rows = analyze([write_report(tmp_path / "report.json", [slow_host, slow_gpu])])["reports"][0]
    assert [r["case"] for r in rows["eligible_cases"]] == ["gpu", "host"]


@pytest.mark.parametrize("change", [
    lambda r: r.update(status="failed"),
    lambda r: r.update(timing_valid=False),
    lambda r: r["correctness_after"].update(passed=False),
    lambda r: r.pop("correctness_before"),
    lambda r: r["samples"][1].update(thermal_status=1),
    lambda r: r["samples"][1].update(charging=True),
    lambda r: r["samples"][1].pop("charging"),
    lambda r: r["samples"][0].update(candidate_selected=True),
    lambda r: r["samples"][0].update(dispatch_count=4),
    lambda r: r["samples"][0].update(model_ms=float("nan")),
    lambda r: r["samples"][0]["dispatches"][0].update(ms=-1),
    lambda r: r["samples"][0].update(gpu_ms=10),
])
def test_excludes_invalid_or_incomparable_measurements(change):
    row = case()
    change(row)
    result = summarize_case(row)
    assert not result["eligible"]
    assert result["reasons"]
    assert "timings" not in result


def test_does_not_pool_sessions_or_count_duplicate_reports_twice(tmp_path):
    a = write_report(tmp_path / "a.json")
    b = write_report(tmp_path / "b.json", run_id="two", device=dict(driver=2))
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(a.read_bytes())
    result = analyze([a, b, duplicate])
    assert len(result["reports"]) == 2
    assert all(r["eligible_cases"][0]["sample_count"] == 3 for r in result["reports"])
    duplicate.write_text(duplicate.read_text().replace('"trial": 0', '"trial": 1'))
    with pytest.raises(ValueError, match="Conflicting original"):
        analyze([a, duplicate])


def test_partial_runs_and_cpu_are_not_ranked(tmp_path):
    partial = write_report(tmp_path / "partial.json", status="canceled")
    cpu = write_report(tmp_path / "cpu.json", run_id="cpu", backend="executorch_cpu")
    result = analyze([partial, cpu])
    assert result["reports"] == []
    assert len(result["ignored_reports"]) == 2


def test_verifies_archived_bytes(tmp_path):
    path = write_report(tmp_path / "report.json")
    (tmp_path / "source.json").write_text(json.dumps(dict(raw_sha256="wrong")))
    with pytest.raises(ValueError, match="checksum mismatch"):
        analyze([path])


def test_absent_kernel_contributions_include_zero_samples():
    row = case()
    row["samples"][1]["dispatches"][1]["kernel_name"] = "rare_kernel"
    result = summarize_case(row)
    kernels = {k["name"]: k for k in result["kernels"]}
    assert kernels["rare_kernel"]["timings"]["count"] == 3
    assert kernels["rare_kernel"]["timings"]["median"] == 0
    assert kernels["future_compute"]["timings"]["median"] == .3


def test_native_catalog_is_not_a_model_replacement_registration(tmp_path):
    kernels = tmp_path / "kernels"
    for folder, contract in {
        "linear": {"kind": "xrgpu.model-override", "operation": "linear"},
        "ai.onnx.Add": {"kind": "xrgpu.native-operator", "id": "ai.onnx.Add", "operation": "new_operation"},
        "unmarked": {"operation": "unmarked"},
    }.items():
        path = kernels / folder
        path.mkdir(parents=True)
        (path / "contract.json").write_text(json.dumps(contract))
    assert existing_operations(tmp_path) == ["linear"]
    report = write_report(tmp_path / "report.json", [case("new_operation"), case("linear")])
    result = analyze([report], root=tmp_path)
    found = {r["operation"]: r["existing_replacement"] for r in result["reports"][0]["eligible_cases"]}
    assert found == {"new_operation": False, "linear": True}


def test_existing_model_override_discovery_is_preserved():
    assert existing_operations() == ["gelu", "grouped_conv2d", "linear"]
