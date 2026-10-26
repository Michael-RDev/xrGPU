"""Test faithful fixture parsing and comparison; native execution is separate."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kernels._shared import verify_upstream as fixtures


def tensor(values, shape=None, dtype="float32"):
    return dict(dtype=dtype, shape=[len(values)] if shape is None else shape,
                data=dict(kind="values", values=values))


def test_published_values_and_local_json_references_preserve_exact_int64():
    value = 2**63 - 1
    document = dict(fixtureArrays={"large": [str(value), str(-value)]})
    spec = tensor({"$ref": "#/fixtureArrays/large"}, [2], "int64")
    actual = fixtures.materialize(spec, document)
    assert actual.dtype == np.dtype("int64")
    assert list(map(int, actual)) == [value, -value]


@pytest.mark.parametrize("reference", ["https://example.invalid/values", "../values", "#/other/x", "#/fixtureArrays/a/b"])
def test_only_restricted_local_fixture_arrays_can_be_resolved(reference):
    with pytest.raises(ValueError):
        fixtures.resolve_values({"$ref": reference}, {})


def test_fixture_reference_cycles_and_missing_arrays_are_rejected():
    document = dict(fixtureArrays=dict(a={"$ref": "#/fixtureArrays/b"}, b={"$ref": "#/fixtureArrays/a"}))
    with pytest.raises(ValueError, match="Cyclic"):
        fixtures.resolve_values({"$ref": "#/fixtureArrays/a"}, document)
    with pytest.raises(ValueError, match="Missing"):
        fixtures.resolve_values({"$ref": "#/fixtureArrays/missing"}, {})


def test_explicit_generators_scalar_empty_cycle_constant_and_linspace():
    scalar = fixtures.materialize(tensor([2.5], []), {})
    assert scalar.shape == () and float(scalar) == 2.5
    assert fixtures.materialize(tensor([], [0, 3]), {}).shape == (0, 3)
    cycle = dict(dtype="int8", shape=[7], data=dict(kind="cycle", values=[-1, 2, 3]))
    np.testing.assert_array_equal(fixtures.materialize(cycle, {}), [-1, 2, 3, -1, 2, 3, -1])
    constant = dict(dtype="float16", shape=[2, 3], data=dict(kind="constant", value=0.5))
    np.testing.assert_array_equal(fixtures.materialize(constant, {}), np.full((2, 3), .5, np.float16))
    line = dict(dtype="int32", shape=[4], data=dict(kind="linspace", start=-2, end=2))
    np.testing.assert_array_equal(fixtures.materialize(line, {}), [-2, 0, 0, 2])


def test_unimplemented_generator_semantics_are_explicitly_unsupported():
    spec = dict(dtype="float32", shape=[4], data=dict(kind="fillFloat32", sinStep=.1, cosStep=.2))
    with pytest.raises(fixtures.UnsupportedFixture, match="fillFloat32"):
        fixtures.materialize(spec, {})
    spec["data"] = dict(kind="cycle", values=[1.0], nanStart=1, nanCount=1)
    with pytest.raises(fixtures.UnsupportedFixture, match="options"):
        fixtures.materialize(spec, {})


def test_values_must_match_shape_and_allocations_are_bounded():
    with pytest.raises(ValueError, match="count"):
        fixtures.materialize(tensor([1, 2], [3]), {})
    with pytest.raises(fixtures.UnsupportedFixture, match="limit"):
        fixtures.materialize(tensor([1], [fixtures.MAX_ELEMENTS + 1]), {})
    with pytest.raises(ValueError, match="shape"):
        fixtures.materialize(tensor([1], [True]), {})


def test_special_values_and_infinities_are_compared_without_nonfinite_json_metrics():
    spec = tensor(["NaN", "Infinity", "-Infinity", -0.0])
    values = fixtures.materialize(spec, {})
    assert not fixtures.compare(values, values)["passed"]
    comparison = fixtures.compare(values, values, allow_nan=True)
    assert comparison["passed"] and comparison["nonfinite_mismatches"] == 0
    json.dumps(comparison, allow_nan=False)
    wrong = values.copy()
    wrong[1] = -np.inf
    assert not fixtures.compare(wrong, values, allow_nan=True)["passed"]
    large = np.array([2**63 - 1], np.int64)
    assert not fixtures.compare(large, large - 1)["passed"]


def test_comparison_uses_output_policy_before_case_and_marks_missing_tolerance():
    policy = fixtures.tolerance(dict(tolerance=1e-6, relTolerance=.01, allowNaN=True), dict(tolerance=1))
    assert policy == dict(atol=1e-6, rtol=.01, allow_nan=True, tolerance_origin="output")
    assert fixtures.tolerance({}, dict(tolerance=.002))["atol"] == .002
    absent = fixtures.tolerance({}, {})
    assert absent["atol"] == absent["rtol"] == 0
    assert absent["tolerance_origin"] == "absent: exact default"


def relu_fixture():
    spec = dict(op="ai.onnx.Relu", inputs=["x"], mask=1)
    case = dict(name="explicit_oracle", inputs=dict(x=tensor([-2, 3])),
                outputs=dict(y=tensor([7, 8])))
    return spec, case


def simulated_process(monkeypatch, values, backend="software_vulkan"):
    def run(command, **kwargs):
        prefix = Path(command[4])
        array = np.asarray(values, dtype=np.float32)
        Path(str(prefix) + ".0.bin").write_bytes(array.tobytes())
        evidence = dict(backend=backend, physical_qualification=False, cpu_compute_fallback=False,
                        outputs=[dict(name="y", dtype="float32", shape=list(array.shape))])
        return SimpleNamespace(returncode=0, stdout=json.dumps(evidence), stderr="")
    monkeypatch.setattr(fixtures.subprocess, "run", run)


def test_explicit_expected_outputs_take_priority_over_computed_reference(monkeypatch, tmp_path):
    spec, case = relu_fixture()
    simulated_process(monkeypatch, [7, 8])
    result = fixtures.case_check(tmp_path, tmp_path / "runner", spec, case, {})
    assert result["status"] == "reference_conflict" and result["comparison"]["passed"]
    assert result["oracle"] == "published_explicit_output"
    simulated_process(monkeypatch, [0, 3])
    result = fixtures.case_check(tmp_path, tmp_path / "runner", spec, case, {}, failure_root=tmp_path / "failure")
    assert result["status"] == "reference_conflict"
    assert result["native_independent_comparison"]["passed"]
    assert (tmp_path / "failure/expected.bin").is_file()


def test_native_backend_identity_and_unsupported_fixtures_cannot_pass(monkeypatch, tmp_path):
    spec, case = relu_fixture()
    simulated_process(monkeypatch, [7, 8], backend="executorch_cpu")
    result = fixtures.case_check(tmp_path, tmp_path / "runner", spec, case, {})
    assert result["status"] == "failed" and "software Vulkan" in result["error"]
    case["inputs"]["x"]["data"] = dict(kind="fillFloat32", sinStep=.1, cosStep=.2)
    result = fixtures.case_check(tmp_path, tmp_path / "runner", spec, case, {})
    assert result["status"] == "unsupported" and "fillFloat32" in result["reason"]


def test_upstream_webgpu_skip_is_provenance_not_a_native_pass(monkeypatch, tmp_path):
    spec, case = relu_fixture()
    case["skipGpu"] = dict(category="permanent", reason="Upstream WebGPU limitation")
    case["outputs"]["y"] = tensor([0, 3])
    simulated_process(monkeypatch, [0, 3])
    result = fixtures.case_check(tmp_path, tmp_path / "runner", spec, case, {})
    assert result["status"] == "passed" and result["upstream_skip_gpu"] == case["skipGpu"]


def test_bad_published_floor_golden_is_a_reference_conflict(monkeypatch, tmp_path):
    spec = dict(op="ai.onnx.Floor", inputs=["x"], mask=1)
    case = dict(name="large_integer", inputs=dict(x=tensor([8388609.])),
                outputs=dict(y=tensor([8388610.])))
    simulated_process(monkeypatch, [8388609.])
    result = fixtures.case_check(tmp_path, tmp_path / "runner", spec, case, {}, failure_root=tmp_path / "floor")
    assert result["status"] == "reference_conflict"
    assert not result["comparison"]["passed"]
    assert result["native_independent_comparison"]["passed"]
    assert (tmp_path / "floor/independent-reference.bin").is_file()


def test_frozen_upstream_fixture_checksum_detects_local_changes(tmp_path):
    path = tmp_path / "tests.json"
    path.write_text('{"cases":[]}')
    entry = dict(tests="tests.json", files={"tests.json": fixtures.sha(path)})
    assert fixtures.load_document(tmp_path, entry) == dict(cases=[])
    path.write_text('{"cases":[{}]}')
    with pytest.raises(ValueError, match="checksum"):
        fixtures.load_document(tmp_path, entry)
