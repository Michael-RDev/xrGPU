# SPDX-License-Identifier: Apache-2.0
"""Slice reference semantics and frozen-fixture conflict reporting."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kernels import provenance
from kernels._shared.extended import verify_frozen


@pytest.fixture(scope="module")
def oracle():
    pytest.importorskip("scipy.special")
    from kernels._shared.extended.tensor_oracle import oracle

    return oracle


@pytest.mark.parametrize("control_type", [np.int32, np.int64])
@pytest.mark.parametrize(
    "size,start,end,step,expected",
    [
        (3, 2, 2**31 - 1, -1, []),
        (3, 2**31 - 1, -(2**31), -1, [2, 1, 0]),
        (3, 2, -(2**31), -2, [2, 0]),
        (3, 2, -1, -1, []),
        (3, -1, -3, -1, [2, 1]),
        (3, -(2**31), -(2**31), -1, [0]),
        (3, -100, 100, 2, [0, 2]),
        (3, 0, 2**31 - 1, 2**31 - 1, [0]),
        (3, 2, -(2**31), -(2**31), [2]),
        (0, 2**31 - 1, -(2**31), -1, []),
        (0, 0, 2**31 - 1, 1, []),
    ],
)
def test_slice_bounds(oracle, control_type, size, start, end, step, expected):
    inputs = dict(
        data=np.arange(size, dtype=np.float32),
        starts=np.array([start], dtype=control_type),
        ends=np.array([end], dtype=control_type),
        steps=np.array([step], dtype=control_type),
    )
    actual = oracle("ai.onnx.Slice", inputs, outputs={"output": {}})["output"]
    np.testing.assert_array_equal(actual, np.array(expected, dtype=np.float32))


def test_slice_full_width_int64_controls(oracle):
    inputs = dict(
        data=np.array([10, 20, 30], dtype=np.int64),
        starts=np.array([2**63 - 1], dtype=np.int64),
        ends=np.array([-(2**63)], dtype=np.int64),
        steps=np.array([-(2**63)], dtype=np.int64),
    )
    actual = oracle("ai.onnx.Slice", inputs, outputs={"output": {}})["output"]
    np.testing.assert_array_equal(actual, [30])


def test_slice_normalizes_multiple_and_negative_axes(oracle):
    inputs = dict(
        data=np.arange(12, dtype=np.int64).reshape(3, 4),
        starts=np.array([-1, -100], dtype=np.int32),
        ends=np.array([-(2**31), 2**31 - 1], dtype=np.int32),
        axes=np.array([-1, 0], dtype=np.int32),
        steps=np.array([-2, 2], dtype=np.int32),
    )
    actual = oracle("ai.onnx.Slice", inputs, outputs={"output": {}})["output"]
    np.testing.assert_array_equal(actual, [[3, 1], [11, 9]])


def test_slice_default_axes_and_steps(oracle):
    inputs = dict(
        data=np.arange(12, dtype=np.float32).reshape(3, 4),
        starts=np.array([0, 1], dtype=np.int32),
        ends=np.array([2, 4], dtype=np.int32),
    )
    actual = oracle("ai.onnx.Slice", inputs, outputs={"output": {}})["output"]
    np.testing.assert_array_equal(actual, [[1, 2, 3], [5, 6, 7]])


def test_slice_rejects_zero_step(oracle):
    inputs = dict(
        data=np.arange(3, dtype=np.float32),
        starts=np.array([0], dtype=np.int32),
        ends=np.array([3], dtype=np.int32),
        steps=np.array([0], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="step is zero"):
        oracle("ai.onnx.Slice", inputs)


@pytest.fixture(scope="module")
def frozen_slice():
    catalog = provenance.catalog()
    entry = next(e for e in catalog["entries"] if e["id"] == "ai.onnx.Slice")
    return (
        dict(catalog, entries=[entry]),
        provenance.document(entry, "contract"),
        provenance.document(entry, "tests"),
    )


@pytest.fixture
def check_frozen_slice(monkeypatch, tmp_path, oracle, frozen_slice):
    catalog, manifest, document = frozen_slice
    monkeypatch.setattr(verify_frozen, "frozen_catalog", lambda _: catalog)
    monkeypatch.setattr(verify_frozen.provenance, "document", lambda *_: manifest)
    monkeypatch.setattr(verify_frozen, "reference_for", lambda _: oracle)
    runner = tmp_path / "runner"
    runner.write_bytes(b"simulated native test adapter")

    def check(case, values=None, backend="software_vulkan"):
        monkeypatch.setattr(
            verify_frozen, "load_document", lambda *_: dict(document, cases=[case])
        )

        def run(command, **kwargs):
            if values is None:
                pytest.fail("Conflicting output shapes must be detected before native execution")
            job = json.loads(Path(command[1]).read_text())
            actual = np.asarray(values, dtype=np.float32)
            assert job["outputs"][0]["shape"] == list(actual.shape)
            path = tmp_path / "actual.bin"
            actual.tofile(path)
            answer = dict(
                info=dict(backend=backend, physical_qualification=False),
                outputs=[dict(name="output", storage_dtype="float32",
                              shape=list(actual.shape), file=str(path))],
            )
            return SimpleNamespace(returncode=0, stdout=json.dumps(answer), stderr="")

        monkeypatch.setattr(verify_frozen.subprocess, "run", run)
        return verify_frozen.check(
            tmp_path / "package", runner, tmp_path / "report.json", selected=["ai.onnx.Slice"]
        )

    return check


def retained_case(document, name):
    return copy.deepcopy(next(case for case in document["cases"] if case["name"] == name))


@pytest.mark.parametrize("explicit", [True, False])
def test_slice_shape_conflict_is_reported_before_execution(
    check_frozen_slice, frozen_slice, explicit
):
    case = retained_case(frozen_slice[2], "ort_1d_reverse_all_axes_end_int32_max")
    if not explicit:
        case["outputs"]["output"].pop("data")
    report = check_frozen_slice(case)
    assert report["reference_conflicts"] == 1
    assert report["failed"] == report["passed"] == 0
    assert report["complete_coverage"] is False
    row = report["results"][0]
    assert row["status"] == "reference_conflict"
    assert row["oracle_conflicts"] == [
        dict(name="output", declared_shape=[4], independent_shape=[0])
    ]
    assert "evidence" not in row
    artifacts = Path(row["failure_artifacts"])
    assert json.loads((artifacts / "case.json").read_text()) == case
    assert (artifacts / "data.input.bin").is_file()
    assert (artifacts / "output.independent.bin").read_bytes() == b""
    assert not (artifacts / "output.actual.bin").exists()
    expected = np.fromfile(artifacts / "output.expected.bin", dtype=np.float32)
    np.testing.assert_array_equal(expected, [4, 3, 2, 1] if explicit else [])


@pytest.mark.parametrize(
    "values,backend,status",
    [
        ([4, 3, 2, 1], "software_vulkan", "passed"),
        ([0, 0, 0, 0], "software_vulkan", "failed"),
        ([4, 3, 2, 1], "host_shape_plan", "failed"),
    ],
)
def test_consistent_slice_fixture_still_checks_native_results(
    check_frozen_slice, frozen_slice, values, backend, status
):
    case = retained_case(frozen_slice[2], "ort_1d_reverse_all_axes_end_int32_min")
    report = check_frozen_slice(case, values, backend)
    assert report["results"][0]["status"] == status
    assert report["reference_conflicts"] == 0
    assert report["complete_coverage"] is (status == "passed")


def test_slice_value_conflict_preserves_both_references(check_frozen_slice, frozen_slice):
    case = retained_case(frozen_slice[2], "ort_1d_reverse_all_axes_end_int32_min")
    case["outputs"]["output"]["data"]["values"] = [42, 42, 42, 42]
    report = check_frozen_slice(case, [4, 3, 2, 1])
    row = report["results"][0]
    assert row["status"] == "reference_conflict"
    artifacts = Path(row["failure_artifacts"])
    np.testing.assert_array_equal(
        np.fromfile(artifacts / "output.expected.bin", dtype=np.float32), [42, 42, 42, 42]
    )
    for kind in ("actual", "independent"):
        np.testing.assert_array_equal(
            np.fromfile(artifacts / f"output.{kind}.bin", dtype=np.float32), [4, 3, 2, 1]
        )
