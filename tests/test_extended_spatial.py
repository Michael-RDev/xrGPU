"""Independent contracts and host shape checks; GPU evidence is collected separately."""

import copy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from kernels import catalog
from kernels._shared.extended.verify_spatial import cases, tensor
from kernels._shared.extended.verify_spatial_published import contract_storage, independent_conflict

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    compiler = shutil.which("c++")
    includes = [
        ROOT.parent / "xrBench/artifacts/deps/executorch/third-party/json/single_include",
        ROOT / "artifacts/deps/executorch/third-party/json/single_include",
    ]
    include = next((p for p in includes if (p / "nlohmann/json.hpp").is_file()), None)
    if compiler is None or include is None:
        pytest.skip("Host C++ compiler and pinned JSON headers required")
    directory = tmp_path_factory.mktemp("spatial-plan")
    stub = directory / "stubs.cpp"
    stub.write_text(
        '#include "framework.h"\nnamespace xrgpu::operators::extended::detail {\nbool plan_tensor(const Request&,ExtendedPlan&){return false;}\nbool plan_complex(const Request&,ExtendedPlan&){return false;}\n}\n'
    )
    output = directory / "runner"
    command = [
        compiler,
        "-std=c++17",
        "-O0",
        "-DXRGPU_EXTENDED_PLAN_ONLY=1",
        "-I" + str(ROOT / "runtime/include"),
        "-I" + str(ROOT / "kernels/_shared/extended"),
        "-I" + str(include),
        str(ROOT / "kernels/_shared/planner.cpp"),
        str(ROOT / "kernels/_shared/extended/planner.cpp"),
        str(ROOT / "kernels/_shared/extended/contracts.cpp"),
        str(ROOT / "kernels/_shared/extended/spatial_plan.cpp"),
        str(stub),
        str(ROOT / "kernels/_shared/extended/tests/execute.cpp"),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    return output


def invocation(case):
    result = {k: v for k, v in case.items() if k not in {"expected", "case"}}
    result["plan_only"] = True
    result["outputs"] = [
        dict(name=n, dtype=str(a.dtype), shape=list(a.shape)) for n, a in case["expected"].items()
    ]
    return result


def execute(runner, tmp_path, job):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job, allow_nan=False))
    return subprocess.run([str(runner), str(path)], capture_output=True, text=True, timeout=10)


def test_every_owned_operator_has_real_reference_case_and_compiled_shader_mapping():
    owned = set(json.loads((ROOT / "kernels/_shared/extended/spatial_ops.json").read_text()))
    values = cases()
    found = [c["operation"] for c in values]
    assert len(values) == 57 and set(found) == owned and len(found) == len(set(found))
    mapping = json.loads((ROOT / "kernels/_shared/extended/spatial_shader_map.json").read_text())
    assert set(mapping) == owned
    for op, shader in mapping.items():
        contract = json.loads((ROOT / "kernels" / op / "contract.json").read_text())
        variant = next(v for v in contract["variants"] if v["output"] == "extended/" + shader)
        source = catalog.regular(ROOT / "kernels", variant["source"])
        assert "#define SPATIAL_OPCODE" in source.read_text()


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["operation"])
def test_native_output_inference_matches_independent_oracle_shape_type(runner, tmp_path, case):
    result = execute(runner, tmp_path, invocation(case))
    assert result.returncode == 0, result.stderr
    answer = json.loads(result.stdout)
    assert answer["info"]["backend"] == "host_shape_plan"
    assert answer["info"]["physical_qualification"] is False
    for actual, (name, array) in zip(answer["outputs"], case["expected"].items()):
        assert (
            actual["name"] == name
            and actual["shape"] == list(array.shape)
            and actual["dtype"] == str(array.dtype)
        )


def test_int64_shape_controls_support_checked_uint32_storage(runner, tmp_path):
    case = next(c for c in cases() if c["operation"] == "ai.onnx.Col2Im")
    job = invocation(case)
    for t in job["inputs"]:
        if t["name"] in {"image_shape", "block_shape"}:
            t["storage_dtype"] = "uint32"
    result = execute(runner, tmp_path, job)
    assert result.returncode == 0, result.stderr


def test_frozen_wire_index_dtype_is_adapted_without_losing_logical_int64():
    declaration = {"dtype": "I", "storage": "uint32"}
    contract = {"typeConstraints": {"I": ["int64"]}}
    assert contract_storage({"dtype": "uint32"}, declaration, contract) == ("int64", "uint32")
    assert contract_storage({"dtype": "int64"}, declaration, contract) == ("int64", "uint32")
    assert contract_storage({"dtype": "int32"}, {"dtype": "I", "storage": "int32"}, contract) == (
        "int64",
        "int32",
    )
    assert contract_storage(
        {"dtype": "uint32"}, {"dtype": "T"}, {"typeConstraints": {"T": ["int32", "uint32"]}}
    ) == ("uint32", "uint32")


@pytest.mark.parametrize(
    "operation,attrs",
    [
        ("ai.onnx.Softmax", {"axis": 2}),
        ("ai.onnx.Conv", {"group": 2}),
        ("ai.onnx.Conv", {"strides": [0, 1]}),
        ("ai.onnx.Resize", {"scales": [1.0, 1.0, 0.0, 1.0]}),
        ("ai.onnx.STFT", {"onesided": 1, "frameStep": 0}),
        ("ai.onnx.DFT", {"axis": 2}),
    ],
)
def test_invalid_spatial_controls_fail_before_gpu_submission(runner, tmp_path, operation, attrs):
    case = copy.deepcopy(next(c for c in cases() if c["operation"] == operation))
    job = invocation(case)
    if operation == "ai.onnx.DFT":
        job["inputs"].append(tensor("axis", np.array(2, np.int64)))
    elif operation == "ai.onnx.STFT":
        job["args"]["frameStep"] = 0
    else:
        job["attrs"].update(attrs)
    result = execute(runner, tmp_path, job)
    assert result.returncode != 0 and "error" in json.loads(result.stderr)


def test_reference_conflict_requires_exact_representable_prefix_and_native_agreement():
    x = np.array([[1e-40, 1e-40, 1e-40]], np.float32)
    actual = np.cumsum(x, axis=1)
    published = np.array([[1e-40, 2e-40, 3e-40]], np.float32)
    case = {"args": {"axis": 1}}
    proof = independent_conflict("ai.onnx.CumSum", case, {"x": x}, {"y": published}, {"y": actual})
    assert proof is not None and "exact_fraction" in proof["method"]
    assert (
        independent_conflict("ai.onnx.CumSum", case, {"x": x}, {"y": actual}, {"y": actual}) is None
    )
    assert (
        independent_conflict(
            "ai.onnx.CumSum", case, {"x": x}, {"y": published}, {"y": np.zeros_like(actual)}
        )
        is None
    )
    ordinary = np.array([[1, 0.1]], np.float32)
    assert (
        independent_conflict(
            "ai.onnx.CumSum", case, {"x": ordinary}, {"y": ordinary}, {"y": ordinary}
        )
        is None
    )


def test_onnx_group_normalization_half_stage_reference_conflict_is_not_a_pass():
    arrays = {
        "x": np.arange(1, 7, dtype=np.float16).reshape(1, 2, 3),
        "scale": np.ones(2, np.float16),
        "bias": np.array([2, 1], np.float16),
    }
    normalized = (np.arange(1, 7, dtype=np.float64) - 3.5) / np.sqrt(35 / 12 + 1e-5)
    typed = (normalized.astype(np.float16) + np.repeat(arrays["bias"], 3)).reshape(1, 2, 3)
    unstaged = (normalized + np.repeat(arrays["bias"], 3)).astype(np.float16).reshape(1, 2, 3)
    proof = independent_conflict(
        "ai.onnx.GroupNormalization",
        {"attrs": {"num_groups": 1}},
        arrays,
        {"y": unstaged},
        {"y": typed},
    )
    assert proof is not None and "ONNX21" in proof["method"]
    assert (
        independent_conflict(
            "ai.onnx.GroupNormalization",
            {"attrs": {"num_groups": 1}},
            arrays,
            {"y": unstaged},
            {"y": unstaged},
        )
        is None
    )


@pytest.mark.parametrize("scale_shape", [(3,), (1, 3), (3, 1)])
def test_gated_rms_normalizes_flat_scale_sized_heads(runner, tmp_path, scale_shape):
    x = np.arange(24, dtype=np.float32).reshape(2, 2, 6)
    case = {
        "operation": "com.microsoft.GatedRMSNorm",
        "inputs": [
            tensor("xT", x),
            tensor("scaleT", np.ones(scale_shape, np.float32)),
            tensor("gateT", np.ones_like(x)),
        ],
        "expected": {"yT": np.empty_like(x)},
    }
    result = execute(runner, tmp_path, invocation(case))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["outputs"][0]["shape"] == list(x.shape)
