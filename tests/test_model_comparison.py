import copy

import pytest

from analysis.optimization import compare_models
from tests.test_report import evidence


def model_evidence():
    report = evidence()
    report.pop("sustained")
    report.update(
        kind="model_optimization",
        status="complete",
        profile="sdpa",
        candidate_variant=0,
        candidate_precision="fp32",
        reference_sha256="a" * 64,
        candidate_sha256="b" * 64,
        layer_audit={
            "correctness": True,
            "atol": 1e-4,
            "rtol": 1e-4,
            "checkpoints": 62,
            "cases": [
                {"checks": [{"correctness": True, "max_tolerance_ratio": 0.5} for _ in range(62)]}
                for _ in range(3)
            ],
        },
    )
    for t in report["trials"]:
        candidate = t["arm"] == "candidate"
        t.update(
            backend="vulkan",
            model_sha256=("b" if candidate else "a") * 64,
            model_profile="sdpa" if candidate else "reference",
            variant=0,
            custom_nodes=0,
        )
        t["samples"] = [
            dict(
                end_to_end_ms=8 if candidate else 10,
                model_ms=7 if candidate else 9,
                thermal_status=0,
                charging=False,
            )
            for _ in range(100)
        ]
    report["trials"].sort(
        key=lambda t: (t["pair"], (t["arm"] == "candidate") != bool(t["pair"] % 2))
    )
    return report


def test_explicit_program_comparison_preserves_sustained_gate():
    report = model_evidence()
    assert compare_models(report)["accepted"]
    assert not compare_models(report)["selection_eligible"]
    assert not compare_models(report, True)["accepted"]
    report["sustained"] = copy.deepcopy(report)
    assert compare_models(report, True)["selection_eligible"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_sha256", "c" * 64),
        ("precision", "fp16"),
        ("model_profile", "other"),
        ("variant", 1),
        ("custom_nodes", 12),
        ("driver", 99),
        ("charging", True),
        ("inference_interval_ms", 1),
        ("correctness", False),
    ],
)
def test_rejects_undeclared_or_confounded_model_comparisons(key, value):
    report = model_evidence()
    report["trials"][1][key] = value
    assert not compare_models(report)["valid_evidence"]


def test_audit_and_model_tail_are_independent_requirements():
    report = model_evidence()
    report["layer_audit"]["cases"][0]["checks"][0]["max_tolerance_ratio"] = 1.01
    assert not compare_models(report)["valid_evidence"]
    report = model_evidence()
    for t in report["trials"]:
        if t["arm"] == "candidate":
            for s in t["samples"]:
                s["model_ms"] = 20
    result = compare_models(report)
    assert result["valid_evidence"] and not result["accepted"] and not result["tails_within_limits"]


def test_declared_fp16_still_requires_exact_layer_thresholds():
    report = model_evidence()
    report["candidate_precision"] = "fp16"
    for t in report["trials"]:
        if t["arm"] == "candidate":
            t["precision"] = "fp16"
    assert compare_models(report)["accepted"]
    report["layer_audit"]["atol"] = 1e-2
    assert not compare_models(report)["valid_evidence"]
