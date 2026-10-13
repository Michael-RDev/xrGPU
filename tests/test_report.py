import copy
import math

import pytest
from analysis.report import compare, percentile


def evidence(mode="isolated"):
    common = dict(device="Oculus/Quest 3", driver=1, os="test-os", model_revision="model",
                  model_sha256="weights", runtime_revision="runtime", kernel_revision="kernel",
                  precision="fp32", sequence_length=128, mode=mode, scene_revision="scene",
                  refresh_hz=72 if mode == "vr" else 0, inference_interval_ms=100,
                  suite_revision="suite", status="ok", correctness=True, charging=False, thermal_start=0,
                  thermal_discontinuity=False, warmup_iterations=20)
    trials = []
    for pair in range(4):
        for arm in ("stock", "candidate"):
            trials.append(dict(common, pair=pair, arm=arm, variant=0 if arm == "stock" else 1,
                               custom_nodes=0 if arm == "stock" else 12,
                               samples=[dict(end_to_end_ms=10 if arm == "stock" else 8)] * 100,
                               measurement_seconds=75, frame_ms=[5.] * 100,
                               missed_frame_rate=.001))
    return dict(schema_version=1, trials=trials,
                sustained=dict(schema_version=1, duration_seconds=600, trials=copy.deepcopy(trials)))


def test_accepts_only_complete_repeatable_evidence():
    for mode in ("isolated", "vr"):
        result = compare(evidence(mode))
        assert result["accepted"]
        assert result["median_speedup"] == 1.25


@pytest.mark.parametrize("field,value", [("correctness", False), ("status", "canceled"),
    ("thermal_discontinuity", True), ("custom_nodes", 0), ("warmup_iterations", 0),
    ("driver", 99), ("precision", "fp16"), ("sequence_length", 256),
    ("thermal_start", 1), ("charging", True), ("variant", -1), ("variant", 4)])
def test_rejects_invalid_or_incomparable_runs(field, value):
    report = evidence()
    report["trials"][1][field] = value
    assert not compare(report)["accepted"]


def test_no_silent_sustained_or_frame_success():
    report = evidence("vr")
    report["trials"][1]["missed_frame_rate"] = None
    assert not compare(report)["accepted"]
    report = evidence()
    report["sustained"] = {"passed": True, "correctness": True, "duration_seconds": 600}
    assert not compare(report)["accepted"]


def test_sustained_must_match_device_and_candidate():
    for key, value in (("device", "Oculus/Quest 3S"), ("variant", 2), ("measurement_seconds", 1)):
        report = evidence()
        for t in report["sustained"]["trials"]:
            if key != "variant" or t["arm"] == "candidate":
                t[key] = value
        assert not compare(report)["accepted"]
    report = evidence()
    report["sustained"]["status"] = "failed"
    assert not compare(report)["accepted"]


def test_latency_tail_and_frame_regressions_reject():
    report = evidence("vr")
    report["trials"][1]["frame_ms"] = [10.] * 100
    assert not compare(report)["accepted"]
    report = evidence()
    report["trials"][1]["samples"] = [dict(end_to_end_ms=8)] * 75 + [dict(end_to_end_ms=30)] * 25
    assert not compare(report)["accepted"]


def test_duplicate_or_nonfinite_evidence_is_invalid():
    report = evidence()
    report["trials"].append(report["trials"][0])
    with pytest.raises(ValueError):
        compare(report)
    with pytest.raises(ValueError):
        percentile([1, math.nan], .5)
    with pytest.raises(ValueError):
        percentile([], .5)


def test_cross_pair_identity_and_variant_must_match():
    report = evidence()
    report["trials"][2]["driver"] = 22
    report["trials"][3]["driver"] = 22
    assert not compare(report)["accepted"]
    report = evidence()
    report["trials"][3]["variant"] = 2
    assert not compare(report)["accepted"]
