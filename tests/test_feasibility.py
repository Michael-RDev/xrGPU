import copy
import math
import sys

import pytest

from analysis.report import compare, confirmed_verdict, feasibility, main
from tests.test_report import evidence


def short_report(speedup=1.25, run_id="first"):
    report = evidence()
    del report["sustained"]
    report.update(status="complete", run_id=run_id)
    for trial in report["trials"]:
        trial["backend"] = "vulkan"
        ms = 10 if trial["arm"] == "stock" else 10 / speedup
        trial["samples"] = [
            dict(
                end_to_end_ms=ms,
                model_ms=ms * 0.9,
                thermal_status=trial["thermal_start"],
                charging=trial["charging"],
            )
            for _ in range(100)
        ]
    return report


def test_diagnostic_can_never_establish_full_acceptance():
    report = short_report()
    diagnostic = feasibility(report)
    assert diagnostic["valid_evidence"] and diagnostic["meets_short_run_target"]
    assert diagnostic["diagnostic_only"] and not diagnostic["selection_eligible"]
    assert not compare(report)["accepted"]
    assert confirmed_verdict(report, short_report(run_id="repeat")) == "strong preliminary result"


@pytest.mark.parametrize(
    "speedup,verdict", [(1.05, "smaller repeatable gain"), (0.95, "no demonstrated gain")]
)
def test_measured_outcomes(speedup, verdict):
    first, repeat = short_report(speedup), short_report(speedup, "repeat")
    assert feasibility(first)["valid_evidence"]
    assert confirmed_verdict(first, repeat) == verdict


@pytest.mark.parametrize(
    "key,value",
    [("model_ms", math.nan), ("end_to_end_ms", 0), ("thermal_status", 1), ("charging", True)],
)
def test_invalid_samples_do_not_establish_feasibility(key, value):
    report = short_report()
    report["trials"][1]["samples"][0][key] = value
    assert not feasibility(report)["valid_evidence"]


def test_model_tail_checked_separately_from_end_to_end():
    report = short_report()
    for t in report["trials"]:
        if t["arm"] == "candidate":
            for s in t["samples"][80:]:
                s["model_ms"] = 20
    result = feasibility(report)
    assert result["valid_evidence"] and not result["tails_within_limits"]
    assert not result["meets_short_run_target"]


def test_baseline_must_not_dispatch_candidate_kernels():
    report = short_report()
    report["trials"][0]["custom_nodes"] = 12
    assert not feasibility(report)["valid_evidence"]


def test_confirmation_requires_complete_independent_matching_runs():
    first = short_report()
    assert confirmed_verdict(first, copy.deepcopy(first)) == "inconclusive"
    repeat = short_report(run_id="repeat")
    for trial in repeat["trials"]:
        trial["driver"] = 99
    assert confirmed_verdict(first, repeat) == "inconclusive"
    first["status"] = "running"
    assert not feasibility(first)["valid_evidence"]


def test_feasibility_cli_forbids_selection(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["report", "unused.json", "--feasibility", "--selection", "out.json"]
    )
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
