import argparse
import json
import math
from pathlib import Path

from analysis.report import IDENTITY, percentile, summarize_trial_group

COMMON_IDENTITY = tuple(k for k in IDENTITY if k not in ("model_sha256", "precision"))


def compare_models(report, require_sustained=False):
    reasons = []
    performance = []
    try:
        if report.get("schema_version") != 1 or report.get("kind") != "model_optimization":
            raise ValueError("not a model optimization report")
        if report.get("status") != "complete":
            reasons.append("incomplete or failed report")
        audit = report.get("layer_audit", {})
        cases = audit.get("cases", [])
        if (
            audit.get("correctness") is not True
            or audit.get("atol") != 1e-4
            or audit.get("rtol") != 1e-4
            or audit.get("checkpoints", 0) < 62
            or len(cases) < 3
            or any(
                len(c.get("checks", [])) != audit["checkpoints"]
                or any(
                    p.get("correctness") is not True
                    or not math.isfinite(p.get("max_tolerance_ratio", float("inf")))
                    or not 0 <= p["max_tolerance_ratio"] <= 1
                    for p in c["checks"]
                )
                for c in cases
            )
        ):
            reasons.append("strict all-layer checkpoint audit did not pass")
        trials = report.get("trials", [])
        identity = trials[0] if trials else {}
        pairs = {}
        for t in trials:
            if any(t.get(k) is None or t[k] != identity.get(k) for k in COMMON_IDENTITY):
                reasons.append("incomparable device/workload/runtime metadata")
            if (
                t.get("mode") != "isolated"
                or t.get("backend") != "vulkan"
                or t.get("inference_interval_ms") != 100
            ):
                reasons.append("unexpected benchmark mode or cadence")
            arm = t["arm"]
            if arm not in ("stock", "candidate") or arm in pairs.setdefault(t["pair"], {}):
                raise ValueError("duplicate or invalid arm")
            pairs[t["pair"]][arm] = t
            expected_hash = report["reference_sha256" if arm == "stock" else "candidate_sha256"]
            expected_precision = "fp32" if arm == "stock" else report["candidate_precision"]
            expected_profile = "reference" if arm == "stock" else report["profile"]
            expected_variant = 0 if arm == "stock" else report["candidate_variant"]
            if (
                len(expected_hash) != 64
                or t.get("model_sha256") != expected_hash
                or t.get("precision") != expected_precision
                or t.get("model_profile") != expected_profile
                or t.get("variant") != expected_variant
            ):
                reasons.append("program identity differs from the declared comparison")
            if expected_variant and (
                expected_variant not in t.get("candidate_variants", [])
                or t.get("custom_nodes", 0) <= 0
            ):
                reasons.append("requested custom kernels were not dispatched")
            if not expected_variant and t.get("custom_nodes") != 0:
                reasons.append("unexpected custom dispatch")
            if t.get("status") != "ok" or t.get("correctness") is not True:
                reasons.append("embedding correctness did not pass")
            if len(t.get("samples", [])) < 100 or t.get("warmup_iterations", 0) < 20:
                reasons.append("insufficient warmed samples")
            if t.get("thermal_discontinuity", True) or t.get("thermal_start", -1) < 0:
                reasons.append("thermal or charging state changed or was unavailable")
            for s in t.get("samples", []):
                if s.get("thermal_status") != t.get("thermal_start") or s.get("charging") != t.get(
                    "charging"
                ):
                    reasons.append("sample thermal or charging state changed")
                for k in ("model_ms", "end_to_end_ms"):
                    value = s.get(k)
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(value)
                        or value <= 0
                    ):
                        raise ValueError("invalid model/end-to-end timing")
        if len(pairs) < 4 or set(pairs) != set(range(len(pairs))):
            reasons.append("requires at least four consecutive paired trials")
        ratios = []
        for number, pair in sorted(pairs.items()):
            if set(pair) != {"stock", "candidate"}:
                reasons.append("incomplete pair")
                continue
            expected_order = ["stock", "candidate"] if number % 2 == 0 else ["candidate", "stock"]
            if [t["arm"] for t in trials if t["pair"] == number] != expected_order:
                reasons.append("trial order did not alternate")
            if pair["stock"]["thermal_start"] != pair["candidate"]["thermal_start"]:
                reasons.append("paired starting thermal states differ")
            ratios.append(
                percentile([s["end_to_end_ms"] for s in pair["stock"]["samples"]], 0.5)
                / percentile([s["end_to_end_ms"] for s in pair["candidate"]["samples"]], 0.5)
            )
        summary = {
            arm: summarize_trial_group([t for t in trials if t["arm"] == arm])
            for arm in ("stock", "candidate")
        }
        a, b = (summary[arm]["timings_ms"] for arm in ("stock", "candidate"))
        speedup = a["end_to_end_ms"]["median"] / b["end_to_end_ms"]["median"]
        tails_ok = all(b[k]["p95"] <= a[k]["p95"] * 1.05 for k in ("model_ms", "end_to_end_ms"))
        if speedup < 1 / 0.9:
            performance.append("median latency reduction below 10%")
        if not ratios or min(ratios) <= 1:
            performance.append("not faster in every pair")
        if not tails_ok:
            performance.append("model or end-to-end p95 regression exceeds 5%")
        if require_sustained:
            long = report.get("sustained", {})
            duration = sum(t.get("measurement_seconds", 0) for t in long.get("trials", []))
            matching = all(
                long.get(k) == report.get(k)
                for k in (
                    "profile",
                    "candidate_variant",
                    "candidate_sha256",
                    "reference_sha256",
                    "candidate_precision",
                )
            )
            matching &= all(
                all(t.get(k) == identity.get(k) for k in COMMON_IDENTITY)
                for t in long.get("trials", [])
            )
            if (
                not matching
                or not math.isfinite(duration)
                or duration < 600
                or not compare_models(long)["accepted"]
            ):
                reasons.append("matching ten-minute sustained comparison has not passed")
        return {
            "valid_evidence": not reasons,
            "accepted": not reasons and not performance,
            "selection_eligible": require_sustained and not reasons and not performance,
            "median_speedup": speedup,
            "latency_reduction_percent": 100 * (1 - 1 / speedup),
            "paired_speedups": ratios,
            "tails_within_limits": tails_ok,
            "reasons": sorted(set(reasons)),
            "performance_reasons": performance,
            "summary": summary,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        return {
            "valid_evidence": False,
            "accepted": False,
            "selection_eligible": False,
            "reasons": ["malformed evidence: " + str(error)],
            "summary": {},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--sustained", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compare_models(json.loads(args.report.read_text()), args.sustained), indent=2))


if __name__ == "__main__":
    main()
