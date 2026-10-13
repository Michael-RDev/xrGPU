"""Compare actual paired device runs; never infer a pass from missing metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

IDENTITY = ("device", "driver", "os", "model_revision", "model_sha256", "runtime_revision",
            "kernel_revision", "precision", "sequence_length", "mode", "scene_revision",
            "refresh_hz", "inference_interval_ms", "suite_revision", "charging")


def percentile(values, q):
    values = sorted(values)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("timings must be finite, nonnegative, and nonempty")
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def compare(report, require_sustained=True):
    if report.get("schema_version") != 1:
        raise ValueError("unsupported report schema")
    trials = report.get("trials", [])
    reasons = []
    if report.get("status") == "failed":
        reasons.append("run failed or was interrupted")
    pairs = {}
    identity = trials[0] if trials else {}
    candidate_variant = None
    for trial in trials:
        if any(k not in trial or trial[k] != identity.get(k) for k in IDENTITY):
            reasons.append("incomparable device/workload/runtime metadata")
        if trial.get("arm") == "candidate":
            if candidate_variant is None:
                candidate_variant = trial.get("variant")
            if trial.get("variant") != candidate_variant:
                reasons.append("mixed candidate variants")
        key = trial["pair"]
        bucket = pairs.setdefault(key, {})
        arm = trial["arm"]
        if arm not in ("stock", "candidate") or arm in bucket:
            raise ValueError("duplicate/unknown trial arm")
        bucket[arm] = trial
    if len(pairs) < 4:
        reasons.append("requires at least four paired trials")
    baseline, candidate, ratios = [], [], []
    for pair in pairs.values():
        if set(pair) != {"stock", "candidate"}:
            reasons.append("incomplete pair")
            continue
        a, b = pair["stock"], pair["candidate"]
        if a.get("thermal_start", -1) < 0 or a.get("thermal_start") != b.get("thermal_start"):
            reasons.append("different or unavailable starting thermal state")
        if any(a.get(k) != b.get(k) or k not in a for k in IDENTITY):
            reasons.append("incomparable device/workload/runtime metadata")
        if a.get("variant") != 0 or b.get("variant") not in (1, 2, 3):
            reasons.append("invalid variant attribution")
        for item in (a, b):
            if item.get("status") != "ok" or item.get("correctness") is not True:
                reasons.append("correctness did not pass")
            samples = item.get("samples", [])
            if len(samples) < 100:
                reasons.append("requires at least 100 warmed samples per trial")
            if item.get("warmup_iterations", 0) < 20:
                reasons.append("insufficient warmup")
            if item.get("thermal_discontinuity", True):
                reasons.append("thermal state changed or was unavailable")
        av = [s["end_to_end_ms"] for s in a.get("samples", [])]
        bv = [s["end_to_end_ms"] for s in b.get("samples", [])]
        if not av or not bv:
            reasons.append("missing timings")
            continue
        if b.get("custom_nodes", 0) <= 0:
            reasons.append("candidate was not dispatched")
        am, bm = percentile(av, 0.5), percentile(bv, 0.5)
        if am <= 0 or bm <= 0:
            reasons.append("zero duration is not evidence")
            continue
        ratios.append(am / bm)
        baseline.extend(av)
        candidate.extend(bv)
        if a["mode"] == "vr":
            if abs(a.get("refresh_hz", 0) - 72) > .1 or abs(b.get("refresh_hz", 0) - 72) > .1:
                reasons.append("72 Hz was not maintained")
            af, bf = a.get("frame_ms", []), b.get("frame_ms", [])
            if not af or not bf or a.get("missed_frame_rate") is None or b.get("missed_frame_rate") is None:
                reasons.append("missing VR frame metrics")
            else:
                if any(not math.isfinite(t["missed_frame_rate"]) or not 0 <= t["missed_frame_rate"] <= 1
                       for t in (a, b)):
                    reasons.append("invalid missed-frame metric")
                if percentile(bf, .95) > percentile(af, .95) * 1.05:
                    reasons.append("VR p95 regression exceeds 5%")
                if b["missed_frame_rate"] - a["missed_frame_rate"] > .01:
                    reasons.append("missed-frame increase exceeds one percentage point")
    speedup = percentile(baseline, .5) / percentile(candidate, .5) if baseline and candidate else None
    if speedup is None or speedup < 1 / .9:
        reasons.append("median latency reduction is below 10%")
    if baseline and candidate and percentile(candidate, .95) > percentile(baseline, .95) * 1.05:
        reasons.append("model p95 regression exceeds 5%")
    if ratios and min(ratios) <= 1:
        reasons.append("speedup did not repeat in every paired trial")
    if require_sustained:
        sustained = report.get("sustained", {})
        long_trials = sustained.get("trials", [])
        matches = all(all(t.get(k) == identity.get(k) for k in IDENTITY) for t in long_trials)
        matches &= all(t.get("variant") == candidate_variant for t in long_trials if t.get("arm") == "candidate")
        duration = sum(t.get("measurement_seconds", 0) for t in long_trials)
        if (not long_trials or not matches or not math.isfinite(duration) or duration < 600 or
                not compare(sustained, False)["accepted"]):
            reasons.append("matching ten-minute sustained comparison has not passed")
    return {"accepted": not reasons, "median_speedup": speedup,
            "paired_speedups": ratios, "reasons": sorted(set(reasons)),
            "trial_speedup_range": [min(ratios), max(ratios)] if ratios else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--selection", type=Path, help="Write a local selection only after acceptance")
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    result = compare(report)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.selection:
        if not result["accepted"]:
            raise SystemExit("No selection written: validation failed")
        first = next(t for t in report["trials"] if t["arm"] == "candidate")
        selection = {"schema_version": 1, "identity": {k: first[k] for k in IDENTITY},
                     "variant": first["variant"], "report_sha256": hashlib.sha256(raw).hexdigest()}
        args.selection.parent.mkdir(parents=True, exist_ok=True)
        args.selection.write_text(json.dumps(selection, indent=2) + "\n")


if __name__ == "__main__":
    main()
