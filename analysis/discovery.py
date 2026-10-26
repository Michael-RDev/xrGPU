"""Find new kernel targets in original xrBench reports, without claiming speedups.

Example: python -m analysis.discovery --archive artifacts/benchmarks/archives \
    --output artifacts/discovery/quest3.json

Ranks every eligible case by median non-staging GPU work within its own run.
Runs, shapes and thermal/charging states are never pooled. Trace timings are
exploratory observations; confirmation must use fresh, untraced paired runs.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("model_ms", "gpu_ms", "compute_ms", "layout_ms", "staging_ms", "target_ms")


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * .95
    low, high = math.floor(position), math.ceil(position)
    return dict(count=len(values), median=statistics.median(values),
                p95=ordered[low] + (ordered[high] - ordered[low]) * (position - low),
                min=ordered[0], max=ordered[-1])


def finite_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def kernel_category(name):
    if name.startswith(("nchw_to_", "image_to_nchw", "buffer_to_", "image_to_buffer")):
        return "staging"
    if name.startswith(("clone_", "view_", "permute_", "transpose_", "copy_")):
        return "layout"
    return "compute"


def existing_operations(root=ROOT):
    # Standalone native operators are not enrolled model graph replacements.
    # Only explicitly marked model override contracts affect discovery status.
    operations = set()
    for path in (root / "kernels").glob("*/contract.json"):
        contract = json.loads(path.read_text())
        if contract.get("kind") != "xrgpu.model-override":
            continue
        operation = contract.get("operation")
        if not isinstance(operation, str) or not operation:
            raise ValueError("Model override operation is missing: " + str(path))
        operations.add(operation)
    return sorted(operations)


def summarize_case(row):
    samples = row.get("samples", [])
    reasons = []
    if row.get("status") != "passed":
        reasons.append("case did not pass")
    if row.get("timing_valid") is False:
        reasons.append("timings marked invalid")
    for key in ("correctness", "correctness_before", "correctness_after"):
        if row.get(key, {}).get("passed") is not True:
            reasons.append(key + " missing or failed")
    if row.get("thermal_discontinuity") or row.get("focus_lost"):
        reasons.append("thermal or focus discontinuity")
    if not samples:
        reasons.append("no measurements")
    for start, key in (("thermal_start", "thermal_status"), ("charging_start", "charging")):
        states = [row.get(start), *[s.get(key) for s in samples]]
        if any(v is None for v in states) or len(set(states)) != 1:
            reasons.append("missing or changing " + key)
    if any(s.get("candidate_selected") is not False or s.get("kernel_variant") != "stock"
           or s.get("validation_control", "none") != "none" for s in samples):
        reasons.append("requires ordinary stock measurements")

    measurements = defaultdict(list)
    kernel_samples = defaultdict(list)
    dispatch_counts = []
    for sample in samples:
        dispatches = sample.get("dispatches")
        if not isinstance(dispatches, list) or not dispatches:
            reasons.append("dispatch trace unavailable")
            continue
        if sample.get("dispatch_count") != len(dispatches):
            reasons.append("incomplete dispatch trace")
        if any(not finite_time(sample.get(key)) for key in ("model_ms", "gpu_ms")):
            reasons.append("invalid invocation timing")
            continue
        if any(not isinstance(d.get("kernel_name"), str) or not d["kernel_name"]
               or not finite_time(d.get("ms")) for d in dispatches):
            reasons.append("invalid kernel trace")
            continue
        totals = dict(compute=0., layout=0., staging=0.)
        by_kernel = defaultdict(float)
        for dispatch in dispatches:
            name, elapsed = dispatch["kernel_name"], dispatch["ms"]
            totals[kernel_category(name)] += elapsed
            by_kernel[name] += elapsed
        traced = sum(totals.values())
        # Older runners calibrated dispatch timestamps separately from gpu_ms.
        # Keep both scopes; reject material disagreement rather than rescale.
        if not math.isclose(traced, sample["gpu_ms"], rel_tol=.05, abs_tol=.002):
            reasons.append("dispatch sum disagrees with GPU timing")
        for key in ("model_ms", "gpu_ms"):
            measurements[key].append(sample[key])
        for key, value in totals.items():
            measurements[key + "_ms"].append(value)
        measurements["target_ms"].append(totals["compute"] + totals["layout"])
        dispatch_counts.append(len(dispatches))
        # Include zeros for kernels absent in a sample, rather than inflating
        # their contribution by conditioning on their appearance.
        for name in set(kernel_samples) | set(by_kernel):
            kernel_samples[name].extend([0.] * (len(dispatch_counts) - 1 - len(kernel_samples[name])))
            kernel_samples[name].append(by_kernel.get(name, 0.))
    reasons = sorted(set(reasons))
    result = dict(case=row["case"], operation=row["operation"],
                  phase=row.get("phase"), trial=row.get("trial"),
                  inputs=row.get("inputs"), params=row.get("params", {}),
                  fixture_seed=row.get("fixture_seed"), sample_count=len(samples),
                  eligible=not reasons, reasons=reasons,
                  correctness=row.get("correctness"),
                  thermal_start=row.get("thermal_start"), charging_start=row.get("charging_start"))
    if reasons:
        return result
    result["timings"] = {key: distribution(measurements[key]) for key in METRICS}
    result["dispatch_count"] = distribution(dispatch_counts)
    result["kernels"] = sorted((dict(name=name, category=kernel_category(name),
                                    timings=distribution(values))
                                for name, values in kernel_samples.items()),
                               key=lambda k: (-k["timings"]["median"], k["name"]))
    return result


def analyze(paths, root=ROOT):
    reports, ignored, seen = [], [], {}
    operations = existing_operations(root)
    for path in sorted({Path(p).resolve() for p in paths}):
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        report = json.loads(raw)
        run_id = report.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Missing run identity: " + str(path))
        if run_id in seen:
            if seen[run_id] != sha:
                raise ValueError("Conflicting original bytes for run: " + run_id)
            continue
        seen[run_id] = sha
        source = path.with_name("source.json")
        if source.is_file():
            expected = json.loads(source.read_text()).get("raw_sha256")
            if expected != sha:
                raise ValueError("Original report checksum mismatch: " + str(path))
        config = report.get("configuration", {})
        reason = None
        if report.get("schema_version") != 2:
            reason = "requires native report schema 2"
        elif report.get("backend") != "executorch_vulkan" or report.get("execution_location") != "headset":
            reason = "requires original headset Vulkan measurements"
        elif report.get("status") != "complete":
            reason = "partial run; excluded from ranking"
        elif config.get("mode") != "isolated":
            reason = "use isolated baselines for discovery"
        elif config.get("validation_control", "none") != "none" or report.get("experiment"):
            reason = "control or experiment run; excluded from discovery"
        if reason:
            ignored.append(dict(run_id=run_id, path=str(path), raw_sha256=sha, reason=reason))
            continue
        rows = [summarize_case(row) for row in report.get("results", [])]
        eligible = [row for row in rows if row["eligible"]]
        for row in eligible:
            row["existing_replacement"] = row["operation"] in operations or (
                row["operation"] == "conv2d" and row["params"].get("groups", 1) > 1
                and "grouped_conv2d" in operations)
        eligible.sort(key=lambda r: (-r["timings"]["target_ms"]["median"], r["case"]))
        reports.append(dict(run_id=run_id, path=str(path), raw_sha256=sha,
                            device=report.get("device", {}), configuration=config,
                            suite_revision=report.get("suite_revision"),
                            runtime_revision=report.get("runtime_revision"),
                            profiling=sorted({s.get("profiling", "unrecorded")
                                              for r in report.get("results", [])
                                              for s in r.get("samples", [])}),
                            operation_count=len({r["operation"] for r in rows}),
                            eligible_cases=eligible,
                            excluded_cases=[row for row in rows if not row["eligible"]]))
    return dict(schema_version=1, kind="xrgpu.kernel_discovery", status="exploratory",
                ranking_metric="median per-invocation compute + layout dispatch milliseconds, within each run",
                existing_operations=operations, reports=reports, ignored_reports=ignored,
                limitations=["Trace profiling adds overhead; these observations do not establish speedups.",
                             "Runs and charging/thermal states are not pooled; samples within a run are not independent sessions.",
                             "The suite is not a model invocation-frequency profile; rankings do not imply model speedups.",
                             "Timing scopes overlap; model_ms and gpu_ms must not be added.",
                             "Kernel categories use names; operator labels in legacy dispatch traces may be stale.",
                             "New targets include layout/fusion opportunities. Confirm with fresh fixtures, latency runs and complete models."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--archive", type=Path, help="Read */*/report.json from an imported archive")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = args.reports + (list(args.archive.glob("*/*/report.json")) if args.archive else [])
    if not paths:
        parser.error("Provide report paths or --archive")
    result = analyze(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("New targets ranked by non-staging GPU work (exploratory trace timings):")
    for report in result["reports"]:
        print(report["run_id"])
        for row in [r for r in report["eligible_cases"] if not r["existing_replacement"]][:10]:
            t = row["timings"]
            print(f"  {row['case']:28} target={t['target_ms']['median']:.4f} ms"
                  f"  model={t['model_ms']['median']:.4f} ms"
                  f"  dispatches={row['dispatch_count']['median']:g}")
    print(json.dumps(dict(output=str(args.output), reports=len(result["reports"]),
                          ignored_reports=len(result["ignored_reports"]))))


if __name__ == "__main__":
    main()
