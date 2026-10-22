"""Shared layer-report statistics, durable writes, and collector protocol."""

import hashlib
import json
import math
import os
import statistics
import urllib.parse
import urllib.request
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def distribution(values):
    if not values or any(v is None or not math.isfinite(v) or v < 0 for v in values):
        return None
    ordered = sorted(values)

    def percentile(p):
        position = (len(ordered) - 1) * p
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

    return {"count": len(values), "median": percentile(.5), "p95": percentile(.95),
            "min": ordered[0], "max": ordered[-1], "mean": statistics.mean(values),
            "stdev": statistics.pstdev(values)}


def summarize(report):
    for case in report["results"]:
        samples = case.get("samples", [])
        case["timings_ms"] = {key: distribution([s.get(key) for s in samples])
                              for key in ("execution_ms", "model_ms", "gpu_ms")}
        timing = case["timings_ms"]["execution_ms"]
        valid = case.get("status") == "passed" and not case.get("thermal_discontinuity", False)
        case["timing_valid"] = valid and timing is not None
        median = timing["median"] if case["timing_valid"] else None
        flops = case.get("work", {}).get("estimated_matmul_conv_flops")
        case["estimated_gflops_per_second"] = flops / (median * 1e6) if flops and median else None
        # These are not bandwidth or peak-efficiency measurements.
        case["power_watts"] = None
        case["peak_utilization_percent"] = None
    report["counts"] = {status: sum(c.get("status") == status for c in report["results"])
                        for status in ("passed", "failed", "skipped")}
    return report


def validate_report(report):
    if not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("kind") != "layer_suite":
        raise ValueError("Expected an xrGPU layer_suite report (schema_version 1)")
    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not 1 <= len(run_id) <= 80 or not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid run_id")
    if report.get("status") not in ("complete", "failed", "canceled"):
        raise ValueError("Only finished reports can be submitted")
    if report.get("execution_location") not in ("host", "headset"):
        raise ValueError("Missing execution_location")
    if not isinstance(report.get("device"), dict) or not isinstance(report.get("suite_revision"), str):
        raise ValueError("Missing device or suite identity")
    if not isinstance(report.get("backend"), str) or not report["backend"]:
        raise ValueError("Missing backend identity")
    cases = report.get("results")
    if not isinstance(cases, list) or len(cases) > 10000:
        raise ValueError("Invalid results")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case"), str):
            raise ValueError("Invalid case")
        if case["case"] in ids or case.get("status") not in ("passed", "failed", "skipped"):
            raise ValueError("Duplicate case or invalid case status")
        ids.add(case["case"])
        if case["status"] == "passed":
            samples = case.get("samples")
            check = case.get("correctness")
            if not isinstance(check, dict) or check.get("passed") is not True or not isinstance(samples, list) or not samples:
                raise ValueError("Passed cases require correctness and timing samples")
            for sample in samples:
                value = sample.get("execution_ms") if isinstance(sample, dict) else None
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid execution timing")
    # Reject NaN/Infinity anywhere, including ancillary metadata.
    return json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def upload(report, url, token=None):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.fragment:
        raise ValueError("Use an HTTP(S) collector URL without embedded credentials or fragments")
    payload = validate_report(report)
    digest = hashlib.sha256(payload).hexdigest()
    headers = {"Content-Type": "application/json", "Idempotency-Key": report["run_id"]}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
        receipt = json.loads(response.read(65536))
    if receipt.get("run_id") != report["run_id"] or receipt.get("sha256") != digest or receipt.get("stored") is not True:
        raise RuntimeError("Collector did not acknowledge storage of this exact report")
    return receipt


def submit_file(path, url):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    receipt = upload(report, url, os.environ.get("XRGPU_UPLOAD_TOKEN"))
    save(path.with_suffix(".receipt.json"), receipt)
    return receipt
