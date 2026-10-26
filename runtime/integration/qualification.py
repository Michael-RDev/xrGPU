"""Offline qualification contract shared by xrGPU and xrBench.

Qualification is shipped by the library maintainer, never supplied by a model.
Hashes bind evidence and selection rules to an immutable kernel package. They
provide integrity, not hardware attestation.
"""

import hashlib
import json
import math
import re

TARGET_FIELDS = ("model", "gpu", "vendor_id", "device_id", "driver", "api_version")
BASE_GATES = ("hardware", "provenance", "correctness", "controls", "holdout")
PERFORMANCE_GATES = BASE_GATES + ("performance", "models", "sustained")
LINEAR_VARIANTS = frozenset(
    f"m{m}_wg{x}x{y}_{storage}"
    for m in (1, 2, 4)
    for x, y in ((8, 4), (16, 4), (8, 8))
    for storage in ("texture2d", "buffer")
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def target_identity(target):
    if not isinstance(target, dict) or any(k not in target for k in TARGET_FIELDS):
        raise ValueError("Complete device, GPU and driver identity required")
    result = {key: target[key] for key in TARGET_FIELDS}
    for key in ("model", "gpu"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError("Missing target identity: " + key)
    driver = result["driver"]
    if not ((isinstance(driver, str) and driver.strip()) or (type(driver) is int and driver >= 0)):
        raise ValueError("Missing target identity: driver")
    for key in ("vendor_id", "device_id", "api_version"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError("Invalid target identity: " + key)
    return result


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite operation descriptor")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    if isinstance(value, list):
        for item in value:
            _finite(item)


def validate_rule(rule):
    if not isinstance(rule, dict) or set(rule) != {"request", "variant", "kind"}:
        raise ValueError("Qualification rule requires request, variant and kind")
    request = rule["request"]
    if not isinstance(request, dict) or set(request) != {
        "operation", "input_shape", "output_shape", "dtype", "layout", "attributes"
    }:
        raise ValueError("Incomplete operation descriptor")
    operation = request["operation"]
    variants = {"gelu": {"semantics_v1"}, "grouped_conv2d": {"grouped_v1"},
                "linear": LINEAR_VARIANTS}
    if operation not in variants or rule["variant"] not in variants[operation]:
        raise ValueError("Unknown operation or kernel variant")
    if rule["kind"] not in ("correctness_repair", "performance"):
        raise ValueError("Unknown qualification rule kind")
    if rule["kind"] == "correctness_repair" and operation == "linear":
        raise ValueError("Linear variants are performance candidates")
    if request["dtype"] != "float32" or request["layout"] not in ("texture3d", "buffer"):
        raise ValueError("Unsupported qualification dtype or layout")
    for key in ("input_shape", "output_shape"):
        shape = request[key]
        if not isinstance(shape, list) or not 1 <= len(shape) <= 8 or any(
            type(dim) is not int or dim <= 0 for dim in shape
        ):
            raise ValueError("Positive static operation shapes required")
    if not isinstance(request["attributes"], dict):
        raise ValueError("Operation attributes must be an object")
    _finite(request)
    canonical(request)
    return rule


def _required(rule):
    return BASE_GATES if rule["kind"] == "correctness_repair" else PERFORMANCE_GATES


def validate_record(record, package, allow_pending=True):
    if not isinstance(record, dict) or record.get("schema_version") != 2 or record.get(
        "kind"
    ) != "xrgpu.qualification":
        raise ValueError("Unsupported qualification record")
    for field in ("package_id", "package_sha256", "runtime_revision"):
        if not package.get(field) or record.get(field) != package[field]:
            raise ValueError("Qualification package/runtime mismatch: " + field)
    if not is_digest(record.get("package_sha256")):
        raise ValueError("Invalid package digest")
    if digest({k: v for k, v in record.items() if k != "qualification_sha256"}) != record.get(
        "qualification_sha256"
    ):
        raise ValueError("Qualification digest mismatch")
    target = target_identity(record.get("target"))
    for key, value in package.get("device", {}).items():
        if key in TARGET_FIELDS and target[key] != value:
            raise ValueError("Qualification target is outside the kernel package: " + key)
    gates = record.get("gates")
    if not isinstance(gates, dict) or any(type(v) is not bool for v in gates.values()):
        raise ValueError("Qualification gates must be booleans")
    reports = record.get("report_sha256")
    if not isinstance(reports, list) or any(not is_digest(v) for v in reports) or len(
        set(reports)
    ) != len(reports):
        raise ValueError("Distinct original report digests required")
    rules = record.get("rules")
    if not isinstance(rules, list):
        raise ValueError("Qualification rules must be an array")
    seen = set()
    for rule in rules:
        validate_rule(rule)
        key = canonical(rule["request"])
        if key in seen:
            raise ValueError("Ambiguous qualification for operation descriptor")
        seen.add(key)
        if any(gates.get(g) is not True for g in _required(rule)):
            raise ValueError("Unpassed gates cannot authorize a kernel")
    status = record.get("status")
    if status == "qualified":
        if not rules or not reports or record.get("synthetic") is not False:
            raise ValueError("Production qualification requires original hardware evidence")
    elif status == "pending" and allow_pending:
        if rules:
            raise ValueError("Pending qualification cannot contain eligible rules")
    else:
        raise ValueError("Qualified hardware evidence required")
    return record


def make_record(package, target, rules, gates, report_sha256, *, synthetic=False):
    """Seal assessed evidence; failed or simulated evidence grants no eligibility."""
    for rule in rules:
        validate_rule(rule)
    eligible = [r for r in rules if not synthetic and all(gates.get(g) is True for g in _required(r))]
    record = dict(
        schema_version=2, kind="xrgpu.qualification",
        **{key: package[key] for key in ("package_id", "package_sha256", "runtime_revision")},
        target=target_identity(target), status="qualified" if eligible and report_sha256 else "pending",
        rules=eligible if report_sha256 else [], gates=dict(gates),
        report_sha256=sorted(report_sha256), synthetic=bool(synthetic),
    )
    record["qualification_sha256"] = digest(record)
    return validate_record(record, package)


def eligible_rules(record, package, device, runtime):
    if not record:
        return []
    validate_record(record, package)
    if record["status"] != "qualified" or record["runtime_revision"] != runtime:
        return []
    if any(device.get(k) != record["target"][k] for k in TARGET_FIELDS):
        return []
    return record["rules"]
