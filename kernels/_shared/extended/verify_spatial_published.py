# SPDX-License-Identifier: Apache-2.0
"""Run frozen explicit-output spatial fixtures under their unchanged policies."""

import argparse
from fractions import Fraction
import hashlib
import json
from kernels import provenance
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from kernels._shared.verify_upstream import (
    UnsupportedFixture,
    compare,
    frozen_catalog,
    load_document,
    materialize,
    tolerance,
)

ROOT = Path(__file__).resolve().parents[3]


def contract_storage(spec, declaration, contract):
    supported = {"float32", "float16", "int64", "int32", "uint32", "int16", "int8", "uint8", "bool"}
    wire = spec["dtype"]
    parameter = declaration.get("dtype", wire)
    allowed = contract.get("typeConstraints", {}).get(parameter, [parameter])
    physical = spec.get("storage", declaration.get("storage", wire))
    if physical not in supported:
        physical = wire
    logical = wire
    if wire not in allowed and len(allowed) == 1 and physical == wire and allowed[0] in supported:
        logical = allowed[0]
    return logical, physical


def independent_conflict(operation, case, arrays, expected, actual):
    """Identify proven/staged reference disagreements without changing test policy."""
    attrs = {**case.get("attrs", {}), **case.get("args", {})}
    if operation == "ai.onnx.CumSum" and arrays["x"].dtype == np.float32:
        x = arrays["x"]
        if not np.isfinite(x).all():
            return None
        axis = int(attrs.get("axis", 0)) % x.ndim
        view = np.moveaxis(x, axis, -1)
        reference = np.empty_like(view)
        reverse, exclusive = bool(attrs.get("reverse", 0)), bool(attrs.get("exclusive", 0))
        for prefix in np.ndindex(view.shape[:-1]):
            total = Fraction(0)
            for i in range(view.shape[-1] - 1, -1, -1) if reverse else range(view.shape[-1]):
                current = Fraction(float(view[prefix + (i,)]))
                if not exclusive:
                    total += current
                value = np.float32(float(total))
                # A representable exact prefix proves this mismatch is not a
                # reduction-order or intermediate-rounding ambiguity.
                if Fraction(float(value)) != total:
                    return None
                reference[prefix + (i,)] = value
                if exclusive:
                    total += current
        reference = np.moveaxis(reference, -1, axis)
        name = next(iter(expected))
        if np.array_equal(reference, actual[name]) and not np.array_equal(
            reference, expected[name]
        ):
            return dict(
                method="exact_fraction_prefix_of_stored_float32_inputs",
                reference=reference,
                name=name,
            )
    if (
        operation == "ai.onnx.GroupNormalization"
        and arrays["x"].dtype == np.float16
        and attrs.get("stash_type", 1) == 1
    ):
        x = arrays["x"].astype(np.float64)
        n, c = x.shape[:2]
        groups = int(attrs["num_groups"])
        grouped = x.reshape(n, groups, -1)
        mean = grouped.mean(-1, keepdims=True)
        variance = (grouped * grouped).mean(-1, keepdims=True) - mean * mean
        normalized = (
            ((grouped - mean) / np.sqrt(variance + attrs.get("epsilon", 1e-5)))
            .reshape(x.shape)
            .astype(np.float16)
        )
        shape = (1, c) + (1,) * (x.ndim - 2)
        reference = (
            (normalized * arrays["scale"].reshape(shape)).astype(np.float16)
            + arrays["bias"].reshape(shape)
        ).astype(np.float16)
        name = next(iter(expected))
        if np.array_equal(reference, actual[name]) and not np.array_equal(
            reference, expected[name]
        ):
            return dict(
                method="independent_float64_statistics_with_ONNX21_explicit_half_cast_and_affine_stages",
                reference=reference,
                name=name,
            )
    return None


def check(package, runner, output, *, maximum=1_000_000, selected=None, selected_cases=None):
    owned = set(json.loads((ROOT / "kernels/_shared/extended/spatial_ops.json").read_text()))
    catalog = frozen_catalog(provenance.PROVENANCE)
    records = []
    failure_root = output.with_name(output.stem + "-failures")
    if output.exists() or failure_root.exists():
        raise FileExistsError("Conformance evidence exists; choose a new output name")
    failure_root.mkdir(parents=True, exist_ok=True)
    for entry in catalog["entries"]:
        if entry["id"] not in owned or selected and entry["id"] not in selected:
            continue
        document = load_document(provenance.PROVENANCE, entry)
        contract = provenance.document(entry, "contract")
        for case in document["cases"]:
            if selected_cases is not None and (entry["id"], case["name"]) not in selected_cases:
                continue
            row = dict(
                operator=entry["id"],
                case=case["name"],
                status="unsupported_fixture",
                source_revision=entry["revision"],
                source_test_sha256=entry["files"][entry["tests"]],
            )
            if case.get("skipGpu"):
                row["upstream_skip_gpu"] = case["skipGpu"]
            try:
                outputs = case.get("outputs", {})
                if not outputs or not all(
                    isinstance(spec, dict) and spec.get("data", {}).get("kind") == "values"
                    for spec in outputs.values()
                ):
                    raise UnsupportedFixture("No complete published explicit output oracle")
                if case.get("expectError") or case.get("expect_error"):
                    raise UnsupportedFixture(
                        "Error-only fixture handled separately from numerical checks"
                    )
                arrays = {
                    name: materialize(spec, document, maximum)
                    for name, spec in case.get("inputs", {}).items()
                    if spec is not None
                }
                expected = {
                    name: materialize(spec, document, maximum) for name, spec in outputs.items()
                }
                expected = {
                    name: expected[name] for name in contract["outputs"] if name in expected
                }
                with tempfile.TemporaryDirectory(prefix="xrgpu-spatial-published-") as temporary:
                    base = Path(temporary)
                    inputs = []
                    for index, (name, array) in enumerate(arrays.items()):
                        spec = case["inputs"][name]
                        logical, physical = contract_storage(
                            spec, contract["inputs"].get(name, {}), contract
                        )
                        arrays[name] = array.astype(logical)
                        array = arrays[name]
                        if physical not in {
                            "float32",
                            "float16",
                            "int64",
                            "int32",
                            "uint32",
                            "int16",
                            "int8",
                            "uint8",
                            "bool",
                        }:
                            physical = str(array.dtype)
                        physical_dtype = np.dtype(physical)
                        if array.dtype.kind in "iu" and physical_dtype.kind in "iu":
                            info = np.iinfo(physical_dtype)
                            if array.size and (
                                int(array.min()) < info.min or int(array.max()) > info.max
                            ):
                                raise UnsupportedFixture(
                                    "Input does not fit declared checked physical storage"
                                )
                        path = base / f"input{index}.bin"
                        array.astype(physical_dtype).tofile(path)
                        inputs.append(
                            dict(
                                name=name,
                                dtype=logical,
                                storage_dtype=physical,
                                shape=list(array.shape),
                                file=str(path),
                            )
                        )
                    requested = []
                    for name, array in expected.items():
                        logical, physical = contract_storage(
                            outputs[name], contract["outputs"].get(name, {}), contract
                        )
                        expected[name] = array.astype(logical)
                        array = expected[name]
                        if physical not in {
                            "float32",
                            "float16",
                            "int64",
                            "int32",
                            "uint32",
                            "int16",
                            "int8",
                            "uint8",
                            "bool",
                        }:
                            physical = str(array.dtype)
                        requested.append(
                            dict(
                                name=name,
                                dtype=str(array.dtype),
                                storage_dtype=physical,
                                shape=list(array.shape),
                            )
                        )
                    job = dict(
                        operation=entry["id"],
                        package=str(package),
                        inputs=inputs,
                        outputs=requested,
                        attrs=case.get("attrs", {}),
                        args=case.get("args", {}),
                        output_directory=str(base / "actual"),
                    )
                    path = base / "job.json"
                    path.write_text(json.dumps(job, allow_nan=False))
                    process = subprocess.run(
                        [str(runner), str(path)], capture_output=True, text=True, timeout=60
                    )
                    if process.returncode:
                        raise RuntimeError(process.stderr.strip())
                    answer = json.loads(process.stdout)
                    info = answer.get("info", {})
                    if (
                        info.get("backend") != "software_vulkan"
                        or info.get("physical_qualification") is not False
                        or info.get("cpu_compute_fallback") is not False
                        or info.get("executed") is not True
                    ):
                        raise RuntimeError("Expected native software Vulkan evidence")
                    comparisons = []
                    actual_arrays = {}
                    if len(answer["outputs"]) != len(expected):
                        raise ValueError("Output count mismatch")
                    for actual, (name, reference) in zip(answer["outputs"], expected.items()):
                        if actual["name"] != name:
                            raise ValueError("Output name mismatch")
                        physical = np.dtype(actual.get("storage_dtype", actual["dtype"]))
                        values = (
                            np.fromfile(actual["file"], dtype=physical)
                            .astype(reference.dtype)
                            .reshape(actual["shape"])
                        )
                        actual_arrays[name] = values
                        policy = tolerance(outputs[name], case)
                        comparisons.append(
                            dict(name=name, policy=policy, **compare(values, reference, **policy))
                        )
                    row.update(
                        status="passed" if all(c["passed"] for c in comparisons) else "failed",
                        comparisons=comparisons,
                        evidence=info,
                    )
                    conflict = (
                        independent_conflict(entry["id"], case, arrays, expected, actual_arrays)
                        if row["status"] == "failed"
                        else None
                    )
                    if conflict is not None:
                        row.update(
                            status="reference_conflict", independent_oracle=conflict["method"]
                        )
                    if row["status"] in {"failed", "reference_conflict"}:
                        destination = failure_root / f"case-{len(records):05d}"
                        destination.mkdir(exist_ok=True)
                        for name, array in arrays.items():
                            array.tofile(destination / (name + ".input.bin"))
                        for name, array in expected.items():
                            array.tofile(destination / (name + ".expected.bin"))
                        for name, array in actual_arrays.items():
                            array.tofile(destination / (name + ".actual.bin"))
                        if conflict is not None:
                            conflict["reference"].tofile(
                                destination / (conflict["name"] + ".independent.bin")
                            )
                        (destination / "case.json").write_text(
                            json.dumps(case, indent=2, allow_nan=False) + "\n"
                        )
                        row["failure_artifacts"] = str(destination)
            except UnsupportedFixture as error:
                row["reason"] = str(error)
            except Exception as error:
                row.update(status="failed", error=str(error))
            records.append(row)
        report = dict(
            schema_version=1,
            kind="xrgpu.spatial-published-software-vulkan",
            physical_qualification=False,
            catalog_sha256=catalog["catalog_sha256"],
            package=str(package),
            runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
            source_sha256=json.loads((package / "development.json").read_text()).get(
                "source_sha256"
            )
            if (package / "development.json").is_file()
            else None,
            results=records,
            passed=sum(r["status"] == "passed" for r in records),
            failed=sum(r["status"] == "failed" for r in records),
            unsupported=sum(r["status"] == "unsupported_fixture" for r in records),
            reference_conflicts=sum(r["status"] == "reference_conflict" for r in records),
            complete=selected_cases is None and all(r["status"] == "passed" for r in records),
            scope="previous_failure_regressions"
            if selected_cases is not None
            else "all_owned_frozen_cases",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator", action="append")
    parser.add_argument(
        "--failed-from",
        type=Path,
        help="Retest only failures/conflicts from an immutable prior report",
    )
    args = parser.parse_args()
    selected_cases = None
    if args.failed_from:
        baseline = json.loads(args.failed_from.read_text())
        selected_cases = {
            (r["operator"], r["case"])
            for r in baseline["results"]
            if r["status"] in {"failed", "reference_conflict"}
        }
        if not selected_cases:
            raise ValueError("Prior report has no failed/conflicting cases")
    report = check(
        args.package.resolve(),
        args.runner.resolve(),
        args.output.resolve(),
        selected=set(args.operator or []),
        selected_cases=selected_cases,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))
    return bool(report["failed"] or report["reference_conflicts"])


if __name__ == "__main__":
    raise SystemExit(main())
