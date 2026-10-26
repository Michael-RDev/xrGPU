# SPDX-License-Identifier: Apache-2.0
"""Run frozen local contracts through native Vulkan with independent references.

No downloads or browser APIs. Unspecified generators, resource caps and missing
independent references remain explicit gaps and never become passing evidence.
"""

import argparse
import hashlib
import importlib
import inspect
import platform
import json
from kernels import provenance
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from kernels._shared.verify_upstream import (
    UnsupportedFixture,
    materialize,
    compare,
    tolerance,
    frozen_catalog,
    load_document,
)

ROOT = Path(__file__).resolve().parents[3]
DTYPES = {"float32", "float16", "int32", "uint32", "int16", "int8", "uint8", "int64", "bool"}


def logical_type(manifest, name, physical, output=False):
    field = manifest.get("outputs" if output else "inputs", {}).get(name, {})
    variable = field.get("dtype")
    allowed = manifest.get("typeConstraints", {}).get(
        variable, [variable] if variable in DTYPES else []
    )
    if len(allowed) == 1 and allowed[0] == "int64" and physical in {"int32", "uint32"}:
        return "int64"
    return physical


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, np.generic):
        return value.item()
    return value


def reference_for(family):
    try:
        return importlib.import_module("kernels._shared.extended." + family + "_oracle").oracle
    except (ModuleNotFoundError, AttributeError):
        try:
            return importlib.import_module("kernels._shared.extended." + family + "_reference").oracle
        except (ModuleNotFoundError, AttributeError):
            return None


def check(package, runner, output, family="tensor", maximum=1_000_000, selected=None, limit=None):
    package, runner, output = map(lambda p: Path(p).resolve(), (package, runner, output))
    membership = set(json.loads((ROOT / "kernels/_shared/extended" / f"{family}_ops.json").read_text()))
    catalog = frozen_catalog(provenance.PROVENANCE)
    oracle = reference_for(family)
    reference_sha256 = (
        hashlib.sha256(Path(inspect.getfile(oracle)).read_bytes()).hexdigest() if oracle else None
    )
    harness_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = []
    failures = output.parent / (output.stem + "-failures")
    if output.exists() or failures.exists():
        raise FileExistsError("Conformance evidence exists; choose a new output name")
    output.parent.mkdir(parents=True, exist_ok=True)
    for entry in catalog["entries"]:
        if entry["id"] not in membership or selected and entry["id"] not in selected:
            continue
        manifest = provenance.document(entry, "contract")
        document = load_document(provenance.PROVENANCE, entry)
        for case in document["cases"]:
            if limit is not None and len(rows) >= limit:
                break
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
                if case.get("expectError") or case.get("expect_error"):
                    raise UnsupportedFixture(
                        "Invalid-invocation fixture needs separate negative validation"
                    )
                arrays = {
                    n: materialize(s, document, maximum)
                    for n, s in case.get("inputs", {}).items()
                    if s is not None
                }
                declared = case["outputs"]
                explicit = {}
                independent = None
                for name, spec in declared.items():
                    if "data" in spec:
                        explicit[name] = materialize(spec, document, maximum)
                if oracle:
                    try:
                        independent = oracle(
                            entry["id"],
                            arrays,
                            case.get("attrs", {}),
                            case.get("args", {}),
                            declared,
                        )
                    except (NotImplementedError, KeyError) as error:
                        if len(explicit) != len(declared):
                            raise UnsupportedFixture(
                                "Independent oracle unavailable: " + str(error)
                            )
                if len(explicit) == len(declared):
                    expected = explicit
                    row["oracle"] = "published_explicit"
                elif independent is not None:
                    expected = {
                        n: np.asarray(independent[n]).astype(s["dtype"])
                        for n, s in declared.items()
                    }
                    row["oracle"] = "independent_reference"
                else:
                    raise UnsupportedFixture("No complete published or independent reference")
                for name, array in expected.items():
                    if list(array.shape) != declared[name]["shape"]:
                        raise ValueError(
                            "Independent oracle shape differs from declared contract for " + name
                        )
                with tempfile.TemporaryDirectory(prefix="xrgpu-frozen-") as temporary:
                    base = Path(temporary)
                    inputs = []
                    for index, (name, array) in enumerate(arrays.items()):
                        path = base / f"input{index}.bin"
                        array.tofile(path)
                        physical = str(array.dtype)
                        inputs.append(
                            dict(
                                name=name,
                                dtype=logical_type(manifest, name, physical),
                                storage_dtype=physical,
                                shape=list(array.shape),
                                file=str(path),
                            )
                        )
                    requested = [
                        dict(
                            name=n,
                            dtype=logical_type(manifest, n, str(v.dtype), True),
                            storage_dtype=str(v.dtype),
                            shape=list(v.shape),
                        )
                        for n, v in expected.items()
                    ]
                    job = dict(
                        package=str(package),
                        operation=entry["id"],
                        inputs=inputs,
                        outputs=requested,
                        attrs=case.get("attrs", {}),
                        args=case.get("args", {}),
                        output_directory=str(base / "output"),
                    )
                    jobpath = base / "job.json"
                    jobpath.write_text(json.dumps(safe_json(job), allow_nan=False))
                    process = subprocess.run(
                        [str(runner), str(jobpath)], capture_output=True, text=True, timeout=60
                    )
                    if process.returncode:
                        raise RuntimeError(process.stderr.strip())
                    answer = json.loads(process.stdout)
                    info = answer["info"]
                    if (
                        info.get("backend") != "software_vulkan"
                        or info.get("physical_qualification") is not False
                    ):
                        raise RuntimeError(
                            "Expected explicitly labelled native software Vulkan test build"
                        )
                    if len(answer["outputs"]) != len(expected):
                        raise ValueError("Native output count mismatch")
                    comparisons = []
                    actuals = {}
                    conflicts = []
                    for desc, (name, wanted) in zip(answer["outputs"], expected.items()):
                        if desc["name"] != name:
                            raise ValueError("Native output name mismatch")
                        actual = (
                            np.fromfile(desc["file"], dtype=desc["storage_dtype"])
                            .reshape(desc["shape"])
                            .astype(wanted.dtype)
                        )
                        actuals[name] = actual
                        policy = tolerance(declared[name], case)
                        comparisons.append(
                            dict(name=name, policy=policy, **compare(actual, wanted, **policy))
                        )
                        if name in explicit and independent is not None:
                            independent_value = np.asarray(independent[name]).astype(wanted.dtype)
                            cross = compare(wanted, independent_value, **policy)
                            if not cross["passed"]:
                                conflicts.append(dict(name=name, **cross))
                    row.update(
                        status="reference_conflict"
                        if conflicts
                        else "passed"
                        if all(c["passed"] for c in comparisons)
                        else "failed",
                        comparisons=comparisons,
                        evidence=info,
                    )
                    if conflicts:
                        row["oracle_conflicts"] = conflicts
                    if row["status"] != "passed":
                        dest = failures / f"case-{len(rows):05d}"
                        dest.mkdir(parents=True, exist_ok=True)
                        (dest / "case.json").write_text(
                            json.dumps(safe_json(case), indent=2, allow_nan=False)
                        )
                        for n, v in arrays.items():
                            v.tofile(dest / (n + ".input.bin"))
                        for n, v in expected.items():
                            v.tofile(dest / (n + ".expected.bin"))
                        for n, v in actuals.items():
                            v.tofile(dest / (n + ".actual.bin"))
                        if independent is not None:
                            for n, v in independent.items():
                                np.asarray(v).tofile(dest / (n + ".independent.bin"))
                        row["failure_artifacts"] = str(dest)
            except UnsupportedFixture as error:
                row["reason"] = str(error)
            except Exception as error:
                row.update(status="failed", error=str(error))
            rows.append(row)
        report = dict(
            kind="xrgpu.extended-frozen-software-vulkan",
            schema_version=1,
            family=family,
            selected_operators=selected,
            fixture_limit=limit,
            element_limit=maximum,
            reference_sha256=reference_sha256,
            harness_sha256=harness_sha256,
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            physical_qualification=False,
            package=str(package),
            runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
            catalog_sha256=catalog["catalog_sha256"],
            results=rows,
        )
        for status, label in [
            ("passed", "passed"),
            ("failed", "failed"),
            ("unsupported_fixture", "unsupported"),
            ("reference_conflict", "reference_conflicts"),
        ]:
            report[label] = sum(r["status"] == status for r in rows)
        report["total"] = len(rows)
        report["complete_coverage"] = (
            report["failed"] == report["unsupported"] == report["reference_conflicts"] == 0
        )
        output.write_text(json.dumps(safe_json(report), indent=2, allow_nan=False) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--family", choices=["tensor", "spatial", "complex"], default="tensor")
    p.add_argument("--operator", action="append")
    p.add_argument("--max-elements", type=int, default=1_000_000)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    r = check(a.package, a.runner, a.output, a.family, a.max_elements, a.operator, a.limit)
    print(json.dumps({k: v for k, v in r.items() if k != "results"}))
    return bool(r["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
