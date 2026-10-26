# SPDX-License-Identifier: Apache-2.0
"""Execute frozen published fixtures with the native test-only software Vulkan runner.

Generator semantics are deliberately limited to explicit values, constant,
repeating cycle, and inclusive linspace. Other generators remain uncovered.
WebGPU-specific skip annotations are retained as provenance, not applied to
native execution. This harness never grants physical GPU/XR qualification.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

import numpy as np

from kernels import provenance
from kernels._shared.build import verify
from kernels._shared.verify_local import TYPES, operators, reference

ROOT = Path(__file__).resolve().parents[2]
MAX_ELEMENTS = 1_000_000


class UnsupportedFixture(ValueError):
    """A published fixture that this harness cannot faithfully materialize."""


class ReferenceConflict(ValueError):
    """Published explicit outputs disagree with an independent reference."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def shape_size(shape, maximum=MAX_ELEMENTS):
    if not isinstance(shape, list) or len(shape) > 16 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid published tensor shape")
    count = math.prod(shape)
    if count > maximum:
        raise UnsupportedFixture("Tensor exceeds configured element limit")
    return count


def resolve_values(value, document, seen=()):
    if isinstance(value, list):
        return value
    if not isinstance(value, dict) or set(value) != {"$ref"}:
        raise ValueError("Fixture values must be an array or local fixtureArrays reference")
    reference_path = value["$ref"]
    prefix = "#/fixtureArrays/"
    if not isinstance(reference_path, str) or not reference_path.startswith(prefix):
        raise ValueError("Only local fixtureArrays JSON references are allowed")
    key = reference_path[len(prefix):]
    if "/" in key or re.search(r"~(?![01])", key):
        raise ValueError("Invalid fixtureArrays JSON pointer")
    key = key.replace("~1", "/").replace("~0", "~")
    if key in seen or len(seen) >= 16:
        raise ValueError("Cyclic or excessive fixtureArrays reference")
    arrays = document.get("fixtureArrays", {})
    if key not in arrays:
        raise ValueError("Missing fixtureArrays reference")
    return resolve_values(arrays[key], document, (*seen, key))


def number(value, dtype):
    if isinstance(value, str):
        special = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": -float("inf")}
        if value in special and dtype.kind == "f":
            return special[value]
        if re.fullmatch(r"[+-]?[0-9]+", value) and dtype.kind in "iu":
            return int(value)
        raise ValueError("Unsupported numeric string in published fixture")
    if type(value) not in (int, float, bool):
        raise ValueError("Invalid published fixture number")
    return value


def materialize(spec, document, maximum=MAX_ELEMENTS):
    dtype_name = spec.get("dtype")
    if dtype_name not in TYPES:
        raise UnsupportedFixture("Fixture dtype is not supported by the typed harness")
    dtype = np.dtype(dtype_name).newbyteorder("<")
    shape = spec.get("shape")
    count = shape_size(shape, maximum)
    data = spec.get("data")
    if not isinstance(data, dict):
        raise UnsupportedFixture("Fixture has no explicit data generator")
    kind = data.get("kind")
    allowed = {"values": {"kind", "values"}, "constant": {"kind", "value"},
               "cycle": {"kind", "values"}, "linspace": {"kind", "start", "end"}}
    if kind not in allowed:
        raise UnsupportedFixture("Unsupported published generator: " + str(kind))
    if set(data) != allowed[kind]:
        raise UnsupportedFixture("Unsupported options for published generator: " + str(kind))
    if kind in ("values", "cycle"):
        values = [number(n, dtype) for n in resolve_values(data["values"], document)]
        if kind == "values" and len(values) != count:
            raise ValueError("Published values count differs from tensor shape")
        if kind == "cycle" and not values and count:
            raise ValueError("Cannot repeat an empty cycle for a nonempty tensor")
        source = np.asarray(values, dtype=dtype)
        array = source if kind == "values" else np.resize(source, count)
    elif kind == "constant":
        array = np.full(count, number(data["value"], dtype), dtype=dtype)
    else:
        start, end = number(data["start"], dtype), number(data["end"], dtype)
        # Use a real-valued inclusive sequence, then convert once to the declared
        # storage type. Integer np.linspace(dtype=...) instead floors negatives.
        array = np.linspace(start, end, count, dtype=np.float64).astype(dtype)
    return np.asarray(array, dtype=dtype).reshape(tuple(shape))


def tolerance(output, case):
    absolute = output.get("tolerance", case.get("tolerance", 0))
    relative = output.get("relTolerance", case.get("relTolerance", 0))
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (absolute, relative)):
        raise ValueError("Invalid published comparison tolerance")
    allow_nan = output.get("allowNaN", case.get("allowNaN", False))
    if type(allow_nan) is not bool:
        raise ValueError("Published allowNaN must be a boolean")
    return dict(atol=absolute, rtol=relative, allow_nan=allow_nan,
                tolerance_origin="output" if "tolerance" in output else "case" if "tolerance" in case else "absent: exact default")


def compare(actual, expected, *, atol=0, rtol=0, allow_nan=False, **_):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return dict(passed=False, reason="Output shape or dtype mismatch")
    if expected.dtype.kind != "f":
        return dict(passed=bool(np.array_equal(actual, expected)), comparison="exact")
    finite = np.isfinite(actual) & np.isfinite(expected)
    special = (actual == expected) & ~finite
    if allow_nan:
        special |= np.isnan(actual) & np.isnan(expected)
    error = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
    limit = atol + rtol * np.abs(expected[finite].astype(np.float64))
    matched = bool(np.all(finite | special) and np.all(error <= limit))
    return dict(passed=matched, comparison="absolute_relative", atol=atol, rtol=rtol,
                allow_nan=allow_nan, max_absolute_error=float(error.max()) if error.size else 0.0,
                nonfinite_mismatches=int(np.count_nonzero(~finite & ~special)))


def frozen_catalog(root=provenance.PROVENANCE):
    return provenance.catalog(root)


def load_document(root, entry):
    return provenance.document(entry, "tests", root=root)


def inputs_for(spec, case, document, maximum):
    if case.get("attrs"):
        raise UnsupportedFixture("Native registry does not implement these published attributes")
    declared = case.get("inputs", {})
    if set(declared) != set(spec["inputs"]):
        raise UnsupportedFixture("Published input interface differs from native operator interface")
    dtype_names = {v.get("dtype") for v in declared.values()}
    if len(dtype_names) != 1:
        raise UnsupportedFixture("Raw typed runner requires a shared input dtype")
    dtype_name = next(iter(dtype_names))
    if dtype_name not in TYPES or not spec["mask"] & (1 << TYPES.index(dtype_name)):
        raise UnsupportedFixture("Published dtype is outside this operator's native implementation")
    return dtype_name, [materialize(declared[n], document, maximum) for n in spec["inputs"]]


def case_check(package, runner, spec, case, document, *, maximum=MAX_ELEMENTS, timeout=30, failure_root=None):
    row = dict(operator=spec["op"], case=case["name"], status="failed")
    if case.get("skipGpu"):
        row["upstream_skip_gpu"] = case["skipGpu"]
    if case.get("requires"):
        row["upstream_requires"] = case["requires"]
    temporary = None
    try:
        dtype_name, inputs = inputs_for(spec, case, document, maximum)
        row["dtype"] = dtype_name
        outputs = case.get("outputs", {})
        if len(outputs) != 1:
            raise UnsupportedFixture("This native operator registry produces one output")
        output_name, output_spec = next(iter(outputs.items()))
        expected_shape = tuple(output_spec["shape"])
        shape_size(output_spec["shape"], maximum)
        if output_spec["dtype"] != dtype_name:
            raise UnsupportedFixture("Native output dtype differs from published fixture")
        with np.errstate(all="ignore"):
            if "data" in output_spec:
                expected = materialize(output_spec, document, maximum)
                row["oracle"] = "published_explicit_output"
            elif math.prod(expected_shape) == 0:
                expected = np.empty(expected_shape, dtype=dtype_name)
                row["oracle"] = "published_empty_output"
            else:
                expected = np.asarray(reference(spec["op"].split(".")[-1], inputs))
                row["oracle"] = "independent_numpy_reference"
        if expected.shape != expected_shape or expected.dtype != np.dtype(dtype_name):
            raise ValueError("Independent reference differs from published output metadata")
        rules = tolerance(output_spec, case)
        row["comparison_policy"] = rules
        independent = None
        conflict = False
        if "data" in output_spec:
            with np.errstate(all="ignore"):
                independent = (np.empty(expected_shape, dtype=dtype_name) if expected.size == 0
                               else np.asarray(reference(spec["op"].split(".")[-1], inputs)))
            row["published_reference_comparison"] = compare(expected, independent, **rules)
            conflict = not row["published_reference_comparison"]["passed"]
            row["independent_reference_sha256"] = hashlib.sha256(independent.tobytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="xrgpu-upstream-") as work:
            temporary = Path(work)
            prefix = temporary / "actual"
            command = [str(runner), str(package), spec["op"], dtype_name, str(prefix)]
            for i, (name, array) in enumerate(zip(spec["inputs"], inputs)):
                path = temporary / f"input{i}.bin"
                path.write_bytes(array.tobytes(order="C"))
                shape = ",".join(map(str, array.shape)) if array.shape else "scalar"
                command += [name, shape, str(path)]
            process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            (temporary / "stderr.txt").write_text(process.stderr)
            if process.returncode:
                raise RuntimeError(process.stderr.strip() or "Native fixture runner failed")
            evidence = json.loads(process.stdout)
            if (evidence.get("backend") != "software_vulkan"
                    or evidence.get("physical_qualification") is not False
                    or evidence.get("cpu_compute_fallback") is not False):
                raise RuntimeError("Published-fixture harness requires explicit software Vulkan execution")
            row["evidence"] = evidence
            descriptors = evidence.get("outputs", [])
            if (len(descriptors) != 1 or descriptors[0].get("name") != output_name
                    or descriptors[0].get("shape") != list(expected_shape)
                    or descriptors[0].get("dtype") != dtype_name):
                raise ValueError("Native output metadata differs from published output interface")
            raw = (temporary / "actual.0.bin").read_bytes()
            if len(raw) != expected.size * expected.dtype.itemsize:
                raise ValueError("Native output byte count differs from published output")
            row["output_sha256"] = hashlib.sha256(raw).hexdigest()
            actual = np.frombuffer(raw, dtype=dtype_name).reshape(expected_shape)
            row["comparison"] = compare(actual, expected, **rules)
            if conflict or not row["comparison"]["passed"]:
                (temporary / "expected.bin").write_bytes(expected.tobytes(order="C"))
                if independent is not None:
                    (temporary / "independent-reference.bin").write_bytes(independent.tobytes(order="C"))
                    row["native_independent_comparison"] = compare(actual, independent, **rules)
                if failure_root is not None:
                    destination = Path(failure_root)
                    shutil.copytree(temporary, destination)
                    row["failure_artifacts"] = str(destination)
                if conflict:
                    raise ReferenceConflict("Published explicit output disagrees with the independent typed oracle")
                raise ValueError("Native result differs from published fixture/oracle")
            row["status"] = "passed"
    except ReferenceConflict as error:
        row.update(status="reference_conflict", reason=str(error))
    except UnsupportedFixture as error:
        row.update(status="unsupported", reason=str(error))
    except Exception as error:
        row["error"] = str(error)
    return row


def check(package, runner, output, selected=None, *, catalog_root=provenance.PROVENANCE, maximum=MAX_ELEMENTS, timeout=30):
    package, runner, output = Path(package).resolve(), Path(runner).resolve(), Path(output).absolute()
    manifest = verify(package)
    catalog = frozen_catalog(catalog_root)
    by_id = {e["id"]: e for e in catalog["entries"]}
    specs = [s for s in operators() if s["op"] in manifest["operations"]]
    if selected and not set(selected).issubset({s["op"] for s in specs}):
        raise ValueError("Choose implemented native operator identifiers")
    if maximum < 0 or timeout <= 0:
        raise ValueError("Invalid published fixture runtime limits")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(kind="xrgpu.published-fixture-software-vulkan", schema_version=1,
                  backend="software_vulkan", physical_qualification=False,
                  package_sha256=manifest["package_sha256"], runner_sha256=sha(runner),
                  oracle_source_sha256=sha(Path(reference.__code__.co_filename)),
                  harness_source_sha256=sha(Path(__file__)),
                  environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()),
                  catalog_sha256=catalog["catalog_sha256"], max_elements=maximum,
                  note="Unsupported fixtures and omitted generators are not passing conformance evidence.",
                  results=[])
    failure_parent = output.parent / (output.stem + "-failures")
    if failure_parent.exists():
        raise FileExistsError("Choose a new report name; existing failure evidence is preserved")
    counter = 0
    libraries = [runner.parent / "libxrgpu_operators.dylib", runner.parent / "libxrgpu_operators.so"]
    software_library = next((p for p in libraries if p.is_file()), None)
    if software_library is not None:
        report["software_library_sha256"] = sha(software_library)
    for spec in specs:
        if selected and spec["op"] not in selected:
            continue
        entry = by_id[spec["op"]]
        document = load_document(catalog_root, entry)
        for case in document["cases"]:
            counter += 1
            before = sha(software_library) if software_library is not None else None
            row = case_check(package, runner, spec, case, document, maximum=maximum, timeout=timeout,
                             failure_root=failure_parent / f"case-{counter:04d}")
            if software_library is not None:
                row["software_library_sha256"] = before
                if sha(software_library) != before:
                    row.update(status="failed", error="Software Vulkan runtime changed during fixture execution")
            row.update(source_revision=entry["revision"], source_test_sha256=entry["files"][entry["tests"]])
            report["results"].append(row)
            counts = Counter(r["status"] for r in report["results"])
            report.update(total=len(report["results"]), passed=counts["passed"], failed=counts["failed"],
                          unsupported=counts["unsupported"], reference_conflicts=counts["reference_conflict"],
                          complete_coverage=counts["unsupported"] == counts["reference_conflict"] == 0)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator", action="append")
    parser.add_argument("--catalog-root", type=Path, default=provenance.PROVENANCE)
    parser.add_argument("--max-elements", type=int, default=MAX_ELEMENTS)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    result = check(args.package, args.runner, args.output, args.operator, catalog_root=args.catalog_root,
                   maximum=args.max_elements, timeout=args.timeout)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    return bool(result["failed"] or result["unsupported"] or result["reference_conflicts"])


if __name__ == "__main__":
    raise SystemExit(main())
