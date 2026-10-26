# SPDX-License-Identifier: Apache-2.0
"""Exercise real native tensor-family shaders against independent references."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from .tensor_reference import representatives


def run(package, runner, output, plan_only=False, selected=None, cases=None):
    if Path(output).exists():
        raise FileExistsError("Conformance evidence exists; choose a new output name")
    reference_paths = [
        Path(__file__).with_name("tensor_reference.py"),
        Path(__file__).with_name("tensor_features.py"),
    ]
    reference_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference_paths}
    rows = []
    for case in representatives() if cases is None else cases:
        if selected and case["operation"] not in selected:
            continue
        row = dict(
            operation=case["operation"], name=case.get("name", "representative"), status="failed"
        )
        try:
            with tempfile.TemporaryDirectory(prefix="xrgpu-tensor-check-") as temporary:
                root = Path(temporary)
                inputs = []
                for n, (name, array) in enumerate(case["inputs"].items()):
                    array = np.asarray(array)
                    path = root / f"input{n}.bin"
                    path.write_bytes(array.tobytes())
                    inputs.append(
                        dict(
                            name=name,
                            dtype=str(array.dtype),
                            shape=list(array.shape),
                            file=str(path),
                        )
                    )
                wanted = [
                    dict(
                        name=name,
                        dtype=str(np.asarray(array).dtype),
                        shape=list(np.asarray(array).shape),
                    )
                    for name, array in case["expected"].items()
                ]
                job = dict(
                    package=str(Path(package).resolve()),
                    operation=case["operation"],
                    inputs=inputs,
                    attrs=case["attrs"],
                    args=case["args"],
                    outputs=wanted,
                    output_directory=str(root / "outputs"),
                    plan_only=plan_only,
                )
                path = root / "job.json"
                path.write_text(json.dumps(job, allow_nan=False))
                process = subprocess.run(
                    [str(Path(runner).resolve()), str(path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if process.returncode:
                    raise RuntimeError(process.stderr.strip())
                answer = json.loads(process.stdout)
                row["info"] = answer["info"]
                if not plan_only and (
                    answer["info"].get("backend") != "software_vulkan"
                    or answer["info"].get("physical_qualification") is not False
                ):
                    raise RuntimeError(
                        "Independent local checks require the explicit software Vulkan test target"
                    )
                for desc, (name, expected) in zip(answer["outputs"], case["expected"].items()):
                    expected = np.asarray(expected)
                    capacity_ok = (
                        plan_only
                        and case["operation"]
                        in {
                            "ai.onnx.Compress",
                            "ai.onnx.NonZero",
                            "ai.onnx.Unique",
                            "ai.onnx.NonMaxSuppression",
                        }
                        and len(desc["shape"]) == expected.ndim
                        and all(a >= b for a, b in zip(desc["shape"], expected.shape))
                    )
                    if desc["name"] != name or (
                        desc["shape"] != list(expected.shape) and not capacity_ok
                    ):
                        raise ValueError(
                            "Native output metadata differs from independent reference"
                        )
                    if not plan_only:
                        actual = np.frombuffer(
                            Path(desc["file"]).read_bytes(), dtype=desc["storage_dtype"]
                        ).reshape(desc["shape"])
                        if expected.dtype.kind == "f":
                            passed = np.allclose(
                                actual,
                                expected,
                                rtol=case.get("rtol", 2e-5),
                                atol=case.get("atol", 2e-5),
                                equal_nan=True,
                            )
                        else:
                            passed = np.array_equal(actual, expected)
                        if not passed:
                            row.setdefault("mismatches", {})[name] = dict(
                                actual=actual.tolist(), expected=expected.tolist()
                            )
                if row.get("mismatches"):
                    raise ValueError("Native tensor kernel differs from independent oracle")
                row["status"] = "passed"
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)
    report = dict(
        kind="xrgpu.tensor-family-check",
        physical_qualification=False,
        reference_sha256=reference_hashes,
        plan_only=plan_only,
        runner_sha256=hashlib.sha256(Path(runner).read_bytes()).hexdigest(),
        passed=sum(row["status"] == "passed" for row in rows),
        failed=sum(row["status"] != "passed" for row in rows),
        results=rows,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--operator", action="append")
    parser.add_argument("--features", action="store_true")
    args = parser.parse_args()
    from .tensor_features import features

    report = run(
        args.package,
        args.runner,
        args.output,
        args.plan_only,
        args.operator,
        features() if args.features else None,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))
    return bool(report["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
