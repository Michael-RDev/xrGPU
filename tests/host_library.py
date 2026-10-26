"""Run all exported model fixtures through the built native CPU Session API."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from runtime.model_bundle import verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def validate(directory, runner, output):
    directory, runner, output = Path(directory).resolve(), Path(runner).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for bundle in sorted(directory.iterdir()):
        if not (bundle / "bundle.json").is_file():
            continue
        manifest = verify_bundle(bundle)
        for fixture in manifest["fixtures"]:
            destination = output / bundle.name / str(fixture["seed"])
            destination.mkdir(parents=True, exist_ok=True)
            command = [str(runner), str(bundle), *(str(bundle / item["file"]) for item in fixture["inputs"])]
            result = subprocess.run(command, cwd=destination, text=True, capture_output=True)
            row = dict(bundle=bundle.name, model_id=manifest["model_id"],
                       bundle_sha256=manifest["bundle_sha256"],
                       runtime_revision=manifest["runtime_revision"],
                       kernel_package=manifest.get("kernel_package"),
                       seed=fixture["seed"], passed=False)
            try:
                if result.returncode:
                    raise RuntimeError(result.stderr[-4000:])
                info = json.loads(result.stdout)
                if info["backend"] != "cpu" or info["build_type"] != "production":
                    raise ValueError("Expected production CPU host execution")
                metrics = []
                for filename, spec in zip(info["output_files"], fixture["outputs"], strict=True):
                    dtype = "<f4" if spec["dtype"] == "float32" else "<i8"
                    expected = np.fromfile(bundle / spec["file"], dtype=dtype).reshape(spec["shape"])
                    actual = np.fromfile(destination / filename, dtype=dtype).reshape(spec["shape"])
                    passed = bool(np.array_equal(actual, expected) if spec["dtype"] == "int64" else
                                  np.allclose(actual, expected, atol=1e-4, rtol=1e-4))
                    if not passed:
                        raise ValueError(f"Native reference mismatch for {spec['name']}")
                    metrics.append(dict(name=spec["name"], max_abs_error=float(np.max(np.abs(actual - expected)))))
                row.update(passed=True, backend=info["backend"], fallback_reason=info["fallback_reason"], outputs=metrics)
            except Exception as error:
                row["error"] = str(error)
            rows.append(row)
            print(f"Native {bundle.name} seed {fixture['seed']}: {'passed' if row['passed'] else row['error']}", flush=True)
    report = dict(execution_target="host_cpu", establishes_hardware_qualification=False,
                  runner=str(runner), passed=bool(rows) and all(row["passed"] for row in rows), fixtures=rows)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("Native model validation failed; inspect " + str(output / "validation.json"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, default=ROOT / "artifacts/model-bundles")
    parser.add_argument("--runner", type=Path, default=ROOT / "artifacts/build/host-library/xrgpu_run")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native-model-validation")
    args = parser.parse_args()
    report = validate(args.bundles, args.runner, args.output)
    print(f"Native model fixtures passed: {len(report['fixtures'])}")


if __name__ == "__main__":
    main()
