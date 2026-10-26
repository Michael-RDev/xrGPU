"""Independent NumPy checks for native stable GPU sorted Unique."""
# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np


def cases():
    rng = np.random.default_rng(20261026)
    yield "large_unsorted_int32", rng.integers(-2048, 2048, size=32768, dtype=np.int32), dict(sorted=0), False
    yield "large_distinct_first_seen", rng.permutation(8192).astype(np.int32), dict(sorted=0), False
    yield "large_int32_flat", rng.integers(-2048, 2048, size=32768, dtype=np.int32), {}, True
    floats = rng.integers(-30, 30, size=2049).astype(np.float32) / 7
    floats[[0, 9, 300]] = np.nan
    floats[1:3] = [-0.0, 0.0]
    yield "float_nan_and_signed_zero", floats, {}, True
    yield "half_flat", (rng.integers(-100, 100, size=1025) / 8).astype(np.float16), {}, True
    yield "uint32_high_bits", rng.integers(0, 2**32, size=1025, dtype=np.uint32), {}, True
    yield "uint8_flat", rng.integers(0, 256, size=1025, dtype=np.uint8), {}, True
    yield "bool_flat", (rng.integers(0, 2, size=1025) > 0), {}, True
    rows = rng.integers(-5, 5, size=(19, 4), dtype=np.int16)
    yield "axis0_rows", rows[rng.integers(0, len(rows), size=1025)], dict(axis=0), True
    slices = rng.integers(-3, 3, size=(2, 13, 3), dtype=np.int8)
    yield "axis1_slices", slices[:, rng.integers(0, 13, size=1025), :], dict(axis=1), True
    yield "empty_axis_slices", np.empty((1025, 0), np.float32), dict(axis=0), True
    yield "small_unsorted", np.array([4, 1, 4, 2, 1, 7, 2], np.int32), dict(sorted=0), False
    yield "empty_flat", np.empty((0,), np.float32), {}, True


def verify(package, runner, output):
    records = []
    with tempfile.TemporaryDirectory(prefix="xrgpu-unique-") as temporary:
        root = Path(temporary)
        for name, array, attrs, sorted_values in cases():
            unique, indices, inverse, counts = np.unique(array, axis=attrs.get("axis"), return_index=True, return_inverse=True, return_counts=True)
            inverse = inverse.reshape(-1)
            if not sorted_values:
                order = np.argsort(indices)
                remap = np.empty(len(order), dtype=np.int64)
                remap[order] = np.arange(len(order))
                unique, indices, inverse, counts = unique[order], indices[order], remap[inverse], counts[order]
            expected = dict(y=unique, indices=indices, inverse_indices=inverse, counts=counts)
            case = root / name
            case.mkdir()
            data = case / "input.bin"
            data.write_bytes(array.tobytes())
            outputs = [dict(name=key, dtype=str(value.dtype) if key == "y" else "int64", shape=list(value.shape),
                **({} if key == "y" else dict(storage_dtype="uint32"))) for key, value in expected.items()]
            job = case / "job.json"
            job.write_text(json.dumps(dict(package=str(package), operation="ai.onnx.Unique", inputs=[dict(name="x", dtype=str(array.dtype), shape=list(array.shape), file=str(data))], attrs=attrs, outputs=outputs, output_directory=str(case / "actual"), repeat=2)))
            row = dict(name=name, input_shape=list(array.shape), dtype=str(array.dtype))
            start = time.monotonic()
            try:
                result = subprocess.run([str(runner), str(job)], capture_output=True, text=True, timeout=180)
                if result.returncode:
                    raise AssertionError(result.stderr)
                answer = json.loads(result.stdout)
                info = answer["info"]
                if info.get("backend") != "software_vulkan" or info.get("test_only") is not True or info.get("physical_qualification") is not False:
                    raise AssertionError("Explicit software-only evidence labels required")
                for descriptor in answer["outputs"]:
                    actual = np.fromfile(descriptor["file"], dtype=np.dtype(descriptor["storage_dtype"])).reshape(descriptor["shape"])
                    np.testing.assert_array_equal(actual, expected[descriptor["name"]])
                row.update(status="passed", info=info)
            except Exception as error:
                row.update(status="failed", error=str(error))
            row["invocation_seconds"] = time.monotonic() - start
            records.append(row)
            print(name + ": " + row["status"], flush=True)
    report = dict(kind="xrgpu.independent-unique-conformance", package=str(package), runner=str(runner),
        physical_qualification=False, oracle_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results=records, passed=sum(r["status"] == "passed" for r in records), failed=sum(r["status"] == "failed" for r in records))
    metadata = Path(package) / "development.json"
    if metadata.exists():
        report["development"] = json.loads(metadata.read_text())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.package, args.runner, args.output)
    raise SystemExit(bool(result["failed"]))


if __name__ == "__main__":
    main()
