"""Exercise generic native storage and contract errors on software Vulkan."""
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def verify(package, runner, output):
    package, runner, output = Path(package).resolve(), Path(runner).resolve(), Path(output).resolve()
    rows = []
    with tempfile.TemporaryDirectory(prefix="xrgpu-extended-core-") as directory:
        root = Path(directory)

        def invoke(name, operation, arrays, expected=None, attrs=None, args=None, outputs=None, error=None):
            case = root / name
            case.mkdir()
            inputs = []
            for index, (slot, array, logical) in enumerate(arrays):
                array = np.asarray(array)
                file = case / ("input" + str(index) + ".bin")
                file.write_bytes(array.tobytes())
                inputs.append(dict(name=slot, shape=list(array.shape), dtype=logical or str(array.dtype),
                    storage_dtype=str(array.dtype), file=str(file)))
            job = dict(package=str(package), operation=operation, inputs=inputs, attrs=attrs or {},
                args=args or {}, output_directory=str(case / "outputs"), repeat=2)
            if outputs is not None:
                job["outputs"] = outputs
            path = case / "job.json"
            path.write_text(json.dumps(job))
            result = subprocess.run([str(runner), str(path)], text=True, capture_output=True, timeout=180)
            row = dict(name=name, operation=operation)
            try:
                if error:
                    if result.returncode == 0:
                        raise AssertionError("Expected failure, received successful native output: " + result.stdout)
                    answer = json.loads(result.stderr.strip().splitlines()[-1])
                    if result.returncode == 0 or error not in answer.get("error", ""):
                        raise AssertionError("Expected explicit contract error: " + error + "; " + str(answer))
                    row.update(status="pass", rejected_error=answer["error"])
                else:
                    if result.returncode:
                        raise AssertionError(result.stderr)
                    answer = json.loads(result.stdout)
                    info = answer["info"]
                    if info.get("backend") != "software_vulkan" or info.get("test_only") is not True or info.get("physical_qualification") is not False:
                        raise AssertionError("Software-only evidence labels required")
                    if len(answer["outputs"]) != len(expected):
                        raise AssertionError("Output count differs")
                    for actual, reference in zip(answer["outputs"], expected):
                        array = np.fromfile(actual["file"], dtype=np.dtype(actual["storage_dtype"])).reshape(actual["shape"])
                        np.testing.assert_array_equal(array, reference)
                    row.update(status="pass", info=info)
            except Exception as exc:
                row.update(status="fail", error=str(exc))
            rows.append(row)
            print(name + ": " + row["status"], flush=True)

        i64 = np.array([-(2**63), -1, 2**40 + 1, 2**63 - 1], dtype=np.int64)
        invoke("int64_raw_identity", "ai.onnx.Identity", [("x", i64, None)], [i64])
        narrow = np.array([0, 1, 2**32 - 1], dtype=np.uint32)
        invoke("logical_int64_uint32_storage", "ai.onnx.Identity", [("x", narrow, "int64")], [narrow])
        indices = np.array([3, 1, 0], dtype=np.int32)
        invoke("int64_raw_gather", "ai.onnx.Gather", [("data", i64, None), ("indices", indices, None)], [i64[indices]])
        values = np.array([0, 2**31, 7, 2**32 - 1, 2**31 - 1], dtype=np.uint32)
        order = np.array(sorted(range(len(values)), key=lambda i: (-int(values[i]), i))[:3], dtype=np.uint32)
        selections = [dict(name="values", dtype="uint32", shape=[3]),
            dict(name="indices", dtype="int64", storage_dtype="uint32", shape=[3])]
        invoke("uint32_topk_int64_indices", "ai.onnx.TopK", [("x", values, None)],
            [values[order], order], args=dict(k=3), outputs=selections)
        signed = np.array([-128, -1, 0, 127], dtype=np.int8)
        invoke("signed_int8_cast_float", "ai.onnx.Cast", [("x", signed, None)],
            [signed.astype(np.float32)], attrs=dict(to=1))
        invoke("published_invalid_gather_zero_fill", "ai.onnx.Gather", [("data", i64, None),
            ("indices", np.array([4], np.int32), None)], [np.zeros(1, np.int64)])
        unique_input = np.array([3, 1, 3, 2, 1], np.int32)
        capacity_outputs = [dict(name="y", dtype="int32", shape=[5])] + [dict(name=name, dtype="int64", storage_dtype="uint32", shape=[5]) for name in ["indices", "inverse_indices", "counts"]]
        invoke("unique_optional_capacity_bounds", "ai.onnx.Unique", [("x", unique_input, None)],
            [np.array([1, 2, 3], np.int32), np.array([1, 3, 0], np.uint32), np.array([2, 0, 2, 1, 0], np.uint32), np.array([2, 1, 2], np.uint32)], outputs=capacity_outputs)
        too_small = [dict(descriptor) for descriptor in capacity_outputs]
        too_small[0]["shape"] = [2]
        invoke("unique_reject_insufficient_capacity", "ai.onnx.Unique", [("x", unique_input, None)],
            outputs=too_small, error="GPU dynamic output exceeds requested capacity")
        invoke("invalid_gpu_scatter_index", "ai.onnx.ScatterElements", [("data", i64, None),
            ("indices", np.array([4], np.int32), None), ("updates", np.array([9], np.int64), None)],
            attrs=dict(axis=0), error="GPU operator contract validation failed")
        invoke("invalid_index_logical_type", "ai.onnx.Gather", [("data", i64, None),
            ("indices", np.array([0], np.int64), None)], error="Logical dtype outside frozen contract")
        invoke("unknown_named_input", "ai.onnx.Identity", [("x", i64, None),
            ("unexpected", np.array([1], np.int32), None)], error="Unknown input outside frozen contract")
        invoke("reject_unsupported_cast_contract", "ai.onnx.Cast", [("x", i64, None)],
            attrs=dict(to=1), error="Logical dtype outside frozen contract")
    report = dict(schema_version=1, kind="xrgpu.extended-core-conformance", package=str(package),
        runner=str(runner), backend="software_vulkan", physical_qualification=False,
        passed=sum(row["status"] == "pass" for row in rows), failed=sum(row["status"] == "fail" for row in rows), cases=rows)
    provenance = package / "development.json"
    if provenance.exists():
        report["development"] = json.loads(provenance.read_text())
    manifest = package / "manifest.json"
    if manifest.exists():
        report["package_sha256"] = json.loads(manifest.read_text())["package_sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.package, args.runner, args.output)
    raise SystemExit(1 if result["failed"] else 0)


if __name__ == "__main__":
    main()
