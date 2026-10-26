# SPDX-License-Identifier: Apache-2.0
"""Run independent numerical checks through test-only native software Vulkan.

This is shader execution evidence, never headset/GPU qualification. No browser
process or WebGPU API is used. Supply an explicitly built software-test runner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TYPES = ["float32", "float16", "int32", "uint32", "int16", "int8", "uint8", "int64", "bool"]


def operators():
    pattern = re.compile(r'XRGPU_OPERATOR\("([^"]+)",\s*"([^"]+)",\s*(\w+),\s*(\d+),\s*(\d+),\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)"\)')
    result = []
    for path in sorted((ROOT / "kernels/_shared").glob("registry*.def")):
        for match in pattern.finditer(path.read_text()):
            op, shader, kind, opcode, mask, first, second, output = match.groups()
            result.append(dict(op=op, shader=shader, kind=kind, mask=int(mask), inputs=[first] + ([second] if kind != "Unary" else [])))
    return result


def values(dtype, shape, *, divisor=False, positive=False):
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if dtype.startswith("float"):
        array = np.linspace(0.25, 0.95, max(count, 1), dtype=np.float64)
        if not positive and not divisor:
            array = np.linspace(-0.9, 0.9, max(count, 1), dtype=np.float64)
        return array[:count].astype(dtype).reshape(shape)
    if dtype == "bool":
        return (np.arange(count) % 3 == 0).reshape(shape)
    limits = np.iinfo(dtype)
    cycle = [limits.min, limits.max, 0, 1, 2, 3, 7]
    if limits.min < 0:
        cycle += [-1, -2, -3, -7]
    if divisor:
        cycle = [x for x in cycle if x]
    return np.asarray([cycle[i % len(cycle)] for i in range(count)], dtype=dtype).reshape(shape)


def reference(name, inputs):
    a = inputs[0]
    b = inputs[1] if len(inputs) > 1 else None
    dtype = a.dtype
    if name == "MatMul":
        return (np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(dtype)
                if dtype.kind == "f" else np.matmul(a, b))
    binary = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply,
              "BitwiseAnd": np.bitwise_and, "BitwiseOr": np.bitwise_or, "BitwiseXor": np.bitwise_xor}
    if name in binary:
        return binary[name](a, b).astype(dtype)
    if name == "Div":
        if dtype.kind == "f":
            return (a.astype(np.float64) / b.astype(np.float64)).astype(dtype)
        x, y = np.broadcast_arrays(a, b)
        bits = dtype.itemsize * 8
        def divide(left, right):
            left, right = int(left), int(right)
            if not right:
                return left
            magnitude = abs(left) // abs(right)
            value = -magnitude if (left < 0) != (right < 0) else magnitude
            value %= 1 << bits
            if dtype.kind == "i" and value >= 1 << (bits-1):
                value -= 1 << bits
            return value
        return np.asarray([divide(xi, yi) for xi, yi in zip(x.flat, y.flat)], dtype=dtype).reshape(x.shape)
    if name == "PRelu":
        return np.where(a < 0, a * b, a).astype(dtype)
    if name == "Relu":
        return np.where(a < 0, np.zeros((), dtype=dtype), a).astype(dtype)
    unary = {"Abs": np.abs, "Neg": np.negative, "Ceil": np.ceil, "Floor": np.floor,
             "Round": np.rint, "Sign": np.sign, "Sqrt": np.sqrt, "Exp": np.exp,
             "Log": np.log, "Sin": np.sin, "Cos": np.cos, "Tan": np.tan,
             "Asin": np.arcsin, "Acos": np.arccos, "Atan": np.arctan,
             "Sinh": np.sinh, "Cosh": np.cosh, "Tanh": np.tanh,
             "Asinh": np.arcsinh, "Acosh": np.arccosh, "Atanh": np.arctanh,
             "Reciprocal": np.reciprocal, "BitwiseNot": np.bitwise_not, "Not": np.logical_not,
             "Identity": lambda x: x, "Sigmoid": lambda x: 1 / (1 + np.exp(-x)),
             "Softplus": lambda x: np.logaddexp(0, x), "Softsign": lambda x: x / (1 + np.abs(x)),
             "Mish": lambda x: x * np.tanh(np.logaddexp(0, x)),
             "HardSwish": lambda x: x * np.clip((x + 3) / 6, 0, 1),
             "Erf": lambda x: np.vectorize(__import__("math").erf)(x)}
    if name not in unary:
        raise KeyError(name)
    working = a.astype(np.float64) if dtype.kind == "f" else a
    return unary[name](working).astype(dtype)


def check(package, runner, output, selected=None):
    from kernels._shared.build import verify
    manifest = verify(package)
    results = []
    for spec in operators():
        if selected and spec["op"] not in selected:
            continue
        name = spec["op"].split(".")[-1]
        for bit, dtype in enumerate(TYPES):
            if not spec["mask"] & (1 << bit):
                continue
            shapes = [(2, 1, 19), (1, 3, 19)] if spec["kind"] == "Binary" else [(67,)]
            if name == "PRelu":
                shapes = [(2, 3, 19), (1, 3, 1)]
            elif spec["kind"] == "MatMul":
                shapes = [(2, 1, 3, 5), (1, 2, 5, 7)]
            inputs = [values(dtype, shape, divisor=name == "Div" and i == 1,
                             positive=name in {"Sqrt", "Log", "Reciprocal"}) for i, shape in enumerate(shapes)]
            if name == "Acosh":
                inputs[0] += np.asarray(1.0, dtype=dtype)
            entry = dict(operator=spec["op"], dtype=dtype, status="failed")
            try:
                with np.errstate(all="ignore"):
                    expected = reference(name, inputs)
                with tempfile.TemporaryDirectory(prefix="xrgpu-check-") as temporary:
                    root = Path(temporary)
                    command = [str(runner), str(package), spec["op"], dtype, str(root / "actual")]
                    for index, (input_name, array) in enumerate(zip(spec["inputs"], inputs)):
                        path = root / f"input{index}.bin"
                        path.write_bytes(array.tobytes())
                        shape = ",".join(map(str, array.shape)) if array.shape else "scalar"
                        command += [input_name, shape, str(path)]
                    process = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    if process.returncode:
                        raise RuntimeError(process.stderr.strip())
                    evidence = json.loads(process.stdout)
                    if evidence.get("backend") != "software_vulkan" or evidence.get("physical_qualification") is not False or evidence.get("test_only") is not True:
                        raise RuntimeError("This harness requires an explicitly labelled software Vulkan test build")
                    descriptors = evidence.get("outputs", [])
                    if len(descriptors) != 1 or descriptors[0].get("dtype") != str(expected.dtype) or descriptors[0].get("shape") != list(expected.shape):
                        raise RuntimeError("Native output descriptor differs from independent inference")
                    actual = np.frombuffer((root / "actual.0.bin").read_bytes(), dtype=dtype).reshape(expected.shape)
                    if expected.dtype.kind == "f":
                        tolerance = 0.007 if dtype == "float16" else 2e-5
                        matched = np.allclose(actual, expected, atol=tolerance, rtol=tolerance, equal_nan=True)
                        with np.errstate(invalid="ignore"):
                            differences = np.abs(actual.astype(float)-expected.astype(float))
                        finite = differences[np.isfinite(differences)]
                        entry["max_absolute_error"] = float(finite.max()) if finite.size else 0.0
                        entry["atol"] = entry["rtol"] = tolerance
                    else:
                        matched = np.array_equal(actual, expected)
                    if not matched:
                        raise ValueError("Native shader result differs from independent NumPy oracle")
                    entry.update(status="passed", evidence=evidence)
            except Exception as error:
                entry["error"] = str(error)
            results.append(entry)
    report = dict(kind="xrgpu.software-vulkan-conformance", schema_version=1,
                  physical_qualification=False, package_sha256=manifest["package_sha256"],
                  runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
                  numpy_version=np.__version__, results=results,
                  passed=sum(row["status"] == "passed" for row in results),
                  failed=sum(row["status"] != "passed" for row in results))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator", action="append")
    args = parser.parse_args()
    result = check(args.package.resolve(), args.runner.resolve(), args.output, args.operator)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    return bool(result["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
