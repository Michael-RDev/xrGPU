"""Independent grouped/packed CSR sparse attention NumPy cases."""
# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def reference(inputs, attrs):
    hq, hk = attrs["num_heads"], attrs["kv_num_heads"]
    raw = inputs["queryT"]
    batch, seq, hidden = raw.shape
    width = hidden // (hq if "keyT" in inputs else hq + 2 * hk)
    if "keyT" in inputs:
        q, k, v = [np.asarray(inputs[key], np.float64).reshape(batch, seq, heads, width).copy() for key, heads in [("queryT", hq), ("keyT", hk), ("valueT", hk)]]
    else:
        q = raw[:, :, :hq * width].astype(np.float64).reshape(batch, seq, hq, width)
        k = raw[:, :, hq * width:(hq + hk) * width].astype(np.float64).reshape(batch, seq, hk, width)
        v = raw[:, :, (hq + hk) * width:].astype(np.float64).reshape(batch, seq, hk, width)
    totals = inputs["keyTotalSequenceLengthsT"].astype(np.int64)
    past = np.zeros(batch, np.int64) if int(inputs["totalSequenceLengthT"][0]) == seq else np.maximum(totals - seq, 0)
    if attrs.get("do_rotary", 0):
        rotary = min(width, 2 * inputs["cosCacheT"].shape[1])
        for array in (q, k):
            old = array.copy()
            for b in range(batch):
                for t in range(seq):
                    position = past[b] + t
                    for d in range(rotary):
                        pair = d // 2 if attrs.get("rotary_interleaved", 0) else d % (rotary // 2)
                        mate = d ^ 1 if attrs.get("rotary_interleaved", 0) else (d + rotary // 2) % rotary
                        negative = d % 2 == 0 if attrs.get("rotary_interleaved", 0) else d < rotary // 2
                        array[b, t, :, d] = old[b, t, :, d] * float(inputs["cosCacheT"][position, pair]) + (-1 if negative else 1) * old[b, t, :, mate] * float(inputs["sinCacheT"][position, pair])
    q = q.astype(raw.dtype).astype(np.float64)
    k = k.astype(inputs["pastKeyT"].dtype)
    v = v.astype(inputs["pastValueT"].dtype)
    present_k = inputs["pastKeyT"].copy()
    present_v = inputs["pastValueT"].copy()
    for b in range(batch):
        count = min(seq, int(totals[b] - past[b]))
        present_k[b, :, past[b]:past[b] + count] = k[b, :count].transpose(1, 0, 2)
        present_v[b, :, past[b]:past[b] + count] = v[b, :count].transpose(1, 0, 2)
    result = np.zeros((batch, seq, hq, width), np.float64)
    pointers, columns = inputs["blockRowIndicesT"], inputs["blockColIndicesT"]
    block = attrs["sparse_block_size"]
    for b in range(batch):
        for head in range(hq):
            kvhead, layout = head // (hq // hk), head % pointers.shape[0]
            for qi in range(seq):
                csrrow = int(past[b] + qi) // block
                allowed = set(map(int, columns[layout, pointers[layout, csrrow]:pointers[layout, csrrow + 1]]))
                keys = [t for t in range(int(totals[b])) if t <= past[b] + qi and t // block in allowed]
                if not keys:
                    continue
                scores = present_k[b, kvhead, keys].astype(np.float64) @ q[b, qi, head] * attrs.get("scale", 1 / np.sqrt(width))
                weights = np.exp(scores - scores.max())
                weights /= weights.sum()
                result[b, qi, head] = weights @ present_v[b, kvhead, keys].astype(np.float64)
    return dict(outputT=result.reshape(batch, seq, hq * width).astype(raw.dtype), pastKeyT=present_k, pastValueT=present_v)


def cases():
    rng = np.random.default_rng(20261026)
    for name, packed, prompt, rotary, interleaved, dtype in [
        ("grouped_ragged_prompt", False, True, False, False, np.float32),
        ("packed_grouped_decode", True, False, False, False, np.float32),
        ("half_partial_rotary", True, False, True, False, np.float16),
        ("interleaved_partial_rotary", False, False, True, True, np.float32),
    ]:
        batch, hq, hk, width, capacity = 2, 4, 2, 8, 40
        seq = 17 if prompt else 3
        q = rng.normal(size=(batch, seq, hq * width)).astype(dtype)
        k = rng.normal(size=(batch, seq, hk * width)).astype(dtype)
        v = rng.normal(size=(batch, seq, hk * width)).astype(dtype)
        totals = np.array([16, 17] if prompt else [21, 22], np.int32)
        inputs = dict(queryT=np.concatenate([q, k, v], axis=-1) if packed else q,
            pastKeyT=rng.normal(size=(batch, hk, capacity, width)).astype(dtype),
            pastValueT=rng.normal(size=(batch, hk, capacity, width)).astype(dtype),
            blockRowIndicesT=np.array([[0, 1, 3, 6], [0, 1, 2, 3]], np.int32),
            blockColIndicesT=np.array([[0, 0, 1, 0, 1, 2], [0, 1, 2, 0, 0, 0]], np.int32),
            totalSequenceLengthT=np.array([seq if prompt else int(totals.max())], np.int32),
            keyTotalSequenceLengthsT=totals)
        if not packed:
            inputs.update(keyT=k, valueT=v)
        attrs = dict(num_heads=hq, kv_num_heads=hk, sparse_block_size=16, scale=0.375)
        if rotary:
            attrs.update(do_rotary=1, rotary_interleaved=int(interleaved))
            angles = np.arange(capacity * 2).reshape(capacity, 2) / 19
            inputs.update(cosCacheT=np.cos(angles).astype(dtype), sinCacheT=np.sin(angles).astype(dtype))
        yield name, inputs, attrs, reference(inputs, attrs)


def verify(package, runner, output):
    records = []
    with tempfile.TemporaryDirectory(prefix="xrgpu-sparse-") as work:
        for name, arrays, attrs, expected in cases():
            root = Path(work) / name
            root.mkdir()
            inputs = []
            for key, array in arrays.items():
                file = root / (key + ".bin")
                file.write_bytes(array.tobytes())
                inputs.append(dict(name=key, dtype=str(array.dtype), shape=list(array.shape), file=str(file)))
            outputs = [dict(name=key, dtype=str(array.dtype), shape=list(array.shape)) for key, array in expected.items()]
            job = root / "job.json"
            job.write_text(json.dumps(dict(package=str(package), operation="com.microsoft.SparseAttention", inputs=inputs, attrs=attrs, outputs=outputs, output_directory=str(root / "actual"), repeat=2)))
            row = dict(name=name)
            try:
                process = subprocess.run([str(runner), str(job)], capture_output=True, text=True, timeout=180)
                if process.returncode:
                    raise AssertionError(process.stderr)
                answer = json.loads(process.stdout)
                info = answer["info"]
                if info.get("backend") != "software_vulkan" or info.get("test_only") is not True or info.get("physical_qualification") is not False:
                    raise AssertionError("Software-only evidence labels required")
                for result in answer["outputs"]:
                    actual = np.fromfile(result["file"], dtype=np.dtype(result["storage_dtype"])).reshape(result["shape"])
                    half = actual.dtype == np.float16
                    np.testing.assert_allclose(actual, expected[result["name"]], atol=1e-3 if half else 1e-5, rtol=1e-3 if half else 1e-5)
                row.update(status="passed", info=info)
            except Exception as error:
                row.update(status="failed", error=str(error))
            records.append(row)
            print(name + ": " + row["status"], flush=True)
    report = dict(kind="xrgpu.independent-sparse-conformance", package=str(package), runner=str(runner),
        physical_qualification=False, oracle_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        results=records, passed=sum(r["status"] == "passed" for r in records), failed=sum(r["status"] == "failed" for r in records))
    metadata = Path(package) / "development.json"
    if metadata.exists():
        report["development"] = json.loads(metadata.read_text())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.package, args.runner, args.output)
    raise SystemExit(bool(result["failed"]))


if __name__ == "__main__":
    main()
