"""Independent NumPy GQA cache, mask, norm and rotary conformance cases."""
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def scale_channels(scales, mode, head, width):
    if mode == "PER_TENSOR":
        return np.full(width, np.asarray(scales).reshape(-1)[0], np.float64)
    return np.asarray(scales).reshape(-1)[head * width : (head + 1) * width].astype(np.float64)


def encode(values, dtype, bits, scales, mode, head):
    if mode == "NONE":
        return np.asarray(values, dtype=dtype)
    quantized = np.clip(np.rint(np.asarray(values, np.float64) / scale_channels(scales, mode, head, len(values))), -8 if bits == 4 else -128, 7 if bits == 4 else 127).astype(np.int16)
    if bits == 4:
        biased = quantized + 8
        result = np.zeros((len(values) + 1) // 2, np.uint8)
        result[:] = biased[::2]
        result[:len(biased[1::2])] |= biased[1::2].astype(np.uint8) << 4
        return result.astype(dtype)
    return quantized.astype(np.int8).view(np.uint8).astype(dtype) if np.dtype(dtype) == np.uint8 else quantized.astype(dtype)


def decode(values, width, bits, scales, mode, head):
    if mode == "NONE":
        return np.asarray(values, np.float64)
    raw = np.asarray(values).astype(np.uint8)
    if bits == 4:
        quantized = np.empty(width, np.int16)
        quantized[::2] = (raw & 15).astype(np.int16) - 8
        quantized[1::2] = (raw[:width // 2] >> 4).astype(np.int16) - 8
    else:
        quantized = raw.view(np.int8).astype(np.int16)
    return quantized * scale_channels(scales, mode, head, width)


def reference(inputs, attrs, cache_dtype, capacity):
    q, k, v = [np.asarray(inputs[name], np.float64) for name in ("queryT", "keyT", "valueT")]
    batch, queries, _ = q.shape
    heads, kvheads = attrs["num_heads"], attrs["kv_num_heads"]
    width = q.shape[-1] // heads
    value_width = v.shape[-1] // kvheads
    q = q.reshape(batch, queries, heads, width).copy()
    k = k.reshape(batch, k.shape[1], kvheads, width).copy()
    v = v.reshape(batch, v.shape[1], kvheads, value_width).copy()
    total = np.asarray(inputs["seqlensKT"]).astype(np.int64) + 1
    past = np.maximum(total - k.shape[1], 0)
    for role, array, norm_name in [(0, q, "qNormWeightT"), (1, k, "kNormWeightT")]:
        if norm_name in inputs:
            array *= 1 / np.sqrt(np.mean(array * array, axis=-1, keepdims=True) + attrs.get("qk_norm_epsilon", 1e-6))
            array *= np.asarray(inputs[norm_name], np.float64)
        if attrs.get("do_rotary", 0):
            old = array.copy()
            rotary = min(width, inputs["cosCacheT"].shape[1] * 2)
            for b in range(batch):
                for token in range(array.shape[1]):
                    position = int(past[b]) + token
                    for d in range(rotary):
                        pair = d // 2 if attrs.get("rotary_interleaved", 0) else d % (rotary // 2)
                        mate = d ^ 1 if attrs.get("rotary_interleaved", 0) else (d + rotary // 2) % rotary
                        negative = d % 2 == 0 if attrs.get("rotary_interleaved", 0) else d < rotary // 2
                        array[b, token, :, d] = old[b, token, :, d] * float(inputs["cosCacheT"][position, pair]) + (-1 if negative else 1) * old[b, token, :, mate] * float(inputs["sinCacheT"][position, pair])
    transformed = [k, v]
    caches = []
    attended = []
    for role, current in enumerate(transformed):
        mode = attrs.get("k_quant_type" if role == 0 else "v_quant_type", "NONE")
        bits = attrs.get("kv_cache_bit_width", 8)
        scale = inputs.get("kScaleT" if role == 0 else "vScaleT")
        physical_width = (current.shape[-1] + 1) // 2 if mode != "NONE" and bits == 4 else current.shape[-1]
        cache = np.zeros((batch, kvheads, capacity, physical_width), cache_dtype)
        previous = inputs.get("pastKeyT" if role == 0 else "pastValueT")
        old_capacity = previous.shape[2] if previous is not None else 0
        old_first = np.maximum(past - old_capacity, 0) if attrs.get("sliding_window_cache", 0) else np.zeros(batch, np.int64)
        new_first = np.maximum(total - capacity, 0) if attrs.get("sliding_window_cache", 0) else np.zeros(batch, np.int64)
        current_values = current.copy()
        for b in range(batch):
            for head in range(kvheads):
                for row in range(capacity):
                    old_row = int(new_first[b] + row - old_first[b])
                    if previous is not None and 0 <= old_row < old_capacity:
                        cache[b, head, row] = previous[b, head, old_row]
                for token in range(current.shape[1]):
                    encoded = encode(current[b, token, head], cache_dtype, bits, scale, mode, head)
                    if mode != "NONE":
                        current_values[b, token, head] = decode(encoded, current.shape[-1], bits, scale, mode, head)
                    row = int(past[b] + token - new_first[b])
                    if 0 <= row < capacity and past[b] + token < total[b]:
                        cache[b, head, row] = encoded
        caches.append(cache)
        attended.append((current_values, previous, old_first, mode, bits, scale))
    answer = np.zeros((batch, queries, heads, value_width), np.float64)
    for b in range(batch):
        for head in range(heads):
            kh = head // (heads // kvheads)
            for qi in range(queries):
                first = int(attended[0][2][b])
                if attrs.get("local_window_size", -1) >= 0:
                    first = max(first, int(past[b]) + qi - attrs["local_window_size"])
                keys = list(range(first, int(total[b])))
                scores, vectors = [], []
                for token in keys:
                    values = []
                    for current, previous, old_first, mode, bits, scales in attended:
                        values.append(current[b, token - past[b], kh] if token >= past[b] else decode(previous[b, kh, token - old_first[b]], current.shape[-1], bits, scales, mode, kh))
                    score = np.dot(q[b, qi, head], values[0]) * attrs.get("scale", 1 / np.sqrt(width))
                    if attrs.get("softcap", 0) > 0:
                        score = attrs["softcap"] * np.tanh(score / attrs["softcap"])
                    if attrs.get("causal", 1) and token > past[b] + qi:
                        score = -np.inf
                    elif "attentionBiasT" in inputs:
                        bias = inputs["attentionBiasT"]
                        score += float(bias[b if bias.shape[0] > 1 else 0, head if bias.shape[1] > 1 else 0, qi if bias.shape[2] > 1 else 0, token if bias.shape[3] > 1 else 0])
                    scores.append(score)
                    vectors.append(values[1])
                extras = []
                if "headSinkT" in inputs:
                    extras.append(float(inputs["headSinkT"][head]))
                if attrs.get("smooth_softmax", -1) == 1:
                    extras.append(0.0)
                maximum = max(scores + extras, default=-np.inf)
                if np.isfinite(maximum):
                    weights = np.exp(np.asarray(scores) - maximum)
                    denominator = weights.sum() + np.exp(np.asarray(extras) - maximum).sum()
                    answer[b, qi, head] = np.sum(weights[:, None] * np.asarray(vectors), axis=0) / denominator
    return {"outputT": answer.reshape(batch, queries, heads * value_width).astype(inputs["queryT"].dtype), "presentKeyT": caches[0], "presentValueT": caches[1]}


def cases():
    rng = np.random.default_rng(20261026)
    result = []
    for name, mode, bits, sliding, rotary, half in [
        ("unquantized_half_cache", "NONE", 8, False, False, True),
        ("int4_per_tensor", "PER_TENSOR", 4, False, False, False),
        ("int8_per_channel", "PER_CHANNEL", 8, False, False, False),
        ("int4_sliding_channel", "PER_CHANNEL", 4, True, False, False),
        ("rotary_norm_split", "NONE", 8, False, True, False),
        ("rotary_norm_interleaved", "NONE", 8, False, True, False),
    ]:
        batch, heads, kvheads, seq, width, capacity = 2, 4, 2, 2, 4, 4
        past = np.array([3, 4] if sliding else [1, 2], np.int32)
        attrs = dict(num_heads=heads, kv_num_heads=kvheads, causal=1)
        if mode != "NONE":
            attrs.update(k_quant_type=mode, v_quant_type=mode, kv_cache_bit_width=bits)
        if sliding:
            attrs.update(sliding_window_cache=1, local_window_size=2)
        cache_dtype = np.float16 if half else np.float32 if mode == "NONE" else np.uint8 if bits == 4 else np.int8
        inputs = {"queryT": rng.normal(size=(batch, seq, heads * width)).astype(np.float32),
            "keyT": rng.normal(size=(batch, seq, kvheads * width)).astype(np.float32),
            "valueT": rng.normal(size=(batch, seq, kvheads * width)).astype(np.float32),
            "seqlensKT": past + seq - 1, "totalSequenceLengthT": np.array([max(past) + seq], np.int32)}
        if mode != "NONE":
            count = 1 if mode == "PER_TENSOR" else kvheads * width
            inputs["kScaleT"] = rng.choice([0.25, 0.5, 1.0], size=count).astype(np.float32)
            inputs["vScaleT"] = rng.choice([0.25, 0.5, 1.0], size=count).astype(np.float32)
        physical_width = width // 2 if mode != "NONE" and bits == 4 else width
        for slot, scales in [("pastKeyT", "kScaleT"), ("pastValueT", "vScaleT")]:
            cache = np.empty((batch, kvheads, capacity, physical_width), cache_dtype)
            for b in range(batch):
                for h in range(kvheads):
                    for t in range(capacity):
                        cache[b, h, t] = encode(rng.normal(size=width), cache_dtype, bits, inputs.get(scales), mode, h)
            inputs[slot] = cache
        if rotary:
            attrs.update(do_rotary=1, rotary_interleaved=int(name.endswith("interleaved")), softcap=2.0, smooth_softmax=1)
            angles = np.arange(16 * width // 2).reshape(16, width // 2) / 13.0
            inputs.update(cosCacheT=np.cos(angles).astype(np.float32), sinCacheT=np.sin(angles).astype(np.float32),
                qNormWeightT=np.array([1.1, 0.7, 0.9, 1.3], np.float32), kNormWeightT=np.array([0.9, 1.2, 0.8, 1.1], np.float32),
                attentionBiasT=rng.normal(size=(1, heads, seq, 4)).astype(np.float32) / 7,
                headSinkT=np.array([-1.0, 0.4, -0.5, 1.2], np.float32))
        expected = reference(inputs, attrs, cache_dtype, capacity)
        result.append((name, inputs, attrs, expected))
    return result


def verify(package, runner, output):
    rows = []
    with tempfile.TemporaryDirectory(prefix="xrgpu-gqa-") as directory:
        root = Path(directory)
        for name, arrays, attrs, expected in cases():
            folder = root / name
            folder.mkdir()
            inputs = []
            for slot, array in arrays.items():
                file = folder / (slot + ".bin")
                file.write_bytes(array.tobytes())
                inputs.append(dict(name=slot, dtype=str(array.dtype), shape=list(array.shape), file=str(file)))
            outputs = [dict(name=slot, dtype=str(array.dtype), shape=list(array.shape)) for slot, array in expected.items()]
            job = folder / "job.json"
            job.write_text(json.dumps(dict(package=str(package), operation="com.microsoft.GroupQueryAttention", inputs=inputs, attrs=attrs, outputs=outputs, output_directory=str(folder / "actual"), repeat=2)))
            row = dict(name=name)
            try:
                result = subprocess.run([str(runner), str(job)], capture_output=True, text=True, timeout=180)
                if result.returncode:
                    raise AssertionError(result.stderr)
                answer = json.loads(result.stdout)
                info = answer["info"]
                if info.get("backend") != "software_vulkan" or info.get("physical_qualification") is not False or info.get("test_only") is not True:
                    raise AssertionError("Software-only evidence labels required")
                for descriptor in answer["outputs"]:
                    actual = np.fromfile(descriptor["file"], dtype=np.dtype(descriptor["storage_dtype"])).reshape(descriptor["shape"])
                    reference_array = expected[descriptor["name"]]
                    if np.issubdtype(actual.dtype, np.integer):
                        np.testing.assert_array_equal(actual, reference_array)
                    else:
                        np.testing.assert_allclose(actual, reference_array, atol=1e-5 if actual.dtype == np.float32 else 1e-3, rtol=1e-5 if actual.dtype == np.float32 else 1e-3)
                row.update(status="passed", info=info)
            except Exception as exc:
                row.update(status="failed", error=str(exc))
            rows.append(row)
            print(name + ": " + row["status"], flush=True)
    report = dict(kind="xrgpu.independent-gqa-conformance", physical_qualification=False, package=str(package), runner=str(runner), cases=rows,
        passed=sum(r["status"] == "passed" for r in rows), failed=sum(r["status"] == "failed" for r in rows))
    development = Path(package) / "development.json"
    if development.exists():
        report["development"] = json.loads(development.read_text())
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
