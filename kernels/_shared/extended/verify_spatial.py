# SPDX-License-Identifier: Apache-2.0
"""Independent local numerical cases for every spatial/reduction/signal kernel.

The runner must explicitly report software_vulkan; these results cannot qualify
physical GPU support. All expected arrays are computed independently in NumPy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tempfile

import numpy as np


def tensor(name, value, dtype=None):
    value = np.asarray(value, dtype=dtype)
    return dict(
        name=name,
        dtype=str(value.dtype),
        shape=list(value.shape),
        values=value.reshape(-1).tolist(),
    )


def cases():
    result = []

    def add(name, inputs, expected, attrs=None, args=None):
        result.append(
            dict(
                operation=name if "." in name else "ai.onnx." + name,
                inputs=[tensor(n, a) for n, a in inputs],
                attrs=attrs or {},
                args=args or {},
                expected={n: np.asarray(a) for n, a in expected},
            )
        )

    x = np.array([[-2.0, 1.0, 3.0], [4.0, -1.0, 2.0]], dtype=np.float32)
    reductions = {
        "ReduceSum": lambda x: np.sum(x, axis=1, keepdims=True),
        "ReduceProd": lambda x: np.prod(x, axis=1, keepdims=True),
        "ReduceMean": lambda x: np.mean(x, axis=1, keepdims=True),
        "ReduceMin": lambda x: np.min(x, axis=1, keepdims=True),
        "ReduceMax": lambda x: np.max(x, axis=1, keepdims=True),
        "ReduceL1": lambda x: np.sum(abs(x), axis=1, keepdims=True),
        "ReduceL2": lambda x: np.sqrt(np.sum(x * x, axis=1, keepdims=True)),
        "ReduceLogSum": lambda x: np.log(np.sum(x, axis=1, keepdims=True)),
        "ReduceLogSumExp": lambda x: np.log(
            np.sum(np.exp(x.astype(np.float64)), axis=1, keepdims=True)
        ).astype(np.float32),
        "ReduceSumSquare": lambda x: np.sum(x * x, axis=1, keepdims=True),
    }
    for name, reference in reductions.items():
        add(name, [("x", x)], [("y", reference(x))], dict(axes=[1]))
    add("CumSum", [("x", x)], [("y", np.cumsum(x, axis=1))], args=dict(axis=1))
    add("CumProd", [("x", x)], [("y", np.cumprod(x, axis=1))], args=dict(axis=1))
    soft = np.exp(x - x.max(axis=1, keepdims=True))
    soft /= soft.sum(axis=1, keepdims=True)
    add("Softmax", [("x", x)], [("y", soft)], dict(axis=1))
    add("LogSoftmax", [("x", x)], [("y", np.log(soft))], dict(axis=1))
    add(
        "ArgMax",
        [("x", x)],
        [("y", np.argmax(x, axis=1, keepdims=True).astype(np.int64))],
        dict(axis=1),
    )
    add(
        "ArgMin",
        [("x", x)],
        [("y", np.argmin(x, axis=1, keepdims=True).astype(np.int64))],
        dict(axis=1),
    )
    hard = np.zeros_like(x)
    np.put_along_axis(hard, np.argmax(x, axis=1, keepdims=True), 1, axis=1)
    add("Hardmax", [("x", x)], [("y", hard)], dict(axis=1))
    bias = np.array([0.5, -0.25, 0.75], np.float32)
    bx = x + bias
    bs = np.exp(bx - bx.max(axis=1, keepdims=True))
    bs /= bs.sum(axis=1, keepdims=True)
    add(
        "com.microsoft.BiasSoftmax",
        [("data", x), ("bias", bias)],
        [("output", bs)],
        dict(axis=1, is_inner_broadcast=1),
    )
    nx = np.array([[[1.0, 2.0], [3.0, 5.0]]], np.float32)
    scale = np.array([1.5, 0.5], np.float32)
    offset = np.array([0.25, -0.5], np.float32)
    epsilon = 1e-5
    mean = np.array([1.0, 3.0], np.float32)
    variance = np.array([2.0, 4.0], np.float32)
    add(
        "BatchNormalization",
        [("x", nx), ("scale", scale), ("b", offset), ("inputMean", mean), ("inputVar", variance)],
        [
            (
                "y",
                (nx - mean[None, :, None])
                * scale[None, :, None]
                / np.sqrt(variance[None, :, None] + epsilon)
                + offset[None, :, None],
            )
        ],
    )
    im = nx.mean(axis=2, keepdims=True)
    iv = nx.var(axis=2, keepdims=True)
    add(
        "InstanceNormalization",
        [("input", nx), ("scale", scale), ("b", offset)],
        [
            (
                "output",
                (nx - im) / np.sqrt(iv + epsilon) * scale[None, :, None] + offset[None, :, None],
            )
        ],
    )
    gm = nx.mean(axis=(1, 2), keepdims=True)
    gv = nx.var(axis=(1, 2), keepdims=True)
    gs = np.array([1.25, 1.25], np.float32)
    gb = np.array([-0.25, -0.25], np.float32)
    add(
        "GroupNormalization",
        [("x", nx), ("scale", gs), ("bias", gb)],
        [("y", (nx - gm) / np.sqrt(gv + epsilon) * gs[0] + gb[0])],
        dict(num_groups=1),
    )
    sx = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4) / 7
    gamma = np.array([1.0, 0.5, 1.5, 0.75], np.float32)
    beta = np.array([0.1, -0.2, 0.3, -0.4], np.float32)

    def group(a):
        return (a - a.mean(axis=(1, 2, 3), keepdims=True)) / np.sqrt(
            a.var(axis=(1, 2, 3), keepdims=True) + epsilon
        ) * gamma + beta

    add(
        "com.microsoft.GroupNorm",
        [("X", sx), ("gamma", gamma), ("beta", beta)],
        [("Y", group(sx))],
        dict(groups=1, activation=0, channels_last=1),
    )
    skip = np.full_like(sx, 0.25)
    cb = np.array([0.5, 0.1, -0.2, -0.5], np.float32)
    res = sx + skip + cb
    add(
        "com.microsoft.SkipGroupNorm",
        [("X", sx), ("gamma", gamma), ("beta", beta), ("skip", skip), ("bias", cb)],
        [("Y", group(res)), ("S", res)],
        dict(groups=1, activation=0, channels_last=1),
    )
    sc = np.array([1.0, 0.5, 1.5], np.float32)
    bb = np.array([0.1, -0.2, 0.3], np.float32)
    m = x.mean(axis=1, keepdims=True)
    inv = (1 / np.sqrt(x.var(axis=1, keepdims=True) + epsilon)).astype(np.float32)
    add(
        "LayerNormalization",
        [("x", x), ("scale", sc), ("b", bb)],
        [("y", (x - m) * inv * sc + bb), ("mean", m), ("invStdDev", inv)],
    )
    rms = (1 / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + epsilon)).astype(np.float32)
    add("RMSNormalization", [("x", x), ("scale", sc)], [("y", x * rms * sc)])
    add(
        "SimplifiedLayerNormalization",
        [("x", x), ("scale", sc)],
        [("y", x * rms * sc), ("invStdVar", rms)],
    )
    ss = np.array([[0.5, -0.5, 0.5], [1.0, -0.25, 0.75]], np.float32)
    small = np.float32(1e-12)
    res = x + ss + bb
    m = res.mean(axis=1, keepdims=True)
    inv = (1 / np.sqrt(res.var(axis=1, keepdims=True) + small)).astype(np.float32)
    add(
        "com.microsoft.SkipLayerNormalization",
        [("inputT", x), ("skipT", ss), ("gammaT", sc), ("betaT", bb), ("biasT", bb)],
        [
            ("outputT", (res - m) * inv * sc + bb),
            ("meanT", m),
            ("invStdT", inv),
            ("residualT", res),
        ],
    )
    ri = (1 / np.sqrt(np.mean(res * res, axis=1, keepdims=True) + small)).astype(np.float32)
    add(
        "com.microsoft.SkipSimplifiedLayerNormalization",
        [("inputT", x), ("skipT", ss), ("gammaT", sc), ("biasT", bb)],
        [
            ("outputT", res * ri * sc),
            ("meanT", np.zeros_like(m)),
            ("invStdT", ri),
            ("residualT", res),
        ],
    )
    gate = np.full_like(x, 0.5)
    add(
        "com.microsoft.GatedRMSNorm",
        [("xT", x), ("scaleT", sc), ("gateT", gate)],
        [("yT", x * rms * sc * gate / (1 + np.exp(-gate)))],
    )
    mv = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    vm = mv.mean(axis=(0, 2, 3), keepdims=True)
    vv = mv.var(axis=(0, 2, 3), keepdims=True)
    add("MeanVarianceNormalization", [("x", mv)], [("y", (mv - vm) / np.sqrt(vv))])
    add(
        "LpNormalization",
        [("input", x)],
        [("output", x / np.sqrt(np.sum(x * x, axis=1, keepdims=True)))],
        dict(axis=1, p=2),
    )
    lx = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2) / 5
    ly = np.empty_like(lx)
    for c in range(3):
        ly[:, c] = (
            lx[:, c]
            / (1 + 1e-4 / 3 * np.sum(lx[:, max(0, c - 1) : min(3, c + 2)] ** 2, axis=1)) ** 0.75
        )
    add("LRN", [("x", lx)], [("y", ly)], dict(size=3))
    image = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
    windows = np.array([image[0, 0, y : y + 2, x : x + 2] for y in range(2) for x in range(2)])
    add(
        "AveragePool",
        [("x", image)],
        [("y", windows.mean(axis=(1, 2)).reshape(1, 1, 2, 2))],
        dict(kernel_shape=[2, 2]),
    )
    add(
        "MaxPool",
        [("x", image)],
        [
            ("y", windows.max(axis=(1, 2)).reshape(1, 1, 2, 2)),
            ("indices", np.array([4, 5, 7, 8], np.int64).reshape(1, 1, 2, 2)),
        ],
        dict(kernel_shape=[2, 2]),
    )
    add(
        "LpPool",
        [("x", image)],
        [("y", np.sqrt((windows * windows).sum(axis=(1, 2))).reshape(1, 1, 2, 2))],
        dict(kernel_shape=[2, 2]),
    )
    for name, ref in [
        ("GlobalAveragePool", np.mean(image)),
        ("GlobalMaxPool", np.max(image)),
        ("GlobalLpPool", np.sqrt(np.sum(image * image))),
    ]:
        add(name, [("x", image)], [("y", np.array([[[[ref]]]], np.float32))])
    unpool = np.zeros((1, 1, 3, 3), np.float32)
    unpool.reshape(-1)[[0, 2, 6, 8]] = [1, 2, 3, 4]
    add(
        "MaxUnpool",
        [
            ("x", np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)),
            ("indices", np.array([0, 2, 6, 8], np.int64).reshape(1, 1, 2, 2)),
            ("output_shape", np.array([1, 1, 3, 3], np.int64)),
        ],
        [("output", unpool)],
        dict(kernel_shape=[2, 2], strides=[2, 2]),
    )
    weight = np.array([[[[1.0, -1.0], [0.5, 2.0]]]], np.float32)
    conv = np.array([np.sum(w * weight[0, 0]) for w in windows], np.float32).reshape(1, 1, 2, 2)
    biasc = np.array([0.25], np.float32)
    add("Conv", [("x", image), ("w", weight), ("bias", biasc)], [("y", conv + 0.25)])
    deform = np.zeros((1, 8, 2, 2), np.float32)
    mask = np.ones((1, 4, 2, 2), np.float32)
    add(
        "DeformConv",
        [("x", image), ("w", weight), ("offset", deform), ("bias", biasc), ("mask", mask)],
        [("y", conv + 0.25)],
    )
    residual = np.full_like(conv, -0.5)
    add(
        "com.microsoft.FusedConv",
        [("x", image), ("w", weight), ("bias", biasc), ("zResidual", residual)],
        [("y", np.maximum(conv - 0.25, 0))],
        dict(activation="Relu"),
    )
    tx = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    tw = np.ones((1, 1, 2, 2), np.float32)
    trans = np.zeros((1, 1, 3, 3), np.float32)
    for y in range(2):
        for xx in range(2):
            trans[0, 0, y : y + 2, xx : xx + 2] += tx[0, 0, y, xx]
    add("ConvTranspose", [("x", tx), ("w", tw)], [("y", trans)])
    qi = image.astype(np.uint8)
    qw = np.ones((1, 1, 2, 2), np.uint8)
    ci = windows.sum(axis=(1, 2)).astype(np.int32).reshape(1, 1, 2, 2)
    add("ConvInteger", [("x", qi), ("w", qw)], [("y", ci)])
    for name in ["ai.onnx.QLinearConv", "com.microsoft.QLinearConv"]:
        add(
            name,
            [
                ("x", qi),
                ("w", qw),
                ("x_scale", np.array(1, np.float32)),
                ("x_zero_point", np.array(0, np.uint8)),
                ("w_zero_point", np.array(0, np.uint8)),
                ("w_scale", np.array(1, np.float32)),
                ("y_scale", np.array(1, np.float32)),
                ("y_zero_point", np.array(0, np.uint8)),
            ],
            [("y", ci.astype(np.uint8))],
        )
    add(
        "Col2Im",
        [
            ("input", np.ones((1, 4, 4), np.float32)),
            ("image_shape", np.array([3, 3], np.int64)),
            ("block_shape", np.array([2, 2], np.int64)),
        ],
        [("output", np.array([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], np.float32))],
    )
    gx = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    grid = np.array([[[[-1.0, -1.0], [1.0, 1.0]]]], np.float32)
    add(
        "GridSample",
        [("x", gx), ("grid", grid)],
        [("y", np.array([[[[1.0, 4.0]]]], np.float32))],
        dict(align_corners=1),
    )
    theta = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], np.float32)
    ag = np.array([[[[-1, -1], [1, -1]], [[-1, 1], [1, 1]]]], np.float32)
    add("AffineGrid", [("theta", theta)], [("grid", ag)], dict(align_corners=1, size=[1, 1, 2, 2]))
    resized = np.repeat(np.repeat(gx, 2, axis=2), 2, axis=3)
    add(
        "Resize",
        [("x", gx)],
        [("y", resized)],
        dict(
            scales=[1.0, 1.0, 2.0, 2.0],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        ),
    )
    add(
        "Upsample",
        [("x", gx), ("scales", np.array([1.0, 1.0, 2.0, 2.0], np.float32))],
        [("y", resized)],
    )
    add(
        "RoiAlign",
        [
            ("x", gx),
            ("rois", np.array([[0, 0, 1, 1]], np.float32)),
            ("batch_indices", np.array([0], np.int64)),
        ],
        [("y", np.array([[[[2.5]]]], np.float32))],
        dict(coordinate_transformation_mode="output_half_pixel", sampling_ratio=1),
    )
    signal = np.arange(1, 5, dtype=np.float32).reshape(1, 4, 1)
    spectrum = np.fft.fft(signal[..., 0], axis=1)
    df = np.stack([spectrum.real, spectrum.imag], axis=-1).astype(np.float32)
    add("DFT", [("input", signal)], [("output", df)])
    signal = np.arange(1, 7, dtype=np.float32).reshape(1, 6, 1)
    spectra = np.stack([np.fft.rfft(signal[0, i : i + 3, 0]) for i in [0, 2]], axis=0)[None]
    st = np.stack([spectra.real, spectra.imag], axis=-1).astype(np.float32)
    add(
        "STFT",
        [("signal", signal), ("window", np.ones(3, np.float32))],
        [("output", st)],
        args=dict(frameStep=2),
    )
    phase = 2 * np.pi * np.arange(4) / 4
    for name, window in [
        ("HannWindow", 0.5 - 0.5 * np.cos(phase)),
        ("HammingWindow", 25 / 46 - 21 / 46 * np.cos(phase)),
        ("BlackmanWindow", 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2 * phase)),
    ]:
        add(name, [("size", np.array(4, np.int32))], [("y", window.astype(np.float32))])
    return result


def run(package, runner, output, fixtures=None):
    failure_root = output.with_name(output.stem + "-failures")
    if output.exists() or failure_root.exists():
        raise FileExistsError("Conformance evidence exists; choose a new output name")
    records = []
    with tempfile.TemporaryDirectory(prefix="xrgpu-spatial-check-") as tmp:
        root = Path(tmp)
        for index, case in enumerate(cases() if fixtures is None else fixtures):
            expected = case["expected"]
            directory = root / str(index)
            directory.mkdir()
            job = {k: v for k, v in case.items() if k not in {"expected", "case"}}
            job.update(
                package=str(package),
                output_directory=str(directory / "actual"),
                outputs=[
                    dict(name=n, dtype=str(a.dtype), shape=list(a.shape))
                    for n, a in expected.items()
                ],
            )
            path = directory / "job.json"
            path.write_text(json.dumps(job, allow_nan=False))
            row = dict(
                operator=case["operation"], case=case.get("case", "representative"), status="failed"
            )
            result = subprocess.run(
                [str(runner), str(path)], text=True, capture_output=True, timeout=120
            )
            try:
                if result.returncode:
                    raise RuntimeError(result.stderr.strip())
                answer = json.loads(result.stdout)
                info = answer.get("info", {})
                if (
                    info.get("backend") != "software_vulkan"
                    or info.get("physical_qualification") is not False
                    or info.get("cpu_compute_fallback") is not False
                    or info.get("executed") is not True
                ):
                    raise RuntimeError("Expected explicitly labelled native software Vulkan")
                if len(answer["outputs"]) != len(expected):
                    raise ValueError("Output count mismatch")
                maxima = []
                for actual, (name, reference) in zip(answer["outputs"], expected.items()):
                    if (
                        actual["name"] != name
                        or actual["shape"] != list(reference.shape)
                        or actual["dtype"] != str(reference.dtype)
                    ):
                        raise ValueError("Output metadata mismatch")
                    values = np.fromfile(actual["file"], dtype=reference.dtype).reshape(
                        reference.shape
                    )
                    if reference.dtype.kind == "f":
                        np.testing.assert_allclose(
                            values, reference, rtol=2e-5, atol=2e-5, equal_nan=True
                        )
                        finite = np.isfinite(values) & np.isfinite(reference)
                        difference = np.abs(
                            values[finite].astype(np.float64) - reference[finite].astype(np.float64)
                        )
                        maxima.append(float(difference.max()) if difference.size else 0.0)
                    else:
                        np.testing.assert_array_equal(values, reference)
                row.update(
                    status="passed", max_absolute_error=max(maxima, default=0), evidence=info
                )
            except Exception as error:
                row["error"] = str(error)
                destination = failure_root / f"case-{index:05d}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(directory, destination)
                specifications = {}
                for name, array in expected.items():
                    array.tofile(destination / (name + ".expected.bin"))
                    specifications[name] = dict(dtype=str(array.dtype), shape=list(array.shape))
                (destination / "expected.json").write_text(
                    json.dumps(specifications, indent=2) + "\n"
                )
                row["failure_artifacts"] = str(destination)
            records.append(row)
    report = dict(
        schema_version=1,
        kind="xrgpu.spatial-software-vulkan-check",
        physical_qualification=False,
        package=str(package),
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        source_sha256=json.loads((package / "development.json").read_text()).get("source_sha256")
        if (package / "development.json").is_file()
        else None,
        scope="independent_representative_and_parameter_cases",
        results=records,
        passed=sum(r["status"] == "passed" for r in records),
        failed=sum(r["status"] != "passed" for r in records),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", action="store_true")
    args = parser.parse_args()
    fixtures = None
    if args.features:
        from .spatial_features import cases as feature_cases

        fixtures = cases() + feature_cases()
    report = run(args.package.resolve(), args.runner.resolve(), args.output.resolve(), fixtures)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))
    return bool(report["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
