# SPDX-License-Identifier: Apache-2.0
"""Independent small NumPy/SciPy references for the native tensor family."""

from pathlib import Path
import json
from kernels import provenance
import numpy as np
from scipy.special import erf, expit

ROOT = Path(__file__).resolve().parents[3]


def representatives():
    result = []

    def f(x):
        return np.asarray(x, dtype=np.float32)

    def i(x):
        return np.asarray(x, dtype=np.int64)

    def add(name, arrays, expected, attrs=None, args=None):
        op = name if "." in name else "ai.onnx." + name
        manifest = provenance.document(op, "contract")
        inputs = arrays if isinstance(arrays, dict) else dict(zip(manifest["inputs"], arrays))
        outputs = (
            expected
            if isinstance(expected, dict)
            else dict(
                zip(manifest["outputs"], expected if isinstance(expected, tuple) else [expected])
            )
        )
        result.append(
            dict(operation=op, inputs=inputs, expected=outputs, attrs=attrs or {}, args=args or {})
        )

    x = f([[-1, 0, 2], [3, -4, 5]])
    y = f([[2, -1, 2]])
    for name, fn in {
        "Equal": np.equal,
        "Greater": np.greater,
        "GreaterOrEqual": np.greater_equal,
        "Less": np.less,
        "LessOrEqual": np.less_equal,
    }.items():
        add(name, [x, y], fn(x, y))
    a, b = np.array([[True, False], [False, True]]), np.array([True, False])
    for name, fn in {"And": np.logical_and, "Or": np.logical_or, "Xor": np.logical_xor}.items():
        add(name, [a, b], fn(a, b))
    add("Not", [a], ~a)
    ints = np.array([-2147483648, -1, 0, 16777217, 2147483647], dtype=np.int32)
    add("BitwiseNot", [ints], ~ints)
    shifts = np.array([1, 2, 3], dtype=np.uint32)
    values = np.array([3, 7, 15], dtype=np.uint32)
    add("BitShift", [values, shifts], values << shifts, {"direction": "LEFT"})
    add("Cast", [f([-2.75, 0.5, 127])], np.array([-2, 0, 127], dtype=np.int8), {"to": 3})
    add("CastLike", [x, np.array(0, dtype=np.float16)], x.astype(np.float16))
    add("BitCast", [ints], ints.view(np.float32), {"to": 1})
    add(
        "Identity",
        [np.array([2**53 + 1, -(2**53) - 3], dtype=np.int64)],
        np.array([2**53 + 1, -(2**53) - 3], dtype=np.int64),
    )
    special = f([-np.inf, -1, 0, np.inf, np.nan])
    add("IsInf", [special], np.isinf(special))
    add("IsNaN", [special], np.isnan(special))
    v = f([-2, -1, -0.25, 0, 0.25, 1, 2])

    def gelu(t):
        return t.astype(np.float64) * (1 + erf(t.astype(np.float64) / np.sqrt(2))) / 2

    def fast_gelu(t):
        return (
            t.astype(np.float64)
            * (
                1
                + np.tanh(
                    np.sqrt(2 / np.pi)
                    * (t.astype(np.float64) + 0.044715 * t.astype(np.float64) ** 3)
                )
            )
            / 2
        )

    for name, ref, attrs in [
        ("Celu", np.maximum(v, 0) + np.minimum(0, np.expm1(v)), {}),
        ("Elu", np.where(v >= 0, v, np.expm1(v)), {}),
        ("Gelu", gelu(v), {}),
        ("HardSigmoid", np.clip(0.2 * v + 0.5, 0, 1), {}),
        ("LeakyRelu", np.where(v >= 0, v, 0.01 * v), {}),
        ("Selu", 1.0507010221481323 * np.where(v > 0, v, 1.6732631921768188 * np.expm1(v)), {}),
        ("Shrink", np.where(v < -0.5, v, np.where(v > 0.5, v, 0)), {}),
        ("Swish", v * expit(v), {}),
        ("ThresholdedRelu", np.where(v > 1, v, 0), {}),
    ]:
        add(name, [v], np.asarray(ref, dtype=np.float32), attrs)
    add("Clip", [v, f(-0.5), f(0.75)], np.clip(v, -0.5, 0.75))
    for name, fn in {"Max": np.maximum, "Min": np.minimum}.items():
        add(name, [x, y], fn(x, y))
    add("Mean", [x, y], ((x + y) / 2).astype(np.float32))
    add("Sum", [x, y], x + y)
    add(
        "Mod",
        [np.array([-5, 5], dtype=np.int32), np.array([3, -3], dtype=np.int32)],
        np.mod(np.array([-5, 5], dtype=np.int32), np.array([3, -3], dtype=np.int32)),
    )
    add("Pow", [f([-2, 2, 4]), np.array([3, -2, 2], dtype=np.int32)], f([-8, 0.25, 16]))
    add("SwiGLU", [x, np.broadcast_to(y, x.shape).copy()], (x * expit(x) * y).astype(np.float32))
    add("Dropout", [x], {"y": x.copy(), "mask": np.ones_like(x, dtype=np.bool_)})
    condition = np.array([[True], [False]])
    add("Where", [condition, x, -x], np.where(condition, x, -x))
    add(
        "Constant",
        [],
        f([[1, 2], [3, 4]]),
        {"value": {"dtype": "float32", "shape": [2, 2], "values": [1, 2, 3, 4]}},
    )
    add("ConstantOfShape", [i([2, 3])], np.full((2, 3), -2.5, dtype=np.float32), {"value": -2.5})
    add("EyeLike", [np.zeros((3, 4), dtype=np.int32)], np.eye(3, 4, k=1, dtype=np.int32), {"k": 1})
    add("Shape", [x], i([2, 3]))
    add("Size", [x], i(6))
    add("Reshape", [x, i([3, 2])], x.reshape(3, 2))
    add("Flatten", [x.reshape(1, 2, 3)], x.reshape(2, 3), {"axis": 2})
    add("Squeeze", [x.reshape(1, 2, 1, 3), i([0, 2])], x)
    add("Unsqueeze", [x, i([0, -1])], x.reshape(1, 2, 3, 1))
    add("Transpose", [x], x.T, {"perm": [1, 0]})
    add("Concat", [x, x + 1], np.concatenate([x, x + 1], axis=1), {"axis": 1})
    add("Expand", [y, i([2, 3])], np.broadcast_to(y, (2, 3)).copy())
    add("Tile", [x, i([2, 2])], np.tile(x, (2, 2)))
    add("Pad", [x, f(-9)], np.pad(x, ((1, 0), (1, 2)), constant_values=-9), {"pads": [1, 1, 0, 2]})
    add("Slice", [x, np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)], x[:, 1:3])
    add("Gather", [x, np.array([2, 0], dtype=np.int32)], x[:, [2, 0]], {"axis": 1})
    idx = np.array([[2, 0], [1, 2]], dtype=np.int32)
    add("GatherElements", [x, idx], np.take_along_axis(x, idx, axis=1), {"axis": 1})
    add("GatherND", [x, i([[0, 1], [1, 2]])], f([x[0, 1], x[1, 2]]))
    seq = f(np.arange(12).reshape(3, 2, 2))
    reverse = seq.copy()
    reverse[:2, 0] = seq[:2, 0][::-1]
    reverse[:3, 1] = seq[:3, 1][::-1]
    add("ReverseSequence", [seq, i([2, 3])], reverse)
    image = f(np.arange(32).reshape(1, 2, 4, 4))
    space = image.reshape(1, 2, 2, 2, 2, 2).transpose(0, 3, 5, 1, 2, 4).reshape(1, 8, 2, 2)
    add("SpaceToDepth", [image], space, {"blocksize": 2})
    add("DepthToSpace", [space], image, {"blocksize": 2})
    add("Trilu", [x, i(0)], np.triu(x))
    add(
        "TopK",
        [x],
        {
            "values": np.take_along_axis(x, np.argsort(-x, axis=1, kind="stable")[:, :2], axis=1),
            "indices": np.argsort(-x, axis=1, kind="stable")[:, :2].astype(np.int64),
        },
        {"axis": 1},
        {"k": 2},
    )
    add("Split", [x, i([1, 2])], {"y0": x[:, :1], "y1": x[:, 1:]}, {"axis": 1})
    add(
        "SplitToSequence",
        [x, np.array([1, 2], dtype=np.uint32)],
        {"y0": x[:, :1], "y1": x[:, 1:]},
        {"axis": 1},
    )
    add("Compress", [x, np.array([True, False, True])], x[:, [0, 2]], {"axis": 1})
    add("NonZero", [x], np.asarray(np.nonzero(x), dtype=np.int64))
    uniq = np.array([3, 1, 3, -1, 2, 1], dtype=np.int32)
    values, first, inverse, counts = np.unique(
        uniq, return_index=True, return_inverse=True, return_counts=True
    )
    add(
        "Unique",
        [uniq],
        {
            "y": values,
            "indices": first.astype(np.int64),
            "inverse_indices": inverse.astype(np.int64),
            "counts": counts.astype(np.int64),
        },
    )
    boxes = f([[[0, 0, 1, 1], [0, 0.1, 1, 1.1], [2, 2, 3, 3]]])
    scores = f([[[0.9, 0.8, 0.7]]])
    add("NonMaxSuppression", [boxes, scores, i(3), f(0.5), f(0)], i([[0, 0, 0], [0, 0, 2]]))
    add(
        "Range",
        [np.array(-3, dtype=np.int32), np.array(6, dtype=np.int32), np.array(2, dtype=np.int32)],
        np.arange(-3, 6, 2, dtype=np.int32),
    )
    indices = np.array([0, 2, -1], dtype=np.int32)
    add(
        "OneHot",
        [indices, np.array(3, dtype=np.int32), f([-1, 5])],
        f([[5, -1, -1], [-1, -1, 5], [-1, -1, 5]]),
    )
    updates = f([[8, 9], [10, 11]])
    scatter = x.copy()
    np.put_along_axis(scatter, idx, updates, axis=1)
    add("Scatter", [x, idx, updates], scatter, {"axis": 1})
    add("ScatterElements", [x, idx, updates], scatter, {"axis": 1})
    nd = x.copy()
    nd[[0, 1]] = f([[2, 3, 4], [5, 6, 7]])
    add("ScatterND", [x, i([[0], [1]]), f([[2, 3, 4], [5, 6, 7]])], nd)
    past = f(np.arange(8).reshape(1, 1, 4, 2))
    update = f([[[[20, 21], [22, 23]]]])
    present = past.copy()
    present[:, :, 1:3] = update
    add("TensorScatter", [past, update, i([1])], present)
    add("If", [np.array(True), x, -x], x)
    initial, step = f([1, -1]), f([0.5, 2])
    scan = np.asarray([initial + (n + 1) * step for n in range(4)], dtype=np.float32)
    add(
        "Loop",
        [np.array(4, dtype=np.uint32), np.array(True), initial, step],
        {"v_final": scan[-1], "scan_output": scan},
    )
    scan_input = f([[1, 2], [3, 4], [5, 6]])
    sequence = np.cumsum(scan_input, axis=0) + initial
    add("Scan", [initial, scan_input], {"final_state": sequence[-1], "scan_output": sequence})
    scales = f([0.5, 0.25, 2])
    zp = np.array([10, 20, 30], dtype=np.uint8)
    quant = np.clip(np.rint(x / scales) + zp, 0, 255).astype(np.uint8)
    add("QuantizeLinear", [x, scales, zp], quant, {"axis": 1})
    add(
        "DequantizeLinear",
        [quant, scales, zp],
        ((quant.astype(np.int32) - zp.astype(np.int32)) * scales).astype(np.float32),
        {"axis": 1},
    )
    low, high = min(0, float(x.min())), max(0, float(x.max()))
    scale = np.float32((high - low) / 255)
    zero = np.uint8(np.clip(np.rint(-low / scale), 0, 255))
    q = np.clip(np.rint(x / scale) + zero, 0, 255).astype(np.uint8)
    add(
        "DynamicQuantizeLinear",
        [x],
        {"y": q, "y_scale": np.asarray(scale), "y_zero_point": np.asarray(zero)},
    )
    left, right, bias = f([[1, 2, 3], [-1, 0, 1]]), f([[1, -1], [2, 0], [3, 1]]), f([0.5, -0.5])
    gemm = left @ right + bias
    add("Gemm", [left, right, bias], gemm)
    add("Einsum", [left, right], np.einsum("ij,jk->ik", left, right), {"equation": "ij,jk->ik"})
    add("com.microsoft.FusedGemm", [left, right, bias], np.maximum(gemm, 0), {"activation": "Relu"})
    add("com.microsoft.FusedMatMul", [left, right], left @ right)
    add("com.microsoft.TransposeMatMul", [left, right], left @ right)
    add("com.microsoft.GemmFastGelu", [left, right, bias], fast_gelu(gemm).astype(np.float32))
    add(
        "com.microsoft.BiasAdd",
        [x.reshape(1, 2, 3), y.reshape(3), np.full((1, 2, 3), 0.5, np.float32)],
        (x + y + 0.5).reshape(1, 2, 3),
    )
    add("com.microsoft.BiasGelu", [x, y.reshape(3)], gelu(x + y).astype(np.float32))
    add("com.microsoft.FastGelu", [x, y.reshape(3)], fast_gelu(x + y).astype(np.float32))
    add("com.microsoft.Gelu", [v], gelu(v).astype(np.float32))
    add("com.microsoft.QuickGelu", [v], (v * expit(1.702 * v)).astype(np.float32))
    add("com.microsoft.GatedAdd", [x, y, f([[0.5], [2]])], x + y * f([[0.5], [2]]))
    split = f(np.arange(12).reshape(1, 2, 6)) / 4
    split_bias = f(np.arange(6)) / 10
    add(
        "com.microsoft.BiasSplitGelu",
        [split, split_bias],
        ((split[..., :3] + split_bias[:3]) * gelu(split[..., 3:] + split_bias[3:])).astype(
            np.float32
        ),
    )
    declared = set(json.loads((ROOT / "kernels/_shared/extended/tensor_ops.json").read_text()))
    actual = {case["operation"] for case in result}
    if declared != actual:
        raise AssertionError(
            {"missing": sorted(declared - actual), "extra": sorted(actual - declared)}
        )
    return result
