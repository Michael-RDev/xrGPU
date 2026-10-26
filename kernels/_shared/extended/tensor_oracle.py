# SPDX-License-Identifier: Apache-2.0
"""Independent CPU reference semantics for local tensor-kernel conformance."""

import numpy as np
from scipy.special import erf, expit


def oracle(operation, x, attrs=None, args=None, outputs=None):
    a = dict(attrs or {}) | dict(args or {})
    name = operation.split(".")[-1]
    ms = operation.startswith("com.microsoft.")
    values = list(x.values())
    first = values[0] if values else None
    dtype = first.dtype if first is not None else np.dtype("float32")
    outnames = list(outputs or {})

    def result(value, names=None):
        if isinstance(value, dict):
            return value
        if isinstance(value, tuple):
            return dict(zip(names or outnames, value))
        return {(names or outnames or ["y"])[0]: np.asarray(value)}

    def target_type(key="to", fallback=None):
        value = a.get(key, 0)
        codes = {
            1: "float32",
            2: "uint8",
            3: "int8",
            5: "int16",
            6: "int32",
            7: "int64",
            9: "bool",
            10: "float16",
            12: "uint32",
        }
        return np.dtype(value if isinstance(value, str) else codes.get(value, fallback or dtype))

    def typed(value, t=dtype):
        return np.asarray(value).astype(t)

    def gelu(v, fast=False):
        v = v.astype(np.float64)
        return (
            v * (1 + np.tanh(np.sqrt(2 / np.pi) * (v + 0.044715 * v**3))) / 2
            if fast
            else v * (1 + erf(v / np.sqrt(2))) / 2
        )

    with np.errstate(all="ignore"):
        binary = {
            "And": np.logical_and,
            "Or": np.logical_or,
            "Xor": np.logical_xor,
            "Equal": np.equal,
            "Greater": np.greater,
            "GreaterOrEqual": np.greater_equal,
            "Less": np.less,
            "LessOrEqual": np.less_equal,
            "Pow": np.power,
        }
        if name in binary:
            if name == "Pow" and dtype.kind in "iu" and np.any(values[1] < 0):
                return result(np.power(values[0].astype(np.float64), values[1]).astype(dtype))
            return result(
                binary[name](values[0], values[1]).astype(dtype)
                if name == "Pow"
                else binary[name](values[0], values[1])
            )
        if name == "Not":
            return result(np.logical_not(first))
        if name == "BitwiseNot":
            return result(np.bitwise_not(first))
        if name == "BitShift":
            return result(
                (np.right_shift if a["direction"] == "RIGHT" else np.left_shift)(
                    values[0], values[1]
                ).astype(dtype)
            )
        if name in {"Cast", "CastLike"}:
            return result(first.astype(values[1].dtype if name == "CastLike" else target_type()))
        if name == "BitCast":
            t = target_type()
            s = list(first.shape)
            if s:
                s[-1] = s[-1] * dtype.itemsize // t.itemsize
            return result(np.frombuffer(first.tobytes(), dtype=t).reshape(s))
        if name == "Identity":
            return result(first.copy())
        if name == "IsNaN":
            return result(np.isnan(first))
        if name == "IsInf":
            return result(
                (
                    np.isneginf(first)
                    if a.get("detect_negative", 1)
                    else np.zeros_like(first, dtype=bool)
                )
                | (
                    np.isposinf(first)
                    if a.get("detect_positive", 1)
                    else np.zeros_like(first, dtype=bool)
                )
            )
        if name == "Celu":
            alpha = a.get("alpha", 1)
            return result(
                typed(
                    np.maximum(first, 0)
                    + np.minimum(0, alpha * np.expm1(first.astype(float) / alpha))
                )
            )
        if name == "Elu":
            return result(
                typed(
                    np.where(first >= 0, first, a.get("alpha", 1) * np.expm1(first.astype(float)))
                )
            )
        if name == "Gelu":
            return result(typed(gelu(first, a.get("approximate", "none") == "tanh")))
        if name == "HardSigmoid":
            return result(
                typed(
                    np.clip(
                        np.float32(a.get("alpha", 0.2)) * first.astype(np.float32)
                        + np.float32(a.get("beta", 0.5)),
                        0,
                        1,
                    )
                )
            )
        if name == "LeakyRelu":
            return result(
                typed(np.where(first >= 0, first, a.get("alpha", 0.01) * first.astype(float)))
            )
        if name == "Selu":
            # The contract computes its two FP32 products separately. This also
            # preserves gradual underflow between alpha and gamma.
            source = first.astype(np.float32)
            negative = np.float32(a.get("alpha", 1.6732631921768188)) * np.expm1(source)
            value = np.float32(a.get("gamma", 1.0507010221481323)) * np.where(
                source > 0, source, negative
            )
            return result(value.astype(dtype))
        if name == "Shrink":
            work = first.astype(np.float32) if dtype.kind != "f" else first.astype(float)
            return result(
                typed(
                    np.where(
                        work < -a.get("lambd", 0.5),
                        work + a.get("bias", 0),
                        np.where(work > a.get("lambd", 0.5), work - a.get("bias", 0), 0),
                    )
                )
            )
        if name in {"Swish", "QuickGelu"}:
            return result(
                typed(
                    first.astype(float)
                    * expit(a.get("alpha", 1.702 if ms else 1) * first.astype(float))
                )
            )
        if name == "ThresholdedRelu":
            return result(typed(np.where(first > a.get("alpha", 1), first, 0)))
        if name == "SwiGLU":
            if dtype == np.float16:
                scaled = (np.float16(a.get("alpha", 1)) * first).astype(np.float16)
                gate = expit(scaled.astype(np.float32)).astype(np.float16)
                swish = (first * gate).astype(np.float16)
                return result((swish * values[1]).astype(np.float16))
            return result(typed(first * expit(np.float32(a.get("alpha", 1)) * first) * values[1]))
        if name == "Clip":
            return result(np.clip(first, x.get("min", None), x.get("max", None)))
        if name in {"Max", "Min", "Sum", "Mean"}:
            arrays = np.broadcast_arrays(*values)
            if name == "Max":
                v = np.maximum.reduce(arrays)
            elif name == "Min":
                v = np.minimum.reduce(arrays)
            else:
                v = np.sum(np.stack(arrays).astype(float), axis=0) / (
                    len(arrays) if name == "Mean" else 1
                )
            return result(typed(v))
        if name == "Mod":
            return result(typed((np.fmod if a.get("fmod", 0) else np.mod)(values[0], values[1])))
        if name == "Where":
            return result(np.where(values[0], values[1], values[2]))
        if name == "Dropout":
            return result(
                {
                    "y": first.copy(),
                    **({"mask": np.ones_like(first, dtype=bool)} if "mask" in outnames else {}),
                }
            )
        if name == "Constant":
            if isinstance(a.get("value"), dict):
                v = a["value"]
                return result(np.asarray(v["values"], dtype=v["dtype"]).reshape(v["shape"]))
            v = a.get("value_floats", a.get("value_float", a.get("value", 0)))
            return result(
                np.asarray(v, dtype=(outputs or {}).get("output", {}).get("dtype", "float32"))
            )
        if name == "ConstantOfShape":
            t = (outputs or {}).get("output", {}).get("dtype", "float32")
            v = a.get("value", 0)
            if isinstance(v, dict):
                t = v["dtype"]
                v = np.asarray(v["values"]).flat[0]
            return result(np.full(tuple(int(v) for v in first.flat), float(v), dtype=t))
        if name == "EyeLike":
            return result(np.eye(*first.shape, k=a.get("k", 0), dtype=target_type("dtype", dtype)))
        if name == "Shape":
            shape = list(first.shape)
            return result(
                np.asarray(
                    shape[slice(a.get("start", 0), a.get("end", len(shape)))], dtype=np.int64
                )
            )
        if name == "Size":
            return result(np.asarray(first.size, dtype=np.int64))
        if name == "Reshape":
            s = [int(v) for v in values[1].flat]
            if not a.get("allowzero", 0):
                s = [first.shape[i] if d == 0 else d for i, d in enumerate(s)]
            return result(first.reshape(s))
        if name == "Flatten":
            axis = a.get("axis", 1)
            axis = axis + first.ndim if axis < 0 else axis
            return result(
                first.reshape(int(np.prod(first.shape[:axis])), int(np.prod(first.shape[axis:])))
            )
        if name == "Squeeze":
            axes = (
                tuple(sorted({int(v) % first.ndim for v in x["axes"].flat}))
                if "axes" in x and x["axes"].size
                else None
            )
            return result(np.squeeze(first, axis=axes))
        if name == "Unsqueeze":
            return result(np.expand_dims(first, tuple(int(v) for v in values[1].flat)))
        if name == "Transpose":
            return result(np.transpose(first, a.get("perm")))
        if name == "Concat":
            return result(np.concatenate(values, axis=a["axis"]))
        if name == "Expand":
            return result(
                np.broadcast_to(
                    first, np.broadcast_shapes(first.shape, tuple(int(v) for v in values[1].flat))
                ).copy()
            )
        if name == "Tile":
            return result(np.tile(first, tuple(int(v) for v in values[1].flat)))
        if name == "Pad":
            pads = a["pads"]
            before = pads[: first.ndim]
            after = pads[first.ndim :]
            output_shape = tuple(n + int(b) + int(e) for n, b, e in zip(first.shape, before, after))
            if any(n == 0 for n in output_shape):
                return result(np.empty(output_shape, dtype=dtype))
            slices = tuple(
                slice(max(0, -b), first.shape[d] - max(0, -e))
                for d, (b, e) in enumerate(zip(before, after))
            )
            v = first[slices]
            p = [(max(0, b), max(0, e)) for b, e in zip(before, after)]
            mode = a.get("mode", "constant")
            return result(
                np.pad(
                    v,
                    p,
                    mode=mode,
                    **(
                        {"constant_values": values[1].item() if len(values) > 1 else 0}
                        if mode == "constant"
                        else {}
                    ),
                )
            )
        if name == "Slice":
            slices = [slice(None)] * first.ndim
            starts = x["starts"].flat
            ends = x["ends"].flat
            axes = x["axes"].flat if "axes" in x else range(x["starts"].size)
            steps = x["steps"].flat if "steps" in x else [1] * x["starts"].size
            for start, end, axis, step in zip(starts, ends, axes, steps):
                end = int(end)
                end = None if int(step) < 0 and (end <= -2147483648 or end >= 2147483647) else end
                slices[int(axis)] = slice(int(start), end, int(step))
            return result(first[tuple(slices)])
        if name == "Gather":
            indices = values[1].astype(np.int64)
            axis = a.get("axis", 0) % first.ndim
            valid = (indices >= -first.shape[axis]) & (indices < first.shape[axis])
            safe = np.where(valid, indices, 0)
            shape = (1,) * axis + indices.shape + (1,) * (first.ndim - axis - 1)
            return result(np.where(valid.reshape(shape), np.take(first, safe, axis=axis), 0))
        if name == "GatherElements":
            indices = values[1].astype(np.int64)
            axis = a.get("axis", 0) % first.ndim
            value = np.empty(indices.shape, dtype=dtype)
            for position in np.ndindex(indices.shape):
                source = list(position)
                source[axis] = indices[position]
                value[position] = first[tuple(source)]
            return result(value)
        if name == "GatherND":
            indices = values[1].astype(np.int64)
            batch = a.get("batch_dims", 0)
            shape = indices.shape[:-1] + first.shape[batch + indices.shape[-1] :]
            v = np.zeros(shape, dtype=dtype)
            for prefix in np.ndindex(indices.shape[:-1]):
                at = indices[prefix]
                valid = all(
                    -first.shape[batch + d] <= j < first.shape[batch + d] for d, j in enumerate(at)
                )
                if valid:
                    v[prefix] = first[prefix[:batch] + tuple(at)]
            return result(v)
        if name == "ReverseSequence":
            v = first.copy()
            batch = a.get("batch_axis", 1)
            time = a.get("time_axis", 0)
            for b, length in enumerate(values[1].flat):
                at = [slice(None)] * v.ndim
                at[batch] = b
                at[time] = slice(0, int(length))
                sub = first[tuple(at)]
                v[tuple(at)] = np.flip(sub, axis=time - (batch < time))
            return result(v)
        if name in {"SpaceToDepth", "DepthToSpace"}:
            b = a["blocksize"]
            n, c, h, w = first.shape
            if name == "SpaceToDepth":
                v = (
                    first.reshape(n, c, h // b, b, w // b, b)
                    .transpose(0, 3, 5, 1, 2, 4)
                    .reshape(n, c * b * b, h // b, w // b)
                )
            elif a.get("mode", "DCR") == "DCR":
                v = (
                    first.reshape(n, b, b, c // (b * b), h, w)
                    .transpose(0, 3, 4, 1, 5, 2)
                    .reshape(n, c // (b * b), h * b, w * b)
                )
            else:
                v = (
                    first.reshape(n, c // (b * b), b, b, h, w)
                    .transpose(0, 1, 4, 2, 5, 3)
                    .reshape(n, c // (b * b), h * b, w * b)
                )
            return result(v)
        if name == "Trilu":
            return result(
                (np.triu if a.get("upper", 1) else np.tril)(
                    first, int(values[1].item()) if len(values) > 1 else 0
                )
            )
        if name in {"Split", "SplitToSequence"}:
            axis = a.get("axis", 0) % first.ndim
            n = first.shape[axis]
            parts = list(x["split"].flat) if "split" in x else []
            if name == "SplitToSequence" and len(parts) == 1:
                width = int(parts[0])
                parts = [min(width, n - j) for j in range(0, n, width)]
            if not parts and outputs:
                parts = [
                    s["shape"][axis] if name == "Split" or a.get("keepdims", 1) else 1
                    for s in outputs.values()
                ]
            if not parts:
                count = a.get("num_outputs", len(outnames) if name == "Split" else n)
                width = (n + count - 1) // count
                parts = [min(width, n - j * width) for j in range(count)]
            parts = parts[: len(outnames)]
            out = np.split(first, np.cumsum(parts), axis=axis)[: len(parts)]
            if name == "SplitToSequence" and not a.get("keepdims", 1):
                out = [np.squeeze(v, axis=axis) for v in out]
            return result(dict(zip(outnames, out)))
        if name == "Compress":
            condition = values[1].reshape(-1)
            length = first.size if "axis" not in a else first.shape[a["axis"]]
            return result(np.compress(condition[:length], first, axis=a.get("axis")))
        if name == "NonZero":
            return result(
                np.asarray(np.nonzero(first), dtype=np.int64)
                if first.ndim
                else np.empty((0, int(first != 0)), dtype=np.int64)
            )
        if name == "Unique":
            v, first_indices, inverse, counts = np.unique(
                first,
                return_index=True,
                return_inverse=True,
                return_counts=True,
                axis=a.get("axis"),
            )
            if not a.get("sorted", 1):
                order = np.argsort(first_indices)
                remap = np.empty_like(order)
                remap[order] = np.arange(len(order))
                v = np.take(v, order, axis=a.get("axis", 0))
                first_indices = first_indices[order]
                counts = counts[order]
                inverse = remap[inverse]
            allout = {
                "y": v,
                "indices": first_indices.astype(np.int64),
                "inverse_indices": inverse.reshape(-1).astype(np.int64),
                "counts": counts.astype(np.int64),
            }
            return result({n: allout[n] for n in outnames})
        if name == "TopK":
            axis = a.get("axis", -1)
            key = -first.astype(np.float64) if a.get("largest", 1) else first.astype(np.float64)
            order = np.argsort(key, axis=axis, kind="stable")
            at = [slice(None)] * first.ndim
            at[axis] = slice(0, a["k"])
            order = order[tuple(at)]
            return result(
                {
                    "values": np.take_along_axis(first, order, axis),
                    "indices": order.astype(np.int64),
                }
            )
        if name == "Range":
            return result(
                np.arange(values[0].item(), values[1].item(), values[2].item(), dtype=dtype)
            )
        if name == "OneHot":
            depth = int(values[1].item())
            off, on = values[2].reshape(-1)
            index = first.astype(np.int64)
            valid = (index >= -depth) & (index < depth)
            index = np.where(index < 0, index + depth, index)
            out = np.full(first.shape + (depth,), off, dtype=values[2].dtype)
            for pos in np.ndindex(first.shape):
                if valid[pos]:
                    out[pos + (int(index[pos]),)] = on
            return result(np.moveaxis(out, -1, a.get("axis", -1)))
        if name in {"Scatter", "ScatterElements", "ScatterND"}:
            data, indices, updates = values
            v = data.copy()
            red = a.get("reduction", "none")
            axis = a.get("axis", 0)
            for pos in np.ndindex(updates.shape):
                if name == "ScatterND":
                    prefix = pos[: indices.ndim - 1]
                    key = tuple(int(q) for q in indices[prefix]) + pos[indices.ndim - 1 :]
                else:
                    key = tuple(
                        int(indices[pos]) if d == axis % data.ndim else p for d, p in enumerate(pos)
                    )
                value = updates[pos]
                if red == "none":
                    v[key] = value
                elif red == "add":
                    v[key] += value
                elif red == "mul":
                    v[key] *= value
                elif red == "min":
                    v[key] = np.minimum(v[key], value)
                elif red == "max":
                    v[key] = np.maximum(v[key], value)
                else:
                    raise ValueError("Unknown scatter reduction")
            return result(v)
        if name == "TensorScatter":
            past, update = values[:2]
            v = past.copy()
            axis = a.get("axis", -2) % past.ndim
            indices = values[2].reshape(-1) if len(values) > 2 else None
            for pos in np.ndindex(update.shape):
                b = pos[0]
                u = pos[axis]
                target = (
                    (int(indices[b]) + u)
                    if indices is not None and indices.size == past.shape[0]
                    else int(indices[b * update.shape[axis] + u])
                    if indices is not None
                    else u
                )
                if a.get("mode", "linear") == "circular":
                    target %= past.shape[axis]
                if not 0 <= target < past.shape[axis]:
                    continue
                key = list(pos)
                key[axis] = target
                v[tuple(key)] = update[pos]
            return result(v)
        if name == "If":
            return result(values[1 if bool(values[0].item()) else 2].copy())
        if name == "Loop":
            capacity = outputs["scan_output"]["shape"][0]
            m = min(int(values[0].item()), capacity) if bool(values[1].item()) else 0
            state = values[2].copy()
            scan = np.zeros((capacity, *state.shape), dtype=state.dtype)
            for t in range(m):
                state = np.asarray(state + values[3], dtype=state.dtype)
                scan[t] = state
            return result({"v_final": state, "scan_output": scan})
        if name == "Scan":
            state = first.copy()
            scan = np.zeros_like(values[1])
            positions = range(len(scan) - 1, -1, -1) if a.get("reverse", 0) else range(len(scan))
            for t in positions:
                state = typed(state + values[1][t])
                scan[t] = state
            return result({"final_state": state, "scan_output": scan})
        if name in {"QuantizeLinear", "DequantizeLinear"}:
            data, scale = values[:2]
            zero = (
                values[2]
                if len(values) > 2
                else np.asarray(
                    0,
                    dtype=(outputs or {}).get(outnames[0], {}).get("dtype", "uint8")
                    if name == "QuantizeLinear"
                    else data.dtype,
                )
            )
            axis = a.get("axis", 1) % max(1, data.ndim)
            block = a.get("block_size", 0)
            if scale.size == 1:
                scale = scale.reshape(())
                zero = zero.reshape(()) if zero.size == 1 else zero
            elif scale.ndim == 1:
                shape = [1] * data.ndim
                shape[axis] = scale.size
                scale = scale.reshape(shape)
                zero = zero.reshape(shape) if zero.size > 1 else zero
            if block:
                scale = np.repeat(scale, block, axis=axis)
                at = [slice(None)] * data.ndim
                at[axis] = slice(0, data.shape[axis])
                scale = scale[tuple(at)]
                if zero.size > 1:
                    zero = np.repeat(zero, block, axis=axis)[tuple(at)]
            if name == "DequantizeLinear":
                return result(
                    (
                        (data.astype(np.int64) - zero.astype(np.int64)) * scale.astype(np.float64)
                    ).astype((outputs or {})[outnames[0]]["dtype"])
                )
            target = np.dtype((outputs or {})[outnames[0]]["dtype"])
            limits = np.iinfo(target)
            precision = a.get("precision", 0)
            calculation = (
                np.float16
                if precision == 10 or precision == 0 and scale.dtype == np.float16
                else np.float32
            )
            quotient = data.astype(calculation) / scale.astype(calculation)
            return result(
                np.clip(np.rint(quotient).astype(np.float32) + zero, limits.min, limits.max).astype(
                    target
                )
            )
        if name == "DynamicQuantizeLinear":
            lo = min(float(first.min()) if first.size else 0, 0)
            hi = max(float(first.max()) if first.size else 0, 0)
            scale = np.float32((hi - lo) / 255 if hi != lo else 1)
            zero = np.uint8(np.clip(np.rint(-lo / scale), 0, 255))
            q = np.clip(np.rint(first / scale) + zero, 0, 255).astype(np.uint8)
            return result({"y": q, "y_scale": np.asarray(scale), "y_zero_point": np.asarray(zero)})
        if name == "NonMaxSuppression":
            boxes, scores = values[:2]
            maximum = (
                int(x["max_output_boxes_per_class"].item())
                if "max_output_boxes_per_class" in x
                else 0
            )
            iou = float(x["iou_threshold"].item()) if "iou_threshold" in x else 0
            threshold = float(x["score_threshold"].item()) if "score_threshold" in x else None
            rows = []
            for b in range(boxes.shape[0]):
                box = boxes[b].astype(float)
                if a.get("center_point_box", 0):
                    box = np.c_[box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2]
                else:
                    box = np.c_[
                        np.minimum(box[:, :2], box[:, 2:]), np.maximum(box[:, :2], box[:, 2:])
                    ]
                for c in range(scores.shape[1]):
                    candidates = [
                        j
                        for j in np.argsort(-scores[b, c], kind="stable")
                        if threshold is None or scores[b, c, j] > threshold
                    ]
                    selected = []
                    while candidates and len(selected) < maximum:
                        j = candidates.pop(0)
                        selected.append(j)
                        remaining = []
                        for k in candidates:
                            wh = np.maximum(
                                0,
                                np.minimum(box[j, 2:], box[k, 2:])
                                - np.maximum(box[j, :2], box[k, :2]),
                            )
                            inter = np.prod(wh)
                            area = (
                                np.prod(box[j, 2:] - box[j, :2])
                                + np.prod(box[k, 2:] - box[k, :2])
                                - inter
                            )
                            if not (area > 0 and inter / area > iou):
                                remaining.append(k)
                        candidates = remaining
                    rows.extend((b, c, j) for j in selected)
            return result(np.asarray(rows, dtype=np.int64).reshape(-1, 3))
        if name == "Einsum":
            return result(typed(np.einsum(a["equation"], *[v.astype(float) for v in values])))
        if name in {"Gemm", "FusedGemm", "FusedMatMul", "TransposeMatMul", "GemmFastGelu"}:
            left, right = values[:2]
            vector_left, vector_right = left.ndim == 1, right.ndim == 1
            if vector_left:
                left = left[None, :]
            if vector_right:
                right = right[:, None]
            if a.get("transBatchA", 0) and left.ndim >= 3:
                left = np.swapaxes(left, -3, -2)
            if a.get("transBatchB", 0) and right.ndim >= 3:
                right = np.swapaxes(right, -3, -2)
            if a.get("transA", 0) and not vector_left:
                left = np.swapaxes(left, -1, -2)
            if a.get("transB", 0) and not vector_right:
                right = np.swapaxes(right, -1, -2)
            alpha = np.float32(a.get("alpha", 1))
            beta = np.float32(a.get("beta", 1))
            shape = np.broadcast_shapes(left.shape[:-2], right.shape[:-2]) + (
                left.shape[-2],
                right.shape[-1],
            )
            v = (
                np.zeros(shape, dtype=np.float32)
                if alpha == 0
                else alpha * np.matmul(left.astype(np.float32), right.astype(np.float32))
            )
            if len(values) > 2 and beta != 0:
                v += beta * values[2].astype(np.float32)
            if vector_left:
                v = np.squeeze(v, axis=-2)
            if vector_right:
                v = np.squeeze(v, axis=-1)
            if name == "GemmFastGelu":
                v = gelu(v, True)
            if name == "FusedGemm":
                act = a.get("activation", "")
                alpha = a.get("activation_alpha", 0.01)
                beta = a.get("activation_beta", 0)
                if act == "Relu":
                    v = np.maximum(v, 0)
                elif act == "LeakyRelu":
                    v = np.where(v >= 0, v, alpha * v)
                elif act == "Tanh":
                    v = np.tanh(v)
                elif act == "Sigmoid":
                    v = expit(v)
                elif act == "Clip":
                    v = np.clip(v, alpha, beta)
                elif act == "HardSigmoid":
                    v = np.clip(alpha * v + beta, 0, 1)
            return result(typed(v))
        if ms and name == "BiasAdd":
            return result(typed(values[0] + values[1] + (values[2] if len(values) > 2 else 0)))
        if ms and name in {"BiasGelu", "FastGelu"}:
            return result(
                typed(gelu(values[0] + (values[1] if len(values) > 1 else 0), name == "FastGelu"))
            )
        if ms and name == "GatedAdd":
            return result(typed(values[0] + typed(values[1] * values[2])))
        if ms and name == "BiasSplitGelu":
            v = values[0] + (values[1] if len(values) > 1 else 0)
            left, right = np.split(v, 2, axis=-1)
            return result(typed(left * gelu(right)))
    raise NotImplementedError("Independent tensor oracle not implemented: " + operation)
