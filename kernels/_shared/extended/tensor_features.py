# SPDX-License-Identifier: Apache-2.0
"""Independent numerical and failure-contract probes for tensor Vulkan kernels."""

import numpy as np


def features():
    cases = []

    def add(operation, name, inputs, expected, attrs=None, args=None):
        cases.append(
            dict(
                operation=operation,
                name=name,
                inputs=inputs,
                expected=expected,
                attrs=attrs or {},
                args=args or {},
            )
        )

    for operation in ["ai.onnx.ScatterElements", "ai.onnx.ScatterND"]:
        data = np.array([2**40 + 7, -(2**45), 2**35], dtype=np.int64)
        updates = np.array([2**39 + 1, -(2**39) - 4, 9], dtype=np.int64)
        indices = (
            np.array([0, 0, 2], dtype=np.int32)
            if operation.endswith("Elements")
            else np.array([[0], [0], [2]], dtype=np.int64)
        )
        for reduction in ["none", "add", "mul", "max", "min"]:
            expected = data.copy()
            with np.errstate(over="ignore"):
                for at, value in zip([0, 0, 2], updates):
                    if reduction == "none":
                        expected[at] = value
                    elif reduction == "add":
                        expected[at] += value
                    elif reduction == "mul":
                        expected[at] *= value
                    elif reduction == "max":
                        expected[at] = max(expected[at], value)
                    else:
                        expected[at] = min(expected[at], value)
            add(
                operation,
                "int64_full_width_" + reduction,
                dict(data=data, indices=indices, updates=updates),
                dict(output=expected),
                dict(reduction=reduction),
            )
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = np.array([0, 3, -4, -1], dtype=np.int32)
    add(
        "ai.onnx.Gather",
        "published_zero_fill_both_bounds",
        dict(data=data, indices=indices),
        dict(output=np.stack([data[0], np.zeros(4), np.zeros(4), data[-1]]).astype(np.float32)),
    )
    data = np.arange(24, dtype=np.int64).reshape(2, 3, 4) * (2**35 + 1)
    indices = np.array([[[1, 0]], [[0, 2]]], dtype=np.int64)
    add(
        "ai.onnx.GatherND",
        "int64_payload_and_indices",
        dict(data=data, indices=indices),
        dict(output=np.stack([data[1, 0], data[0, 2]])[:, None, :]),
    )
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    indices = np.array([[[2, 1], [0, -1]]], dtype=np.int32)
    expected = np.empty(indices.shape, np.float32)
    for at in np.ndindex(indices.shape):
        source = list(at)
        source[1] = indices[at]
        expected[at] = data[tuple(source)]
    add(
        "ai.onnx.GatherElements",
        "non_axis_subset_dimensions",
        dict(data=data, indices=indices),
        dict(output=expected),
        dict(axis=1),
    )
    for k in [2**32, -(2**32), 1, -1]:
        add(
            "ai.onnx.Trilu",
            "full_width_k_" + str(k),
            dict(input=np.ones((3, 4), np.float32), k=np.array(k, np.int64)),
            dict(output=np.triu(np.ones((3, 4), np.float32), k=k)),
        )
    for dtype, value in [(np.uint32, 2**32 - 3), (np.int32, -(2**30) + 1), (np.bool_, True)]:
        add(
            "ai.onnx.ConstantOfShape",
            "exact_scalar_fill_" + np.dtype(dtype).name,
            dict(input=np.array([2, 3], np.int64)),
            dict(output=np.full((2, 3), value, dtype=dtype)),
            dict(value=int(value)),
        )
    add(
        "ai.onnx.ConstantOfShape",
        "typed_fill_literal",
        dict(input=np.array([2, 2], np.int64)),
        dict(output=np.full((2, 2), -321, np.int16)),
        dict(value=dict(dtype="int16", shape=[1], values=[-321])),
    )
    add(
        "ai.onnx.Unique",
        "requested_inverse_without_indices",
        dict(x=np.array([9, 1, 9, 4, 1], np.int32)),
        dict(y=np.array([1, 4, 9], np.int32), inverse_indices=np.array([2, 0, 2, 1, 0], np.int64)),
    )
    add(
        "ai.onnx.Unique",
        "requested_counts_without_inverse",
        dict(x=np.array([9, 1, 9, 4, 1], np.int32)),
        dict(y=np.array([1, 4, 9], np.int32), counts=np.array([2, 1, 2], np.int64)),
    )
    for alpha, beta in [(0, 1), (1, 0)]:
        a = np.eye(2, dtype=np.float32) if alpha else np.full((2, 2), np.inf, np.float32)
        b = np.eye(2, dtype=np.float32)
        bias = np.array([3, 7], np.float32) if beta else np.array([np.inf, np.nan], np.float32)
        expected = np.broadcast_to(bias, (2, 2)).copy() if alpha == 0 else a.copy()
        add(
            "ai.onnx.Gemm",
            f"zero_coefficient_{alpha}_{beta}",
            dict(a=a, b=b, bias=bias),
            dict(y=expected),
            dict(alpha=alpha, beta=beta),
        )
    for operation in ["ai.onnx.Celu", "ai.onnx.Elu", "ai.onnx.Selu"]:
        x = np.array([-1e-39, -1e-40, 0, 1e-40], np.float32)
        if operation.endswith("Selu"):
            attrs = dict(alpha=1.6732, gamma=1.0507)
            expected = np.float32(1.0507) * np.where(x > 0, x, np.float32(1.6732) * np.expm1(x))
        else:
            attrs = dict(alpha=1)
            expected = np.where(x >= 0, x, np.expm1(x))
        add(operation, "subnormal_tail", dict(x=x), dict(y=expected), attrs)
    for capacity, count, condition in [(5, 2, True), (2, 8, True), (3, 2, False), (0, 5, True)]:
        initial = np.array([3, -4], np.float32)
        step = np.array([0.25, 2], np.float32)
        scan = np.zeros((capacity, 2), np.float32)
        state = initial.copy()
        for t in range(min(capacity, count) if condition else 0):
            state += step
            scan[t] = state
        add(
            "ai.onnx.Loop",
            f"scan_capacity_{capacity}_count_{count}_{condition}",
            dict(
                m=np.array(count, np.uint32),
                cond=np.array(condition, np.bool_),
                v_initial=initial,
                step=step,
            ),
            dict(v_final=state, scan_output=scan),
        )
    source = np.arange(12, dtype=np.float32).reshape(4, 3)
    scan = np.cumsum(source[::-1], axis=0)[::-1].copy()
    add(
        "ai.onnx.Scan",
        "reverse_output_matches_input_row",
        dict(initial_state=np.zeros(3, np.float32), scan_input=source),
        dict(final_state=scan[0], scan_output=scan),
        dict(reverse=1),
    )
    for dtype in [np.int8, np.int16, np.int32]:
        a = np.array([-4, 7, 5, 4, -7, 8], dtype=dtype)
        b = np.array([2, -3, 8, -2, 3, 5], dtype=dtype)
        for fmod in [0, 1]:
            expected = (np.fmod if fmod else np.remainder)(a, b)
            add(
                "ai.onnx.Mod",
                f"signed_{np.dtype(dtype).name}_{fmod}",
                dict(a=a, b=b),
                dict(c=expected),
                dict(fmod=fmod),
            )
    a = np.array([1e20, -1e20, 1e30, -1e30, 2**27 + 16, -(2**27 + 16)], np.float32)
    b = np.array([3, 7, -11, 9, 5, -3], np.float32)
    for fmod in [0, 1]:
        add(
            "ai.onnx.Mod",
            f"large_dividend_{fmod}",
            dict(a=a, b=b),
            dict(c=(np.fmod if fmod else np.remainder)(a, b)),
            dict(fmod=fmod),
        )
    a = np.array([np.nan, 1, 3, 4], np.float32)
    b = np.array([4, np.nan, 2, 7], np.float32)
    for op in ["Min", "Max"]:
        add(
            "ai.onnx." + op,
            "nan_survives_remaining_variadic_inputs",
            dict(a=a, b=b, c=np.array(2, np.float32)),
            dict(
                y=(np.minimum if op == "Min" else np.maximum)(
                    (np.minimum if op == "Min" else np.maximum)(a, b), 2
                )
            ),
        )
    rng = np.random.default_rng(731)
    for kind, axis in [(np.int32, -1), (np.uint32, 0), (np.float16, -1)]:
        shape = (1025, 2) if axis == 0 else (2, 1025)
        if kind == np.uint32:
            data = rng.integers(2**31, 2**32, size=shape, dtype=np.uint32)
        elif kind == np.int32:
            data = rng.integers(-(2**30), 2**30, size=shape, dtype=np.int32)
        else:
            data = rng.normal(size=shape).astype(kind)
            data.flat[5] = np.nan
            data.flat[8] = np.nan
        order = np.argsort(-data.astype(np.float64), axis=axis, kind="stable")
        part = [slice(None)] * data.ndim
        part[axis] = slice(0, 7)
        order = order[tuple(part)]
        add(
            "ai.onnx.TopK",
            f"parallel_sort_{np.dtype(kind).name}_axis{axis}",
            dict(x=data),
            dict(values=np.take_along_axis(data, order, axis), indices=order.astype(np.int64)),
            dict(axis=axis),
            dict(k=7),
        )
    values = np.full(500000, 3, np.int32)
    add(
        "ai.onnx.TopK",
        "parallel_large_ties_stable_indices",
        dict(x=values),
        dict(values=np.full(256, 3, np.int32), indices=np.arange(256, dtype=np.int64)),
        args=dict(k=256),
    )
    add(
        "ai.onnx.Sum",
        "fifteen_input_native_heap_extension",
        {chr(ord("a") + j): np.full((2, 3), j + 1, np.float32) for j in range(15)},
        dict(y=np.full((2, 3), 120, np.float32)),
    )
    for name in ["com.microsoft.FusedMatMul", "com.microsoft.TransposeMatMul"]:
        a = np.arange(3, dtype=np.float32)
        b = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        add(name, "left_vector_batched_right", dict(A=a, B=b), dict(Y=np.matmul(a, b)))
        add(name, "vector_dot_scalar", dict(A=a, B=a), dict(Y=np.matmul(a, a)))
    add(
        "ai.onnx.Pad",
        "large_crop_empty_output_with_large_secondary_dimension",
        dict(
            data=np.resize(np.arange(1, 7, dtype=np.float32), (2, 18, 4)),
            constant_value=np.array(100, np.float32),
        ),
        dict(output=np.empty((0, 1048594, 3), np.float32)),
        dict(pads=[1, 1048576, -2, -3, 0, 1], mode="constant"),
    )
    powers = np.array([1, -1, 2, -2, 3, -3, 1, -1], np.int32)
    add(
        "ai.onnx.Pow",
        "integer_negative_constant_exponent",
        dict(x=powers, y=np.array([-1], np.int32)),
        dict(z=np.power(powers.astype(np.float32), -1).astype(np.int32)),
    )
    for case in cases:
        case["atol"] = 0
        case["rtol"] = 0
    return cases
