# SPDX-License-Identifier: Apache-2.0
"""Independent parameterized CPU oracles for the original spatial Vulkan kernels."""

import numpy as np

from .verify_spatial import tensor


def cases():
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(1)
    rng = np.random.default_rng(72391)
    result = []

    def add(name, inputs, outputs, attrs=None, args=None, label="parameters"):
        result.append(
            dict(
                operation=name if "." in name else "ai.onnx." + name,
                case=label,
                inputs=[tensor(n, a) for n, a in inputs],
                expected={n: np.asarray(a) for n, a in outputs},
                attrs=attrs or {},
                args=args or {},
            )
        )

    def random(shape):
        return rng.uniform(-1, 1, shape).astype(np.float32)

    def torch_array(a):
        return torch.from_numpy(a.copy())

    def array(a):
        return a.detach().numpy().astype(np.float32)

    x = random((1, 2, 6))
    w = random((4, 1, 3))
    b = random((4,))
    y = array(
        functional.conv1d(
            functional.pad(torch_array(x), (2, 1)),
            torch_array(w),
            torch_array(b),
            stride=2,
            dilation=2,
            groups=2,
        )
    )
    add(
        "Conv",
        [("x", x), ("w", w), ("bias", b)],
        [("y", y)],
        dict(auto_pad="SAME_LOWER", group=2, strides=[2], dilations=[2]),
        label="grouped_dilated_same_lower_1d",
    )
    x = random((1, 4, 4, 5))
    w = random((4, 2, 2, 2))
    b = random((4,))
    y = array(
        functional.conv2d(
            functional.pad(torch_array(x), (0, 1, 1, 0)),
            torch_array(w),
            torch_array(b),
            stride=(1, 2),
            groups=2,
        )
    )
    add(
        "Conv",
        [("x", x), ("w", w), ("bias", b)],
        [("y", y)],
        dict(group=2, strides=[1, 2], pads=[1, 0, 0, 1]),
        label="asymmetric_grouped_2d",
    )
    x = random((1, 2, 3, 4))
    w = random((2, 3, 3, 3))
    b = random((3,))
    y = array(
        functional.conv_transpose2d(
            torch_array(x), torch_array(w), torch_array(b), stride=2, padding=1, output_padding=1
        )
    )
    add(
        "ConvTranspose",
        [("x", x), ("w", w), ("bias", b)],
        [("y", y)],
        dict(strides=[2, 2], pads=[1, 1, 1, 1], output_padding=[1, 1]),
        label="transpose_output_padding",
    )
    x = random((1, 1, 3, 3))
    w = random((1, 1, 2, 2))
    b = random((1,))
    conv = array(functional.conv2d(torch_array(x), torch_array(w), torch_array(b)))
    y = conv * np.clip((conv + 3) / 6, 0, 1)
    add(
        "com.microsoft.FusedConv",
        [("x", x), ("w", w), ("bias", b)],
        [("y", y)],
        dict(activation="HardSwish"),
        label="hard_swish_fusion",
    )
    x = np.arange(24, dtype=np.int8).reshape(1, 2, 3, 4) - 7
    w = np.array([[[[1, -2], [2, 1]]], [[[-1, 2], [1, -2]]]], np.int8)
    bias = np.array([2, -1], np.int32)
    z = np.array(2, np.int8)
    wz = np.array([-1, 1], np.int8)
    acc = np.empty((1, 2, 2, 3), np.int32)
    for c in range(2):
        for yy in range(2):
            for xx in range(3):
                acc[0, c, yy, xx] = np.sum(
                    (x[0, c, yy : yy + 2, xx : xx + 2].astype(np.int64) - 2)
                    * (w[c, 0].astype(np.int64) - int(wz[c]))
                )
    add(
        "ConvInteger",
        [("x", x), ("w", w), ("x_zero_point", z), ("w_zero_point", wz)],
        [("y", acc)],
        dict(group=2),
        label="depthwise_signed_zero_points",
    )
    xs = np.array(0.25, np.float32)
    ws = np.array([0.5, 0.125], np.float32)
    ys = np.array(0.2, np.float32)
    yz = np.array(-3, np.int8)
    y = np.clip(
        np.rint(
            (acc.astype(np.float64) + bias[None, :, None, None])
            * float(xs)
            * ws.astype(np.float64)[None, :, None, None]
            / float(ys)
        )
        + int(yz),
        -128,
        127,
    ).astype(np.int8)
    common = [
        ("x", x),
        ("w", w),
        ("B", bias),
        ("x_zero_point", z),
        ("w_zero_point", wz),
        ("x_scale", xs),
        ("w_scale", ws),
        ("y_scale", ys),
        ("y_zero_point", yz),
    ]
    add("QLinearConv", common, [("y", y)], dict(group=2), label="per_channel_quantization")
    nhwc = [(n, a.transpose(0, 2, 3, 1) if n == "x" else a) for n, a in common]
    add(
        "com.microsoft.QLinearConv",
        nhwc,
        [("y", y.transpose(0, 2, 3, 1))],
        dict(group=2, channels_last=1),
        label="nhwc_depthwise_per_channel_quantization",
    )
    x = random((1, 2, 4, 5))
    tx = torch_array(x)
    y = array(
        functional.avg_pool2d(
            tx, kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=True
        )
    )
    add(
        "AveragePool",
        [("x", x)],
        [("y", y)],
        dict(
            kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], ceil_mode=1, count_include_pad=1
        ),
        label="ceil_include_pad",
    )
    y, indices = functional.max_pool2d(tx, 2, stride=1, dilation=2, return_indices=True)
    add(
        "MaxPool",
        [("x", x)],
        [
            ("y", array(y)),
            (
                "indices",
                indices.numpy().astype(np.int64)
                + np.arange(x.shape[0] * x.shape[1], dtype=np.int64).reshape(
                    x.shape[0], x.shape[1], 1, 1
                )
                * np.prod(x.shape[2:]),
            ),
        ],
        dict(kernel_shape=[2, 2], dilations=[2, 2]),
        label="dilated_indices",
    )
    x = random((1, 2, 2, 3))
    grid = np.array([[[[-1.4, -1.2], [-0.3, 0.25], [1.2, 1.4]]]], np.float32)
    for mode in ["bilinear", "nearest", "bicubic"]:
        for padding in ["zeros", "border", "reflection"]:
            y = array(
                functional.grid_sample(
                    torch_array(x),
                    torch_array(grid),
                    mode=mode,
                    padding_mode=padding,
                    align_corners=False,
                )
            )
            add(
                "GridSample",
                [("x", x), ("grid", grid)],
                [("y", y)],
                dict(
                    mode={"bilinear": "linear", "nearest": "nearest", "bicubic": "cubic"}[mode],
                    padding_mode=padding,
                ),
                label=mode + "_" + padding,
            )
    x = random((1, 1, 2, 3, 2))
    grid = np.array([[[[[-1.2, 0.2, -0.4], [0.3, -1.4, 1.2]]]]], np.float32)
    y = array(
        functional.grid_sample(
            torch_array(x),
            torch_array(grid),
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )
    )
    add(
        "GridSample",
        [("x", x), ("grid", grid)],
        [("y", y)],
        dict(mode="linear", padding_mode="reflection", align_corners=1),
        label="trilinear_reflection",
    )
    x = random((1, 2, 3, 4))
    for mode in ["bilinear", "bicubic"]:
        y = array(
            functional.interpolate(torch_array(x), scale_factor=2, mode=mode, align_corners=False)
        )
        add(
            "Resize",
            [("x", x)],
            [("y", y)],
            dict(scales=[1.0, 1.0, 2.0, 2.0], mode="linear" if mode == "bilinear" else "cubic"),
            label=mode + "_half_pixel",
        )
    x = random((1, 1, 5, 7))
    y = array(
        functional.interpolate(
            torch_array(x), size=(2, 3), mode="bilinear", align_corners=False, antialias=True
        )
    )
    add(
        "Resize",
        [("x", x)],
        [("y", y)],
        dict(scales=[1.0, 1.0, 0.4, 3 / 7], mode="linear", antialias=1, exclude_outside=1),
        label="antialiased_downsample",
    )
    x = np.arange(7, dtype=np.float32).reshape(1, 1, 1, 7)
    add(
        "Resize",
        [("x", x)],
        [("y", np.array([[[[0, 4]]]], np.float32))],
        dict(
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="round_prefer_ceil",
        ),
        label="inferred_ratio_nearest_tie",
    )
    x = random((2, 3, 4))
    s = random((3, 4))
    b = random((1, 3, 1))
    m = x.mean(axis=(1, 2), keepdims=True)
    inv = 1 / np.sqrt(x.var(axis=(1, 2), keepdims=True) + 1e-5)
    add(
        "LayerNormalization",
        [("x", x), ("scale", s), ("b", b)],
        [("y", (x - m) * inv * s + b), ("mean", m), ("invStdDev", inv)],
        dict(axis=1),
        label="suffix_axis_and_broadcast_affine",
    )
    x = random((1, 4, 2, 3))
    s = random((4,))
    b = random((4,))
    y = array(functional.group_norm(torch_array(x), 2, torch_array(s), torch_array(b)))
    add(
        "GroupNormalization",
        [("x", x), ("scale", s), ("bias", b)],
        [("y", y)],
        dict(num_groups=2),
        label="channel_affine_groups",
    )
    x = random((2, 3, 4))
    skip = random((3, 4))
    s = random((4,))
    b = random((4,))
    res = x + skip
    m = res.mean(axis=2, keepdims=True)
    inv = 1 / np.sqrt(res.var(axis=2, keepdims=True) + 1e-12)
    add(
        "com.microsoft.SkipLayerNormalization",
        [("inputT", x), ("skipT", skip), ("gammaT", s), ("betaT", b)],
        [("outputT", (res - m) * inv * s + b)],
        label="skip_batch_broadcast",
    )
    x = random((2, 4))
    axis = 0
    values = np.exp(x - x.max(axis=axis, keepdims=True))
    values /= values.sum(axis=axis, keepdims=True)
    add("Softmax", [("x", x)], [("y", values)], dict(axis=axis), label="strided_axis_zero")
    y = np.flip(np.cumsum(np.flip(x, axis=1), axis=1), axis=1) - x
    add(
        "CumSum",
        [("x", x)],
        [("y", y)],
        dict(exclusive=1, reverse=1),
        dict(axis=1),
        label="exclusive_reverse",
    )
    add(
        "ReduceSum",
        [("x", x)],
        [("y", x.copy())],
        dict(axes=[], noop_with_empty_axes=1, keepdims=0),
        label="empty_axes_noop",
    )
    x = random((2, 3, 4))
    y = np.sum(x, axis=(0, 2))
    add(
        "ReduceSum",
        [("x", x)],
        [("y", y)],
        dict(axes=[0, 2], keepdims=0),
        label="multiple_axes_remove_dims",
    )
    x = np.array([[3, 3, 1], [2, 2, 2]], np.float32)
    add(
        "ArgMax",
        [("x", x)],
        [("y", np.array([1, 2], np.int64))],
        dict(axis=1, select_last_index=1, keepdims=0),
        label="last_tie_indices",
    )
    real = random((1, 5))
    fft = np.fft.rfft(real, axis=1)
    packed = np.stack((fft.real, fft.imag), axis=-1).astype(np.float32)
    irfft = np.fft.irfft(packed[..., 0] + 1j * packed[..., 1], n=5, axis=1).astype(np.float32)[
        ..., None
    ]
    add(
        "DFT",
        [("input", packed), ("dft_length", np.array(5, np.int32))],
        [("output", irfft)],
        dict(inverse=1, onesided=1),
        label="inverse_real_odd_length",
    )
    z = random((3, 2)) + 1j * random((3, 2))
    packed = np.stack((z.real, z.imag), axis=-1).astype(np.float32)
    answer = np.fft.ifft(z, axis=0)
    out = np.stack((answer.real, answer.imag), axis=-1).astype(np.float32)
    add(
        "DFT",
        [("input", packed), ("axis", np.array(0, np.int64))],
        [("output", out)],
        dict(inverse=1),
        label="arbitrary_axis_complex_inverse",
    )
    signal = random((1, 6, 2))
    z = signal[..., 0] + 1j * signal[..., 1]
    fft = np.stack([np.fft.fft(z[:, i : i + 4], axis=1) for i in [0, 2]], axis=1)
    answer = np.stack((fft.real, fft.imag), axis=-1).astype(np.float32)
    add(
        "STFT",
        [("signal", signal)],
        [("output", answer)],
        dict(onesided=0),
        dict(frameStep=2, frameLength=4),
        label="complex_no_window_full_spectrum",
    )
    for name in ["HannWindow", "HammingWindow", "BlackmanWindow"]:
        add(
            name,
            [("size", np.array(1, np.int32))],
            [("y", np.array([np.nan], np.float32))],
            dict(periodic=0),
            label="symmetric_size_one_nan",
        )
    integers = np.array(
        [[2**30 + 3, 2**30 + 4, -(2**30)], [-2147483648, -2147483647, -2147483646]], np.int32
    )
    add(
        "ArgMax",
        [("x", integers)],
        [("y", np.argmax(integers, axis=1).astype(np.int64))],
        dict(axis=1, keepdims=0),
        label="exact_i32_beyond_float32",
    )
    add(
        "ArgMin",
        [("x", integers)],
        [("y", np.argmin(integers, axis=1).astype(np.int64))],
        dict(axis=1, keepdims=0),
        label="exact_i32_beyond_float32",
    )
    for name, values, reference in [
        ("ReduceSum", [2**30, 2**30 - 2], 2147483646),
        ("ReduceSum", [2**30, 2**30], 2147483647),
        ("ReduceMean", [2147483647, 2147483646], 2147483646),
        ("ReduceL1", [-2147483648], 2147483647),
        ("ReduceProd", [2147483647] * 6, 2147483647),
    ]:
        add(
            name,
            [("x", np.array(values, np.int32))],
            [("y", np.array(reference, np.int32))],
            dict(axes=[0], keepdims=0),
            label="widened_saturated_integer",
        )
    x = random((2, 2, 6))
    gate = random(x.shape) * 4
    scale = np.array([0.7, 1.3, 1.1], np.float32)
    grouped = x.astype(np.float64).reshape(-1, scale.size)
    y = (
        grouped / np.sqrt(np.mean(grouped * grouped, axis=1, keepdims=True) + 1e-5) * scale
    ).reshape(x.shape)
    y = (y * gate / (1 + np.exp(-gate.astype(np.float64)))).astype(np.float32)
    for shape in [(3,), (1, 3), (3, 1)]:
        add(
            "com.microsoft.GatedRMSNorm",
            [("xT", x), ("scaleT", scale.reshape(shape)), ("gateT", gate)],
            [("yT", y)],
            label="per_head_scale_layout_" + str(shape),
        )
    return result
