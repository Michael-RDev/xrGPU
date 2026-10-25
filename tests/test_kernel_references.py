"""Independent CPU reference checks; these do not execute Vulkan shaders."""

import json
from pathlib import Path

import numpy as np
import pytest

from kernels.gelu.reference import gelu
from kernels.grouped_conv2d.reference import conv2d
from kernels.linear.reference import linear

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "kernels/linear/test.json").read_text())
SEEDS = SPEC["seeds"] + SPEC["holdout_seeds"]


@pytest.fixture(scope="module")
def torch():
    module = pytest.importorskip("torch")
    module.set_num_threads(1)
    return module


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_reference_modes(torch, seed, approximate):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(size=513), [-20, -8, -3, 0, 3, 8, 20]])
    expected = torch.nn.functional.gelu(torch.from_numpy(x), approximate=approximate).numpy()
    np.testing.assert_allclose(gelu(x, approximate), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape,bias", [((1, 1, 384, 17), True), ((1, 31, 13, 17), False)])
def test_linear_reference_tails(torch, seed, shape, bias):
    b, m, k, n = shape
    rng = np.random.default_rng(seed)
    x, weight = rng.normal(size=(b, m, k)), rng.normal(size=(n, k))
    offset = rng.normal(size=n) if bias else None
    expected = torch.nn.functional.linear(
        torch.from_numpy(x),
        torch.from_numpy(weight),
        None if offset is None else torch.from_numpy(offset),
    ).numpy()
    np.testing.assert_allclose(linear(x, weight, offset), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "ci,co,groups,stride,padding,dilation,bias",
    [
        (8, 12, 4, (1, 1), (1, 1), (1, 1), True),  # The saved failing case.
        (6, 10, 2, (2, 1), (2, 1), (2, 1), False),  # Groups cross vec4 boundaries.
        (3, 3, 3, (1, 2), (1, 1), (1, 1), True),  # Depthwise and channel tails.
        (3, 7, 1, (2, 2), (1, 1), (1, 1), False),
    ],
)
def test_grouped_convolution_reference(
    torch, seed, ci, co, groups, stride, padding, dilation, bias
):
    rng = np.random.default_rng(seed)
    x, weight = rng.normal(size=(1, ci, 15, 17)), rng.normal(size=(co, ci // groups, 3, 3))
    offset = rng.normal(size=co) if bias else None
    expected = torch.nn.functional.conv2d(
        torch.from_numpy(x),
        torch.from_numpy(weight),
        None if offset is None else torch.from_numpy(offset),
        stride,
        padding,
        dilation,
        groups,
    ).numpy()
    actual = conv2d(x, weight, offset, groups, stride, padding, dilation)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
