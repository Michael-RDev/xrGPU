"""Host semantic and shader compilation checks; GPU execution is still required."""

import json
from pathlib import Path

import numpy as np
import pytest

from kernels.experimental.attention.build import build
from kernels.experimental.attention.reference import attention, parameters

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def torch():
    module = pytest.importorskip("torch")
    module.set_num_threads(1)
    return module


@pytest.mark.parametrize("shape", [(1, 6, 32, 32, 64, 64), (1, 6, 128, 128, 64, 64),
                                  (2, 3, 7, 11, 13, 17), (1, 1, 3, 256, 128, 128),
                                  (1, 1, 1, 1, 1, 1)])
@pytest.mark.parametrize("mode", ["none", "batch", "broadcast", "large", "all_masked"])
def test_reference_matches_pytorch_sdpa(torch, shape, mode):
    b, h, m, s, d, dv = shape
    rng = np.random.default_rng(41731)
    q, k, v = [rng.normal(size=sh).astype(np.float32)
               for sh in ((b, h, m, d), (b, h, s, d), (b, h, s, dv))]
    mask = None
    if mode != "none":
        mask = np.zeros((1 if mode == "broadcast" else b, 1, 1, s), dtype=np.float32)
        mask[..., -(s // 3 or 1):] = -10000
        if mode == "all_masked":
            mask.fill(-10000)  # Finite additive masks retain ordinary softmax semantics.
    if mode == "large":
        q *= 100
        k *= 100
    kwargs = dict(attn_mask=None if mask is None else torch.from_numpy(mask).double())
    expected = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(q).double(), torch.from_numpy(k).double(),
        torch.from_numpy(v).double(), **kwargs).numpy()
    np.testing.assert_allclose(attention(q, k, v, mask), expected, atol=1e-4, rtol=1e-4)
    push = parameters(q, k, v, mask)
    assert push[:6] == (m, s, d, dv, h, b)
    assert push[6:] == (s if mask is not None and mask.shape[0] > 1 else 0, int(mask is not None))


@pytest.mark.parametrize("bad", ["dtype", "noncontiguous", "infinite", "mask_shape",
                                 "mask_infinite", "key_length", "head_dim", "mismatch"])
def test_rejects_inputs_outside_shader_contract(bad):
    q = np.zeros((2, 3, 7, 13), dtype=np.float32)
    k, v = q.copy(), q.copy()
    mask = None
    if bad == "dtype":
        q = q.astype(np.float64)
    elif bad == "noncontiguous":
        q = q[..., ::-1]
    elif bad == "infinite":
        q.flat[0] = np.inf
    elif bad == "mask_shape":
        mask = np.zeros((2, 3, 7, 7), dtype=np.float32)
    elif bad == "mask_infinite":
        mask = np.full((1, 1, 1, 7), -np.inf, dtype=np.float32)
    elif bad == "key_length":
        k = np.zeros((2, 3, 257, 13), dtype=np.float32)
        v = k.copy()
    elif bad == "head_dim":
        q = np.zeros((2, 3, 7, 129), dtype=np.float32)
        k = q.copy()
    elif bad == "mismatch":
        v = v[:, :2].copy()
    with pytest.raises(ValueError):
        parameters(q, k, v, mask)


def test_builds_three_variants_and_discovery_fixtures(tmp_path):
    compiler = ROOT / "artifacts/toolchain/shaderc/2026.4/bin/glslc"
    if not compiler.is_file():
        pytest.skip("Pinned local shader compiler unavailable")
    output = tmp_path / "attention"
    result = build(output, compiler)
    assert result["variants"] == 3
    assert result["fixtures"] == 14
    manifest = json.loads((output / "manifest.json").read_text())
    contract = json.loads((output / "contract.json").read_text())
    assert manifest["status"] == "unqualified"
    assert manifest["runtime_integration"] is False
    assert manifest["gpu_correctness_validated"] is False
    assert manifest["performance_validated"] is False
    assert set(f["seed"] for f in manifest["fixtures"]) == set(contract["discovery_seeds"])
    assert not set(contract["discovery_seeds"]) & set(contract["confirmation_seeds"])
    for fixture in manifest["fixtures"]:
        with np.load(output / fixture["file"]) as tensors:
            mask = tensors["mask"] if fixture["push_constants"][-1] else None
            np.testing.assert_array_equal(
                attention(tensors["q"], tensors["k"], tensors["v"], mask), tensors["expected"])
            assert list(parameters(tensors["q"], tensors["k"], tensors["v"], mask)) == fixture["push_constants"]
    with pytest.raises(FileExistsError):
        build(output, compiler)
