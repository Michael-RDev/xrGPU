import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/bundle"


@pytest.mark.skipif(not (BUNDLE / "minilm_cpu_128.pte").exists(), reason="Run models.export first")
def test_cpu_program_matches_pytorch_references():
    torch = pytest.importorskip("torch")
    runtime = pytest.importorskip("executorch.runtime").Runtime.get()
    torch.set_num_threads(1)
    fixtures = json.loads((BUNDLE / "references.json").read_text())
    for length in (32, 128, 256):
        program = runtime.load_program(BUNDLE / f"minilm_cpu_{length}.pte")
        method = program.load_method("forward")
        assert method is not None
        for case in fixtures:
            if case["sequence_length"] != length:
                continue
            ids = torch.tensor([case["input_ids"]], dtype=torch.int64)
            mask = torch.tensor([case["attention_mask"]], dtype=torch.int64)
            actual = method.execute((ids, mask, torch.zeros_like(ids)))[0].numpy()[0]
            expected = np.array(case["embedding"])
            assert actual.shape == (384,)
            assert np.isfinite(actual).all()
            assert abs(np.linalg.norm(actual) - 1) <= 0.001
            cosine = np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))
            assert cosine >= 0.999, (length, case["text"], cosine)


def test_linear_oracle_shape_validation():
    from kernels.linear.reference import linear

    assert np.allclose(linear(np.ones((1, 3, 5)), np.ones((7, 5)), np.ones(7)), 6)
    with pytest.raises(ValueError):
        linear(np.ones((1, 3, 5)), np.ones((7, 6)), np.ones(7))
