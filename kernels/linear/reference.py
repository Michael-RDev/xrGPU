"""Independent linear contract oracle."""
import numpy as np


def linear(x, weight, bias=None):
    x, weight = np.asarray(x, dtype=np.float64), np.asarray(weight, dtype=np.float64)
    if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[-1]:
        raise ValueError('Expected [..., M, K] input and [N, K] weight')
    if bias is not None and np.shape(bias) != (weight.shape[0],):
        raise ValueError('Bias must have N elements')
    return x @ weight.T + (0 if bias is None else np.asarray(bias, dtype=np.float64))
