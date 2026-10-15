import numpy as np


def linear(x, weight, bias):
    x, weight, bias = (np.asarray(t, dtype=np.float64) for t in (x, weight, bias))
    if x.ndim != 3 or weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError("expected X[B,M,K], W[N,K], bias[N]")
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError("incompatible reduction dimension")
    return (x @ weight.T + bias).astype(np.float32)
