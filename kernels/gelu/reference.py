"""Independent float64 oracle; inputs and results use PyTorch GELU semantics."""
import math
import numpy as np


def gelu(x, approximate='none'):
    x = np.asarray(x, dtype=np.float64)
    if approximate == 'none':
        return .5 * x * (1 + np.vectorize(math.erf, otypes=[float])(x / math.sqrt(2)))
    if approximate == 'tanh':
        return .5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + .044715 * x**3)))
    raise ValueError('approximate must be none or tanh')
