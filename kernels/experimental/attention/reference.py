"""Host contract and independent mathematical reference; does not execute GLSL."""

import numpy as np


def parameters(q, k, v, mask=None):
    tensors = (q, k, v) + (() if mask is None else (mask,))
    if any(x.dtype != np.float32 or not x.flags.c_contiguous or x.ndim != 4 for x in tensors):
        raise ValueError("Contiguous rank-four FP32 inputs required")
    if any(not np.isfinite(x).all() or np.any(np.abs(x) > 1000) for x in (q, k, v)):
        raise ValueError("Q/K/V must be finite with absolute values <= 1000")
    b, h, m, d = q.shape
    if k.shape[:2] != (b, h) or v.shape[:3] != k.shape[:3] or k.shape[3] != d:
        raise ValueError("Incompatible attention shapes")
    s, dv = k.shape[2], v.shape[3]
    if not (b > 0 and h > 0 and 1 <= m <= 256 and 1 <= s <= 256
            and 1 <= d <= 128 and 1 <= dv <= 128 and b * h <= 65535):
        raise ValueError("Attention shape exceeds prototype limits")
    if mask is not None and (mask.shape not in ((1, 1, 1, s), (b, 1, 1, s))
                             or not np.isfinite(mask).all() or np.any(np.abs(mask) > 1e6)):
        raise ValueError("Finite per-key additive mask [1 or B,1,1,S] with abs <= 1e6 required")
    storage = sum(x.nbytes for x in tensors) + b * h * m * dv * 4 + (4 if mask is None else 0)
    if storage > 64 * 1024 * 1024:
        raise ValueError("Attention storage exceeds 64 MiB")
    return (m, s, d, dv, h, b, s if mask is not None and mask.shape[0] > 1 else 0,
            int(mask is not None))


def attention(q, k, v, mask=None):
    parameters(q, k, v, mask)
    scores = (q.astype(np.float64) @ k.astype(np.float64).swapaxes(-1, -2)) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores += mask
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return (weights @ v.astype(np.float64)).astype(np.float32)
