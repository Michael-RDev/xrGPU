"""Scalar-channel NumPy oracle, independent of shader packing."""
import numpy as np


def conv2d(x, weight, bias=None, groups=1, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    x, weight = np.asarray(x, dtype=np.float64), np.asarray(weight, dtype=np.float64)
    if x.ndim != 4 or weight.ndim != 4 or x.shape[0] != 1 or groups < 1:
        raise ValueError('Expected batch-one NCHW and OIHW tensors')
    n, ci, h, w = x.shape
    co, cig, kh, kw = weight.shape
    if ci % groups or co % groups or cig != ci // groups:
        raise ValueError('Invalid groups or weight channels')
    if min(*stride, *dilation) < 1 or min(padding) < 0:
        raise ValueError('Invalid convolution geometry')
    oh = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    if min(oh, ow) < 1 or (bias is not None and np.shape(bias) != (co,)):
        raise ValueError('Invalid output shape or bias')
    padded = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
    out = np.zeros((n, co, oh, ow), dtype=np.float64)
    for oc in range(co):
        base = (oc // (co // groups)) * cig
        for ic in range(cig):
            for ky in range(kh):
                for kx in range(kw):
                    out[:, oc] += weight[oc, ic, ky, kx] * padded[:, base + ic,
                        ky*dilation[0]:ky*dilation[0]+oh*stride[0]:stride[0],
                        kx*dilation[1]:kx*dilation[1]+ow*stride[1]:stride[1]]
    return out + (0 if bias is None else np.asarray(bias)[None, :, None, None])
