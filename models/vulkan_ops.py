"""Private Vulkan operators used only by optimized MiniLM programs."""

import torch
from executorch.backends.vulkan.op_registry import OpFeatures, update_features
from executorch.backends.vulkan import utils
from executorch.exir.dialects._ops import ops as exir_ops

LIBRARY = torch.library.Library("xrgpu", "DEF")
LIBRARY.define("gelu(Tensor x) -> Tensor")
LIBRARY.impl(
    "gelu", lambda x: torch.nn.functional.gelu(x, approximate="none"), "CompositeExplicitAutograd"
)


@update_features([torch.ops.xrgpu.gelu.default, exir_ops.edge.xrgpu.gelu.default, "xrgpu::gelu"])
def gelu_features():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE, inputs_dtypes=utils.FP_T, supports_resize=True
    )


def packed_attention(qkv, mask, heads):
    length, packed_width = qkv.shape
    width = packed_width // 3
    q, k, v = [
        x.reshape(1, length, heads, width // heads).transpose(1, 2)
        for x in qkv.split(width, dim=-1)
    ]
    scores = (q @ k.transpose(-2, -1)) * ((width // heads) ** -0.5)
    scores = scores + (1.0 - mask.reshape(1, 1, 1, length)) * -10000.0
    return (scores.softmax(-1) @ v).transpose(1, 2).reshape(length, width)


LIBRARY.define("attention(Tensor qkv, Tensor mask, int heads) -> Tensor")
LIBRARY.impl("attention", packed_attention, "CompositeExplicitAutograd")


@update_features("xrgpu::attention")
def attention_features():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_BUFFER, inputs_dtypes={torch.float32}, supports_resize=False
    )


LIBRARY.define("residual_norm(Tensor x, Tensor residual, Tensor weight, Tensor bias, float eps) -> Tensor")
LIBRARY.impl("residual_norm", lambda x, residual, weight, bias, eps:
             torch.nn.functional.layer_norm(x + residual, (x.shape[-1],), weight, bias, eps),
             "CompositeExplicitAutograd")


@update_features("xrgpu::residual_norm")
def residual_norm_features():
    return OpFeatures(inputs_storage=utils.CONTIGUOUS_BUFFER, inputs_dtypes={torch.float32},
                      supports_prepacking=True, supports_resize=False)
