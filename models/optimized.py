"""Equivalent batch-one MiniLM graphs with flat hidden states and fused attention."""

import torch
from torch import nn
from torch.nn import functional as F
from models import vulkan_ops  # noqa: F401


class OptimizedMiniLM(nn.Module):
    def __init__(self, encoder, length, *, attention="sdpa", packed_qkv=False, audit=False, fuse_norm=False):
        super().__init__()
        self.embeddings = encoder.embeddings
        self.layers = encoder.encoder.layer
        self.length = length
        self.heads = encoder.config.num_attention_heads
        self.width = encoder.config.hidden_size
        self.head_width = self.width // self.heads
        self.attention = attention
        self.packed_qkv = packed_qkv
        self.audit = audit
        self.fuse_norm = fuse_norm
        if attention == "flash" and (not packed_qkv or self.head_width != 32 or length > 256):
            raise ValueError("Packed attention requires 32-wide heads and at most 256 tokens")
        self.register_buffer("positions", torch.arange(length))
        self.qkv = nn.ModuleList()
        if packed_qkv:
            for layer in self.layers:
                projections = [
                    layer.attention.self.query,
                    layer.attention.self.key,
                    layer.attention.self.value,
                ]
                combined = nn.Linear(self.width, 3 * self.width)
                combined.weight = nn.Parameter(torch.cat([p.weight.detach() for p in projections]))
                combined.bias = nn.Parameter(torch.cat([p.bias.detach() for p in projections]))
                self.qkv.append(combined)
        if attention == "sdpa":
            import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.embeddings.word_embeddings(input_ids.reshape(self.length))
        hidden = hidden + self.embeddings.token_type_embeddings(token_type_ids.reshape(self.length))
        hidden = hidden + self.embeddings.position_embeddings(self.positions)
        hidden = self.embeddings.LayerNorm(hidden)
        checkpoints = [hidden]
        bias = (1.0 - attention_mask.to(hidden.dtype)).reshape(1, 1, 1, self.length)
        # Finite in FP16 too; exp(-10000) is zero in either precision.
        bias = bias * -10000.0
        for i, layer in enumerate(self.layers):
            if self.packed_qkv:
                projected = self.qkv[i](hidden)
                q, k, v = projected.split(self.width, dim=-1)
            else:
                q = layer.attention.self.query(hidden)
                k = layer.attention.self.key(hidden)
                v = layer.attention.self.value(hidden)
            checkpoints.extend([q, k, v])
            if self.attention == "flash":
                context = torch.ops.xrgpu.attention.default(
                    projected, attention_mask.to(hidden.dtype), self.heads
                )
            else:
                q, k, v = [
                    x.reshape(1, self.length, self.heads, self.head_width).transpose(1, 2)
                    for x in (q, k, v)
                ]
                if self.attention == "sdpa":
                    context = torch.ops.et_vk.sdpa.default(q, k, v, bias, self.head_width**-0.5)
                else:
                    scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_width**0.5
                    context = torch.matmul(torch.softmax(scores + bias, dim=-1), v)
                context = context.transpose(1, 2).reshape(self.length, self.width)
            attention_out = layer.attention.output.dense(context)
            hidden = self.residual_norm(attention_out, hidden, layer.attention.output.LayerNorm)
            up = layer.intermediate.dense(hidden)
            activated = torch.ops.xrgpu.gelu.default(up)
            down = layer.output.dense(activated)
            output = self.residual_norm(down, hidden, layer.output.LayerNorm)
            checkpoints.extend([context, attention_out, hidden, up, activated, down, output])
            hidden = output
        mask = attention_mask.reshape(self.length, 1).to(hidden.dtype)
        pooled = (hidden * mask).sum(0, keepdim=True) / mask.sum(0, keepdim=True).clamp(min=1e-9)
        embedding = F.normalize(pooled, p=2, dim=1)
        if self.audit:
            return torch.cat([x.reshape(-1) for x in checkpoints] + [embedding.reshape(-1)])
        return embedding

    def residual_norm(self, x, residual, norm):
        if self.fuse_norm:
            return torch.ops.xrgpu.residual_norm.default(x, residual, norm.weight, norm.bias, norm.eps)
        return norm(x + residual)

    def checkpoint_layout(self):
        names = [("embedding_norm", self.length * self.width)]
        for i in range(len(self.layers)):
            for name in (
                "query",
                "key",
                "value",
                "attention_context",
                "attention_projection",
                "attention_norm",
                "ffn_up",
                "gelu",
                "ffn_down",
                "output_norm",
            ):
                count = self.length * (4 * self.width if name in ("ffn_up", "gelu") else self.width)
                names.append((f"block_{i}.{name}", count))
        names.append(("normalized_embedding", self.width))
        offset, layout = 0, []
        for name, count in names:
            layout.append({"name": name, "offset": offset, "count": count})
            offset += count
        return layout
