import copy

import pytest


@pytest.mark.parametrize("attention,packed,fuse_norm", [("eager", False, False), ("sdpa", False, False), ("sdpa", True, False), ("flash", True, False), ("flash", True, True)])
def test_all_checkpoints_match_original_bert(attention, packed, fuse_norm):
    torch = pytest.importorskip("torch")
    from transformers import BertConfig, BertModel
    from models.optimized import OptimizedMiniLM

    torch.manual_seed(19)
    torch.set_num_threads(1)
    config = BertConfig(
        hidden_size=96,
        intermediate_size=384,
        num_attention_heads=3,
        num_hidden_layers=6,
        vocab_size=100,
        max_position_embeddings=32,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    config._attn_implementation = "eager"
    original = BertModel(config).eval()
    expected = []
    handles = [original.embeddings.register_forward_hook(lambda m, args, out: expected.append(out))]
    for layer in original.encoder.layer:
        for module in (
            layer.attention.self.query,
            layer.attention.self.key,
            layer.attention.self.value,
        ):
            handles.append(module.register_forward_hook(lambda m, args, out: expected.append(out)))
        handles.append(
            layer.attention.output.dense.register_forward_pre_hook(
                lambda m, args: expected.append(args[0])
            )
        )
        for module in (
            layer.attention.output.dense,
            layer.attention.output.LayerNorm,
            layer.intermediate.dense,
        ):
            handles.append(module.register_forward_hook(lambda m, args, out: expected.append(out)))
        handles.append(
            layer.output.dense.register_forward_pre_hook(lambda m, args: expected.append(args[0]))
        )
        for module in (layer.output.dense, layer.output.LayerNorm):
            handles.append(module.register_forward_hook(lambda m, args, out: expected.append(out)))
    ids = torch.randint(0, 100, (1, 16))
    mask = torch.tensor([[1] * 9 + [0] * 7])
    types = torch.tensor([[0] * 8 + [1] * 8])
    with torch.inference_mode():
        hidden = original(ids, mask, types).last_hidden_state
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        expected.append(torch.nn.functional.normalize(pooled, dim=1))
    for handle in handles:
        handle.remove()
    candidate = OptimizedMiniLM(
        copy.deepcopy(original), 16, attention=attention, packed_qkv=packed, audit=True, fuse_norm=fuse_norm
    ).eval()
    with torch.inference_mode():
        actual = candidate(ids, mask, types)
    oracle = torch.cat([x.flatten() for x in expected])
    assert len(expected) == len(candidate.checkpoint_layout()) == 62
    torch.testing.assert_close(actual, oracle, rtol=1e-4, atol=1e-4)
