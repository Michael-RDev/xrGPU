"""Sentence-transformers MiniLM with mean pooling and L2 normalization."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
WEIGHTS_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
SEQUENCE_LENGTHS = (32, 128, 256)
TEXTS = (
    "A handheld flashlight illuminates dark areas.",
    "Café déjà vu — naïve façade.",
    "",
    "你好，世界！",
    "one\ttwo\nthree",
    "headset " * 300,
)
FIXTURE_SEEDS = (20261025, 20261026, 20261027, 20261125, 20261126, 20261127)


def load(offline=False):
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/cache/huggingface"))
    options = dict(revision=REVISION, local_files_only=offline,
                   cache_dir=ROOT / "artifacts/cache/huggingface/hub")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
    encoder = AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager", **options).eval()
    from runtime.model_bundle import sha256
    weights = (ROOT / "artifacts/cache/huggingface/hub"
               / "models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
               / REVISION / "model.safetensors")
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("MiniLM pretrained weight digest mismatch")

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask, token_type_ids):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=token_type_ids, return_dict=False)[0]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(1, keepdim=True)
                      / mask.sum(1, keepdim=True).clamp(min=1e-9)).squeeze(1)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    return Embedding().eval(), tokenizer


def inputs(tokenizer, text, length):
    if length not in SEQUENCE_LENGTHS:
        raise ValueError("MiniLM supports sequence lengths 32, 128 and 256")
    values = tokenizer(text, padding="max_length", truncation=True, max_length=length,
                       return_tensors="pt")
    return tuple(values[key].contiguous() for key in ("input_ids", "attention_mask", "token_type_ids"))


def export(destination, *, offline=False, kernel_package=None, overwrite=False, lengths=SEQUENCE_LENGTHS):
    import tempfile
    from runtime.model_bundle import export_bundle, sha256

    model, tokenizer = load(offline)
    weight_path = (ROOT / "artifacts/cache/huggingface/hub"
                   / "models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
                   / REVISION / "model.safetensors")
    result = {}
    with tempfile.TemporaryDirectory(prefix="xrgpu-tokenizer-") as temporary:
        tokenizer.save_pretrained(temporary)
        assets = [(path, "tokenizer/" + path.name) for path in sorted(Path(temporary).iterdir()) if path.is_file()]
        for length in lengths:
            print(f"Export MiniLM sequence length {length}", flush=True)
            fixtures = [inputs(tokenizer, text, length) for text in TEXTS]
            result[f"minilm-{length}"] = export_bundle(
                model, fixtures[0], Path(destination) / f"minilm-{length}", model_id=MODEL_ID,
                model_revision=REVISION, input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["embedding"], kernel_package=kernel_package, seed=20261025,
                additional_fixtures=list(zip(FIXTURE_SEEDS[1:], fixtures[1:])),
                extra_files=assets + [(ROOT / "licenses/model-Apache-2.0.txt", "licenses/model-Apache-2.0.txt")], overwrite=overwrite,
                preprocessing=dict(tokenizer="WordPiece", tokenizer_directory="tokenizer",
                                   sequence_length=length, padding="max_length", truncation=True,
                                   pooling="attention_mask_mean", output_normalization="L2"),
                metadata=dict(weights_sha256=sha256(weight_path),
                              source=f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
                              license="Apache-2.0", fixture_texts=list(TEXTS)),
            )
    return result
