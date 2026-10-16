import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_model(module, inputs, destination, vulkan, options=None):
    import torch
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    exported = torch.export.export(module.eval(), inputs, strict=True)
    partitions = (
        [VulkanPartitioner({"force_fp16": False, **(options or {})})]
        if vulkan
        else [XnnpackPartitioner()]
    )
    edge = to_edge_transform_and_lower(exported, partitioner=partitions)
    # Keep the graph for diagnosis of CPU fallbacks and layout conversions
    destination.with_suffix(".graph.txt").write_text(str(edge.exported_program().graph))
    program = edge.to_executorch()
    destination.write_bytes(program.buffer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/bundle")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/cache/huggingface"))
    flatc = ROOT / "artifacts/build/flatbuffers-host/flatc.exe"
    if os.name == "nt" and flatc.exists():
        os.environ.setdefault("FLATC_EXECUTABLE", str(flatc))
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(20261013)
    torch.set_num_threads(1)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=args.offline
    )
    encoder = AutoModel.from_pretrained(
        MODEL, revision=REVISION, local_files_only=args.offline, attn_implementation="eager"
    ).eval()

    class Embedding(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            hidden = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=False,
            )[0]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (
                (hidden * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1e-9)
            ).squeeze(1)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    model = Embedding(encoder).eval()
    tokenizer.save_pretrained(out / "tokenizer")
    corpus = json.loads((ROOT / "models/corpus.json").read_text(encoding="utf-8"))
    suite = json.loads((ROOT / "benchmarks/suite.json").read_text(encoding="utf-8"))
    texts = [d["text"] for d in corpus["documents"]] + [q["text"] for q in corpus["queries"]]
    texts += corpus["validation_texts"] + ["headset " * 300]
    fixtures = []
    for length in suite["sequence_lengths"]:
        for text in texts:
            tokens = tokenizer(
                text, padding="max_length", truncation=True, max_length=length, return_tensors="pt"
            )
            inputs = tuple(tokens[k] for k in ("input_ids", "attention_mask", "token_type_ids"))
            with torch.inference_mode():
                expected = model(*inputs)[0].tolist()
            fixtures.append(
                {
                    "text": text,
                    "sequence_length": length,
                    "input_ids": inputs[0][0].tolist(),
                    "attention_mask": inputs[1][0].tolist(),
                    "embedding": expected,
                }
            )
        if not args.references_only:
            for backend in ("cpu", "vulkan"):
                print(f"Exporting MiniLM {backend} seq={length}", flush=True)
                export_model(
                    model, inputs, out / f"minilm_{backend}_{length}.pte", backend == "vulkan"
                )
    (out / "references.json").write_text(json.dumps(fixtures, allow_nan=False))
    (out / "corpus.json").write_text(json.dumps(corpus, indent=2))
    (out / "suite.json").write_bytes((ROOT / "benchmarks/suite.json").read_bytes())

    layers = []
    # Actual pretrained FFN weights, not synthetic weights of a different scale.
    for name, layer in (
        ("up", encoder.encoder.layer[0].intermediate.dense),
        ("down", encoder.encoder.layer[0].output.dense),
    ):
        for length in (1, 31, 32, 128, 256):
            x = torch.randn(1, length, layer.in_features)
            with torch.inference_mode():
                y = layer(x)
            stem = f"linear_{name}_{length}"
            x.numpy().astype("<f4").tofile(out / f"{stem}.input.bin")
            y.numpy().astype("<f4").tofile(out / f"{stem}.expected.bin")
            if not args.references_only:
                export_model(layer, (x,), out / f"{stem}.pte", True)
            layers.append(
                {
                    "id": stem,
                    "shape": list(x.shape),
                    "output_shape": list(y.shape),
                }
            )
    (out / "layers.json").write_text(json.dumps(layers, indent=2))
    for k, n in ((13, 17), (384, 1535)):
        layer = torch.nn.Linear(k, n).eval()
        x = torch.randn(1, 7, k)
        with torch.inference_mode():
            y = layer(x)
        stem = f"linear_edge_{k}_{n}"
        x.numpy().astype("<f4").tofile(out / f"{stem}.input.bin")
        y.numpy().astype("<f4").tofile(out / f"{stem}.expected.bin")
        if not args.references_only:
            export_model(layer, (x,), out / f"{stem}.pte", True)
        layers.append(
            {
                "id": stem,
                "shape": list(x.shape),
                "output_shape": list(y.shape),
            }
        )
    (out / "layers.json").write_text(json.dumps(layers, indent=2))
    import shutil

    shutil.copytree(ROOT / "licenses", out / "licenses", dirs_exist_ok=True)
    # Verify labeled retrieval against the reference; fail before shipping a
    # demo whose labels do not agree with the unchanged model.
    primary = {x["text"]: np.array(x["embedding"]) for x in fixtures if x["sequence_length"] == 128}
    document_vectors = np.stack([primary[d["text"]] for d in corpus["documents"]])
    for query in corpus["queries"]:
        found = corpus["documents"][int(np.argmax(document_vectors @ primary[query["text"]]))]["id"]
        if found != query["expected_id"]:
            raise RuntimeError(f"Reference retrieval mismatch: {query['text']}: {found}")
    manifest = {
        "schema_version": 1,
        "model_id": MODEL,
        "model_revision": REVISION,
        "precision": "fp32",
        "seed": 20261013,
        "runtime_revision": json.loads((ROOT / "dependencies.json").read_text())["executorch"][
            "commit"
        ],
        "environment": {
            k: importlib.metadata.version(k)
            for k in ("torch", "executorch", "transformers", "numpy")
        },
        "suite_revision": sha256(out / "suite.json"),
        "references_only": args.references_only,
        "files": {
            p.relative_to(out).as_posix(): sha256(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "bundle.json"
        },
    }
    (out / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out}; reference retrieval checks passed", flush=True)


if __name__ == "__main__":
    main()
