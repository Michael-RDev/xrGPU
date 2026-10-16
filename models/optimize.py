"""Export full-model optimization experiments alongside the immutable FP32 baseline."""

import argparse
import copy
import json
import os
from pathlib import Path

from models.export import MODEL, REVISION, ROOT, export_model, sha256

PROFILES = {
    "flat": {"attention": "eager", "packed_qkv": False, "fp16": False},
    "sdpa": {"attention": "sdpa", "packed_qkv": False, "fp16": False},
    "qkv": {"attention": "sdpa", "packed_qkv": True, "fp16": False},
    "qkv_fp16": {"attention": "sdpa", "packed_qkv": True, "fp16": True},
    "buffer": {"attention": "sdpa", "packed_qkv": True, "fp16": False, "buffer": True},
    "sdpa_buffer": {"attention": "sdpa", "packed_qkv": False, "fp16": False, "buffer": True},
    "flash": {"attention": "flash", "packed_qkv": True, "fp16": False, "buffer": True},
    "fused": {"attention": "flash", "packed_qkv": True, "fp16": False, "buffer": True, "fuse_norm": True},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=["flat", "sdpa", "qkv", "qkv_fp16"]
    )
    parser.add_argument("--lengths", nargs="+", type=int, choices=(32, 128, 256), default=[128])
    parser.add_argument("--bundle", type=Path, default=ROOT / "artifacts/bundle")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/cache/huggingface"))
    os.environ.setdefault(
        "FLATC_EXECUTABLE", str(ROOT / "artifacts/build/flatbuffers-host/flatc.exe")
    )
    import torch
    from transformers import AutoModel, AutoTokenizer
    from models.optimized import OptimizedMiniLM
    from executorch.backends.vulkan.serialization.vulkan_graph_schema import VkStorageType

    torch.set_num_threads(1)
    torch.manual_seed(20261013)
    encoder = AutoModel.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    out = args.bundle
    manifest = json.loads((out / "bundle.json").read_text())
    profiles = manifest.setdefault("optimization_profiles", {})
    references = json.loads((out / "references.json").read_text())
    for length in args.lengths:
        reference_audit = OptimizedMiniLM(encoder, length, attention="eager", audit=True).eval()
        audit_cases = []
        audit_texts = [
            "",
            "Find the flashlight so I can see in the dark virtual room.",
            "headset " * 300,
        ]
        for index, text in enumerate(audit_texts):
            t = tokenizer(
                text, padding="max_length", truncation=True, max_length=length, return_tensors="pt"
            )
            inputs = tuple(t[k] for k in ("input_ids", "attention_mask", "token_type_ids"))
            with torch.inference_mode():
                expected = reference_audit(*inputs)
            filename = f"audit_{length}_{index}.expected.bin"
            expected.numpy().astype("<f4").tofile(out / filename)
            audit_cases.append({"text": text, "expected": filename})
        for name in args.profiles:
            config = PROFILES[name]
            model = OptimizedMiniLM(
                copy.deepcopy(encoder),
                length,
                attention=config["attention"],
                packed_qkv=config["packed_qkv"],
                fuse_norm=config.get("fuse_norm", False),
            ).eval()
            audit = copy.deepcopy(model)
            audit.audit = True
            # Exact graph rewrites must satisfy the existing elementwise layer
            # bound before export. Device precision changes are audited separately.
            max_error = 0.0
            with torch.inference_mode():
                for case in references:
                    if case["sequence_length"] != length:
                        continue
                    ids = torch.tensor([case["input_ids"]], dtype=torch.long)
                    mask = torch.tensor([case["attention_mask"]], dtype=torch.long)
                    sample = (ids, mask, torch.zeros_like(ids))
                    actual, expected = audit(*sample), reference_audit(*sample)
                    error = (actual - expected).abs()
                    max_error = max(max_error, error.max().item())
                    if not torch.all(error <= 1e-4 + 1e-4 * expected.abs()):
                        raise RuntimeError(f"{name}/{length}: graph rewrite failed layer tolerance")
                    embedding = model(*sample)[0]
                    reference = torch.tensor(case["embedding"])
                    if F_cosine(embedding, reference) < 0.999:
                        raise RuntimeError(f"{name}/{length}: embedding reference mismatch")
            options = {"force_fp16": config["fp16"]}
            if config.get("buffer"):
                options["storage_type_override"] = VkStorageType.BUFFER
            stem = f"minilm_opt_{name}_{length}"
            print(f"Exporting {name}/{length}; host checkpoint max error {max_error:g}", flush=True)
            export_model(model, inputs, out / f"{stem}.pte", True, options)
            export_model(audit, inputs, out / f"{stem}_audit.pte", True, options)
            for suffix in ("", "_audit"):
                graph = (out / f"{stem}{suffix}.graph.txt").read_text()
                if (
                    graph.count(
                        "call_function[target=torch.ops.higher_order.executorch_call_delegate]"
                    )
                    != 1
                ):
                    raise RuntimeError(f"{stem}{suffix}: expected one fully delegated GPU graph")
            entry = profiles.setdefault(
                name,
                {
                    "precision": "fp16" if config["fp16"] else "fp32",
                    "config": config,
                    "lengths": {},
                },
            )
            entry["lengths"][str(length)] = {
                "model": f"{stem}.pte",
                "audit_model": f"{stem}_audit.pte",
                "audit_cases": audit_cases,
                "checkpoints": audit.checkpoint_layout(),
                "host_max_abs_error": max_error,
            }
            manifest["files"] = {
                p.relative_to(out).as_posix(): sha256(p)
                for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "bundle.json"
            }
            (out / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Optimization programs and strict checkpoint oracles exported", flush=True)


def F_cosine(a, b):
    import torch.nn.functional as F

    return F.cosine_similarity(a, b, dim=0).item()


if __name__ == "__main__":
    main()
