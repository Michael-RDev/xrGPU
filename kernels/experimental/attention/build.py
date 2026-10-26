"""Compile the isolated attention prototype and export discovery OR holdout fixtures.

python -m kernels.experimental.attention.build --glslc /absolute/path/glslc \
    --output artifacts/experiments/attention-discovery

No inference hooks are installed, no shaders are dispatched, and no performance
claims are inferred from compilation or host references. Use --stage confirmation
only after freezing a shortlist from hardware discovery measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import numpy as np

from .reference import attention, parameters

SOURCE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, glslc, stage="discovery"):
    output, glslc = Path(output).resolve(), Path(glslc).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh experiment directory")
    if stage not in ("discovery", "confirmation"):
        raise ValueError("Unknown fixture stage")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".attention-", dir=output.parent))
    try:
        contract = json.loads((SOURCE / "contract.json").read_text())
        for name in ("attention.comp", "contract.json", "reference.py", "build.py"):
            shutil.copyfile(SOURCE / name, staging / name)
        compiler = subprocess.check_output([str(glslc), "--version"], text=True).strip()
        variants = []
        for size in contract["variants"]:
            binary = staging / f"attention_wg{size}.spv"
            subprocess.run([str(glslc), "-O", "--target-env=vulkan1.1",
                            f"-DWORKGROUP_SIZE={size}", str(staging / "attention.comp"),
                            "-o", str(binary)], check=True)
            if binary.read_bytes()[:4] != struct.pack("<I", 0x07230203):
                raise ValueError("Invalid SPIR-V binary")
            variants.append(dict(workgroup_size=size, binary=binary.name,
                                 shared_memory_bytes=4 * (128 + 256 + size)))
        fixtures = []
        # First three reproduce the shapes seen in baseline traces, with fresh
        # data. Others exercise mask broadcast, lane tails and supported limits.
        cases = [("attention_short", 1, 6, 32, 32, 64, 64, "none"),
                 ("attention_masked", 1, 6, 128, 128, 64, 64, "batch"),
                 ("attention_odd", 2, 3, 7, 11, 13, 17, "none"),
                 ("attention_batch_mask", 2, 3, 7, 11, 13, 17, "batch"),
                 ("attention_broadcast_mask", 2, 3, 7, 11, 13, 17, "broadcast"),
                 ("attention_limits", 1, 1, 3, 256, 128, 128, "batch"),
                 ("attention_single", 1, 1, 1, 1, 1, 1, "none")]
        for seed in contract[stage + "_seeds"]:
            rng = np.random.default_rng(seed)
            for name, b, h, m, s, d, dv, mode in cases:
                q = rng.normal(size=(b, h, m, d)).astype(np.float32)
                k = rng.normal(size=(b, h, s, d)).astype(np.float32)
                v = rng.normal(size=(b, h, s, dv)).astype(np.float32)
                mask = None
                if mode != "none":
                    mask = np.zeros((b if mode == "batch" else 1, 1, 1, s), dtype=np.float32)
                    mask[..., -(s // 3 or 1):] = -10000
                push = parameters(q, k, v, mask)
                filename = f"{name}-{seed}.npz"
                expected = attention(q, k, v, mask)
                np.savez(staging / filename, q=q, k=k, v=v, expected=expected,
                         mask=mask if mask is not None else np.zeros(1, dtype=np.float32),
                         push_constants=np.asarray(push, dtype="<u4"))
                fixtures.append(dict(case=name, seed=seed, file=filename,
                                     dispatch=[m, b * h, 1], push_constants=list(push)))
        files = {p.name: sha(p) for p in sorted(staging.iterdir()) if p.is_file()}
        manifest = dict(schema_version=1, kind="xrgpu.attention_prototype", status="unqualified",
                        compiler=compiler, flags=["-O", "--target-env=vulkan1.1"],
                        stage=stage, variants=variants, fixtures=fixtures, files=files,
                        runtime_integration=False, gpu_correctness_validated=False,
                        performance_validated=False)
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["experiment_sha256"] = hashlib.sha256(canonical).hexdigest()
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
        return dict(path=str(output), variants=len(variants), fixtures=len(fixtures), stage=stage,
                    experiment_sha256=manifest["experiment_sha256"], status="unqualified")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glslc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("discovery", "confirmation"), default="discovery")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.glslc, args.stage), indent=2))


if __name__ == "__main__":
    main()
