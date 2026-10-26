"""Export pinned xrGPU examples; all programs and fixtures stay local."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "minilm", "mobilenetv2"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/model-bundles")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--package", type=Path, help="Verified immutable xrGPU kernel package")
    parser.add_argument("--lengths", type=int, nargs="+", choices=(32, 128, 256), default=(32, 128, 256))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(20261025)
    package = None
    if args.package:
        from runtime.kernel_package import verify
        package = verify(args.package)
    results = {}
    options = dict(offline=args.offline, kernel_package=package, overwrite=args.overwrite)
    if args.model in ("all", "minilm"):
        from models.minilm import export
        results.update(export(args.output, lengths=args.lengths, **options))
    if args.model in ("all", "mobilenetv2"):
        from models.mobilenetv2 import export
        results.update(export(args.output, **options))
    summary = {key: dict(bundle_sha256=value["bundle_sha256"], programs=list(value["programs"]),
                         export_errors=value["export_errors"]) for key, value in results.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
