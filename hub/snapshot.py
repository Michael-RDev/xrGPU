"""Freeze public upstream contracts; never execute upstream shaders or expressions."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from kernels import provenance

NAMESPACE = "webgpu-kernels"
EXPECTED_COUNT = 221
RETAIN = {"README.md", "LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE", "NOTICE.txt",
          "build/webgpu/metadata.json", "build/webgpu/manifest.json",
          "build/webgpu/test.json", "build/webgpu/bench.json"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe_path(name):
    return provenance.safe_path(name)


def get(url):
    for attempt in range(6):
        try:
            with urlopen(Request(url, headers={"User-Agent": "xrGPU-contract-snapshot/1"}), timeout=90) as response:
                return response.read()
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(f"Upstream download failed: HTTP {error.code} at {url}") from None
            time.sleep(min(int(error.headers.get("Retry-After", "5")), 30) * (attempt + 1))
    raise RuntimeError("Unreachable download retry")


def source_entry(row, staging):
    repo = row["id"]
    name = repo.removeprefix(NAMESPACE + "/")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Unexpected upstream operator identifier")
    api = "https://huggingface.co/api/kernels/" + repo
    info = json.loads(get(api + "/revision/v1"))
    revision = info.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Immutable upstream revision required")
    tree = json.loads(get(api + "/tree/" + revision + "?recursive=true&expand=false"))
    if not isinstance(tree, list) or len(tree) >= 1000:
        raise ValueError("Unexpected or truncated upstream file inventory")
    paths = {item["path"] for item in tree if item["type"] == "file"}
    required = {"README.md", "build/webgpu/metadata.json", "build/webgpu/manifest.json",
                "build/webgpu/test.json", "build/webgpu/bench.json"}
    if not required.issubset(paths):
        raise ValueError("Incomplete upstream operator: " + repo)
    payloads = {}
    all_hashes = {}
    base = "https://huggingface.co/kernels/" + repo + "/resolve/" + revision + "/"
    for relative in sorted(paths):
        safe_path(relative)
        if relative == ".gitattributes":
            continue
        payload = get(base + quote(relative, safe="/"))
        all_hashes[relative] = hashlib.sha256(payload).hexdigest()
        payloads[relative] = payload
    metadata = json.loads(payloads["build/webgpu/metadata.json"])
    if metadata.get("name") != name or metadata.get("version") != 1:
        raise ValueError("Upstream metadata identity mismatch: " + repo)
    if metadata.get("digest", {}).get("algorithm") != "sha256":
        raise ValueError("Unsupported upstream digest algorithm: " + repo)
    for relative, expected in metadata["digest"]["files"].items():
        safe_path(relative)
        actual = base64.b64encode(hashlib.sha256(payloads["build/webgpu/" + relative]).digest()).decode()
        if actual != expected:
            raise ValueError("Upstream metadata digest mismatch: " + repo + "/" + relative)
    manifest = json.loads(payloads["build/webgpu/manifest.json"])
    if manifest.get("domain", "") + "." + manifest.get("name", "") != name:
        raise ValueError("Upstream contract identity mismatch: " + repo)
    if metadata.get("license") != "Apache-2.0" or b"license: apache-2.0" not in payloads["README.md"].lower():
        raise ValueError("Review unexpected upstream license: " + repo)
    tests = json.loads(payloads["build/webgpu/test.json"])
    benchmarks = json.loads(payloads["build/webgpu/bench.json"])
    if not isinstance(tests.get("cases"), list) or not isinstance(benchmarks.get("cases"), list):
        raise ValueError("Invalid upstream case inventory: " + repo)
    retained = {}
    for relative in sorted(paths.intersection(RETAIN)):
        destination = staging / name / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[relative])
        retained[relative] = all_hashes[relative]
    prefix = "upstream/" + name + "/"
    return dict(id=name, repo_id=repo, revision=revision, source_version="v1", license="Apache-2.0",
                contract=prefix + "build/webgpu/manifest.json", tests=prefix + "build/webgpu/test.json",
                benchmarks=prefix + "build/webgpu/bench.json", metadata=prefix + "build/webgpu/metadata.json",
                files={prefix + p: h for p, h in retained.items()}, upstream_files=all_hashes,
                upstream_kernel_revision=metadata.get("provenance", {}).get("kernel", {}).get("sha"),
                test_count=len(tests["cases"]), benchmark_count=len(benchmarks["cases"]))


def freeze(output, workers=6):
    output = Path(output).resolve()
    if (output / "catalog.json").exists() or (output / "upstream").exists():
        raise FileExistsError("Frozen catalog already exists; verify it or select a new output")
    rows = json.loads(get("https://huggingface.co/api/kernels?author=" + NAMESPACE + "&limit=1000&full=true"))
    if len(rows) != EXPECTED_COUNT or len({r["id"] for r in rows}) != EXPECTED_COUNT:
        raise ValueError("Upstream catalog changed; review the expected operator count explicitly")
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=output))
    try:
        entries = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(source_entry, row, staging) for row in rows]
            for future in as_completed(futures):
                entries.append(future.result())
                if len(entries) % 20 == 0:
                    print(f"Verified {len(entries)}/{len(rows)} upstream operators", flush=True)
        catalog = dict(schema_version=1, kind="xrgpu.upstream-catalog", source_namespace=NAMESPACE,
                       source_version="v1", operator_count=len(entries),
                       test_count=sum(e["test_count"] for e in entries),
                       benchmark_count=sum(e["benchmark_count"] for e in entries),
                       note="v1 is resolved to immutable commits; WGSL digests are audit provenance, never native executable artifacts.",
                       entries=sorted(entries, key=lambda e: e["id"]))
        catalog["catalog_sha256"] = digest(catalog)
        staging.rename(output / "upstream")
        (output / "catalog.json").write_bytes(canonical(catalog) + b"\n")
        return catalog
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify(directory=None):
    return provenance.catalog(directory if directory is not None else provenance.PROVENANCE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    parser.add_argument("--output", type=Path, default=provenance.PROVENANCE)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    value = freeze(args.output, args.workers) if args.command == "freeze" else verify(args.output)
    print(json.dumps({k: value[k] for k in ("operator_count", "test_count", "benchmark_count", "catalog_sha256")}, indent=2))


if __name__ == "__main__":
    main()
