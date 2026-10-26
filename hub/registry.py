"""Prepare allowlisted, native-only xrML kernel repositories and publish releases."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from hub import snapshot
from kernels import catalog as native_catalog, provenance

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "xrgpu.benchmark-kernels.v1"
NATIVE_ID = "xrgpu.native-operators.v1"
AGGREGATE = "xrGPU-kernels"


def canonical(value):
    return snapshot.canonical(value)


def digest(value):
    return snapshot.digest(value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regular(root, relative):
    snapshot.safe_path(relative)
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Package roots cannot be symlinks")
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("Package symlinks are forbidden")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Missing package file: " + relative)
    return path


def verify_native(directory):
    root = Path(directory)
    manifest = json.loads(regular(root, "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("package_id") != NATIVE_ID:
        raise ValueError("Unsupported native operator package")
    if digest({k: v for k, v in manifest.items() if k != "package_sha256"}) != manifest.get("package_sha256"):
        raise ValueError("Native package manifest digest mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Native package file inventory required")
    if len({p.casefold() for p in files}) != len(files):
        raise ValueError("Ambiguous native package paths")
    for name, expected in files.items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha(regular(root, name)) != expected:
            raise ValueError("Native package file digest mismatch: " + name)
    if not any(p.endswith("libxrgpu_operators.so") for p in files):
        raise ValueError("Native Android library required; source-only cards cannot be published as implemented")
    if not any(p.endswith(".spv") for p in files):
        raise ValueError("Compiled native SPIR-V required")
    if any(p.endswith((".wgsl", ".wgsl.jinja", ".js", ".mjs", ".ts", ".html")) for p in files):
        raise ValueError("Browser implementations are not native distribution artifacts")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Native package symlinks are forbidden")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Native package contains unlisted or missing files")
    return manifest


def inventory(directory):
    """Seal only a producer-created directory; the transport independently verifies it."""
    root = Path(directory)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Prepared repository symlinks are forbidden")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            snapshot.safe_path(relative)
            if relative == "hub-inventory.json":
                continue
            if any(p.startswith(".") for p in relative.split("/")):
                raise ValueError("Hidden files are forbidden in prepared repositories")
            files[relative] = sha(path)
    if not files:
        raise ValueError("Prepared repository is empty")
    value = dict(schema_version=1, files=files)
    (root / "hub-inventory.json").write_bytes(canonical(value) + b"\n")
    return value


def verify_prepared(directory):
    root = Path(directory)
    value = json.loads(regular(root, "hub-inventory.json").read_text())
    if value.get("schema_version") != 1 or not isinstance(value.get("files"), dict) or not value["files"]:
        raise ValueError("Prepared repository inventory required")
    expected_names = set(value["files"]) | {"hub-inventory.json"}
    actual_names = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Prepared repository symlinks are forbidden")
        if path.is_file():
            actual_names.add(path.relative_to(root).as_posix())
    if actual_names != expected_names:
        raise ValueError("Prepared repository contains unlisted or missing files")
    for relative, expected in value["files"].items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid prepared repository checksum")
        if sha(regular(root, relative)) != expected:
            raise ValueError("Prepared repository digest mismatch: " + relative)
    return value


def copy_inventory(source, destination, files):
    destination = Path(destination)
    for name in sorted(files):
        original = regular(source, name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def native_operations(manifest, catalog=None):
    catalog = catalog or native_catalog.load()
    records = {entry["id"]: entry for entry in catalog["entries"]}
    operations = manifest.get("operations", {})
    if isinstance(operations, list):
        if len(set(operations)) != len(operations):
            raise ValueError("Duplicate native operator declaration")
        operations = {name: manifest.get("supported_contracts", {}).get(name, {}) for name in operations}
    if not isinstance(operations, dict) or not operations or not set(operations).issubset(records):
        raise ValueError("Native package must declare implemented native catalog contracts")
    tags = dict(float32="f32", float16="f16", int32="i32", uint32="u32", int8="i8", uint8="u8", int16="i16")
    for identity, support in operations.items():
        if not isinstance(support, dict) or not support:
            raise ValueError("Missing implemented native contract: " + identity)
        entry = records[identity]
        if entry["native"]["api"] == "legacy":
            row = entry["native"]["registry"]
            types = support.get("dtypes", support.get("supported_dtypes", []))
            if not types or not set(types).issubset(row["dtypes"]):
                raise ValueError("Unsupported declared native dtypes: " + identity)
            required = {"shaders/" + row["slug"] + "_" + tags[t] + ".spv" for t in types}
        else:
            required = {"shaders/" + v["output"] for v in entry["variants"]}
            required.add("lib/android-arm64/libxrgpu_extended_operators.so")
            if support.get("native_api") != "extended" or support.get("shader") != "extended/" + entry["native"]["shader"]:
                raise ValueError("Native extended implementation metadata mismatch: " + identity)
        if not required.issubset(manifest.get("files", {})):
            raise ValueError("Missing compiled native implementation: " + identity)
    return operations


def verify_benchmark(directory):
    """An existing benchmark package is optional and always explicitly supplied."""
    root = Path(directory)
    manifest = json.loads(regular(root, "manifest.json").read_text())
    if manifest.get("package_id") != BENCHMARK_ID or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Unsupported benchmark package identity/inventory")
    for name, expected in manifest["files"].items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or sha(regular(root, name)) != expected:
            raise ValueError("Benchmark source digest mismatch")
    if not any(Path(name).name.startswith("LICENSE") for name in manifest["files"]):
        raise ValueError("Benchmark package must inventory its license")
    return manifest


def copy_native_catalog(source, destination, catalog):
    """Copy catalog-owned public source trees, excluding caches/private file types."""
    source, destination = Path(source), Path(destination)
    suffixes = {".py", ".cpp", ".h", ".def", ".glsl", ".glslh", ".comp", ".md", ".json", ".yaml", ".txt", ".cmake"}
    names = {"catalog.py", "catalog.json", "provenance.py", "__init__.py"}
    for entry in catalog["entries"]:
        names.update(p.relative_to(source).as_posix() for p in (source / entry["path"]).rglob("*")
                     if p.is_file() and (p.suffix in suffixes or p.name in {"LICENSE", "NOTICE", "CMakeLists.txt"})
                     and not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(source).parts))
    names.update(p.relative_to(source).as_posix() for p in (source / "_shared").rglob("*")
                 if p.is_file() and (p.suffix in suffixes or p.name in {"LICENSE", "NOTICE", "CMakeLists.txt"})
                 and not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(source).parts))
    copy_inventory(source, destination, names)
    frozen = provenance.catalog(source / "_provenance")
    copy_inventory(source / "_provenance", destination / "_provenance",
                   {p for item in frozen["entries"] for p in item["files"]} | {"catalog.json", "LICENSE", "NOTICE"})


def operator_package(source, destination, manifest, entry, support):
    """Create an inventoried per-operator binary subset with a distinct digest."""
    allowed = {"shaders/" + v["output"] for v in entry["variants"]}
    names = {name for name in manifest["files"] if name in allowed or name.startswith("include/")
             or name.startswith("lib/android-arm64/") or Path(name).name in {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"}}
    copy_inventory(source, destination, names)
    value = dict(schema_version=1, package_id=NATIVE_ID, version=1, backend="vulkan", target="android-arm64",
                 operations=[entry["id"]], supported_contracts={entry["id"]: support},
                 qualification="unqualified", origin_package_sha256=manifest["package_sha256"],
                 files={name: manifest["files"][name] for name in sorted(names)})
    value["package_sha256"] = digest(value)
    write_json(Path(destination) / "manifest.json", value)
    return value


def operator_card(entry, support, package, namespace):
    types = support.get("dtypes", support.get("supported_dtypes", []))
    return f'''---
library_name: kernels
license: apache-2.0
tags:
- kernel
- vulkan
- android
- xr
- alpha
---
# {entry['id']} — xrGPU native alpha

Native Android ARM64 Vulkan implementation distributed by `{namespace}`.
This package contains a compiled C++ library and SPIR-V compute shaders.

**Alpha: physical GPU correctness, complete contract conformance, performance,
and sustained XR behavior remain unverified.** Kernel availability does not
establish device support or an optimization claim.

The supported native contract is recorded in `operator.json`; unsupported
attributes, types and shapes must return an explicit unsupported-contract error.
Declared native types: {', '.join(types) if types else 'see the native manifest'}.

The complete upstream reference contract and its {entry['test_count']} test cases
are preserved for audit, including cases outside the current native alpha.
Upstream `skipGpu` flags describe WebGPU limitations; they are not native skip approvals.

- Native package: `{package['package_id']}` / `{package['package_sha256']}`.
- Upstream: [`{entry['repo_id']}` at `{entry['revision']}`](https://huggingface.co/kernels/{entry['repo_id']}/tree/{entry['revision']}).
- Backend: Vulkan on Android ARM64. No browser or WebGPU runtime is used.
- License: Apache-2.0 for new native code and upstream contract snapshots;
  retained third-party notices govern any marked dependencies.

Use the native C++ `xrgpu::operators` API in the included headers and see the
aggregate [`{namespace}/{AGGREGATE}`](https://huggingface.co/kernels/{namespace}/{AGGREGATE})
for the xrBench fetch workflow, full catalog and incomplete operator roadmap.
'''


def prepare(native_package, output, *, catalog_root=None, namespace="xrML", benchmark_package=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", namespace):
        raise ValueError("Invalid Hub namespace")
    native_package, output = Path(native_package).absolute(), Path(output).absolute()
    catalog_root = Path(catalog_root).absolute() if catalog_root else native_catalog.KERNELS
    if output.exists():
        raise FileExistsError("Use a fresh immutable release directory")
    package = verify_native(native_package)
    catalog = native_catalog.load(catalog_root)
    operations = native_operations(package, catalog)
    frozen = provenance.catalog(catalog_root / "_provenance")
    references = {entry["id"]: entry for entry in frozen["entries"]}
    benchmark = verify_benchmark(benchmark_package) if benchmark_package is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".hub-release-", dir=output.parent))
    try:
        aggregate = staging / "aggregate"
        aggregate.mkdir()
        copy_inventory(native_package, aggregate / "native", set(package["files"]) | {"manifest.json"})
        copy_native_catalog(catalog_root, aggregate / "kernels", catalog)
        packages = dict(native=dict(package_id=NATIVE_ID, path="native", package_sha256=package["package_sha256"]))
        if benchmark is not None:
            copy_inventory(benchmark_package, aggregate / "kernels/benchmark", set(benchmark["files"]) | {"manifest.json"})
            packages["benchmark"] = dict(package_id=BENCHMARK_ID, path="kernels/benchmark", manifest_sha256=digest(benchmark))
        for name in ("LICENSE", "NOTICE"):
            shutil.copyfile(regular(catalog_root / "_provenance", name), aggregate / name)
        entries, prepared = [], []
        for entry in catalog["entries"]:
            identity, reference = entry["id"], references[entry["id"]]
            implemented = identity in operations
            support = operations.get(identity, {})
            status = dict(implementation="implemented" if implemented else "not_in_supplied_package",
                          contract_conformance="incomplete", hardware="unverified", performance="unverified", sustained_xr="unverified")
            entries.append(dict(id=identity, native=entry["native"], contract="kernels/" + entry["contract"],
                                provenance=entry["provenance"], status=status, supported_contract=support,
                                repo_id=namespace + "/" + identity if implemented else None))
            if not implemented:
                continue
            destination = staging / "operators" / identity
            destination.mkdir(parents=True)
            subset = operator_package(native_package, destination / "native", package, entry, support)
            copy_inventory(catalog_root / "_provenance", destination / "reference", reference["files"])
            copy_inventory(catalog_root, destination / "kernels", {entry[k] for k in ("contract", "tests", "benchmarks")})
            operator = dict(schema_version=1, kind="xrgpu.native-operator", id=identity, version=1,
                            backend="vulkan", platform="android-arm64", status=status, native=entry["native"],
                            contract="kernels/" + entry["contract"], supported_contract=support,
                            native_package=dict(package_id=NATIVE_ID, package_sha256=subset["package_sha256"], path="native", origin_package_sha256=package["package_sha256"]),
                            upstream=dict(repo_id=reference["repo_id"], revision=reference["revision"],
                                          contract="reference/" + reference["contract"], tests="reference/" + reference["tests"], benchmarks="reference/" + reference["benchmarks"]))
            write_json(destination / "operator.json", operator)
            (destination / "README.md").write_text(operator_card(reference, support, subset, namespace))
            for name in ("LICENSE", "NOTICE"):
                shutil.copyfile(regular(catalog_root / "_provenance", name), destination / name)
            native_files = {p.relative_to(destination).as_posix(): sha(p) for p in destination.rglob("*") if p.is_file()}
            metadata = dict(name=identity, id="_" + identity.replace(".", "_").lower() + "_vulkan_alpha", version=1,
                            license="Apache-2.0", backend=dict(type="vulkan"),
                            digest=dict(algorithm="sha256", files={p: base64.b64encode(bytes.fromhex(h)).decode() for p, h in native_files.items()}),
                            xrgpu=dict(status=status, native_package_sha256=subset["package_sha256"], origin_package_sha256=package["package_sha256"]))
            write_json(destination / "build/android-vulkan/metadata.json", metadata)
            prepared.append(dict(repo_id=namespace + "/" + identity, directory="operators/" + identity, inventory_sha256=digest(inventory(destination))))
        registry = dict(schema_version=1, kind="xrgpu.hub-registry", release="native-kernels-v1", namespace=namespace,
                        backend="vulkan", platform="android-arm64", status="alpha", operator_count=len(entries), implemented_count=len(operations),
                        catalog_sha256=catalog["catalog_sha256"], upstream_catalog_sha256=frozen["catalog_sha256"], entries=entries, packages=packages)
        registry["registry_sha256"] = digest(registry)
        write_json(aggregate / "registry.json", registry)
        benchmark_note = ("An explicitly supplied benchmark package is preserved at `kernels/benchmark`, with its original identity and license."
                          if benchmark is not None else "No optional benchmark source package was supplied or restored.")
        (aggregate / "README.md").write_text(f"""---
library_name: kernels
license: apache-2.0
tags: [kernel, vulkan, android, xr, alpha]
---
# xrGPU native kernels

Native Android ARM64 Vulkan distribution: {len(operations)} operators in this supplied binary package,
with {catalog['operator_count']} native operator definitions in `kernels/catalog.json`.
Each definition links actual shared C++ implementation metadata and compiled shader variants.
No browser compute runtime is included.

`registry.json` binds the original native package digest `{package['package_sha256']}`
and the native catalog digest `{catalog['catalog_sha256']}`. Per-operator binary
subsets have their own manifests and record this origin package digest.

Physical XR GPU correctness, complete contract conformance, performance and sustained
XR behavior remain unverified. Software Vulkan evidence is distinct from physical
qualification. Unsupported contracts raise explicitly; native APIs provide no CPU fallback.

Original contracts and fixtures remain unchanged audit data under `kernels/_provenance`;
its catalog digest is `{frozen['catalog_sha256']}`. Original WebGPU shader hashes are
provenance only. Native portable semantics live in each operator's `contract.json`.

{benchmark_note}

Pin a published aggregate commit and the native package SHA-256 before using the
xrBench fetch workflow. Preparation and explicit fetch can access storage; execution
uses verified local files offline. This local preparation command does not publish.
See LICENSE, NOTICE, native manifests and retained upstream attribution.
""")
        prepared.insert(0, dict(repo_id=namespace + "/" + AGGREGATE, directory="aggregate", inventory_sha256=digest(inventory(aggregate))))
        release = dict(schema_version=1, kind="xrgpu.hub-release", namespace=namespace, visibility="public", repo_type="kernel",
                       status="prepared", registry_sha256=registry["registry_sha256"], repositories=prepared)
        release["release_sha256"] = digest(release)
        write_json(staging / "release.json", release)
        staging.rename(output)
        return release
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_release(directory):
    root = Path(directory)
    release = json.loads(regular(root, "release.json").read_text())
    if release.get("schema_version") != 1 or release.get("kind") != "xrgpu.hub-release":
        raise ValueError("Unsupported Hub release")
    if digest({k: v for k, v in release.items() if k != "release_sha256"}) != release.get("release_sha256"):
        raise ValueError("Hub release digest mismatch")
    if release.get("visibility") != "public" or release.get("repo_type") != "kernel":
        raise ValueError("xrML releases require explicit public kernel repositories")
    seen = set()
    for repo in release["repositories"]:
        name = repo["repo_id"]
        if name in seen or not name.startswith(release["namespace"] + "/"):
            raise ValueError("Unexpected or duplicate release repository")
        seen.add(name)
        snapshot.safe_path(repo["directory"])
        path = root / repo["directory"]
        value = verify_prepared(path)
        if digest(value) != repo["inventory_sha256"]:
            raise ValueError("Release repository inventory mismatch")
    return release


def publish(directory, *, transport=None, env_file=None):
    """Explicit publication only; never invoked by preparation or benchmarks."""
    root = Path(directory).resolve()
    release = verify_release(root)
    transport = Path(transport).resolve() if transport else ROOT / "hub/transport.py"
    if not transport.is_file():
        raise FileNotFoundError("The isolated Hugging Face transport is required")
    common = ["uv", "run", "--isolated", "--locked", "--script", str(transport)]
    if env_file:
        common += ["--env-file", str(Path(env_file).resolve())]
    results = []
    for repo in release["repositories"]:
        args = common + ["publish", "--repo-id", repo["repo_id"], "--directory", str(root / repo["directory"]),
                         "--message", "Publish verified native xrGPU alpha kernel artifacts"]
        completed = subprocess.run(args, text=True, capture_output=True, timeout=600)
        if completed.returncode:
            # Transport errors can include credential-bearing HTTP diagnostics.
            raise RuntimeError("Kernel publication failed for " + repo["repo_id"] +
                               "; check kernel namespace entitlement and transport status")
        try:
            result = json.loads(completed.stdout)
        except (ValueError, TypeError):
            raise RuntimeError("Kernel transport returned an invalid publication receipt") from None
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--native-package", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--catalog", type=Path, default=native_catalog.KERNELS)
    p.add_argument("--namespace", default="xrML")
    p.add_argument("--benchmark-package", type=Path, help="Explicit existing verified benchmark package; omitted by default")
    v = commands.add_parser("verify")
    v.add_argument("directory", type=Path)
    p = commands.add_parser("publish")
    p.add_argument("directory", type=Path)
    p.add_argument("--transport", type=Path)
    p.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.native_package, args.output, catalog_root=args.catalog, namespace=args.namespace, benchmark_package=args.benchmark_package)
    elif args.command == "verify":
        result = verify_release(args.directory)
    else:
        result = publish(args.directory, transport=args.transport, env_file=args.env_file)
    if args.command != "publish":
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
