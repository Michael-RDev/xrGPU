"""Strict local access to immutable upstream reference metadata.

These files are audit data, never executable kernels or native support claims.
The explicit root argument also supports old self-contained package snapshots.
"""
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path, PurePath, PurePosixPath
import re

PROVENANCE = Path(__file__).resolve().parent / "_provenance"
RETAIN = frozenset({"README.md", "LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE", "NOTICE.txt",
    "build/webgpu/metadata.json", "build/webgpu/manifest.json", "build/webgpu/test.json", "build/webgpu/bench.json"})
REQUIRED = frozenset({"README.md", "build/webgpu/metadata.json", "build/webgpu/manifest.json", "build/webgpu/test.json", "build/webgpu/bench.json"})
DOCUMENTS = {"contract": "build/webgpu/manifest.json", "tests": "build/webgpu/test.json", "benchmarks": "build/webgpu/bench.json", "metadata": "build/webgpu/metadata.json"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe_path(relative):
    if isinstance(relative, PurePath):
        relative = relative.as_posix()
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Unsafe provenance file path")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError("Unsafe provenance file path")
    return path


def regular(relative, root=PROVENANCE):
    """Resolve a regular file below an explicit root, rejecting symlink aliases."""
    relative = safe_path(relative)
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError("Provenance root symlinks are forbidden")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Provenance file symlinks are forbidden")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Missing regular provenance file: " + str(relative))
    return path


def _bytes(relative, expected, root):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid provenance checksum")
    payload = regular(relative, root=root).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Frozen upstream checksum/digest mismatch: " + str(relative))
    return payload


def catalog(root=PROVENANCE, *, verify_files=True):
    """Validate the original catalog identity and optionally every retained file.

    There is no global cache: a modified catalog or reference cannot retain a
    previously trusted result. Metadata-only lookup still checks catalog paths,
    source revisions, counts, licenses and its canonical SHA-256.
    """
    value = json.loads(regular("catalog.json", root=root).read_bytes())
    if value.get("kind") != "xrgpu.upstream-catalog" or value.get("schema_version") != 1:
        raise ValueError("Unsupported frozen upstream catalog")
    content = {key: item for key, item in value.items() if key != "catalog_sha256"}
    if digest(content) != value.get("catalog_sha256"):
        raise ValueError("Frozen upstream catalog digest mismatch")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != value.get("operator_count"):
        raise ValueError("Frozen catalog count mismatch")
    seen = set()
    total_tests = total_benchmarks = 0
    for item in entries:
        identity = item.get("id")
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", identity) or identity in seen:
            raise ValueError("Invalid or duplicate frozen operator identity")
        seen.add(identity)
        if not re.fullmatch(r"[0-9a-f]{40}", item.get("revision", "")):
            raise ValueError("Immutable upstream revision required")
        if item.get("license") != "Apache-2.0":
            raise ValueError("Unexpected upstream license")
        prefix = "upstream/" + identity + "/"
        files = item.get("files")
        if not isinstance(files, dict) or not {prefix + name for name in REQUIRED}.issubset(files) or not set(files).issubset({prefix + name for name in RETAIN}):
            raise ValueError("Unexpected retained upstream file inventory")
        for kind, suffix in DOCUMENTS.items():
            if item.get(kind) != prefix + suffix:
                raise ValueError("Frozen document identity/path mismatch")
        for relative, expected in files.items():
            safe_path(relative)
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("Invalid frozen file digest")
            if verify_files:
                _bytes(relative, expected, root)
        for kind, count_key in (("tests", "test_count"), ("benchmarks", "benchmark_count")):
            count = item.get(count_key)
            if type(count) is not int or count < 0:
                raise ValueError("Invalid frozen case count")
            if verify_files:
                data = document(item, kind, root=root)
                if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or len(data["cases"]) != count:
                    raise ValueError("Frozen upstream case count mismatch")
        total_tests += item["test_count"]
        total_benchmarks += item["benchmark_count"]
    if total_tests != value.get("test_count") or total_benchmarks != value.get("benchmark_count"):
        raise ValueError("Frozen catalog total case count mismatch")
    return value


def entry(identity, root=PROVENANCE):
    for item in catalog(root, verify_files=False)["entries"]:
        if item["id"] == identity:
            return item
    raise ValueError("Unknown frozen operator: " + str(identity))


def document(operator, kind="contract", root=PROVENANCE):
    if kind not in DOCUMENTS:
        raise ValueError("Unknown frozen document kind: " + str(kind))
    item = entry(operator, root) if isinstance(operator, str) else operator
    if not isinstance(item, dict) or kind not in item:
        raise ValueError("Missing frozen document declaration")
    relative = item[kind]
    expected = item.get("files", {}).get(relative)
    return json.loads(_bytes(relative, expected, root))
