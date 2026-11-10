"""Byte-hashed catalog inputs must survive Git's Windows checkout conversion."""

# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_hashed_sources_survive_autocrlf_checkout(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required to exercise checkout filters")
    repository = tmp_path / "repo"
    repository.mkdir()

    def run(*args):
        return subprocess.run(
            [git, "-c", "core.autocrlf=true", "-c", "core.safecrlf=false", *args],
            cwd=repository, check=True, capture_output=True,
        )

    run("init")
    shutil.copyfile(ROOT / ".gitattributes", repository / ".gitattributes")
    native = json.loads((ROOT / "kernels/catalog.json").read_bytes())
    upstream = json.loads((ROOT / "kernels/_provenance/catalog.json").read_bytes())
    # One real file per retained filename covers READMEs, licenses and nested fixtures.
    samples = {}
    for entry in upstream["entries"]:
        for relative, digest in entry["files"].items():
            suffix = relative.removeprefix("upstream/" + entry["id"] + "/")
            samples.setdefault(suffix, ("kernels/_provenance/" + relative, digest))
    entry = native["entries"][0]
    expected = dict(samples.values())
    for key, digest in entry["descriptor_digests"].items():
        expected["kernels/" + entry[key]] = digest
    for relative, digest in expected.items():
        source = (ROOT / relative).read_bytes()
        assert hashlib.sha256(source).hexdigest() == digest, relative
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)

    # Reindexing hashes raw bytes too, so even freshly authored CRLF descriptors
    # must survive check-in and checkout without Git silently changing their hashes.
    literals = {"kernels/_provenance/upstream/example/NOTICE.txt": b"CRLF\r\nLF\n"}
    for name in ("contract", "test", "bench"):
        literals[f"kernels/example/{name}.json"] = b'{\r\n  "example": true\r\n}\r\n'
    for literal, payload in literals.items():
        target = repository / literal
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        expected[literal] = hashlib.sha256(payload).hexdigest()
    run("add", "-f", ".gitattributes", *expected)
    # Force a fresh working-tree write, using exactly Git's checkout filters.
    checkout = tmp_path / "checkout"
    run("checkout-index", "--all", "--prefix=" + checkout.as_posix() + "/")
    for relative, digest in expected.items():
        assert hashlib.sha256((checkout / relative).read_bytes()).hexdigest() == digest, relative
