"""Exercise the model fingerprint selector independently of ExecuTorch builds."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def contract(root, name, kind):
    path = root / "kernels" / name
    path.mkdir(parents=True)
    (path / "contract.json").write_text(json.dumps({"kind": kind, "operation": name}))
    (path / "kernel.glsl").write_text("original shader")
    return path


def fingerprint(root, directory):
    cmake = shutil.which("cmake") or str(ROOT.parent / "xrBench/.venv/bin/cmake")
    if not Path(cmake).is_file():
        pytest.skip("CMake is required to exercise revision source selection")
    script = directory / "select.cmake"
    output = directory / "sources.txt"
    script.write_text(
        f'include("{ROOT / "cmake/model_revision_sources.cmake"}")\n'
        f'xrgpu_model_kernel_sources("{root}" selected)\n'
        f'file(WRITE "{output}" "${{selected}}")\n'
    )
    run = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    sources = output.read_text().split(";") if output.read_text() else []
    return sources, hashlib.sha256(
        "".join(hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources).encode()
    ).hexdigest()


def test_catalog_and_provenance_do_not_change_model_kernel_fingerprint(tmp_path):
    model = contract(tmp_path, "linear", "xrgpu.model-override")
    native = contract(tmp_path, "ai.onnx.Add", "xrgpu.native-operator")
    shared = tmp_path / "kernels/_shared"
    provenance = tmp_path / "kernels/_provenance"
    shared.mkdir()
    provenance.mkdir()
    (shared / "kernel.glsl").write_text("native common shader")
    (provenance / "catalog.json").write_text("immutable upstream bytes")
    files, before = fingerprint(tmp_path, tmp_path)
    assert set(files) == {str(model / "contract.json"), str(model / "kernel.glsl")}
    (native / "kernel.glsl").write_text("changed native implementation")
    (native / "contract.json").write_text(
        json.dumps({"kind": "xrgpu.native-operator", "operation": "renamed"})
    )
    (shared / "kernel.glsl").write_text("changed common native code")
    (provenance / "catalog.json").write_text("changed provenance")
    assert fingerprint(tmp_path, tmp_path)[1] == before
    (model / "kernel.glsl").write_text("changed model override")
    assert fingerprint(tmp_path, tmp_path)[1] != before


def test_unmarked_or_experimental_contracts_are_not_model_overrides(tmp_path):
    unknown = contract(tmp_path, "old", "experimental")
    (unknown / "contract.json").write_text('{"operation":"old"}')
    assert fingerprint(tmp_path, tmp_path)[0] == []
