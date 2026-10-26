import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_native_sha256_matches_standard_library(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required")
    source = tmp_path / "sha.cpp"
    source.write_text('#include "runtime/native/sha256.h"\n#include <iostream>\n'
                      'int main(int n,char** p){for(int i=1;i<n;++i)std::cout<<xrgpu::file_sha256(p[i])<<"\\n";}\n')
    executable = tmp_path / "sha"
    subprocess.run([compiler, "-std=c++17", "-I", str(ROOT), str(source), "-o", str(executable)], check=True)
    paths = []
    expected = []
    for length in (0, 1, 55, 56, 63, 64, 65, 127, 128, 1024 * 1024 + 13):
        value = bytes(i % 251 for i in range(length))
        path = tmp_path / str(length)
        path.write_bytes(value)
        paths.append(str(path))
        expected.append(hashlib.sha256(value).hexdigest())
    result = subprocess.run([str(executable), *paths], check=True, text=True, capture_output=True)
    assert result.stdout.splitlines() == expected


def test_java_library_owned_tensors_and_static_validation(tmp_path):
    if not shutil.which("javac") or not shutil.which("java"):
        pytest.skip("JDK required")
    sources = sorted((ROOT / "library/src/main/java/org/xrgpu").glob("*.java"))
    sources.append(ROOT / "tests/java/LibraryTensorTests.java")
    subprocess.run(["javac", "-Xlint:all", "-d", str(tmp_path), *map(str, sources)], check=True)
    result = subprocess.run(["java", "-cp", str(tmp_path), "org.xrgpu.LibraryTensorTests"],
                            check=True, text=True, capture_output=True)
    assert "Library tensor checks passed: 17" in result.stdout


def test_native_session_multi_input_output_and_rejects_invalid_bundles(tmp_path):
    executable = ROOT / "artifacts/build/host-library/xrgpu_session_test"
    runner = ROOT / "artifacts/build/host-library/xrgpu_run"
    if not executable.is_file() or not runner.is_file():
        pytest.skip("Build the host library and native Session acceptance executable first")
    torch = pytest.importorskip("torch")
    from runtime.model_bundle import export_bundle

    class Mixed(torch.nn.Module):
        def forward(self, values, tokens):
            return values.transpose(0, 1) + 1, tokens + 3

    bundle = tmp_path / "mixed"
    manifest = export_bundle(Mixed(), (torch.arange(6, dtype=torch.float32).reshape(2, 3),
                                      torch.tensor([5, 9], dtype=torch.int64)), bundle,
                             model_id="native-mixed-test", model_revision="test-fixture",
                             input_names=["values", "tokens"])
    result = subprocess.run([str(executable), str(bundle)], check=True, text=True, capture_output=True)
    assert "Native Session lifecycle" in result.stdout
    run = tmp_path / "run"
    run.mkdir()
    inputs = [str(bundle / item["file"]) for item in manifest["fixtures"][0]["inputs"]]
    subprocess.run([str(runner), str(bundle), *inputs], cwd=run, check=True, capture_output=True)
    import numpy as np

    for index, spec in enumerate(manifest["fixtures"][0]["outputs"]):
        dtype = "<f4" if spec["dtype"] == "float32" else "<i8"
        assert np.array_equal(np.fromfile(run / f"output{index}.bin", dtype=dtype),
                              np.fromfile(bundle / spec["file"], dtype=dtype))
    import json

    original = (bundle / "bundle.json").read_text()
    for damage in ("runtime", "policy", "interface", "checksum", "program"):
        changed = json.loads(original)
        if damage == "runtime":
            changed["runtime_revision"] = "wrong"
        elif damage == "policy":
            changed["qualification"] = {"status": "qualified"}
        elif damage == "interface":
            changed["inputs"][0]["shape"] = [1, 6]
        elif damage == "checksum":
            filename = changed["programs"]["cpu"]["file"]
            changed["files"][filename] = "0" * 64
        else:
            filename = changed["programs"]["cpu"]["file"]
            (bundle / filename).write_bytes(b"invalid serialized program")
            changed["files"][filename] = hashlib.sha256((bundle / filename).read_bytes()).hexdigest()
        (bundle / "bundle.json").write_text(json.dumps(changed))
        failed = subprocess.run([str(runner), str(bundle)], capture_output=True, text=True)
        assert failed.returncode == 1, (damage, failed.stderr)
        assert failed.stderr
    (bundle / "bundle.json").write_text(original)


def test_native_configuration_rejects_wrong_runtime_checkout(tmp_path):
    cmake = ROOT / ".venv/bin/cmake"
    if not cmake.is_file():
        pytest.skip("Project CMake environment required")
    result = subprocess.run([str(cmake), "-S", str(ROOT), "-B", str(tmp_path / "build"),
                             "-DXRGPU_BUILD_DEMO=OFF", "-DXRGPU_ENABLE_VULKAN=OFF",
                             f"-DEXECUTORCH_ROOT={ROOT}"], text=True, capture_output=True)
    assert result.returncode != 0
    assert "does not match dependencies.json pinned revision" in result.stdout + result.stderr
