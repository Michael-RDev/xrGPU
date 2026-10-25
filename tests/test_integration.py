import json
from pathlib import Path
import shutil
import subprocess

import pytest

from runtime import prepare
from runtime.build import verify_shader_registry
from runtime.integration.hooks import TRANSFORMS

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY = ROOT / "artifacts/deps/executorch"


@pytest.fixture
def dependency():
    if not (DEPENDENCY / ".git").exists():
        pytest.skip("Pinned ExecuTorch sources required; run runtime.prepare")
    return DEPENDENCY


@pytest.mark.parametrize("experimental", [False, True])
def test_native_package_selection(tmp_path, dependency, experimental):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("Host C++ compiler required")
    output = tmp_path / "package"
    args = [
        compiler,
        "-std=c++17",
        "-pthread",
        "-I",
        str(ROOT),
        "-I",
        str(dependency / "third-party/json/single_include"),
    ]
    if experimental:
        args.append("-DXRGPU_ENABLE_EXPERIMENTS")
    subprocess.run([*args, str(ROOT / "tests/native/package.cpp"), "-o", str(output)], check=True)
    subprocess.run([str(output), str(ROOT / "kernels/linear/contract.json")], check=True)


def test_shader_registry_requires_all_current_variants(dependency):
    # Freeze the public shader names requested by the C++ hooks. This check must
    # also work on a fresh checkout without any historical package artifacts.
    shaders = (
        {
            f"xrgpu_linear_m{rows}{bias}_texture3d_{weight}_float"
            for rows in (1, 2, 4)
            for bias in ("", "_bias")
            for weight in ("texture2d", "buffer")
        }
        | {
            f"xrgpu_gelu_{mode}_float_{storage}"
            for mode in ("erf", "tanh")
            for storage in ("buffer", "texture3d")
        }
        | {"xrgpu_grouped_conv2d_float", "xrgpu_grouped_conv2d_clamp_float"}
    )
    registry = "\n".join(json.dumps(name) for name in shaders)
    assert verify_shader_registry(registry) == set(shaders)
    for name in shaders:
        with pytest.raises(RuntimeError, match="missing from registry"):
            verify_shader_registry(registry.replace(json.dumps(name), ""))


def test_integration_migration_is_idempotent_and_preserves_unknown_edits(
    tmp_path, dependency, monkeypatch
):
    root = tmp_path / "project"
    dest = root / "artifacts/deps/executorch"
    dest.mkdir(parents=True)
    prefix = "backends/vulkan/runtime/graph/ops/impl/"
    paths = [prefix + name for name in TRANSFORMS] + ["backends/vulkan/runtime/VulkanBackend.cpp"]
    for relative in paths:
        original = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=dependency)
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(original)
    patch = root / "runtime/patches/0001-xrgpu-linear.patch"
    patch.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "runtime/patches/0001-xrgpu-linear.patch", patch)
    subprocess.run(["git", "init", "-q"], cwd=dest, check=True)
    subprocess.run(["git", "add", "."], cwd=dest, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Host Test",
            "-c",
            "user.email=test@localhost",
            "commit",
            "-qm",
            "Pinned fixture",
        ],
        cwd=dest,
        check=True,
    )
    subprocess.run(["git", "apply", str(patch)], cwd=dest, check=True)
    monkeypatch.setattr(prepare, "ROOT", root)
    prepare.install_integration(dest)
    installed = {p: (dest / p).read_bytes() for p in paths}
    prepare.install_integration(dest)
    assert installed == {p: (dest / p).read_bytes() for p in paths}
    changed = dest / paths[-1]
    changed.write_text(changed.read_text() + "\n// local edit\n")
    before = {p: (dest / p).read_bytes() for p in paths}
    with pytest.raises(RuntimeError, match="unknown dependency edits"):
        prepare.install_integration(dest)
    assert before == {p: (dest / p).read_bytes() for p in paths}
