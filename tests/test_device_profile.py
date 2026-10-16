import json
import shutil
import subprocess
import runpy
from pathlib import Path

import pytest

from runtime.prepare import vulkan_shader_sources

ROOT = Path(__file__).resolve().parents[1]


def test_native_profiles_and_dispatch(tmp_path):
    compiler = shutil.which("c++") or shutil.which("cl")
    if compiler is None:
        pytest.skip("Host C++ compiler required for native profile checks")
    msvc = Path(compiler).stem.lower() == "cl"
    executable = tmp_path / ("profiles.exe" if msvc else "profiles")
    command = (
        [
            compiler,
            "/nologo",
            "/std:c++17",
            "/EHsc",
            "/I" + str(ROOT),
            str(ROOT / "tests/native/profiles.cpp"),
            "/Fe:" + str(executable),
            "/Fo:" + str(tmp_path) + "/",
        ]
        if msvc
        else [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT),
            str(ROOT / "tests/native/profiles.cpp"),
            "-o",
            str(executable),
        ]
    )
    subprocess.run(command, check=True)
    lines = subprocess.check_output([str(executable)], text=True).splitlines()
    manifest = json.loads((ROOT / "kernels/linear/vulkan/quest3/manifest.json").read_text())
    assert lines[:3] == ["quest3", manifest["kernel_set"], manifest["shader_prefix"]]
    expected = [
        [v["id"], v["tile_m"], v["tile_n"], *v["workgroup"], int(v["weight_storage"] == "buffer")]
        for v in manifest["variants"]
    ]
    assert sorted([[int(value) for value in line.split()] for line in lines[3:]]) == sorted(expected)


def test_shader_discovery_and_namespace_collisions(tmp_path):
    sources = vulkan_shader_sources()
    assert {p.name for p in sources} == {
        "xrgpu_linear.glsl",
        "xrgpu_linear.yaml",
        "xrgpu_gelu.glsl",
        "xrgpu_gelu.yaml",
        "xrgpu_attention.glsl",
        "xrgpu_attention.yaml",
        "xrgpu_residual_norm.glsl",
        "xrgpu_residual_norm.yaml",
    }
    assert all(p.parent.name in ("quest3", "portable") for p in sources)
    for device in ("one", "two"):
        folder = tmp_path / "kernels/linear/vulkan" / device
        folder.mkdir(parents=True)
        (folder / "same_name.glsl").write_text("shader")
    with pytest.raises(ValueError, match="unique shader filenames"):
        vulkan_shader_sources(tmp_path)


@pytest.mark.skipif(
    not (ROOT / "artifacts/deps/executorch/backends/vulkan/runtime/gen_vulkan_spv.py").exists(),
    reason="Run runtime.prepare first",
)
def test_shader_macro_preprocessing_handles_windows_newlines():
    module = runpy.run_path(
        str(ROOT / "artifacts/deps/executorch/backends/vulkan/runtime/gen_vulkan_spv.py")
    )
    source = "#define ADD(x) " + "\\" + "\n    ((x) + 1)\n"
    assert module["preprocess"](source.replace("\n", "\r\n"), {}) == source
