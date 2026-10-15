import json
import shutil
import subprocess
from pathlib import Path

import pytest

from runtime.prepare import vulkan_shader_sources

ROOT = Path(__file__).resolve().parents[1]


def test_native_profiles_and_dispatch(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("Host C++ compiler required for native profile checks")
    executable = tmp_path / "profiles"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT),
            str(ROOT / "tests/native/profiles.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    lines = subprocess.check_output([str(executable)], text=True).splitlines()
    manifest = json.loads((ROOT / "kernels/linear/vulkan/quest3/manifest.json").read_text())
    assert lines[:3] == ["quest3", manifest["kernel_set"], manifest["shader_prefix"]]
    expected = [
        [v["id"], v["tile_m"], v["tile_n"], *v["workgroup"], int(v["weight_storage"] == "buffer")]
        for v in manifest["variants"]
    ]
    assert [[int(value) for value in line.split()] for line in lines[3:]] == expected


def test_shader_discovery_and_namespace_collisions(tmp_path):
    sources = vulkan_shader_sources()
    assert {p.name for p in sources} == {"xrgpu_linear.glsl", "xrgpu_linear.yaml"}
    assert all(p.parent.name == "quest3" for p in sources)
    for device in ("one", "two"):
        folder = tmp_path / "kernels/linear/vulkan" / device
        folder.mkdir(parents=True)
        (folder / "same_name.glsl").write_text("shader")
    with pytest.raises(ValueError, match="unique shader filenames"):
        vulkan_shader_sources(tmp_path)
