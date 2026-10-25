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
    contract = json.loads((ROOT / "kernels/linear/contract.json").read_text())
    assert lines[:3] == ["quest3", "quest3.linear.vulkan.fp32.v2", "xrgpu_linear"]
    variants = {v["id"]: v for v in contract["variants"]}
    for line in lines[3:]:
        candidate, name, *values = line.split()
        assert int(candidate) in (1, 2, 3)
        spec = variants[name]
        assert list(map(int, values)) == [
            spec["tile_m"],
            4,
            *spec["workgroup"],
            int(spec["weight_storage"] == "buffer"),
        ]
    assert len(lines[3:]) == 3


def test_shader_discovery_and_namespace_collisions(tmp_path):
    sources = vulkan_shader_sources()
    assert {p.name for p in sources} == {
        f"xrgpu_{operation}.{extension}"
        for operation in ("linear", "gelu", "grouped_conv2d")
        for extension in ("glsl", "yaml")
    }
    assert all(p.parent.name == "quest3" for p in sources)
    for device in ("one", "two"):
        folder = tmp_path / "kernels/linear/vulkan" / device
        folder.mkdir(parents=True)
        (folder / "same_name.glsl").write_text("shader")
    with pytest.raises(ValueError, match="unique shader filenames"):
        vulkan_shader_sources(tmp_path)
