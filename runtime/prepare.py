import json
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from runtime.integration.hooks import TRANSFORMS, once

ROOT = Path(__file__).resolve().parents[1]
SUBMODULES = [
    "backends/vulkan/third-party/Vulkan-Headers",
    "backends/vulkan/third-party/VulkanMemoryAllocator",
    "backends/vulkan/third-party/volk",
    "third-party/flatbuffers",
    "third-party/flatcc",
    "third-party/json",
    "third-party/pocketfft",
    "third-party/gflags",
    "backends/xnnpack/third-party/FXdiv",
    "backends/xnnpack/third-party/cpuinfo",
    "backends/xnnpack/third-party/pthreadpool",
    "backends/xnnpack/third-party/FP16",
    "backends/xnnpack/third-party/XNNPACK",
]


def run(*args, cwd=ROOT):
    subprocess.run(args, cwd=cwd, check=True)


def prepare():
    lock = json.loads((ROOT / "dependencies.json").read_text())
    for name in ("executorch", "openxr"):
        spec = lock[name]
        dest = ROOT / "artifacts/deps" / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            run("git", "clone", "--depth", "1", "--branch", spec["ref"], spec["url"], str(dest))
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
        if actual != spec["commit"]:
            raise RuntimeError(
                f"{dest}: expected {spec['commit']}, found {actual}; refusing to reset"
            )
        if name == "executorch":
            run("git", "submodule", "update", "--init", "--depth", "1", *SUBMODULES, cwd=dest)
    apply_patch()


def apply_patch():
    lock = json.loads((ROOT / "dependencies.json").read_text())
    for name in ("executorch", "openxr"):
        source = ROOT / "artifacts/deps" / name
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        if actual != lock[name]["commit"]:
            raise RuntimeError(f"Unexpected {name} commit {actual}; see dependencies.json")
    dest = ROOT / "artifacts/deps/executorch"
    install_integration(dest)
    shader_dir = dest / "backends/vulkan/runtime/graph/ops/glsl"
    for file in vulkan_shader_sources():
        copy_if_changed(file, shader_dir / file.name)
    copy_if_changed(
        ROOT / "runtime/native/dispatch.h",
        dest / "backends/vulkan/runtime/graph/ops/impl/XrGpuDispatch.h",
    )
    copy_if_changed(
        ROOT / "runtime/integration/xrgpu_kernel_package.h",
        dest / "backends/vulkan/runtime/graph/ops/impl/xrgpu_kernel_package.h",
    )


def timestamp_hook(source):
    source = '#include "graph/ops/impl/XrGpuDispatch.h"\n' + once(
        source,
        "config.enable_querypool = true;",
        "config.enable_querypool = xrgpu::gpu_timestamps_supported;",
    )
    # Several qualified linear variants share one SPIR-V binary. Preserve the
    # actual dispatch geometry so xrBench can verify workgroup specialization.
    return once(
        source,
        r'''", \"dispatch_id\": " + std::to_string(r.dispatch_id) + "}";''',
        r'''", \"dispatch_id\": " + std::to_string(r.dispatch_id) +
          ", \"global_size\": [" + std::to_string(r.metadata.global_workgroup_size[0]) + "," +
          std::to_string(r.metadata.global_workgroup_size[1]) + "," + std::to_string(r.metadata.global_workgroup_size[2]) +
          "], \"local_size\": [" + std::to_string(r.metadata.local_workgroup_size[0]) + "," +
          std::to_string(r.metadata.local_workgroup_size[1]) + "," + std::to_string(r.metadata.local_workgroup_size[2]) + "]}";''',
    )


def legacy_source(source, relative):
    """Reconstruct the known old patch without modifying the dependency."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / relative
        path.parent.mkdir(parents=True)
        path.write_text(source)
        subprocess.run(
            [
                "git",
                "apply",
                "--include=" + relative,
                str(ROOT / "runtime/patches/0001-xrgpu-linear.patch"),
            ],
            cwd=folder,
            check=True,
            capture_output=True,
        )
        return path.read_text()


def install_integration(dest):
    """Migrate known legacy hooks; refuse unknown dependency edits before writing."""
    prefix = "backends/vulkan/runtime/graph/ops/impl/"
    transforms = {prefix + name: transform for name, transform in TRANSFORMS.items()}
    transforms["backends/vulkan/runtime/VulkanBackend.cpp"] = timestamp_hook
    state_path = ROOT / "artifacts/integration-sources.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    planned = {}
    for relative, transform in transforms.items():
        original = subprocess.check_output(
            ["git", "show", "HEAD:" + relative],
            cwd=dest,
            text=True,
        )
        expected = transform(original)
        current = (dest / relative).read_text()
        known = {original, expected}
        if relative.endswith(("Linear.cpp", "VulkanBackend.cpp")):
            known.add(legacy_source(original, relative))
        digest = hashlib.sha256(current.encode()).hexdigest()
        if current not in known and digest != previous.get(relative):
            raise RuntimeError("Refusing to overwrite unknown dependency edits: " + relative)
        planned[relative] = expected
    for relative, expected in planned.items():
        path = dest / relative
        if path.read_text() != expected:
            path.write_text(expected)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {name: hashlib.sha256(source.encode()).hexdigest() for name, source in planned.items()},
            indent=2,
        )
        + "\n"
    )
    temporary.replace(state_path)


def copy_if_changed(source, destination):
    if not destination.exists() or source.read_bytes() != destination.read_bytes():
        shutil.copyfile(source, destination)


def vulkan_shader_sources(root=ROOT):
    """ExecuTorch has a flat shader namespace; refuse colliding implementations."""
    sources = sorted((root / "kernels").glob("*/vulkan/*/*"))
    shaders = [p for p in sources if p.suffix in (".glsl", ".yaml")]
    names = [p.name for p in shaders]
    if len(names) != len(set(names)):
        raise ValueError("Vulkan implementations must use unique shader filenames")
    return shaders


if __name__ == "__main__":
    prepare()
