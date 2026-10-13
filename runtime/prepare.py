"""Fetch pinned sources, then apply the reviewed xrGPU integration patch.

Run with python -m runtime.prepare. Never resets an existing source checkout.
"""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMODULES = [
    "backends/vulkan/third-party/Vulkan-Headers",
    "backends/vulkan/third-party/VulkanMemoryAllocator",
    "backends/vulkan/third-party/volk",
    "third-party/flatbuffers", "third-party/flatcc", "third-party/json",
    "third-party/pocketfft", "third-party/gflags",
    "backends/xnnpack/third-party/FXdiv", "backends/xnnpack/third-party/cpuinfo",
    "backends/xnnpack/third-party/pthreadpool",
    "backends/xnnpack/third-party/FP16", "backends/xnnpack/third-party/XNNPACK",
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
            raise RuntimeError(f"{dest}: expected {spec['commit']}, found {actual}; refusing to reset")
        if name == "executorch":
            run("git", "submodule", "update", "--init", "--depth", "1", *SUBMODULES, cwd=dest)
    apply_patch()


def apply_patch():
    lock = json.loads((ROOT / "dependencies.json").read_text())
    for name in ("executorch", "openxr"):
        source = ROOT / "artifacts/deps" / name
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if actual != lock[name]["commit"]:
            raise RuntimeError(f"Unexpected {name} commit {actual}; see dependencies.json")
    dest = ROOT / "artifacts/deps/executorch"
    patch = ROOT / "runtime/patches/0001-xrgpu-linear.patch"
    reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=dest,
                             capture_output=True)
    if reverse.returncode:
        run("git", "apply", "--check", str(patch), cwd=dest)
        run("git", "apply", str(patch), cwd=dest)
    shader_dir = dest / "backends/vulkan/runtime/graph/ops/glsl"
    for file in (ROOT / "kernels/linear").glob("xrgpu_linear.*"):
        copy_if_changed(file, shader_dir / file.name)
    copy_if_changed(ROOT / "runtime/native/dispatch.h",
                    dest / "backends/vulkan/runtime/graph/ops/impl/XrGpuDispatch.h")


def copy_if_changed(source, destination):
    if not destination.exists() or source.read_bytes() != destination.read_bytes():
        shutil.copyfile(source, destination)


if __name__ == "__main__":
    prepare()
