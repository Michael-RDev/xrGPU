import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/toolchain"


def download(url, path, expected=None, algorithm="sha256", headers=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {path.name}", flush=True)
        request = urllib.request.Request(url, headers=headers or {})
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(request, timeout=120) as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        temporary.replace(path)
    digest = hashlib.new(algorithm)
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if expected and digest.hexdigest() != expected:
        raise RuntimeError(f"Checksum mismatch: {path}")


def unzip(url, filename, checksum, target, folder, algorithm="sha1"):
    if target.exists():
        return
    archive = BASE / "downloads" / filename
    download(url, archive, checksum, algorithm)
    staging = BASE / ("unpack-" + filename)
    staging.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(staging)
    else:
        subprocess.run(["unzip", "-qo", str(archive), "-d", str(staging)], check=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not staging.resolve().is_relative_to(BASE.resolve()) or not target.resolve().is_relative_to(
        BASE.resolve()
    ):
        raise RuntimeError("Archive staging directory escaped toolchain directory")
    for attempt in range(6):
        try:
            (staging / folder).rename(target)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(2)
    shutil.rmtree(staging)


def windows(adb_only=False):
    base = "https://dl.google.com/android/repository/"
    sdk = BASE / "android-sdk"
    unzip(
        base + "platform-tools_r37.0.1-win.zip",
        "platform-tools-windows.zip",
        "e03e78b1d80b396f1c3358e31251cb31740e1110",
        sdk / "platform-tools",
        "platform-tools",
    )
    if adb_only:
        return
    unzip(
        base + "android-ndk-r28c-windows.zip",
        "ndk-windows.zip",
        "086bba43ff2f5eb0e387b15c8278bb4e0d89ba1d",
        sdk / "ndk/28.2.13676358",
        "android-ndk-r28c",
    )
    unzip(
        base + "platform-35_r02.zip",
        "platform.zip",
        "0bb560a90a7a2cbd0dd8348224d518b638fe7949",
        sdk / "platforms/android-35",
        "android-35",
    )
    unzip(
        base + "build-tools_r35_windows.zip",
        "build-tools-windows.zip",
        "af059bb67cf7786f45ee0db85e2d24985df1b4b6",
        sdk / "build-tools/35.0.0",
        "android-15",
    )
    unzip(
        "https://services.gradle.org/distributions/gradle-8.11.1-bin.zip",
        "gradle.zip",
        "f397b287023acdba1e9f6fc5ea72d22dd63669d59ed4a289a29b1a76eee151c6",
        BASE / "gradle",
        "gradle-8.11.1",
        "sha256",
    )
    spec = json.loads((ROOT / "runtime/windows-toolchain.json").read_text())
    package = spec["jdk"]
    unzip(package["url"], "jdk.zip", package["sha256"], BASE / "jdk", package["folder"], "sha256")
    shader = spec["shaderc"]
    source = ROOT / "artifacts/deps/shaderc"
    if not source.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", shader["ref"], shader["url"], str(source)],
            check=True,
        )
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != shader["commit"]:
        raise RuntimeError("Unexpected shaderc source revision")
    if not (BASE / "shaderc/bin/glslc.exe").exists():
        subprocess.run(
            [sys.executable, str(source / "utils/git-sync-deps")], cwd=source, check=True
        )
        build = ROOT / "artifacts/build/shaderc"
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DSHADERC_SKIP_TESTS=ON",
                "-DSHADERC_SKIP_EXAMPLES=ON",
                "-DSHADERC_SKIP_COPYRIGHT_CHECK=ON",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build), "--target", "glslc_exe", "-j", "4"], check=True
        )
        (BASE / "shaderc/bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(build / "glslc/glslc.exe", BASE / "shaderc/bin/glslc.exe")
    flatc = ROOT / "artifacts/build/flatbuffers-host/flatc.exe"
    if not flatc.exists():
        flatbuffers = ROOT / "artifacts/deps/executorch/third-party/flatbuffers"
        if not (flatbuffers / "CMakeLists.txt").exists():
            raise RuntimeError("Run python -m runtime.prepare before the full bootstrap")
        subprocess.run(
            [
                "cmake",
                "-S",
                str(flatbuffers),
                "-B",
                str(flatc.parent),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DFLATBUFFERS_BUILD_TESTS=OFF",
                "-DFLATBUFFERS_BUILD_FLATLIB=OFF",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(flatc.parent), "--target", "flatc", "-j", "4"], check=True
        )
    paths = [
        BASE / "jdk/bin",
        BASE / "shaderc/bin",
        BASE / "gradle/bin",
        sdk / "platform-tools",
        ROOT / ".venv/Scripts",
    ]

    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    values = {
        "ANDROID_HOME": sdk,
        "ANDROID_NDK": sdk / "ndk/28.2.13676358",
        "JAVA_HOME": BASE / "jdk",
        "GRADLE_USER_HOME": ROOT / "artifacts/cache/gradle",
        "ANDROID_USER_HOME": ROOT / "artifacts/cache/android",
        "FLATC_EXECUTABLE": flatc,
    }
    lines = [f"$env:{key} = {quote(value)}" for key, value in values.items()]
    lines.append("$env:PYTHONUTF8 = '1'")
    lines.append(f"$env:PATH = {quote(';'.join(map(str, paths)) + ';')} + $env:PATH")
    (BASE / "env.ps1").write_text("\n".join(lines) + "\n")
    print(f"Toolchain ready. Dot-source {BASE / 'env.ps1'}")


def main():
    parser = argparse.ArgumentParser(description="Install verified workspace-local Android tools")
    parser.add_argument("--adb-only", action="store_true")
    args = parser.parse_args()
    if sys.platform == "win32":
        windows(args.adb_only)
        return
    if sys.platform != "darwin":
        raise SystemExit("Bootstrap supports Windows and macOS; use an existing SDK on Linux")
    base = "https://dl.google.com/android/repository/"
    sdk = BASE / "android-sdk"
    unzip(
        base + "android-ndk-r28c-darwin.zip",
        "ndk.zip",
        "fc20a6bf15a30fb3428c9b60a7308793a362dc6d",
        sdk / "ndk/28.2.13676358",
        "android-ndk-r28c",
    )
    unzip(
        base + "platform-35_r02.zip",
        "platform.zip",
        "0bb560a90a7a2cbd0dd8348224d518b638fe7949",
        sdk / "platforms/android-35",
        "android-35",
    )
    unzip(
        base + "build-tools_r35_macosx.zip",
        "build-tools.zip",
        "93ab8ce91230e067b5add4bfa79919c52b27f072",
        sdk / "build-tools/35.0.0",
        "android-15",
    )
    unzip(
        base + "platform-tools_r37.0.1-darwin.zip",
        "platform-tools.zip",
        "6ae73f4de6452dc57e62ec02b68eed92a4c21661",
        sdk / "platform-tools",
        "platform-tools",
    )
    gradle_url = "https://services.gradle.org/distributions/gradle-8.11.1-bin.zip"
    checksum = "f397b287023acdba1e9f6fc5ea72d22dd63669d59ed4a289a29b1a76eee151c6"
    unzip(gradle_url, "gradle.zip", checksum, BASE / "gradle", "gradle-8.11.1", "sha256")
    # A relocatable public Homebrew bottle supplies only the shader compiler.
    shader_sha = "1313ffbae9ac1559f7160cc89657ecd5fc324895ff46e15a4bd198ec3bbebec0"
    archive = BASE / "downloads/shaderc.tar.gz"
    if not archive.exists():
        auth = json.loads(
            urllib.request.urlopen(
                "https://ghcr.io/token?service=ghcr.io&scope=repository:homebrew/core/shaderc:pull"
            ).read()
        )
        download(
            "https://ghcr.io/v2/homebrew/core/shaderc/blobs/sha256:" + shader_sha,
            archive,
            shader_sha,
            headers={"Authorization": "Bearer " + auth["token"]},
        )
    download(
        "https://ghcr.io/v2/homebrew/core/shaderc/blobs/sha256:" + shader_sha, archive, shader_sha
    )
    if not (BASE / "shaderc").exists():
        with tarfile.open(archive) as tar:
            tar.extractall(BASE, filter="data")
    glslc = next((BASE / "shaderc").glob("*/bin/glslc"))
    env_file = BASE / "env.sh"
    import shlex

    q = shlex.quote
    env_file.write_text(
        f"export ANDROID_HOME={q(str(sdk))}\n"
        f"export ANDROID_NDK={q(str(sdk / 'ndk/28.2.13676358'))}\n"
        f"export GRADLE_USER_HOME={q(str(ROOT / 'artifacts/cache/gradle'))}\n"
        f'export PATH={q(str(glslc.parent))}:{q(str(BASE / "gradle/bin"))}:{q(str(sdk / "platform-tools"))}:"$PATH"\n'
    )
    print(f"Toolchain ready. Source {env_file}", flush=True)


if __name__ == "__main__":
    main()
