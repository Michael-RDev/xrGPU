"""Compile Android sources and run the tokenizer/acceptance JVM integration test."""
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_SHA = "3ea61b2a06e31edf1c91134fe9106b0ebb16628be169f3db75bc7a2b06b45796"


def main():
    jar = ROOT / "artifacts/test-deps/json.jar"
    if not jar.exists():
        jar.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen("https://repo.maven.apache.org/maven2/org/json/json/20250517/json-20250517.jar") as src, jar.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    if hashlib.sha256(jar.read_bytes()).hexdigest() != JSON_SHA:
        raise RuntimeError("JSON test dependency checksum mismatch")
    sdk = Path(os.environ.get("ANDROID_HOME", ROOT / "artifacts/toolchain/android-sdk"))
    android = sdk / "platforms/android-35/android.jar"
    dest = ROOT / "artifacts/host-java"
    dest.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "app/src/main/java/org/xrgpu").glob("*.java"))
    subprocess.run(["javac", "-Xlint:all", "-cp", str(android), "-d", str(dest), *map(str, sources)], check=True)
    from tests.test_report import evidence
    fixture = ROOT / "artifacts/test-evidence.json"
    fixture.write_text(json.dumps(evidence("vr")))
    cp = os.pathsep.join(map(str, (jar, dest)))
    subprocess.run(["javac", "-cp", cp, "-d", str(dest), str(ROOT / "tests/java/HostTests.java")], check=True)
    subprocess.run(["java", "-cp", cp, "org.xrgpu.HostTests", str(ROOT / "artifacts/bundle"), str(fixture)], check=True)


if __name__ == "__main__":
    main()
