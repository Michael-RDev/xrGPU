import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify():
    apk = ROOT / "app/build/outputs/apk/debug/app-debug.apk"
    bundle = ROOT / "artifacts/bundle-parent/bundle"
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    exported = json.loads((ROOT / "artifacts/bundle/bundle.json").read_text(encoding="utf-8"))
    if any(exported["files"].get(name) != digest for name, digest in manifest["files"].items()):
        raise RuntimeError("Selected assets differ from the verified exports")
    identity = json.loads((ROOT / "artifacts/build/android/build-identity.json").read_text())
    sdk = Path(os.environ.get("ANDROID_HOME", ROOT / "artifacts/toolchain/android-sdk"))
    build_tools = sdk / "build-tools/35.0.0"
    aapt = build_tools / ("aapt.exe" if os.name == "nt" else "aapt")
    signer = build_tools / ("apksigner.bat" if os.name == "nt" else "apksigner")
    native = {}
    with zipfile.ZipFile(apk) as archive:
        if archive.read("assets/bundle/bundle.json") != (bundle / "bundle.json").read_bytes():
            raise RuntimeError("Packaged bundle manifest differs from exported manifest")
        for name, expected in manifest["files"].items():
            if "\\" in name or name.startswith("/") or ".." in name.split("/"):
                raise RuntimeError(f"Non-portable asset path: {name}")
            with archive.open("assets/bundle/" + name) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError(f"Packaged asset hash mismatch: {name}")
        for name in ("libxrgpu.so", "libopenxr_loader.so", "libc++_shared.so"):
            data = archive.read("lib/arm64-v8a/" + name)
            packaged_hash = hashlib.sha256(data).hexdigest()
            stripped = (
                ROOT
                / "app/build/intermediates/stripped_native_libs/debug/stripDebugDebugSymbols/out/lib/arm64-v8a"
                / name
            )
            source = (
                stripped if stripped.exists() else ROOT / "artifacts/install/lib/arm64-v8a" / name
            )
            if packaged_hash != digest_file(source):
                raise RuntimeError(f"Packaged native library differs from build output: {name}")
            if name == "libxrgpu.so" and any(
                value.encode() not in data for value in identity.values()
            ):
                raise RuntimeError("Packaged native source/runtime fingerprint mismatch")
            native[name] = packaged_hash
    xml = subprocess.check_output(
        [str(aapt), "dump", "xmltree", str(apk), "AndroidManifest.xml"], text=True
    )
    if "android.permission.INTERNET" in xml or "org.xrgpu.DebugActivity" not in xml:
        raise RuntimeError("Expected an offline debug APK containing the diagnostic activity")
    signature = subprocess.check_output([str(signer), "verify", "--verbose", str(apk)], text=True)
    record = {
        "status": "passed",
        "apk": str(apk),
        "apk_sha256": digest_file(apk),
        "identity": identity,
        "verified_asset_count": len(manifest["files"]),
        "native_sha256": native,
        "internet_permission": False,
        "debug_commands": True,
        "signature_verification": signature.strip(),
        "device_performance_validated": False,
    }
    (ROOT / "artifacts/build-validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    verify()
