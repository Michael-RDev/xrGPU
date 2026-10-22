import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = re.compile(r"run-[0-9a-f]{32}\Z")
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "artifacts/benchmarks"
MAX_REPORT_BYTES = 128 * 1024 * 1024


def import_archive(path, cache=DEFAULT_CACHE):
    """Import original bytes, without manufacturing a database receipt."""
    path, cache = Path(path), Path(cache) / "archives"
    archive_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    staged, seen = [], set()
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 512 * 1024 * 1024:
            raise ValueError("Archive exceeds the uncompressed size limit")
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            parts = entry.filename.split("/")
            if len(parts) != 2 or parts[0] not in {"completed", "partial"}:
                raise ValueError("Expected completed/ or partial/ JSON reports")
            run_id = parts[1].removesuffix(".json")
            if not parts[1].endswith(".json") or not RUN_ID.fullmatch(run_id) or run_id in seen:
                raise ValueError("Invalid or duplicate archive run identity")
            seen.add(run_id)
            if entry.file_size > MAX_REPORT_BYTES:
                raise ValueError("Report exceeds the import size limit")
            raw = archive.read(entry)
            report = json.loads(raw)
            canonical(report)
            if (report.get("schema_version") != 2 or report.get("run_id") != run_id
                    or report.get("execution_location") != "headset"
                    or report.get("backend") not in {"executorch_cpu", "executorch_vulkan"}
                    or report.get("status") not in {"complete", "failed", "canceled"}
                    or not isinstance(report.get("results"), list)
                    or not isinstance(report.get("configuration"), dict)
                    or not isinstance(report.get("device"), dict)):
                raise ValueError("Invalid native archive report")
            if (parts[0] == "completed") != (report["status"] == "complete"):
                raise ValueError("Archive classification disagrees with native status")
            target = cache / parts[0] / run_id
            sha = hashlib.sha256(raw).hexdigest()
            if target.exists() and ((target / "report.json").read_bytes() != raw
                                    or not (target / "source.json").is_file()):
                raise FileExistsError("Run ID already has different or incomplete archive evidence")
            staged.append((target, raw, dict(run_id=run_id, raw_sha256=sha,
                canonical_sha256=digest(report), kind="local_archive", database_receipted=False,
                archive_sha256=archive_hash, archive_member=entry.filename,
                classification=parts[0], backend=report["backend"], native_status=report["status"])))
    # Validate the entire archive before writing anything. Never extract ZIP paths.
    results = []
    for target, raw, provenance in staged:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".import-", dir=target.parent))
            try:
                for name, data in (("report.json", raw), ("source.json", canonical(provenance))):
                    with (temporary / name).open("wb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                temporary.rename(target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        results.append(dict(provenance, path=str(target)))
    return results


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def verify(envelope, run_id):
    if not RUN_ID.fullmatch(run_id) or not isinstance(envelope, dict):
        raise ValueError("Invalid run identity or result envelope")
    report, receipt = envelope.get("report"), envelope.get("receipt")
    if not isinstance(report, dict) or not isinstance(receipt, dict):
        raise ValueError("A complete report and database receipt are required")
    if report.get("schema_version") != 2 or report.get("run_id") != run_id:
        raise ValueError("Expected this run's native benchmark schema version 2 report")
    if report.get("status") not in {"complete", "failed", "canceled"}:
        raise ValueError("Only terminal reports can be imported")
    if report.get("execution_location") != "headset" or report.get("backend") != "executorch_vulkan":
        raise ValueError("Expected native headset Vulkan evidence")
    if not isinstance(report.get("results"), list) or not isinstance(report.get("configuration"), dict) or not isinstance(report.get("device"), dict):
        raise ValueError("Incomplete native report")
    if receipt.get("stored") is not True or receipt.get("run_id") != run_id or receipt.get("sha256") != digest(report):
        raise ValueError("Database receipt does not match this exact report")
    return report, receipt


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self, url="http://localhost:8765"):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "http" or parts.hostname not in {"localhost", "127.0.0.1"} or parts.username or parts.password or parts.path not in {"", "/"} or parts.query or parts.fragment:
            raise ValueError("Use the local xrBench HTTP origin, for example http://localhost:8765")
        if parts.port is not None and not 1 <= parts.port <= 65535:
            raise ValueError("Invalid local port")
        self.url = url.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.token = None

    def get(self, path):
        if self.token is None:
            session = self._get("/api/session")
            self.token = session.get("token")
            if not isinstance(self.token, str) or not self.token:
                raise ValueError("xrBench did not provide a local session")
        return self._get(path)

    def _get(self, path):
        request = urllib.request.Request(self.url + path, headers={"X-XRGPU-Session": self.token or ""})
        with self.opener.open(request, timeout=30) as response:
            if response.headers.get_content_type() != "application/json":
                raise ValueError("Expected JSON from xrBench")
            raw = response.read(MAX_REPORT_BYTES + 1)
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError("Report exceeds the import size limit")
        data = json.loads(raw)
        canonical(data)
        return data

    def runs(self):
        offset, seen = 0, set()
        while True:
            page = self.get(f"/api/v1/results?limit=100&offset={offset}")
            if not isinstance(page, dict) or not isinstance(page.get("runs"), list) or len(page["runs"]) > 100:
                raise ValueError("Invalid results page")
            for row in page["runs"]:
                run_id = row.get("run_id", "")
                if not RUN_ID.fullmatch(run_id):
                    raise ValueError("Invalid run identity in results page")
                if run_id not in seen:
                    seen.add(run_id)
                    yield row
            next_offset = page.get("next_offset")
            if next_offset is None:
                return
            if type(next_offset) is not int or not offset < next_offset <= 100000:
                raise ValueError("Invalid results pagination")
            offset = next_offset

    def result(self, run_id):
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("Invalid run identity")
        envelope = self.get("/api/v1/results/" + run_id)
        verify(envelope, run_id)
        return envelope


def import_result(envelope, cache=DEFAULT_CACHE, source="http://localhost:8765"):
    run_id = envelope.get("report", {}).get("run_id", "")
    report, receipt = verify(envelope, run_id)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / run_id

    def existing():
        try:
            old = {name: json.loads((target / (name + ".json")).read_text()) for name in ("report", "receipt")}
            _, previous = verify(old, run_id)
        except (OSError, ValueError) as error:
            raise FileExistsError("Existing cached evidence is incomplete or corrupt; refusing to overwrite") from error
        if previous != receipt:
            raise FileExistsError("Run ID already has different cached evidence")
        return dict(run_id=run_id, sha256=receipt["sha256"], path=str(target), status="already imported")

    if target.exists():
        return existing()
    temporary = Path(tempfile.mkdtemp(prefix="." + run_id + "-", dir=cache))
    try:
        values = dict(report=report, receipt=receipt, source=dict(origin=source, imported_at=datetime.now(timezone.utc).isoformat(), kind="native_layer_baseline"))
        for name, value in values.items():
            with (temporary / (name + ".json")).open("wb") as stream:
                stream.write(canonical(value))
                stream.flush()
                os.fsync(stream.fileno())
        try:
            temporary.rename(target)
        except OSError as error:
            if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
                return existing()
            raise
        return dict(run_id=run_id, sha256=receipt["sha256"], path=str(target), status="imported")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description="Import committed xrBench baseline evidence without running benchmarks.")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List committed runs")
    importing = commands.add_parser("import", help="Verify and cache reports and database receipts")
    for command in (listing, importing):
        command.add_argument("--url", default="http://localhost:8765")
    selection = importing.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id")
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--archive", type=Path)
    importing.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    try:
        if args.command == "import" and args.archive:
            print(json.dumps(import_archive(args.archive, args.output), indent=2))
            return
        client = Client(args.url)
        if args.command == "list":
            print(json.dumps(list(client.runs()), indent=2))
            return
        rows = list(client.runs()) if args.all else [dict(run_id=args.run_id)]
        for row in rows:
            envelope = client.result(row["run_id"])
            if "receipt" in row and row["receipt"] != envelope["receipt"]:
                raise ValueError("Report receipt changed since the results listing")
            print(json.dumps(import_result(envelope, args.output, client.url)))
        if not rows:
            print("No committed xrBench runs to import.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"xrBench import failed: {error}. Existing cached evidence is preserved.") from error


if __name__ == "__main__":
    main()
