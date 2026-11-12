import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from runtime.benchmark_results import Client, NoRedirect, digest, import_result, report_digest


@pytest.fixture
def envelope():
    report = dict(
        schema_version=2,
        run_id="run-" + "a" * 32,
        status="complete",
        execution_location="headset",
        backend="executorch_vulkan",
        device={"model": "synthetic fixture"},
        configuration={},
        results=[],
    )
    return dict(
        report=report, receipt=dict(stored=True, run_id=report["run_id"], sha256=digest(report))
    )


def test_atomic_cache_idempotence_conflict_and_preserved_report(tmp_path, envelope):
    original = copy.deepcopy(envelope)
    first = import_result(envelope, tmp_path)
    assert first["status"] == "imported"
    assert import_result(envelope, tmp_path)["status"] == "already imported"
    path = tmp_path / envelope["report"]["run_id"]
    assert json.loads((path / "report.json").read_text()) == original["report"]
    assert json.loads((path / "receipt.json").read_text()) == original["receipt"]
    envelope["report"]["device"]["model"] = "changed"
    envelope["receipt"]["sha256"] = digest(envelope["report"])
    with pytest.raises(FileExistsError):
        import_result(envelope, tmp_path)
    assert json.loads((path / "report.json").read_text()) == original["report"]
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("database_first", [False, True])
def test_same_report_can_be_retried_from_another_source(tmp_path, envelope, database_first):
    database = copy.deepcopy(envelope)
    database["receipt"].update(
        storage="postgresql", canonicalization="python", received_at="2026-11-10T00:00:00+00:00"
    )
    sources = [(envelope, "http://localhost:8766"), (database, "postgresql:xrgpu_benchmarks")]
    if database_first:
        sources.reverse()
    assert import_result(sources[0][0], tmp_path, sources[0][1])["status"] == "imported"
    target = tmp_path / envelope["report"]["run_id"]
    original = {path.name: path.read_bytes() for path in target.iterdir()}
    assert import_result(sources[1][0], tmp_path, sources[1][1])["status"] == "already imported"
    assert {path.name: path.read_bytes() for path in target.iterdir()} == original


@pytest.mark.parametrize("database_first", [False, True])
def test_hosted_report_retries_allow_jsonb_number_normalization(
    tmp_path, envelope, database_first
):
    pytest.importorskip("rfc8785")
    envelope["report"]["device"]["readings"] = [1e20, -0.0]
    envelope["receipt"].update(
        sha256=report_digest(envelope["report"], "rfc8785"), canonicalization="rfc8785"
    )
    database = copy.deepcopy(envelope)
    database["report"]["device"]["readings"] = [10**20, 0]
    sources = [envelope, database]
    if database_first:
        sources.reverse()
    assert import_result(sources[0], tmp_path)["status"] == "imported"
    target = tmp_path / envelope["report"]["run_id"]
    original = {path.name: path.read_bytes() for path in target.iterdir()}
    assert import_result(sources[1], tmp_path)["status"] == "already imported"
    assert {path.name: path.read_bytes() for path in target.iterdir()} == original
    changed = copy.deepcopy(database)
    changed["report"]["device"]["readings"][1] = 1
    changed["receipt"]["sha256"] = report_digest(changed["report"], "rfc8785")
    with pytest.raises(FileExistsError):
        import_result(changed, tmp_path)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == original


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"origin": ""},
        {"imported_at": "invalid"},
        {"imported_at": "2026-11-10T00:00:00"},
        {"kind": "local_archive"},
        {"backend": "executorch_cpu"},
        {"execution_location": "phone"},
        {"package_sha256": "b" * 64},
    ],
)
def test_corrupt_source_provenance_is_rejected_without_changing_cache(tmp_path, envelope, change):
    envelope["report"]["configuration"]["kernel_package"] = dict(content_sha256="a" * 64)
    envelope["receipt"]["sha256"] = digest(envelope["report"])
    import_result(envelope, tmp_path)
    target = tmp_path / envelope["report"]["run_id"]
    source = target / "source.json"
    provenance = json.loads(source.read_text())
    source.write_text(json.dumps(dict(provenance, **change) if change else {}))
    original = {path.name: path.read_bytes() for path in target.iterdir()}
    with pytest.raises(FileExistsError, match="incomplete or corrupt"):
        import_result(envelope, tmp_path)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == original


def test_legacy_source_provenance_can_still_be_retried(tmp_path, envelope):
    envelope["report"]["configuration"]["kernel_package"] = dict(content_sha256="a" * 64)
    envelope["receipt"]["sha256"] = digest(envelope["report"])
    import_result(envelope, tmp_path)
    target = tmp_path / envelope["report"]["run_id"]
    (target / "source.json").write_text(json.dumps(dict(
        origin="http://localhost:8766", imported_at="2026-10-26T00:00:00+00:00",
        kind="native_layer_baseline",
    )))
    original = {path.name: path.read_bytes() for path in target.iterdir()}
    assert import_result(envelope, tmp_path)["status"] == "already imported"
    assert {path.name: path.read_bytes() for path in target.iterdir()} == original


@pytest.mark.parametrize("change", ["hash", "id", "status", "schema", "backend", "nan", "missing"])
def test_invalid_evidence_never_written(tmp_path, envelope, change):
    if change == "hash":
        envelope["receipt"]["sha256"] = "0" * 64
    if change == "id":
        envelope["report"]["run_id"] = "../../escape"
    if change == "status":
        envelope["report"]["status"] = "running"
    if change == "schema":
        envelope["report"]["schema_version"] = 1
    if change == "backend":
        envelope["report"]["backend"] = "webgpu"
    if change == "nan":
        envelope["report"]["device"]["value"] = float("nan")
    if change == "missing":
        envelope.pop("receipt")
    with pytest.raises(ValueError):
        import_result(envelope, tmp_path)
    assert not list(tmp_path.iterdir())


def test_failed_write_leaves_no_partial_evidence(tmp_path, envelope, monkeypatch):
    import runtime.benchmark_results as module

    monkeypatch.setattr(
        module.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk unavailable"))
    )
    with pytest.raises(OSError):
        import_result(envelope, tmp_path)
    assert not list(tmp_path.iterdir())


def test_corrupt_existing_cache_is_not_repaired_silently(tmp_path, envelope):
    import_result(envelope, tmp_path)
    path = tmp_path / envelope["report"]["run_id"] / "receipt.json"
    path.write_text("{}")
    with pytest.raises(FileExistsError):
        import_result(envelope, tmp_path)
    assert path.read_text() == "{}"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://127.0.0.1.evil",
        "http://localhost:8765/api",
        "http://user:secret@localhost",
        "file:///tmp/data",
        "http://localhost?x=1",
    ],
)
def test_importer_only_connects_to_local_origin(url):
    with pytest.raises(ValueError):
        Client(url)


def test_read_only_http_session_pagination_and_disconnect(tmp_path, envelope):
    calls = []
    second = copy.deepcopy(envelope)
    second["report"]["run_id"] = "run-" + "b" * 32
    second["receipt"].update(run_id=second["report"]["run_id"], sha256=digest(second["report"]))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            calls.append(self.path)
            if self.path == "/api/session":
                data = {"token": "test-session"}
            else:
                assert self.headers["X-XRGPU-Session"] == "test-session"
                if self.path.endswith("offset=0"):
                    data = {"runs": [{"run_id": envelope["report"]["run_id"]}], "next_offset": 1}
                elif self.path.endswith("offset=1"):
                    data = {"runs": [{"run_id": second["report"]["run_id"]}], "next_offset": None}
                else:
                    data = envelope
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = Client(f"http://127.0.0.1:{server.server_port}")
    try:
        rows = list(client.runs())
        assert len(rows) == 2
        downloaded = client.result(rows[0]["run_id"])
        import_result(downloaded, tmp_path)
        assert calls.count("/api/session") == 1
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    with pytest.raises(OSError):
        client.result(rows[0]["run_id"])
    assert import_result(downloaded, tmp_path)["status"] == "already imported"
    assert NoRedirect().redirect_request(None, None, 302, None, None, "http://elsewhere") is None
