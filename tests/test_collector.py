import copy
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from collector.server import Store, handler
from runtime.layer_report import upload, validate_report


def report():
    return {"schema_version": 1, "kind": "layer_suite", "run_id": "test-run",
            "status": "complete", "execution_location": "host", "device": {"architecture": "test"},
            "backend": "test", "suite_revision": "test-suite", "results": [
                {"case": "linear", "status": "passed", "correctness": {"passed": True},
                 "samples": [{"execution_ms": 1.0}]}]}


def test_database_persists_and_retries_cannot_overwrite(tmp_path):
    path = tmp_path / "runs.sqlite3"
    store = Store(path)
    first = store.put(report())
    assert first["stored"] and not first["duplicate"]
    reopened = Store(path)
    assert reopened.get("test-run") == report()
    assert reopened.put(report())["duplicate"]
    altered = report()
    altered["backend"] = "different"
    with pytest.raises(FileExistsError):
        reopened.put(altered)
    assert len(reopened.recent(20)) == 1
    assert reopened.get("test-run") == report()


@pytest.mark.parametrize("change", [
    {"status": "running"}, {"execution_location": "unspecified"}, {"schema_version": 9},
    {"device": "test"}, {"results": [{"case": "bad", "status": "passed"}]},
    {"results": [{"case": "bad", "status": "passed", "correctness": {"passed": True}, "samples": [{"execution_ms": -1}]}]},
    {"device": {"invalid": float("nan")}},
])
def test_invalid_reports_rejected(change):
    with pytest.raises(ValueError):
        validate_report({**report(), **change})


def test_duplicate_case_rejected():
    value = report()
    value["results"].append(copy.deepcopy(value["results"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_report(value)


def test_client_rejects_false_storage_acknowledgement(monkeypatch):
    import io
    from types import SimpleNamespace

    fake = SimpleNamespace(open=lambda *a, **k: io.BytesIO(b'{"stored":true,"run_id":"test-run","sha256":"wrong"}'))
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: fake)
    with pytest.raises(RuntimeError, match="exact report"):
        upload(report(), "https://collector.example/v1/runs", "secret")


def test_client_does_not_follow_redirects():
    from runtime.layer_report import NoRedirect
    assert NoRedirect().redirect_request(None, None, 302, None, None, "https://elsewhere.example") is None


def test_live_http_upload_auth_receipt_and_retrieval(tmp_path):
    token = "test-token-at-least-16-characters"
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(Store(tmp_path / "runs.db"), token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/runs"
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            upload(report(), url)
        assert error.value.code == 401
        assert upload(report(), url, token)["stored"]
        assert upload(report(), url, token)["duplicate"]
        request = urllib.request.Request(url + "/test-run", headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.load(response) == report()
        request = urllib.request.Request(url, data=b'{"wrong":1}', headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
