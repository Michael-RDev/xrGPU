import copy
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from runtime.layer_report import upload, validate_report


def report():
    return {
        "schema_version": 1,
        "kind": "layer_suite",
        "run_id": "test-run",
        "status": "complete",
        "execution_location": "host",
        "device": {"architecture": "test"},
        "backend": "test",
        "suite_revision": "test-suite",
        "results": [
            {
                "case": "linear",
                "status": "passed",
                "correctness": {"passed": True},
                "samples": [{"execution_ms": 1.0}],
            }
        ],
    }


@pytest.mark.parametrize(
    "change",
    [
        {"status": "running"},
        {"execution_location": "unspecified"},
        {"schema_version": 9},
        {"device": "test"},
        {"results": [{"case": "bad", "status": "passed"}]},
        {
            "results": [
                {
                    "case": "bad",
                    "status": "passed",
                    "correctness": {"passed": True},
                    "samples": [{"execution_ms": -1}],
                }
            ]
        },
        {"device": {"invalid": float("nan")}},
    ],
)
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

    fake = SimpleNamespace(
        open=lambda *a, **k: io.BytesIO(b'{"stored":true,"run_id":"test-run","sha256":"wrong"}')
    )
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: fake)
    with pytest.raises(RuntimeError, match="exact report"):
        upload(report(), "https://collector.example/v1/runs", "secret")


def test_client_does_not_follow_redirects():
    from runtime.layer_report import NoRedirect

    assert (
        NoRedirect().redirect_request(None, None, 302, None, None, "https://elsewhere.example")
        is None
    )


def test_live_http_client_sends_auth_identity_and_exact_payload():
    """Exercise xrGPU's client. The collector database belongs to xrBench."""
    import hashlib

    token = "test-token-at-least-16-characters"
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + token:
                self.send_error(401)
                return
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, self.headers["Idempotency-Key"], body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "stored": True,
                        "run_id": "test-run",
                        "sha256": hashlib.sha256(body).hexdigest(),
                    }
                ).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/runs"
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            upload(report(), url)
        assert error.value.code == 401
        assert upload(report(), url, token)["stored"]
        assert received == [("/v1/runs", "test-run", validate_report(report()))]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
