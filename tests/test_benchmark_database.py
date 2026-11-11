"""Database pull protocol tests; synthetic records do not qualify device kernels."""

import copy
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone

import pytest

from runtime import benchmark_database as database
from runtime.benchmark_results import canonical, import_result, report_digest

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("rfc8785")
from psycopg.conninfo import conninfo_to_dict, make_conninfo  # noqa: E402


@pytest.fixture(autouse=True)
def clean_pg_environment(monkeypatch):
    for key in os.environ:
        if key.startswith("PG"):
            monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    "dsn",
    [
        "host=db.example sslmode=disable",
        "host=localhost",
        "dbname=xrbench",
        "host=127.0.0.1 hostaddr=203.0.113.1",
        "host=127.0.0.1,db.example",
        "service=production host=127.0.0.1",
        "host=127.0.0.1,",
    ],
)
def test_remote_and_ambiguous_addresses_force_verified_tls(dsn):
    assert database.connection_options(dsn) == dict(sslmode="verify-full", gssencmode="disable")


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "/tmp/postgres", "127.0.0.1,::1"])
def test_explicit_local_transport(host):
    assert database.connection_options("host=" + host) == dict(host=host, hostaddr="")


def test_libpq_environment_cannot_redirect_to_plaintext_remote(monkeypatch):
    monkeypatch.setenv("PGHOSTADDR", "203.0.113.1")
    assert database.connection_options("host=127.0.0.1")["sslmode"] == "verify-full"
    monkeypatch.setenv("PGSERVICE", "remote")
    assert database.connection_options("host=127.0.0.1")["sslmode"] == "verify-full"


def test_connection_failures_and_invalid_dsns_never_expose_secrets(monkeypatch):
    with pytest.raises(database.DatabaseError, match="Invalid PostgreSQL") as error:
        database.connection_options("private-password")
    assert "private-password" not in str(error.value)
    calls = []

    def fail(dsn, **options):
        calls.append(options)
        raise psycopg.OperationalError("postgresql://secret:private-password@private-host")

    monkeypatch.setattr(psycopg, "connect", fail)
    with pytest.raises(database.DatabaseError) as error:
        list(database.PostgresClient("host=db.example password=private-password").results("a" * 64))
    assert calls == [dict(connect_timeout=5, sslmode="verify-full", gssencmode="disable")]
    assert "private-password" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize(
    "package,run", [("latest", None), ("A" * 64, None), ("a" * 64, "../../escape"), (None, None)]
)
def test_invalid_selection_never_connects(monkeypatch, package, run):
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: pytest.fail("unexpected connection"))
    with pytest.raises(ValueError):
        list(database.PostgresClient("host=127.0.0.1").results(package, run))


def test_hosted_hash_matches_the_actual_javascript_canonicalizer():
    if not shutil.which("node"):
        pytest.skip("Node required for independent cross-language hash check")
    value = {
        "samples": [1.0, -0.0, 1e-7, 1e20, 0.000001, 0.123456789],
        "device": "Casque é 🥽",
        "\U0001f600": 1,
        "\ue000": 2,
    }
    script = """const {createHash} = require('node:crypto');
    const value = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
    function canonical(v) {
      if (Array.isArray(v)) return '[' + v.map(canonical).join(',') + ']';
      if (v && typeof v === 'object') return '{' + Object.keys(v).sort().map(
        k => JSON.stringify(k) + ':' + canonical(v[k])).join(',') + '}';
      return JSON.stringify(v);
    }
    process.stdout.write(createHash('sha256').update(canonical(value)).digest('hex'));
    """
    actual = subprocess.run(
        ["node", "-e", script], input=json.dumps(value), capture_output=True, text=True, check=True
    ).stdout
    assert report_digest(value, "rfc8785") == actual
    assert report_digest(value) != actual


@pytest.fixture
def native_report():
    package = "a" * 64
    return dict(
        schema_version=3,
        run_id="run-" + "1" * 32,
        status="complete",
        execution_location="headset",
        backend="xrgpu_vulkan",
        suite_revision="s" * 64,
        runtime_revision="b" * 40,
        native_package_sha256=package,
        native_catalog_sha256="c" * 64,
        device={"model": "Synthetic é 🥽"},
        started_at="2026-10-27T00:00:00Z",
        finished_at="2026-10-27T00:00:01Z",
        configuration=dict(native_package_sha256=package, mode="isolated", preset="standard"),
        results=[
            dict(
                case="ai.onnx.Abs",
                phase="isolated",
                trial=0,
                operation="ai.onnx.Abs",
                status="passed",
                correctness={"passed": True},
                samples=[{"gpu_ms": 1.0}],
            )
        ],
    )


@pytest.fixture
def store():
    if os.environ.get("XRGPU_TEST_POSTGRES") != "1":
        pytest.skip("Opt in to isolated local PostgreSQL tests with XRGPU_TEST_POSTGRES=1")
    base = os.environ.get("XRGPU_TEST_DATABASE_URL")
    if not base or "host" not in database.connection_options(base):
        pytest.fail("An explicitly local XRGPU_TEST_DATABASE_URL is required")
    name = "xrgpu_pull_test_" + uuid.uuid4().hex
    params = conninfo_to_dict(base)
    params["dbname"] = name
    dsn = make_conninfo(**params)
    with psycopg.connect(base, autocommit=True) as connection:
        connection.execute(
            psycopg.sql.SQL("CREATE DATABASE {}").format(psycopg.sql.Identifier(name))
        )
    try:
        with psycopg.connect(dsn) as connection:
            connection.execute("""CREATE SCHEMA xrgpu_benchmarks;
                CREATE TABLE xrgpu_benchmarks.benchmark_runs (
                    run_id text PRIMARY KEY, report jsonb, report_sha256 text, received_at timestamptz,
                    schema_version integer, suite_revision text, runtime_revision text, backend text,
                    status text, device jsonb, configuration jsonb, mode text, preset text,
                    started_at timestamptz, finished_at timestamptz);
                CREATE TABLE xrgpu_benchmarks.benchmark_results (
                    run_id text, case_id text, phase text, trial integer, operation text, status text,
                    correctness jsonb, samples jsonb, result jsonb,
                    PRIMARY KEY (run_id, case_id, phase, trial));""")
        yield dsn
    finally:
        with psycopg.connect(base, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                    psycopg.sql.Identifier(name)
                )
            )


@pytest.fixture
def canonical_store(store):
    with psycopg.connect(store) as connection:
        connection.execute(
            "ALTER TABLE xrgpu_benchmarks.benchmark_runs ADD COLUMN report_canonical text"
        )
    return store


def seed(dsn, report, algorithm="python", preserve_canonical=False):
    config = report["configuration"]
    received = datetime(2026, 10, 27, tzinfo=timezone.utc)
    with psycopg.connect(dsn) as connection:
        connection.execute(
            """INSERT INTO xrgpu_benchmarks.benchmark_runs
            (run_id,report,report_sha256,received_at,schema_version,suite_revision,
             runtime_revision,backend,status,device,configuration,mode,preset,started_at,finished_at)
            VALUES
            (%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s)""",
            (
                report["run_id"],
                json.dumps(report),
                report_digest(report, algorithm),
                received,
                report["schema_version"],
                report["suite_revision"],
                report["runtime_revision"],
                report["backend"],
                report["status"],
                json.dumps(report["device"]),
                json.dumps(config),
                config["mode"],
                config["preset"],
                report["started_at"],
                report["finished_at"],
            ),
        )
        if preserve_canonical:
            connection.execute(
                """UPDATE xrgpu_benchmarks.benchmark_runs SET report_canonical=%s
                WHERE run_id=%s""",
                (canonical(report).decode("utf-8"), report["run_id"]),
            )
        for row in report["results"]:
            connection.execute(
                """INSERT INTO xrgpu_benchmarks.benchmark_results VALUES
                (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)""",
                (
                    report["run_id"],
                    row["case"],
                    row["phase"],
                    row["trial"],
                    row["operation"],
                    row["status"],
                    json.dumps(row.get("correctness")),
                    json.dumps(row["samples"]),
                    json.dumps(row),
                ),
            )


@pytest.mark.parametrize(
    "schema,backend,algorithm",
    [
        (3, "xrgpu_vulkan", "python"),
        (3, "xrgpu_vulkan", "rfc8785"),
        (2, "executorch_vulkan", "python"),
        (2, "executorch_cpu", "rfc8785"),
        (2, "vulkan_resident", "python"),
    ],
)
def test_pull_exact_version_preserves_original_and_retries(
    store, native_report, tmp_path, schema, backend, algorithm
):
    report = native_report
    report.update(schema_version=schema, backend=backend, status="failed")
    if schema == 2:
        report["configuration"]["kernel_package"] = dict(content_sha256="a" * 64)
    original = copy.deepcopy(report)
    seed(store, report, algorithm)
    other = copy.deepcopy(report)
    other["run_id"] = "run-" + "2" * 32
    other["native_package_sha256"] = other["configuration"]["native_package_sha256"] = "b" * 64
    if schema == 2:
        other["configuration"]["kernel_package"]["content_sha256"] = "b" * 64
    seed(store, other, algorithm)
    envelopes = list(database.PostgresClient(store).results("a" * 64))
    assert len(envelopes) == 1 and envelopes[0]["report"] == original
    assert envelopes[0]["receipt"]["sha256"] == report_digest(report, algorithm)
    assert (
        import_result(envelopes[0], tmp_path, "postgresql:xrgpu_benchmarks")["status"] == "imported"
    )
    again = list(database.PostgresClient(store).results("a" * 64, report["run_id"]))
    assert import_result(again[0], tmp_path)["status"] == "already imported"
    assert canonical(
        json.loads((tmp_path / report["run_id"] / "report.json").read_text())
    ) == canonical(original)
    assert store not in (tmp_path / report["run_id"] / "source.json").read_text()
    assert not list(database.PostgresClient(store).results("b" * 64, report["run_id"]))


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE xrgpu_benchmarks.benchmark_runs SET report_sha256=repeat('0',64)",
        "UPDATE xrgpu_benchmarks.benchmark_runs SET status='failed'",
        "UPDATE xrgpu_benchmarks.benchmark_runs SET finished_at=finished_at + interval '1 second'",
        "UPDATE xrgpu_benchmarks.benchmark_results SET status='failed'",
        "UPDATE xrgpu_benchmarks.benchmark_results SET samples='[]'::jsonb",
        "UPDATE xrgpu_benchmarks.benchmark_results SET result='{}'::jsonb",
        "DELETE FROM xrgpu_benchmarks.benchmark_results",
    ],
)
def test_corrupt_database_evidence_never_cached(store, native_report, tmp_path, mutation):
    seed(store, native_report)
    with psycopg.connect(store) as connection:
        connection.execute(mutation)
    with pytest.raises(ValueError):
        for envelope in database.PostgresClient(store).results("a" * 64):
            import_result(envelope, tmp_path)
    assert not list(tmp_path.iterdir())


def test_pull_is_read_only_even_with_writable_credentials(store, native_report, monkeypatch):
    seed(store, native_report)

    def attempt_write(connection, report):
        assert connection.execute("SHOW transaction_read_only").fetchone()[0] == "on"
        connection.execute("CREATE TABLE public.must_not_exist (id integer)")

    monkeypatch.setattr(database, "_verify_results", attempt_write)
    with pytest.raises(database.DatabaseError):
        list(database.PostgresClient(store).results("a" * 64))
    with psycopg.connect(store) as connection:
        assert (
            connection.execute("SELECT to_regclass('public.must_not_exist')").fetchone()[0] is None
        )


def test_snapshot_excludes_concurrent_insert_and_next_pull_sees_it(
    store, native_report, monkeypatch
):
    seed(store, native_report)
    original_verify = database._verify_results
    calls = []

    def verify_and_insert(connection, report):
        original_verify(connection, report)
        assert connection.execute("SHOW transaction_isolation").fetchone()[0] == "repeatable read"
        if not calls:
            other = copy.deepcopy(report)
            other["run_id"] = "run-" + "2" * 32
            seed(store, other)
        calls.append(report["run_id"])

    monkeypatch.setattr(database, "_verify_results", verify_and_insert)
    assert len(list(database.PostgresClient(store).results("a" * 64))) == 1
    assert len(list(database.PostgresClient(store).results("a" * 64))) == 2


def test_oversized_report_is_rejected_before_transfer(store, native_report, monkeypatch):
    seed(store, native_report)
    monkeypatch.setattr(database, "MAX_REPORT_BYTES", 10)
    with pytest.raises(ValueError, match="size limit"):
        list(database.PostgresClient(store).results("a" * 64))


def test_hosted_exponent_hash_survives_jsonb_number_expansion(store, native_report, tmp_path):
    native_report["device"]["large_counter"] = 1e20
    seed(store, native_report, "rfc8785")
    envelope = next(database.PostgresClient(store).results("a" * 64))
    # PostgreSQL expands 1e20 and the driver decodes it as an integer.
    assert type(envelope["report"]["device"]["large_counter"]) is int
    assert import_result(envelope, tmp_path)["status"] == "imported"
    assert import_result(envelope, tmp_path)["status"] == "already imported"


@pytest.mark.parametrize("value", [1e20, -0.0])
def test_python_canonical_report_preserves_numbers_across_sources(
    canonical_store, native_report, tmp_path, value
):
    native_report["device"]["reading"] = value
    native_report["configuration"]["reading"] = value
    native_report["results"][0]["correctness"]["max_abs_error"] = value
    native_report["native_report"] = {"nested": [{"reading": value}]}
    original = canonical(native_report)
    seed(canonical_store, native_report, preserve_canonical=True)
    envelope = next(database.PostgresClient(canonical_store).results("a" * 64))
    assert canonical(envelope["report"]) == original
    assert envelope["receipt"]["canonicalization"] == "python"
    http = dict(report=native_report, receipt=dict(
        stored=True, run_id=native_report["run_id"], sha256=report_digest(native_report)
    ))
    assert import_result(http, tmp_path)["status"] == "imported"
    target = tmp_path / native_report["run_id"]
    cached = {path.name: path.read_bytes() for path in target.iterdir()}
    for _ in range(2):
        assert import_result(envelope, tmp_path, "postgresql:xrgpu_benchmarks")["status"] == "already imported"
    assert {path.name: path.read_bytes() for path in target.iterdir()} == cached


@pytest.mark.parametrize("value", [1e20, -0.0])
def test_legacy_python_report_requires_original_payload(store, native_report, value):
    native_report["device"]["reading"] = value
    seed(store, native_report)
    with pytest.raises(ValueError, match="original canonical report"):
        list(database.PostgresClient(store).results("a" * 64))


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE xrgpu_benchmarks.benchmark_runs SET report_canonical='{}'",
        "UPDATE xrgpu_benchmarks.benchmark_runs SET report=report || '{\"device\":{\"model\":\"changed\"}}'::jsonb",
        "UPDATE xrgpu_benchmarks.benchmark_runs SET report=report || '{\"configuration\":true}'::jsonb",
    ],
)
def test_canonical_payload_and_jsonb_must_agree(canonical_store, native_report, mutation):
    seed(canonical_store, native_report, preserve_canonical=True)
    with psycopg.connect(canonical_store) as connection:
        connection.execute(mutation)
    with pytest.raises(ValueError, match="canonical report|Canonical report"):
        list(database.PostgresClient(canonical_store).results("a" * 64))


def test_canonical_payload_size_is_bounded_before_transfer(canonical_store, native_report, monkeypatch):
    seed(canonical_store, native_report, preserve_canonical=True)
    with psycopg.connect(canonical_store) as connection:
        limit = connection.execute(
            "SELECT octet_length(report::text) + 1 FROM xrgpu_benchmarks.benchmark_runs"
        ).fetchone()[0]
        connection.execute(
            "UPDATE xrgpu_benchmarks.benchmark_runs SET report_canonical=report_canonical || repeat(' ', %s)",
            (limit,),
        )
    monkeypatch.setattr(database, "MAX_REPORT_BYTES", limit)
    with pytest.raises(ValueError, match="size limit"):
        list(database.PostgresClient(canonical_store).results("a" * 64))


def test_jsonb_numeric_comparison_preserves_boolean_types():
    assert database._same_jsonb({"values": [1e20, -0.0]}, {"values": [10**20, 0.0]})
    assert not database._same_jsonb({"values": [True]}, {"values": [1]})
    assert not database._same_jsonb({"values": [False]}, {"values": [0.0]})


def test_report_package_mismatch_is_rejected_even_with_valid_hash(store, native_report):
    native_report["native_package_sha256"] = "b" * 64
    seed(store, native_report)
    with pytest.raises(ValueError, match="package identity"):
        list(database.PostgresClient(store).results("a" * 64))


def test_stream_has_no_history_page_limit(store, native_report):
    for index in range(105):
        native_report["run_id"] = "run-" + f"{index:032x}"
        seed(store, native_report)
    envelopes = list(database.PostgresClient(store).results("a" * 64))
    assert len(envelopes) == len({e["report"]["run_id"] for e in envelopes}) == 105


def test_cli_pull_and_missing_run_exit_status(store, native_report, tmp_path):
    seed(store, native_report, "rfc8785")
    command = [
        sys.executable,
        "-m",
        "runtime.benchmark_results",
        "pull",
        "--package-sha256",
        "a" * 64,
        "--output",
        str(tmp_path),
    ]
    env = dict(os.environ, XRGPU_DATABASE_URL=store)
    for status in ("imported", "already imported"):
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == status
        assert store not in result.stdout + result.stderr
    missing = subprocess.run(
        command + ["--run-id", "run-" + "0" * 32], env=env, capture_output=True, text=True
    )
    assert missing.returncode != 0 and "No committed run matches" in missing.stderr


def test_cli_requires_secrets_in_environment(tmp_path):
    env = dict(os.environ)
    env.pop("XRGPU_DATABASE_URL", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "runtime.benchmark_results",
            "pull",
            "--package-sha256",
            "a" * 64,
            "--output",
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "environment variable" in result.stderr
    assert not list(tmp_path.iterdir())


def test_partial_cache_cannot_be_reported_as_imported(tmp_path, native_report):
    envelope = dict(
        report=native_report,
        receipt=dict(
            stored=True, run_id=native_report["run_id"], sha256=report_digest(native_report)
        ),
    )
    import_result(envelope, tmp_path)
    (tmp_path / native_report["run_id"] / "source.json").unlink()
    with pytest.raises(FileExistsError):
        import_result(envelope, tmp_path)
