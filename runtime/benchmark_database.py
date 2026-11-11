"""Read committed xrBench evidence; this module never writes to PostgreSQL."""

import hashlib
import ipaddress
import json
import os
from datetime import datetime
from decimal import Decimal

from runtime.benchmark_results import (
    MAX_REPORT_BYTES,
    RUN_ID,
    SHA256,
    canonical,
    package_digest,
    report_digest,
    verify,
)


class DatabaseError(RuntimeError):
    """A database failure safe to display without connection secrets."""


def connection_options(dsn):
    """Match xrBench's verified TLS policy, including libpq environment overrides."""
    try:
        from psycopg.conninfo import conninfo_to_dict
    except ImportError:
        raise DatabaseError(
            "Install database support with `uv sync --extra database --inexact`."
        ) from None
    try:
        settings = conninfo_to_dict(dsn)
    except Exception:
        raise DatabaseError("Invalid PostgreSQL connection configuration.") from None

    def effective(key):
        return settings.get(key, os.environ.get("PG" + key.upper(), ""))

    def local(value, sockets=False):
        if sockets and value.startswith(("/", "@")):
            return True
        try:
            return ipaddress.ip_address(value).is_loopback
        except ValueError:
            return False

    hosts, addresses = effective("host"), effective("hostaddr")
    if (
        not (settings.get("service") or os.environ.get("PGSERVICE"))
        and hosts
        and all(local(host, sockets=True) for host in hosts.split(","))
        and (not addresses or all(local(address) for address in addresses.split(",")))
    ):
        return dict(host=hosts, hostaddr=addresses)
    return dict(sslmode="verify-full", gssencmode="disable")


def _same_jsonb(left, right):
    """Compare JSON values while allowing PostgreSQL's numeric normalization."""
    if type(left) in (int, float) and type(right) in (int, float):
        return Decimal(str(left)) == Decimal(str(right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_jsonb(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_jsonb(a, b) for a, b in zip(left, right))
    return left == right


def _envelope(row, expected_package, canonical_report=None):
    run_id, report, sha, received, *indexed = row
    if not isinstance(report, dict):
        raise ValueError("Stored report is missing or exceeds the import size limit")
    if len(canonical(report)) > MAX_REPORT_BYTES:
        raise ValueError("Report exceeds the import size limit")
    stored_report = report
    if canonical_report is not None:
        if not isinstance(canonical_report, str) or not canonical_report:
            raise ValueError("Canonical report is invalid or exceeds the import size limit")
        payload = canonical_report.encode("utf-8")
        if len(payload) > MAX_REPORT_BYTES:
            raise ValueError("Canonical report exceeds the import size limit")
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError("Stored canonical report integrity check failed")
        report = json.loads(payload)
        if not _same_jsonb(report, stored_report):
            raise ValueError("Canonical report disagrees with the stored JSONB report")
    # New Python ingestion preserves the exact canonical payload. Older rows and
    # hosted JCS reports can be verified directly only when JSONB retains enough information.
    algorithm = "python"
    if report_digest(report) != sha:
        algorithm = "rfc8785"
        if report_digest(report, algorithm) != sha:
            raise ValueError(
                "Stored report integrity check failed. A legacy Python-hashed report may need "
                "its original canonical report restored to report_canonical by the database owner."
            )
    receipt = dict(
        stored=True,
        storage="postgresql",
        run_id=run_id,
        sha256=sha,
        canonicalization=algorithm,
        received_at=received.isoformat(),
    )
    envelope = dict(report=report, receipt=receipt)
    verify(envelope, run_id)
    if package_digest(report) != expected_package:
        raise ValueError("Stored report does not match the requested package digest")
    config = stored_report["configuration"]
    expected = [
        stored_report.get(k)
        for k in (
            "schema_version",
            "suite_revision",
            "runtime_revision",
            "backend",
            "status",
            "device",
        )
    ] + [config, config.get("mode"), config.get("preset")]
    # JSON equality must not equate a boolean to an integer.
    if canonical(indexed[:9]) != canonical(expected):
        raise ValueError("Indexed run metadata disagrees with the stored report")
    try:
        times = [
            datetime.fromisoformat(report[k].replace("Z", "+00:00"))
            for k in ("started_at", "finished_at")
        ]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid stored report timestamps") from None
    if any(t.tzinfo is None for t in times) or times[1] < times[0] or indexed[9:] != times:
        raise ValueError("Indexed run timestamps disagree with the stored report")
    return envelope


def _verify_results(connection, report):
    stored = connection.execute(
        """SELECT case_id, phase, trial, operation, status,
        correctness, samples, result FROM xrgpu_benchmarks.benchmark_results
        WHERE run_id=%s ORDER BY case_id, phase, trial LIMIT %s""",
        (report["run_id"], len(report["results"]) + 1),
    ).fetchall()
    expected = []
    for result in report["results"]:
        if not isinstance(result, dict):
            raise ValueError("Invalid stored result")
        expected.append(
            [
                result.get(k)
                for k in ("case", "phase", "trial", "operation", "status", "correctness", "samples")
            ]
            + [result]
        )
    # Compare by identity without relying on the database's collation order.
    actual = {tuple(row[:3]): list(row) for row in stored}
    wanted = {tuple(row[:3]): row for row in expected}
    if (
        len(actual) != len(stored)
        or len(wanted) != len(expected)
        or actual.keys() != wanted.keys()
        or any(canonical(actual[key]) != canonical(value) for key, value in wanted.items())
    ):
        raise ValueError("Stored result rows disagree with the report")


class PostgresClient:
    def __init__(self, dsn):
        self._dsn = dsn

    def results(self, package_sha256, run_id=None):
        """Stream one exact package from a bounded, repeatable, read-only snapshot."""
        if not isinstance(package_sha256, str) or not SHA256.fullmatch(package_sha256):
            raise ValueError("A lowercase SHA-256 package digest is required")
        if run_id is not None and (not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id)):
            raise ValueError("Invalid run identity")
        options = connection_options(self._dsn)
        import psycopg

        try:
            with psycopg.connect(self._dsn, connect_timeout=5, **options) as connection:
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                connection.execute("SET LOCAL statement_timeout = '30s'")
                connection.execute("SET LOCAL lock_timeout = '5s'")
                has_canonical = connection.execute(
                    """SELECT EXISTS (SELECT 1 FROM information_schema.columns
                    WHERE table_schema='xrgpu_benchmarks' AND table_name='benchmark_runs'
                        AND column_name='report_canonical')"""
                ).fetchone()[0]
                payload_column = (
                    "CASE WHEN report_canonical IS NULL THEN NULL "
                    "WHEN octet_length(report_canonical) <= %s THEN report_canonical ELSE '' END"
                    if has_canonical else "NULL::text"
                )
                parameters = [MAX_REPORT_BYTES]
                if has_canonical:
                    parameters.append(MAX_REPORT_BYTES)
                parameters.extend([package_sha256, package_sha256, run_id, run_id])
                with connection.cursor(name="xrgpu_pull") as cursor:
                    cursor.itersize = 1  # Do not buffer a page of potentially large reports.
                    cursor.execute(
                        f"""SELECT run_id,
                        CASE WHEN octet_length(report::text) <= %s THEN report END,
                        report_sha256, received_at, schema_version, suite_revision,
                        runtime_revision, backend, status, device, configuration, mode, preset,
                        started_at, finished_at, {payload_column}
                        FROM xrgpu_benchmarks.benchmark_runs
                        WHERE ((schema_version=3 AND configuration->>'native_package_sha256'=%s)
                            OR (schema_version=2 AND
                                configuration->'kernel_package'->>'content_sha256'=%s))
                            AND (%s::text IS NULL OR run_id=%s)
                        ORDER BY run_id""",
                        parameters,
                    )
                    for row in cursor:
                        envelope = _envelope(row[:-1], package_sha256, row[-1])
                        _verify_results(connection, row[1])
                        yield envelope
        except psycopg.Error:
            # Driver diagnostics can contain credentials, server addresses or SQL
            # data. Do not leak them through the CLI or exception chaining.
            raise DatabaseError(
                "PostgreSQL pull failed. Check connectivity, verified TLS, the xrBench schema, "
                "and SELECT access to benchmark_runs and benchmark_results."
            ) from None
