# Pull versioned xrBench results

xrBench owns PostgreSQL ingestion. xrGPU can pull committed reports directly,
without starting xrBench's local HTTP server or running a benchmark. Select an
exact immutable kernel package digest; the current checkout and a mutable
"latest" version are never used to identify measurements.

```sh
uv sync --locked --extra database --inexact
# Supply XRGPU_DATABASE_URL through your private environment or secret manager.
uv run --locked --extra database --inexact python -m runtime.benchmark_results pull \
  --package-sha256 PACKAGE_SHA256
```

For native operator reports (schema 3), use `package_sha256` from the measured
`xrgpu.native-operators.v1` package manifest. xrBench records it as
`native_package_sha256`. For older layer/resident reports (schema 2), use the
benchmark source package's `kernel_package.content_sha256`, the digest of its
manifest. These identities refer to different package types and are not
interchangeable with a compiled model package or a qualification record.

Add `--run-id run-<32 lowercase hex digits>` to pull one run while still checking
its package version. `--database-env PRIVATE_DB_VARIABLE` selects a different
environment variable. Credentials are never accepted as CLI arguments or saved
with the evidence. `--output /absolute/cache/path` changes the cache location.
No matches for a package produces a message and exits successfully; a requested
run that does not match the package exits with an error.

The connection needs SELECT access to
`xrgpu_benchmarks.benchmark_runs` and `xrgpu_benchmarks.benchmark_results`, plus
schema usage. Use separately provisioned reader credentials that can see the
intended rows under the deployment's RLS policies. The hosted Supabase ingestion
service role has RPC access only; its API key does not grant this SQL reader
access. This command does not create roles, change grants/RLS, apply migrations,
or use administrative migration secrets automatically. Remote and ambiguous
hosts require `sslmode=verify-full` and a trusted CA; explicit numeric loopback
addresses or Unix sockets support local development.

The pull uses one REPEATABLE READ, READ ONLY transaction with query/lock timeouts
and streams one report at a time. It verifies the database's existing SHA-256
using either Python's local encoding or the hosted receiver's ECMAScript/JCS
encoding, the run/package identity, indexed run fields and timestamps, and each
original result row and its identity/status/correctness/sample columns. A
corrupt or oversized report fails before that run is cached. No new checksum is
substituted for a mismatched stored checksum.

xrBench migration 004 adds `benchmark_runs.report_canonical`, containing the
exact JSON representation used for Python report hashing. Apply that migration
through xrBench's administrative setup before using its updated Python ingest
service. The pull verifies this payload against the existing checksum and the
JSONB copy, then caches the original numeric representations, including exponent
notation and signed zero. Hosted JCS reports and verifiable older schemas remain
readable without this column.

JSONB cannot recover every legacy Python report's original numeric representation.
If such a row fails verification, its database owner must restore the canonical
payload from the original report, with a matching existing checksum. The pull
does not guess the lost formatting or replace the database checksum.

Each verified run is atomically saved to:

```text
artifacts/benchmarks/run-<id>/
  report.json
  receipt.json
  source.json
```

`report.json` preserves the database's full JSON content, including nested
`native_report` originals when present. When available, the saved canonical payload
preserves numeric representations that JSONB discards. `receipt.json` records the stored hash,
hash encoding, committed receipt time and database run identity. `source.json`
records import time, package digest, backend and execution location without the
DSN. Identical retries across HTTP imports and database pulls leave the original
cached report, receipt, and source files intact. Conflicting or incomplete
cached evidence is never overwritten. If a later run fails, earlier verified
runs remain cached and can be reused on retry.

Complete, failed and canceled native runs are retained with their existing
statuses, including CPU and resident backend evidence. Importing proves stored
content integrity, not hardware attestation, correctness, a performance gain or
kernel eligibility. Qualification remains an explicit, separate release input.
The existing `list`/`import --url` commands and archive imports remain available.

## Verification

Run the protocol/cache suites with the database extra installed:

```sh
.venv/bin/python -m pytest tests/test_benchmark_results.py tests/test_benchmark_database.py
```

Database integration cases are opt-in. Set `XRGPU_TEST_POSTGRES=1` and
`XRGPU_TEST_DATABASE_URL` to an explicitly local administrative test database.
Tests create and remove only UUID-named temporary databases; they do not use
`XRGPU_DATABASE_URL`, the hosted database, or xrBench environment files. They
exercise both hash formats, package/run selection, retries, corruption, read-only
enforcement, concurrent inserts and the CLI. Node, when available, independently
checks ECMAScript hashing against the JavaScript canonicalizer.

Verified on 2026-10-27: 89 pull/cache/HTTP/qualification tests passed against an
isolated PostgreSQL 14 cluster. A separate compatibility check applied xrBench's
actual engine and hosted migrations locally, ingested both report formats, and
pulled the hosted 221-operator / 22,100-sample report through a SELECT-only role
with explicit test RLS policies. No production database was used for these checks.
