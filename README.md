# xrGPU

Pull committed benchmark results for an exact kernel package version:

```sh
uv run --locked --extra database --inexact python -m runtime.benchmark_results pull \
  --package-sha256 PACKAGE_SHA256
```

Supply `XRGPU_DATABASE_URL` privately. See [database pull setup and verification](docs/benchmark-results.md).
