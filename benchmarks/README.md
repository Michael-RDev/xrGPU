# Benchmarks moved

The benchmark engine and both websites now live in [xrBench](../../xrBench/README.md).
Run `uv run python main.py` from that repository to start the developer service
on `http://localhost:8766`.

xrGPU imports committed reports with `uv run python -m runtime.benchmark_results import --all`.
