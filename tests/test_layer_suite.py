"""The retained compatibility command redirects users; removed stubs stay removed."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_remaining_benchmark_command_explains_relocation():
    result = subprocess.run(
        [sys.executable, "-m", "runtime.layer_benchmark"], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "xrBench" in result.stderr
    assert "runtime.benchmark_results import --all" in result.stderr


def test_removed_benchmark_package_has_no_legacy_command():
    # The user deliberately deleted this stub; native catalog migration must not
    # restore it or continue promising its old relocation-message contract.
    assert not (ROOT / "benchmarks/__main__.py").exists()
    result = subprocess.run([sys.executable, "-m", "benchmarks"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "No module named benchmarks" in result.stderr
