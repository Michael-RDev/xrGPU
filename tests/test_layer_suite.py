import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["benchmarks", "runtime.layer_benchmark"])
def test_old_benchmark_commands_explain_relocation(module):
    result = subprocess.run([sys.executable, "-m", module], capture_output=True, text=True)
    assert result.returncode == 1
    assert "xrBench" in result.stderr
    assert "runtime.benchmark_results import --all" in result.stderr
