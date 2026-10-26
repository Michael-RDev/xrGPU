"""Compatibility namespace; native implementations live in kernels._shared."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[1] / "kernels/_shared")]
