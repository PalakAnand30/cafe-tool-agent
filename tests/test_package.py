"""Smoke-check the installed distribution without a source-path override."""

import subprocess
import sys
from pathlib import Path


def test_installed_package_imports_without_gpu_dependencies(tmp_path: Path) -> None:
    """A clean interpreter outside the checkout can import every package module."""
    script = """
import importlib
import pkgutil
import sys
from importlib.metadata import version

import cafe_agent

assert version("cafe-tool-agent")
for module in pkgutil.walk_packages(cafe_agent.__path__, cafe_agent.__name__ + "."):
    importlib.import_module(module.name)

forbidden = {"torch", "bitsandbytes", "vllm", "unsloth", "mlx", "mlx_lm"}
assert not forbidden.intersection(name.split(".")[0] for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
