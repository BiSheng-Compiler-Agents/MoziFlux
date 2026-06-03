"""Shared test fixtures and configuration."""
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
_PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_DIR))

# Ensure Hermes env is set for plugin imports
os.environ.setdefault("HERMES_HOME", "/opt/data")
os.environ.setdefault("KERNEL_SANDBOX_ROOT",
                      str(_PROJECT_DIR / "datasets" / "KernelBench_Triton"))


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace directory with a baseline kernel file."""
    ws = tmp_path / "test_kernel"
    ws.mkdir()
    # Create a baseline file (Number_name.py pattern)
    (ws / "25_Swish.py").write_text("# baseline kernel\nimport triton\n")
    # Create a reference file (base_Number_name.py pattern)
    (ws / "base_25_Swish.py").write_text("# reference kernel\nimport triton\n")
    return ws


@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a path for a temporary episodes database."""
    return str(tmp_path / "test_episodes.db")


@pytest.fixture
def populated_workspace(tmp_workspace):
    """Workspace with deliverables present."""
    ws = tmp_workspace
    (ws / "opt_25_Swish.py").write_text("# optimized kernel\n")
    (ws / "profile_kernels.py").write_text("# profile\n")
    (ws / "Optimizations.md").write_text("# Optimizations\n")
    (ws / "performance_report.md").write_text("# Performance Report\n")
    (ws / "review.md").write_text("# Review\n")
    return ws
