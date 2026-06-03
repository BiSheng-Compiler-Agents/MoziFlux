"""Tests for cannsim-remote plugin.

Tests helper functions that don't require SSH access.
"""
import os
import sys
import importlib.util
from pathlib import Path

_PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_DIR))

os.environ.setdefault("HERMES_HOME", "/opt/data")

_spec = importlib.util.spec_from_file_location(
    "cannsim_remote",
    str(_PROJECT_DIR / ".hermes" / "plugins" / "cannsim-remote" /
        "__init__.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_cannsim_record_completed = _mod._cannsim_record_completed


class TestCannsimRecordCompleted:
    """Test CANN 9.0.0 cleanup bug detection (remote variant)."""

    def test_detects_cleanup_bug(self):
        log = "all tasks are finished!\ncurrent_dir = os.getcwd()\nFileNotFoundError\n_cleanup_user_env\n"
        assert _cannsim_record_completed(log) is True

    def test_normal_completion_not_detected(self):
        log = "all tasks are finished!\n"
        assert _cannsim_record_completed(log) is False

    def test_failure_not_detected(self):
        log = "Simulation failed\nError: kernel crash\n"
        assert _cannsim_record_completed(log) is False
