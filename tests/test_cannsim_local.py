"""Tests for cannsim-local plugin.

Tests helper functions that don't require CANN/cannsim to be installed.
"""
import os
import sys
import importlib.util
from pathlib import Path

_PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_DIR))

os.environ.setdefault("HERMES_HOME", "/opt/data")

_spec = importlib.util.spec_from_file_location(
    "cannsim_local",
    str(_PROJECT_DIR / ".hermes" / "plugins" / "cannsim-local" /
        "__init__.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_cannsim_record_completed = _mod._cannsim_record_completed
_build_env_with_cann = _mod._build_env_with_cann


class TestCannsimRecordCompleted:
    """Test CANN 9.0.0 cleanup bug detection."""

    def test_detects_cleanup_bug(self):
        stdout = "all tasks are finished!\ncurrent_dir = os.getcwd()\n"
        stderr = "FileNotFoundError: [Errno 2] No such directory\n_cleanup_user_env\n"
        assert _cannsim_record_completed(stdout, stderr) is True

    def test_normal_completion_not_detected(self):
        stdout = "all tasks are finished!\n"
        stderr = ""
        assert _cannsim_record_completed(stdout, stderr) is False

    def test_failure_not_detected(self):
        stdout = "Simulation failed\n"
        stderr = "Error: kernel crash\n"
        assert _cannsim_record_completed(stdout, stderr) is False


class TestBuildEnvWithCann:
    """Test CANN environment building."""

    def test_returns_env_dict(self):
        env = _build_env_with_cann()
        assert isinstance(env, dict)
        assert "PATH" in env

    def test_handles_missing_setenv_path(self, monkeypatch):
        monkeypatch.delenv("CANNSIM_SETENV_PATH", raising=False)
        env = _build_env_with_cann()
        assert isinstance(env, dict)

    def test_handles_invalid_setenv_path(self, monkeypatch):
        monkeypatch.setenv("CANNSIM_SETENV_PATH", "/nonexistent/path.sh")
        env = _build_env_with_cann()
        assert isinstance(env, dict)
