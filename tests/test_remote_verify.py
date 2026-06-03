"""Tests for remote-verify plugin.

Tests the remote verification logic without requiring actual SSH connections.
Uses mocking for paramiko SSH/SFTP operations.
"""
import os
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

_PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_DIR))

os.environ.setdefault("HERMES_HOME", "/opt/data")
os.environ.setdefault("REMOTE_VERIFY_HOST", "192.168.1.10")
os.environ.setdefault("REMOTE_VERIFY_USER", "testuser")
os.environ.setdefault("REMOTE_VERIFY_PASS", "testpass")
os.environ.setdefault("REMOTE_VERIFY_PORT", "22")
os.environ.setdefault("REMOTE_VERIFY_BASE_DIR", "~/kernel_verify")
os.environ.setdefault("REMOTE_VERIFY_CONDA_ENV", "compilerclaw")

_spec = importlib.util.spec_from_file_location(
    "remote_verify",
    str(_PROJECT_DIR / ".hermes" / "plugins" / "remote-verify" /
        "__init__.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_remote_verify = _mod._remote_verify
_on_pre_tool_call = _mod._on_pre_tool_call
_remote_host = _mod._remote_host
_remote_user = _mod._remote_user
_remote_pass = _mod._remote_pass


class TestEnvHelpers:
    """Test environment variable resolution."""

    def test_remote_host(self):
        assert _remote_host() == "192.168.1.10"

    def test_remote_user(self):
        assert _remote_user() == "testuser"

    def test_remote_pass(self):
        assert _remote_pass() == "testpass"


class TestRemoteVerifyInputValidation:
    """Test input validation for remote_verify tool."""

    def test_missing_local_dir(self):
        result = _remote_verify("/nonexistent/dir")
        assert result["success"] is False
        assert "does not exist" in result["error"]

    def test_missing_profile_kernels(self, tmp_path):
        result = _remote_verify(str(tmp_path))
        assert result["success"] is False
        assert "profile_kernels.py not found" in result["error"]


class TestRemoteVerifyMockedSSH:
    """Test remote_verify with mocked SSH/SFTP."""

    def _make_mock_ssh(self, test_pass=True, bench_pass=True):
        """Create a mock SSH client."""
        mock_ssh = MagicMock()
        mock_sftp = MagicMock()

        # Mock SFTP open
        mock_ssh.open_sftp.return_value = mock_sftp

        # Mock exec_command responses
        call_count = {"n": 0}

        def mock_exec(cmd, timeout=300):
            call_count["n"] += 1
            stdin = MagicMock()
            stdout = MagicMock()
            stderr = MagicMock()

            if "echo $HOME" in cmd:
                stdout.channel.recv_exit_status.return_value = 0
                stdout.read.return_value = b"/home/testuser\n"
                stderr.read.return_value = b""
            elif "rm -rf" in cmd:
                stdout.channel.recv_exit_status.return_value = 0
                stdout.read.return_value = b""
                stderr.read.return_value = b""
            elif "mkdir -p" in cmd:
                stdout.channel.recv_exit_status.return_value = 0
                stdout.read.return_value = b""
                stderr.read.return_value = b""
            elif "ls -la" in cmd:
                stdout.channel.recv_exit_status.return_value = 0
                stdout.read.return_value = b"profile_kernels.py\nopt_kernel.py\n"
                stderr.read.return_value = b""
            elif "command -v conda" in cmd:
                stdout.channel.recv_exit_status.return_value = 0
                stdout.read.return_value = b"/opt/miniconda3/bin/conda\n"
                stderr.read.return_value = b""
            elif "profile_kernels.py --test" in cmd:
                stdout.channel.recv_exit_status.return_value = 0 if test_pass else 1
                if test_pass:
                    stdout.read.return_value = b"UNIT TEST\nPASS\nall shapes correct\n"
                else:
                    stdout.read.return_value = b"UNIT TEST\nFAIL\nshape B=64 mismatch\n"
                stderr.read.return_value = b""
            elif "profile_kernels.py --bench" in cmd:
                stdout.channel.recv_exit_status.return_value = 0 if bench_pass else 1
                stdout.read.return_value = b"Benchmark results\nPyTorch/ACL: 1.2ms\nOptimized: 0.8ms\n"
                stderr.read.return_value = b""
            elif "find" in cmd and ("*.txt" in cmd or "*.csv" in cmd):
                stdout.channel.recv_exit_status.return_value = 0
                stdout.read.return_value = b""
                stderr.read.return_value = b""
            else:
                stdout.channel.recv_exit_status.return_value = 0
                stdout.read.return_value = b""
                stderr.read.return_value = b""

            return stdin, stdout, stderr

        mock_ssh.exec_command.side_effect = mock_exec
        return mock_ssh, mock_sftp

    def test_successful_test_and_bench(self, tmp_path):
        (tmp_path / "profile_kernels.py").write_text("# test profile\n")
        (tmp_path / "opt_kernel.py").write_text("# optimized\n")

        mock_ssh, mock_sftp = self._make_mock_ssh(test_pass=True,
                                                  bench_pass=True)

        with patch.object(_mod, "_ssh_connect", return_value=mock_ssh):
            result = _remote_verify(str(tmp_path),
                                    run_test=True,
                                    run_bench=True)

        assert result["success"] is True
        assert result["test_passed"] is True
        assert "bench_output" in result
        assert result["remote_dir"] is not None

    def test_test_fails_no_bench(self, tmp_path):
        (tmp_path / "profile_kernels.py").write_text("# test profile\n")

        mock_ssh, mock_sftp = self._make_mock_ssh(test_pass=False,
                                                  bench_pass=False)

        with patch.object(_mod, "_ssh_connect", return_value=mock_ssh):
            result = _remote_verify(str(tmp_path),
                                    run_test=True,
                                    run_bench=True)

        assert result["success"] is True
        assert result["test_passed"] is False
        assert "bench_output" not in result  # Bench should be skipped

    def test_test_only(self, tmp_path):
        (tmp_path / "profile_kernels.py").write_text("# test\n")

        mock_ssh, mock_sftp = self._make_mock_ssh(test_pass=True)

        with patch.object(_mod, "_ssh_connect", return_value=mock_ssh):
            result = _remote_verify(str(tmp_path),
                                    run_test=True,
                                    run_bench=False)

        assert result["success"] is True
        assert result["test_passed"] is True
        assert "bench_output" not in result

    def test_paramiko_not_installed(self, tmp_path):
        (tmp_path / "profile_kernels.py").write_text("# test\n")

        with patch.dict(sys.modules, {"paramiko": None}):
            result = _remote_verify(str(tmp_path))
            assert result["success"] is False
            assert "paramiko not installed" in result["error"]


class TestPreToolCallHook:
    """Test pre_tool_call hook for sandbox enforcement."""

    def test_blocks_write_to_plugins(self):
        result = _on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/opt/moziflux/.hermes/plugins/foo.py"},
            session_id="kernelbench-l1_25_Swish",
        )
        assert result is not None
        assert result["action"] == "block"

    def test_blocks_dangerous_terminal(self):
        result = _on_pre_tool_call(
            tool_name="terminal",
            args={"command": "rm -rf /opt/moziflux/.hermes/plugins/"},
            session_id="kernelbench-l1_25_Swish",
        )
        assert result is not None

    def test_allows_safe_operations(self):
        result = _on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/tmp/test.py"},
            session_id="kernelbench-l1_25_Swish",
        )
        assert result is None

    def test_non_sandbox_session_allows_anything(self):
        result = _on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/opt/moziflux/.hermes/plugins/foo.py"},
            session_id="random-session",
        )
        assert result is None
