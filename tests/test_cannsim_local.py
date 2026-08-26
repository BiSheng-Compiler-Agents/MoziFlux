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
_find_instr_bin = _mod._find_instr_bin
_wait_for_instr_bin = _mod._wait_for_instr_bin
_cannsim_local_run = _mod._cannsim_local_run
_apply_local_patches = _mod._apply_local_patches


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

    def test_patches_requested_conda_environment_before_host_triton(
            self, tmp_path, monkeypatch):
        conda_root = tmp_path / "conda"
        conda_bin = conda_root / "bin" / "conda"
        conda_bin.parent.mkdir(parents=True)
        conda_bin.write_text("", encoding="utf-8")
        triton_dir = (conda_root / "envs" / "target" / "lib" / "python3.11" /
                      "site-packages" / "triton")
        tools = triton_dir / "tools"
        tools.mkdir(parents=True)
        devices = tools / "get_ascend_devices.py"
        devices.write_text(
            "import os\n"
            "pci_condition = False\n"
            "npu_smi_condition = False\n"
            "is_compile_on_910_95 = pci_condition or npu_smi_condition\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("CONDA_BIN", str(conda_bin))
        ok, log = _apply_local_patches("target")
        assert ok is True
        assert "get_ascend_devices.py: applied OK" in log
        assert "env_condition" in devices.read_text(encoding="utf-8")

    def test_patch_application_is_idempotent(self, tmp_path, monkeypatch):
        conda_root = tmp_path / "conda"
        conda_bin = conda_root / "bin" / "conda"
        conda_bin.parent.mkdir(parents=True)
        conda_bin.write_text("", encoding="utf-8")
        triton_dir = (conda_root / "envs" / "target" / "lib" / "python3.11" /
                      "site-packages" / "triton")
        tools = triton_dir / "tools"
        runtime = triton_dir / "backends" / "ascend" / "runtime"
        tools.mkdir(parents=True)
        runtime.mkdir(parents=True)
        devices = tools / "get_ascend_devices.py"
        devices.write_text(
            "import os\n"
            "pci_condition = False\n"
            "npu_smi_condition = False\n"
            "is_compile_on_910_95 = pci_condition or npu_smi_condition\n",
            encoding="utf-8",
        )
        runtime_utils = runtime / "utils.py"
        runtime_utils.write_text(
            "import torch\n\n"
            "def _init_npu_params():\n"
            "    from triton.runtime.driver import driver\n\n"
            "    target = driver.active.get_current_target()\n"
            "    return target\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("CONDA_BIN", str(conda_bin))

        first_ok, first_log = _apply_local_patches("target")
        assert first_ok is True
        assert "get_ascend_devices.py: applied OK" in first_log
        assert "runtime/utils.py: applied OK" in first_log
        first_devices = devices.read_text(encoding="utf-8")
        first_runtime = runtime_utils.read_text(encoding="utf-8")

        second_ok, second_log = _apply_local_patches("target")
        assert second_ok is True
        assert "get_ascend_devices.py: already applied" in second_log
        assert "runtime/utils.py: already applied" in second_log
        assert devices.read_text(encoding="utf-8") == first_devices
        assert runtime_utils.read_text(encoding="utf-8") == first_runtime


class TestInstrBinSafety:

    def test_prefers_non_empty_instr_bin_over_newer_empty_file(self, tmp_path):
        experiment = tmp_path / "cannsim_1_test_kernel"
        experiment.mkdir()
        valid = experiment / "instr.bin"
        valid.write_bytes(b"x" * _mod.INSTR_LOG_RECORD_SIZE)
        empty = tmp_path / "instr.bin"
        empty.write_bytes(b"")
        os.utime(empty, (valid.stat().st_mtime + 10, ) * 2)
        assert _find_instr_bin(str(tmp_path)) == str(valid)

    def test_accepts_only_aligned_stable_instr_bin(self, tmp_path):
        instr = tmp_path / "instr.bin"
        instr.write_bytes(b"x" * (_mod.INSTR_LOG_RECORD_SIZE * 2))
        safe, reason = _wait_for_instr_bin(str(tmp_path),
                                           poll=0,
                                           max_wait=0.1,
                                           quiet_window=0)
        assert safe is True
        assert "stable" in reason

        instr.write_bytes(b"x" * (_mod.INSTR_LOG_RECORD_SIZE + 1))
        safe, reason = _wait_for_instr_bin(str(tmp_path),
                                           poll=0.001,
                                           max_wait=0.01,
                                           quiet_window=0)
        assert safe is False
        assert "not aligned" in reason


class TestCannsimLocalRun:

    def _run_with_report(self, tmp_path, monkeypatch, *, create_trace: bool):
        source = tmp_path / "source"
        source.mkdir()
        (source / "run.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        monkeypatch.setattr(_mod.tempfile, "gettempdir", lambda: str(tmp_path))
        monkeypatch.setattr(_mod, "_build_env_with_cann", lambda: {"PATH": ""})
        monkeypatch.setattr(_mod, "_find_cannsim_bin", lambda _env: "cannsim")
        monkeypatch.setattr(_mod, "_apply_local_patches", lambda _env:
                            (True, "patched"))
        monkeypatch.setattr(_mod, "_run_cannsim_record",
                            lambda *_args, **_kwargs: (0, "recorded", ""))

        def fake_run(command, cwd, env, timeout):
            if " build'" in command:
                (Path(cwd) / "test_kernel").write_text("binary",
                                                       encoding="utf-8")
                return 0, "built", ""
            if " report " in command:
                if create_trace:
                    report = Path(cwd) / "report"
                    report.mkdir(parents=True, exist_ok=True)
                    (report / "trace_core0.json").write_text(
                        '[{"ph":"X","ts":0,"dur":1}]', encoding="utf-8")
                return 0, "reported", ""
            raise AssertionError(f"unexpected command: {command}")

        monkeypatch.setattr(_mod, "_run", fake_run)
        return _cannsim_local_run(str(source),
                                  "run.sh",
                                  job_name="test-job",
                                  gen_report=True)

    def test_requested_report_without_trace_is_failure(self, tmp_path,
                                                       monkeypatch):
        result = self._run_with_report(tmp_path,
                                       monkeypatch,
                                       create_trace=False)
        assert result["success"] is False
        assert "trace_core0.json" in result["error"]

    def test_requested_report_with_trace_is_success(self, tmp_path,
                                                    monkeypatch):
        result = self._run_with_report(tmp_path,
                                       monkeypatch,
                                       create_trace=True)
        assert result["success"] is True
        assert result["trace_json_size_bytes"] > 0
        assert result["trace_local_path"].endswith("trace_core0.json")
