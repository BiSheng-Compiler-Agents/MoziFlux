"""Tests for kernel-sandbox plugin.

Tests the sandbox hooks, state machine, path protection,
and deliverable verification without needing a running Hermes agent.
"""
import os
import sys
import json
import importlib.util
from pathlib import Path

_PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_DIR))

os.environ.setdefault("KERNEL_SANDBOX_ROOT",
                      str(_PROJECT_DIR / "datasets" / "KernelBench_Triton"))
os.environ.setdefault("HERMES_HOME", "/opt/data")

_spec = importlib.util.spec_from_file_location(
    "kernel_sandbox",
    str(_PROJECT_DIR / ".hermes" / "plugins" / "kernel-sandbox" /
        "__init__.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Export all testable functions
_session_workspace = _mod._session_workspace
_sandbox_root = _mod._sandbox_root
_state_file = _mod._state_file
_load_state = _mod._load_state
_save_state = _mod._save_state
_baseline_name = _mod._baseline_name
_project_dir = _mod._project_dir
_path_matches_protected = _mod._path_matches_protected
_path_is_baseline = _mod._path_is_baseline
_check_deliverables = _mod._check_deliverables
_current_stage = _mod._current_stage
_advance_stage = _mod._advance_stage
_on_pre_tool_call = _mod._on_pre_tool_call
_on_pre_llm_call = _mod._on_pre_llm_call
_on_post_llm_call = _mod._on_post_llm_call
_on_session_end = _mod._on_session_end
_handle_status = _mod._handle_status


class TestSessionWorkspace:
    """Test workspace path derivation from session IDs."""

    def test_kernelbench_session(self):
        ws = _session_workspace("kernelbench-l1_25_Swish")
        assert ws is not None
        assert ws.name == "l1_25_Swish"

    def test_followup_session_resolves_to_same_workspace(self):
        ws1 = _session_workspace("kernelbench-l1_25_Swish")
        ws2 = _session_workspace("kernelbench-l1_25_Swish-followup1")
        assert ws1 == ws2

    def test_non_sandbox_session_returns_none(self):
        assert _session_workspace("random-session") is None

    def test_empty_session_returns_none(self):
        assert _session_workspace("") is None


class TestBaselineName:
    """Test baseline file detection."""

    def test_detects_base_prefixed_file(self, tmp_workspace):
        baseline = _baseline_name(tmp_workspace)
        assert baseline == "base_25_Swish.py"

    def test_detects_number_name_file(self, tmp_path):
        ws = tmp_path / "test"
        ws.mkdir()
        (ws / "99_MyKernel.py").write_text("# kernel")
        baseline = _baseline_name(ws)
        assert baseline == "99_MyKernel.py"

    def test_returns_none_when_no_files(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        assert _baseline_name(ws) is None

    def test_state_file_takes_precedence(self, tmp_workspace):
        state = {"baseline": "25_Swish.py"}
        sf = _state_file(tmp_workspace)
        with open(sf, "w") as f:
            json.dump(state, f)
        baseline = _baseline_name(tmp_workspace)
        assert baseline == "25_Swish.py"


class TestPathProtection:
    """Test project file protection logic."""

    def test_blocks_project_plugins_dir(self):
        assert _path_matches_protected(
            "/opt/moziflux/.hermes/plugins/foo.py") is True

    def test_blocks_project_skills_dir(self):
        assert _path_matches_protected(
            "/opt/moziflux/.hermes/skills/foo.md") is True

    def test_blocks_optimize_kernels_py(self):
        assert _path_matches_protected(
            "/opt/moziflux/optimize_kernels.py") is True

    def test_blocks_agents_md(self):
        assert _path_matches_protected("/opt/moziflux/AGENTS.md") is True

    def test_allows_outside_project(self):
        assert _path_matches_protected(
            "/opt/data/.hermes/plugins/foo.py") is False

    def test_allows_tmp_files(self):
        assert _path_matches_protected("/tmp/test.py") is False

    def test_allows_workspace_files(self, tmp_workspace):
        assert _path_matches_protected(str(tmp_workspace /
                                           "opt_25_Swish.py")) is False


class TestPathIsBaseline:
    """Test baseline file detection for write blocking."""

    def test_identifies_baseline(self, tmp_workspace):
        assert _path_is_baseline(str(tmp_workspace / "base_25_Swish.py"),
                                 tmp_workspace) is True

    def test_non_baseline_allowed(self, tmp_workspace):
        assert _path_is_baseline(str(tmp_workspace / "opt_25_Swish.py"),
                                 tmp_workspace) is False

    def test_outside_workspace_not_baseline(self, tmp_workspace):
        # A file with the baseline name outside the workspace IS still blocked
        # because _path_is_baseline checks the filename, not the directory
        assert _path_is_baseline("/tmp/base_25_Swish.py",
                                 tmp_workspace) is True


class TestDeliverablesCheck:
    """Test deliverable verification."""

    def test_all_present(self, populated_workspace):
        complete, missing = _check_deliverables(populated_workspace)
        assert complete is True
        assert missing == []

    def test_missing_opt(self, tmp_workspace):
        # Only create 4 of 5 deliverables
        (tmp_workspace / "profile_kernels.py").write_text("")
        (tmp_workspace / "Optimizations.md").write_text("")
        (tmp_workspace / "performance_report.md").write_text("")
        (tmp_workspace / "review.md").write_text("")
        complete, missing = _check_deliverables(tmp_workspace)
        assert complete is False
        assert any("opt_" in m for m in missing)

    def test_missing_all(self, tmp_workspace):
        complete, missing = _check_deliverables(tmp_workspace)
        assert complete is False
        assert len(missing) == 5


class TestStateMachine:
    """Test pipeline stage transitions."""

    def test_initial_stage_is_optimize(self, tmp_workspace):
        state = _load_state(tmp_workspace)
        assert _current_stage(state) == "optimize"

    def test_advance_to_verify_when_complete(self, populated_workspace):
        state = _load_state(populated_workspace)
        new_stage, msg = _advance_stage(state, populated_workspace)
        assert new_stage == "verify"
        assert "VERIFY" in msg

    def test_stays_optimize_when_incomplete(self, tmp_workspace):
        state = _load_state(tmp_workspace)
        new_stage, msg = _advance_stage(state, tmp_workspace)
        assert new_stage == "optimize"
        assert "INCOMPLETE" in msg

    def test_advance_to_record_when_verified(self, populated_workspace):
        state = _load_state(populated_workspace)
        _advance_stage(state, populated_workspace)  # -> verify
        state["verified"] = True
        _save_state(populated_workspace, state)
        state = _load_state(populated_workspace)
        new_stage, msg = _advance_stage(state, populated_workspace)
        assert new_stage == "record"

    def test_revert_to_optimize_when_verify_failed(self, populated_workspace):
        state = _load_state(populated_workspace)
        _advance_stage(state, populated_workspace)  # -> verify
        state["verify_failed"] = True
        state["verify_error"] = "cannsim timeout"
        _save_state(populated_workspace, state)
        state = _load_state(populated_workspace)
        new_stage, msg = _advance_stage(state, populated_workspace)
        assert new_stage == "optimize"
        assert "FAILED" in msg

    def test_advance_to_done_when_recorded(self, populated_workspace):
        state = _load_state(populated_workspace)
        _advance_stage(state, populated_workspace)  # -> verify
        state["verified"] = True
        _save_state(populated_workspace, state)
        state = _load_state(populated_workspace)
        _advance_stage(state, populated_workspace)  # -> record
        state["recorded"] = True
        _save_state(populated_workspace, state)
        state = _load_state(populated_workspace)
        new_stage, msg = _advance_stage(state, populated_workspace)
        assert new_stage == "done"


class TestPreToolCallHook:
    """Test pre_tool_call hook blocks dangerous operations."""

    def test_blocks_write_to_plugins(self):
        result = _on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/opt/moziflux/.hermes/plugins/foo.py"},
            session_id="kernelbench-l1_25_Swish",
        )
        assert result is not None
        assert result["action"] == "block"

    def test_blocks_write_to_baseline(self, tmp_workspace):
        # Use a session_id that resolves to our tmp_workspace
        # _session_workspace expects "kernelbench-<kernel_name>" under KERNEL_SANDBOX_ROOT
        # We need to monkeypatch the sandbox root to use our tmp_workspace's parent
        original_root = _sandbox_root()

        # Create a fake session_id based on tmp_workspace name
        session_id = f"kernelbench-{tmp_workspace.name}"

        # Temporarily override the sandbox root
        tmp_parent = tmp_workspace.parent
        os.environ["KERNEL_SANDBOX_ROOT"] = str(tmp_parent)

        try:
            result = _on_pre_tool_call(
                tool_name="write_file",
                args={"path": str(tmp_workspace / "base_25_Swish.py")},
                session_id=session_id,
            )
            assert result is not None
            assert result["action"] == "block"
        finally:
            # Restore
            os.environ["KERNEL_SANDBOX_ROOT"] = str(original_root)

    def test_allows_write_to_workspace(self, tmp_workspace):
        result = _on_pre_tool_call(
            tool_name="write_file",
            args={"path": str(tmp_workspace / "opt_test.py")},
            session_id="kernelbench-test_kernel",
        )
        assert result is None

    def test_blocks_dangerous_terminal(self):
        result = _on_pre_tool_call(
            tool_name="terminal",
            args={"command": "rm -rf /opt/moziflux/.hermes/plugins/"},
            session_id="kernelbench-l1_25_Swish",
        )
        assert result is not None
        assert result["action"] == "block"

    def test_allows_safe_terminal(self):
        result = _on_pre_tool_call(
            tool_name="terminal",
            args={"command": "ls -la"},
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


class TestPreLlmCallHook:
    """Test pre_llm_call hook context injection."""

    def test_injects_context_for_sandbox_session(self):
        result = _on_pre_llm_call(
            session_id="kernelbench-l1_25_Swish",
            user_message="test",
            is_first_turn=True,
            model="test-model",
            platform="python",
        )
        assert result is not None
        assert "context" in result
        assert "KERNEL SANDBOX" in result["context"]

    def test_returns_none_for_non_sandbox_session(self):
        result = _on_pre_llm_call(
            session_id="random-session",
            user_message="test",
            is_first_turn=True,
            model="test",
            platform="python",
        )
        assert result is None


class TestKernelStatusTool:
    """Test kernel_status tool handler."""

    def test_read_status(self, tmp_workspace):
        result = json.loads(
            _handle_status({
                "session_id": "kernelbench-test_kernel",
                "action": "read",
            }))
        assert result["stage"] == "optimize"
        assert result["deliverables_complete"] is False

    def test_set_verified_flag(self, tmp_workspace):
        # Create the workspace under the real sandbox root
        session_id = f"kernelbench-{tmp_workspace.name}"
        tmp_parent = tmp_workspace.parent
        os.environ["KERNEL_SANDBOX_ROOT"] = str(tmp_parent)
        try:
            result = json.loads(
                _handle_status({
                    "session_id": session_id,
                    "action": "set",
                    "key": "verified",
                    "value": True,
                }))
            assert result["ok"] is True
        finally:
            os.environ["KERNEL_SANDBOX_ROOT"] = str(_PROJECT_DIR / "datasets" /
                                                    "KernelBench_Triton")

    def test_set_invalid_key(self, tmp_workspace):
        result = json.loads(
            _handle_status({
                "session_id": "kernelbench-test_kernel",
                "action": "set",
                "key": "invalid_key",
                "value": True,
            }))
        assert "error" in result

    def test_unknown_session(self):
        result = json.loads(
            _handle_status({
                "session_id": "unknown-session",
                "action": "read",
            }))
        assert "error" in result
