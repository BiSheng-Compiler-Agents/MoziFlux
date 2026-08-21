"""Tests for kernel-sandbox plugin.

Tests the sandbox hooks, state machine, path protection,
and deliverable verification without needing a running Hermes agent.
"""
import os
import sys
import json
import importlib.util
import hashlib
import sqlite3
from pathlib import Path

import pytest

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
_on_llm_execution = _mod._on_llm_execution
_on_post_tool_call = _mod._on_post_tool_call
_on_post_llm_call = _mod._on_post_llm_call
_on_session_end = _mod._on_session_end
_handle_status = _mod._handle_status


@pytest.fixture(autouse=True)
def _isolate_remote_verify_environment(monkeypatch):
    for name in ("REMOTE_VERIFY_HOST", "REMOTE_VERIFY_USER",
                 "REMOTE_VERIFY_PASS"):
        monkeypatch.delenv(name, raising=False)


def test_successful_remote_verify_clears_stale_validation_errors(
        tmp_path, monkeypatch):
    workspace = tmp_path / "kernel"
    workspace.mkdir()
    _save_state(workspace, {
        "current_stage": "verify",
        "results_txt_errors": ["stale parser failure"],
        "verify_failed": True,
        "verify_error": "old failure",
    })
    monkeypatch.setattr(_mod, "_session_workspace", lambda _session_id: workspace)
    monkeypatch.setattr(_mod, "_save_results_txt", lambda _workspace, _raw: None)
    monkeypatch.setattr(_mod, "_remote_verify_available", lambda: True)
    monkeypatch.setattr(_mod, "_validate_results_txt", lambda _workspace: (True, []))
    monkeypatch.setattr(_mod, "_advance_stage", lambda _state, _workspace: ("record", "ok"))

    _on_post_tool_call(
        tool_name="remote_verify",
        result={"success": True, "test_passed": True, "bench_passed": True},
        session_id="kernelbench-test",
    )

    state = _load_state(workspace)
    assert state["verified"] is True
    assert "results_txt_errors" not in state
    assert "verify_failed" not in state
    assert "verify_error" not in state


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

    def test_detects_agent_readable_numbered_file(self, tmp_workspace):
        baseline = _baseline_name(tmp_workspace)
        assert baseline == "25_Swish.py"

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
        assert _path_is_baseline(str(tmp_workspace / "25_Swish.py"),
                                 tmp_workspace) is True

    def test_non_baseline_allowed(self, tmp_workspace):
        assert _path_is_baseline(str(tmp_workspace / "opt_25_Swish.py"),
                                 tmp_workspace) is False

    def test_outside_workspace_not_baseline(self, tmp_workspace):
        # A file with the baseline name outside the workspace IS still blocked
        # because _path_is_baseline checks the filename, not the directory
        assert _path_is_baseline("/tmp/25_Swish.py", tmp_workspace) is True


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

    def test_guided_search_blocks_advance_while_run_active(
            self, populated_workspace, monkeypatch, tmp_path):
        db_path = tmp_path / "guided.sqlite3"
        monkeypatch.setenv("GUIDED_SEARCH_DB_PATH", str(db_path))
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE search_runs (id INTEGER PRIMARY KEY, status TEXT, "
                "failure_reason TEXT, promotion_domain TEXT, winner_candidate_id INTEGER, "
                "winner_source_hash TEXT)")
            conn.execute("INSERT INTO search_runs VALUES "
                         "(1, 'running', '', 'hardware', NULL, NULL)")
        state = _load_state(populated_workspace)
        state["current_stage"] = "search"
        state["guided_search"] = {"enabled": True, "run_id": 1}
        _save_state(populated_workspace, state)
        new_stage, msg = _advance_stage(state, populated_workspace)
        assert new_stage == "search"
        assert "GUIDED SEARCH" in msg

    def test_guided_search_completed_winner_advances_to_finalize(
            self, populated_workspace, monkeypatch, tmp_path):
        db_path = tmp_path / "guided.sqlite3"
        monkeypatch.setenv("GUIDED_SEARCH_DB_PATH", str(db_path))
        winner = populated_workspace / "opt_25_Swish.py"
        winner_hash = hashlib.sha256(winner.read_bytes()).hexdigest()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE search_runs (id INTEGER PRIMARY KEY, status TEXT, "
                "failure_reason TEXT, promotion_domain TEXT, winner_candidate_id INTEGER, "
                "winner_source_hash TEXT)")
            conn.execute(
                "INSERT INTO search_runs VALUES "
                "(1, 'completed', '', 'hardware', 7, ?)",
                (winner_hash, ),
            )
        state = _load_state(populated_workspace)
        state["current_stage"] = "search"
        state["guided_search"] = {"enabled": True, "run_id": 1}
        _save_state(populated_workspace, state)
        new_stage, msg = _advance_stage(state, populated_workspace)
        assert new_stage == "finalize"
        assert "FINALIZE" in msg

    def test_guided_verify_rejects_changed_winner(self, populated_workspace):
        winner = populated_workspace / "opt_25_Swish.py"
        winner_hash = hashlib.sha256(winner.read_bytes()).hexdigest()
        state = {
            "baseline": "25_Swish.py",
            "current_stage": "verify",
            "verified": True,
            "guided_search": {
                "enabled": True,
                "run_id": 1,
                "winner_source_hash": winner_hash,
            },
        }
        winner.write_text("# replaced after finalization\n", encoding="utf-8")
        new_stage, msg = _advance_stage(state, populated_workspace)
        assert new_stage == "verify"
        assert "winner hash" in msg


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

    def test_allows_v4a_patch_to_opt_file_during_search(self, tmp_workspace):
        original_root = _sandbox_root()
        os.environ["KERNEL_SANDBOX_ROOT"] = str(tmp_workspace.parent)
        try:
            _save_state(tmp_workspace, {
                "baseline": "25_Swish.py",
                "current_stage": "search",
                "guided_search": {"enabled": True, "run_id": 1},
            })
            for target in (
                str(tmp_workspace / "opt_25_Swish.py"),
                os.path.relpath(
                    tmp_workspace / "opt_25_Swish.py", Path.cwd()),
            ):
                result = _on_pre_tool_call(
                    tool_name="patch",
                    args={
                        "mode": "patch",
                        "patch": (
                            "*** Begin Patch\n"
                            f"*** Update File: {target}\n"
                            "@@\n-old\n+new\n"
                            "*** End Patch"
                        ),
                    },
                    session_id=f"kernelbench-{tmp_workspace.name}",
                )
                assert result is None
        finally:
            os.environ["KERNEL_SANDBOX_ROOT"] = str(original_root)

    def test_v4a_patch_checks_every_target(self, tmp_workspace):
        original_root = _sandbox_root()
        os.environ["KERNEL_SANDBOX_ROOT"] = str(tmp_workspace.parent)
        try:
            result = _on_pre_tool_call(
                tool_name="patch",
                args={
                    "mode": "patch",
                    "patch": (
                        "*** Begin Patch\n"
                        f"*** Update File: {tmp_workspace / 'opt_25_Swish.py'}\n"
                        f"*** Update File: {tmp_workspace / '25_Swish.py'}\n"
                        "*** End Patch"
                    ),
                },
                session_id=f"kernelbench-{tmp_workspace.name}",
            )
            assert result is not None
            assert result["action"] == "block"
            assert "input kernel" in result["message"]
        finally:
            os.environ["KERNEL_SANDBOX_ROOT"] = str(original_root)

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

    def test_allows_read_only_terminal_on_guided_files(self, tmp_workspace):
        original_root = _sandbox_root()
        os.environ["KERNEL_SANDBOX_ROOT"] = str(tmp_workspace.parent)
        try:
            for stage, filename in (
                ("search", "profile_kernels.py"),
                ("verify", "opt_25_Swish.py"),
            ):
                _save_state(tmp_workspace, {
                    "baseline": "25_Swish.py",
                    "current_stage": stage,
                    "guided_search": {"enabled": True, "run_id": 1},
                })
                result = _on_pre_tool_call(
                    tool_name="terminal",
                    args={"command": f"sha256sum {tmp_workspace / filename}"},
                    session_id=f"kernelbench-{tmp_workspace.name}",
                )
                assert result is None
        finally:
            os.environ["KERNEL_SANDBOX_ROOT"] = str(original_root)

    def test_blocks_terminal_write_to_hash_locked_winner(self, tmp_workspace):
        original_root = _sandbox_root()
        os.environ["KERNEL_SANDBOX_ROOT"] = str(tmp_workspace.parent)
        try:
            _save_state(tmp_workspace, {
                "baseline": "25_Swish.py",
                "current_stage": "verify",
                "guided_search": {"enabled": True, "run_id": 1},
            })
            result = _on_pre_tool_call(
                tool_name="terminal",
                args={
                    "command": (
                        f"cp /tmp/replacement.py {tmp_workspace / 'opt_25_Swish.py'}")
                },
                session_id=f"kernelbench-{tmp_workspace.name}",
            )
            assert result is not None
            assert result["action"] == "block"
            assert "hash-locked" in result["message"]
        finally:
            os.environ["KERNEL_SANDBOX_ROOT"] = str(original_root)

    def test_non_sandbox_session_allows_anything(self):
        result = _on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/opt/moziflux/.hermes/plugins/foo.py"},
            session_id="random-session",
        )
        assert result is None

    def test_blocks_final_deliverable_during_guided_search(
            self, tmp_workspace):
        original_root = _sandbox_root()
        os.environ["KERNEL_SANDBOX_ROOT"] = str(tmp_workspace.parent)
        try:
            state = {
                "baseline": "base_25_Swish.py",
                "current_stage": "search",
                "guided_search": {
                    "enabled": True,
                    "run_id": 1
                },
            }
            _save_state(tmp_workspace, state)
            result = _on_pre_tool_call(
                tool_name="write_file",
                args={"path": str(tmp_workspace / "review.md")},
                session_id=f"kernelbench-{tmp_workspace.name}",
            )
            assert result is not None
            assert result["action"] == "block"
            assert "FINALIZE" in result["message"]
        finally:
            os.environ["KERNEL_SANDBOX_ROOT"] = str(original_root)

    def test_allows_search_harness_outside_workspace(self, tmp_workspace,
                                                     tmp_path):
        original_root = _sandbox_root()
        os.environ["KERNEL_SANDBOX_ROOT"] = str(tmp_workspace.parent)
        try:
            _save_state(
                tmp_workspace, {
                    "baseline": "25_Swish.py",
                    "current_stage": "search",
                    "guided_search": {
                        "enabled": True,
                        "run_id": 1
                    },
                })
            harness = tmp_path / "guided-artifacts" / "profile_kernels.py"
            result = _on_pre_tool_call(
                tool_name="write_file",
                args={"path": str(harness)},
                session_id=f"kernelbench-{tmp_workspace.name}",
            )
            assert result is None
        finally:
            os.environ["KERNEL_SANDBOX_ROOT"] = str(original_root)

    def test_blocks_winner_write_during_guided_verify(self, tmp_workspace):
        original_root = _sandbox_root()
        os.environ["KERNEL_SANDBOX_ROOT"] = str(tmp_workspace.parent)
        try:
            _save_state(
                tmp_workspace, {
                    "baseline": "25_Swish.py",
                    "current_stage": "verify",
                    "guided_search": {
                        "enabled": True,
                        "run_id": 1,
                        "winner_source_hash": "abc",
                    },
                })
            result = _on_pre_tool_call(
                tool_name="write_file",
                args={"path": str(tmp_workspace / "opt_25_Swish.py")},
                session_id=f"kernelbench-{tmp_workspace.name}",
            )
            assert result is not None
            assert result["action"] == "block"
            assert "hash-locked" in result["message"]
        finally:
            os.environ["KERNEL_SANDBOX_ROOT"] = str(original_root)


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

    def test_request_middleware_refreshes_stage_without_accumulation(
        self, tmp_workspace
    ):
        original_root = _sandbox_root()
        os.environ["KERNEL_SANDBOX_ROOT"] = str(tmp_workspace.parent)
        try:
            _save_state(tmp_workspace, {
                "baseline": "25_Swish.py",
                "current_stage": "optimize",
            })
            request = {
                "messages": [{"role": "user", "content": "optimize"}],
            }
            first = _on_llm_execution(
                request=request,
                next_call=lambda request: request,
                session_id=f"kernelbench-{tmp_workspace.name}",
                api_call_count=1,
            )
            assert first is not None
            assert "STAGE:     optimize" in first["messages"][-1]["content"]

            state = _load_state(tmp_workspace)
            state["current_stage"] = "verify"
            _save_state(tmp_workspace, state)
            second = _on_llm_execution(
                request=request,
                next_call=lambda request: request,
                session_id=f"kernelbench-{tmp_workspace.name}",
                api_call_count=2,
            )
            assert second is not None
            second_content = second["messages"][-1]["content"]
            assert "STAGE:     verify" in second_content
            assert "STAGE:     optimize" not in second_content
            assert request["messages"][0]["content"] == "optimize"

            tool_only = {
                "input": [{
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "tool result",
                }],
            }
            third = _on_llm_execution(
                request=tool_only,
                next_call=lambda request: request,
                session_id=f"kernelbench-{tmp_workspace.name}",
                api_call_count=3,
                api_mode="codex_responses",
            )
            assert third["input"][-1]["role"] == "user"
            assert "KERNEL SANDBOX" in third["input"][-1]["content"][-1]["text"]
        finally:
            os.environ["KERNEL_SANDBOX_ROOT"] = str(original_root)


class TestRegistration:
    def test_uses_llm_request_middleware_not_once_per_turn_hook(self):
        class Context:
            def __init__(self):
                self.hooks = []
                self.middleware = []
                self.tools = []

            def register_hook(self, name, callback):
                self.hooks.append((name, callback))

            def register_middleware(self, name, callback):
                self.middleware.append((name, callback))

            def register_tool(self, **kwargs):
                self.tools.append(kwargs)

        ctx = Context()
        _mod.register(ctx)
        assert "pre_llm_call" not in {name for name, _ in ctx.hooks}
        assert ctx.middleware == [("llm_execution", _on_llm_execution)]


class TestKernelStatusTool:
    """Test kernel_status tool handler."""

    def test_read_status(self, tmp_workspace):
        original_root = _sandbox_root()
        os.environ["KERNEL_SANDBOX_ROOT"] = str(tmp_workspace.parent)
        try:
            result = json.loads(
                _handle_status({
                    "session_id": "kernelbench-test_kernel",
                    "action": "read",
                }))
            assert result["stage"] == "optimize"
            assert result["deliverables_complete"] is False
        finally:
            os.environ["KERNEL_SANDBOX_ROOT"] = str(original_root)

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
