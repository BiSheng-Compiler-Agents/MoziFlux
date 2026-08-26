"""Tests for optimize_kernels.py orchestrator logic.

Tests state management, kernel discovery, baseline detection,
and workspace setup without running the full pipeline.
"""
import json
import os
from pathlib import Path

import sys
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from optimize_kernels import (
    mark_kernel,
    get_baseline_file,
    kernel_is_complete,
    setup_workspace,
    disable_batch_agent_background_reviews,
    KERNEL_AGENT_TOOLSETS,
    MODEL,
    main,
)


@pytest.fixture(autouse=True)
def _isolate_remote_verify_environment(monkeypatch):
    monkeypatch.delenv("REMOTE_VERIFY_HOST", raising=False)


class TestStateManagement:
    """Test optimize_state.json helpers."""

    def test_mark_kernel_records_model_name(self, tmp_path, monkeypatch):
        import optimize_kernels

        state_file = tmp_path / "optimize_state.json"
        monkeypatch.setattr(optimize_kernels, "STATE_FILE", state_file)

        state = {"kernels": {}}
        mark_kernel(state, "l1_25_Swish", "running", "started")

        assert state["kernels"]["l1_25_Swish"]["model"] == MODEL
        with open(state_file) as f:
            saved = json.load(f)
        assert saved["kernels"]["l1_25_Swish"]["model"] == MODEL

    def test_batch_agent_disables_background_memory_and_skill_reviews(self):

        class Agent:
            _memory_nudge_interval = 5
            _skill_nudge_interval = 10

        agent = Agent()
        disable_batch_agent_background_reviews(agent)
        assert agent._memory_nudge_interval == 0
        assert agent._skill_nudge_interval == 0

    def test_kernel_agent_uses_project_files_not_global_skill_tools(self):
        assert KERNEL_AGENT_TOOLSETS == [
            "terminal", "file", "todo", "triton_ascend"
        ]
        assert "skills" not in KERNEL_AGENT_TOOLSETS
        assert "hermes-cli" not in KERNEL_AGENT_TOOLSETS
        assert "cronjob" not in KERNEL_AGENT_TOOLSETS


class TestCliConfig:
    """Test command-line overrides for formerly hardcoded config."""

    def test_cli_overrides_config(self, tmp_path, monkeypatch):
        import optimize_kernels

        dataset_dir = tmp_path / "dataset"
        state_file = tmp_path / "state" / "optimize_state.json"
        hermes_home = tmp_path / "hermes"
        dataset_dir.mkdir()

        monkeypatch.setattr(sys, "argv", [
            "optimize_kernels.py",
            "--dry-run",
            "--dataset-dir",
            str(dataset_dir),
            "--state-file",
            str(state_file),
            "--hermes-home",
            str(hermes_home),
            "--model",
            "openai/gpt-5.5",
            "--provider",
            "openrouter",
            "--max-iterations",
            "12",
            "--max-pipeline-turns",
            "3",
        ])

        main()

        assert optimize_kernels.DATASET_DIR == dataset_dir
        assert optimize_kernels.STATE_FILE == state_file
        assert optimize_kernels.MODEL == "openai/gpt-5.5"
        assert optimize_kernels.PROVIDER == "openrouter"
        assert optimize_kernels.MAX_ITERATIONS == 12
        assert optimize_kernels.MAX_PIPELINE_TURNS == 3
        assert os.environ["HERMES_HOME"] == str(hermes_home)

    def test_guided_mode_does_not_mutate_plugin_environment_or_config(
            self, tmp_path, monkeypatch):
        import optimize_kernels

        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS",
                           "user-project-value")
        monkeypatch.setattr(sys, "argv", [
            "optimize_kernels.py",
            "--dry-run",
            "--guided-search",
            "--dataset-dir",
            str(dataset_dir),
            "--state-file",
            str(tmp_path / "state.json"),
            "--hermes-home",
            str(tmp_path / "home"),
        ])

        main()

        assert os.environ[
            "HERMES_ENABLE_PROJECT_PLUGINS"] == "user-project-value"
        assert optimize_kernels.GUIDED_SEARCH is True


class TestGetBaselineFile:
    """Test baseline kernel file detection."""

    def test_ignores_base_prefixed_file(self, tmp_path):
        (tmp_path / "25_Swish.py").write_text("# baseline")
        (tmp_path / "base_25_Swish.py").write_text("# reference")
        result = get_baseline_file(tmp_path)
        assert result is not None
        assert result.name == "25_Swish.py"

    def test_detects_number_name_file(self, tmp_path):
        (tmp_path / "99_MyKernel.py").write_text("# kernel")
        result = get_baseline_file(tmp_path)
        assert result is not None
        assert result.name == "99_MyKernel.py"

    def test_returns_none_when_no_files(self, tmp_path):
        result = get_baseline_file(tmp_path)
        assert result is None

    def test_returns_none_when_multiple_candidates(self, tmp_path):
        (tmp_path / "1_A.py").write_text("")
        (tmp_path / "2_B.py").write_text("")
        result = get_baseline_file(tmp_path)
        assert result is None

    def test_ignores_non_number_name_files(self, tmp_path):
        (tmp_path / "helper.py").write_text("")
        result = get_baseline_file(tmp_path)
        assert result is None

    def test_ignores_opt_files(self, tmp_path):
        (tmp_path / "opt_something.py").write_text("")
        (tmp_path / "profile_kernels.py").write_text("")
        result = get_baseline_file(tmp_path)
        assert result is None


class TestKernelIsComplete:
    """Test coarse completeness check."""

    def test_complete_directory(self, tmp_path):
        (tmp_path / "opt_something.py").write_text("")
        (tmp_path / "profile_kernels.py").write_text("")
        (tmp_path / "Optimizations.md").write_text("")
        (tmp_path / "performance_report.md").write_text("")
        (tmp_path / "review.md").write_text("")
        complete, missing = kernel_is_complete(tmp_path)
        assert complete is True
        assert missing == []

    def test_missing_opt(self, tmp_path):
        (tmp_path / "profile_kernels.py").write_text("")
        (tmp_path / "Optimizations.md").write_text("")
        (tmp_path / "performance_report.md").write_text("")
        (tmp_path / "review.md").write_text("")
        complete, missing = kernel_is_complete(tmp_path)
        assert complete is False
        assert "opt_*.py" in missing

    def test_missing_profile(self, tmp_path):
        (tmp_path / "opt_something.py").write_text("")
        (tmp_path / "Optimizations.md").write_text("")
        (tmp_path / "performance_report.md").write_text("")
        (tmp_path / "review.md").write_text("")
        complete, missing = kernel_is_complete(tmp_path)
        assert complete is False
        assert "profile_kernels.py" in missing

    def test_missing_md_deliverables(self, tmp_path):
        (tmp_path / "opt_something.py").write_text("")
        (tmp_path / "profile_kernels.py").write_text("")
        complete, missing = kernel_is_complete(tmp_path)
        assert complete is False
        assert missing == [
            "Optimizations.md", "performance_report.md", "review.md"
        ]

    def test_missing_both(self, tmp_path):
        complete, missing = kernel_is_complete(tmp_path)
        assert complete is False
        assert len(missing) == 5


class TestSetupWorkspace:
    """Test workspace initialization."""

    def test_creates_pipeline_state(self, tmp_path):
        # Create a valid kernel dir
        (tmp_path / "25_Swish.py").write_text("# kernel")
        (tmp_path / "base_25_Swish.py").write_text("# ref")

        state = {"kernels": {}}
        result = setup_workspace(tmp_path, state)
        assert result is True

        sf = tmp_path / ".pipeline_state.json"
        assert sf.exists()
        with open(sf) as f:
            ps = json.load(f)
        assert ps["baseline"] == "25_Swish.py"
        assert ps["current_stage"] == "optimize"

    def test_fails_without_baseline(self, tmp_path):
        state = {"kernels": {}}
        result = setup_workspace(tmp_path, state)
        assert result is False

    def test_idempotent(self, tmp_path):
        (tmp_path / "25_Swish.py").write_text("# kernel")
        state = {"kernels": {}}
        setup_workspace(tmp_path, state)
        setup_workspace(tmp_path, state)  # Should not crash

        sf = tmp_path / ".pipeline_state.json"
        assert sf.exists()

    def test_guided_search_is_explicit_opt_in(self, tmp_path):
        (tmp_path / "25_Swish.py").write_text("# kernel")
        state = {"kernels": {}}
        result = setup_workspace(
            tmp_path,
            state,
            guided_search=True,
            guided_budget=20,
        )
        assert result is True
        pipeline = json.loads(
            (tmp_path / ".pipeline_state.json").read_text(encoding="utf-8"))
        assert pipeline["current_stage"] == "search"
        assert pipeline["guided_search"] == {
            "enabled": True,
            "run_id": None,
            "budget": 20,
            "revision": 0,
            "pre_search_deliverable_hashes": {},
        }

    def test_new_guided_run_clears_terminal_pipeline_flags(self, tmp_path):
        (tmp_path / "25_Swish.py").write_text("# kernel")
        (tmp_path / ".pipeline_state.json").write_text(
            json.dumps({
                "baseline": "25_Swish.py",
                "current_stage": "done",
                "verified": True,
                "verify_failed": True,
                "verify_error": "old failure",
                "recorded": True,
                "results_txt_errors": ["old"],
                "stages_completed": ["optimize", "verify", "record"],
            }))
        setup_workspace(
            tmp_path,
            {"kernels": {}},
            guided_search=True,
            guided_resume=False,
        )
        pipeline = json.loads(
            (tmp_path / ".pipeline_state.json").read_text(encoding="utf-8"))
        for key in (
                "verified",
                "verify_failed",
                "verify_error",
                "recorded",
                "results_txt_errors",
        ):
            assert key not in pipeline
        assert pipeline["stages_completed"] == []

    def test_legacy_mode_has_no_guided_state(self, tmp_path):
        (tmp_path / "25_Swish.py").write_text("# kernel")
        state = {"kernels": {}}
        setup_workspace(tmp_path, state)
        pipeline = json.loads(
            (tmp_path / ".pipeline_state.json").read_text(encoding="utf-8"))
        assert pipeline["current_stage"] == "optimize"
        assert "guided_search" not in pipeline
