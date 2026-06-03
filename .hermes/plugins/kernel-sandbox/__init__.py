"""
kernel-sandbox plugin
=====================
Sandboxed kernel optimization flow for Triton kernels on Ascend NPU.

Enforces:
  - Agent can only read/write inside its kernel WORKSPACE directory
  - Agent cannot modify the baseline kernel file
  - Agent cannot modify project-level files (plugins/, skills/, optimize_kernels.py)
  - Automatic deliverable verification after each agent turn
  - results.txt is auto-saved from remote_verify and validated for correctness

State is tracked in .pipeline_state.json inside each kernel workspace.

Required environment variables:
  KERNEL_SANDBOX_ROOT  -- absolute path to the kernels/
                          (default: ~/CompilerClaw/kernels)

Plugin hooks registered:
  pre_tool_call   -- block writes to project files, block baseline modification
  pre_llm_call    -- inject workspace context + current stage instructions
  post_llm_call   -- check deliverables + validate results.txt, advance pipeline state
  on_session_end  -- flush state, log summary (works for both CLI and AIAgent API)

Tools registered:
  kernel_status   -- let agent check its own pipeline state
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _sandbox_root() -> Path:
    """Return the sandbox root directory."""
    return Path(
        os.environ.get(
            "KERNEL_SANDBOX_ROOT",
            os.path.expanduser("~/CompilerClaw/kernels"),
        )).resolve()


# Protected-paths check: block writes to ANY file under the project directory
# EXCEPT the kernel sandbox root (where kernels live).
_SANDBOX_ROOT = _sandbox_root()

# Deliverable file patterns (checked after each agent turn)
_DELIVERABLES = [
    ("opt_*.py", "opt_{baseline}.py"),
    ("profile_kernels.py", "profile_kernels.py"),
    ("Optimizations.md", "Optimizations.md"),
    ("performance_report.md", "performance_report.md"),
    ("review.md", "review.md"),
    ("results.txt", "results.txt"),
]


def _path_matches_protected(path_str: str) -> bool:
    """Return True if path_str is inside the project directory but OUTSIDE
    the kernel sandbox root."""
    try:
        resolved = Path(path_str).expanduser().resolve()
    except (OSError, RuntimeError):
        return True

    project = _project_dir()
    if not str(resolved).startswith(str(project)):
        return False

    sandbox = _SANDBOX_ROOT
    if str(resolved).startswith(str(sandbox)):
        return False

    return True


def _session_workspace(session_id: str) -> Path | None:
    """Derive the workspace Path from a session_id like 'kernelbench-l1_25_Swish'."""
    prefix = "kernelbench-"
    if session_id.startswith(prefix):
        kernel_name = session_id[len(prefix):]
        kernel_name = re.sub(r"-followup\d+$", "", kernel_name)
        return _sandbox_root() / kernel_name
    return None


def _state_file(workspace: Path) -> Path:
    return workspace / ".pipeline_state.json"


def _load_state(workspace: Path) -> dict:
    sf = _state_file(workspace)
    if sf.exists():
        try:
            with open(sf) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_state(workspace: Path, state: dict) -> None:
    sf = _state_file(workspace)
    tmp = sf.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(sf)


def _baseline_name(workspace: Path) -> str | None:
    """Return the input kernel .py filename in the workspace, or None."""
    state = _load_state(workspace)
    if state.get("baseline"):
        return state["baseline"]

    all_py = [
        f for f in workspace.glob("*.py")
        if not f.name.startswith("opt_") and not f.name.startswith("profile_")
        and not f.name.startswith("base_") and not f.name.startswith(".")
    ]

    input_files = [f for f in all_py if re.match(r'^\d+_', f.name)]
    if len(input_files) == 1:
        return input_files[0].name

    if len(all_py) == 1:
        return all_py[0].name
    return None


def _remote_verify_available() -> bool:
    return bool(
        os.environ.get("REMOTE_VERIFY_HOST")
        and os.environ.get("REMOTE_VERIFY_USER")
        and os.environ.get("REMOTE_VERIFY_PASS"))


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_is_baseline(path_str: str, workspace: Path) -> bool:
    """Return True if path_str is the input kernel file the agent should optimize."""
    try:
        resolved = Path(path_str).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    baseline = _baseline_name(workspace)
    return baseline is not None and resolved.name == baseline


def _path_is_reference(path_str: str, workspace: Path) -> bool:
    """Return True if path_str is a base_*.py reference kernel file."""
    try:
        resolved = Path(path_str).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    return resolved.name.startswith("base_") and resolved.name.endswith(".py")


# ---------------------------------------------------------------------------
# results.txt: save + validate
# ---------------------------------------------------------------------------

DELIVERABLE_RESULTS_TXT = "results.txt"


def _check_deliverables(
        workspace: Path,
        require_results: bool = True) -> tuple[bool, list[str]]:
    """Return (all_present, [missing_patterns]).

    When *require_results* is True and remote verify hardware is available,
    results.txt is also required.
    """
    py_files = {f.name for f in workspace.glob("*.py")}
    md_files = {f.name for f in workspace.glob("*.md")}
    all_files = {f.name for f in workspace.iterdir() if f.is_file()}
    missing: list[str] = []

    for pattern, canonical in _DELIVERABLES:
        if pattern.startswith("opt_"):
            if not any(f.startswith("opt_") for f in py_files):
                missing.append(canonical)
        elif pattern == "profile_kernels.py":
            if "profile_kernels.py" not in py_files:
                missing.append(canonical)
        elif pattern == DELIVERABLE_RESULTS_TXT:
            # Only require results.txt when requested AND remote verify is available
            if require_results and _remote_verify_available():
                if DELIVERABLE_RESULTS_TXT not in all_files:
                    missing.append(canonical)
        else:
            if pattern not in md_files:
                missing.append(pattern)

    return len(missing) == 0, missing


def _save_results_txt(workspace: Path, raw: dict) -> None:
    """Write remote_verify output to results.txt."""
    results_path = workspace / DELIVERABLE_RESULTS_TXT
    test_output = raw.get("test_output", "") or ""
    bench_output = raw.get("bench_output", "") or ""

    combined = test_output
    if bench_output:
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += bench_output

    results_path.write_text(combined, encoding="utf-8")
    logger.info("kernel-sandbox: saved results.txt for %s", workspace.name)


def _validate_results_txt(workspace: Path) -> tuple[bool, list[str]]:
    """Validate results.txt content.

    Checks:
      1. File exists and is non-empty
      2. UNIT_TEST PASS is present
      3. All three providers (baseline1, baseline2, optimized) have TEST entries
      4. optimized has at least one PASS and no MISMATCH
      5. Benchmark table is present

    Returns (valid, [error_messages]).
    """
    results_path = workspace / DELIVERABLE_RESULTS_TXT
    errors: list[str] = []

    if not results_path.exists():
        return False, ["results.txt does not exist"]

    content = results_path.read_text(encoding="utf-8")
    lines = content.strip().splitlines()

    if not lines:
        return False, ["results.txt is empty"]

    # Detect canonical format: must have both UNIT_TEST lines AND
    # " optimized " (with spaces) in TEST lines.
    # Old-format results.txt (from previous agent runs) use different
    # structure and should not be validated.
    has_unit_test = any(line.startswith("UNIT_TEST") for line in lines)
    has_canonical_test = any(
        line.startswith("TEST ") and " optimized " in line for line in lines)
    if not has_unit_test or not has_canonical_test:
        logger.info(
            "kernel-sandbox: results.txt for %s uses old format, skipping validation",
            workspace.name,
        )
        return True, []

    # Check UNIT_TEST result
    unit_test_lines = [ln for ln in lines if ln.startswith("UNIT_TEST")]
    if not unit_test_lines:
        errors.append("No UNIT_TEST line found in results.txt")
    elif not any("PASS" in ln for ln in unit_test_lines):
        errors.append(f"UNIT_TEST did not pass: {unit_test_lines}")

    # Check that all three providers have TEST entries
    test_lines = [ln for ln in lines if ln.startswith("TEST ")]
    providers_found = set()
    for line in test_lines:
        for provider in ("baseline1", "baseline2", "optimized"):
            if f" {provider} " in line:
                providers_found.add(provider)

    for provider in ("baseline1", "baseline2", "optimized"):
        if provider not in providers_found:
            errors.append(f"No TEST entries found for provider: {provider}")

    # If base_*.py exists in the workspace, baseline2 MUST be in the results
    base_files = list(workspace.glob("base_*.py"))
    if base_files and "baseline2" not in providers_found:
        errors.append(
            f"base_*.py exists in workspace ({base_files[0].name}) but baseline2 "
            f"has no TEST entries in results.txt — profile must compare against base_*.py"
        )

    # Check optimized has at least one PASS and no MISMATCH
    opt_lines = [ln for ln in test_lines if " optimized " in ln]
    if opt_lines:
        opt_pass = [ln for ln in opt_lines if "PASS" in ln]
        opt_mismatch = [ln for ln in opt_lines if "MISMATCH" in ln]
        if not opt_pass:
            errors.append("optimized provider has no PASS lines")
        if opt_mismatch:
            errors.append("optimized provider has MISMATCH")
    else:
        errors.append("No TEST lines for optimized provider")

    # Check benchmark table is present
    bench_content_lines = [
        ln for ln in lines
        if not ln.startswith("TEST ") and not ln.startswith("UNIT_TEST") and
        not ln.startswith("BENCH ") and not ln.startswith("/") and ln.strip()
    ]
    has_table = any("PyTorch" in ln or "Baseline" in ln or "Optimized" in ln
                    for ln in bench_content_lines)
    if not has_table:
        has_table = any("label" in ln.lower() and (
            "torch" in ln.lower() or "acl" in ln.lower())
                        for ln in bench_content_lines)
    if not has_table:
        errors.append("No benchmark table found in results.txt")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Pipeline stage management
# ---------------------------------------------------------------------------


def _current_stage(state: dict) -> str:
    return state.get("current_stage", "optimize")


def _advance_stage(state: dict, workspace: Path) -> tuple[str, str]:
    """Maybe advance the pipeline stage based on deliverable completeness
    and results.txt validity. Returns (new_stage, message)."""
    stage = _current_stage(state)
    rv_avail = _remote_verify_available()

    # Only require results.txt when leaving verify (not entering it)
    require_results = stage == "verify"
    complete, missing = _check_deliverables(workspace,
                                            require_results=require_results)

    if stage == "optimize":
        if complete:
            state["current_stage"] = "verify"
            state["stages_completed"] = state.get("stages_completed",
                                                  []) + ["optimize"]
            _save_state(workspace, state)
            if rv_avail:
                verify_instr = (
                    "ALL DELIVERABLES PRESENT. Stage advanced to VERIFY.\n"
                    "Run remote_verify(local_dir=..., run_test=True, run_bench=True) "
                    "on physical NPU hardware. The pipeline auto-detects the result. "
                    "results.txt will be auto-saved and validated.")
            else:
                verify_instr = (
                    "ALL DELIVERABLES PRESENT. Stage advanced to VERIFY.\n"
                    "Remote NPU hardware is not available. Run cannsim_local_run and "
                    "if correct set verified=True via kernel_status.")
            return "verify", verify_instr
        else:
            return stage, (
                f"INCOMPLETE - missing: {', '.join(missing)}. "
                f"Write ONLY the missing files. Do not re-optimize the kernel."
            )

    elif stage == "verify":
        if state.get("verified"):
            state["current_stage"] = "record"
            state["stages_completed"] = state.get("stages_completed",
                                                  []) + ["verify"]
            _save_state(workspace, state)
            return "record", (
                "VERIFICATION PASSED. Stage advanced to RECORD.\n"
                "Use episode_write to record this optimization attempt.")
        elif state.get("verify_failed"):
            state["current_stage"] = "optimize"
            state["stages_completed"] = state.get("stages_completed",
                                                  []) + ["verify_fail"]
            _save_state(workspace, state)
            return "optimize", (
                f"VERIFICATION FAILED: {state.get('verify_error', 'unknown')}. "
                f"Re-enter OPTIMIZE stage. Fix the issues and re-produce deliverables."
            )
        else:
            if rv_avail:
                msg = (
                    "VERIFY stage: run remote_verify(local_dir=..., run_test=True, run_bench=True) "
                    "to validate on physical NPU hardware. "
                    "results.txt will be auto-saved and validated. "
                    "Do NOT use kernel_status to set verified.")
            else:
                msg = (
                    "VERIFY stage: remote NPU hardware is not available. "
                    "Run cannsim_local_run and if confident set verified=True via kernel_status."
                )
            return stage, msg

    elif stage == "record":
        if state.get("recorded"):
            state["current_stage"] = "done"
            state["stages_completed"] = state.get("stages_completed",
                                                  []) + ["record"]
            _save_state(workspace, state)
            return "done", "All stages complete. Pipeline DONE."
        else:
            return stage, (
                "RECORD stage: use episode_write to save this optimization, "
                "then mark recorded via kernel_status.")

    return stage, ""


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------


def _on_pre_tool_call(
    tool_name: str = "",
    args: dict | None = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> dict | None:
    """Block tools that would modify protected project files, the input kernel
    file, or reference kernel files."""
    args = args or {}
    workspace = _session_workspace(session_id)
    if workspace is None:
        return None

    if tool_name in ("write_file", "patch"):
        target = args.get("path", "")

        if _path_matches_protected(target):
            msg = f"BLOCKED: '{target}' is a project-level file and cannot be modified."
            logger.warning("kernel-sandbox: %s", msg)
            return {"action": "block", "message": msg}

        if _path_is_baseline(target, workspace):
            baseline = _baseline_name(workspace)
            msg = (
                f"BLOCKED: '{target}' is the input kernel file ({baseline}). "
                f"You must NOT modify it. Write to opt_{baseline} instead.")
            logger.warning("kernel-sandbox: %s", msg)
            return {"action": "block", "message": msg}

        if _path_is_reference(target, workspace):
            msg = (
                f"BLOCKED: '{target}' is a reference kernel file (base_*.py). "
                f"You must NOT read or modify it.")
            logger.warning("kernel-sandbox: %s", msg)
            return {"action": "block", "message": msg}

    if tool_name == "read_file":
        target = args.get("path", "")
        if _path_is_reference(target, workspace):
            msg = (
                f"BLOCKED: '{target}' is a reference kernel file (base_*.py). "
                f"You must NOT read it.")
            logger.warning("kernel-sandbox: %s", msg)
            return {"action": "block", "message": msg}

    if tool_name == "terminal":
        cmd = args.get("command", "")

        if re.search(r"\b(cat|head|tail|less|more|grep)\s+.*base_\S*\.py\b",
                     cmd):
            msg = (
                "BLOCKED: terminal command would read a reference kernel file "
                "(base_*.py). You must NOT read the reference kernel.")
            logger.warning("kernel-sandbox: %s", msg)
            return {"action": "block", "message": msg}

        _dangerous_patterns = [
            r"\brm\s+(-rf?|--recursive)\s+.*(\.hermes|skills|plugins|optimize_kernels)",
            r"\b(cp|mv|install|pip\s+install)\s+.*(\.hermes|skills|plugins)",
            r"\b(write_file|patch)\s*\(.*(\.hermes|skills|plugins|optimize_kernels)",
            r"cd\s+.*(&&|\|\|)\s*(rm|cp|mv|pip|install)",
        ]
        for pat in _dangerous_patterns:
            if re.search(pat, cmd):
                msg = f"BLOCKED: command targets project files.\nCommand: {cmd[:200]}"
                logger.warning("kernel-sandbox: %s", msg)
                return {"action": "block", "message": msg}

    return None


def _on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    conversation_history: list | None = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    **_: Any,
) -> dict | None:
    """Inject workspace context and current stage instructions."""
    workspace = _session_workspace(session_id)
    if workspace is None:
        return None

    state = _load_state(workspace)
    stage = _current_stage(state)
    baseline = _baseline_name(workspace)

    parts: list[str] = []
    parts.append(f"[KERNEL SANDBOX - session: {session_id}]")
    parts.append(f"WORKSPACE: {workspace}")
    parts.append(
        f"INPUT:  {baseline}  (read this file, write to opt_{baseline})")
    parts.append("REFERENCE FILES:  base_*.py  (DO NOT read or modify)")
    parts.append(f"STAGE:     {stage}")

    if stage == "optimize":
        parts.append(
            "TASK: Optimize the baseline kernel using cannsim_local_run. "
            "Produce all 5 deliverables: opt_<name>.py, profile_kernels.py, "
            "Optimizations.md, performance_report.md, and review.md."
            f"Write ONLY inside {workspace}.")
    elif stage == "verify":
        rv_avail = _remote_verify_available()
        if rv_avail:
            parts.append(
                "TASK: Run remote_verify(local_dir=..., run_test=True, run_bench=True) "
                "on physical NPU hardware. results.txt will be auto-saved and validated. "
                "The pipeline auto-detects pass/fail from the tool output. "
                "cannsim_local_run is fine for bottleneck analysis, but does NOT count "
                "as the verification result.")
        else:
            parts.append(
                "TASK: Verify the optimized kernel. Remote NPU hardware is not available. "
                "Run cannsim_local_run on opt_*.py, analyze the trace, and if correct "
                "set verified=True via kernel_status.")
    elif stage == "record":
        parts.append("TASK: Record this optimization with episode_write. "
                     "Then mark with kernel_status(recorded=True).")
    elif stage == "done":
        parts.append("STATUS: Pipeline complete. No further action needed.")

    # Surface results.txt validation errors if any
    if state.get("results_txt_errors"):
        parts.append(
            f"RESULTS.TXT ERRORS: {'; '.join(state['results_txt_errors'])}")

    ctx = "\n".join(parts)
    logger.debug("kernel-sandbox: pre_llm_call injecting context for %s",
                 session_id)
    return {"context": ctx}


def _on_post_tool_call(
    tool_name: str = "",
    args: dict | None = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> None:
    """Auto-set pipeline verification flags based on remote_verify results.
    Saves and validates results.txt."""
    if tool_name != "remote_verify":
        return
    workspace = _session_workspace(session_id)
    if workspace is None:
        return
    state = _load_state(workspace)
    stage = _current_stage(state)
    if stage != "verify":
        return

    try:
        raw = json.loads(result) if isinstance(result, str) else result
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning(
            "kernel-sandbox: post_tool_call - failed to parse remote_verify result"
        )
        return

    if not isinstance(raw, dict):
        return

    # Save raw remote_verify output to results.txt
    try:
        _save_results_txt(workspace, raw)
    except Exception as exc:
        logger.warning("kernel-sandbox: failed to save results.txt for %s: %s",
                       session_id, exc)

    # Validate results.txt content (only when remote verify is available)
    if _remote_verify_available():
        try:
            valid, errors = _validate_results_txt(workspace)
            if not valid:
                logger.warning(
                    "kernel-sandbox: results.txt VALIDATION FAILED for %s: %s",
                    session_id,
                    "; ".join(errors),
                )
                state["results_txt_errors"] = errors
                # Override: don't mark verified if results.txt has issues
                if raw.get("test_passed") is True:
                    logger.warning(
                        "kernel-sandbox: overriding verified=True - results.txt has errors: %s",
                        "; ".join(errors),
                    )
                    _save_state(workspace, state)
                    return
        except Exception as exc:
            logger.warning(
                "kernel-sandbox: results.txt validation error for %s: %s",
                session_id, exc)

    if raw.get("test_passed") is True:
        state["verified"] = True
        logger.info(
            "kernel-sandbox: remote_verify PASSED for %s - auto-set verified=True",
            session_id,
        )
    elif raw.get("success") is False:
        state["verify_failed"] = True
        state["verify_error"] = raw.get("error", "remote_verify failed")
        logger.warning(
            "kernel-sandbox: remote_verify FAILED for %s - auto-set verify_failed=True",
            session_id,
        )
    elif raw.get("test_passed") is False:
        state["verify_failed"] = True
        state["verify_error"] = "correctness test failed (test_passed=False)"
        logger.warning(
            "kernel-sandbox: remote_verify test FAILED for %s - auto-set verify_failed=True",
            session_id,
        )
    else:
        return  # no conclusive pass/fail yet

    _save_state(workspace, state)
    _advance_stage(state, workspace)


def _on_post_llm_call(
    session_id: str = "",
    user_message: str = "",
    assistant_response: str = "",
    conversation_history: list | None = None,
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    """After each turn: check deliverables + validate results.txt, advance pipeline."""
    workspace = _session_workspace(session_id)
    if workspace is None:
        return

    state = _load_state(workspace)
    complete, missing = _check_deliverables(workspace)
    stage = _current_stage(state)

    state["deliverables_complete"] = complete
    state["deliverables_missing"] = missing
    state["last_turn_at"] = datetime.now(timezone.utc).isoformat()
    state["turn_count"] = state.get("turn_count", 0) + 1

    # Stage-conditional advancement
    _save_state(workspace, state)

    logger.info(
        "kernel-sandbox: post_llm_call session=%s stage=%s complete=%s missing=%s turns=%d",
        session_id,
        stage,
        complete,
        missing,
        state["turn_count"],
    )


def _on_session_end(
    session_id: str = "",
    completed: bool = True,
    interrupted: bool = False,
    **_: Any,
) -> None:
    workspace = _session_workspace(session_id)
    if workspace is None:
        return
    state = _load_state(workspace)
    logger.info(
        "kernel-sandbox: session_end session=%s workspace=%s completed=%s interrupted=%s state=%s",
        session_id,
        workspace,
        completed,
        interrupted,
        json.dumps(state),
    )


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _handle_status(args: dict, **_) -> str:
    session_id = args.get("session_id", "")
    workspace = _session_workspace(session_id)
    if workspace is None:
        return json.dumps(
            {"error": f"unknown workspace for session: {session_id}"})

    state = _load_state(workspace)
    action = args.get("action", "read")

    if action == "read":
        complete, missing = _check_deliverables(workspace)
        return json.dumps(
            {
                "workspace": str(workspace),
                "stage": _current_stage(state),
                "baseline": _baseline_name(workspace),
                "deliverables_complete": complete,
                "deliverables_missing": missing,
                "results_txt_errors": state.get("results_txt_errors", []),
                "stages_completed": state.get("stages_completed", []),
                "turn_count": state.get("turn_count", 0),
            },
            indent=2)

    elif action == "advance":
        new_stage, msg = _advance_stage(state, workspace)
        return json.dumps({"stage": new_stage, "message": msg})

    elif action == "set":
        key = args.get("key", "")
        value = args.get("value")
        allowed_keys = ["recorded", "verify_failed"]
        if key == "verified":
            if not _remote_verify_available():
                allowed_keys.append("verified")
            else:
                return json.dumps({
                    "error":
                    ("Remote verify hardware IS available. You MUST call remote_verify. "
                     "Setting verified=True manually is blocked.")
                })
        if key in allowed_keys:
            state[key] = value
            if key == "verify_failed" and value:
                state["verify_error"] = args.get("error", "unknown")
            _save_state(workspace, state)
            return json.dumps({"ok": True, "key": key, "value": value})
        return json.dumps({"error": f"unknown key: {key}"})

    return json.dumps({"error": f"unknown action: {action}"})


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)

    ctx.register_tool(
        name="kernel_status",
        toolset="triton_ascend",
        schema={
            "name":
            "kernel_status",
            "description":
            ("Check or advance the kernel optimization pipeline state. "
             "Actions: read (check status), advance (move to next stage), "
             "set (mark verified/recorded/failed). "
             "Always pass session_id explicitly."),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type":
                        "string",
                        "description":
                        "The session ID, e.g. 'kernelbench-l1_25_Swish'.",
                    },
                    "action": {
                        "type":
                        "string",
                        "enum": ["read", "advance", "set"],
                        "description":
                        "What to do: read status, advance stage, or set a flag.",
                    },
                    "key": {
                        "type": "string",
                        "enum": ["recorded", "verify_failed"],
                        "description": "Flag to set (only with action=set).",
                    },
                    "value": {
                        "type": "boolean",
                        "description": "Value to set (only with action=set).",
                    },
                    "error": {
                        "type":
                        "string",
                        "description":
                        "Error message (only with action=set key=verify_failed).",
                    },
                },
                "required": ["session_id"],
            },
        },
        handler=_handle_status,
        requires_env=[],
        check_fn=lambda: True,
    )
