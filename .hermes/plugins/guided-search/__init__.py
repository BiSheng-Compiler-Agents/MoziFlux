"""Guided MAP-Elites search plugin for Triton-Ascend optimization."""

from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from .descriptors import (
    BehaviorDescriptor,
    DESCRIPTOR_SCHEMA,
    MECHANISM_SCHEMA,
    NPU_OPTIONS,
    annotate_target,
    candidate_fingerprint,
    classify_candidate,
    describe_coordinate,
)
from .fitness import (
    hardware_fitness,
    hardware_fitness_from_benchmark,
    simulation_fitness,
    trace_wall_cycles,
)
from .gradients import estimate_gradients
from .archive import select_parent, select_target
from .prompts import render_attempt_prompt
from .state import (
    abandon_active_attempt,
    db_path,
    ensure_active_attempt,
    ensure_schema,
    finalize_run,
    get_active_attempt,
    get_candidate,
    get_candidate_by_content_hash,
    get_run,
    increment_attempt_counter,
    initialize_search_run,
    mark_parent_checked_out,
    record_candidate,
    record_llm_iteration,
    record_tool_event,
    run_summary,
    update_attempt_phase,
    update_current_source_hash,
    get_cell_elites,
    has_matching_evaluator_event,
    successful_tool_evidence,
)

logger = logging.getLogger(__name__)
_REQUIRED_ENV: list[str] = []


def _attempt_tool_budget() -> int:
    raw = os.environ.get("GUIDED_SEARCH_ATTEMPT_TOOL_BUDGET", "20")
    try:
        return max(1, int(raw))
    except ValueError:
        return 20


def _guided_search_available() -> bool:
    """Runtime tool-exposure predicate used by every registered tool."""
    try:
        ensure_schema()
        path = db_path()
        return path.exists() and os.access(path.parent, os.W_OK)
    except Exception as exc:
        logger.debug("guided-search unavailable: %s", exc)
        return False


def _remote_verify_available() -> bool:
    return bool(
        os.environ.get("REMOTE_VERIFY_HOST")
        and os.environ.get("REMOTE_VERIFY_USER")
        and os.environ.get("REMOTE_VERIFY_PASS"))


def _sandbox_root() -> Path:
    return Path(
        os.environ.get("KERNEL_SANDBOX_ROOT", "~/CompilerClaw/kernels")
    ).expanduser().resolve()


def _session_workspace(session_id: str) -> Path | None:
    prefix = "kernelbench-"
    if not session_id.startswith(prefix):
        return None
    name = session_id[len(prefix):]
    if "-followup" in name:
        name = name.rsplit("-followup", 1)[0]
    return _sandbox_root() / name


def _load_pipeline(workspace: Path) -> dict[str, Any]:
    path = workspace / ".pipeline_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_pipeline(workspace: Path, state: dict[str, Any]) -> None:
    path = workspace / ".pipeline_state.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _guided_context(session_id: str) -> tuple[Path, dict[str, Any], int] | None:
    if not _guided_search_available():
        return None
    workspace = _session_workspace(session_id)
    if workspace is None or not workspace.is_dir():
        return None
    pipeline = _load_pipeline(workspace)
    guided = pipeline.get("guided_search")
    if not isinstance(guided, dict) or not guided.get("enabled"):
        return None
    if pipeline.get("current_stage") != "search":
        return None
    run_id = guided.get("run_id")
    if not run_id:
        run_id = initialize_search_run(
            workspace=workspace,
            session_id=session_id,
            budget=int(guided.get("budget", 20)),
            target_soc=os.environ.get("CANNSIM_SOC_VERSION", ""),
            wall_time_limit_seconds=guided.get("wall_time_limit_seconds"),
            promotion_domain=(
                "hardware" if _remote_verify_available() else "simulation"),
        )
        pipeline = _load_pipeline(workspace)
    return workspace, pipeline, int(run_id)


def initialize_run_for_workspace(
    workspace: str | Path, session_id: str, budget: int = 20
) -> int:
    """Public setup helper for programmatic orchestrators and tests."""
    return initialize_search_run(
        workspace=Path(workspace), session_id=session_id, budget=budget,
        target_soc=os.environ.get("CANNSIM_SOC_VERSION", ""),
        promotion_domain=(
            "hardware" if _remote_verify_available() else "simulation"),
    )


def _on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    conversation_history: list | None = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    api_request_id: str = "",
    **_: Any,
) -> dict[str, str] | None:
    context = _guided_context(session_id)
    if context is None:
        return None
    workspace, pipeline, run_id = context
    try:
        attempt = ensure_active_attempt(run_id)
        if int(attempt["tool_calls_used"]) >= _attempt_tool_budget():
            abandon_active_attempt(
                run_id,
                f"attempt tool budget exhausted ({_attempt_tool_budget()})",
            )
            attempt = ensure_active_attempt(run_id)
    except RuntimeError as exc:
        stopped_run = get_run(run_id) or {}
        stopped_status = str(stopped_run.get("status", ""))
        if stopped_status.startswith("failed_"):
            instruction = (
                f"SEARCH FAILED ({stopped_status}): "
                f"{stopped_run.get('failure_reason') or exc}. Do not call "
                "guided_search_finalize or kernel_status advance."
            )
        elif stopped_status == "completed":
            instruction = (
                "A hardware-confirmed winner has been materialized. Call "
                "kernel_status(action='advance') to enter FINALIZE."
            )
        else:
            if stopped_run.get("promotion_domain") == "hardware":
                instruction = (
                    "Obtain physical-NPU confirmation for an elite, then call "
                    "guided_search_finalize. Do not call kernel_status advance "
                    "before the winner is materialized."
                )
            else:
                instruction = (
                    "remote_verify is unavailable; finalize the best cannsim elite "
                    "with guided_search_finalize, then call kernel_status advance."
                )
        return {
            "context": (
                "[GUIDED SEARCH - ASCEND MAP-ELITES]\n"
                f"RUN: {run_id}\nSTATUS: {exc}\n"
                f"{instruction}"
            )
        }
    record_llm_iteration(run_id, api_request_id)
    run = get_run(run_id)
    assert run is not None
    parent = get_candidate(attempt.get("parent_candidate_id"))
    return {"context": render_attempt_prompt(run=run, attempt=attempt, parent=parent)}


def _inject_latest_user_context(
    request: dict[str, Any], context: str, *, api_mode: str = ""
) -> dict[str, Any]:
    """Append bounded dynamic context to a request-local user message copy."""
    message_key = (
        "messages" if isinstance(request.get("messages"), list)
        else "input" if isinstance(request.get("input"), list)
        else None)
    if message_key is None:
        return request
    messages = list(request[message_key])
    synthetic = (
        {"role": "user", "content": [{"type": "input_text", "text": context}]}
        if message_key == "input" or api_mode == "codex_responses"
        else {"role": "user", "content": context}
    )
    messages.append(synthetic)
    effective = dict(request)
    effective[message_key] = messages
    return effective


def _on_llm_execution(
    request: dict[str, Any],
    next_call,
    session_id: str = "",
    api_request_id: str = "",
    api_call_count: int = 0,
    api_mode: str = "",
    **_: Any,
) -> Any:
    rendered = _on_pre_llm_call(
        session_id=session_id,
        api_request_id=api_request_id,
    )
    if not rendered or not rendered.get("context"):
        return next_call(request)
    effective = _inject_latest_user_context(
        request, str(rendered["context"]), api_mode=api_mode)
    if effective is request:
        return next_call(request)
    return next_call(effective)


def _on_pre_tool_call(
    tool_name: str = "", args: dict | None = None, session_id: str = "", **_: Any
) -> dict[str, str] | None:
    context = _guided_context(session_id)
    if context is None:
        return None
    workspace, pipeline, run_id = context
    increment_attempt_counter(run_id, "tool_calls_used")
    return None


def _parse_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return parsed if isinstance(parsed, dict) else {"text": result[:4000]}
        except json.JSONDecodeError:
            return {"text": result[:4000]}
    return {"text": str(result)[:4000]}


def _content_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    except OSError:
        return None


def _single_candidate_hash(directory: Path) -> str | None:
    try:
        candidates = sorted(directory.resolve().glob("opt_*.py"))
    except OSError:
        return None
    return _content_hash(candidates[0]) if len(candidates) == 1 else None


def _evaluator_source_hash(
    tool_name: str, args: dict[str, Any], workspace: Path
) -> str | None:
    if tool_name in {"cannsim_local_run", "cannsim_remote_run", "remote_verify"}:
        local_dir = args.get("local_dir")
        if local_dir:
            return _single_candidate_hash(Path(str(local_dir)).expanduser())
        return None
    return _single_candidate_hash(workspace)


def _on_post_tool_call(
    tool_name: str = "",
    args: dict | None = None,
    result: Any = None,
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> None:
    context = _guided_context(session_id)
    if context is None:
        return
    workspace, pipeline, run_id = context
    args = args or {}
    raw = _parse_result(result)
    source_content_hash = _evaluator_source_hash(tool_name, args, workspace)
    if not record_tool_event(
        run_id=run_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        result=raw,
        source_content_hash=source_content_hash,
    ):
        return
    if tool_name in {"patch", "write_file"}:
        if source_content_hash:
            update_current_source_hash(run_id, source_content_hash, "compiling")
        else:
            update_attempt_phase(run_id, "compiling")
    elif tool_name in {"cannsim_local_run", "cannsim_remote_run"}:
        text = json.dumps(raw)
        if raw.get("success") is True and "[HOST] PASS" in text:
            update_attempt_phase(run_id, "testing")
        else:
            update_attempt_phase(run_id, "debugging_compile", raw.get("error", text[:1000]))
    elif tool_name == "remote_verify":
        if raw.get("test_passed") is True:
            update_attempt_phase(run_id, "profiling")
        elif raw.get("test_passed") is False or raw.get("success") is False:
            update_attempt_phase(run_id, "debugging_correctness", raw.get("error", "correctness failed"))


def _on_post_llm_call(session_id: str = "", **_: Any) -> None:
    context = _guided_context(session_id)
    if context is None:
        return
    workspace, pipeline, run_id = context
    summary = run_summary(run_id)
    guided = pipeline.setdefault("guided_search", {})
    guided.update({
        "run_id": run_id,
        "status": summary["status"],
        "failure_reason": summary.get("failure_reason", ""),
        "revision": summary["revision"],
        "completed_attempts": summary["completed_attempts"],
    })
    active = summary.get("active_attempt")
    if active:
        guided["active_attempt"] = {
            "id": active.get("id"),
            "phase": active.get("phase"),
            "status": active.get("status"),
            "tool_calls_used": active.get("tool_calls_used", 0),
            "llm_iterations_used": active.get("llm_iterations_used", 0),
            "last_error": active.get("last_error", ""),
        }
    else:
        guided.pop("active_attempt", None)
    _save_pipeline(workspace, pipeline)


def _on_pre_verify(session_id: str = "", **_: Any) -> dict[str, str] | None:
    context = _guided_context(session_id)
    if context is None:
        return None
    workspace, pipeline, run_id = context
    summary = run_summary(run_id)
    attempt = summary.get("active_attempt")
    if not attempt:
        return None
    return {
        "action": "continue",
        "message": (
            f"Guided-search attempt {attempt['id']} is still in phase "
            f"{attempt['phase']}. Obtain a conclusive evaluation or submit the "
            "attempt as invalid before stopping."
        ),
    }


def _on_session_end(session_id: str = "", **_: Any) -> None:
    # State is transactionally durable after every event; session end is only a
    # diagnostic boundary and must not close an active attempt.
    context = _guided_context(session_id)
    if context is not None:
        workspace, pipeline, run_id = context
        logger.info("guided-search session paused: %s", json.dumps(run_summary(run_id), default=str))


def _resolve_run(session_id: str) -> tuple[Path, dict[str, Any], int]:
    context = _guided_context(session_id)
    if context is None:
        raise RuntimeError("guided search is unavailable or not enabled for this workspace")
    return context


def _handle_status(args: dict[str, Any], **_: Any) -> str:
    try:
        workspace, pipeline, run_id = _resolve_run(str(args.get("session_id", "")))
        return json.dumps(run_summary(run_id), indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _handle_checkout(args: dict[str, Any], **_: Any) -> str:
    try:
        workspace, pipeline, run_id = _resolve_run(str(args.get("session_id", "")))
        attempt = get_active_attempt(run_id)
        if not attempt:
            raise RuntimeError("no active guided-search attempt")
        if attempt["phase"] != "checkout":
            raise RuntimeError(
                f"attempt {attempt['id']} is already in phase {attempt['phase']}; "
                "refusing to overwrite the working candidate")
        parent = get_candidate(attempt.get("parent_candidate_id"))
        if not parent:
            raise RuntimeError("active attempt has no materializable parent")
        baseline = pipeline.get("baseline")
        if not baseline:
            raise RuntimeError("pipeline baseline is unknown")
        output = workspace / f"opt_{baseline}"
        output.write_text(parent["source_text"], encoding="utf-8")
        actual_hash = hashlib.sha256(parent["source_text"].encode("utf-8")).hexdigest()
        mark_parent_checked_out(run_id, actual_hash)
        return json.dumps({
            "ok": True,
            "attempt_id": attempt["id"],
            "parent_candidate_id": parent["id"],
            "path": str(output),
            "source_hash": actual_hash,
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _handle_submit(args: dict[str, Any], **_: Any) -> str:
    try:
        workspace, pipeline, run_id = _resolve_run(str(args.get("session_id", "")))
        candidate_path = Path(str(args.get("candidate_path", ""))).expanduser().resolve()
        if workspace not in candidate_path.parents:
            raise ValueError("candidate_path must be inside the kernel workspace")
        if not candidate_path.name.startswith("opt_") or not candidate_path.is_file():
            raise ValueError("candidate_path must be an existing opt_*.py file")
        source = candidate_path.read_text(encoding="utf-8")
        source_content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        attempt = get_active_attempt(run_id)
        attempt_id = int(attempt["id"]) if attempt else None
        if attempt and attempt.get("kind") == "baseline":
            baseline_parent = get_candidate(attempt.get("parent_candidate_id"))
            if (baseline_parent is None
                    or baseline_parent.get("content_hash") != source_content_hash):
                raise RuntimeError(
                    "baseline calibration must submit the unchanged checked-out "
                    "baseline source")
        archived_source = get_candidate_by_content_hash(run_id, source_content_hash)
        if attempt is None and archived_source is None:
            raise RuntimeError(
                "submission without an active attempt is allowed only for an "
                "already archived source awaiting hardware confirmation")
        launch_options = args.get("launch_options") or (
            json.loads(archived_source["launch_options_json"])
            if archived_source else {})
        environment = args.get("environment") or (
            json.loads(archived_source["environment_json"])
            if archived_source else {})
        descriptor = classify_candidate(source, launch_options, environment)
        simulation_score = None
        simulation = args.get("simulation")
        if isinstance(simulation, dict):
            candidate_metrics = simulation.get("candidate") or {}
            baseline_metrics = simulation.get("baseline") or {}
            forbidden_metrics = {"useful_work", "wall_cycles"}
            if (forbidden_metrics & set(candidate_metrics)
                    or forbidden_metrics & set(baseline_metrics)):
                raise ValueError(
                    "simulation useful_work and wall_cycles are evaluator-owned; "
                    "submit only probe_contract and baseline source_content_hash")
            if (not candidate_metrics.get("probe_contract")
                    or baseline_metrics.get("probe_contract")
                    != candidate_metrics.get("probe_contract")):
                raise ValueError(
                    "simulation promotion requires one matching non-empty probe contract")
            evidence = successful_tool_evidence(
                run_id,
                {"cannsim_local_run", "cannsim_remote_run"},
                source_content_hash=source_content_hash,
                attempt_id=attempt_id,
            )
            if evidence is None:
                raise RuntimeError(
                    "simulation promotion requires successful non-scaffold cannsim "
                    "evidence for this attempt and exact source hash")
            baseline_source_hash = baseline_metrics.get("source_content_hash")
            if not baseline_source_hash:
                raise ValueError(
                    "simulation baseline requires source_content_hash so its trace "
                    "cannot be borrowed from another candidate")
            baseline_evidence = successful_tool_evidence(
                run_id,
                {"cannsim_local_run", "cannsim_remote_run"},
                source_content_hash=str(baseline_source_hash),
                attempt_id=None,
            )
            if baseline_evidence is None:
                raise RuntimeError(
                    "simulation promotion requires source-bound baseline cannsim evidence")
            candidate_metrics = {
                **candidate_metrics,
                # Equal work is defined by the shared probe contract. Use a
                # normalized unit here rather than trusting an LLM-supplied count.
                "useful_work": 1.0,
                "wall_cycles": trace_wall_cycles(evidence),
            }
            baseline_metrics = {
                **baseline_metrics,
                "useful_work": 1.0,
                "wall_cycles": trace_wall_cycles(baseline_evidence),
            }
            simulation_score = simulation_fitness(
                candidate_metrics, baseline_metrics
            )
        hardware_score = None
        hardware = args.get("hardware")
        if isinstance(hardware, dict):
            evidence = successful_tool_evidence(
                run_id,
                {"remote_verify"},
                source_content_hash=source_content_hash,
                attempt_id=attempt_id,
            )
            if evidence is None:
                raise RuntimeError(
                    "hardware promotion requires remote_verify test/benchmark evidence "
                    "for this attempt and exact source hash")
            hardware_score = hardware_fitness_from_benchmark(
                str(evidence.get("bench_output") or "")
            )
        run = get_run(run_id)
        if run is None:
            raise RuntimeError(f"guided-search run {run_id} disappeared")
        required_domain = str(run["promotion_domain"])
        required_score = (
            hardware_score if required_domain == "hardware" else simulation_score)
        if bool(args.get("correct")) and required_score is None:
            if required_domain == "hardware":
                raise RuntimeError(
                    "remote_verify is available; a correct candidate requires "
                    "source-bound physical-NPU correctness and PyTorch/ACL timing")
            raise RuntimeError(
                "remote_verify is unavailable; a correct candidate requires "
                "source-bound equal-work cannsim evidence")
        if (not bool(args.get("correct")) and not has_matching_evaluator_event(
                run_id,
                source_content_hash=source_content_hash,
                attempt_id=attempt_id)):
            raise RuntimeError(
                "an invalid candidate requires source-bound evaluator failure evidence")
        candidate_id = record_candidate(
            run_id=run_id,
            source=source,
            descriptor=descriptor,
            correct=bool(args.get("correct")),
            safe=bool(args.get("safe", True)),
            simulation_score=simulation_score,
            hardware_score=hardware_score,
            launch_options=launch_options,
            environment=environment,
        )
        summary = run_summary(run_id)
        guided = pipeline.setdefault("guided_search", {})
        guided.update({
            "run_id": run_id,
            "revision": summary["revision"],
            "completed_attempts": summary["completed_attempts"],
            "status": summary["status"],
            "failure_reason": summary.get("failure_reason", ""),
        })
        _save_pipeline(workspace, pipeline)
        return json.dumps({
            "ok": True,
            "candidate_id": candidate_id,
            "coordinate": descriptor.coordinate(),
            "descriptor_evidence": [
                item.as_dict() for item in descriptor.evidence
            ],
            "simulation_score": simulation_score,
            "hardware_score": hardware_score,
            "completed_attempts": summary["completed_attempts"],
            "budget": summary["budget"],
        }, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _handle_finalize(args: dict[str, Any], **_: Any) -> str:
    try:
        workspace, pipeline, run_id = _resolve_run(str(args.get("session_id", "")))
        summary = run_summary(run_id)
        if str(summary["status"]).startswith("failed_"):
            raise RuntimeError(
                f"guided-search run failed ({summary['status']}): "
                f"{summary.get('failure_reason') or 'no valid elite'}")
        if (int(summary["completed_attempts"]) < int(summary["budget"])
                and summary["status"] != "ready_to_finalize"):
            raise RuntimeError(
                f"search budget not exhausted: {summary['completed_attempts']}/{summary['budget']}"
            )
        result = finalize_run(run_id)
        refreshed = _load_pipeline(workspace)
        guided = refreshed.setdefault("guided_search", {})
        guided.update({
            "run_id": run_id,
            "status": "completed",
            "failure_reason": "",
            "completed_attempts": summary["completed_attempts"],
            "winner_candidate_id": result.get("candidate_id"),
            "winner_source_hash": result.get("winner_source_hash"),
        })
        guided.pop("active_attempt", None)
        _save_pipeline(workspace, refreshed)
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _tool_schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def register(ctx) -> None:
    for hook_name, callback in (
        ("pre_tool_call", _on_pre_tool_call),
        ("post_tool_call", _on_post_tool_call),
        ("post_llm_call", _on_post_llm_call),
        ("pre_verify", _on_pre_verify),
        ("on_session_end", _on_session_end),
    ):
        ctx.register_hook(hook_name, callback)
    ctx.register_middleware("llm_execution", _on_llm_execution)

    common = {
        "toolset": "triton_ascend",
        "check_fn": _guided_search_available,
        "requires_env": _REQUIRED_ENV,
    }
    ctx.register_tool(
        name="guided_search_status",
        schema=_tool_schema(
            "guided_search_status",
            "Read the active Ascend MAP-Elites run, archive, budget, and sticky attempt.",
            {"session_id": {"type": "string"}},
            ["session_id"],
        ),
        handler=_handle_status,
        **common,
    )
    ctx.register_tool(
        name="guided_search_checkout_parent",
        schema=_tool_schema(
            "guided_search_checkout_parent",
            "Atomically materialize the active attempt's selected parent as opt_*.py. Valid only in CHECKOUT phase.",
            {"session_id": {"type": "string"}},
            ["session_id"],
        ),
        handler=_handle_checkout,
        **common,
    )
    ctx.register_tool(
        name="guided_search_submit_candidate",
        schema=_tool_schema(
            "guided_search_submit_candidate",
            "Submit the current opt_*.py candidate with structured correctness, cannsim, and optional physical-NPU evidence.",
            {
                "session_id": {"type": "string"},
                "candidate_path": {"type": "string"},
                "correct": {"type": "boolean"},
                "safe": {"type": "boolean"},
                "simulation": {
                    "type": "object",
                    "description": (
                        "Request simulation promotion after source-bound cannsim "
                        "runs for candidate and baseline under one fixed-work probe "
                        "contract. Work and wall cycles are not accepted from the caller."),
                    "properties": {
                        "candidate": {
                            "type": "object",
                            "properties": {
                                "probe_contract": {"type": "string"},
                            },
                            "required": ["probe_contract"],
                            "additionalProperties": False,
                        },
                        "baseline": {
                            "type": "object",
                            "properties": {
                                "probe_contract": {"type": "string"},
                                "source_content_hash": {"type": "string"},
                            },
                            "required": ["probe_contract", "source_content_hash"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["candidate", "baseline"],
                    "additionalProperties": False,
                },
                "hardware": {
                    "type": "object",
                    "description": (
                        "Request hardware promotion after source-bound remote_verify. "
                        "Fitness is parsed from raw PyTorch/ACL and Optimized Triton "
                        "benchmark output; caller-provided latency arrays are ignored."),
                },
                "launch_options": {"type": "object"},
                "environment": {"type": "object"},
            },
            ["session_id", "candidate_path", "correct"],
        ),
        handler=_handle_submit,
        **common,
    )
    ctx.register_tool(
        name="guided_search_finalize",
        schema=_tool_schema(
            "guided_search_finalize",
            "After the configured budget, materialize the best hardware-confirmed elite and hand off to kernel-sandbox FINALIZE.",
            {"session_id": {"type": "string"}},
            ["session_id"],
        ),
        handler=_handle_finalize,
        **common,
    )


__all__ = [
    "BehaviorDescriptor",
    "DESCRIPTOR_SCHEMA",
    "MECHANISM_SCHEMA",
    "NPU_OPTIONS",
    "annotate_target",
    "candidate_fingerprint",
    "classify_candidate",
    "describe_coordinate",
    "estimate_gradients",
    "get_cell_elites",
    "hardware_fitness",
    "initialize_run_for_workspace",
    "initialize_search_run",
    "record_candidate",
    "select_parent",
    "select_target",
    "simulation_fitness",
]
