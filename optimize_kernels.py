"""
optimize_kernels.py — Thin orchestrator for sandboxed Triton kernel optimization.

Uses the kernel-sandbox plugin to enforce workspace boundaries, inject
per-stage context, and verify deliverables. The plugin handles all
sandboxing; this script only:
  1. Discovers kernels
  2. Creates workspaces with .pipeline_state.json
  3. Launches the Hermes agent (plugin does the rest)
  4. Reads final state for reporting

Usage:
    python optimize_kernels.py                          # all pending kernels
    python optimize_kernels.py --dry-run                # show what would run
    python optimize_kernels.py --level 1                # only Level-1
    python optimize_kernels.py --kernel l1_25_Swish     # single kernel
    python optimize_kernels.py --retry-failed           # re-run failed/incomplete
    python optimize_kernels.py --force                  # ignore state, re-run all
    python optimize_kernels.py --max 5                  # limit to N kernels
    python optimize_kernels.py --workers 2              # parallel (default: 1)
    python optimize_kernels.py --model openai/gpt-5.5 --provider openrouter
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ── Hermes env ──────────────────────────────────────────────────────────────
DEFAULT_HERMES_HOME = os.environ.get("HERMES_HOME", "/opt/data")
os.environ.setdefault("HERMES_HOME", DEFAULT_HERMES_HOME)
_PROJECT_DIR = Path(__file__).parent.resolve()
os.chdir(_PROJECT_DIR)
sys.path.insert(0, str(_PROJECT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimize_kernels")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATASET_DIR = Path(
    os.environ.get("OPTIMIZE_DATASET_DIR",
                   ROOT / "datasets" / "KernelBench_Triton"))
STATE_FILE = Path(
    os.environ.get("OPTIMIZE_STATE_FILE", ROOT / "optimize_state.json"))

# ── Agent config ─────────────────────────────────────────────────────────────
MODEL = os.environ.get("OPTIMIZE_MODEL", "deepseek/deepseek-v4-flash")
PROVIDER = os.environ.get("OPTIMIZE_PROVIDER", "openrouter")
MAX_ITERATIONS = int(os.environ.get("OPTIMIZE_MAX_ITERATIONS", "90"))

# Max times we re-invoke the agent to drive the pipeline forward within one
# kernel. The agent's single run_conversation call ends its turn whenever it
# stops emitting tool calls — which can leave the pipeline parked mid-stage
# (e.g. at "verify" after deliverables are written but before remote_verify is
# run). We re-enter the conversation with a continuation nudge until the
# pipeline reaches "done" or stalls (no stage progress between turns).
MAX_PIPELINE_TURNS = int(os.environ.get("OPTIMIZE_MAX_PIPELINE_TURNS", "6"))
GUIDED_SEARCH = False
GUIDED_SEARCH_BUDGET = int(os.environ.get("GUIDED_SEARCH_BUDGET", "20"))
GUIDED_SEARCH_WALL_TIME_MINUTES: float | None = None
GUIDED_SEARCH_RESUME = False
GUIDED_SEARCH_SEED_RUN_ID: int | None = None
KERNEL_AGENT_TOOLSETS = ["terminal", "file", "todo", "triton_ascend"]


def disable_batch_agent_background_reviews(agent) -> None:
    """Keep autonomous batch runs from mutating global memory or skills."""
    agent._memory_nudge_interval = 0
    agent._skill_nudge_interval = 0

# Continuation prompt for re-entering a parked pipeline. The kernel-sandbox
# pre_llm_call hook injects the authoritative stage instructions; this just
# tells the agent to act on them.
CONTINUE_PROMPT = (
    "Your pipeline is not yet DONE. Read the STAGE and TASK in the sandbox "
    "context above and perform the next required action now (do not just "
    "report status). If the stage is VERIFY, run the verification tool the "
    "context names. Continue until the pipeline reaches the 'done' stage.")

# ── Per-kernel task prompt (plugin injects workspace + stage context) ───────
TASK_PROMPT = """
Optimize the Triton kernel in this directory.

Required deliverables (all 5 must be produced):
1. opt_<baseline>.py       — Optimized kernel + ModelNew host interface
2. profile_kernels.py       — @perf_report benchmark, all dispatch paths, unit test
3. Optimizations.md         — Each optimization applied, with code snippets and rationale
4. performance_report.md    — cannsim trace tables (baseline vs optimized), hardware latency
5. review.md                — Static P0/P1/P2 review of the optimized kernel

Use cannsim_local_run to simulate the kernel and get trace data.
Use kernel_status to check your pipeline state and mark stages complete.
"""

GUIDED_TASK_PROMPT = """
Run guided MAP-Elites search for the Triton kernel in this directory.

The guided-search and kernel-sandbox hooks provide the authoritative active
attempt, parent, mutation direction, and stage instructions. During SEARCH,
edit only opt_<baseline>.py, keep each attempt sticky through compile,
correctness, and evaluation, then call guided_search_submit_candidate. Do not
create final deliverables or call kernel_status advance until guided search has
selected and materialized a hardware-confirmed winner.
"""

# ── State management (orchestrator-level, separate from plugin state) ────────


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"kernels": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)


def mark_kernel(state: dict, name: str, status: str, detail: str = "") -> None:
    state["kernels"][name] = {
        "status": status,
        "detail": detail,
        "model": MODEL,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)


# ── Kernel discovery ────────────────────────────────────────────────────────


def discover_kernels(level: int = None) -> list[Path]:
    dirs = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())
    if level is not None:
        dirs = [d for d in dirs if d.name.startswith(f"l{level}_")]
    return dirs


def get_baseline_file(kernel_dir: Path) -> Path | None:
    """Return the agent-readable kernel file (<Number>_<name>.py), or None.

    base_*.py files are read-only references and must NOT be selected here.
    """
    all_py = [
        f for f in kernel_dir.glob("*.py")
        if not f.name.startswith("opt_") and not f.name.startswith("base_")
        and "profile" not in f.name.lower() and not f.name.startswith(".")
    ]
    candidates = [f for f in all_py if re.match(r"^\d+_.+\.py$", f.name)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def kernel_is_complete(kernel_dir: Path) -> tuple[bool, list[str]]:
    """Check that all required optimization deliverables are present."""
    py_files = {f.name for f in kernel_dir.glob("*.py")}
    all_files = {f.name for f in kernel_dir.iterdir() if f.is_file()}
    missing = []
    if not any(f.startswith("opt_") for f in py_files):
        missing.append("opt_*.py")
    if "profile_kernels.py" not in py_files:
        missing.append("profile_kernels.py")
    required_files = ["Optimizations.md", "performance_report.md", "review.md"]
    if os.environ.get("REMOTE_VERIFY_HOST"):
        required_files.append("results.txt")
    for name in required_files:
        if name not in all_files:
            missing.append(name)
    return len(missing) == 0, missing


# ── Workspace setup ──────────────────────────────────────────────────────────


def setup_workspace(
    kernel_dir: Path,
    state: dict,
    *,
    guided_search: bool = False,
    guided_budget: int = 20,
    guided_wall_time_minutes: float | None = None,
    guided_resume: bool = False,
    guided_seed_run_id: int | None = None,
) -> bool:
    """
    Ensure the kernel directory has a .pipeline_state.json.
    Returns True if ready to run.
    """
    baseline = get_baseline_file(kernel_dir)
    if baseline is None:
        log.warning("  ✗ %s — cannot find unique baseline file",
                    kernel_dir.name)
        mark_kernel(state, kernel_dir.name, "failed",
                    "no unique baseline file")
        return False

    sf = kernel_dir / ".pipeline_state.json"
    if not sf.exists():
        pipeline_state = {
            "baseline":
            baseline.name,
            "current_stage":
            "optimize",
            "stages_completed": [],
            "deliverables_complete":
            False,
            "deliverables_missing": [
                "opt_*.py", "profile_kernels.py", "Optimizations.md",
                "performance_report.md", "review.md"
            ],
            "turn_count":
            0,
            "created_at":
            datetime.now(timezone.utc).isoformat(),
        }
        with open(sf, "w") as f:
            json.dump(pipeline_state, f, indent=2)
        log.info("  Initialized .pipeline_state.json for %s", kernel_dir.name)

    if guided_search:
        try:
            with open(sf) as f:
                pipeline_state = json.load(f)
        except (json.JSONDecodeError, OSError):
            pipeline_state = {
                "baseline": baseline.name,
                "stages_completed": []
            }
        previous = pipeline_state.get("guided_search")
        run_id = (previous.get("run_id")
                  if guided_resume and isinstance(previous, dict) else None)
        if not guided_resume:
            for key in (
                    "verified",
                    "verify_failed",
                    "verify_error",
                    "recorded",
                    "results_txt_errors",
                    "last_turn_at",
            ):
                pipeline_state.pop(key, None)
            pipeline_state["stages_completed"] = []
            pipeline_state["turn_count"] = 0
        pipeline_state["current_stage"] = "search"
        final_names = (
            "profile_kernels.py",
            "Optimizations.md",
            "performance_report.md",
            "review.md",
            "results.txt",
        )
        pre_search_hashes = {}
        for name in final_names:
            path = kernel_dir / name
            if path.is_file():
                pre_search_hashes[name] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
        pipeline_state["guided_search"] = {
            "enabled":
            True,
            "run_id":
            run_id,
            "budget":
            max(1, int(guided_budget)),
            "revision": (int(previous.get("revision", 0)) if isinstance(
                previous, dict) else 0),
            "pre_search_deliverable_hashes":
            pre_search_hashes,
        }
        if guided_wall_time_minutes is not None:
            pipeline_state["guided_search"]["wall_time_limit_seconds"] = (
                max(0.0, float(guided_wall_time_minutes)) * 60.0)
        if guided_seed_run_id is not None:
            pipeline_state["guided_search"]["seed_run_id"] = int(
                guided_seed_run_id)
        pipeline_state["deliverables_complete"] = False
        pipeline_state["deliverables_missing"] = ["guided-search completion"]
        with open(sf, "w") as f:
            json.dump(pipeline_state, f, indent=2)
        log.info("  Guided search enabled for %s (budget=%d, resume=%s)",
                 kernel_dir.name, guided_budget, guided_resume)

    return True


# ── Per-kernel optimization ──────────────────────────────────────────────────


def optimize_kernel(kernel_dir: Path, state: dict) -> dict:
    """
    Run the Hermes agent on a single kernel directory.
    The kernel-sandbox plugin handles all sandboxing, context injection,
    and deliverable verification via hooks.
    """
    from run_agent import AIAgent

    name = kernel_dir.name
    log.info("▶ Starting %s", name)
    t0 = time.time()

    # Setup workspace (creates .pipeline_state.json if needed)
    if not setup_workspace(
            kernel_dir,
            state,
            guided_search=GUIDED_SEARCH,
            guided_budget=GUIDED_SEARCH_BUDGET,
            guided_wall_time_minutes=GUIDED_SEARCH_WALL_TIME_MINUTES,
            guided_resume=GUIDED_SEARCH_RESUME,
            guided_seed_run_id=GUIDED_SEARCH_SEED_RUN_ID):
        return {
            "kernel": name,
            "status": "failed",
            "detail": "workspace setup failed",
            "elapsed_s": 0.0
        }

    prompt = (GUIDED_TASK_PROMPT if GUIDED_SEARCH else TASK_PROMPT).strip()

    agent = AIAgent(
        model=MODEL,
        provider=PROVIDER,
        quiet_mode=True,
        # Project skills are ordinary files under .hermes/skills and are read through
        # the file toolset. Excluding the global `skills` toolset prevents unrelated
        # or mutable home-profile skills from contaminating reproducible kernel runs.
        enabled_toolsets=KERNEL_AGENT_TOOLSETS,
        session_id=f"kernelbench-{name}",
        max_iterations=MAX_ITERATIONS,
        save_trajectories=True,
    )
    disable_batch_agent_background_reviews(agent)

    try:
        mark_kernel(state, name, "running",
                    f"started {datetime.now(timezone.utc).isoformat()}")

        sf = kernel_dir / ".pipeline_state.json"

        def _read_progress() -> tuple[str, int, int, str, int, str, int, int]:
            if sf.exists():
                try:
                    with open(sf) as f:
                        current = json.load(f)
                    guided = current.get("guided_search") or {}
                    active = guided.get("active_attempt") or {}
                    return (
                        current.get("current_stage", "unknown"),
                        int(guided.get("revision", 0)),
                        int(guided.get("completed_attempts", 0)),
                        str(guided.get("status", "")),
                        int(active.get("id", 0) or 0),
                        str(active.get("phase", "")),
                        int(active.get("tool_calls_used", 0) or 0),
                        int(active.get("llm_iterations_used", 0) or 0),
                    )
                except (json.JSONDecodeError, OSError):
                    pass
            return "unknown", 0, 0, "", 0, "", 0, 0

        # Drive the pipeline forward across multiple turns. The agent often ends
        # its turn with the pipeline parked mid-stage (deliverables written but
        # verification not run). We re-enter the same conversation, feeding the
        # message history back, until the pipeline reaches "done" or stalls.
        result = {}
        history = None
        msg = prompt
        prev_progress = None
        for turn in range(1, MAX_PIPELINE_TURNS + 1):
            result = agent.run_conversation(
                user_message=msg,
                conversation_history=history,
                task_id=f"kernelbench-{name}",
            )
            history = result.get("messages")
            progress = _read_progress()
            stage = progress[0]
            log.info(
                "  %s turn %d → stage=%s guided_revision=%d attempts=%d "
                "status=%s active=%d phase=%s tools=%d llm=%d",
                name, turn, *progress)

            if stage == "done":
                break
            if progress[3].startswith("failed_"):
                log.error("  %s guided search failed: %s", name, progress[3])
                break
            # Stall guard: if the agent made no stage progress this turn (and it
            # isn't the very first optimize turn that still needs deliverables),
            # don't keep burning identical turns.
            if progress == prev_progress and turn > 1:
                log.warning(
                    "  %s stalled at stage=%s after turn %d — stopping", name,
                    stage, turn)
                break
            prev_progress = progress
            msg = CONTINUE_PROMPT

        # Fire on_session_finalize for the plugin
        try:
            from hermes_cli.plugins import invoke_hook
            invoke_hook("on_session_finalize",
                        session_id=f"kernelbench-{name}",
                        platform="python")
        except Exception:
            pass

        elapsed = time.time() - t0

        # Read the plugin's pipeline state for reporting
        pipeline_state = {}
        if sf.exists():
            with open(sf) as f:
                pipeline_state = json.load(f)

        current_stage = pipeline_state.get("current_stage", "unknown")
        complete = pipeline_state.get("deliverables_complete", False)
        missing = pipeline_state.get("deliverables_missing", [])
        verified = pipeline_state.get("verified", False)
        guided_state = pipeline_state.get("guided_search") or {}
        guided_status = str(guided_state.get("status", ""))
        turns = pipeline_state.get("turn_count", 0)

        log.info(
            "  %s done — stage=%s complete=%s verified=%s turns=%d elapsed=%.0fs",
            name,
            current_stage,
            complete,
            verified,
            turns,
            elapsed,
        )

        if guided_status.startswith("failed_"):
            failure = guided_state.get("failure_reason") or guided_status
            mark_kernel(state, name, "failed", detail=str(failure))
            return {
                "kernel": name,
                "status": "failed",
                "detail": str(failure),
                "elapsed_s": elapsed,
            }
        if current_stage == "done":
            mark_kernel(
                state,
                name,
                "done",
                detail=
                f"stage={current_stage} turns={turns} elapsed={elapsed:.0f}s")
            return {
                "kernel": name,
                "status": "done",
                "detail": f"complete in {elapsed:.0f}s",
                "elapsed_s": elapsed
            }
        elif not complete:
            mark_kernel(state,
                        name,
                        "incomplete",
                        detail=f"stage={current_stage} missing={missing}")
            return {
                "kernel": name,
                "status": "incomplete",
                "detail": f"missing: {missing}",
                "elapsed_s": elapsed
            }
        else:
            # Deliverables exist but the pipeline never reached "done" — most
            # commonly stuck at "verify" because verification was not run/passed.
            # This is NOT a success: report it as unverified so it can be retried,
            # rather than silently marking it done.
            mark_kernel(
                state,
                name,
                "unverified",
                detail=
                f"stage={current_stage} verified={verified} turns={turns}")
            return {
                "kernel": name,
                "status": "unverified",
                "detail":
                f"deliverables complete but stage={current_stage} (not verified)",
                "elapsed_s": elapsed
            }

    except Exception as e:
        elapsed = time.time() - t0
        log.error("  ✗ %s failed after %.0fs: %s", name, elapsed, e)
        mark_kernel(state, name, "failed", str(e))
        return {
            "kernel": name,
            "status": "failed",
            "detail": str(e),
            "elapsed_s": elapsed
        }


# ── Summary ─────────────────────────────────────────────────────────────────


def print_summary(results: list[dict], state: dict) -> None:
    done = [r for r in results if r["status"] == "done"]
    incomplete = [r for r in results if r["status"] == "incomplete"]
    unverified = [r for r in results if r["status"] == "unverified"]
    failed = [r for r in results if r["status"] == "failed"]

    print()
    print("=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"  Done:       {len(done)}")
    print(f"  Unverified: {len(unverified)}")
    print(f"  Incomplete: {len(incomplete)}")
    print(f"  Failed:     {len(failed)}")
    print()

    if unverified:
        print("Unverified (deliverables present but pipeline not done):")
        for r in unverified:
            print(f"  {r['kernel']}: {r['detail']}")
        print()

    if incomplete:
        print("Incomplete:")
        for r in incomplete:
            print(f"  {r['kernel']}: {r['detail']}")
        print()

    if failed:
        print("Failed:")
        for r in failed:
            print(f"  {r['kernel']}: {r['detail']}")
        print()

    counts: dict[str, int] = {}
    for info in state["kernels"].values():
        s = info["status"]
        counts[s] = counts.get(s, 0) + 1

    print("OVERALL STATE:")
    for s, c in sorted(counts.items()):
        print(f"  {s:<12} {c}")
    print(f"\nState file: {STATE_FILE}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    global DATASET_DIR, STATE_FILE, MODEL, PROVIDER, MAX_ITERATIONS, MAX_PIPELINE_TURNS
    global GUIDED_SEARCH, GUIDED_SEARCH_BUDGET, GUIDED_SEARCH_RESUME
    global GUIDED_SEARCH_SEED_RUN_ID, GUIDED_SEARCH_WALL_TIME_MINUTES

    parser = argparse.ArgumentParser(
        description="Sandboxed Triton kernel optimization pipeline")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    parser.add_argument("--hermes-home",
                        type=Path,
                        default=Path(DEFAULT_HERMES_HOME))
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--provider", type=str, default=PROVIDER)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--max-pipeline-turns",
                        type=int,
                        default=MAX_PIPELINE_TURNS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--level", type=int, choices=[1, 2])
    parser.add_argument("--kernel", type=str)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--guided-search",
        action="store_true",
        help=
        "Opt into SQL-backed Ascend MAP-Elites search before finalization.",
    )
    parser.add_argument(
        "--guided-search-budget",
        type=int,
        default=GUIDED_SEARCH_BUDGET,
        help="Completed candidate attempts per kernel (default: 20).",
    )
    parser.add_argument(
        "--guided-search-wall-time-minutes",
        type=float,
        default=None,
        help="Optional wall-time limit per kernel; no limit by default.",
    )
    parser.add_argument(
        "--resume-guided-search",
        action="store_true",
        help="Resume the workspace's incomplete guided-search run.",
    )
    parser.add_argument(
        "--seed-guided-search",
        type=int,
        default=None,
        metavar="RUN_ID",
        help="Seed a new guided-search run from a retained prior run.",
    )
    args = parser.parse_args()

    DATASET_DIR = args.dataset_dir
    STATE_FILE = args.state_file
    MODEL = args.model
    PROVIDER = args.provider
    MAX_ITERATIONS = args.max_iterations
    MAX_PIPELINE_TURNS = args.max_pipeline_turns
    GUIDED_SEARCH = bool(args.guided_search)
    GUIDED_SEARCH_BUDGET = max(1, int(args.guided_search_budget))
    GUIDED_SEARCH_WALL_TIME_MINUTES = args.guided_search_wall_time_minutes
    GUIDED_SEARCH_RESUME = bool(args.resume_guided_search)
    GUIDED_SEARCH_SEED_RUN_ID = args.seed_guided_search
    if GUIDED_SEARCH:
        MAX_PIPELINE_TURNS = max(MAX_PIPELINE_TURNS, GUIDED_SEARCH_BUDGET + 4)
    os.environ["HERMES_HOME"] = str(args.hermes_home)

    state = load_state()

    # ── Discover kernels ──────────────────────────────────────────────────
    if args.kernel:
        dirs = [DATASET_DIR / args.kernel]
        if not dirs[0].is_dir():
            log.error("Kernel directory not found: %s", dirs[0])
            sys.exit(1)
    else:
        dirs = discover_kernels(level=args.level)

    # ── Filter based on state + flags ─────────────────────────────────────
    to_process = []
    for d in dirs:
        name = d.name
        if args.force:
            to_process.append(d)
            continue
        complete, _ = kernel_is_complete(d)
        # Files present is NOT sufficient — the pipeline must have reached the
        # "done" stage (i.e. verified). A kernel parked at "verify" has all its
        # deliverable files but was never validated, so it must NOT be skipped.
        pstate_file = d / ".pipeline_state.json"
        pstage = "unknown"
        if pstate_file.exists():
            try:
                with open(pstate_file) as pf:
                    pstage = json.load(pf).get("current_stage", "unknown")
            except (json.JSONDecodeError, OSError):
                pass
        if complete and pstage == "done":
            continue
        existing = state["kernels"].get(name, {}).get("status", "pending")
        if existing == "done":
            to_process.append(d)  # state says done but files missing → re-run
        elif existing in ("failed", "incomplete",
                          "unverified") and args.retry_failed:
            to_process.append(d)
        elif existing in ("pending", "running", ""):
            to_process.append(d)
        elif existing != "done":
            to_process.append(d)

    if args.max:
        to_process = to_process[:args.max]

    log.info("Kernels to process: %d / %d total", len(to_process), len(dirs))

    if args.dry_run:
        print(f"\nDRY RUN — would process {len(to_process)} kernels:")
        for d in to_process:
            _, missing = kernel_is_complete(d)
            print(f"  {d.name}  (missing: {', '.join(missing)})")
        return

    if not to_process:
        log.info("Nothing to do — all kernels are complete.")
        print_summary([], state)
        return

    # ── Run ───────────────────────────────────────────────────────────────
    results = []
    t_start = time.time()

    if args.workers == 1:
        for kernel_dir in to_process:
            r = optimize_kernel(kernel_dir, state)
            results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(optimize_kernel, d, state): d
                for d in to_process
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    d = futures[fut]
                    log.error("Unexpected error for %s: %s", d.name, e)
                    results.append({
                        "kernel": d.name,
                        "status": "failed",
                        "detail": str(e),
                        "elapsed_s": 0.0
                    })

    total_elapsed = time.time() - t_start
    log.info("Total elapsed: %.0fs (%.1f min)", total_elapsed,
             total_elapsed / 60)
    print_summary(results, state)


if __name__ == "__main__":
    main()
