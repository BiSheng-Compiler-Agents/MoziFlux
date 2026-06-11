"""
optimize_kernels.py — Automated Triton kernel optimization pipeline

Runs the Hermes agent on every kernel in datasets/KernelBench_Triton/,
producing three files per kernel:
  1. <N>_<name>.py          — baseline (already exists, must not be modified)
  2. opt_<N>_<name>.py      — optimized Triton kernel (all shapes)
  3. profile_kernels.py     — perf_report benchmark vs torch_ref + baseline

The agent uses cannsim for trace-driven optimization and validates correctness
before writing the final files.

State is tracked in optimize_state.json so interrupted runs can be resumed.

Usage:
    # Run all pending kernels (resume-safe):
    python optimize_kernels.py

    # Dry-run: print which kernels would be processed:
    python optimize_kernels.py --dry-run

    # Process only Level-1 kernels:
    python optimize_kernels.py --level 1

    # Process only Level-2 kernels:
    python optimize_kernels.py --level 2

    # Process a single kernel by directory name:
    python optimize_kernels.py --kernel l2_1_Conv2D_ReLU_BiasAdd

    # Re-run failed kernels (skip completed ones):
    python optimize_kernels.py --retry-failed

    # Force re-run everything (ignore state):
    python optimize_kernels.py --force

    # Limit number of kernels to process in this run:
    python optimize_kernels.py --max 10

    # Control parallelism (default: 1 — sequential):
    python optimize_kernels.py --workers 2
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure HERMES_HOME is set for AIAgent .env discovery ──────────────────────
# run_agent.py loads .env from <HERMES_HOME>/.env at import time.
# Without this, it falls back to ~/.hermes/ which may not exist.
_HERMES_HOME = os.environ.get("HERMES_HOME", "/opt/data")
os.environ.setdefault("HERMES_HOME", _HERMES_HOME)

# ── CRITICAL: cd to project root so project plugins (.hermes/plugins/) ────
# Plugin discovery uses Path.cwd() to find .hermes/plugins/. Without this,
# cannsim_local_run and other project plugins won't be visible to the agent.
_PROJECT_DIR = Path(__file__).parent.resolve()
os.chdir(_PROJECT_DIR)
sys.path.insert(0, str(_PROJECT_DIR))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimize_kernels")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATASET_DIR = ROOT / "datasets" / "KernelBench_Triton"
STATE_FILE  = ROOT / "optimize_state.json"

# ── Agent config ───────────────────────────────────────────────────────────────
MODEL    = "deepseek/deepseek-v4-flash"
PROVIDER = "openrouter"

# ── Per-kernel task prompt ─────────────────────────────────────────────────────
TASK_PROMPT = """
Optimize the Triton kernel in the directory: {kernel_dir}

The baseline kernel file that you should start optimizing is: {baseline_file}

Use cannsim_local_run to simulate the kernel and get trace data.
Write the final optimized kernel in opt_{baseline_filename} in the same directory.

Required deliverables (all 5 must be produced):
1. opt_{baseline_filename}       — Optimized kernel + ModelNew host interface
2. profile_kernels.py             — @perf_report benchmark, all dispatch paths, unit test
3. Optimizations.md               — Each optimization applied, with code snippets and rationale
4. performance_report.md          — cannsim trace tables (baseline vs optimized), hardware latency
5. review.md                      — Static P0/P1/P2 review of the optimized kernel

Do not modify the baseline file.
"""

# ── State management ───────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load optimization state from JSON, or return empty state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"kernels": {}}


def save_state(state: dict) -> None:
    """Atomically save state to JSON."""
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)


def mark_kernel(state: dict, kernel_name: str, status: str,
                detail: str = "", files: list = None) -> None:
    """Update a kernel's status in the state dict and save."""
    state["kernels"][kernel_name] = {
        "status": status,
        "detail": detail,
        "files": files or [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_state(state)


# ── Kernel discovery ───────────────────────────────────────────────────────────

def discover_kernels(level: int = None) -> list[Path]:
    """Return sorted list of kernel directories, optionally filtered by level."""
    dirs = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())
    if level is not None:
        dirs = [d for d in dirs if d.name.startswith(f"l{level}_")]
    return dirs


def get_baseline_file(kernel_dir: Path) -> Path | None:
    """Find the baseline .py file (not opt_*, not profile_*)."""
    candidates = [
        f for f in kernel_dir.glob("*.py")
        if not f.name.startswith("opt_")
        and "profile" not in f.name.lower()
    ]
    return candidates[0] if len(candidates) == 1 else None


def kernel_is_complete(kernel_dir: Path) -> tuple[bool, list[str]]:
    """
    Coarse pre-run completeness check used by the discovery filter.
    Returns True only when the two must-have files (opt_*.py and
    profile_kernels.py) are present. For the strict 5-deliverable check
    used after a run, use verify_deliverables() instead.
    """
    files    = [f.name for f in kernel_dir.glob("*.py")]
    missing  = []

    has_opt     = any(f.startswith("opt_") for f in files)
    has_profile = any("profile" in f.lower() for f in files)

    if not has_opt:
        missing.append("opt_*.py")
    if not has_profile:
        missing.append("profile_kernels.py")

    return len(missing) == 0, missing


# ── Required deliverables (per triton-operator/orchestration SKILL.md) ─────
# Every optimized kernel directory must contain all 5 of these. If any are
# missing after a run, the orchestrator sends a follow-up prompt asking the
# agent to produce them (no re-optimization of the kernel itself).
REQUIRED_DELIVERABLES = (
    "opt_*.py",                 # Optimized kernel + ModelNew host interface
    "profile_kernels.py",       # @perf_report benchmark, all dispatch paths, unit test
    "Optimizations.md",         # Each optimization applied, with code snippets and rationale
    "performance_report.md",    # cannsim trace tables (baseline vs optimized)
    "review.md",                # Static P0/P1/P2 review of the optimized kernel
)

# Max number of follow-up prompts the orchestrator will send asking the
# agent to deliver missing files. Bounds the retry loop in optimize_kernel().
MAX_DELIVERY_FOLLOWUPS = 2


def verify_deliverables(kernel_dir: Path) -> tuple[bool, list[str], str | None]:
    """
    Strict post-run check for all 5 required deliverables in kernel_dir.

    Returns (is_complete, missing_filenames, followup_message_for_agent):
      - is_complete=True  → every deliverable present, followup_message is None
      - is_complete=False → missing_filenames lists the absent items using
        their canonical pattern or filename, and followup_message is a
        self-contained prompt that can be fed back to the same Hermes
        agent as a follow-up user_message, asking it to produce ONLY the
        missing files (no re-optimization).

    The follow-up message tells the agent to inspect kernel_dir first, see
    what is already on disk, and write only the missing artifacts — so
    work the previous run already completed is preserved.
    """
    py_files = {f.name for f in kernel_dir.glob("*.py")}
    md_files = {f.name for f in kernel_dir.glob("*.md")}

    missing: list[str] = []

    # 1. opt_<baseline_filename>.py — at least one opt_*.py must exist
    if not any(f.startswith("opt_") for f in py_files):
        missing.append("opt_*.py")

    # 2. profile_kernels.py — exact filename
    if "profile_kernels.py" not in py_files:
        missing.append("profile_kernels.py")

    # 3-5. Markdown reports — exact filenames
    for md in ("Optimizations.md", "performance_report.md", "review.md"):
        if md not in md_files:
            missing.append(md)

    if not missing:
        return True, [], None

    baseline       = get_baseline_file(kernel_dir)
    baseline_name  = baseline.name if baseline else "<baseline>.py"
    baseline_str   = str(baseline) if baseline else "(not found)"

    followup = (
        f"Your previous run for kernel '{kernel_dir.name}' did NOT deliver "
        f"all required files. {len(missing)} of {len(REQUIRED_DELIVERABLES)} "
        f"deliverables are missing.\n\n"
        f"  Kernel directory: {kernel_dir}\n"
        f"  Baseline file:    {baseline_str}  (do NOT modify it)\n\n"
        f"MISSING:\n"
        + "\n".join(f"  - {m}" for m in missing)
        + "\n\n"
        f"REQUIRED DELIVERABLES (per the triton-operator orchestration skill):\n"
        f"  1. opt_{baseline_name}    — Optimized kernel + ModelNew host interface\n"
        f"  2. profile_kernels.py      — @perf_report benchmark, all dispatch paths, unit test\n"
        f"  3. Optimizations.md        — Each optimization applied, with code snippets and rationale\n"
        f"  4. performance_report.md   — cannsim trace tables (baseline vs optimized), hardware latency\n"
        f"  5. review.md               — Static P0/P1/P2 review of the optimized kernel\n\n"
        f"DO NOT re-run the optimization workflow. The kernel is already "
        f"optimized in the existing opt_*.py and profile_kernels.py. Start "
        f"by listing {kernel_dir} to see what is already on disk, then write "
        f"ONLY the missing files. Each missing file must be self-contained "
        f"and reference the existing artifacts in {kernel_dir}."
    )
    return False, missing, followup


# ── Per-kernel optimization via Hermes agent ───────────────────────────────────

def optimize_kernel(kernel_dir: Path, state: dict) -> dict:
    """
    Run the Hermes agent on a single kernel directory.

    Returns a result dict:
      { "kernel": name, "status": "done"|"failed"|"skipped",
        "detail": str, "files": [str], "elapsed_s": float }
    """
    from run_agent import AIAgent

    name = kernel_dir.name
    log.info("▶ Starting %s", name)
    t0 = time.time()

    # Check if already complete on disk (regardless of state file)
    complete, missing = kernel_is_complete(kernel_dir)
    if complete:
        log.info("  ✓ %s already complete (skipping)", name)
        mark_kernel(state, name, "done", "already complete on disk",
                    files=[f.name for f in kernel_dir.glob("*.py")])
        return {"kernel": name, "status": "skipped",
                "detail": "already complete", "files": [], "elapsed_s": 0.0}

    baseline = get_baseline_file(kernel_dir)
    if baseline is None:
        log.warning("  ✗ %s — cannot find unique baseline file", name)
        mark_kernel(state, name, "failed", "no unique baseline file found")
        return {"kernel": name, "status": "failed",
                "detail": "no unique baseline file found", "files": [],
                "elapsed_s": time.time() - t0}

    prompt = TASK_PROMPT.format(
        kernel_dir=str(kernel_dir),
        baseline_file=str(baseline),
        baseline_filename=baseline.name,
    )

    agent = AIAgent(
        model=MODEL,
        provider=PROVIDER,
        quiet_mode=True,
        enabled_toolsets=["hermes-cli", "triton_ascend"],
        # Each kernel gets its own isolated task_id so tool calls,
        # working directories and sessions don't bleed across kernels.
        session_id=f"kernelbench-{name}",
        max_iterations=90, 
        save_trajectories=True
    )

    try:
        mark_kernel(state, name, "running", f"started {datetime.now(timezone.utc).isoformat()}")
        result = agent.run_conversation(
            user_message=prompt,
            task_id=f"kernelbench-{name}",
        )

        # Fire phoenix-tracer on_session_finalize to close root span and flush.
        # conversation_loop.py fires on_session_end but never on_session_finalize,
        # so the root hermes.session span would stay open forever without this.
        try:
            from hermes_cli.plugins import invoke_hook
            invoke_hook("on_session_finalize", session_id=f"kernelbench-{name}", platform="python")
        except Exception:
            pass

        # Verify all 5 required deliverables. If any are missing, send up
        # to MAX_DELIVERY_FOLLOWUPS follow-up prompts asking the agent to
        # produce ONLY the missing files (no re-optimization of the kernel).
        for followup_idx in range(1, MAX_DELIVERY_FOLLOWUPS + 2):  # 1..N+1
            complete_after, missing_after, followup = verify_deliverables(kernel_dir)
            if complete_after:
                break
            if followup_idx > MAX_DELIVERY_FOLLOWUPS:
                log.error(
                    "  ✗ %s still missing %d deliverable(s) after %d follow-ups: %s",
                    name, len(missing_after), MAX_DELIVERY_FOLLOWUPS, missing_after,
                )
                break
            log.warning(
                "  ⚠ %s missing %d deliverable(s) — followup %d/%d: %s",
                name, len(missing_after), followup_idx, MAX_DELIVERY_FOLLOWUPS, missing_after,
            )
            try:
                agent.run_conversation(
                    user_message=followup,
                    task_id=f"kernelbench-{name}-followup{followup_idx}",
                )
            except Exception as e:
                log.error("  ✗ %s followup %d failed: %s", name, followup_idx, e)
                break

        elapsed = time.time() - t0

        # Re-verify after the follow-up loop (the agent may have written
        # only some of the missing files, in which case we still report
        # incomplete rather than done).
        complete_after, missing_after, _ = verify_deliverables(kernel_dir)
        all_files = sorted(f.name for f in kernel_dir.iterdir() if f.is_file())

        if complete_after:
            log.info("  ✓ %s done (%.0fs, %d files)", name, elapsed, len(all_files))
            mark_kernel(state, name, "done",
                        detail=f"elapsed {elapsed:.0f}s",
                        files=all_files)
            return {"kernel": name, "status": "done",
                    "detail": f"elapsed {elapsed:.0f}s",
                    "files": all_files, "elapsed_s": elapsed}
        else:
            detail = f"agent finished but missing: {missing_after}"
            log.warning("  ⚠ %s incomplete — %s", name, detail)
            mark_kernel(state, name, "incomplete", detail,
                        files=all_files)
            return {"kernel": name, "status": "incomplete",
                    "detail": detail, "files": all_files, "elapsed_s": elapsed}

    except Exception as e:
        elapsed = time.time() - t0
        log.error("  ✗ %s failed after %.0fs: %s", name, elapsed, e)
        mark_kernel(state, name, "failed", str(e))
        return {"kernel": name, "status": "failed",
                "detail": str(e), "files": [], "elapsed_s": elapsed}


# ── Summary printing ───────────────────────────────────────────────────────────

def print_summary(results: list[dict], state: dict) -> None:
    """Print a final run summary and overall state tallies."""
    done       = [r for r in results if r["status"] == "done"]
    skipped    = [r for r in results if r["status"] == "skipped"]
    incomplete = [r for r in results if r["status"] == "incomplete"]
    failed     = [r for r in results if r["status"] == "failed"]

    print()
    print("=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"  Done:       {len(done)}")
    print(f"  Skipped:    {len(skipped)}")
    print(f"  Incomplete: {len(incomplete)}")
    print(f"  Failed:     {len(failed)}")
    print()

    if incomplete:
        print("Incomplete (missing files):")
        for r in incomplete:
            print(f"  {r['kernel']}: {r['detail']}")
        print()

    if failed:
        print("Failed:")
        for r in failed:
            print(f"  {r['kernel']}: {r['detail']}")
        print()

    # Overall state tally
    counts: dict[str, int] = {}
    for info in state["kernels"].values():
        s = info["status"]
        counts[s] = counts.get(s, 0) + 1

    print("OVERALL STATE (all kernels ever attempted):")
    for s, c in sorted(counts.items()):
        print(f"  {s:<12} {c}")
    print(f"\nState file: {STATE_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automated Triton kernel optimization pipeline via Hermes agent"
    )
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print which kernels would be processed and exit")
    parser.add_argument("--level",         type=int, choices=[1, 2],
                        help="Process only Level-1 or Level-2 kernels")
    parser.add_argument("--kernel",        type=str,
                        help="Process a single kernel by directory name")
    parser.add_argument("--retry-failed",  action="store_true",
                        help="Re-run kernels marked failed or incomplete in state")
    parser.add_argument("--force",         action="store_true",
                        help="Re-run all kernels, ignoring state")
    parser.add_argument("--max",           type=int, default=None,
                        help="Maximum number of kernels to process in this run")
    parser.add_argument("--workers",       type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    state = load_state()

    # ── Discover kernels ──────────────────────────────────────────────────────
    if args.kernel:
        dirs = [DATASET_DIR / args.kernel]
        if not dirs[0].is_dir():
            log.error("Kernel directory not found: %s", dirs[0])
            sys.exit(1)
    else:
        dirs = discover_kernels(level=args.level)

    # ── Filter based on state + flags ─────────────────────────────────────────
    to_process = []
    for d in dirs:
        name = d.name
        # --force: include everything
        if args.force:
            to_process.append(d)
            continue
        # Already done on disk → always skip
        complete, _ = kernel_is_complete(d)
        if complete:
            continue
        # Check state
        existing = state["kernels"].get(name, {})
        existing_status = existing.get("status", "pending")
        if existing_status == "done":
            # State says done but files are missing → re-run
            to_process.append(d)
        elif existing_status in ("failed", "incomplete") and args.retry_failed:
            to_process.append(d)
        elif existing_status in ("pending", "running", ""):
            to_process.append(d)
        elif existing_status not in ("done",):
            # running/unknown → include by default
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

    # ── Run ───────────────────────────────────────────────────────────────────
    results = []
    t_start = time.time()

    if args.workers == 1:
        # Sequential — simplest, avoids any thread-safety concerns with AIAgent
        for kernel_dir in to_process:
            r = optimize_kernel(kernel_dir, state)
            results.append(r)
    else:
        # Parallel — each worker creates its own AIAgent instance
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(optimize_kernel, d, state): d
                for d in to_process
            }
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    results.append(r)
                except Exception as e:
                    d = futures[fut]
                    log.error("Unexpected error for %s: %s", d.name, e)
                    results.append({"kernel": d.name, "status": "failed",
                                    "detail": str(e), "files": [],
                                    "elapsed_s": 0.0})

    total_elapsed = time.time() - t_start
    log.info("Total elapsed: %.0fs (%.1f min)", total_elapsed, total_elapsed / 60)
    print_summary(results, state)


if __name__ == "__main__":
    main()
