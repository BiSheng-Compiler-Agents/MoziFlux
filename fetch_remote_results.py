#!/usr/bin/env python3
"""
fetch_remote_results.py — backfill results.txt for finished kernels.

Finds kernel directories that have all deliverables but no results.txt, runs
profile_kernels.py on the remote Ascend machine, and downloads the captured
output as <kernel_dir>/results.txt.

It reuses the EXACT SSH mechanism from the remote-verify plugin (the same
_ssh_connect / _ssh_exec with transport keepalive + wall-clock timeout, plus
_sftp_upload_dir), so the connection handling and the recv_exit_status timeout
fix apply here too.

results.txt is simply the stdout of `python profile_kernels.py` with no flags
(runs both the correctness check and the benchmark table).

Usage:
    python fetch_remote_results.py                       # all pending kernels
    python fetch_remote_results.py --dry-run             # list what would run
    python fetch_remote_results.py --kernel l1_3_Batched_matrix_multiplication
    python fetch_remote_results.py --level 1             # only l1_* kernels
    python fetch_remote_results.py --max 5               # limit to N kernels
    python fetch_remote_results.py --force               # re-run even if results.txt exists
    python fetch_remote_results.py --timeout 900         # per-kernel remote timeout (s)

Env vars (same as the remote-verify plugin; loaded from ~/.hermes/.env):
    REMOTE_VERIFY_HOST / USER / PASS         (required)
    REMOTE_VERIFY_PORT                       (optional, default 22)
    REMOTE_VERIFY_BASE_DIR                   (optional, default ~/kernel_verify)
    REMOTE_VERIFY_CONDA_ENV                  (optional, default compilerclaw)
    REMOTE_VERIFY_CANN_ENV                   (optional)
    REMOTE_VERIFY_PROXY_COMMAND              (optional)
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = PROJECT_DIR / "datasets" / "KernelBench_Triton"
PLUGIN_INIT = PROJECT_DIR / ".hermes" / "plugins" / "remote-verify" / "__init__.py"
RESULTS_FILENAME = "results.txt"

# Deliverables that must all be present before we bother running remotely.
REQUIRED_MD = ["Optimizations.md", "performance_report.md", "review.md"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_remote_results")


# ── Load ~/.hermes/.env (not auto-loaded outside Hermes) ──────────────────────
def _load_dotenv() -> None:
    """Load env vars from the Hermes .env if present (does not clobber exports).

    Already-exported vars (the normal case when run from a Hermes shell) win.
    Checks $HERMES_HOME/.hermes/.env then ~/.hermes/.env.
    """
    candidates = []
    hh = os.environ.get("HERMES_HOME")
    if hh:
        candidates.append(os.path.join(hh, ".hermes", ".env"))
    candidates.append(os.path.expanduser("~/.hermes/.env"))

    for env_path in candidates:
        if not os.path.isfile(env_path):
            continue
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
        break


# ── Import the remote-verify plugin module to reuse its SSH mechanism ─────────
def _load_plugin():
    if not PLUGIN_INIT.is_file():
        log.error("remote-verify plugin not found at %s", PLUGIN_INIT)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("remote_verify_plugin",
                                                  PLUGIN_INIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Kernel discovery ──────────────────────────────────────────────────────────
def has_all_deliverables(kernel_dir: Path) -> bool:
    """opt_*.py (not opt_base_*), profile_kernels.py, and the 3 .md reports."""
    files = {f.name for f in kernel_dir.iterdir() if f.is_file()}
    has_opt = any(
        f.startswith("opt_") and f.endswith(".py")
        and not f.startswith("opt_base_") for f in files)
    has_profile = "profile_kernels.py" in files
    has_md = all(m in files for m in REQUIRED_MD)
    return has_opt and has_profile and has_md


def discover(level: int | None, kernel: str | None, force: bool) -> list[Path]:
    if not DATASET_DIR.is_dir():
        log.error("dataset dir not found: %s", DATASET_DIR)
        sys.exit(1)

    dirs = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())
    if kernel:
        dirs = [d for d in dirs if d.name == kernel]
        if not dirs:
            log.error("kernel not found: %s", kernel)
            sys.exit(1)
    if level is not None:
        dirs = [d for d in dirs if d.name.startswith(f"l{level}_")]

    pending: list[Path] = []
    for d in dirs:
        if not has_all_deliverables(d):
            continue
        if (d / RESULTS_FILENAME).exists() and not force:
            continue
        pending.append(d)
    return pending


# ── Remote run for one kernel (reuses plugin SSH functions) ──────────────────
def run_one(rv, ssh, kernel_dir: Path, timeout: int) -> dict:
    """Upload kernel_dir, run profile_kernels.py, write stdout to results.txt.

    Reuses the plugin's _ssh_exec (keepalive + wall-clock timeout) and
    _sftp_upload_dir so connection handling matches remote_verify exactly.
    """
    name = kernel_dir.name
    sftp = ssh.open_sftp()
    try:
        home = rv._resolve_home(ssh)
        base = rv._remote_base_dir().replace("~", home)
        remote_dir = f"{base}/{name}_{int(time.time())}"

        rv._ssh_exec(ssh, f"rm -rf {remote_dir}", timeout=30)
        rv._ssh_exec(ssh, f"mkdir -p {remote_dir}", timeout=30)

        log.info("  uploading %s -> %s", name, remote_dir)
        rv._sftp_upload_dir(sftp, str(kernel_dir), remote_dir)

        # Build the same conda + CANN run prefix the plugin uses.
        conda_env = rv._remote_conda_env()
        cann_env = rv._remote_cann_env().replace("~", home)
        rc, conda_path, _ = rv._ssh_exec(
            ssh,
            "command -v conda 2>/dev/null || ls ~/miniconda3/bin/conda 2>/dev/null | head -1",
            timeout=10,
        )
        conda_path = conda_path.strip()
        if not conda_path:
            return {
                "kernel": name,
                "status": "failed",
                "detail": "conda not found on remote"
            }

        run_prefix = f"source {cann_env} 2>/dev/null; {conda_path} run -n {conda_env}"

        # No flags → profile_kernels.py runs BOTH correctness + benchmark, which is
        # exactly the content of results.txt. Capture combined stdout+stderr.
        cmd = f"cd {remote_dir} && {run_prefix} python profile_kernels.py 2>&1"
        log.info("  running profile_kernels.py (timeout %ds)", timeout)
        rc, out, err = rv._ssh_exec(ssh, cmd, timeout=timeout)
        combined = (out + ("\n" + err if err else "")).strip() + "\n"

        # Write results.txt into the kernel directory.
        results_path = kernel_dir / RESULTS_FILENAME
        results_path.write_text(combined, encoding="utf-8")

        passed = rc == 0 and "FAIL" not in combined.upper()
        log.info("  wrote %s (rc=%s, %s)", results_path, rc,
                 "PASS" if passed else "non-pass")
        return {
            "kernel": name,
            "status": "done" if passed else "ran_with_failures",
            "detail": f"rc={rc}, results.txt={results_path.stat().st_size}B",
            "remote_dir": remote_dir,
        }
    finally:
        try:
            sftp.close()
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Backfill results.txt via remote Ascend run")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="List pending kernels, don't run")
    parser.add_argument("--kernel",
                        type=str,
                        help="Run a single kernel by dir name")
    parser.add_argument("--level",
                        type=int,
                        help="Only kernels matching l<level>_*")
    parser.add_argument("--max",
                        type=int,
                        default=None,
                        help="Limit to N kernels")
    parser.add_argument("--force",
                        action="store_true",
                        help="Re-run even if results.txt exists")
    parser.add_argument("--timeout",
                        type=int,
                        default=900,
                        help="Per-kernel remote timeout (s)")
    args = parser.parse_args()

    _load_dotenv()

    pending = discover(args.level, args.kernel, args.force)
    if args.max:
        pending = pending[:args.max]

    if not pending:
        log.info(
            "Nothing to do — no kernels with deliverables-but-no-results.txt.")
        return

    log.info("%d kernel(s) pending:", len(pending))
    for d in pending:
        log.info("  • %s", d.name)

    if args.dry_run:
        log.info("(dry-run) exiting without connecting.")
        return

    # Validate required env before connecting.
    rv = _load_plugin()
    if not (rv._remote_host() and rv._remote_user() and rv._remote_pass()):
        log.error(
            "Missing REMOTE_VERIFY_HOST / USER / PASS (set them in ~/.hermes/.env)."
        )
        sys.exit(1)

    log.info("connecting to %s@%s:%s ...", rv._remote_user(),
             rv._remote_host(), rv._remote_port())
    ssh = rv._ssh_connect()

    results = []
    try:
        for i, d in enumerate(pending, 1):
            log.info("[%d/%d] %s", i, len(pending), d.name)
            try:
                results.append(run_one(rv, ssh, d, args.timeout))
            except Exception as e:
                log.error("  ✗ %s failed: %s", d.name, e)
                results.append({
                    "kernel": d.name,
                    "status": "failed",
                    "detail": str(e)
                })
    finally:
        try:
            ssh.close()
        except Exception:
            pass

    # Summary
    done = [r for r in results if r["status"] == "done"]
    with_fail = [r for r in results if r["status"] == "ran_with_failures"]
    failed = [r for r in results if r["status"] == "failed"]
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  results.txt written (PASS): {len(done)}")
    print(f"  ran but non-pass output   : {len(with_fail)}")
    print(f"  failed                    : {len(failed)}")
    for r in with_fail + failed:
        print(f"    {r['kernel']}: {r['detail']}")


if __name__ == "__main__":
    main()
