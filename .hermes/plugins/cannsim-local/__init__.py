"""
cannsim-local plugin
====================
Build a Triton test binary locally, run cannsim record + report, and return
trace_core0.json for optimization analysis. No remote machine required.

CANN toolkit must be installed and CANNSIM_SETENV_PATH must point to the
CANN set_env.sh script (which puts cannsim on PATH). The plugin sources
this automatically — no hardcoded paths.

Required environment variables:
  CANNSIM_SETENV_PATH  — path to CANN set_env.sh
                           (default: /opt/miniconda3/Ascend/cann/set_env.sh)
  CONDA_BIN            — path to conda binary
                           (default: /opt/miniconda3/bin/conda)
  CONDA_ENV            — conda env name (default: compilerclaw)

cannsim is located via shutil.which("cannsim") after sourcing set_env.sh —
no CANNSIM_BIN env var, no hardcoded paths.

Important lessons (from remote plugin):
  - cannsim record flags MUST come before the -- separator:
      cannsim record -s <soc> -- <binary>
    Do NOT use `-o <dir>`: with -o, cannsim's CWD becomes the timestamped
    subdir and log_ca/ can't be found, producing a tiny instr.bin and no
    valid trace. Instead, cd into job_dir and let cannsim create its own
    `cannsim_<ts>_<bin>/` subdir there.
  - Do NOT use `-g`: A/B tested empirically — `-g` only triggers cannsim's
    auto-report. The instr.bin fed to `cannsim report` is byte-identical
    with or without `-g`. Since we always call `cannsim report` ourselves,
    `-g` is redundant.
  - cannsim.log cycle counts are not actionable. Use trace_core0.json from
    `cannsim report -e <exp_dir> -o <report_out_dir> -n 0` for optimization.
  - CANN 9.0.0 cleanup bug: cannsim record exits with code 1 after a
    successful simulation. Detected by "all tasks are finished!" +
    "current_dir = os.getcwd()" + "FileNotFoundError" + "_cleanup_user_env".
  - Default timeout is 1800s. Large GEMM kernels take ~1500s to simulate.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from glob import glob
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CANN environment
# ---------------------------------------------------------------------------

def _build_env_with_cann() -> dict[str, str]:
    """Return a copy of os.environ with CANN variables sourced from set_env.sh."""
    setenv = os.environ.get("CANNSIM_SETENV_PATH", "")
    if not setenv:
        logger.warning(
            "cannsim-local: CANNSIM_SETENV_PATH not set; "
            "cannsim may not be on PATH. Set CANNSIM_SETENV_PATH to the "
            "CANN set_env.sh path."
        )
        return dict(os.environ)
    if not os.path.isfile(setenv):
        logger.warning(f"cannsim-local: set_env.sh not found at {setenv}, using current env")
        return dict(os.environ)

    dump_cmd = f"source {setenv} && env -0"
    proc = subprocess.run(
        dump_cmd, shell=True, executable="/bin/bash",
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        logger.warning(f"cannsim-local: sourcing {setenv} failed: {proc.stderr}")
        return dict(os.environ)

    env = dict(os.environ)
    for pair in proc.stdout.split("\0"):
        if "=" in pair:
            key, _, value = pair.partition("=")
            env[key] = value
    return env


# ---------------------------------------------------------------------------
# cannsim binary resolution
# ---------------------------------------------------------------------------

def _find_cannsim_bin(env: dict[str, str] | None = None) -> str:
    """Resolve the cannsim binary path.

    After sourcing CANNSIM_SETENV_PATH (CANN set_env.sh), cannsim is on PATH.
    We only use shutil.which("cannsim") with the sourced environment PATH —
    no hardcoded paths, no CANNSIM_BIN env var fallback.
    """
    search_env = env if env is not None else os.environ
    on_path = shutil.which("cannsim", path=search_env.get("PATH", ""))
    if on_path:
        return on_path

    raise FileNotFoundError(
        "cannsim binary not found after sourcing CANN set_env.sh. "
        "Ensure CANNSIM_SETENV_PATH points to a valid set_env.sh."
    )


# ---------------------------------------------------------------------------
# Triton patches for local simulation (no physical NPU needed)
#
# Two files need patching in the triton package:
#
# 1. triton/tools/get_ascend_devices.py
#    Add env_condition so is_compile_on_910_95=True when TRITON_ASCEND_ARCH is set.
#
# 2. triton/backends/ascend/compiler.py
#    - Import get_ascend_arch_from_env from driver
#    - Use it first: get_ascend_arch_from_env() or NPUUtils().get_arch()
# ---------------------------------------------------------------------------

def _apply_local_patches(conda_env: str) -> tuple[bool, str]:
    """Apply triton patches in-place, like the remote plugin does on SSH.

    Patches the two triton files directly in the conda env's site-packages.
    Idempotent — skips if already applied.
    """
    # Find triton package location
    triton_dir = None
    try:
        import triton
        triton_dir = os.path.dirname(triton.__file__)
    except ImportError:
        # Derive conda env lib path from CONDA_BIN env var
        conda_bin = os.environ.get("CONDA_BIN", "")
        if conda_bin:
            conda_root = os.path.dirname(os.path.dirname(conda_bin))
            conda_env_lib = os.path.join(conda_root, "envs", conda_env, "lib")
            if os.path.isdir(conda_env_lib):
                for ver_dir in os.listdir(conda_env_lib):
                    candidate = os.path.join(conda_env_lib, ver_dir, "site-packages", "triton")
                    if os.path.isdir(candidate):
                        triton_dir = candidate
                        break

    if triton_dir is None:
        return False, "Could not find triton package location"

    log_parts = []

    # Patch 1: get_ascend_devices.py — add env_condition
    p1 = os.path.join(triton_dir, "tools", "get_ascend_devices.py")
    if not os.path.isfile(p1):
        log_parts.append("[PATCH] get_ascend_devices.py: file not found — skipping")
    else:
        with open(p1) as f:
            c1 = f.read()
        if "env_condition" in c1:
            log_parts.append("[PATCH] get_ascend_devices.py: already applied")
        elif "is_compile_on_910_95 = pci_condition or npu_smi_condition" in c1:
            c1 = c1.replace(
                "is_compile_on_910_95 = pci_condition or npu_smi_condition",
                "env_condition = os.getenv(\\\"TRITON_ASCEND_ARCH\\\", \\\"\\\").strip().lower() in (\\n"
                "    \\\"ascend910_9589\\\", \\\"ascend910b\\\", \\\"ascend950\\\", \\\"ascend910_95\\\"\\n"
                ")\\n"
                "is_compile_on_910_95 = pci_condition or npu_smi_condition or env_condition",
                1,
            )
            with open(p1, "w") as f:
                f.write(c1)
            log_parts.append("[PATCH] get_ascend_devices.py: applied OK")
        else:
            log_parts.append("[PATCH] get_ascend_devices.py: OLD string not found — skipping")

    # Patch 2: compiler.py — import + use get_ascend_arch_from_env
    p2 = os.path.join(triton_dir, "backends", "ascend", "compiler.py")
    if not os.path.isfile(p2):
        log_parts.append("[PATCH] compiler.py: file not found — skipping")
    else:
        with open(p2) as f:
            c2 = f.read()
        if "get_ascend_arch_from_env" in c2:
            log_parts.append("[PATCH] compiler.py: already applied")
        else:
            old_import = "from triton.backends.ascend.driver import (\n    NPUUtils\n)"
            new_import = "from triton.backends.ascend.driver import (\n    NPUUtils,\n    get_ascend_arch_from_env,\n)"
            if old_import in c2:
                c2 = c2.replace(old_import, new_import, 1)
                log_parts.append("[PATCH] compiler.py: import applied OK")
            else:
                log_parts.append("[PATCH] compiler.py: import OLD string not found")

            old_target = 'f"--target={NPUUtils().get_arch()}"'
            new_target = 'f"--target={get_ascend_arch_from_env() or NPUUtils().get_arch()}"'
            if old_target in c2:
                c2 = c2.replace(old_target, new_target, 1)
                log_parts.append("[PATCH] compiler.py: target applied OK")
            else:
                log_parts.append("[PATCH] compiler.py: target OLD string not found")

            with open(p2, "w") as f:
                f.write(c2)

    log = "\n".join(log_parts)
    all_ok = all(
        "applied OK" in p or "already applied" in p or "skipping" in p
        for p in log_parts
    )
    return all_ok, log


# ---------------------------------------------------------------------------
# CANN 9.0.0 cleanup bug detection
# ---------------------------------------------------------------------------

def _cannsim_record_completed(stdout: str, stderr: str) -> bool:
    """Return True when the simulation ran to completion despite a non-zero exit.

    CANN 9.0.0 cannsim has a Python bug in record.py _cleanup_user_env():
    after the simulation finishes it calls os.getcwd() on a directory that it
    already deleted, raising FileNotFoundError and exiting with code 1 even
    though every kernel program ran successfully.
    """
    log = stdout + "\n" + stderr
    has_completion = "all tasks are finished!" in log
    is_cleanup_bug = (
        "current_dir = os.getcwd()" in log
        and "FileNotFoundError" in log
        and "_cleanup_user_env" in log
    )
    return has_completion and is_cleanup_bug


# ---------------------------------------------------------------------------
# Core tool logic
# ---------------------------------------------------------------------------

def _run(cmd: str, cwd: str, env: dict[str, str], timeout: int) -> tuple[int, str, str]:
    """Run a shell command with timeout. Returns (rc, stdout, stderr)."""
    logger.info(f"cannsim-local: running: {cmd}")
    proc = subprocess.Popen(
        cmd, shell=True, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        logger.error(f"cannsim-local: command timed out after {timeout}s")
        return -1, stdout, stderr + f"\n[TIMEOUT after {timeout}s]"
    return proc.returncode, stdout, stderr


def _cannsim_local_run(
    local_dir: str,
    run_script: str,
    binary_name: str = "test_kernel",
    build_cmd: str | None = None,
    job_name: str | None = None,
    soc_version: str | None = None,
    cannsim_output_subdir: str = "output",
    gen_report: bool = True,
    timeout: int = 1800,
    report_timeout: int = 300,
) -> dict[str, Any]:
    """
    Build a Triton test binary locally, run cannsim record, run cannsim report,
    and return trace_core0.json content.

    Parameters
    ----------
    local_dir             : local directory containing C++ sources + run_script
    run_script            : filename of the build/run shell script inside local_dir
    binary_name           : name of the compiled test binary (default: 'test_kernel')
    build_cmd             : shell command to build the binary (default: 'bash <run_script> build')
    job_name              : job subdirectory name (default: basename + timestamp)
    soc_version           : cannsim -s value (default: CANNSIM_SOC_VERSION or 'Ascend950')
    cannsim_output_subdir : subdirectory for cannsim output (default: 'output')
    gen_report            : if True, run cannsim report to produce trace_core0.json
    timeout               : timeout for cannsim record in seconds (default: 1800)
    report_timeout        : timeout for cannsim report in seconds (default: 300)
    """
    local_dir = os.path.expanduser(local_dir)
    if not os.path.isdir(local_dir):
        return {"success": False, "error": f"local_dir does not exist: {local_dir}"}

    run_script_path = os.path.join(local_dir, run_script)
    if not os.path.isfile(run_script_path):
        return {"success": False, "error": f"run_script not found: {run_script_path}"}

    soc = soc_version or os.environ.get("CANNSIM_SOC_VERSION", "Ascend950")
    conda_env = os.environ.get("CONDA_ENV", "compilerclaw")
    conda_bin = os.environ.get("CONDA_BIN", "")

    # Build environment with CANN variables
    env = _build_env_with_cann()

    # Resolve cannsim binary using the sourced CANN PATH
    try:
        cannsim_bin = _find_cannsim_bin(env)
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}

    # Ensure conda env's bin is on PATH
    if conda_bin:
        conda_root = os.path.dirname(os.path.dirname(conda_bin))
        conda_env_bin = os.path.join(conda_root, "envs", conda_env, "bin")
    else:
        conda_env_bin = ""
    if conda_env_bin and conda_env_bin not in env.get("PATH", ""):
        env["PATH"] = conda_env_bin + ":" + env.get("PATH", "")

    # Job directory — use a temp dir to avoid polluting the source tree
    job_name = job_name or (os.path.basename(local_dir.rstrip("/")) + "_" + str(int(time.time())))
    job_dir = os.path.join(tempfile.gettempdir(), "cannsim_local", job_name)
    os.makedirs(job_dir, exist_ok=True)

    # Copy sources into job dir
    for item in os.listdir(local_dir):
        src = os.path.join(local_dir, item)
        dst = os.path.join(job_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # Make run script executable
    local_run_script = os.path.join(job_dir, run_script)
    os.chmod(local_run_script, 0o755)

    # ── Step 0: Apply triton patches ──────────────────────────────────────
    logger.info("cannsim-local: applying triton patches...")
    patch_ok, patch_log = _apply_local_patches(conda_env)
    logger.info(f"cannsim-local: patch result:\n{patch_log}")
    if not patch_ok:
        return {"success": False, "error": f"Failed to apply triton patches:\n{patch_log}"}

    # ── Step 1: Build ──────────────────────────────────────────────────────
    effective_build_cmd = build_cmd or f"bash {run_script} build"
    build_shell_cmd = (
        f"{conda_bin} run -n {conda_env} bash -c "
        f"'cd {job_dir} && {effective_build_cmd}'"
    )
    rc, build_out, build_err = _run(build_shell_cmd, job_dir, env, timeout=300)
    build_log = (build_out + "\n" + build_err).strip()
    if rc != 0:
        return {
            "success": False,
            "error": f"Build step failed (exit {rc}):\n{build_log}",
            "build_log": build_log,
            "job_dir": job_dir,
        }
    logger.info(f"cannsim-local: build OK:\n{build_log[-1000:]}")

    # ── Step 2: cannsim record ────────────────────────────────────────────
    binary_path = os.path.join(job_dir, binary_name)
    if not os.path.isfile(binary_path):
        return {
            "success": False,
            "error": f"Binary not found after build: {binary_path}",
            "build_log": build_log[-2000:],
            "job_dir": job_dir,
        }

    # Flags MUST come before the -- separator.
    # Do NOT use `-o <dir>`: causes CWD issues and broken instr.bin.
    # Do NOT use `-g`: redundant since we always call cannsim report ourselves.
    record_cmd = f"{cannsim_bin} record -s {soc} -- {binary_path}"
    rc, record_out, record_err = _run(record_cmd, job_dir, env, timeout=timeout)
    record_log = (record_out + "\n" + record_err).strip()

    if rc != 0:
        if _cannsim_record_completed(record_out, record_err):
            logger.warning(
                f"cannsim-local: cannsim record exited {rc} but simulation "
                "completed (CANN 9.0.0 cleanup bug) — continuing to report step"
            )
        else:
            return {
                "success": False,
                "error": f"cannsim record failed (exit {rc}):\n{record_log[-3000:]}",
                "job_dir": job_dir,
                "build_log": build_log[-2000:],
                "cannsim_log_tail": record_log[-4000:],
            }

    # ── Step 3: cannsim report ────────────────────────────────────────────
    trace_json_content = ""
    trace_local_path = ""
    report_log = ""
    exp_dir = ""

    if gen_report:
        # Find the most recently created cannsim_*_<binary_name> dir
        pattern = os.path.join(job_dir, f"cannsim_*_{binary_name}")
        candidates = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
        exp_dir = candidates[0] if candidates else ""

        if not exp_dir:
            logger.warning("cannsim-local: experiment dir not found; skipping report")
            report_log = "experiment dir not found"
        else:
            report_out_dir = os.path.join(exp_dir, "report")
            report_cmd = f"{cannsim_bin} report -e {exp_dir} -o {report_out_dir} -n 0"
            rc_rep, rep_out, rep_err = _run(report_cmd, exp_dir, env, timeout=report_timeout)
            report_log = (rep_out + "\n" + rep_err).strip()

            if rc_rep != 0:
                logger.warning(f"cannsim report returned non-zero: {rc_rep}\n{report_log[-1000:]}")
            else:
                # Find and read trace_core0.json
                trace_candidates = []
                for root, dirs, files in os.walk(report_out_dir):
                    for f in files:
                        if f == "trace_core0.json":
                            trace_candidates.append(os.path.join(root, f))
                for root, dirs, files in os.walk(exp_dir):
                    for f in files:
                        if f == "trace_core0.json":
                            trace_candidates.append(os.path.join(root, f))

                if trace_candidates:
                    trace_path = max(trace_candidates, key=os.path.getmtime)
                    trace_local_path = trace_path
                    with open(trace_path) as f:
                        raw = f.read()
                    trace_json_content = raw[:200_000]
                    logger.info(
                        f"cannsim-local: trace read from {trace_path} ({len(raw)} bytes)"
                    )
                else:
                    logger.warning("cannsim-local: trace_core0.json not found after report")

    return {
        "success": True,
        "job_name": job_name,
        "job_dir": job_dir,
        "experiment_dir": exp_dir,
        "patch_log": patch_log,
        "build_log": build_log[-2000:],
        "cannsim_log_tail": record_log[-4000:],
        "report_log": report_log[-2000:],
        "trace_local_path": trace_local_path,
        "trace_json": trace_json_content,
        "trace_truncated": len(trace_json_content) == 200_000,
    }


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_tool(
        name="cannsim_local_run",
        toolset="triton_ascend",
        schema={
            "name": "cannsim_local_run",
            "description": (
                "Build a Triton test binary locally, run 'cannsim record' wrapping "
                "the compiled binary, then run 'cannsim report -n 0' to produce "
                "trace_core0.json, and return its contents for optimization analysis. "
                "Requires CANN toolkit (set CANNSIM_SETENV_PATH to set_env.sh) and the "
                "conda env with triton-ascend installed (set CONDA_BIN and CONDA_ENV)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "local_dir": {
                        "type": "string",
                        "description": "Local directory containing C++ sources, CMakeLists.txt, and run_script.",
                    },
                    "run_script": {
                        "type": "string",
                        "description": "Filename of the build/run shell script inside local_dir (e.g. 'run_kernel.sh').",
                    },
                    "binary_name": {
                        "type": "string",
                        "description": (
                            "Name of the compiled test binary inside local_dir that cannsim record will wrap. "
                            "Default: 'test_kernel'. cannsim wraps THIS binary — not the run script."
                        ),
                    },
                    "build_cmd": {
                        "type": "string",
                        "description": (
                            "Shell command to build the binary, run inside the job dir with the "
                            "conda env activated. Default: 'bash <run_script> build'. "
                            "The build runs BEFORE cannsim record so compiler noise stays out of the trace."
                        ),
                    },
                    "job_name": {
                        "type": "string",
                        "description": "Name of the job subdirectory under /tmp/cannsim_local/ (default: basename + timestamp).",
                    },
                    "soc_version": {
                        "type": "string",
                        "description": "cannsim -s value (default: CANNSIM_SOC_VERSION env or 'Ascend950').",
                    },
                    "cannsim_output_subdir": {
                        "type": "string",
                        "description": "Subdirectory for cannsim output (default: 'output').",
                    },
                    "gen_report": {
                        "type": "boolean",
                        "description": (
                            "If true (default), run 'cannsim report -n 0' to produce trace_core0.json. "
                            "Always use true for optimization work — cycle counts from cannsim.log are not actionable."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout for the cannsim record step in seconds (default: 1800). Large GEMM kernels (e.g. 4096x4096) can take 1500+ seconds to simulate.",
                    },
                    "report_timeout": {
                        "type": "integer",
                        "description": "Timeout for the cannsim report step in seconds (default: 300).",
                    },
                },
                "required": ["local_dir", "run_script"],
            },
        },
        handler=lambda args, **kw: json.dumps(_cannsim_local_run(
            local_dir=args["local_dir"],
            run_script=args["run_script"],
            binary_name=args.get("binary_name", "test_kernel"),
            build_cmd=args.get("build_cmd"),
            job_name=args.get("job_name"),
            soc_version=args.get("soc_version"),
            cannsim_output_subdir=args.get("cannsim_output_subdir", "output"),
            gen_report=args.get("gen_report", True),
            timeout=args.get("timeout", 1800),
            report_timeout=args.get("report_timeout", 300),
        )),
        check_fn=lambda: bool(
            os.environ.get("CANNSIM_SETENV_PATH", "") and
            os.path.isfile(os.environ["CANNSIM_SETENV_PATH"])
        ),
        requires_env=["CANNSIM_SETENV_PATH", "CONDA_BIN", "CONDA_ENV"],
    )
