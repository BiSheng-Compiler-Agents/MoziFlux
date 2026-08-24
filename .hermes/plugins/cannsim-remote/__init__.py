"""
cannsim_remote plugin
=====================
Transfers a local directory (containing a compiled Triton npubin + C++ host
source files + run script) to a remote machine via SFTP, patches the remote
triton env so it works without a physical NPU, then:

  1. Builds the test binary on the remote (separate step, outside cannsim scope)
  2. Runs  cannsim record -g -o <output> -s <soc> -- <binary>
  3. Runs  cannsim report -n 0 --timeline  to produce trace_core0.json
  4. Downloads trace_core0.json back to a local temp file
  5. Returns the trace JSON content + log tail

Required env vars:
  CANNSIM_REMOTE_HOST  — SSH hostname / IP of the remote machine
  CANNSIM_REMOTE_USER  — SSH username
  CANNSIM_REMOTE_PASS  — SSH password

Optional env vars (with defaults):
  CANNSIM_REMOTE_PORT      — SSH port (default: 22)
  CANNSIM_REMOTE_BASE_DIR  — base dir on remote (default: ~/cannsim_jobs)
  CANNSIM_REMOTE_CONDA_ENV — conda env name on remote (default: compilerclaw)
  CANNSIM_SOC_VERSION      — cannsim -s value (default: Ascend950)
  CANNSIM_SETENV_PATH      — path to setenv.bash on remote
                             (default: ~/miniconda3/Ascend/cann/bin/setenv.bash)

Important lessons:
  - Upload C++ SOURCES (.cpp + CMakeLists.txt), not pre-built binaries.
    A binary built locally with GCC 13 will fail on a remote with GCC 11
    (GLIBCXX_3.4.32 not found). Let the build step compile on the remote.
  - SFTP does not preserve execute permissions. Plugin chmod +x all files.
  - conda is not on PATH in non-interactive SSH. Plugin resolves it by path.
  - base_dir ~ is not expanded by SFTP. Plugin resolves $HOME via SSH first.
  - Job dir reuse causes stale files from failed runs. Wipe before uploading.
  - cannsim record flags MUST come before the -- separator:
      cannsim record -g -o <dir> -s <soc> -- <binary>
    The run script must NOT be passed directly as the binary argument —
    build first (outside cannsim scope), then point cannsim at the binary.
  - cannsim.log cycle counts are not actionable. Use trace_core0.json from
    `cannsim report -n 0 --timeline` for optimization decisions.
  - CANN 9.0.0 cleanup bug: cannsim record exits with code 1 after a successful
    simulation due to a FileNotFoundError in _cleanup_user_env(os.getcwd()).
    The plugin detects this pattern and continues to the report step rather than
    treating it as a failure. Signature: "current_dir = os.getcwd()" +
    "FileNotFoundError" + "_cleanup_user_env" in stderr, combined with
    "all tasks are finished!" in stdout confirming the simulation completed.
  - Default timeout is 1800s. Large GEMM kernels (4096×4096) take ~1500s to
    simulate. The old 600s default would kill mid-simulation.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import tempfile
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SFTP helpers
# ---------------------------------------------------------------------------


def _sftp_upload_dir(sftp, local_dir: str, remote_dir: str) -> None:
    """Recursively upload local_dir to remote_dir via sftp."""
    for item in os.listdir(local_dir):
        local_path = os.path.join(local_dir, item)
        remote_path = remote_dir.rstrip("/") + "/" + item
        if os.path.isdir(local_path):
            try:
                sftp.mkdir(remote_path)
            except OSError:
                pass  # already exists
            _sftp_upload_dir(sftp, local_path, remote_path)
        else:
            sftp.put(local_path, remote_path)


def _ssh_exec(ssh, command: str, timeout: int = 120) -> tuple[int, str, str]:
    """Run a command over SSH and return (exit_code, stdout, stderr)."""
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, stdout.read().decode(), stderr.read().decode()


# ---------------------------------------------------------------------------
# Remote patch script
#
# Two files need patching in the remote triton `compilerclaw` env:
#
# 1. triton/tools/get_ascend_devices.py
#    Add env_condition so is_compile_on_910_95=True when TRITON_ASCEND_ARCH is set,
#    routing compilation through linalg_to_bin_enable_npu_compile_910_95 (not A2_A3).
#
# 2. triton/backends/ascend/compiler.py
#    - Import get_ascend_arch_from_env from driver
#    - Use it first: get_ascend_arch_from_env() or NPUUtils().get_arch()
#      so rtGetSocVersion is never called without a physical NPU.
# ---------------------------------------------------------------------------

# Inline Python patch script — runs on the remote via ssh exec
_PATCH_SCRIPT = r'''
import sys, re

def patch_file(path, old, new, label):
    import os
    if not os.path.exists(path):
        print(f"[PATCH] {label}: file not found ({path}) — skipping (env may be pre-patched)")
        return
    with open(path) as f:
        content = f.read()
    if new.strip() in content:
        print(f"[PATCH] {label}: already applied, skipping")
        return
    if old.strip() not in content:
        print(f"[PATCH] {label}: OLD string not found — may be a different version, skipping", file=sys.stderr)
        return
    patched = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(patched)
    print(f"[PATCH] {label}: applied OK")

import triton, os
triton_dir = os.path.dirname(triton.__file__)

# ── Patch 1: get_ascend_devices.py ──────────────────────────────────────────
p1 = os.path.join(triton_dir, "tools", "get_ascend_devices.py")
patch_file(
    p1,
    old="is_compile_on_910_95 = pci_condition or npu_smi_condition",
    new=(
        "env_condition = os.getenv(\"TRITON_ASCEND_ARCH\", \"\").strip().lower() in (\n"
        "    \"ascend910_9589\", \"ascend910b\", \"ascend950\", \"ascend910_95\"\n"
        ")\n"
        "is_compile_on_910_95 = pci_condition or npu_smi_condition or env_condition"
    ),
    label="get_ascend_devices.py: env_condition",
)

# ── Patch 2a: compiler.py — import get_ascend_arch_from_env ─────────────────
p2 = os.path.join(triton_dir, "backends", "ascend", "compiler.py")
patch_file(
    p2,
    old="from triton.backends.ascend.driver import (\n    NPUUtils\n)",
    new="from triton.backends.ascend.driver import (\n    NPUUtils,\n    get_ascend_arch_from_env,\n)",
    label="compiler.py: import get_ascend_arch_from_env",
)

# ── Patch 2b: compiler.py — use env arch first ───────────────────────────────
patch_file(
    p2,
    old='f"--target={NPUUtils().get_arch()}"',
    new='f"--target={get_ascend_arch_from_env() or NPUUtils().get_arch()}"',
    label="compiler.py: get_ascend_arch_from_env() or get_arch()",
)

print("[PATCH] All patches checked.")
'''


def _apply_remote_patches(ssh, conda_env: str) -> tuple[bool, str]:
    """Apply the three triton patches in the remote conda env."""
    script_path = "/tmp/_hermes_triton_patch.py"
    write_cmd = f"cat > {script_path} << 'HEREDOC'\n{_PATCH_SCRIPT}\nHEREDOC"
    _ssh_exec(ssh, write_cmd, timeout=10)

    # Find conda — try common locations for non-interactive SSH sessions
    find_conda = (
        "CONDA=$(command -v conda 2>/dev/null || "
        "ls ~/miniconda3/bin/conda ~/anaconda3/bin/conda /opt/conda/bin/conda 2>/dev/null | head -1) && "
        "$CONDA run -n {env} python3 {script}").format(env=conda_env,
                                                       script=script_path)

    rc, out, err = _ssh_exec(ssh, find_conda, timeout=30)
    log = (out + "\n" + err).strip()
    return rc == 0, log


# ---------------------------------------------------------------------------
# Core tool logic
# ---------------------------------------------------------------------------


def _cannsim_record_completed(log: str) -> bool:
    """Return True when the simulation ran to completion despite a non-zero exit.

    CANN 9.0.0 cannsim has a Python bug in record.py _cleanup_user_env():
    after the simulation finishes it calls os.getcwd() on a directory that it
    already deleted, raising FileNotFoundError and exiting with code 1 even
    though every kernel program ran successfully and instr.bin / log_ca are
    fully written.  We detect this by confirming:
      1. The kernel actually ran:  "all tasks are finished!" appears in the log
      2. The failure is only the known cleanup bug:
             current_dir = os.getcwd()
             FileNotFoundError
    """
    has_completion = "all tasks are finished!" in log
    is_cleanup_bug = ("current_dir = os.getcwd()" in log
                      and "FileNotFoundError" in log
                      and "_cleanup_user_env" in log)
    return has_completion and is_cleanup_bug


def _ssh_cannsim_record(
    ssh,
    setenv: str,
    remote_job_dir: str,
    remote_binary: str,
    soc: str,
    timeout: int,
) -> tuple[int, str, str]:
    """Run cannsim record on remote via SSH, with early-kill on trace completion.

    CANN runtime atexit handlers hang the binary after the simulation completes
    (see pitfall #23 — simulation/SKILL.md).  This wrapper runs a bash script
    on the remote that:
      1. Launches cannsim record in the background
      2. Polls cannsim.log every 3s for completion markers
      3. Kills the record process as soon as the trace is complete (``Result
         copied back`` or ``all tasks are finished!``), avoiding the atexit
         teardown hang
      4. Returns exit code 0 on early-kill, 1 on timeout

    Returns the same shape as _ssh_exec: (exit_code, stdout, stderr).
    """
    wrapper_script = (
        'set -o pipefail\n'
        f'cd {remote_job_dir} || exit 1\n'
        f'cannsim record -s {soc} -- {remote_binary} 2>&1 &\n'
        'RECORD_PID=$!\n'
        'trap "kill -9 $RECORD_PID 2>/dev/null; exit 0" EXIT\n'
        'POLL_LOG="cannsim.log"\n'
        'START_SEC=$SECONDS\n'
        'while kill -0 $RECORD_PID 2>/dev/null; do\n'
        '  ELAPSED=$((SECONDS - START_SEC))\n'
        '  if [ $ELAPSED -gt MAX_SEC ]; then\n'
        '    echo "[WRAPPER] Timeout after ${ELAPSED}s"\n'
        '    kill -9 $RECORD_PID 2>/dev/null\n'
        '    exit 1\n'
        '  fi\n'
        '  if [ -f "$POLL_LOG" ]; then\n'
        '    if grep -q "Result copied back" "$POLL_LOG" 2>/dev/null || '
        'grep -q "all tasks are finished" "$POLL_LOG" 2>/dev/null; then\n'
        # CRITICAL: the completion marker fires BEFORE cannsim flushes instr.bin
        # to disk (the write happens during the atexit/teardown phase the kill
        # skips). Killing on the marker + a blind sleep races that write and, for
        # teardown-hanging kernels (al.multibuffer), usually wins — leaving an
        # experiment dir with no instr.bin so `cannsim report` fails with
        # "instr log file is not found". Wait for a non-empty, size-stable
        # instr.bin in the newest experiment subdir before killing.
        '      echo "[WRAPPER] Trace marker seen; waiting for instr.bin to stabilize"\n'
        '      LAST=-1\n'
        '      for i in $(seq 1 40); do\n'
        '        EXP=$(ls -1dt cannsim_*/ 2>/dev/null | head -1)\n'
        '        SZ=$(stat -c%s "${EXP}instr.bin" 2>/dev/null || echo 0)\n'
        '        if [ "$SZ" -gt 0 ] && [ "$SZ" = "$LAST" ]; then\n'
        '          echo "[WRAPPER] instr.bin stable at ${SZ} bytes"\n'
        '          break\n'
        '        fi\n'
        '        LAST=$SZ\n'
        '        sleep 3\n'
        '      done\n'
        '      echo "[WRAPPER] Trace complete, killing record process"\n'
        '      kill -9 $RECORD_PID 2>/dev/null\n'
        '      exit 0\n'
        '    fi\n'
        '  fi\n'
        '  sleep 3\n'
        'done\n'
        'wait $RECORD_PID 2>/dev/null\n'
        'EXIT_CODE=${?:-0}\n'
        'echo "[WRAPPER] cannsim record exited naturally with code $EXIT_CODE"\n'
        'exit $EXIT_CODE\n').replace("MAX_SEC", str(timeout))

    full_cmd = f"source {setenv} && bash -c {shlex.quote(wrapper_script)}"
    return _ssh_exec(ssh, full_cmd, timeout=timeout + 30)


def _cannsim_remote_run(
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
    Upload local_dir to the remote machine, build the test binary there,
    run cannsim record, run cannsim report, and return trace_core0.json content.

    Parameters
    ----------
    local_dir             : local directory containing the C++ sources + run_script
    run_script            : filename of the build/run shell script inside local_dir
    binary_name           : name of the compiled test binary (default: 'test_kernel')
                            This binary is what cannsim record wraps — NOT the run script.
    build_cmd             : shell command to build the binary on remote, run inside
                            the job dir (default: 'bash <run_script> build').
                            The command runs with the job dir as cwd.
    job_name              : remote subdirectory name (default: basename of local_dir + timestamp)
    soc_version           : cannsim -s value (default: CANNSIM_SOC_VERSION env or 'Ascend950')
    cannsim_output_subdir : subdirectory inside job dir for cannsim output (default: 'output')
    gen_report            : if True (default), pass -g to cannsim record and then run
                            cannsim report to produce trace_core0.json
    timeout               : SSH timeout for the cannsim record command (default: 600s)
    report_timeout        : SSH timeout for the cannsim report command (default: 300s)
    """
    try:
        import paramiko
    except ImportError:
        return {
            "success": False,
            "error": "paramiko is not installed. Run: pip install paramiko"
        }

    host = os.environ.get("CANNSIM_REMOTE_HOST", "")
    user = os.environ.get("CANNSIM_REMOTE_USER", "")
    password = os.environ.get("CANNSIM_REMOTE_PASS", "")
    port = int(os.environ.get("CANNSIM_REMOTE_PORT", "22"))
    base_dir = os.environ.get("CANNSIM_REMOTE_BASE_DIR", "~/cannsim_jobs")
    conda_env = os.environ.get("CANNSIM_REMOTE_CONDA_ENV", "compilerclaw")
    soc = soc_version or os.environ.get("CANNSIM_SOC_VERSION", "Ascend950")
    setenv = os.environ.get(
        "CANNSIM_SETENV_PATH",
        "~/miniconda3/Ascend/cann/bin/setenv.bash",
    )

    if not host or not user or not password:
        return {
            "success":
            False,
            "error":
            "Missing required env vars: CANNSIM_REMOTE_HOST, CANNSIM_REMOTE_USER, CANNSIM_REMOTE_PASS",
        }

    local_dir = os.path.expanduser(local_dir)
    if not os.path.isdir(local_dir):
        return {
            "success": False,
            "error": f"local_dir does not exist: {local_dir}"
        }

    run_script_path = os.path.join(local_dir, run_script)
    if not os.path.isfile(run_script_path):
        return {
            "success": False,
            "error": f"run_script not found in local_dir: {run_script_path}"
        }

    job_name = job_name or (os.path.basename(local_dir.rstrip("/")) + "_" +
                            str(int(time.time())))

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        logger.info(f"cannsim_remote: connecting to {user}@{host}:{port}")
        ssh.connect(host, port=port, username=user, password=password)

        # 1. Resolve remote $HOME (SFTP cannot expand ~)
        _, home_out, _ = _ssh_exec(ssh, "echo $HOME", timeout=10)
        remote_home = home_out.strip()
        base_dir_abs = base_dir.replace("~", remote_home)
        remote_job_dir = base_dir_abs.rstrip("/") + "/" + job_name

        # 2. Wipe + recreate job dir so no stale binaries remain
        _ssh_exec(ssh, f"rm -rf {remote_job_dir}", timeout=30)
        rc, out, err = _ssh_exec(ssh, f"mkdir -p {remote_job_dir}", timeout=30)
        if rc != 0:
            return {
                "success": False,
                "error": f"Failed to create remote dir: {err}"
            }

        # 3. Upload local_dir contents via SFTP
        logger.info(
            f"cannsim_remote: uploading {local_dir} → {remote_job_dir}")
        sftp = ssh.open_sftp()
        try:
            sftp.chdir(remote_job_dir)
            _sftp_upload_dir(sftp, local_dir, remote_job_dir)
        finally:
            sftp.close()

        # 4. chmod +x everything (SFTP strips execute bits)
        remote_run_script = remote_job_dir.rstrip("/") + "/" + run_script
        _ssh_exec(ssh, f"chmod +x {remote_job_dir}/*", timeout=10)

        # 5. Apply triton patches on remote
        logger.info("cannsim_remote: applying triton patches on remote...")
        patch_ok, patch_log = _apply_remote_patches(ssh, conda_env)
        logger.info(f"cannsim_remote: patch result:\n{patch_log}")
        if not patch_ok:
            return {
                "success": False,
                "error": f"Failed to apply remote patches:\n{patch_log}"
            }

        # 6. BUILD the test binary on remote (separate from cannsim scope)
        #    Default: bash <run_script> build
        #    This keeps compiler noise out of the cannsim trace.
        effective_build_cmd = build_cmd or f"bash {remote_run_script} build"
        # Use conda run to avoid needing shell functions (conda activate requires sourcing)
        # Find conda binary first, then use 'conda run -n env bash -c "..."'
        build_full_cmd = (
            f"source {setenv} && "
            f"CONDA_BIN=$(command -v conda 2>/dev/null || "
            f"ls ~/miniconda3/bin/conda ~/anaconda3/bin/conda /opt/conda/bin/conda 2>/dev/null | head -1) && "
            f"$CONDA_BIN run -n {conda_env} bash -c "
            f"'source {setenv} && cd {remote_job_dir} && {effective_build_cmd}'"
        )
        logger.info(f"cannsim_remote: building binary: {build_full_cmd}")
        rc, build_out, build_err = _ssh_exec(ssh, build_full_cmd, timeout=300)
        build_log = (build_out + "\n" + build_err).strip()
        if rc != 0:
            return {
                "success": False,
                "error": f"Build step failed (exit {rc}):\n{build_log}",
                "build_log": build_log,
            }
        logger.info(f"cannsim_remote: build OK:\n{build_log[-1000:]}")

        # 7. Run cannsim record — wrap the BINARY, not the run script
        #    Flags MUST come before the -- separator.
        #    Do NOT use `-o <dir>`: with -o, cannsim's CWD becomes the timestamped
        #    subdir and log_ca/ can't be found, producing a tiny instr.bin and no
        #    valid trace. Instead, cd into remote_job_dir and let cannsim create
        #    its own `cannsim_<ts>_<bin>/` subdir there.
        #    `-g` is normally redundant because we always call `cannsim report`
        #    ourselves below. BUT instr.bin is only flushed during cannsim's
        #    atexit/teardown phase. For teardown-hanging kernels (al.multibuffer),
        #    the early-kill wrapper waits for a size-stable instr.bin before
        #    killing (see _ssh_cannsim_record) — otherwise the kill races the
        #    write and `cannsim report` fails with "instr log file is not found".
        remote_binary = remote_job_dir.rstrip("/") + "/" + binary_name

        logger.info(
            "cannsim-remote: running cannsim record with early-kill wrapper")
        rc, record_out, record_err = _ssh_cannsim_record(
            ssh, setenv, remote_job_dir, remote_binary, soc, timeout)
        record_log = (record_out + "\n" + record_err).strip()

        if rc != 0:
            if _cannsim_record_completed(record_log):
                # CANN 9.0.0 cleanup bug: cannsim exited non-zero after a
                # successful simulation.  Log a warning and continue — instr.bin
                # and log_ca are intact so cannsim report will succeed normally.
                logger.warning(
                    f"cannsim-remote: cannsim record exited {rc} but simulation "
                    "completed (CANN 9.0.0 os.getcwd() cleanup bug) — continuing "
                    "to report step")
            else:
                return {
                    "success": False,
                    "error":
                    f"cannsim record failed (exit {rc}):\n{record_log[-3000:]}",
                    "job_name": job_name,
                    "remote_job_dir": remote_job_dir,
                    "patch_log": patch_log,
                    "build_log": build_log,
                    "cannsim_log_tail": record_log[-4000:],
                }

        # 8. Run cannsim report to produce trace_core0.json
        #    cannsim record (without -o) creates a `cannsim_<ts>_<binname>/` subdir
        #    inside remote_job_dir containing instr.bin, cannsim.log, and log_ca/.
        #    Find it, then call `cannsim report -e <exp_dir> -o <exp_dir>/report -n 0`.
        trace_json_content = ""
        trace_local_path = ""
        report_log = ""
        exp_dir = ""

        if gen_report:
            # Find the most recently created cannsim_*_<binary_name> dir under remote_job_dir
            find_exp = (
                f"ls -1dt {remote_job_dir}/cannsim_*_{binary_name} 2>/dev/null | head -1"
            )
            _, exp_out, _ = _ssh_exec(ssh, find_exp, timeout=15)
            exp_dir = exp_out.strip()

            if not exp_dir:
                logger.warning(
                    "cannsim_remote: experiment dir not found; skipping report"
                )
                report_log = "experiment dir not found"
            else:
                report_out_dir = exp_dir + "/report"
                # Note: `cannsim record -g` already auto-generates report/trace_core0.json
                # for core 0. We re-run report here for explicit control (e.g. other cores).
                cannsim_report_cmd = (
                    f"source {setenv} && "
                    f"cd {exp_dir} && "
                    f"cannsim report -e {exp_dir} -o {report_out_dir} -n 0")
                logger.info(
                    f"cannsim_remote: running report: {cannsim_report_cmd}")
                rc_rep, rep_out, rep_err = _ssh_exec(ssh,
                                                     cannsim_report_cmd,
                                                     timeout=report_timeout)
                report_log = (rep_out + "\n" + rep_err).strip()

                if rc_rep != 0:
                    logger.warning(
                        f"cannsim report returned non-zero: {rc_rep}\n{report_log[-1000:]}"
                    )
                else:
                    # Find and download trace_core0.json
                    find_cmd = f"find {report_out_dir} {exp_dir} -name 'trace_core0.json' 2>/dev/null | head -1"
                    _, find_out, _ = _ssh_exec(ssh, find_cmd, timeout=15)
                    remote_trace = find_out.strip()

                    if remote_trace:
                        tmp = tempfile.NamedTemporaryFile(
                            suffix="_trace_core0.json",
                            prefix=f"cannsim_{job_name}_",
                            delete=False,
                        )
                        trace_local_path = tmp.name
                        tmp.close()

                        sftp = ssh.open_sftp()
                        try:
                            sftp.get(remote_trace, trace_local_path)
                        finally:
                            sftp.close()

                        with open(trace_local_path) as f:
                            raw = f.read()
                        trace_json_content = raw[:200_000]
                        logger.info(
                            f"cannsim_remote: trace downloaded to {trace_local_path} ({len(raw)} bytes)"
                        )
                    else:
                        logger.warning(
                            "cannsim_remote: trace_core0.json not found after report"
                        )

        return {
            "success": True,
            "job_name": job_name,
            "remote_job_dir": remote_job_dir,
            "remote_experiment_dir": exp_dir,
            "patch_log": patch_log,
            "build_log": build_log[-2000:],
            "cannsim_log_tail": record_log[-4000:],
            "report_log": report_log[-2000:],
            "trace_local_path": trace_local_path,
            "trace_json": trace_json_content,
            "trace_truncated": len(trace_json_content) == 200_000,
        }

    except Exception as e:
        logger.exception("cannsim_remote: unexpected error")
        return {"success": False, "error": str(e)}
    finally:
        ssh.close()


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_tool(
        name="cannsim_remote_run",
        toolset="triton_ascend",
        schema={
            "name":
            "cannsim_remote_run",
            "description":
            ("Upload a local directory of C++ Triton kernel sources to a remote Ascend machine, "
             "build the test binary there (keeping build noise out of the trace), run "
             "'cannsim record -g' wrapping the compiled binary, then run 'cannsim report -n 0 --timeline' "
             "to produce trace_core0.json, download it, and return its contents for optimization analysis. "
             "Requires env vars: CANNSIM_REMOTE_HOST, CANNSIM_REMOTE_USER, CANNSIM_REMOTE_PASS."
             ),
            "parameters": {
                "type": "object",
                "properties": {
                    "local_dir": {
                        "type":
                        "string",
                        "description":
                        "Local directory containing C++ sources, CMakeLists.txt, and run_script.",
                    },
                    "run_script": {
                        "type":
                        "string",
                        "description":
                        "Filename of the build/run shell script inside local_dir (e.g. 'run_kernel.sh').",
                    },
                    "binary_name": {
                        "type":
                        "string",
                        "description":
                        ("Name of the compiled test binary inside local_dir that cannsim record will wrap. "
                         "Default: 'test_kernel'. cannsim wraps THIS binary — not the run script."
                         ),
                    },
                    "build_cmd": {
                        "type":
                        "string",
                        "description":
                        ("Shell command to build the binary on the remote, executed inside the job dir. "
                         "Default: 'bash <run_script> build'. "
                         "The build runs BEFORE cannsim record so compiler noise stays out of the trace."
                         ),
                    },
                    "job_name": {
                        "type":
                        "string",
                        "description":
                        "Name of the job subdirectory on the remote (default: basename + timestamp).",
                    },
                    "soc_version": {
                        "type":
                        "string",
                        "description":
                        "cannsim -s value (default: CANNSIM_SOC_VERSION env or 'Ascend950').",
                    },
                    "cannsim_output_subdir": {
                        "type":
                        "string",
                        "description":
                        "Subdirectory inside the remote job dir for cannsim output (default: 'output').",
                    },
                    "gen_report": {
                        "type":
                        "boolean",
                        "description":
                        ("If true (default), pass -g to cannsim record and run 'cannsim report -n 0 --timeline' "
                         "to produce trace_core0.json. The trace JSON is returned in the 'trace_json' field. "
                         "Always use true for optimization work — cycle counts from cannsim.log are not actionable."
                         ),
                    },
                    "timeout": {
                        "type":
                        "integer",
                        "description":
                        "SSH timeout for the cannsim record step in seconds (default: 1800). Large GEMM kernels (e.g. 4096×4096) can take 1500+ seconds to simulate — set this higher if the run is killed mid-simulation.",
                    },
                    "report_timeout": {
                        "type":
                        "integer",
                        "description":
                        "SSH timeout for the cannsim report step in seconds (default: 300).",
                    },
                },
                "required": ["local_dir", "run_script"],
            },
        },
        handler=lambda args, **kw: json.dumps(
            _cannsim_remote_run(
                local_dir=args["local_dir"],
                run_script=args["run_script"],
                binary_name=args.get("binary_name", "test_kernel"),
                build_cmd=args.get("build_cmd"),
                job_name=args.get("job_name"),
                soc_version=args.get("soc_version"),
                cannsim_output_subdir=args.get("cannsim_output_subdir",
                                               "output"),
                gen_report=args.get("gen_report", True),
                timeout=args.get("timeout", 1800),
                report_timeout=args.get("report_timeout", 300),
            )),
        check_fn=lambda: bool(
            os.environ.get("CANNSIM_REMOTE_HOST") and os.environ
            .get("CANNSIM_REMOTE_USER") and os.environ.get(
                "CANNSIM_REMOTE_PASS") and os.environ.get("CANNSIM_REMOTE_PORT"
                                                          )),
        requires_env=[
            "CANNSIM_REMOTE_HOST",
            "CANNSIM_REMOTE_USER",
            "CANNSIM_REMOTE_PASS",
            "CANNSIM_REMOTE_PORT",
            "CANNSIM_REMOTE_BASE_DIR",
            "CANNSIM_REMOTE_CONDA_ENV",
            "CANNSIM_SOC_VERSION",
            "CANNSIM_SETENV_PATH",
        ],
    )
