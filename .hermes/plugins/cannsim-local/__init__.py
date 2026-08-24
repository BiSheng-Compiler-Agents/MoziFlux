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
  - `-g` is normally redundant because we always call `cannsim report`
    ourselves on the instr.bin left in the experiment dir. For reliable traces,
    early-kill is only allowed after the strong simulator completion marker
    (`all tasks are finished!`) AND after instr.bin is non-empty, 424-byte
    aligned, and unchanged in size+mtime for a conservative quiet window. Blind
    timeout kills are cleanup only and must be reported as failure.
  - cannsim.log cycle counts are not actionable. Use trace_core0.json from
    `cannsim report -e <exp_dir> -o <report_out_dir> -n 0` for optimization.
  - CANN 9.0.0 cleanup bug: cannsim record exits with code 1 after a
    successful simulation. Detected by "all tasks are finished!" +
    "current_dir = os.getcwd()" + "FileNotFoundError" + "_cleanup_user_env".
  - Default timeout is 1800s. Large GEMM kernels take ~1500s to simulate.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import select
import shutil
import signal
import subprocess
import tempfile
import time
from glob import glob
from typing import Any

logger = logging.getLogger(__name__)

INSTR_LOG_RECORD_SIZE = 424  # trace_tools.model2trace: struct "<QIIQ200s200s"

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
            "CANN set_env.sh path.")
        return dict(os.environ)
    if not os.path.isfile(setenv):
        logger.warning(
            f"cannsim-local: set_env.sh not found at {setenv}, using current env"
        )
        return dict(os.environ)

    dump_cmd = f"source {setenv} && env -0"
    proc = subprocess.run(
        dump_cmd,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning(
            f"cannsim-local: sourcing {setenv} failed: {proc.stderr}")
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
        "Ensure CANNSIM_SETENV_PATH points to a valid set_env.sh.")


# ---------------------------------------------------------------------------
# Triton patches for local simulation (no physical NPU needed)
#
# Three compatibility points need patching in this triton package revision:
#
# 1. triton/tools/get_ascend_devices.py
#    Add env_condition so is_compile_on_910_95=True when TRITON_ASCEND_ARCH is set.
#
# 2. triton/backends/ascend/compiler.py
#    - Import get_ascend_arch_from_env from driver
#    - Use it first: get_ascend_arch_from_env() or NPUUtils().get_arch()
#
# 3. triton/backends/ascend/runtime/utils.py
#    In compile-only mode, derive the lazy runtime target properties from
#    TRITON_ASCEND_ARCH instead of initializing a physical NPU while Triton's
#    cache-key scanner imports the CV autotuner.
# ---------------------------------------------------------------------------


def _apply_local_patches(conda_env: str) -> tuple[bool, str]:
    """Apply no-device Triton compatibility patches in-place.

    Patches the relevant Triton files directly in the conda env's site-packages.
    Idempotent — skips if already applied.
    """
    # Compilation runs in ``conda_env``. Resolve that environment first even
    # when the Hermes host process also has Triton installed; patching the host
    # package would leave the compiler environment unchanged.
    triton_dir = None
    conda_bin = os.environ.get("CONDA_BIN", "")
    if conda_bin:
        conda_root = os.path.dirname(os.path.dirname(conda_bin))
        conda_env_lib = os.path.join(conda_root, "envs", conda_env, "lib")
        if os.path.isdir(conda_env_lib):
            for ver_dir in sorted(os.listdir(conda_env_lib)):
                candidate = os.path.join(conda_env_lib, ver_dir,
                                         "site-packages", "triton")
                if os.path.isdir(candidate):
                    triton_dir = candidate
                    break
    if triton_dir is None:
        try:
            import triton
            triton_dir = os.path.dirname(triton.__file__)
        except (ImportError, SyntaxError):
            pass

    if triton_dir is None:
        return False, "Could not find triton package location"

    log_parts = []

    # Patch 1: get_ascend_devices.py — add env_condition
    p1 = os.path.join(triton_dir, "tools", "get_ascend_devices.py")
    if not os.path.isfile(p1):
        log_parts.append(
            "[PATCH] get_ascend_devices.py: file not found — skipping")
    else:
        with open(p1) as f:
            c1 = f.read()
        env_patch = (
            'env_condition = os.getenv("TRITON_ASCEND_ARCH", "").strip().lower() in (\n'
            '    "ascend910_9589", "ascend910b", "ascend950", "ascend910_95"\n'
            ')\n'
            'is_compile_on_910_95 = pci_condition or npu_smi_condition or env_condition'
        )
        old_assignment = (
            "is_compile_on_910_95 = pci_condition or npu_smi_condition")
        if c1.count(env_patch) == 1 and old_assignment not in c1:
            log_parts.append("[PATCH] get_ascend_devices.py: already applied")
        elif c1.count(old_assignment) == 1 and env_patch not in c1:
            c1 = c1.replace(old_assignment, env_patch, 1)
            ast.parse(c1, filename=p1)
            with open(p1, "w") as f:
                f.write(c1)
            log_parts.append("[PATCH] get_ascend_devices.py: applied OK")
        else:
            log_parts.append(
                "[ERROR] get_ascend_devices.py: expected exactly one unpatched "
                "assignment or one complete patched assignment")

    # Patch 2: compiler.py — import + use get_ascend_arch_from_env
    p2 = os.path.join(triton_dir, "backends", "ascend", "compiler.py")
    if not os.path.isfile(p2):
        log_parts.append("[PATCH] compiler.py: file not found — skipping")
    else:
        with open(p2) as f:
            c2 = f.read()
        old_import = "from triton.backends.ascend.driver import (\n    NPUUtils\n)"
        new_import = "from triton.backends.ascend.driver import (\n    NPUUtils,\n    get_ascend_arch_from_env,\n)"
        old_target = 'f"--target={NPUUtils().get_arch()}"'
        new_target = 'f"--target={get_ascend_arch_from_env() or NPUUtils().get_arch()}"'
        import_applied = c2.count(new_import) == 1 and old_import not in c2
        target_applied = c2.count(new_target) == 1 and old_target not in c2
        if import_applied and target_applied:
            log_parts.append("[PATCH] compiler.py: already applied")
        else:
            valid = True
            changed = False
            if not import_applied and c2.count(old_import) == 1:
                c2 = c2.replace(old_import, new_import, 1)
                changed = True
            elif not import_applied:
                log_parts.append(
                    "[ERROR] compiler.py: expected exactly one driver import block"
                )
                valid = False
            if not target_applied and c2.count(old_target) == 1:
                c2 = c2.replace(old_target, new_target, 1)
                changed = True
            elif not target_applied:
                log_parts.append(
                    "[ERROR] compiler.py: expected exactly one BishengIR target expression"
                )
                valid = False
            if valid and changed:
                ast.parse(c2, filename=p2)
                with open(p2, "w") as f:
                    f.write(c2)
                log_parts.append("[PATCH] compiler.py: applied OK")

    # Patch 3: runtime/utils.py — no physical-device query in compile-only mode
    p3 = os.path.join(triton_dir, "backends", "ascend", "runtime", "utils.py")
    if not os.path.isfile(p3):
        log_parts.append("[PATCH] runtime/utils.py: file not found — skipping")
    else:
        with open(p3) as f:
            c3 = f.read()
        old_runtime = """    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
"""
        new_runtime = """    compile_only = os.getenv(\"TRITON_COMPILE_ONLY\", \"\").lower() in (\"1\", \"true\")
    env_arch = os.getenv(\"TRITON_ASCEND_ARCH\", \"\").strip()
    if compile_only and env_arch:
        from triton.backends.compiler import GPUTarget

        is_a5 = env_arch.startswith(\"Ascend910_95\") or env_arch.startswith(\"Ascend950\")
        num_cube_core = 32
        num_vector_core = num_cube_core * 2 if is_a5 else num_cube_core
        _cached_params = {
            'target': GPUTarget(\"npu\", env_arch, 32),
            'device': 0,
            'prop': {
                'num_aicore': num_cube_core,
                'num_vectorcore': num_vector_core,
            },
            'num_cube_core': num_cube_core,
            'num_vector_core': num_vector_core,
            'ub_size_in_kbytes': 256 if is_a5 else 192,
            'rf_size_in_kbytes': 128 if is_a5 else None,
        }
        return _cached_params

    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
"""
        runtime_applied = (c3.count(new_runtime) == 1 and old_runtime not in c3
                           and "import os\n" in c3)
        if runtime_applied:
            log_parts.append("[PATCH] runtime/utils.py: already applied")
        elif c3.count(old_runtime) == 1 and new_runtime not in c3:
            if "import os\n" not in c3:
                if c3.count("import torch\n") != 1:
                    log_parts.append(
                        "[ERROR] runtime/utils.py: expected exactly one torch import"
                    )
                    c3 = ""
                else:
                    c3 = c3.replace("import torch\n",
                                    "import os\n\nimport torch\n", 1)
            if c3:
                c3 = c3.replace(old_runtime, new_runtime, 1)
                ast.parse(c3, filename=p3)
                with open(p3, "w") as f:
                    f.write(c3)
                log_parts.append("[PATCH] runtime/utils.py: applied OK")
        else:
            log_parts.append(
                "[ERROR] runtime/utils.py: expected exactly one _init_npu_params "
                "driver-target block or one complete compile-only replacement")

    log = "\n".join(log_parts)
    all_ok = not any(part.startswith("[ERROR]") for part in log_parts)
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
    is_cleanup_bug = ("current_dir = os.getcwd()" in log
                      and "FileNotFoundError" in log
                      and "_cleanup_user_env" in log)
    return has_completion and is_cleanup_bug


# ---------------------------------------------------------------------------
# Core tool logic
# ---------------------------------------------------------------------------


def _find_experiment_dir(cwd: str) -> str | None:
    """Return the most-recently-created cannsim_*_<bin> experiment dir under cwd.

    cannsim record creates a fresh `cannsim_<ts>_<bin>/` subdir inside the job
    dir for each run. instr.bin and cannsim.log are written there.
    """
    candidates = sorted(glob(os.path.join(cwd, "cannsim_*")),
                        key=os.path.getmtime,
                        reverse=True)
    candidates = [c for c in candidates if os.path.isdir(c)]
    return candidates[0] if candidates else None


def _find_instr_bin(cwd: str) -> str | None:
    """Locate instr.bin. cannsim writes it EITHER at the job root (cwd) OR in
    the cannsim_<ts>_<bin> experiment subdir, depending on version/flags — both
    layouts occur in practice. Prefer whichever is newest and non-empty."""
    candidates = [os.path.join(cwd, "instr.bin")]
    exp = _find_experiment_dir(cwd)
    if exp:
        candidates.append(os.path.join(exp, "instr.bin"))
    present = [p for p in candidates if os.path.isfile(p)]
    if not present:
        return None
    non_empty = [p for p in present if os.path.getsize(p) > 0]
    if non_empty:
        present = non_empty
    # Newest non-empty wins; retain an empty path only when all candidates are
    # empty so the caller can emit an accurate flush diagnostic.
    present.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return present[0]


def _wait_for_instr_bin(
    cwd: str,
    poll: float,
    max_wait: float,
    quiet_window: float = 30.0,
) -> tuple[bool, str]:
    """Block until instr.bin is safe to feed to cannsim report.

    instr.bin is appended in chunks, so two equal size samples are not enough:
    a long gap between chunk flushes can look stable. A reliable trace requires
    a non-empty file whose size is aligned to the decoded record size and whose
    size+mtime have remained unchanged for a full quiet_window.

    Returns (True, reason) when safe, otherwise (False, diagnostic).
    """
    deadline = time.time() + max_wait
    stable_since: float | None = None
    last_state: tuple[int, int] | None = None
    last_diag = "instr.bin not found"

    while time.time() < deadline:
        instr = _find_instr_bin(cwd)
        now = time.time()
        if instr:
            try:
                st = os.stat(instr)
                size = st.st_size
                mtime_ns = st.st_mtime_ns
            except OSError as e:
                last_diag = f"could not stat instr.bin: {e}"
                stable_since = None
                last_state = None
            else:
                state = (size, mtime_ns)
                aligned = size > 0 and size % INSTR_LOG_RECORD_SIZE == 0
                if state != last_state:
                    stable_since = now if aligned else None
                    last_state = state
                elif aligned and stable_since is None:
                    stable_since = now

                if size <= 0:
                    last_diag = f"instr.bin exists but is empty: {instr}"
                elif size % INSTR_LOG_RECORD_SIZE != 0:
                    last_diag = (
                        f"instr.bin size {size} is not aligned to "
                        f"{INSTR_LOG_RECORD_SIZE}-byte records: {instr}")
                elif stable_since is not None:
                    quiet_for = now - stable_since
                    if quiet_for >= quiet_window:
                        return True, (
                            f"instr.bin stable for {quiet_for:.0f}s at {size} bytes "
                            f"({instr})")
                    last_diag = (
                        f"instr.bin aligned at {size} bytes but quiet for only "
                        f"{quiet_for:.0f}s/{quiet_window:.0f}s: {instr}")
        time.sleep(poll)

    return False, f"{last_diag}; did not become safe within {max_wait:.0f}s"


def _kill_pg(proc: subprocess.Popen) -> None:
    """Kill the entire process group created with preexec_fn=os.setsid."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _run(cmd: str, cwd: str, env: dict[str, str],
         timeout: int) -> tuple[int, str, str]:
    """Run a shell command with timeout. Returns (rc, stdout, stderr)."""
    logger.info(f"cannsim-local: running: {cmd}")
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_pg(proc)
        stdout, stderr = proc.communicate()
        logger.error(f"cannsim-local: command timed out after {timeout}s")
        return -1, stdout, stderr + f"\n[TIMEOUT after {timeout}s]"
    return proc.returncode, stdout, stderr


def _run_cannsim_record(
    cmd: str,
    cwd: str,
    env: dict[str, str],
    timeout: int,
    completion_markers: tuple[str, ...] = ("all tasks are finished", ),
    poll_interval: float = 3.0,
    instr_wait: float = 600.0,
    instr_quiet_window: float = 30.0,
) -> tuple[int, str, str]:
    """Run cannsim record, polling the binary's cannsim.log for completion markers.

    CANN runtime registers atexit handlers that synchronise all 32+ simulated AI
    cores.  If any core has an unresolved pipeline state (common with al.multibuffer
    double-buffering), the teardown hangs at ~1700% CPU and the binary never exits.
    The instruction trace (instr.bin) is serialized DURING the early teardown
    steps, so we must confirm it is fully written (size-stable) before killing —
    the hang itself is cosmetic, but the kill must not pre-empt the instr.bin write.

    Instead of waiting for the full timeout, we poll cannsim.log for the strong
    simulator completion marker, then wait for a safe instr.bin: non-empty,
    aligned to the trace record size, and unchanged in size+mtime for a full
    quiet window. Timeout kills are cleanup only and return failure.

    Returns (rc, stdout, stderr) with the same shape as subprocess.communicate().
    On early kill, rc=0 and stderr carries a note about early exit.
    On unsafe early kill or timeout, rc=-1 and stderr carries a diagnostic.
    """
    logger.info(f"cannsim-local: running: {cmd}")
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    start = time.time()

    cannsim_log_path = os.path.join(cwd, "cannsim.log")

    def _read_stream(stream, chunks):
        """Non-blocking read of a pipe into a list."""
        if stream is None:
            return
        try:
            r, _, _ = select.select([stream], [], [], 0)
            if r:
                line = stream.readline()
                while line:
                    chunks.append(line)
                    r, _, _ = select.select([stream], [], [], 0)
                    if not r:
                        break
                    line = stream.readline()
        except (ValueError, OSError):
            pass

    while True:
        elapsed = time.time() - start
        remaining = timeout - elapsed

        if remaining <= 0:
            _kill_pg(proc)
            _read_stream(proc.stdout, stdout_chunks)
            _read_stream(proc.stderr, stderr_chunks)
            stdout, stderr = proc.communicate()
            stdout_chunks.append(stdout or "")
            stderr_chunks.append(stderr or "")
            logger.error(
                f"cannsim-local: cannsim record timed out after {timeout}s")
            return -1, "".join(stdout_chunks), "".join(
                stderr_chunks) + f"\n[TIMEOUT after {timeout}s]"

        # Non-blocking read of stdout/stderr
        _read_stream(proc.stdout, stdout_chunks)
        _read_stream(proc.stderr, stderr_chunks)

        # Check if process exited naturally
        ret = proc.poll()
        if ret is not None:
            stdout_rest, stderr_rest = proc.communicate()
            stdout_chunks.append(stdout_rest or "")
            stderr_chunks.append(stderr_rest or "")
            return ret, "".join(stdout_chunks), "".join(stderr_chunks)

        # Poll cannsim.log for completion markers
        if os.path.isfile(cannsim_log_path):
            try:
                with open(cannsim_log_path) as lf:
                    log_content = lf.read()
                if any(marker in log_content for marker in completion_markers):
                    logger.info(
                        "cannsim-local: trace complete marker detected in cannsim.log — "
                        "waiting for instr.bin to flush before killing binary")
                    # CRITICAL: instr.bin is appended in chunks, so one unchanged
                    # poll interval can false-stabilize during a gap between
                    # flushes. Require a conservative quiet window plus 424-byte
                    # record alignment before treating early-kill as safe. If this
                    # check fails, kill only for cleanup and return failure.
                    safe, instr_reason = _wait_for_instr_bin(
                        cwd,
                        poll_interval,
                        instr_wait,
                        instr_quiet_window,
                    )
                    if safe:
                        kill_reason = instr_reason
                    else:
                        logger.error("cannsim-local: unsafe to report: %s",
                                     instr_reason)
                        _read_stream(proc.stdout, stdout_chunks)
                        _read_stream(proc.stderr, stderr_chunks)
                        _kill_pg(proc)
                        proc.wait()
                        note = (
                            f"\n[UNSAFE EARLY EXIT: {instr_reason}. "
                            "Binary killed for cleanup; trace_core0.json is not reliable.]"
                        )
                        return -1, "".join(
                            stdout_chunks), "".join(stderr_chunks) + note

                    # One last read of the pipe streams
                    _read_stream(proc.stdout, stdout_chunks)
                    _read_stream(proc.stderr, stderr_chunks)

                    _kill_pg(proc)
                    proc.wait()
                    killed_at = time.time() - start
                    note = (f"\n[EARLY EXIT: {kill_reason}. "
                            f"Binary killed after {killed_at:.0f}s to avoid "
                            f"CANN runtime atexit teardown hang.]")
                    logger.info(
                        f"cannsim-local: binary killed after {killed_at:.0f}s "
                        f"({kill_reason})")
                    return 0, "".join(
                        stdout_chunks), "".join(stderr_chunks) + note
            except OSError:
                pass  # log file may be temporarily locked; try again next poll

        time.sleep(min(poll_interval, remaining if remaining > 0 else 1))


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
    return_trace_json: bool = False,
    trace_json_max_bytes: int = 200_000,
) -> dict[str, Any]:
    """
    Build a Triton test binary locally, run cannsim record, run cannsim report,
    and return trace_core0.json path plus concise logs. Full trace JSON is only
    returned when return_trace_json=True.

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
    return_trace_json     : include trace_core0.json content in output (default: False)
    trace_json_max_bytes  : max trace JSON bytes to include when requested
    """
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
            "error": f"run_script not found: {run_script_path}"
        }

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
    job_name = job_name or (os.path.basename(local_dir.rstrip("/")) + "_" +
                            str(int(time.time())))
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
        return {
            "success": False,
            "error": f"Failed to apply triton patches:\n{patch_log}"
        }

    # ── Step 1: Build ──────────────────────────────────────────────────────
    effective_build_cmd = build_cmd or f"bash {run_script} build"
    build_shell_cmd = (f"{conda_bin} run -n {conda_env} bash -c "
                       f"'cd {job_dir} && {effective_build_cmd}'")
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
    rc, record_out, record_err = _run_cannsim_record(record_cmd,
                                                     job_dir,
                                                     env,
                                                     timeout=timeout)
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
                "error":
                f"cannsim record failed (exit {rc}):\n{record_log[-3000:]}",
                "job_dir": job_dir,
                "build_log": build_log[-2000:],
                "cannsim_log_tail": record_log[-4000:],
            }

    # ── Step 3: cannsim report ────────────────────────────────────────────
    trace_json_content = ""
    trace_json_size_bytes = 0
    trace_local_path = ""
    report_log = ""
    report_error = ""
    exp_dir = ""

    if gen_report:
        # cannsim report needs the directory that CONTAINS log_ca/ + instr.bin.
        # Depending on cannsim version/flags this is EITHER the job root OR the
        # cannsim_<ts>_<bin> experiment subdir. Pick whichever actually holds
        # log_ca/ (preferring the job root, which is the common layout).
        pattern = os.path.join(job_dir, f"cannsim_*_{binary_name}")
        candidates = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
        exp_subdir = candidates[0] if candidates else ""

        report_src = ""
        for cand in (job_dir, exp_subdir):
            if cand and os.path.isdir(os.path.join(cand, "log_ca")):
                report_src = cand
                break
        # Fall back to the experiment subdir if neither has log_ca/ (lets the
        # report command surface its own diagnostic).
        if not report_src:
            report_src = exp_subdir or job_dir
        exp_dir = report_src

        if not exp_dir:
            logger.warning(
                "cannsim-local: report source dir not found; skipping report")
            report_log = "report source dir not found"
            report_error = report_log
        else:
            report_out_dir = os.path.join(exp_dir, "report")
            report_cmd = f"{cannsim_bin} report -e {exp_dir} -o {report_out_dir} -n 0"
            rc_rep, rep_out, rep_err = _run(report_cmd,
                                            exp_dir,
                                            env,
                                            timeout=report_timeout)
            report_log = (rep_out + "\n" + rep_err).strip()

            if rc_rep != 0:
                report_error = f"cannsim report failed (exit {rc_rep})"
                logger.warning(
                    f"cannsim report returned non-zero: {rc_rep}\n{report_log[-1000:]}"
                )
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
                    trace_json_size_bytes = os.path.getsize(trace_path)
                    if return_trace_json:
                        with open(trace_path) as f:
                            raw = f.read()
                        trace_json_content = raw[:trace_json_max_bytes]
                    logger.info(
                        f"cannsim-local: trace available at {trace_path} "
                        f"({trace_json_size_bytes} bytes)")
                else:
                    report_error = "trace_core0.json not found after cannsim report"
                    logger.warning(
                        "cannsim-local: trace_core0.json not found after report"
                    )

    success = not gen_report or bool(trace_local_path)
    result = {
        "success":
        success,
        "job_name":
        job_name,
        "job_dir":
        job_dir,
        "experiment_dir":
        exp_dir,
        "patch_log":
        patch_log,
        "build_log":
        build_log[-2000:],
        "cannsim_log_tail":
        record_log[-4000:],
        "report_log":
        report_log[-2000:],
        "trace_local_path":
        trace_local_path,
        "trace_json":
        trace_json_content,
        "trace_json_size_bytes":
        trace_json_size_bytes,
        "trace_json_returned":
        return_trace_json,
        "trace_truncated":
        bool(return_trace_json
             and trace_json_size_bytes > len(trace_json_content)),
    }
    if not success:
        result[
            "error"] = report_error or "cannsim report produced no usable trace"
    return result


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_tool(
        name="cannsim_local_run",
        toolset="triton_ascend",
        schema={
            "name":
            "cannsim_local_run",
            "description":
            ("Build a Triton test binary locally, run 'cannsim record' wrapping "
             "the compiled binary, then run 'cannsim report -n 0' to produce "
             "trace_core0.json, and return its contents for optimization analysis. "
             "Requires CANN toolkit (set CANNSIM_SETENV_PATH to set_env.sh) and the "
             "conda env with triton-ascend installed (set CONDA_BIN and CONDA_ENV)."
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
                        ("Shell command to build the binary, run inside the job dir with the "
                         "conda env activated. Default: 'bash <run_script> build'. "
                         "The build runs BEFORE cannsim record so compiler noise stays out of the trace."
                         ),
                    },
                    "job_name": {
                        "type":
                        "string",
                        "description":
                        "Name of the job subdirectory under /tmp/cannsim_local/ (default: basename + timestamp).",
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
                        "Subdirectory for cannsim output (default: 'output').",
                    },
                    "gen_report": {
                        "type":
                        "boolean",
                        "description":
                        ("If true (default), run 'cannsim report -n 0' to produce trace_core0.json. "
                         "Always use true for optimization work — cycle counts from cannsim.log are not actionable."
                         ),
                    },
                    "timeout": {
                        "type":
                        "integer",
                        "description":
                        "Timeout for the cannsim record step in seconds (default: 1800). Large GEMM kernels (e.g. 4096x4096) can take 1500+ seconds to simulate.",
                    },
                    "report_timeout": {
                        "type":
                        "integer",
                        "description":
                        "Timeout for the cannsim report step in seconds (default: 300).",
                    },
                    "return_trace_json": {
                        "type":
                        "boolean",
                        "description":
                        ("If true, include trace_core0.json content in the tool result. "
                         "Default false returns only paths and concise logs to avoid oversized outputs."
                         ),
                    },
                    "trace_json_max_bytes": {
                        "type":
                        "integer",
                        "description":
                        "Maximum trace JSON bytes to include when return_trace_json=true (default: 200000).",
                    },
                },
                "required": ["local_dir", "run_script"],
            },
        },
        handler=lambda args, **kw: json.dumps(
            _cannsim_local_run(
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
                return_trace_json=args.get("return_trace_json", False),
                trace_json_max_bytes=args.get("trace_json_max_bytes", 200_000),
            )),
        check_fn=lambda: bool(
            os.environ.get("CANNSIM_SETENV_PATH", "") and os.path.isfile(
                os.environ["CANNSIM_SETENV_PATH"])),
        requires_env=["CANNSIM_SETENV_PATH", "CONDA_BIN", "CONDA_ENV"],
    )
