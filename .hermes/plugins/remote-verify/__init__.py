"""
remote-verify plugin
====================
Transfer kernel files to a remote Ascend machine via SSH/SFTP, run
profile_kernels.py for correctness and benchmark results, and download
results back to the local workflow.

Workflow:
  1. Connect to remote via SSH (paramiko), with optional proxy support
  2. Upload all kernel files + profile_kernels.py via SFTP
  3. Run "python profile_kernels.py --test" for correctness
  4. If correctness passes, run "python profile_kernels.py --bench" for benchmark
  5. Download results (perf table, correctness output) back to local workspace
  6. Return results to agent for analysis and episode recording

Required environment variables:
  REMOTE_VERIFY_HOST     — SSH hostname / IP
  REMOTE_VERIFY_USER     — SSH username
  REMOTE_VERIFY_PASS     — SSH password
  REMOTE_VERIFY_PORT     — SSH port (default: 22)
  REMOTE_VERIFY_BASE_DIR — base dir on remote (default: ~/kernel_verify)
  REMOTE_VERIFY_CONDA_ENV — conda env name (default: compilerclaw)

The pre_tool_call hook blocks writes to project files outside the workspace
for kernelbench sessions to maintain sandbox boundaries.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _remote_host() -> str:
    return os.environ.get("REMOTE_VERIFY_HOST", "")


def _remote_user() -> str:
    return os.environ.get("REMOTE_VERIFY_USER", "")


def _remote_pass() -> str:
    return os.environ.get("REMOTE_VERIFY_PASS", "")


def _remote_port() -> int:
    # Optional: blank or unset → default 22. Tolerate "" (set-but-empty in .env).
    val = os.environ.get("REMOTE_VERIFY_PORT", "").strip()
    return int(val) if val else 22


def _remote_base_dir() -> str:
    return os.environ.get("REMOTE_VERIFY_BASE_DIR",
                          "").strip() or "~/kernel_verify"


def _remote_conda_env() -> str:
    return os.environ.get("REMOTE_VERIFY_CONDA_ENV",
                          "").strip() or "compilerclaw"


def _remote_cann_env() -> str:
    """Path to CANN set_env.sh on remote machine.

    Searches known locations: conda env Ascend path, then system-wide installs.
    Resolve at runtime via SSH so it adapts to the remote user's home directory.
    """
    return os.environ.get(
        "REMOTE_VERIFY_CANN_ENV",
        "",
    ).strip() or "~/miniconda3/envs/compilerclaw/Ascend/cann/set_env.sh"


def _remote_proxy_command() -> str:
    return os.environ.get("REMOTE_VERIFY_PROXY_COMMAND", "")


# ---------------------------------------------------------------------------
# SSH/SFTP helpers
# ---------------------------------------------------------------------------


def _ssh_connect():
    """Create an SSH client connection.

    Supports direct connections and proxy commands (e.g. SOCKS5 via nc).
    Set REMOTE_VERIFY_PROXY_COMMAND for proxy support:
    """
    import paramiko

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    proxy_cmd = _remote_proxy_command()
    sock = None
    if proxy_cmd:
        # Resolve %h and %p placeholders in the proxy command
        resolved_cmd = proxy_cmd.replace("%h", _remote_host()).replace(
            "%p", str(_remote_port()))
        proxy = paramiko.ProxyCommand(resolved_cmd)
        sock = proxy

    ssh.connect(
        hostname=_remote_host(),
        port=_remote_port(),
        username=_remote_user(),
        password=_remote_pass(),
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
        sock=sock,
    )
    # Keepalive so a silently-dropped TCP connection (NAT/firewall idle reap,
    # proxy hop dying) surfaces as a transport error instead of an infinite
    # blocking read on recv_exit_status().
    try:
        transport = ssh.get_transport()
        if transport is not None:
            transport.set_keepalive(15)
    except Exception:
        pass
    return ssh


def _ssh_exec(ssh, command: str, timeout: int = 300) -> tuple[int, str, str]:
    """Run a command on the remote. Returns (rc, stdout, stderr).

    Enforces a real wall-clock timeout. paramiko's exec_command(timeout=) only
    arms the channel's read/write timeout — it does NOT bound
    channel.recv_exit_status(), which blocks forever if the remote never sends
    an exit-status message (dropped TCP connection, NAT idle reap, proxy hop
    dying, remote sshd reaping the session). We set the channel timeout AND poll
    exit_status_ready() against a deadline so a broken channel raises instead of
    hanging the whole agent.
    """
    import socket

    logger.info(f"remote-verify: running: {command[:200]}")
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    chan = stdout.channel
    # Arm the channel-level timeout too (covers recv()/read()).
    chan.settimeout(timeout)

    deadline = time.time() + timeout
    while not chan.exit_status_ready():
        if time.time() > deadline:
            try:
                chan.close()
            except Exception:
                pass
            raise TimeoutError(
                f"remote command exceeded {timeout}s with no exit status "
                f"(channel may be dead): {command[:120]}")
        # If the transport died (keepalive detected a dropped connection), bail.
        transport = chan.get_transport()
        if transport is not None and not transport.is_active():
            raise ConnectionError(
                f"SSH transport died while waiting for command to finish: {command[:120]}"
            )
        time.sleep(1)

    rc = chan.recv_exit_status()
    try:
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
    except socket.timeout:
        out, err = "", "[remote-verify: timed out reading command output]"
    return rc, out, err


def _resolve_home(ssh) -> str:
    """Resolve $HOME on the remote machine."""
    rc, out, _ = _ssh_exec(ssh, "echo $HOME", timeout=10)
    return out.strip()


def _sftp_upload_dir(sftp, local_dir: str, remote_dir: str) -> None:
    """Recursively upload a directory via SFTP.

    Skips build artifacts and accumulated remote_results from prior runs to
    keep uploads fast (nested remote_results/ can otherwise compound across
    runs and take minutes).
    """
    local_dir = os.path.expanduser(local_dir)
    _SKIP_DIRS = {
        "remote_results", "build", "__pycache__", "cannsim", "cannsim_work",
        "cannsim_baseline", "cannsim_opt", "cannsim_sim"
    }
    _SKIP_EXTS = {".npubin", ".npu", ".so", ".o", ".a"}
    for root, dirs, files in os.walk(local_dir):
        # Prune skipped directories in-place so os.walk doesn't descend.
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        rel = os.path.relpath(root, local_dir)
        remote_sub = os.path.join(remote_dir, rel).replace("\\", "/")
        try:
            sftp.stat(remote_sub)
        except FileNotFoundError:
            try:
                sftp.mkdir(remote_sub)
            except OSError:
                pass  # May already exist from parallel creation

        for fname in files:
            if os.path.splitext(fname)[1] in _SKIP_EXTS:
                continue
            local_path = os.path.join(root, fname)
            remote_path = f"{remote_sub}/{fname}"
            sftp.put(local_path, remote_path)


def _sftp_download_file(sftp, remote_path: str, local_path: str) -> None:
    """Download a single file via SFTP."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    sftp.get(remote_path, local_path)


# ---------------------------------------------------------------------------
# Core tool: remote_verify
# ---------------------------------------------------------------------------


def _remote_verify(
    local_dir: str,
    run_test: bool = True,
    run_bench: bool = True,
    timeout: int = 600,
) -> dict[str, Any]:
    """
    Transfer kernel files to remote, run profile_kernels.py, download results.

    Parameters
    ----------
    local_dir : str
        Local directory containing kernel files and profile_kernels.py
    run_test : bool
        Run "python profile_kernels.py --test" for correctness (default: True)
    run_bench : bool
        Run "python profile_kernels.py --bench" for benchmark (default: True)
    timeout : int
        Timeout for remote commands in seconds (default: 600)

    Returns
    -------
    dict with keys:
      - success: bool
      - test_passed: bool (only if run_test=True)
      - test_output: str
      - bench_output: str (only if run_bench=True)
      - remote_dir: str
      - error: str (only on failure)
    """
    local_dir = os.path.expanduser(local_dir)
    if not os.path.isdir(local_dir):
        return {
            "success": False,
            "error": f"local_dir does not exist: {local_dir}"
        }

    # Check that profile_kernels.py exists
    profile_path = os.path.join(local_dir, "profile_kernels.py")
    if not os.path.isfile(profile_path):
        return {
            "success": False,
            "error": f"profile_kernels.py not found in {local_dir}"
        }

    ssh = None
    sftp = None
    remote_dir = None

    try:
        ssh = _ssh_connect()
        sftp = ssh.open_sftp()

        # Resolve remote home and create job directory
        home = _resolve_home(ssh)
        base = _remote_base_dir().replace("~", home)
        job_name = os.path.basename(local_dir.rstrip("/"))
        remote_dir = f"{base}/{job_name}_{int(time.time())}"

        # Clean and create remote directory
        _ssh_exec(ssh, f"rm -rf {remote_dir}", timeout=30)
        _ssh_exec(ssh, f"mkdir -p {remote_dir}", timeout=30)

        # Upload all files
        logger.info(f"remote-verify: uploading {local_dir} -> {remote_dir}")
        _sftp_upload_dir(sftp, local_dir, remote_dir)

        # Verify upload
        rc, out, err = _ssh_exec(ssh, f"ls -la {remote_dir}", timeout=10)
        if rc != 0:
            return {
                "success": False,
                "error": f"Failed to list remote dir: {err}"
            }

        # Build the conda run command prefix with CANN environment sourced
        conda_env = _remote_conda_env()
        cann_env_script = _remote_cann_env()
        # Find conda binary on remote
        rc, conda_path, _ = _ssh_exec(
            ssh,
            "command -v conda 2>/dev/null || ls ~/miniconda3/bin/conda 2>/dev/null | head -1",
            timeout=10)
        conda_path = conda_path.strip()
        if not conda_path:
            return {
                "success": False,
                "error": "conda not found on remote machine"
            }

        # Resolve ~ in cann env script via $HOME already resolved above
        cann_env_resolved = cann_env_script.replace("~", home)

        run_prefix = (f"source {cann_env_resolved} 2>/dev/null; "
                      f"{conda_path} run -n {conda_env}")
        results = {"success": True, "remote_dir": remote_dir}

        # Run correctness test
        if run_test:
            test_cmd = (
                f"cd {remote_dir} && {run_prefix} python profile_kernels.py --test"
            )
            logger.info("remote-verify: running correctness test")
            rc, test_out, test_err = _ssh_exec(ssh, test_cmd, timeout=timeout)
            test_output = test_out + "\n" + test_err
            results["test_output"] = test_output
            results[
                "test_passed"] = rc == 0 and "FAIL" not in test_output.upper()

            if not results["test_passed"]:
                logger.warning("remote-verify: correctness test FAILED")
                # Don't run bench if test failed
                run_bench = False

        # Run benchmark
        if run_bench:
            bench_cmd = (
                f"cd {remote_dir} && {run_prefix} python profile_kernels.py --bench"
            )
            logger.info("remote-verify: running benchmark")
            rc, bench_out, bench_err = _ssh_exec(ssh,
                                                 bench_cmd,
                                                 timeout=timeout)
            results["bench_output"] = bench_out + "\n" + bench_err
            results["bench_passed"] = rc == 0

        # Download any generated result files
        rc, remote_files, _ = _ssh_exec(
            ssh,
            f"find {remote_dir} -name '*.txt' -o -name '*.csv' -o -name '*.log'",
            timeout=10)
        if remote_files.strip():
            for remote_file in remote_files.strip().split("\n"):
                remote_file = remote_file.strip()
                if remote_file:
                    rel = os.path.relpath(remote_file, remote_dir)
                    local_result = os.path.join(local_dir, "remote_results",
                                                rel)
                    try:
                        _sftp_download_file(sftp, remote_file, local_result)
                    except Exception as e:
                        logger.warning(
                            f"remote-verify: failed to download {remote_file}: {e}"
                        )

        return results

    except ImportError:
        return {
            "success":
            False,
            "error":
            ("paramiko not installed. "
             "Install into the Hermes agent venv: "
             "sudo /opt/hermes/.venv/bin/python3 -m ensurepip && "
             "sudo /opt/hermes/.venv/bin/python3 -m pip install paramiko"),
        }
    except Exception as e:
        logger.exception("remote-verify: unexpected error")
        return {"success": False, "error": str(e)}
    finally:
        if sftp:
            try:
                sftp.close()
            except Exception:
                pass
        if ssh:
            try:
                ssh.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_tool(
        name="remote_verify",
        toolset="triton_ascend",
        schema={
            "name":
            "remote_verify",
            "description":
            ("Transfer kernel files to a remote Ascend machine via SSH/SFTP, "
             "run profile_kernels.py for correctness and benchmark results, "
             "and download results back. "
             "Requires REMOTE_VERIFY_HOST, REMOTE_VERIFY_USER, REMOTE_VERIFY_PASS env vars."
             ),
            "parameters": {
                "type": "object",
                "properties": {
                    "local_dir": {
                        "type":
                        "string",
                        "description":
                        "Local directory containing kernel files and profile_kernels.py",
                    },
                    "run_test": {
                        "type": "boolean",
                        "description": "Run correctness test (default: True)",
                    },
                    "run_bench": {
                        "type": "boolean",
                        "description": "Run benchmark (default: True)",
                    },
                    "timeout": {
                        "type":
                        "integer",
                        "description":
                        "Timeout for remote commands in seconds (default: 600)",
                    },
                },
                "required": ["local_dir"],
            },
        },
        handler=lambda args, **kw: json.dumps(
            _remote_verify(
                local_dir=args.get("local_dir", ""),
                run_test=args.get("run_test", True),
                run_bench=args.get("run_bench", True),
                timeout=int(args.get("timeout", 600)),
            )),
        requires_env=[
            "REMOTE_VERIFY_HOST",
            "REMOTE_VERIFY_USER",
            "REMOTE_VERIFY_PASS",
        ],
        check_fn=lambda: bool(_remote_host() and _remote_user() and
                              _remote_pass()),
    )
