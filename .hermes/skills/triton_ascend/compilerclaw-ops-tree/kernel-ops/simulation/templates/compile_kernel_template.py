"""
compile_kernel.py — canonical template for compiling a Triton kernel for cannsim.

MANDATORY SETUP RULES:
  1. TRITON_ASCEND_ARCH = "Ascend910_9589" (set BEFORE importing triton)
  2. Clear ~/.triton/cache BEFORE importing triton
  3. Use PID-unique DUMP_DIR ("_triton_dump_<pid>")

ALSO:
  4. cannsim soc_version = "Ascend950"
  5. Copy the binary to the job root in run_kernel.sh

Usage: place in local_dir/, invoked on remote by run_kernel.sh during build step.
"""
import os
import shutil
import pathlib

SCRIPT_DIR = str(pathlib.Path(__file__).parent.resolve())
DUMP_DIR = os.path.join(SCRIPT_DIR, "_triton_dump_" + str(os.getpid()))
NPUBIN_DEST = os.path.join(SCRIPT_DIR, "my_kernel.npubin")

os.environ["TRITON_ASCEND_ARCH"] = "Ascend910_9589"
os.environ["TRITON_COMPILE_ONLY"] = "1"

# Clear triton cache BEFORE import
_cache_dir = os.path.expanduser("~/.triton/cache")
if os.path.isdir(_cache_dir):
    shutil.rmtree(_cache_dir, ignore_errors=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.compiler import compile, ASTSource  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402

# ── REPLACE: define your kernel here ─────────────────────────────────────────


@triton.jit
def my_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)


# ── REPLACE: set your block size and signature ────────────────────────────────

BLOCK = 1024

src = ASTSource(
    fn=my_kernel,
    signature={
        "x_ptr": "*fp32",
        "out_ptr": "*fp32",
        "N": "i32",
    },
    constants={"BLOCK": BLOCK},
)

print(f"[COMPILE] Compiling my_kernel BLOCK={BLOCK} ...")
result = compile(src, target=GPUTarget("npu", "Ascend910_9589", 32))
print("[COMPILE] Compilation OK")

# Extract npubin from compile result — more reliable than TRITON_KERNEL_DUMP
data = result.asm["npubin"]
with open(NPUBIN_DEST, "wb") as f:
    f.write(data)
print(f"[COMPILE] npubin written to {NPUBIN_DEST} ({len(data)} bytes)")
