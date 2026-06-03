"""
compile_kernel.py — canonical template for compiling a Triton kernel for cannsim.

THREE MANDATORY SETUP RULES (discovered June 2026 — all three required):
  1. TRITON_ASCEND_ARCH = "Ascend910_9589"
     Triton's libdevice.py validates this; "Ascend950" raises ValueError.
  2. Clear ~/.triton/cache BEFORE importing triton
     Without this, the JIT cache silently returns cached IR and writes nothing to DUMP_DIR.
  3. Use a PID-unique DUMP_DIR (e.g. "_triton_dump_<pid>")
     Prevents stale hits across back-to-back runs in the same job directory.

ALSO:
  4. cannsim soc_version = "Ascend950" (in cannsim_remote_run call / run_kernel.sh)
     cannsim only supports Ascend950 as -s value; Ascend910_9589 fails with "not supported".
  5. Copy the binary to the job root in run_kernel.sh:
       cp "$BUILD_DIR/my_binary" "$SCRIPT_DIR/my_binary"
     cannsim looks for the binary at the job root, not inside build/.

Usage: place in local_dir/, invoked on remote by run_kernel.sh during build step.
"""
import os
import glob
import shutil
import pathlib
import subprocess

SCRIPT_DIR  = str(pathlib.Path(__file__).parent.resolve())
# Rule 3: PID-unique DUMP_DIR avoids stale cache hits across consecutive runs
DUMP_DIR    = os.path.join(SCRIPT_DIR, "_triton_dump_" + str(os.getpid()))
NPUBIN_DEST = os.path.join(SCRIPT_DIR, "my_kernel.npubin")

# Must be set BEFORE importing triton
os.environ["TRITON_KERNEL_DUMP"]  = "1"
os.environ["TRITON_DUMP_DIR"]     = DUMP_DIR
os.environ["TRITON_ASCEND_ARCH"]  = "Ascend910_9589"   # Rule 1 — do NOT use "Ascend950"
os.environ["TRITON_COMPILE_ONLY"] = "1"

# Rule 2: clear triton cache BEFORE import to force a fresh dump
import shutil as _shutil
_cache_dir = os.path.expanduser("~/.triton/cache")
if os.path.isdir(_cache_dir):
    print(f"[COMPILE] Clearing triton cache: {_cache_dir}")
    _shutil.rmtree(_cache_dir, ignore_errors=True)

import triton
import triton.language as tl
from triton.compiler import compile, ASTSource
from triton.backends.compiler import GPUTarget


# ── REPLACE: define your kernel here ─────────────────────────────────────────

@triton.jit
def my_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x    = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)


# ── REPLACE: set your block size and signature ────────────────────────────────

BLOCK = 1024

src = ASTSource(
    fn=my_kernel,
    signature={
        "x_ptr":   "*fp32",
        "out_ptr": "*fp32",
        "N":       "i32",
    },
    constants={"BLOCK": BLOCK},
)

print(f"[COMPILE] Compiling my_kernel BLOCK={BLOCK} ...")
result = compile(src, target=GPUTarget("npu", "Ascend910_9589", 32))
print("[COMPILE] Compilation OK")

# ── locate and copy npubin ────────────────────────────────────────────────────

all_npubins = sorted(glob.glob(os.path.join(DUMP_DIR, "**", "*.npubin"), recursive=True))
print(f"[COMPILE] Found npubins: {all_npubins}")

if not all_npubins:
    r = subprocess.run(["find", DUMP_DIR, "-type", "f"], capture_output=True, text=True)
    print(f"[COMPILE] Dump dir contents:\n{r.stdout}")
    raise FileNotFoundError(
        f"No .npubin found under {DUMP_DIR}. "
        "Checklist: (1) cache cleared before import? (2) TRITON_ASCEND_ARCH=Ascend910_9589? "
        "(3) TRITON_COMPILE_ONLY=1? (4) kernel body free of syntax errors?"
    )

shutil.copy2(all_npubins[0], NPUBIN_DEST)
print(f"[COMPILE] npubin written to {NPUBIN_DEST}  ({os.path.getsize(NPUBIN_DEST)} bytes)")
