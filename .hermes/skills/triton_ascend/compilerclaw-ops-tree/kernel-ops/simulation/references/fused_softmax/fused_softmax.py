"""
Fused Softmax kernel — Triton-Ascend reference implementation.

Compiles the kernel and dumps the npubin to cannsim_host/ using
TRITON_KERNEL_DUMP without requiring a physical NPU.

This script is intended to run on the remote cannsim machine via run_kernel.sh,
not locally. It requires a CANN-patched triton-ascend environment (compilerclaw).
run_kernel.sh invokes it automatically as part of the build step.
"""
import os, glob, shutil

DUMP_DIR = "/tmp/triton_dump_softmax"
NPUBIN_DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "cannsim_host", "fused_softmax.npubin")

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_DUMP_DIR"] = DUMP_DIR
os.environ["TRITON_ASCEND_ARCH"] = "Ascend910_9589"
os.environ["TRITON_COMPILE_ONLY"] = "1"

import triton
import triton.language as tl
from triton.compiler import compile, ASTSource
from triton.backends.compiler import GPUTarget

# ---------------------------------------------------------------------------
# Fused Softmax kernel
#
# Each program handles one row of the input matrix [M, N].
# grid = (M,)  — one block per row.
#
# Algorithm (numerically stable):
#   1. Load a row of N elements
#   2. Subtract row max  (for numerical stability)
#   3. Exponentiate
#   4. Divide by sum of exponentials
# ---------------------------------------------------------------------------


@triton.jit
def fused_softmax_kernel(
    x_ptr,  # *fp32  input  [M, N]
    out_ptr,  # *fp32  output [M, N]
    M,  # i32   number of rows
    N,  # i32   number of columns (must equal BLOCK_N)
    stride_xm,  # i32   row stride of x   (= N for row-major)
    stride_om,  # i32   row stride of out (= N for row-major)
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    # Pointer to the start of this row
    x_row = x_ptr + row * stride_xm
    out_row = out_ptr + row * stride_om

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # 1. Load row
    x = tl.load(x_row + cols, mask=mask, other=-float("inf"))

    # 2. Numerically stable: subtract max
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # 3. Exponentiate
    num = tl.exp(x)

    # 4. Normalise
    denom = tl.sum(num, axis=0)
    out = num / denom

    # 5. Store
    tl.store(out_row + cols, out, mask=mask)


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

M = 128
N = 1024  # must be a power of 2 and <= BLOCK_N
BLOCK_N = 1024  # constexpr — must cover the full row

target = GPUTarget("npu", "Ascend910_9589", 32)

src = ASTSource(
    fn=fused_softmax_kernel,
    signature={
        "x_ptr": "*fp32",
        "out_ptr": "*fp32",
        "M": "i32",
        "N": "i32",
        "stride_xm": "i32",
        "stride_om": "i32",
    },
    constants={"BLOCK_N": BLOCK_N},
)

print(f"Compiling fused_softmax_kernel (M={M}, N={N}, BLOCK_N={BLOCK_N}) ...")
result = compile(src, target=target)
print("Compilation OK")

# ---------------------------------------------------------------------------
# Copy npubin to cannsim_host/
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(NPUBIN_DEST), exist_ok=True)

npubin_files = sorted(
    glob.glob(f"{DUMP_DIR}/**/fused_softmax_kernel.npubin", recursive=True))
if npubin_files:
    shutil.copy2(npubin_files[0], NPUBIN_DEST)
    print(
        f"npubin copied to: {NPUBIN_DEST}  ({os.path.getsize(NPUBIN_DEST)} bytes)"
    )
else:
    print("WARNING: fused_softmax_kernel.npubin not found in dump dir")

# Show all dumped files
print("\nDumped files:")
for f in sorted(glob.glob(f"{DUMP_DIR}/**/*", recursive=True)):
    if os.path.isfile(f):
        print(f"  {f}  ({os.path.getsize(f)} bytes)")
