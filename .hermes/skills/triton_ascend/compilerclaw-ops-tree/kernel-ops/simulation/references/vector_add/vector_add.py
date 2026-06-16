"""
Vector Add kernel — Triton-Ascend reference implementation.

Compiles the kernel and dumps the npubin to cannsim_host/ using
TRITON_KERNEL_DUMP without requiring a physical NPU.

This script is intended to run on the remote cannsim machine via run_kernel.sh,
not locally. It requires a CANN-patched triton-ascend environment (compilerclaw).
run_kernel.sh invokes it automatically as part of the build step.
"""
import os, glob, shutil

DUMP_DIR = "/tmp/triton_dump_vector_add"
NPUBIN_DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "cannsim_host", "add_kernel.npubin")

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_DUMP_DIR"] = DUMP_DIR
os.environ["TRITON_ASCEND_ARCH"] = "Ascend910_9589"
os.environ["TRITON_COMPILE_ONLY"] = "1"

import triton
import triton.language as tl
from triton.compiler import compile, ASTSource
from triton.backends.compiler import GPUTarget


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


N = 1024
BLOCK = 1024

target = GPUTarget("npu", "Ascend910_9589", 32)

src = ASTSource(
    fn=add_kernel,
    signature={
        "x_ptr": "*fp32",
        "y_ptr": "*fp32",
        "out_ptr": "*fp32",
        "n": "i32"
    },
    constants={"BLOCK": BLOCK},
)

print(f"Compiling add_kernel (N={N}, BLOCK={BLOCK}) ...")
result = compile(src, target=target)
print("Compilation OK")

# Copy npubin to cannsim_host/
os.makedirs(os.path.dirname(NPUBIN_DEST), exist_ok=True)

npubin_files = sorted(
    glob.glob(f"{DUMP_DIR}/**/add_kernel.npubin", recursive=True))
if npubin_files:
    shutil.copy2(npubin_files[0], NPUBIN_DEST)
    print(
        f"npubin copied to: {NPUBIN_DEST}  ({os.path.getsize(NPUBIN_DEST)} bytes)"
    )
else:
    print("WARNING: add_kernel.npubin not found in dump dir")

print("\nDumped files:")
for f in sorted(glob.glob(f"{DUMP_DIR}/**/*", recursive=True)):
    if os.path.isfile(f):
        print(f"  {f}  ({os.path.getsize(f)} bytes)")
