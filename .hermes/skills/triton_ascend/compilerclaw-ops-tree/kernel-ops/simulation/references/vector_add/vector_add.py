"""
Vector Add kernel — Triton-Ascend reference implementation.
Compiles the kernel and writes the npubin for cannsim.
"""
import os
import pathlib

SCRIPT_DIR = str(pathlib.Path(__file__).parent.resolve())
NPUBIN_DEST = os.path.join(SCRIPT_DIR, "cannsim_host", "add_kernel.npubin")

os.environ["TRITON_ASCEND_ARCH"] = "Ascend910_9589"
os.environ["TRITON_COMPILE_ONLY"] = "1"

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.compiler import compile, ASTSource  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


BLOCK = 1024

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

print(f"[COMPILE] Compiling add_kernel BLOCK={BLOCK} ...")
result = compile(src, target=GPUTarget("npu", "Ascend910_9589", 32))
print("[COMPILE] Compilation OK")

data = result.asm["npubin"]
os.makedirs(os.path.dirname(NPUBIN_DEST), exist_ok=True)
with open(NPUBIN_DEST, "wb") as f:
    f.write(data)
print(f"[COMPILE] npubin written to {NPUBIN_DEST} ({len(data)} bytes)")
