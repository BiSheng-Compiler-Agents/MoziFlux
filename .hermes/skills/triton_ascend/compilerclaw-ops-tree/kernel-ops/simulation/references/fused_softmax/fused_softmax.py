"""
Fused Softmax kernel — Triton-Ascend reference implementation.
Compiles the kernel and writes the npubin for cannsim.
"""
import os
import pathlib

SCRIPT_DIR = str(pathlib.Path(__file__).parent.resolve())
NPUBIN_DEST = os.path.join(SCRIPT_DIR, "cannsim_host", "fused_softmax.npubin")

os.environ["TRITON_ASCEND_ARCH"] = "Ascend910_9589"
os.environ["TRITON_COMPILE_ONLY"] = "1"

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.compiler import compile, ASTSource  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402


@triton.jit
def fused_softmax_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm
    out_row = out_ptr + row * stride_om
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_row + cols, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    out = num / denom
    tl.store(out_row + cols, out, mask=mask)


BLOCK_N = 1024

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

print(f"[COMPILE] Compiling fused_softmax_kernel BLOCK_N={BLOCK_N} ...")
result = compile(src, target=GPUTarget("npu", "Ascend910_9589", 32))
print("[COMPILE] Compilation OK")

data = result.asm["npubin"]
os.makedirs(os.path.dirname(NPUBIN_DEST), exist_ok=True)
with open(NPUBIN_DEST, "wb") as f:
    f.write(data)
print(f"[COMPILE] npubin written to {NPUBIN_DEST} ({len(data)} bytes)")
