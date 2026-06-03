# Using c.asm["npubin"] instead of TRITON_KERNEL_DUMP

## Problem

On triton-ascend 3.2.0, `TRITON_KERNEL_DUMP` sometimes produces empty dump directories.
The `.npubin` file is never written to disk, even though `compile()` returns successfully.
The hash-named subdirectory inside `DUMP_DIR` exists but is empty.

## Root Cause

The dump-to-disk mechanism (`TRITON_KERNEL_DUMP`) is a side effect of the compile pipeline.
On some triton-ascend versions (3.2.0 verified), the dump step silently skips the binary
write when specific compiler configurations are used. The `compile()` return value still
holds the npubin bytes in its `asm` dict — they just don't get flushed to disk.

## Fix: Extract npubin from the compile object

Instead of setting `TRITON_KERNEL_DUMP` and globing for the file, read the bytes directly:

```python
import os

OUT_DIR = os.getcwd()  # run_kernel.sh does cd "$SCRIPT_DIR" first

os.environ["TRITON_ASCEND_ARCH"] = "Ascend910_9589"
os.environ["TRITON_COMPILE_ONLY"] = "1"

import triton
import triton.language as tl
from triton.compiler import compile, ASTSource
from triton.backends.compiler import GPUTarget

@triton.jit
def my_kernel(...):
    ...

c = compile(
    ASTSource(fn=my_kernel, signature={...}, constants={...}),
    target=GPUTarget("npu", "Ascend910_9589", 32),
)

npubin_path = os.path.join(OUT_DIR, "my_kernel.npubin")
data = c.asm["npubin"]  # bytes
with open(npubin_path, "wb") as f:
    f.write(data)
print(f"[COMPILE] npubin written to {npubin_path} ({len(data)} bytes)")
```

## Key advantages over TRITON_KERNEL_DUMP

| Aspect | TRITON_KERNEL_DUMP | c.asm dict |
|--------|-------------------|------------|
| Reliability | Sometimes empty on 3.2.0 | Always works |
| Env vars needed | 2 (TRITON_KERNEL_DUMP + TRITON_DUMP_DIR) | 0 |
| File globbing | Yes — hash-named subdirectory varies | No — direct bytes |
| CWD dependency | Uses __file__.parent (may differ from job dir) | Uses os.getcwd() (matches run_kernel.sh cd) |

## Verification

After writing the npubin with this method, confirm the binary exists and has reasonable size:

```bash
ls -la my_kernel.npubin
# Expected: 5000-50000 bytes for typical vector kernels
strings my_kernel.npubin | head -3
# Should contain the kernel function name
```
