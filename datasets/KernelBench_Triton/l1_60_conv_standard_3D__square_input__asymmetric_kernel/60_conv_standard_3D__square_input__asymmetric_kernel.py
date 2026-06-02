import triton
import triton.language as tl

@triton.jit
def _touch_identity_kernel(ptr, n_elements: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        v = tl.load(ptr)
        tl.store(ptr, v)
