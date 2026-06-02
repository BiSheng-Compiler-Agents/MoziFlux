import triton
import triton.language as tl

@triton.jit
def _scale_kernel(x_ptr, y_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Fast path for full tiles: avoid mask overhead entirely
    full = block_start + BLOCK_SIZE <= n_elements
    if full:
        x = tl.load(x_ptr + offsets, cache_modifier=".cg")
        s_cast = tl.full((), s, x.dtype)
        y = x * s_cast
        tl.store(y_ptr + offsets, y)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        s_cast = tl.full((), s, x.dtype)
        y = x * s_cast
        tl.store(y_ptr + offsets, y, mask=mask)
