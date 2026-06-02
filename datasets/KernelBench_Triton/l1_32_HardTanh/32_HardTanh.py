import triton
import triton.language as tl

@triton.jit
def _hardtanh_kernel(x_ptr, y_ptr, n_elements, min_val, max_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Preserve NaN behavior exactly like PyTorch: if x is NaN, keep it as NaN.
    x = tl.where(x > max_val, max_val, tl.where(x < min_val, min_val, x))
    tl.store(y_ptr + offsets, x, mask=mask)
