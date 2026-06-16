import triton
import triton.language as tl


@triton.jit
def _sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Numerically-stable sigmoid:
    # z = exp(-|x|); inv = 1 / (1 + z)
    # if x >= 0: y = inv
    # else:      y = z * inv
    z = tl.exp(-tl.abs(x32))
    inv = 1.0 / (1.0 + z)
    y32 = tl.where(x32 >= 0.0, inv, z * inv)

    y = y32.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)
