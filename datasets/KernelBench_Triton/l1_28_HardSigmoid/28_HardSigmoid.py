import triton
import triton.language as tl


@triton.jit
def _hardsigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Hints for better codegen/coalescing
    tl.max_contiguous(offsets, BLOCK_SIZE)
    tl.multiple_of(offsets, 8)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Piecewise-linear hardsigmoid with NaN propagation:
    # y = 0            if x <= -3
    # y = 1            if x >= 3
    # y = x/6 + 0.5    otherwise
    below = x <= -3.0
    above = x >= 3.0
    y_mid = x * (1.0 / 6.0) + 0.5
    y = tl.where(below, 0.0, y_mid)
    y = tl.where(above, 1.0, y)

    tl.store(y_ptr + offsets, y, mask=mask)
