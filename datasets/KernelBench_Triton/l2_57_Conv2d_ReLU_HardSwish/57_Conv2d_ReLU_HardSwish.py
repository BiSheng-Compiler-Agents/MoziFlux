import triton
import triton.language as tl


@triton.jit
def _relu_hswish_inplace_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hint for better vectorization on contiguous ranges
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Fused ReLU + HardSwish:
    # r = max(x, 0)
    # y = r * clamp((r + 3)/6, 0, 1)
    # For r >= 0, clamp reduces to min((r + 3)/6, 1) = min(r + 3, 6) * (1/6)
    r = tl.maximum(x, 0.0)
    inv6 = 1.0 / 6.0
    y = r * tl.minimum(r + 3.0, 6.0) * inv6

    tl.store(x_ptr + offsets, y, mask=mask)
