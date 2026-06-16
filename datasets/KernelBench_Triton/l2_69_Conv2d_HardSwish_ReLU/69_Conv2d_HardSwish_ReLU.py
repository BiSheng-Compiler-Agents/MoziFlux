import triton
import triton.language as tl


@triton.jit
def _hswish_relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1D launch over the flattened tensor
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Provide vectorization hints for better memory coalescing
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute ReLU(HardSwish(x)) with minimal ops:
    # rx = max(x, 0)
    # r  = min(rx/6 + 0.5, 1)
    # y  = rx * r
    rx = tl.maximum(x, 0.0)
    y = rx * tl.minimum(rx * (1.0 / 6.0) + 0.5, 1.0)

    tl.store(y_ptr + offsets, y, mask=mask)
