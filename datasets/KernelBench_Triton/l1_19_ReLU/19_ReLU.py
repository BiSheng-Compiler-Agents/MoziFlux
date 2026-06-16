import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
    ],
    key=['n_elements'],
)
@triton.jit
def _relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
                 IS_FP: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Vectorization / coalescing hints
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, 16)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    zero = tl.zeros([BLOCK_SIZE], dtype=x.dtype)
    y = tl.maximum(x, zero)

    # Explicitly propagate NaNs for floating types to match torch.relu semantics
    if IS_FP:
        nan_mask = x != x
        y = tl.where(nan_mask, x, y)

    tl.store(y_ptr + offsets, y, mask=mask)
