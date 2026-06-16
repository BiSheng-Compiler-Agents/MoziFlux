import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def _div_leakyrelu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    inv_div,
    neg_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 16)

    # Stream from global via L2
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")

    # Do math in input dtype
    inv = tl.full((), inv_div, x.dtype)
    slope = tl.full((), neg_slope, x.dtype)
    one = tl.full((), 1.0, x.dtype)
    zero = tl.full((), 0.0, x.dtype)

    y = x * inv
    # Branchless LeakyReLU: y + (slope - 1) * min(y, 0)
    k = slope - one
    y_neg = tl.minimum(y, zero)
    out = y + y_neg * k

    tl.store(y_ptr + offs, out, mask=mask, eviction_policy="evict_first")
