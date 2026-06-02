import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _spatial_mean_subtract_kernel(
    x_ptr,       # *: [N, C, D, H, W] contiguous in spatial dims
    y_ptr,       # *: [N, C, D, H, W] output
    stride_n,    # stride along N dimension (elements)
    stride_c,    # stride along C dimension (elements)
    S,           # total spatial elements per (n, c) = D*H*W
    N,           # batch size
    C,           # channels
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Map program id to (n, c)
    n = pid // C
    c = pid % C

    # Base pointers for this (n, c) plane
    base_x = x_ptr + n * stride_n + c * stride_c
    base_y = y_ptr + n * stride_n + c * stride_c

    offsets = tl.arange(0, BLOCK_SIZE)
    tl.static_assert(BLOCK_SIZE % 128 == 0)

    # First pass: compute spatial mean using scalar accumulator (low register pressure)
    sum_acc = tl.full((), 0.0, dtype=tl.float32)
    i = 0
    while i < S:
        idx = i + offsets
        mask = idx < S
        vals = tl.load(base_x + idx, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        sum_acc += tl.sum(vals, axis=0)
        i += BLOCK_SIZE

    denom = tl.full((), S, dtype=tl.float32)
    mean = sum_acc / denom  # scalar in fp32

    # Second pass: subtract mean and write out
    i = 0
    while i < S:
        idx = i + offsets
        mask = idx < S
        vals = tl.load(base_x + idx, mask=mask, other=0.0, eviction_policy="evict_last")
        out = vals.to(tl.float32) - mean
        tl.store(base_y + idx, out.to(vals.dtype), mask=mask)
        i += BLOCK_SIZE
