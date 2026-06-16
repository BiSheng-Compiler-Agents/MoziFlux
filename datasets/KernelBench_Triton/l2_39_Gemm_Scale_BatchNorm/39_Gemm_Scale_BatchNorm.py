import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 256
        },
                      num_warps=8,
                      num_stages=2),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128
        },
                      num_warps=4,
                      num_stages=2),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 256
        },
                      num_warps=4,
                      num_stages=2),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 128
        },
                      num_warps=8,
                      num_stages=2),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128
        },
                      num_warps=4,
                      num_stages=1),
    ],
    key=["M", "N"],
)
@triton.jit
def _affine_per_col_kernel(
    y_ptr,  # [M, N] input/output (row-major)
    alpha_ptr,  # [N] per-column scale
    beta_ptr,  # [N] per-column bias
    M,
    N,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load tile of y
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y = tl.load(y_ptrs, mask=mask, other=0.0, cache_modifier=".cg")

    # Load per-column params once per tile
    alpha = tl.load(alpha_ptr + offs_n, mask=mask_n, other=1.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)

    # Apply: y = y * alpha + beta
    y = y * alpha[None, :] + beta[None, :]

    # Store back
    tl.store(y_ptrs, y, mask=mask, cache_modifier=".cg")
