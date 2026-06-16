import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=3,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_linear_min_sub_kernel(
    X_ptr,
    W_ptr,
    B_ptr,
    C_ptr,
    Y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_b,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w, out_dtype=tl.float32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(B_ptr + offs_n * stride_b, mask=mask_n,
                other=0.0).to(tl.float32)
    c = tl.load(C_ptr).to(tl.float32)
    bc = b - c
    out = tl.minimum(acc + bc[None, :], 0.0)

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym +
                      offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])
