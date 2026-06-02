import triton
import triton.language as tl

@triton.jit
def _linear_fused_kernel(
    A_ptr,         # [M, K]
    WT_ptr,        # we pass W in [N, K] here (keep name for signature compatibility)
    B_ptr,         # [N]
    Y_ptr,         # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,  # stride_wk: stride along K of W, stride_wn: stride along N of W
    stride_ym, stride_yn,
    scale,         # fused scale = 1 + scaling_factor
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak)
        # Use W in row-major [N, K] to avoid a separate transpose on the host
        w_ptrs = WT_ptr + (offs_n[:, None] * stride_wn + (k0 + offs_k[None, :]) * stride_wk)

        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (k0 + offs_k[None, :] < K)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)           # (BM, BK)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)           # (BN, BK)

        # acc += a @ w^T
        acc += tl.dot(a, tl.trans(w))
        k0 += BLOCK_K

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Fused scaling + residual: y = (1 + s) * (A @ W^T + b)
    acc *= scale

    # Store
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)
