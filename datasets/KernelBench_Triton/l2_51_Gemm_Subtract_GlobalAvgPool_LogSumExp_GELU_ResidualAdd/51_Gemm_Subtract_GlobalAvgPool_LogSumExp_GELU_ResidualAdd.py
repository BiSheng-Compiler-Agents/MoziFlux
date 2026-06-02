import triton
import triton.language as tl

@triton.jit
def _fused_matvec_gelu_broadcast_add_cptr(
    x_ptr,          # *f32, [N, K] input / residual
    v_ptr,          # *f32, [K]    mean over rows of W (i.e., mean over out_features)
    out_ptr,        # *f32, [N, K] output
    c_ptr,          # *f32, [1]    device scalar: mean(b - subtract)
    N, K,           # i32
    stride_xm,      # i32
    stride_xk,      # i32
    stride_om,      # i32
    stride_ok,      # i32
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    # Load scalar c from device memory
    c = tl.load(c_ptr)

    # Base pointers for rows
    row_x_ptr = x_ptr + offs_m[:, None] * stride_xm
    row_out_ptr = out_ptr + offs_m[:, None] * stride_om

    # 1) Row-wise dot: s = x @ v + c
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_ptrs = row_x_ptr + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(v_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        acc += tl.sum(x_tile * v_tile[None, :], axis=1)
    acc = acc + c  # [BLOCK_M]

    # Use the standard tanh-form GELU approximation to avoid erf codegen issues.
    acc2 = acc * acc
    u = acc * (0.7978845608028654 + 0.035677408136300125 * acc2)
    gelu_s = acc * (1.0 / (1.0 + tl.exp(-2.0 * u)))  # [BLOCK_M]

    # 3) Broadcast add with original x: out = x + gelu_s[:, None]
    for n0 in range(0, K, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < K
        x_ptrs2 = row_x_ptr + offs_n[None, :] * stride_xk
        x_tile2 = tl.load(x_ptrs2, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        y_tile = (x_tile2.to(tl.float32) + gelu_s[:, None]).to(x_tile2.dtype)
        out_ptrs = row_out_ptr + offs_n[None, :] * stride_ok
        tl.store(out_ptrs, y_tile, mask=mask_m[:, None] & mask_n[None, :])
