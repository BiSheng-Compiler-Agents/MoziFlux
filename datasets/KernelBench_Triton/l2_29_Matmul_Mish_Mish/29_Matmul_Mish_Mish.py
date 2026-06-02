import triton
import triton.language as tl

@triton.jit
def linear_mish2_rowwise(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_row_ptr = x_ptr + pid_m * stride_xm

    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        x_vals = tl.load(
            x_row_ptr + offs_k * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        w_ptrs = w_ptr + (
            offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        )
        w_mask = (offs_n[:, None] < N) & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        k0 += BLOCK_K

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias

    threshold = 20.0

    abs_acc = tl.abs(acc)
    sp1_stable = tl.log(1.0 + tl.exp(-abs_acc)) + tl.maximum(acc, 0.0)
    sp1 = tl.where(acc > threshold, acc, sp1_stable)
    t1 = tl.exp(-2.0 * sp1)
    tanh_sp1 = 1.0 - 2.0 * t1 / (1.0 + t1)
    m1 = acc * tanh_sp1

    abs_m1 = tl.abs(m1)
    sp2_stable = tl.log(1.0 + tl.exp(-abs_m1)) + tl.maximum(m1, 0.0)
    sp2 = tl.where(m1 > threshold, m1, sp2_stable)
    t2 = tl.exp(-2.0 * sp2)
    tanh_sp2 = 1.0 - 2.0 * t2 / (1.0 + t2)
    out = m1 * tanh_sp2

    y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
    tl.store(y_ptrs, out, mask=offs_n < N)
