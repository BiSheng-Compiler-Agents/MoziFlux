import triton
import triton.language as tl

@triton.jit
def _gemv_rowblock_kernel(
    A_ptr,  # *A: (M, K)
    B_ptr,  # *B: (K, 1)
    C_ptr,  # *C: (M, 1)
    M: tl.constexpr,
    K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    row_offsets = rows[:, None] * stride_am

    k0 = 0
    while k0 < K:
        cols = k0 + k_offsets
        mask_k = cols < K
        a_ptrs = A_ptr + row_offsets + cols[None, :] * stride_ak
        b_ptrs = B_ptr + cols * stride_bk
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask_k, other=0.0)
        acc += tl.sum(a_tile.to(tl.float32) * b_tile.to(tl.float32)[None, :], axis=1)
        k0 += BLOCK_K

    c_ptrs = C_ptr + rows * stride_cm
    tl.store(c_ptrs, acc, mask=mask_m)
