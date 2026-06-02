import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Existing good configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 128}, num_stages=5, num_warps=4),
        # Added higher-K tiles and deeper pipelines for large-K performance
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=6, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=6, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=6, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Alignment/coalescing hints for better codegen on Hopper
    tl.multiple_of(offs_m, 8)
    tl.multiple_of(offs_n, 8)
    tl.multiple_of(offs_k, 8)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Static masks for M/N bounds
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k_start * BLOCK_K
        k_mask = (k_offset + offs_k) < K
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k_offset + offs_k)[None, :] * stride_ak)
        b_ptrs = b_ptr + ((k_offset + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a_mask = mask_m & k_mask[None, :]
        b_mask = k_mask[:, None] & mask_n
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
