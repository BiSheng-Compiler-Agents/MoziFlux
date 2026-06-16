import triton
import triton.language as tl


@triton.jit
def _matmul_bias_relu_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [N, K] (weight), accessed as [K, N] via strides
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    ADD_BIAS: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Provide alignment/contiguity hints to the compiler for better codegen
    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)
    tl.multiple_of(offs_k, 16)

    # Masks for M and N bounds (K handled per-iteration)
    a_mask_m = offs_m < M
    b_mask_n = offs_n < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Regular pipelined K loop; let Triton pipeline via num_stages
    k = 0
    while k < K:
        k_offs = k + offs_k

        # Compute tile pointers
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am +
                          k_offs[None, :] * stride_ak)  # [BM, BK]
        b_ptrs = b_ptr + (k_offs[:, None] * stride_bk +
                          offs_n[None, :] * stride_bn)  # [BK, BN]

        # Masks for this tile
        a_mask = (a_mask_m[:, None]) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (b_mask_n[None, :])

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Use Tensor Cores: cast inputs to fp16 and accumulate in fp32
        a = a.to(tl.float16)
        b = b.to(tl.float16)

        acc += tl.dot(a, b)
        k += BLOCK_K

    # Epilogue: bias and ReLU
    if ADD_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=b_mask_n,
                       other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store results
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=a_mask_m[:, None] & b_mask_n[None, :])
