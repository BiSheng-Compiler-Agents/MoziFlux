import triton
import triton.language as tl


@triton.jit
def _rowwise_dot_kernel(
        x_ptr,  # *f32/*f16, shape [M, K]
        s_ptr,  # *f32/*f16, shape [K]
        out_ptr,  # *f32/*f16, shape [M, 1] (we write column 0)
        M: tl.constexpr,  # int
        K: tl.constexpr,  # int
        stride_xm,  # int: stride for dim-0 of x in elements
        stride_xk,  # int: stride for dim-1 of x in elements
        stride_outm,  # int: stride for dim-0 of out in elements
        scale,  # f32 scalar (apply once at the end)
        BLOCK_M: tl.constexpr,  # tile size along M
        BLOCK_K: tl.constexpr,  # tile size along K
):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Accumulator for each row in the block
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm +
                          offs_k[None, :] * stride_xk)
        x = tl.load(x_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                    cache_modifier=".cg").to(tl.float32)

        # Load S tile [BLOCK_K]
        s = tl.load(s_ptr + offs_k,
                    mask=mask_k,
                    other=0.0,
                    cache_modifier=".cg").to(tl.float32)

        # Accumulate row-wise dot products
        acc += tl.sum(x * s[None, :], axis=1)
        k0 += BLOCK_K

    # Apply the final scale once
    acc = acc * scale

    # Store result directly
    tl.store(out_ptr + offs_m * stride_outm, acc, mask=mask_m)
