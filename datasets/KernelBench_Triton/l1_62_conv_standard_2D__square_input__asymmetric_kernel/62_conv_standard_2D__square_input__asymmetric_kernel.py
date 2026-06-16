import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_warps=4,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_warps=4,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_warps=4,
                      num_stages=3),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_warps=8,
                      num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _conv2d_im2col_gemm_kernel(
    x_ptr,  # *tensor
    w_ptr,  # *tensor (flattened weights: [N=CO, K=CI*KH*KW])
    y_ptr,  # *tensor (fp32)
    B,  # batch size
    CI,
    H,
    W,  # input dims
    OH,
    OW,  # output spatial dims
    KH,
    KW,  # kernel dims
    CO,  # out channels
    STRH,
    STRW,  # stride
    PADH,
    PADW,  # padding
    DILH,
    DILW,  # dilation
    M: tl.constexpr,  # total rows in im2col (B*OH*OW)
    N: tl.constexpr,  # total cols (CO)
    K: tl.constexpr,  # reduction dim (CI*KH*KW)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # tile coordinates
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # Strides for NCHW
    y_stride_n = CO * OH * OW
    y_stride_c = OH * OW
    y_stride_h = OW
    y_stride_w = 1

    x_stride_n = CI * H * W
    x_stride_c = H * W
    x_stride_h = W
    x_stride_w = 1

    # Decode m into (b_idx, oh, ow)
    ohow = OH * OW
    b_idx = m_offsets // ohow
    rem = m_offsets - b_idx * ohow
    oh = rem // OW
    ow = rem - oh * OW

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop - statically unrolled to enable pipelining with num_stages
    for k0 in tl.static_range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Decode k into (ci, kh, kw)
        khkw = KH * KW
        ci = k_offsets // khkw
        remk = k_offsets - ci * khkw
        kh = remk // KW
        kw = remk - kh * KW

        # Compute input positions
        ih = oh[:, None] * STRH - PADH + kh[None, :] * DILH
        iw = ow[:, None] * STRW - PADW + kw[None, :] * DILW
        in_bounds = ((ih >= 0) & (iw >= 0) & (ih < H) &
                     (iw < W)) & m_mask[:, None] & k_mask[None, :]

        # Compute input addresses and load A tile [BLOCK_M, BLOCK_K]
        x_offsets = (b_idx[:, None] * x_stride_n + ci[None, :] * x_stride_c +
                     ih * x_stride_h + iw * x_stride_w)
        a_tile = tl.load(x_ptr + x_offsets, mask=in_bounds,
                         other=0.0).to(tl.float32)

        # Load weight tile as transposed block for better dot: [BLOCK_K, BLOCK_N]
        w_offsets_t = k_offsets[:, None] + n_offsets[None, :] * K
        wb_mask = k_mask[:, None] & n_mask[None, :]
        b_tile_t = tl.load(w_ptr + w_offsets_t, mask=wb_mask,
                           other=0.0).to(tl.float32)

        # Accumulate in fp32
        acc += tl.dot(a_tile, b_tile_t)

    # Store to output
    y_offsets = (b_idx[:, None] * y_stride_n +
                 n_offsets[None, :] * y_stride_c + oh[:, None] * y_stride_h +
                 ow[:, None] * y_stride_w)
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + y_offsets, acc, mask=y_mask)


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _conv2d_im2col_gemm_kernel_mma(
        x_ptr,  # *tensor (fp16/bf16)
        w_ptr,  # *tensor (fp16/bf16) flattened [CO, K]
        y_ptr,  # *tensor (fp32 acc)
        B,  # batch size
        CI,
        H,
        W,  # input dims
        OH,
        OW,  # output spatial dims
        KH,
        KW,  # kernel dims
        CO,  # out channels
        STRH,
        STRW,  # stride
        PADH,
        PADW,  # padding
        DILH,
        DILW,  # dilation
        M: tl.constexpr,  # B*OH*OW
        N: tl.constexpr,  # CO
        K: tl.constexpr,  # CI*KH*KW
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        IS_BF16: tl.constexpr,  # specialize for bf16 vs fp16
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # Strides for NCHW
    y_stride_n = CO * OH * OW
    y_stride_c = OH * OW
    y_stride_h = OW
    y_stride_w = 1

    x_stride_n = CI * H * W
    x_stride_c = H * W
    x_stride_h = W
    x_stride_w = 1

    # Decode m -> (b, oh, ow)
    ohow = OH * OW
    b_idx = m_offsets // ohow
    rem = m_offsets - b_idx * ohow
    oh = rem // OW
    ow = rem - oh * OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.static_range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        khkw = KH * KW
        ci = k_offsets // khkw
        remk = k_offsets - ci * khkw
        kh = remk // KW
        kw = remk - kh * KW

        ih = oh[:, None] * STRH - PADH + kh[None, :] * DILH
        iw = ow[:, None] * STRW - PADW + kw[None, :] * DILW
        in_bounds = ((ih >= 0) & (iw >= 0) & (ih < H) &
                     (iw < W)) & m_mask[:, None] & k_mask[None, :]

        x_offsets = (b_idx[:, None] * x_stride_n + ci[None, :] * x_stride_c +
                     ih * x_stride_h + iw * x_stride_w)
        a_tile = tl.load(x_ptr + x_offsets, mask=in_bounds, other=0.0)
        if IS_BF16:
            a_tc = a_tile.to(tl.bfloat16)
        else:
            a_tc = a_tile.to(tl.float16)

        # Load weight tile transposed: [BLOCK_K, BLOCK_N]
        w_offsets_t = k_offsets[:, None] + n_offsets[None, :] * K
        wb_mask = k_mask[:, None] & n_mask[None, :]
        b_tile_t = tl.load(w_ptr + w_offsets_t, mask=wb_mask, other=0.0)
        if IS_BF16:
            b_tc = b_tile_t.to(tl.bfloat16)
        else:
            b_tc = b_tile_t.to(tl.float16)

        acc += tl.dot(a_tc, b_tc, out_dtype=tl.float32)

    y_offsets = (b_idx[:, None] * y_stride_n +
                 n_offsets[None, :] * y_stride_c + oh[:, None] * y_stride_h +
                 ow[:, None] * y_stride_w)
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + y_offsets, acc, mask=y_mask)
