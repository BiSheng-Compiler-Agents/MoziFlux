import triton
import triton.language as tl


@triton.jit
def _add_layernorm_lastdim_kernel(
        x_ptr,  # *flattened* input pointer (contiguous)
        y_ptr,  # *flattened* output pointer (contiguous)
        gamma_ptr,  # weight (normalized_shape)
        beta_ptr,  # bias   (normalized_shape)
        sum_w,  # scalar to add before LN (invariance: LN(x + c) == LN(x))
        M,  # number of columns (normalized dimension)
        N_ROWS,  # number of rows (total elements // M)
        eps,  # epsilon
        BLOCK_SIZE: tl.constexpr,  # >= M
        ROWS_PER_CTA: tl.constexpr  # rows processed per CTA
):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(cols, 16)
    tl.max_contiguous(cols, BLOCK_SIZE)

    # Reuse gamma/beta across rows (kept in registers)
    col_mask = cols < M
    g = tl.load(gamma_ptr + cols, mask=col_mask, other=1.0).to(tl.float32)
    b = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    inv_M = 1.0 / tl.full((), M, dtype=tl.float32)

    # Base row index for this program id
    row_base = pid * ROWS_PER_CTA

    # Prefetch first row
    row_idx0 = row_base
    row_active0 = row_idx0 < N_ROWS
    row_start0 = row_idx0 * M
    mask0 = col_mask & row_active0
    x_raw0 = tl.load(x_ptr + row_start0 + cols, mask=mask0, other=0.0)

    # Software-pipelined loop: prefetch next row while computing current
    for r in tl.static_range(ROWS_PER_CTA):
        row_idx = row_base + r
        row_active = row_idx < N_ROWS
        mask = col_mask & row_active

        # Current row data; exploit LN invariance to skip adding sum_w
        x_fp = x_raw0.to(tl.float32)

        # Prefetch next row early to hide latency
        if r < ROWS_PER_CTA - 1:
            next_idx = row_idx + 1
            next_active = next_idx < N_ROWS
            next_start = next_idx * M
            next_mask = col_mask & next_active
            x_raw1 = tl.load(x_ptr + next_start + cols,
                             mask=next_mask,
                             other=0.0)

        # Mean/Var in fp32
        mean = tl.sum(x_fp, axis=0) * inv_M
        diff = x_fp - mean
        var = tl.sum(diff * diff, axis=0) * inv_M
        rstd = tl.rsqrt(var + eps)

        y_fp = (diff * rstd) * g + b
        y = y_fp.to(x_raw0.dtype)
        tl.store(y_ptr + row_idx * M + cols, y, mask=mask)

        if r < ROWS_PER_CTA - 1:
            x_raw0 = x_raw1


@triton.jit
def _avgpool3d_gelu_kernel(x_ptr, y_ptr, N, C, D, H, W, Do, Ho, Wo, TOT_ROWS,
                           BLOCK_W: tl.constexpr, ROWS_PER_CTA: tl.constexpr,
                           KD: tl.constexpr, KH: tl.constexpr,
                           KW: tl.constexpr):
    pid = tl.program_id(axis=0)
    w = tl.arange(0, BLOCK_W)

    # Precompute input/output strides
    in_stride_w = 1
    in_stride_h = W
    in_stride_d = H * W
    in_stride_c = D * H * W
    in_stride_n = C * D * H * W

    out_stride_w = 1
    out_stride_h = Wo
    out_stride_d = Ho * Wo
    out_stride_c = Do * Ho * Wo
    out_stride_n = C * Do * Ho * Wo

    for r in tl.static_range(ROWS_PER_CTA):
        row = pid * ROWS_PER_CTA + r
        row_mask = row < TOT_ROWS

        # Decode (n, c, do, ho) from row id
        ho = row % Ho
        tmp = row // Ho
        do = tmp % Do
        tmp = tmp // Do
        c = tmp % C
        n = tmp // C

        # Base indices
        di0 = do * KD
        hi0 = ho * KH

        base_in = n * in_stride_n + c * in_stride_c + di0 * in_stride_d + hi0 * in_stride_h
        w_mask = (w < Wo) & row_mask

        acc = tl.zeros([BLOCK_W], dtype=tl.float32)
        # Accumulate over kernel window
        for kd_i in tl.static_range(KD):
            base_kd = base_in + kd_i * in_stride_d
            for kh_i in tl.static_range(KH):
                base_kh = base_kd + kh_i * in_stride_h
                for kw_i in tl.static_range(KW):
                    offs = base_kh + w * KW + kw_i
                    val = tl.load(x_ptr + offs, mask=w_mask, other=0.0)
                    acc += val.to(tl.float32)

        # Average
        scale = 1.0 / (KD * KH * KW)
        avg = acc * scale

        # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        e = tl.erf(avg * 0.7071067811865476)  # 1/sqrt(2)
        out = 0.5 * avg * (1.0 + e)

        base_out = n * out_stride_n + c * out_stride_c + do * out_stride_d + ho * out_stride_h
        tl.store(y_ptr + base_out + w, out, mask=w_mask)
