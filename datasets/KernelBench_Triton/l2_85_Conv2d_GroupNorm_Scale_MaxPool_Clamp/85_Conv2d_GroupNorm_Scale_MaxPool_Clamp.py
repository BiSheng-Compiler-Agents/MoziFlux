import triton
import triton.language as tl


@triton.jit
def _groupnorm_stats_kernel(
    x_ptr,  # *f32
    mean_ptr,  # *f32
    rstd_ptr,  # *f32
    B,
    C,
    H,
    W,
    G,  # i32
    eps,  # f32
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (n, g) pair computes mean and rstd for that group
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    HW = H * W
    Cpg = C // G
    TOT = Cpg * HW

    base_n = n * C * HW
    ch_start = g * Cpg

    offs = tl.arange(0, BLOCK_SIZE)
    s = tl.zeros([1], dtype=tl.float32)
    ss = tl.zeros([1], dtype=tl.float32)

    start = 0
    while start < TOT:
        idx = start + offs
        mask = idx < TOT

        c_rel = idx // HW
        hw = idx - c_rel * HW
        c_idx = ch_start + c_rel

        ptrs = base_n + c_idx * HW + hw
        x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)

        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE

    denom = tl.full([1], TOT, dtype=tl.float32)
    mean = s / denom
    var = ss / denom - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    scalar_offs = tl.arange(0, 1)
    tl.store(mean_ptr + n * G + g + scalar_offs, mean, mask=scalar_offs == 0)
    tl.store(rstd_ptr + n * G + g + scalar_offs, rstd, mask=scalar_offs == 0)


@triton.jit
def _groupnorm_apply_pool_clamp_kernel(
        x_ptr,
        y_ptr,  # *f32
        mean_ptr,
        rstd_ptr,  # *f32
        gamma_ptr,
        beta_ptr,  # *f32 (already fused with scale)
        B,
        C,
        H,
        W,
        G,  # i32
        Ho,
        Wo,  # i32
        clamp_min,
        clamp_max,  # f32
        K: tl.constexpr,  # kernel size (square)
        STRIDE: tl.constexpr,  # stride == kernel size
        BLOCK_W: tl.
    constexpr,  # number of output columns processed per program
):
    # Grid: (B*C, Ho, ceil(Wo/BLOCK_W))
    pid_nc = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)

    c = pid_nc % C
    n = pid_nc // C
    h_out = pid_h

    # Group info
    Cpg = C // G
    g = c // Cpg

    # Strides
    in_plane_stride = H * W
    out_plane_stride = Ho * Wo
    in_base_nc = (n * C + c) * in_plane_stride
    out_base_nc = (n * C + c) * out_plane_stride

    # Scalars per (n, c)
    mean = tl.load(mean_ptr + n * G + g)
    rstd = tl.load(rstd_ptr + n * G + g)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    # Output column offsets
    w_offsets = tl.arange(0, BLOCK_W)
    w_out = pid_w * BLOCK_W + w_offsets
    w_out_mask = w_out < Wo

    tl.max_contiguous(w_out, BLOCK_W)
    tl.multiple_of(w_out, BLOCK_W)

    # Corresponding input row start
    h_in_base = h_out * STRIDE

    # We'll accumulate extreme of raw x (pre-normalization) depending on sign(gamma)
    # This allows postponing the affine to only the selected extreme element.
    if K == 2:
        h0 = h_in_base
        h1 = h_in_base + 1

        h0_mask = h0 < H
        h1_mask = h1 < H

        w_in0 = w_out * STRIDE
        w_in1 = w_in0 + 1

        acc_max = tl.full([BLOCK_W], -float("inf"), dtype=tl.float32)
        acc_min = tl.full([BLOCK_W], float("inf"), dtype=tl.float32)

        # Row 0
        row0_base = in_base_nc + h0 * W
        mask0a = w_out_mask & h0_mask & (w_in0 < W)
        mask0b = w_out_mask & h0_mask & (w_in1 < W)
        v0a = tl.load(x_ptr + row0_base + w_in0, mask=mask0a, other=0.0)
        v0b = tl.load(x_ptr + row0_base + w_in1, mask=mask0b, other=0.0)
        acc_max = tl.where(mask0a, tl.maximum(acc_max, v0a), acc_max)
        acc_min = tl.where(mask0a, tl.minimum(acc_min, v0a), acc_min)
        acc_max = tl.where(mask0b, tl.maximum(acc_max, v0b), acc_max)
        acc_min = tl.where(mask0b, tl.minimum(acc_min, v0b), acc_min)

        # Row 1
        row1_base = in_base_nc + h1 * W
        mask1a = w_out_mask & h1_mask & (w_in0 < W)
        mask1b = w_out_mask & h1_mask & (w_in1 < W)
        v1a = tl.load(x_ptr + row1_base + w_in0, mask=mask1a, other=0.0)
        v1b = tl.load(x_ptr + row1_base + w_in1, mask=mask1b, other=0.0)
        acc_max = tl.where(mask1a, tl.maximum(acc_max, v1a), acc_max)
        acc_min = tl.where(mask1a, tl.minimum(acc_min, v1a), acc_min)
        acc_max = tl.where(mask1b, tl.maximum(acc_max, v1b), acc_max)
        acc_min = tl.where(mask1b, tl.minimum(acc_min, v1b), acc_min)

        # Select based on sign of gamma
        acc_v = tl.where(gamma >= 0, acc_max, acc_min)
        acc = ((acc_v - mean) * rstd) * gamma + beta
    else:
        # Generic KxK pooling path, choose extreme in raw x then apply affine once.
        acc_max = tl.full([BLOCK_W], -float("inf"), dtype=tl.float32)
        acc_min = tl.full([BLOCK_W], float("inf"), dtype=tl.float32)
        for kh in tl.static_range(0, K):
            h_in = h_in_base + kh
            h_mask = h_in < H
            row_h_base = in_base_nc + h_in * W
            for kw in tl.static_range(0, K):
                w_in = w_out * STRIDE + kw
                in_mask = w_out_mask & h_mask & (w_in < W)
                v = tl.load(x_ptr + row_h_base + w_in, mask=in_mask, other=0.0)
                acc_max = tl.where(in_mask, tl.maximum(acc_max, v), acc_max)
                acc_min = tl.where(in_mask, tl.minimum(acc_min, v), acc_min)
        acc_v = tl.where(gamma >= 0, acc_max, acc_min)
        acc = ((acc_v - mean) * rstd) * gamma + beta

    # Clamp
    acc = tl.minimum(acc, clamp_max)
    acc = tl.maximum(acc, clamp_min)

    # Store
    out_row_base = out_base_nc + h_out * Wo
    tl.store(y_ptr + out_row_base + w_out, acc, mask=w_out_mask)


@triton.jit
def _groupnorm_apply_kernel(
    x_ptr,
    y_ptr,  # *f32
    mean_ptr,
    rstd_ptr,  # *f32
    gamma_ptr,
    beta_ptr,  # *f32
    B,
    C,
    H,
    W,
    G,  # i32
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)  # linear over (B, C)
    n = pid // C
    c = pid % C

    HW = H * W
    Cpg = C // G
    g = c // Cpg

    base = (n * C + c) * HW
    mean = tl.load(mean_ptr + n * G + g)
    rstd = tl.load(rstd_ptr + n * G + g)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    offs = tl.arange(0, BLOCK_HW)
    start = 0
    while start < HW:
        idx = start + offs
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = ((x - mean) * rstd) * gamma + beta
        tl.store(y_ptr + base + idx, y, mask=mask)
        start += BLOCK_HW
