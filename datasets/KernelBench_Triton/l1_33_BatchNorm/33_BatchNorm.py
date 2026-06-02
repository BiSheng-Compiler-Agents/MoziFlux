import triton
import triton.language as tl

@triton.jit
def _bn_row_reduce_nhw_store(
    x_ptr,
    partial_sum_ptr,
    partial_sumsq_ptr,
    N, H, W,
    stride_n, stride_c, stride_h, stride_w,
    NH,  # number of (n,h) rows per channel
    BLOCK_W: tl.constexpr,
    NUM_W_CHUNKS: tl.constexpr,
):
    # program ids
    c = tl.program_id(0)
    nh = tl.program_id(1)
    # derive n, h from nh
    n = nh // H
    h = nh - n * H

    # base pointer for this row
    row_base = x_ptr + n * stride_n + c * stride_c + h * stride_h

    # accumulate along W with vector accumulation to reduce number of tl.sum ops
    offs_w = tl.arange(0, BLOCK_W)
    tl.multiple_of(offs_w, BLOCK_W)
    acc_vec_sum = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc_vec_sumsq = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for cw in tl.static_range(0, NUM_W_CHUNKS):
        w_idx = cw * BLOCK_W + offs_w
        mask = w_idx < W
        vals = tl.load(row_base + w_idx * stride_w, mask=mask, other=0.0)
        acc_vec_sum += vals
        acc_vec_sumsq += vals * vals

    # reduce vectors to scalars once
    acc_sum = tl.sum(acc_vec_sum, axis=0)
    acc_sumsq = tl.sum(acc_vec_sumsq, axis=0)

    # store partial reductions per (c, nh), no atomics
    base_idx = c * NH + nh
    tl.store(partial_sum_ptr + base_idx, acc_sum)
    tl.store(partial_sumsq_ptr + base_idx, acc_sumsq)

@triton.jit
def _bn_finalize_params(
    partial_sum_ptr,
    partial_sumsq_ptr,
    scale_ptr,
    shift_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    NH,      # number of (n,h) rows per channel
    M,       # total elements per-channel = N*H*W
    eps,     # epsilon
    exp_avg_factor,  # exponential_average_factor
    use_batch_stats,  # 1 if using batch stats, 0 if using running stats
    do_update,        # 1 to update running stats (training & tracking), else 0
    affine_flag,      # 1 if affine, else 0
    BLOCK_NH: tl.constexpr,
    NUM_NH_CHUNKS: tl.constexpr,
):
    c = tl.program_id(0)

    # compute mean/var either from batch partial sums or from running stats
    if use_batch_stats:
        offs = tl.arange(0, BLOCK_NH)
        acc_sum = 0.0
        acc_sumsq = 0.0
        for chunk in tl.static_range(0, NUM_NH_CHUNKS):
            idx = chunk * BLOCK_NH + offs
            mask = idx < NH
            base = c * NH + idx
            psum = tl.load(partial_sum_ptr + base, mask=mask, other=0.0)
            psumsq = tl.load(partial_sumsq_ptr + base, mask=mask, other=0.0)
            acc_sum += tl.sum(psum, axis=0)
            acc_sumsq += tl.sum(psumsq, axis=0)
        mean = acc_sum / M
        var = acc_sumsq / M - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = 1.0 / tl.sqrt(var + eps)

        # optionally update running stats
        if do_update:
            rm = tl.load(running_mean_ptr + c)
            rv = tl.load(running_var_ptr + c)
            one_minus = 1.0 - exp_avg_factor
            unbiased_var = var
            if M > 1:
                unbiased_var = var * (M / (M - 1))
            rm = rm * one_minus + mean * exp_avg_factor
            rv = rv * one_minus + unbiased_var * exp_avg_factor
            tl.store(running_mean_ptr + c, rm)
            tl.store(running_var_ptr + c, rv)

        mean_use = mean
        invstd_use = invstd
    else:
        # evaluation using running stats
        rm = tl.load(running_mean_ptr + c)
        rv = tl.load(running_var_ptr + c)
        mean_use = rm
        invstd_use = 1.0 / tl.sqrt(rv + eps)

    if affine_flag:
        w = tl.load(weight_ptr + c)
        b = tl.load(bias_ptr + c)
        scale = invstd_use * w
        shift = b - mean_use * scale
    else:
        scale = invstd_use
        shift = -mean_use * invstd_use

    tl.store(scale_ptr + c, scale)
    tl.store(shift_ptr + c, shift)

@triton.jit
def _bn_apply_nhw(
    x_ptr,
    y_ptr,
    scale_ptr,
    shift_ptr,
    N, H, W,
    stride_nx, stride_cx, stride_hx, stride_wx,
    stride_ny, stride_cy, stride_hy, stride_wy,
    BLOCK_W: tl.constexpr,
    NUM_W_CHUNKS: tl.constexpr,
):
    c = tl.program_id(0)
    nh = tl.program_id(1)
    n = nh // H
    h = nh - n * H

    scale_c = tl.load(scale_ptr + c)
    shift_c = tl.load(shift_ptr + c)

    x_row = x_ptr + n * stride_nx + c * stride_cx + h * stride_hx
    y_row = y_ptr + n * stride_ny + c * stride_cy + h * stride_hy

    offs_w = tl.arange(0, BLOCK_W)
    tl.multiple_of(offs_w, BLOCK_W)
    for cw in tl.static_range(0, NUM_W_CHUNKS):
        w_idx = cw * BLOCK_W + offs_w
        mask = w_idx < W
        x = tl.load(x_row + w_idx * stride_wx, mask=mask, other=0.0)
        y = x * scale_c + shift_c
        tl.store(y_row + w_idx * stride_wy, y, mask=mask)
