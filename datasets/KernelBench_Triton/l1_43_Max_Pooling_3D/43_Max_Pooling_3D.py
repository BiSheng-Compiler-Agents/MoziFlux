import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 16}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_W": 32}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_W": 32}, num_warps=1, num_stages=4),
        triton.Config({"BLOCK_W": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_W": 64}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=4),
    ],
    key=["outW"],
)
@triton.jit
def _maxpool3d_fwd_kernel(
    x_ptr,  # *f32 / *f16 contiguous NCDHW
    y_ptr,  # *f32 / *f16 contiguous NCDHW
    N, C, D, H, W,
    outD, outH, outW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    K_D: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_row = tl.program_id(0)  # over N*C*outD*outH
    pid_col = tl.program_id(1)  # over outW tiles

    ow = pid_col * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = ow < outW
    tl.max_contiguous(ow, BLOCK_W)
    tl.multiple_of(ow, 1)

    # decompose pid_row -> (n, c, od, oh)
    oh = pid_row % outH
    t = pid_row // outH
    od = t % outD
    t = t // outD
    c = t % C
    n = t // C

    # starting coords in input for this output location (vector across W)
    in_z0 = od * stride_d - pad_d
    in_y0 = oh * stride_h - pad_h
    in_x0 = ow * stride_w - pad_w  # [BLOCK_W]

    # flattened base offsets assuming contiguous memory
    # for x: ((((n*C + c) * D + z) * H + y) * W + x)
    base_nc = (n * C + c) * D * H * W

    # prepare output linear index (contiguous)
    out_idx = (((n * C + c) * outD + od) * outH + oh) * outW + ow

    # accumulator in fp32 for robustness (store will cast as needed)
    neg_inf = -float("inf")
    acc = tl.full((BLOCK_W,), neg_inf, dtype=tl.float32)

    # precompute level strides
    L1 = H * W
    L2 = W

    # Reorder loops to hoist x/x_valid computation out of kd/kh loops.
    x = in_x0
    for kw in tl.static_range(0, K_W):
        x_valid = (x >= 0) & (x < W)
        for kd in tl.static_range(0, K_D):
            z = in_z0 + kd * dil_d
            z_valid = (z >= 0) & (z < D)
            z_base = z * L1
            for kh in tl.static_range(0, K_H):
                y = in_y0 + kh * dil_h
                y_valid = (y >= 0) & (y < H)
                base_zh = z_base + y * L2
                m = mask_ow & z_valid & y_valid & x_valid
                in_idx = base_nc + base_zh + x
                vals = tl.load(x_ptr + in_idx, mask=m, other=neg_inf, eviction_policy="evict_first")
                vals = vals.to(tl.float32)
                acc = tl.maximum(acc, vals)
        x += dil_w

    tl.store(y_ptr + out_idx, acc, mask=mask_ow)
