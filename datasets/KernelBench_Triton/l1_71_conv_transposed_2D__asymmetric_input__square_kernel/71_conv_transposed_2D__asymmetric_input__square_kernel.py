import triton
import triton.language as tl


@triton.jit
def _flip_transpose_4d_kernel(
    inp_ptr,  # [Cin, Cout, K, K]
    out_ptr,  # [Cout, Cin, K, K]
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    K: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n_elements

    stride_out_kw = 1
    stride_out_kh = K
    stride_out_ci = K * K
    stride_out_co = Cin * stride_out_ci

    co = offs // stride_out_co
    rem = offs - co * stride_out_co
    ci = rem // stride_out_ci
    rem = rem - ci * stride_out_ci
    ky = rem // stride_out_kh
    kx = rem - ky * stride_out_kh

    in_ky = K - 1 - ky
    in_kx = K - 1 - kx

    stride_in_kw = 1
    stride_in_kh = K
    stride_in_co = K * K
    stride_in_ci = Cout * stride_in_co

    in_idx = (ci * stride_in_ci + co * stride_in_co + in_ky * stride_in_kh +
              in_kx * stride_in_kw)
    vals = tl.load(inp_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 64,
            'BLOCK_K': 32
        },
                      num_warps=4,
                      num_stages=3),
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 64,
            'BLOCK_K': 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 128,
            'BLOCK_K': 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'BLOCK_K': 32
        },
                      num_warps=8,
                      num_stages=5),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 64,
            'BLOCK_K': 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 256,
            'BLOCK_K': 32
        },
                      num_warps=8,
                      num_stages=4),
        # Added larger tiles and deeper pipelines for H200
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 256,
            'BLOCK_K': 32
        },
                      num_warps=8,
                      num_stages=5),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 128,
            'BLOCK_K': 32
        },
                      num_warps=8,
                      num_stages=5),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 256,
            'BLOCK_K': 32
        },
                      num_warps=8,
                      num_stages=6),
        # Allow a wider K-chunk for larger Cin cases
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'BLOCK_K': 64
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 64,
            'BLOCK_K': 64
        },
                      num_warps=4,
                      num_stages=4),
    ],
    key=['N', 'Cin', 'Cout', 'H_out', 'W_out', 'K'],
)
@triton.jit
def _convtransp2d_stride1_pad0_groups1_kernel(
    x_ptr,  # * (N, Cin, H, W)
    w_ptr,  # * (Cout, Cin, K, K) -- rotated weight: flip(spatial) + permute(out,in,kh,kw)
    bias_ptr,  # * (Cout,) or dummy
    y_ptr,  # * (N, Cout, H_out, W_out)
    N,
    Cin,
    H,
    W,
    Cout,
    K: tl.constexpr,
    H_out,
    W_out,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wo,
    stride_wi,
    stride_wkh,
    stride_wkw,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Tile ids
    pid_m = tl.program_id(0)  # rows: N * H_out * W_out
    pid_n = tl.program_id(1)  # cols: Cout

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    cols = col_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    total_rows = N * H_out * W_out
    mask_m = rows < total_rows
    mask_n = cols < Cout

    # Hints for codegen
    tl.max_contiguous(rows, BLOCK_M)
    tl.max_contiguous(cols, BLOCK_N)

    # Map rows -> (n, h_out, w_out)
    hw_total = H_out * W_out
    n_idx = rows // hw_total
    hw_idx = rows % hw_total
    h_out_idx = hw_idx // W_out
    w_out_idx = hw_idx % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_range = tl.arange(0, BLOCK_K)
    rc = 0
    while rc < Cin:
        c_idx = rc + k_range  # [BLOCK_K]
        c_mask = c_idx < Cin

        # Iterate over spatial kernel (ConvTranspose2d stride=1,pad=0 equals Conv2d with rotated kernel and padding=K-1)
        for ky in tl.static_range(0, K):
            # input h coordinate for this ky
            h_in = h_out_idx + (ky - (K - 1))
            valid_y = (h_in >= 0) & (h_in < H)
            for kx in tl.static_range(0, K):
                w_in = w_out_idx + (kx - (K - 1))
                valid_x = (w_in >= 0) & (w_in < W)
                vmask = mask_m & valid_y & valid_x

                # Precompute base pointers to reduce integer ops in inner loop
                x_base = (x_ptr + n_idx[:, None] * stride_xn +
                          h_in[:, None] * stride_xh +
                          w_in[:, None] * stride_xw)
                x_ptrs = x_base + c_idx[None, :] * stride_xc
                x_mask = vmask[:, None] & c_mask[None, :]
                a = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

                # Load W tile: (BLOCK_K, BLOCK_N) from rotated weight layout [Cout, Cin, K, K]
                w_base = (w_ptr + cols[None, :] * stride_wo + ky * stride_wkh +
                          kx * stride_wkw)
                w_ptrs = w_base + c_idx[:, None] * stride_wi
                w_mask = c_mask[:, None] & mask_n[None, :]
                b = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

                acc += tl.dot(a, b)
        rc += BLOCK_K

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + cols, mask=mask_n,
                            other=0.0).to(tl.float32)
        acc = acc + bias_vals[None, :]

    # Store Y tile
    y_ptrs = (y_ptr + n_idx[:, None] * stride_yn + cols[None, :] * stride_yc +
              h_out_idx[:, None] * stride_yh + w_out_idx[:, None] * stride_yw)
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)
