import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_W": 32}, num_warps=2, num_stages=2),
    ],
    key=["OW"],
)
@triton.jit
def _avg_pool3d_k4s4_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_W: tl.constexpr,
):
    # Program ids:
    #  - axis 0 over (N * C * OD * OH)
    #  - axis 1 tiles over OW
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    # Decompose pid0 -> n, c, od, oh
    oh_idx = pid0 % OH
    tmp = pid0 // OH
    od_idx = tmp % OD
    tmp = tmp // OD
    nc_idx = tmp
    n_idx = nc_idx // C
    c_idx = nc_idx % C

    # Tile along W dimension
    w_out = pid1 * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_out < OW

    # Base pointers using strides
    x_base = (
        n_idx * stride_n
        + c_idx * stride_c
        + (od_idx * 4) * stride_d
        + (oh_idx * 4) * stride_h
    )
    y_base = (
        n_idx * out_stride_n
        + c_idx * out_stride_c
        + od_idx * out_stride_d
        + oh_idx * out_stride_h
    )

    # Each output corresponds to a 4x4x4 block in input with stride 4
    w_in_base = w_out * 4

    # Accumulate in FP32 for numerical stability
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    for kd in range(4):
        for kh in range(4):
            base_kdh = x_base + kd * stride_d + kh * stride_h
            for kw in range(4):
                ptrs = x_ptr + base_kdh + (w_in_base + kw) * stride_w
                vals = tl.load(ptrs, mask=w_mask, other=0.0).to(tl.float32)
                acc += vals

    # Average over 64 elements
    out_vals = acc * (1.0 / 64.0)

    # Store
    y_ptrs = y_ptr + y_base + w_out * out_stride_w
    tl.store(y_ptrs, out_vals, mask=w_mask)
