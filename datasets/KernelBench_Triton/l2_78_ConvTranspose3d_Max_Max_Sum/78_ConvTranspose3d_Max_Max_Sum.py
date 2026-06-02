import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 2, 'BLOCK_W': 128}, num_warps=8, num_stages=2),
    ],
    key=['H2', 'W2'],
)
@triton.jit
def _maxpool_6x_3d_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    D2, H2, W2,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_ndc = tl.program_id(2)

    # Decode n, c, d_out from pid_ndc
    CD2 = C * D2
    n = pid_ndc // CD2
    rem = pid_ndc % CD2
    c = rem // D2
    d_out = rem % D2

    # Tile of output H/W
    w_out = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_out = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mw = w_out < W2
    mh = h_out < H2
    m_hw = mh[:, None] & mw[None, :]

    # Compute starting indices in input for this output tile
    d_start = d_out * 6
    h_start = h_out * 6  # [BH]
    w_start = w_out * 6  # [BW]

    # Prepare max accumulator in fp32 for numerical stability
    m = tl.full((BLOCK_H, BLOCK_W), -float('inf'), dtype=tl.float32)

    # Base pointer for batch/channel
    base_nc = n * stride_n + c * stride_c

    # Iterate over the 6x6x6 window
    for kd in range(6):
        d_idx_off = (d_start + kd) * stride_d
        for kh in range(6):
            h_idx = h_start[:, None] + kh
            h_off = h_idx * stride_h
            for kw in range(6):
                w_idx = w_start[None, :] + kw
                w_off = w_idx * stride_w
                ptr = x_ptr + base_nc + d_idx_off + h_off + w_off
                val = tl.load(ptr, mask=m_hw, other=0.0)
                m = tl.where(m_hw, tl.maximum(m, val.to(tl.float32)), m)

    # Store results
    out_base = out_ptr + n * out_stride_n + c * out_stride_c + d_out * out_stride_d
    out_ptrs = out_base + h_out[:, None] * out_stride_h + w_out[None, :] * out_stride_w
    tl.store(out_ptrs, m, mask=m_hw)
